# FakeTensor 动态信息采集方案设计文档

## 1. 方案概述

本方案设计了一套独立于 `analyze_trace.py` 的动态信息采集系统 `fake_runner.py`，用于补充和校准静态 DAG 分析。系统分为三个阶段：

| Phase | 功能 | 依赖环境 | 当前状态 |
|-------|------|---------|---------|
| Phase 1 | AST 源码解析 Module 树 | 仅 Python 3 stdlib | ✅ 完成，已跑通 |
| Phase 2 | FakeTensor 调用序列采集 | torch + lagrange_torch Docker | ⚠️ 需要 Docker 环境 |
| Phase 3 | 与静态 DAG 的 merge 接口 | 无额外依赖 | 📐 接口已设计 |

## 2. Phase 1：AST-based Module Tree Extraction

### 2.1 实现原理

通过 Python `ast` 模块解析所有 `.py` 文件，递归追踪 `nn.Module` 子类的 `__init__()` 中的子模块注册：

- `self.attr = SomeModule(...)` → 直接属性绑定
- `setattr(self, name, SomeModule(...))` → 动态属性绑定
- `self.container.append(SomeModule(...))` → ModuleList 元素追踪
- 局部变量追踪：`var = SomeClass(...); self.list.append(var)` → 解析为对应类型

### 2.2 输出样例（5698781 模型，前 20 条）

```
AwemeLiveCVR (root, 64 entries total)
├── sparse_module: SparseModule
│   ├── bias2nn: Bias2NN
│   ├── tokens_module_dict: ModuleDict
│   └── group_token_module: GroupTokenModule
├── transformer: Transformer
│   ├── resblocks: ModuleList
│   └── resblocks[*]: ResidualAttentionBlock
│       ├── ln1: LayerNorm
│       ├── mlp: SparseMoe
│       │   ├── router: Router
│       │   │   └── gate: BatchMatMulDense
│       │   └── ffn: PerTokensFFN
│       ├── attn: MultiHeadAttention / FlashAttentionForTorch
│       └── gau_attention: GAUAttention
│           └── gau_blocks: ModuleList → GAUBlock[*]
├── <dynamic>: GroupTower (×5: relation/gift/stay/pos/other)
│   └── dense_towers: DenseTower
│       └── layers: ModuleList
├── grouped_task_tower_modules: ModuleDict
├── label_and_mask_module: CVRLabel
└── gift_diamond_ordinal_regression_module: GiftOrdinalRegressionModule
    └── ordinal_towers: ModuleList
```

### 2.3 对比静态分析的优势

| 维度 | 静态分析 (analyze_trace.py) | Phase 1 (fake_runner.py) |
|------|--------------------------|--------------------------|
| ModuleList 元素类型 | 需匹配 `for` 循环推断 | 直接从 `.append()` 追踪 |
| 动态 setattr | 已支持但保守 | 同样支持，精确标记 `[DYNAMIC]` |
| 展开层数 | 依赖 trace 验证 | 递归无限深（防环） |
| Container 元素识别 | 推断容器类型 | 精确追踪 append 参数类型 |

### 2.4 直接解决的问题

**Bug 1（ModuleList 展开数错误）**：Phase 1 能精确告知每个 `ModuleList` 实际注册了什么元素类型。例如：
- `transformer.resblocks[*]` → `ResidualAttentionBlock`（不再需要保守猜测展开 1 个还是 N 个）
- `DenseTower.layers[*]` → 包含多层 Linear（静态分析之前错误地只展开 `_layers[0]`）

## 3. Phase 2：FakeTensor 调用序列采集

### 3.1 设计方案

```python
# Hook nn.Module.__call__ 记录调用链
for name, mod in model.named_modules():
    mod.register_forward_hook(make_hook(name))

# 使用 FakeTensorMode 执行模型 forward
with torch._subclasses.FakeTensorMode():
    fake_input = construct_fake_inputs(batch_size=1024)
    model(fake_input)
```

### 3.2 输出格式 (call_sequence.json)

```json
[
  {
    "order": 0,
    "module_path": "sparse_module",
    "class": "SparseModule",
    "input_tensor_ids": [0, 1, 2],
    "output_tensor_ids": [3, 4],
    "input_shapes": [[1024, 646], [1024, 256]],
    "output_shapes": [[1024, 1024], [1024, 128]]
  },
  {
    "order": 1,
    "module_path": "transformer",
    "class": "Transformer",
    "input_tensor_ids": [3],
    "output_tensor_ids": [5],
    "input_shapes": [[1024, 64, 256]],
    "output_shapes": [[1024, 64, 256]]
  }
]
```

### 3.3 当前环境限制

当前 workspace 中模型实例化失败的根本原因链：

```
main_model.py 
  → import config  
    → from bytedance.lagrange_torch.common.core import is_torch_serving
  → from cvr import AwemeLiveCVR
    → models/sparse_module.py
      → from perceiver import SeqPerceiver
        → from attn import MultiHeadAttention
          → class FlashAttention: @torch.compiler.disable
            → torch._dynamo 内部导入链触发 mock 冲突
```

**失败原因**：`bytedance.lagrange_torch` 是一个复杂的框架，深度嵌入模型代码各处。仅靠 `sys.modules` 级别的 mock 无法完全模拟其行为，因为：
1. 框架提供 `LG.dense_feature()` 等返回 `nn.Module` 实例的工厂方法
2. `LG.get_bias()` 等方法动态生成 embedding lookup 模块
3. `LG.result()` 返回具有 `.head()` 接口的 Result 对象
4. catch-all mock 与 PyTorch 内部 `torch._dynamo` 的导入链冲突

### 3.4 Docker 环境替代方案

在 lagrange_torch Docker 环境中运行：

```bash
# 1. 启动 Docker (torch27 镜像)
docker run -it --rm \
  -v $(pwd):/workspace \
  <registry>/lagrange/torch/ci/images/dev:torch27 bash

# 2. 设置环境
export BATCH_SIZE=1024
export FORGE_COMPILE=1  # 使用 CPU 模式

# 3. 运行 fake_runner.py (Phase 2)
cd /workspace
python3 torch-trace-analyzer/scripts/fake_runner.py \
  --code-path testset/extracted/5698781/modelcode/ \
  --output-dir fake_runner_output \
  --root-class AwemeLiveCVR \
  --batch-size 1024
```

在 Docker 中：
- `bytedance.lagrange_torch` 完整可用
- 模型可以正常实例化（设置 `FORGE_COMPILE=1` 避免 CUDA 依赖）
- FakeTensorMode 可以正常运行（PyTorch >= 2.0 支持）

## 4. Phase 3：Merge 接口设计

### 4.1 接口签名

```python
def merge_dynamic_info(static_dag: dict, dynamic_info: dict) -> dict:
    """
    Parameters:
        static_dag: analyze_trace.py 输出的 DAG JSON
        dynamic_info: {"module_tree": [...], "call_sequence": [...]}
    
    Returns:
        calibrated_dag: 校准后的 DAG
    """
```

### 4.2 校准规则

| 规则 | 输入 | 校准动作 |
|------|------|---------|
| ModuleList 展开修正 | Phase 1 module_tree | 精确设置展开数 |
| 假边过滤 | Phase 2 tensor_id | 无 tensor 依赖的边标记 suspect |
| 并行 vs 顺序 | Phase 2 input_tensor_ids | 共享输入且无输出依赖 → 并行 |
| Shape 标注 | Phase 2 shapes | 在节点上添加 input/output shape |
| 置信度评分 | Phase 1 + 2 | static_only / dynamic_confirmed |

### 4.3 校准示例

**Bug 2（假边 `query_norm → action_norm`）**：

静态分析中，`query_norm` 和 `action_norm` 在 `forward()` 中顺序调用：
```python
q = self.query_norm(x)
a = self.action_norm(x)
```

静态分析可能误判为 `query_norm → action_norm`（因为两者顺序出现）。  
动态信息校准：
- `query_norm` 的 `input_tensor_ids = [10]`
- `action_norm` 的 `input_tensor_ids = [10]`
- `query_norm` 的 `output_tensor_ids = [11]`
- `action_norm` 的 `output_tensor_ids = [12]`

**判定**：`action_norm` 的输入中不包含 `query_norm` 的输出 `[11]`，因此 `query_norm → action_norm` 是假边，应删除。两者共享同一输入 `[10]`，属于并行分支。

## 5. 与已知 Bug 的关联分析

### Bug 1：ModuleList 展开数错误

**问题**：DenseTower 的 `nn.ModuleList` 包含多层 Linear，但静态分析只展开 `_layers[0]`。

**Phase 1 解决方案**：
- `module_tree.json` 中 `DenseTower.layers` 标记为 `ModuleList`，且 `[*]` 元素类型为已知
- 与 `named_modules()` 的 live 数据交叉校验：精确知道有多少层
- 即使无法实例化，AST 解析也能告诉我们 `_built_layers()` 会创建多少层（从 `output_dims` 参数推断循环次数）

**完全解决需要**：Phase 2 在 Docker 中运行，`named_modules()` 返回精确数量。

### Bug 2：并行调用被误判为顺序依赖

**问题**：`query_norm` 和 `action_norm` 共享相同输入但无数据依赖，被静态分析错误连线。

**Phase 2 解决方案**：
- 通过 `tensor_id` 追踪，可精确判定 A 的输出是否被 B 消费
- 如果 A.output_ids ∩ B.input_ids = ∅，则 A→B 是假边
- 共享相同 input_ids 的模块是并行分支

**当前可用的部分解决**：
- 静态分析已通过变量追踪（`x = self.query_norm(x)` vs `y = self.action_norm(x)`）识别 LHS 不同变量名来避免假边
- Phase 2 提供终极验证：即使变量名相同（如 loop 中重复使用 `x`），tensor_id 也能区分

## 6. 实现状态

### 已实现

1. ✅ `torch-trace-analyzer/scripts/fake_runner.py` — 完整脚本
   - Phase 1：AST 解析，支持 self.attr / setattr / append / 局部变量追踪
   - Phase 2：含完整 lagrange_torch mock 框架（当前因深层 mock 冲突失败）
   - Phase 3：`merge_dynamic_info()` 接口定义和文档

2. ✅ 输出文件：
   - `fake_runner_output/module_tree.json` — 64 条 module 树条目
   - `fake_runner_output/summary.json` — 执行结果摘要

### 待完成（需要 Docker 环境）

1. ⬜ Phase 2 完整运行（需 lagrange_torch Docker）
2. ⬜ Phase 3 merge 逻辑实现（等 Phase 2 数据后实现）
3. ⬜ 多模型测试（5476790 / 5547919 等）

## 7. 后续步骤

1. **短期**：在 lagrange_torch Docker 中运行 Phase 2，获取 `call_sequence.json`
2. **中期**：基于动态数据实现 `merge_dynamic_info()` 的校准逻辑
3. **长期**：将动态信息集成到前端可视化（边上标注 shape、置信度着色）
