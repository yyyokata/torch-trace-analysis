# PR3 B 类回归 — AST 覆盖缺口修复 · 测例设计文档

> 状态：测例设计阶段（**等待 NaN 审核**，未进入修复代码阶段）
> 作者：Constructor (林楠 整理)
> 工作脚本：`ast_refactor_workdir/scripts/analyze_trace.py`（PR3 当前 8470 行）
> 安全基线：`ast_refactor_workdir/scripts/analyze_trace_pr3_BASELINE.bak`（9018 行，含 regex fallback）
> 关联设计：`ast_evaluator_redesign.md`（PR1/2/3 整体重构）

---

## 0. 重要前置发现

**与 NaN 描述的差异**：NaN 任务描述把 5 个失败定性为「PR3 删掉 regex fallback 后暴露的 AST 覆盖缺口」。但实际验证 PR3 BASELINE（regex fallback 仍然存在的版本）时，**这 5 个 case 同样全部失败**，且失败原因与 PR3 当前一致（同样的 Rule3 环、同样的 Rule6_out 失败）。

| Model | PR3 当前（无 fallback） | PR3 BASELINE（有 fallback） | 是否新引入 |
|---|---|---|---|
| 5547919 | Rule3: 8 节点环 (MMCN/ResNet) | Rule3: 8 节点环 (MMCN/ResNet) | ❌ 旧问题 |
| 5698781 | Rule3: 28 节点环 (GAUBlock) | Rule3: 8 节点环 (GAUBlock) + cycle warning | ❌ 旧问题，PR3 后规模放大 |
| 5820572 | Rule3: 28 节点环 (GAUBlock) | Rule3: 28 节点环 (GAUBlock) | ❌ 旧问题 |
| 5698781+trace | 同 5698781 | 同 5698781 | ❌ 旧问题 |
| 5800920 | Rule6_out FAIL (2) + nodes=271 | Rule6_out FAIL (2) + nodes=291 | ❌ 旧问题，PR3 后 nodes 减少 |

结论：**这些是历史遗留缺口**，PR3 删除 fallback 只是让某些表达式从「regex 兜底成功」变为「AST 主路径无法解析」。但环路本身的根因与 fallback 无关。

修复策略仍然遵循 NaN 的指示：**通过 AST 把这些场景接住，不恢复 fallback**。

---

## 1. 失败 Case 根因汇总

### Case A：GAUBlock `{'norm#1', 'g_swish'}` 假环（5698781 / 5820572 / 5698781+trace）

**根因（已通过插桩定位）**：

`_build_data_dependency_edges` 函数的 **元组解包（tuple unpacking）非模块调用分支** 在 `analyze_trace.py:4146-4149` 错误使用了 `_ast_var_env` 预扫数据：

```python
# transformer.py:568:  q, k, v, g = torch.split(qkvg, [...], dim=-1)
# 处理这一行时，进入 "if not called_attrs" 元组解包分支：

_line_for_v = phys_lineno              # 应该用 568
if _ast_var_env.get(v) and _ast_var_env[v][1] is not None:
    _line_for_v = _ast_var_env[v][1]   # ← 错！被覆盖为 571
var_producers[v] = (rhs_producers, _line_for_v)
```

`_ast_var_env` 是 `build_var_env` 返回的「整方法所有 `<var> = self.attr(...)` 赋值的预扫表」。它把变量名 `g` 映射到下一个 self.attr 赋值点（第 571 行 `g = self.g_swish(g)`）。

**问题链**：
1. 第 568 行 `q, k, v, g = torch.split(...)` — `g` 实际产生于第 568 行（torch.split 的输出）
2. 但 `_ast_var_env['g'] = ('g_swish', 571)` 已被预填
3. 元组解包逻辑盲目用 571 覆盖 `var_producers['g']` 的行号
4. 同时 RHS `qkvg` 通过 `_collect_expr_producers` 拉到 `norm` 作为 producer（因为 `qkvg` 的 lineage 链 → `x@558 from norm` → `qkvg@567 from fuse_dense`，但 producer 集合可能合并了上游链路）
5. 结果 `var_producers['g'] = ({'norm'}, 571)`
6. 真正在第 571 行 `self.g_swish(g)` 执行时，消费者读取 `g` 的 producer = `norm`，from_line=571
7. 在 split-cycle 阶段：`norm` 共 2 次调用（558/609），sorted lines = [558, 571, 609, ...]，cnt=2。`from_line=571` → idx=1 → `norm#1`
8. 假边 `(norm#1, g_swish)` 出现，与合法边 `(g_swish, norm#1)` 构成环

**AST 覆盖缺口**：
非模块赋值（如 `torch.split`、`torch.nn.Tanh()()`、本地辅助函数等）的 LHS 行号污染 — 元组解包逻辑不应使用 `_ast_var_env` 提供的「未来某次 self.attr 赋值」覆盖行号。

---

### Case B：MMCN / ResNet `recycle_layers_[*] ↔ ln_layers_[*]` 假环（5547919）

**根因**：

`mmcn.py` MMCN.forward (L163-167) 与 `main_model.py` ResNet.forward (L447-466) 都包含同一个 pattern：

```python
# mmcn.py L163-167
output1 = self.recycle_layers_[f"{self.name}_RecycleNet_{i}_weight_1"](cur_layer)   # call A
output1_ln = self.ln_layers_[f"{self.name}_RecycleNet_{i}_output1_ln"](output1)      # call B
output1 = smelu(output1_ln)                                                          # 重赋值 output1
recycle_scale = self.recycle_layers_[f"{self.name}_RecycleNet_{i}_weight_2"](output1) # call C
recycle_scale_ln = self.ln_layers_[f"{self.name}_RecycleNet_{i}_scale_ln"](recycle_scale)  # call D
```

**ModuleDict f-string key 不可枚举**：脚本输出 `[ERROR] Cannot statically enumerate keys for ... reason: non-literal key expression`。所有 `self.recycle_layers_[<f-string>]` 调用全部折叠到同一个 `recycle_layers_[*]` 通配节点，`self.ln_layers_[<f-string>]` 全部折叠到 `ln_layers_[*]`。

**Multi-call splitter 失效**：
- 在 `_attr_max_calls_in_single_method` 计数阶段（L3325-3398），同方法 3 次 `recycle_layers_[*]` 调用 → cnt=3。**应该被 split 成 #0/#1/#2**。
- 但实际运行时 splitter 不工作：从 dump_dag.html SCC 输出可看到 `node_232 ↔ node_233` 是 **未带 `#N` 后缀的原始 attr**。说明 `_multi_call_candidates ∩ cycle_nodes = ∅`。

**为什么不 split？** 计数阶段把名字记为 `recycle_layers_[*]`（带 `[*]`），但 cycle_nodes 中的节点 ID 是另一种形式（很可能是 `recycle_layers_[*]` 本身或者底层 attr 名 `recycle_layers_`），两边名字对不上 → 交集为空 → 跳过 split。

需要插桩验证名字一致性，但目前推测：在 `known_attrs` 中实际登记的可能是 `recycle_layers_[*]`，但在 edge_set 中可能是 `recycle_layers_` 或 `recycle_layers_*`。

**AST 覆盖缺口**：
1. `_norm_index_expr` 对 f-string 索引返回 `None`，导致计数走 `_wild = f"{_cont}[*]"` 分支，但下游 edge 构建可能用了不同的 attr 名 ⇒ split 名字空间错位。
2. 即使 split 成功，遍历 `for i in range(...)` 的循环展开未生效（动态 key 的 ModuleDict 没有按循环索引展开成 per-iteration 节点）。

---

### Case C：5800920 Rule6_out FAIL (2) + nodes=271 严重偏移（期望 166）

**当前症状**：
- `Rule6_out failures (2)`:
  - `node_436→node_2__out (concat_module_long_seq_kv_tensors_100k)` 缺 parent-side LHS consumer line
  - `node_437→node_2__out (long_seq_sim_kv_ln)` 缺 parent-side LHS consumer line
- `nodes=271` vs Iter17 生产基线 nodes=166 → 105 个多余节点

**根因（推测，需测例验证）**：
1. `concat_module_long_seq_kv_tensors_100k` 这种 attr 名包含数字（`100k`）— 是动态字符串生成的 setattr 注册（如 `setattr(self, f"concat_module_long_seq_kv_tensors_{N}k", Module())`）；
2. AST 主路径在 `self_param_aliases` / `dynamic_indexed_containers` 识别上有缺口 — `loop_expansion` 把 `for n in [10, 100, 1000]` 字面量列表展开过头，导致同类多实例（每个 `N` 一份）独立成节点；
3. Rule6_out 检查的是「outbound boundary 边的 var_history 必须含父级 LHS 消费行」。这两条边缺 parent-side consumer line，意味着 producer 端识别正确但 consumer 端的 var_lineage 链断了 — 通常是 wrapper 类内部 `setattr(self, "result_X", x)` 或者 `self.results.append(x)` 这类 dynamic 赋值未被 AST 链接回父类。

**AST 覆盖缺口**：
- `setattr(self, <dynamic_name>, <tensor_var>)` 后续在父类被 `getattr(self, <name>)` 消费的链路追踪
- `loop_expansion` 对字面量数字列表（如 `for w_size in [10, 100, 1000, 10000]`）的展开 — 是否应该展开？还是按 wildcard 处理？

---

## 2. 测例设计（24 + 5 新增 = 29 个 PR3 mvp 测例）

为对应 3 个失败类别，新增 **5 个 synthetic mock 测例**到 `ast_refactor_workdir/testset/synthetic_cases/ast_evaluator/test_constant_resolver.py`（或拆出独立文件 `test_dataflow_edge_cases.py`，建议拆，因为这些不是 ConstantResolver 测试）。

### 2.1 测例文件组织

新增独立测试文件：`ast_refactor_workdir/testset/synthetic_cases/dataflow_edge_cases/test_dataflow_edges.py`

每个测例：
- 构造一个最小 mock 模型源码（写入 tmp 目录）
- 调用 `analyze_trace.py --code-path <tmp> --html-flowchart <tmp_html>`
- 解析 HTML 中的 DATA_INFER JSON
- 用具体断言验证 edges/nodes/cycles

### 2.2 五个新测例

#### **Case A — 测例 25：tuple_unpack_no_lineage_pollution**

**目标**：验证 `q, k, v, g = torch.split(...)` 不污染 var_producers

**Mock 源码**（synthetic_a/main_model.py）：
```python
import torch
import torch.nn as nn

class MyBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.norm = nn.LayerNorm(8)
        self.fuse = nn.Linear(8, 32)
        self.out_proj = nn.Linear(8, 8)
        self.swish = nn.SiLU()

    def forward(self, x):
        x = self.norm(x)              # L_norm
        qkvg = self.fuse(x)           # L_fuse
        q, k, v, g = torch.split(qkvg, [8, 8, 8, 8], dim=-1)
        g = self.swish(g)             # L_swish
        out = q * g
        out = self.out_proj(out)
        out = self.norm(out)          # L_norm_2 — second norm call
        return out

class Root(nn.Module):
    def __init__(self):
        super().__init__()
        self.block = MyBlock()
    def forward(self, x):
        return self.block(x)
```

**断言**（infer DAG）：
1. NO edge `(norm, swish)` with `from_line == L_swish` (i.e., from_line 不能等于 swish 调用行)
2. Expected edge `(norm, swish)` with `from_line == L_norm` (558 等价)，`var ∈ {x, qkvg, g}`，`to_line == L_swish`
3. NO cycle in any SCC: `assert no SCC of size > 1`
4. `swish → norm#1` 边存在（L_swish → L_norm_2）
5. `norm#0 → swish` 边存在（L_norm → L_swish）

**通过条件**：Rule3 PASS，且无 `from_line == to_line` 的可疑边。

---

#### **Case A 补充 — 测例 26：tuple_unpack_with_self_call**

**目标**：验证 `out1, out2 = self.attr(x)` 这种合法 tuple unpacking 仍然正确

**Mock 源码**（synthetic_a2/main_model.py）：
```python
import torch
import torch.nn as nn

class Splitter(nn.Module):
    def __init__(self): super().__init__()
    def forward(self, x):
        return x.chunk(2, dim=-1)

class Root(nn.Module):
    def __init__(self):
        super().__init__()
        self.splitter = Splitter()
        self.up = nn.Linear(8, 16)
        self.down1 = nn.Linear(8, 8)
        self.down2 = nn.Linear(8, 8)
    def forward(self, x):
        x = self.up(x)
        a, b = self.splitter(x)       # tuple unpack from self.attr
        a = self.down1(a)
        b = self.down2(b)
        return a + b
```

**断言**：
1. Edge `(up, splitter)` 存在
2. Edge `(splitter, down1)` 存在
3. Edge `(splitter, down2)` 存在
4. NO edge `(up, down1)` / `(up, down2)` (即 splitter 必须是中间节点)
5. var_producers['a'] / var_producers['b'] = {splitter}（不能漏）

---

#### **Case B — 测例 27：moduledict_fstring_key_split**

**目标**：验证 ModuleDict 用 f-string key 时，多次调用同 attr 会被 split

**Mock 源码**（synthetic_b/main_model.py）：
```python
import torch
import torch.nn as nn

class CycleProneBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.name = "blk"
        self.dim = 8
        # 用 f-string key 注册 ModuleDict
        self.lin_dict = nn.ModuleDict()
        self.norm_dict = nn.ModuleDict()
        for i in range(3):
            self.lin_dict[f"{self.name}_lin_{i}"] = nn.Linear(8, 8)
            self.norm_dict[f"{self.name}_norm_{i}"] = nn.LayerNorm(8)

    def forward(self, x):
        cur = x
        for i in range(3):
            o1 = self.lin_dict[f"{self.name}_lin_{i}"](cur)         # call A_i
            o1_ln = self.norm_dict[f"{self.name}_norm_{i}"](o1)     # call B_i
            o1 = torch.relu(o1_ln)
            cur = self.lin_dict[f"{self.name}_lin_{i}"](o1)         # call C_i (重复同 attr)
        return cur

class Root(nn.Module):
    def __init__(self):
        super().__init__()
        self.block = CycleProneBlock()
    def forward(self, x):
        return self.block(x)
```

**断言**：
1. Rule3: NO cycle (DAG 无环)
2. 节点中存在 `lin_dict[*]#0` 和 `lin_dict[*]#1`（说明 split 生效）OR 存在 per-key 展开节点 `lin_dict[blk_lin_0]`, `lin_dict[blk_lin_1]`, `lin_dict[blk_lin_2]`（说明 f-string key 已被 ConstantResolver 解析）
3. 边方向单调：从 `lin_dict[*]#0` → `norm_dict[*]#0` → `lin_dict[*]#1` 等

**接受双解（split OR 展开）**：因为 f-string key 是 `f"{self.name}_lin_{i}"`，理论上 ConstantResolver 可以把 `self.name="blk"` + `i ∈ {0,1,2}` 展开成 3 个具体 key。如果 AST 主路径能做到，更好；否则回落到 `[*]` + split。

---

#### **Case B 补充 — 测例 28：moduledict_dynamic_key_no_cycle**

**目标**：验证 ModuleDict 即使 key 完全动态（运行时输入），多次调用也不构成假环

**Mock 源码**（synthetic_b2/main_model.py）：
```python
import torch.nn as nn

class DynamicKeyBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer_a = nn.ModuleDict()
        self.layer_b = nn.ModuleDict()
        # 动态注册（运行时不可枚举的 key）
        for k in ["x", "y"]:
            self.layer_a[k] = nn.Linear(4, 4)
            self.layer_b[k] = nn.LayerNorm(4)

    def forward(self, x, key):
        h = self.layer_a[key](x)
        h = self.layer_b[key](h)
        h = self.layer_a[key](h)   # 第二次调用同 attr
        return h

class Root(nn.Module):
    def __init__(self):
        super().__init__()
        self.block = DynamicKeyBlock()
    def forward(self, x, key="x"):
        return self.block(x, key)
```

**断言**：
1. Rule3 PASS（不允许 layer_a ↔ layer_b 双向边）
2. `layer_a[*]` 在 forward 中被识别为 multi-call 候选（cnt=2）
3. split 后存在 `layer_a[*]#0` 和 `layer_a[*]#1`
4. 边方向：`layer_a[*]#0 → layer_b[*]#0 → layer_a[*]#1`（顺序合法 DAG）

---

#### **Case C — 测例 29：dynamic_setattr_outbound_boundary**

**目标**：验证 `setattr(self, f"x_{N}", val)` + 父类 `getattr(self, f"x_{N}")` 的 outbound boundary 边 var_history 含父级 LHS 消费行（覆盖 5800920 Rule6_out 失败）

**Mock 源码**（synthetic_c/main_model.py）：
```python
import torch
import torch.nn as nn

class Aggregator(nn.Module):
    def __init__(self):
        super().__init__()
        # setattr 风格注册多个子 module
        for n in [10, 100, 1000]:
            setattr(self, f"branch_{n}", nn.Linear(8, 8))
        self.merge = nn.Linear(24, 8)

    def forward(self, x):
        outs = []
        for n in [10, 100, 1000]:
            mod = getattr(self, f"branch_{n}")
            outs.append(mod(x))
        # 父级 LHS 消费点：merged = ...
        merged = torch.cat(outs, dim=-1)
        merged = self.merge(merged)
        return merged

class Root(nn.Module):
    def __init__(self):
        super().__init__()
        self.agg = Aggregator()
    def forward(self, x):
        return self.agg(x)
```

**断言**：
1. Rule6_out PASS：所有从 `branch_10/100/1000` 出去的 boundary 边 var_history 必须包含父级 LHS 消费行（`merged = torch.cat(outs, ...)` 这一行）
2. 节点数量合理：Aggregator group 包含 4 个子节点（branch_10, branch_100, branch_1000, merge），不应过度扩展
3. 边 `(branch_10, merge)`、`(branch_100, merge)`、`(branch_1000, merge)` 全部存在

---

### 2.3 主回归测例（test_dag_rules.py 已存在，无需新增）

**修复完成的判定标准**：
- `test_dag_rules.py` 全部 7 个 case PASS（5476790、5547919、5698781、5758056、5800920、5820572、5698781+trace）
- `test_constant_resolver.py` 24/24 PASS（保持不回退）
- `test_dataflow_edges.py` 新增 5 个测例全部 PASS

### 2.4 Rule6_out 假阳性二次验证

5800920 当前的 Rule6_out FAIL 涉及 `concat_module_long_seq_kv_tensors_100k` 和 `long_seq_sim_kv_ln`。若测例 29 通过后主回归仍 FAIL，说明真实模型的 setattr pattern 比 mock 复杂（如多文件、跨类）。届时需进一步插桩定位。

---

## 3. 修复策略提案（**等 NaN 确认后再写代码**）

### 修复点 1：tuple_unpack 行号污染（`analyze_trace.py:4146-4149`）

```python
# 修复前
_line_for_v = phys_lineno
if _ast_var_env.get(v) and _ast_var_env[v][1] is not None:
    _line_for_v = _ast_var_env[v][1]    # ← 删除：盲目覆盖
var_producers[v] = (rhs_producers, _line_for_v)

# 修复后
# 元组解包 LHS 的产生行 = 当前物理行（torch.split 所在行），
# 不能用 _ast_var_env 的「未来某次 self.attr 赋值」覆盖。
var_producers[v] = (rhs_producers, phys_lineno)
```

**额外验证**：搜索其他可能用 `_ast_var_env` 覆盖行号的地方，确保仅在 `<var> = self.<attr>(...)` 这种 single-target 赋值场景使用。

### 修复点 2：multi_call_candidates 与 cycle_nodes 名字空间对齐

需要先插桩验证假设，但很可能问题是：
- `_method_call_count` 用 `recycle_layers_[*]`
- `cycle_nodes` 用 `recycle_layers_` 或别的形式

修复方法：统一在 edge_set 与 _multi_call_candidates 的同一命名空间下做交集（统一规范化 attr 名）。

### 修复点 3：dynamic setattr 的 outbound boundary var_history

补齐 `setattr(self, <fstring>, <var>)` + `getattr(self, <fstring>)` 链路在父类调用点的 LHS 消费行追踪。

---

## 4. 工作流程（NaN 确认后）

1. **基线备份**：
   ```bash
   cp ast_refactor_workdir/scripts/analyze_trace.py \
      ast_refactor_workdir/scripts/analyze_trace_pr3_b_class_fix_BASELINE.bak
   ```
2. **新增测例文件**：`testset/synthetic_cases/dataflow_edge_cases/test_dataflow_edges.py`（5 个 mock 测例）
3. **修复点 1**（tuple unpack 行号），跑测例 25/26 PASS
4. **修复点 2**（multi_call splitter 名字空间），跑测例 27/28 PASS
5. **修复点 3**（dynamic setattr boundary），跑测例 29 PASS
6. **主回归全跑**：`python3 testset/test_dag_rules.py` 必须 7/7 PASS
7. **24/24 ConstantResolver 测例不回退**：`python3 ast_refactor_workdir/testset/synthetic_cases/ast_evaluator/test_constant_resolver.py` 24 PASS
8. **存档基线**：`analyze_trace_pr3_b_class_fixed_BASELINE.py`

---

## 5. 风险与边界

- **风险 1**：修复点 1（删除 _ast_var_env 行号覆盖）可能影响其他 case。需要主回归全量验证，特别是 5476790、5758056（当前已 PASS）不回退。
- **风险 2**：修复点 2 若把 multi_call splitter 触发频率提升，可能产生过多 `#N` 节点污染图。需要确认 split 仅作用于真正成环的 attr。
- **边界**：本设计文档不涉及 frontend_html 的修改，纯 analyze_trace.py 的 DAG 构建逻辑修复。

---

## 6. 待 NaN 确认的关键决策点

1. ✅ / ❌ 是否同意将测例 25-29 单独建文件 `test_dataflow_edges.py`？（不混进 ConstantResolver 测试）
2. ✅ / ❌ 修复点 1 的方案（直接删除 `_ast_var_env` 行号覆盖）是否同意？
3. ✅ / ❌ 测例 27 接受「split 节点」OR「f-string 展开节点」二者其一通过？
4. ✅ / ❌ 修复点 2 是否需要先插桩验证 `_multi_call_candidates` 与 `cycle_nodes` 名字空间错位假设？
5. ✅ / ❌ 测例 29 失败时若发现真实模型场景更复杂，是否同意继续补齐测例（迭代式）？
