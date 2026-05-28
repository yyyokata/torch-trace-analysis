
## AST Step2 规划（2026-05-28）

### 两份规划对应关系

两份规划层次不同，不是互斥关系：
- **旧 Phase C/D/F/G** = 架构清理/技术债收口
- **新 Step2-P1/P2/缺口C/D** = ConstantResolver 能力补齐

| 旧 Phase | 对应新规划 | 说明 |
|---|---|---|
| Phase C（删 _parse_local_stmt，搬 8 个 helper） | 无直接对应 | 纯架构清理，P1/P2 工程地基，但不是前置阻塞 |
| Phase D（删所有 regex fallback） | Step2-P1 + Step2-P2 的上层 | P1/P2 是 Phase D 里最影响正确性的子集 |
| Phase F（EvalTrace + --debug-eval） | 缺口C | 高度重叠；缺口C 是 Phase F 的轻量落地版 |
| Phase G（_eval_cache memo） | 已完成 | 脚本里 _eval_cache 已实现，不需要排进 Step2 |

缺口D（清理 class_init_param_anno）：防止后续 P1/P2 改 resolver 时误引入 annotation fallback。

### Step2 主线（当前阶段）

| 优先级 | 任务 | 来源 | 原因 |
|---|---|---|---|
| P0 | 清理 class_init_param_anno 字段/注释 | 缺口D | 改动最小，防 annotation fallback 风险 |
| P1 | 实例级普通形参传播（Block(2)→range(n)=2） | Step2-P1 | 最明确的 correctness 缺口 |
| P2 | list_len 多级属性链（self.config.names） | Step2-P2 | 脚本注释明确写“not supported in PR1” |
| P3 | failure reason counter（_fail(reason)） | 缺口C / Phase F 轻量版 | 和 P1/P2 并行 |

### Step2 后的 AST Cleanup 专项（稍后）

| 优先级 | 任务 | 来源 |
|---|---|---|
| P4 | 删 _parse_local_stmt，搬 8 个 helper 到 ASTFrontend | Phase C |
| P5 | 全量 regex fallback 删除 | Phase D 剩余 |
| P6 | EvalTrace dataclass + --debug-eval CLI | Phase F 完整版 |

**P8：torch native stub 深度补全**

目标：完全覆盖 torch 原生 nn.Module（nn.Linear、nn.Embedding 等）在各种注册方式下的 leaf 识别。

当前已覆盖：
- `self.xxx = nn.Linear(...)` 直接赋值路径

尚未覆盖的缺口（按优先级）：

| 优先级 | 缺口 | 影响 |
|---|---|---|
| 🔴 高 | `setattr(self, name, nn.Linear(...))` 动态注册 | setattr 路径对 `nn.*` 类型不走 `_is_nn_leaf_stub` 判断，误入 `dynamic_attr` |
| 🔴 高 | `nn.ModuleList([nn.Linear(...), ...])` 容器内 native 元素 | ModuleList 展开时子元素是 native class 时不识别为 leaf，直接丢失 |
| 🟡 中 | `torch.nn.Linear` / `from torch.nn import Linear` 等别名前缀 | 非标准 `nn.` 前缀的 import 方式会 miss |
| 🟢 低 | `nn.Sequential` 内嵌 native 子层 | 场景较少 |

实现思路：
1. setattr 路径：在 `dynamic_attrs_per_class` 收集时，对 `cls_ref` 做 `_is_nn_leaf_stub` 判断，命中时改为写入 `torch_native_module_classes`
2. ModuleList 元素：在 ModuleList 展开逻辑中，对元素 `cls_ref` 做 native 判断，命中时补入 native leaf
3. 别名前缀：扩展 `_is_nn_leaf_stub` 的匹配逻辑，支持 `torch.nn.xxx` 完整前缀和 `from torch.nn import xxx` 导入形式
