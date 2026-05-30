# Edge & 高亮重构设计

## 0. 背景

本文档记录在 AST DAG 重构（`ModuleAttr`/`ContainerAttr` 体系）中，对 Edge 类型、高亮 evidence 来源的统一重新设计。属于综合重构的一部分，与 AST 构图重构在同一分支下完成。

---

## 1. Attr 类型体系

新构图引入四种一等公民 Attr 类型，均支持 Container 级别嵌套，leaf 实例唯一对应一个 `CallLoc`：

```
ModuleAttr    → 用户 nn.Module attr
ContainerAttr → ModuleList / ModuleDict 等容器（支持嵌套展开）
InputAttr     → LG input source（LG.dense_feature / LG.get_sample_rate / LG.feature_column / LG.get_bias 等）
ResultAttr    → result.head attr
```

---

## 2. Edge 类型

```
InputAttr  → ModuleAttr    ("dep")   —— Input edge
ModuleAttr → ModuleAttr    ("dep")   —— 内部数据流 edge
ModuleAttr → ResultAttr    ("dep")   —— Result edge
```

三种 edge 统一结构，都携带 `evidence.lineage_by_carrier`，后端统一产出，前端只负责渲染。

**废弃**：`_scan_root_result_edges()`、`_input_edge_var_history()` 两个前端扫描器在新构图实现后不再需要。

---

## 3. InputAttr 特殊处理

### 3.1 扫描范围

LG 全量扫描所有源文件（包含 global var 定义），不局限于 Root class `__init__`。

### 3.2 双 Loc 记录

`InputAttr` 需要记录两个位置：

| 字段 | 含义 | 用途 |
|---|---|---|
| `call_loc` | `__init__` 里 `self.xxx = LG.dense_feature(...)` 定义行 | 节点展示：点击节点跳到注册位置 |
| `forward_use_loc` | `forward()` 里 `fc = self.xxx()` 执行行 | edge evidence 起点行号 |

### 3.3 Edge Evidence 构建

Input edge 的 `lineage_by_carrier` 在链首插入一条 `role="input_source"` step，指向 `call_loc`（init 定义行），之后由 `var_lineage` 从 `forward_use_loc` 开始正常追踪 carrier 链到 consumer module。

```
[init 定义行]   self.v3l3tag_module = LG.dense_feature(...)  role="input_source"
      ↓
[forward 执行行] fc = self.v3l3tag_module()                  role="producer"
      ↓
[中间传递步骤]   out_fc = fc + bias                           role="step"
      ↓
[consumer 调用行] out = self.dense(out_fc)                   role="consumer"
```

global var 场景：source step 指向 global var 定义行，forward_use_loc 指向 forward() 内使用行。

**`var_lineage` 机制本身不改动**，只在构建 Input edge evidence 时在链首插入 source step。

---

## 4. ResultAttr

- `call_loc` = `result.head(...)` 调用行
- 直接复用 `var_lineage` 追踪 carrier（prediction/loss/sample_rate 等 kwargs）到 producer module
- 暂不做 init 路径特殊处理

---

## 5. `first_call_loc` 消除

新构图下，每个 leaf `ModuleAttr` / `InputAttr` / `ResultAttr` 实例唯一对应一个 `CallLoc`，不存在"同一个 attr 被多次调用取第一次"的歧义。`first_call_loc` 字段在新构图中完全消除。

---

## 6. 高亮逻辑

前端高亮逻辑不变（`lineage_by_carrier` 折叠面板 + `renderCodeBlock()`），变化的是 evidence 数据来源：

| 路径 | 现有实现 | 新实现 |
|---|---|---|
| module→module edge | 后端 `var_lineage`（已有） | 不变 |
| Input→module edge | 前端 `_input_edge_var_history()` 临时合成 | 后端产出，链首加 input_source step |
| module→Result edge | 前端 `_scan_root_result_edges()` 临时合成 | 后端产出，复用 `var_lineage` |

"没有 lineage" 和 "flow 边" 两种 fallback 路径：
- **没有 lineage**：新构图下 Input/Result edge 都有后端 evidence，module→module edge 有 `var_lineage`，正常路径覆盖率大幅提升，fallback 仅保留给真正无法静态追踪的复杂表达式
- **flow 边**：仅保留给动态生成类（无真实 `forward()` 源码），属于合法弱 evidence，保留现有展示逻辑

---

## 7. 实现顺序（在 AST 重构 Phase C 阶段）

1. LG 全量扫描器：全文件扫描，建 `InputAttr` 对象，记录 `call_loc` + `forward_use_loc`
2. ResultAttr 识别：从 `result.head(...)` 调用行建 `ResultAttr`
3. 后端 Input edge evidence 构建：在 `_build_data_dependency_edges()` 中为 Input edge 产出 `lineage_by_carrier`，链首插 input_source step
4. 后端 Result edge evidence 构建：复用 `var_lineage` 产出 Result edge evidence
5. 前端扫描器废弃：删除 `_scan_root_result_edges()` 和 `_input_edge_var_history()`
