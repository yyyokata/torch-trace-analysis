# AST Parse 调用统一化评估报告

## 1. 调用点清单 (analyze_trace.py)

| 行号 | 函数/上下文 | 输入类型 | 用途 | 缓存状态 |
| :--- | :--- | :--- | :--- | :--- |
| **52** | `ASTFrontend.__init__` | 全文件源码 | AST 元数据前端初始化，构建类/方法注册表。 | 局部缓存（多个函数内有独立的 `ast_frontends` 字典）。 |
| **904** | `build_module_like_set` | 全文件源码 | 扫描所有文件以识别 `nn.Module` 子类及其继承关系。 | **无**（每次调用都重新 parse 所有文件）。 |
| **1005** | `_AST_ModuleTreeExtractor._parse_all_files` | 全文件源码 | 深度解析 `__init__` 赋值以构建模块树。 | **无**（独立于其他模块重新 parse）。 |
| **2415** | `_rhs_last_called_attr` | 表达式字符串 | `mode="eval"` 解析。分析赋值语句右值的 Producer。 | **无**（片段解析）。 |
| **5875** | `_parse_local_stmt` | 逻辑行字符串 | 语句解析。用于追踪 `__init__` 中的局部变量构造。 | **无**（片段解析）。 |

---

## 2. Q1：全文件 parse (52/904/1005) 是否能统一？

### 评估结论：**完全可行且强烈推荐**。

- **重叠分析**：这三处 parse 的输入完全一致（均来源于 `source_files` 字典）。
  - `build_module_like_set` (904) 只需要类名和 bases。
  - `_AST_ModuleTreeExtractor` (1005) 需要 `ClassDef` 节点及其 `__init__` 内容。
  - `ASTFrontend` (52) 已经提供了涵盖上述所有需求的功能（`class_registry` 存储了 `ClassDef` 节点，且有 `get_init_assignments_ast` 等接口）。
- **合并可行性**：
  - 目前架构下，一个文件会被 parse 至少 3 次：启动时扫基类一次、提取模块树一次、主循环处理类时通过 `ASTFrontend` 又 parse 一次。
  - 建议引入**全局/集中式 ASTFrontend 缓存池**。在 `build_static_module_tree` 启动时一次性（或 lazy）构建所有文件的 `ASTFrontend` 实例，后续所有组件直接共享这些实例。
- **效益**：大幅减少 CPU 耗时（解析数千个文件时，AST parse 是主要瓶颈），并统一了“类-文件”映射逻辑。

---

## 3. Q2：片段/单行 parse (2415/5875) 是否可以消除？

### 评估结论：**可行，且有助于消除位置偏差风险**。

- **行 2415 (`mode="eval"`)**：
  - **现状**：将 RHS 表达式文本重新 parse 以获取 `ast.Call` 节点。
  - **改进**：由于 `build_static_module_tree` 已经在 `_ast_stmt_info_by_line` 中按行号索引了语句的 `stmt_info`，而该 info 中已经包含了完整的 `ast.Assign` 节点。直接传递 `node.value` 给分析函数即可，无需重新 parse 字符串。
- **行 5875 (`_parse_local_stmt`)**：
  - **现状**：为了拿到一个独立的 AST 节点，代码手动拼接逻辑行并 parse。这导致产生的节点 `lineno=1`，无法直接使用基于文件偏移的源码提取工具，被迫额外维护了一个使用 `ast.unparse` 的 `_node_to_text` (行 5892)。
  - **改进**：既然 `ASTFrontend` 已经持有了全文件 AST 树，完全可以根据 `(cname, mname)` 找到对应的 `FunctionDef`，再通过行号匹配找到原始 AST 节点。
  - **优势**：复用 full AST 节点后，可以统一使用 `ASTFrontend._node_to_text`。该方法在有位置信息时使用 O(1) 的切片提取，比 `ast.unparse` 更快且更精确（保留原始注释和格式）。

---

## 4. Q3：改动量评估、阻力分析与迁移次序

### 关键改动位置 (按建议修改次序)

1.  **`ASTFrontend` 实例池化 (小改动)**:
    - 在 `build_static_module_tree` 顶层创建一个 `ast_frontends: Dict[str, ASTFrontend]`。
    - 修改各处的 `_get_ast_frontend(fname)` 本地 helper，使其指向该统一缓存。
2.  **重构 `build_module_like_set` (中等改动)**:
    - 接口改为接收 `ast_frontends` 字典。
    - 遍历 frontend 实例的 `class_registry` 来替代 `ast.parse`。
3.  **重构 `_AST_ModuleTreeExtractor` (中等改动)**:
    - 移除 `_parse_all_files` 中的 `ast.parse`。
    - 直接从 `ast_frontends` 中获取 `ClassDef` 节点。
4.  **优化 `_rhs_last_called_attr` 与 `_parse_local_stmt` (小改动)**:
    - 调整调用链，优先传递已有的 AST 节点而非表达式文本。

### 估算改动量级
- **中等 (100–200 行)**。主要工作在于接口契约的调整和缓存对象的透传。

### 明显阻力
- **数据格式差异**：现有代码中 `source_files` 的值多为 `List[str]` (lines)，而 `ASTFrontend` 构造函数需要 `str` (source)。统一时需要注意 `"\n".join(lines)` 的开销，建议在 `ASTFrontend` 内部做 lazy 处理。
- **位置独立性验证**：行 5875 原本依赖 `lineno=1` 的特性（虽然是为了绕过 bug）。切换到 full AST 后，需确保所有 `_node_to_text` 调用都经过了 `ASTFrontend` 封装的 fast-path 逻辑。
- **循环依赖担忧**：由于 `ASTFrontend` 已经定义在 `analyze_trace.py` 顶部，不存在模块间循环引用阻力。

---

## 5. 总体建议

**结论：强烈推荐统一为“一次 Parse + 多次共享”架构。**

**理由：**
1. **性能**：消除 3 倍以上的冗余全文件解析开销。
2. **正确性**：消除 snippet parse 带来的 `lineno` 偏移问题，使得源码提取逻辑（`_node_to_text`）能够统一化。
3. **可维护性**：目前多处定义的 `_get_ast_frontend` 闭包函数导致缓存互不通气，统一后可以简化为单一的单例/池化模式。
4. **架构演进**：符合 `Scenario 5 Design` 中提出的“新增抽象层 `ConstantTable` 挂载在 `ASTFrontend` 之上”的长远规划。
