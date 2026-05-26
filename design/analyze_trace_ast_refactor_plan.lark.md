## 目标与结论

这份计划基于对 `torch-trace-analyzer/scripts/analyze_trace.py`、`torch-trace-analyzer/scripts/fake_runner.py` 和已有 AST 分析报告的通读整理，目标是把 **正则+启发式主干** 逐步迁移为 **AST 驱动、文本 fallback 保底**。

<callout icon="bulb" bgc="3">  
  **推荐总策略：**  
  1. 先收口 **类/方法边界元数据层**；  
  2. 再改 **模块注册 / 容器展开 / dynamic setattr**；  
  3. 最后改 **forward 数据流与 evidence 定位**。  
  
  每个函数改造后，都用 Iter17 基线产物做逐模型 diff：`nodes / groups / top_edges / internal_edges / Rule1~Rule12 / var_history / first_call_loc / edge evidence`。  
</callout>

## 优先级总览

<table header-row="true" header-col="false" col-widths="180,110,220,120,160">  
  <tr>  
    <td>优先级</td>  
    <td>函数/逻辑</td>  
    <td>改造目标</td>  
    <td>难度</td>  
    <td>是否可复用 fake_runner</td>  
  </tr>  
  <tr>  
    <td>P0</td>  
    <td>`_build_class_map()`</td>  
    <td>统一类/方法边界，建立 AST 元数据源</td>  
    <td>低</td>  
    <td>是</td>  
  </tr>  
  <tr>  
    <td>P0</td>  
    <td>`build_static_module_tree()`</td>  
    <td>主入口改为消费 AST registry，而不是重复文本扫描</td>  
    <td>中</td>  
    <td>是</td>  
  </tr>  
  <tr>  
    <td>P0</td>  
    <td>`_build_source_dependency_order()`</td>  
    <td>改为 AST call collector 的副产物</td>  
    <td>中</td>  
    <td>部分可复用</td>  
  </tr>  
  <tr>  
    <td>P1</td>  
    <td>`_extract_kw_int_args()` / `_eval_int_atom()` / `_eval_list_len()` / `_extract_kw_list_lens()` / `_resolve_range_n()`</td>  
    <td>把容器规模推断改成受限 AST 求值器</td>  
    <td>中</td>  
    <td>部分可复用</td>  
  </tr>  
  <tr>  
    <td>P1</td>  
    <td>`_parse_list_ctor_items()` / `_parse_dict_ctor_items()` / `_resolve_str_list()` / `_expand_fstring_template()`</td>  
    <td>把容器元素与动态命名的文本解析改成 AST 表达式解析</td>  
    <td>中</td>  
    <td>部分可复用</td>  
  </tr>  
  <tr>  
    <td>P1</td>  
    <td>`build_static_module_tree()` 中 `first_call_loc` 段</td>  
    <td>把调用点定位改为 AST call-site span</td>  
    <td>中</td>  
    <td>部分可复用</td>  
  </tr>  
  <tr>  
    <td>P2</td>  
    <td>`_build_data_dependency_edges()`</td>  
    <td>forward/helper 数据流改为 AST 解释器</td>  
    <td>高</td>  
    <td>否</td>  
  </tr>  
  <tr>  
    <td>P2</td>  
    <td>`_collect_dict_reads()` / `_collect_expr_producers()` / `_parse_dict_literal_items()` / `_rhs_last_called_attr()`</td>  
    <td>删除文本猜测，改为 AST expression evaluator</td>  
    <td>高</td>  
    <td>否</td>  
  </tr>  
</table>

---

## Phase 划分建议

### Phase A：先建立统一 AST 元数据层
- 目标：先让 `class_map`、method ranges、module class registry、`__init__` assignments 有一个统一可信来源。
- 预期收益：后续 `build_static_module_tree()` 和 `_build_data_dependency_edges()` 都不再自己重复做边界识别。
- 验证重点：`class_map`、`method_ranges`、`attr_def_loc` 是否与基线一致或更准。

### Phase B：改造模块注册与容器展开
- 目标：替换 `self.xxx = ...`、`setattr(...)`、append、ModuleList/ModuleDict/Sequential 相关正则扫描。
- 预期收益：Iter17 的 DenseTower / dynamic setattr / per-instance 裁剪逻辑更稳定。
- 验证重点：`attrs`、`container_elems`、`container_kinds`、`dynamic_setattr_attrs`、`class_str_list_attrs`。

### Phase C：改造 forward 数据流与 evidence
- 目标：替换 `_build_data_dependency_edges()` 内部对 tuple unpack、dict slot、tensor alias、helper 可达性、first_call_loc 的文本解析。
- 预期收益：Rule1/1b/1c、Rule6/6_out、ProdConsCompleteness 的稳定性显著提升。
- 验证重点：`dep_edges`、`dep_edge_locs`、`var_history`、`lineage`、`first_call_loc`。

---

## 逐函数改造清单（按优先级排序）

## 函数名：_build_class_map()
- 位置：`analyze_trace.py` L837-L876
- 当前实现方式：
  - 纯正则 + 缩进启发式。
  - 关键 pattern：`^class (\w+)\(`、`def (\w+)\(`。
  - 结束位置靠“回到 0 缩进 / 新 class 出现”来回推。
- 已知局限/bug 场景：
  - `class X:` 无显式基类时会漏。
  - 装饰器、多行函数签名、`async def`、嵌套定义、复杂注释场景都可能错位。
  - 方法结束位置是文本回推，不是语义边界。
- AST 改造方案：
  - 直接遍历 `ast.Module.body` 中的 `ast.ClassDef`。
  - 用 `ast.FunctionDef` / `ast.AsyncFunctionDef` 记录 `methods`。
  - 使用 `lineno` / `end_lineno` 精确记录类、方法边界。
  - 额外输出统一的 AST registry，例如：`class_ast_map[(fname, cname)] = {start,end,methods,bases,node}`。
- 改造后预期输出变化：
  - `class_map` 的起止行会更精确，尤其是多行签名和装饰器场景。
  - 极少数过去被误扫进 class 的行会被排除。
- 影响的下游规则/字段：
  - `class_map`
  - `method_ranges`
  - `build_static_module_tree()` 扫描范围
  - `_build_data_dependency_edges()` 的 forward/helper 采样范围
- 改造难度：低
- 是否可以复用 fake_runner.py 的现有代码：是

## 函数名：build_static_module_tree()
- 位置：`analyze_trace.py` L4026-L5771
- 当前实现方式：
  - 主入口仍以正则/拼接文本为主，只把 `_AST_ModuleTreeExtractor` 当附加元数据。
  - 关键手法：`_join_logical_lines()`、`_split_top_level()`、多轮 method scan、后处理补洞。
- 已知局限/bug 场景：
  - 同一语义被多套逻辑重复识别：类继承、attr 注册、容器展开、dynamic setattr。
  - 复杂多行表达式、注释、嵌套 call、控制流分支场景仍依赖字符串恢复。
  - AST 已存在但不是唯一事实源，维护成本高。
- AST 改造方案：
  - 把 `_AST_ModuleTreeExtractor` 升级为主数据源，而不是旁路元数据。
  - `build_static_module_tree()` 只负责 orchestration：消费 AST registry、合并容器展开、拼装 tree。
  - 逐步移除文本正则扫描，把文本层保留为 fallback / diff 对照开关。
- 改造后预期输出变化：
  - `attrs`、`attr_def_loc`、`container_elems` 的来源变得稳定。
  - 初期目标应是 **零统计变化**；后续只接受“更准的行号 / 证据位置”变化。
- 影响的下游规则/字段：
  - `tree`
  - `roots`
  - `attrs`
  - `attr_def_loc`
  - `container_kinds`
  - `input_source_attrs`
  - `first_call_loc`
- 改造难度：中
- 是否可以复用 fake_runner.py 的现有代码：是

## 函数名：_build_source_dependency_order()
- 位置：`analyze_trace.py` L879-L940
- 当前实现方式：
  - 文本扫描 `__init__` 中 `self.attr = ClassName(...)`。
  - 文本扫描 `forward()` 中 `self.attr(...)`、`self.attr[idx](...)`、`getattr(self, 'attr')(...)`。
- 已知局限/bug 场景：
  - 只能看到 forward 文本，不理解 helper-method 调用链。
  - 同行多调用、嵌套调用、comprehension 中调用顺序不可靠。
  - 动态 `getattr(self, name_expr)` 只能处理字面量字符串。
- AST 改造方案：
  - 从统一 AST call collector 产出 `module_call_order`。
  - 遍历 `ast.Call`，识别 callee 形态：
    - `ast.Attribute(value=Name('self'), attr=...)`
    - `ast.Subscript(Attribute(self, cont), idx)`
    - `ast.Call(func=Name('getattr'), args=[self, ...])`
  - 用 `lineno + col_offset` 维持真实顺序。
  - helper 可达性由 method call graph 决定，而不是文本 grep。
- 改造后预期输出变化：
  - `module_call_order` 可能更长，能覆盖 helper 中真实调用。
  - 某些误判顺序会收敛。
- 影响的下游规则/字段：
  - `children_ordered`
  - `first_call_loc`
  - DAG 节点展示顺序
- 改造难度：中
- 是否可以复用 fake_runner.py 的现有代码：部分可复用（类/方法 AST 遍历可复用，call graph 需新增）

## 函数名：_eval_int_atom()
- 位置：`analyze_trace.py` L4123-L4136
- 当前实现方式：
  - 正则识别数字字面量；再查文件级/全局常量表。
- 已知局限/bug 场景：
  - 不支持 `A + 1`、`max_n - 1`、`self.layers` 这类受限表达式。
  - 常量收集仍来自文本扫描。
- AST 改造方案：
  - 作为“受限 AST 求值器”的基础函数。
  - 支持 `ast.Constant`、`ast.Name`、`ast.Attribute(self.x / mod.CONST)`、必要时支持 `ast.UnaryOp`/`ast.BinOp`。
  - 常量表也改由 AST module-level assignment 提供。
- 改造后预期输出变化：
  - 能解析更多 `range(N)` / kw=int 场景，减少 wildcard。
- 影响的下游规则/字段：
  - `ctor_kw_int_args`
  - `_resolve_range_n()`
  - 容器展开数量
- 改造难度：低
- 是否可以复用 fake_runner.py 的现有代码：部分可复用（AST 遍历框架可复用，求值器需新增）

## 函数名：_extract_kw_int_args()
- 位置：`analyze_trace.py` L4138-L4163
- 当前实现方式：
  - 正则从 call 文本中抽 `kw=ATOM`。
  - 关键 pattern：`\b(\w+)\s*=\s*([A-Z][A-Z0-9_]*|\d+)\b`。
- 已知局限/bug 场景：
  - 依赖“先移除字符串，再做近似匹配”。
  - 无法稳健处理复杂 kw 表达式、嵌套 call、多行参数和注释。
- AST 改造方案：
  - 直接处理 `ast.Call.keywords`。
  - 对每个 keyword.value 调用受限求值器。
  - 保留“只支持安全子集”的策略，不做全解释器。
- 改造后预期输出变化：
  - `ctor_kw_int_args` 更完整、更少误匹配。
- 影响的下游规则/字段：
  - `ctor_kw_int_args`
  - `_resolve_range_n()`
  - `build_static_module_tree()` 容器展开
- 改造难度：低
- 是否可以复用 fake_runner.py 的现有代码：部分可复用

## 函数名：_eval_list_len()
- 位置：`analyze_trace.py` L4207-L4322
- 当前实现方式：
  - 文本递归求值，支持 `[a,b] + [c]`、`[x] * N`、常量 list 名称等。
  - 依赖 `_split_top_level()` 和正则。
- 已知局限/bug 场景：
  - 本质在手写迷你 parser。
  - 对 `ListComp`、条件表达式、嵌套容器、变量别名支持弱。
- AST 改造方案：
  - 处理 `ast.List`、`ast.Tuple`、`ast.BinOp(Add/Mult)`、`ast.Name`、`ast.Attribute`、`ast.ListComp`（限定 `range/ enumerate / 常量列表`）。
  - 输出“可解析长度”或 `None`，继续保持保守策略。
- 改造后预期输出变化：
  - `ctor_kw_list_lens` 更稳定。
  - Iter17 的 per-instance 裁剪覆盖率会上升。
- 影响的下游规则/字段：
  - `_extract_kw_list_lens()`
  - `_resolve_iter_len()` / `_resolve_range_n()`
  - `instance_kw_list_lens`
  - `container_elems`
- 改造难度：中
- 是否可以复用 fake_runner.py 的现有代码：否（需要新增 AST 求值逻辑）

## 函数名：_extract_kw_list_lens()
- 位置：`analyze_trace.py` L4324-L4341
- 当前实现方式：
  - 先 `_split_top_level(call_text)`，再用正则抓 `kw=value`。
- 已知局限/bug 场景：
  - 对嵌套表达式高度依赖字符串分割是否正确。
  - 复杂 keyword 顺序和跨行写法容易变脆。
- AST 改造方案：
  - 直接消费 `ast.Call.keywords`。
  - 对 `keyword.value` 调 `_eval_list_len_ast()`。
- 改造后预期输出变化：
  - `ctor_kw_list_lens` 更全，误差更小。
- 影响的下游规则/字段：
  - `ctor_kw_list_lens`
  - `instance_kw_list_lens`
  - DenseTower / GroupTower 展开裁剪
- 改造难度：低
- 是否可以复用 fake_runner.py 的现有代码：部分可复用

## 函数名：_parse_list_ctor_items()
- 位置：`analyze_trace.py` L4543-L4549
- 当前实现方式：
  - 顶层逗号切分后，用 `^\s*([\w.]+)\s*\(` 找 `Class(...)`。
- 已知局限/bug 场景：
  - 只认“元素就是裸 ctor call”。
  - 对变量引用、条件表达式、list comprehension、helper 返回值无能为力。
- AST 改造方案：
  - 直接解析 `ast.List` / `ast.Tuple` 元素。
  - 对元素调用 `_infer_class_from_value_ast()`。
  - 对 `ast.Name` 走局部变量环境；对 `ast.ListComp` 走受限展开。
- 改造后预期输出变化：
  - `container_elems` 更完整，wildcard 减少。
- 影响的下游规则/字段：
  - `container_elems`
  - `container_kinds`
  - Rule6 / Rule6_out 相关路径完整性
- 改造难度：中
- 是否可以复用 fake_runner.py 的现有代码：部分可复用（`_infer_class_from_value()` 可复用）

## 函数名：_parse_dict_ctor_items()
- 位置：`analyze_trace.py` L4551-L4590
- 当前实现方式：
  - 手工按顶层逗号和第一个冒号切 `{'k': V()}`。
- 已知局限/bug 场景：
  - 不支持复杂 key/value 表达式。
  - dict merge、条件 value、helper 返回、变量 key 都容易退化。
- AST 改造方案：
  - 直接处理 `ast.Dict`。
  - `key` 支持 `ast.Constant(str)`、必要时支持受限字符串表达式。
  - `value` 交给 `_infer_class_from_value_ast()` 或局部变量解析。
- 改造后预期输出变化：
  - ModuleDict 元素识别更稳定，`[*]` 回退减少。
- 影响的下游规则/字段：
  - `container_elems`
  - `dynamic getattr` 精度
  - Rule3 假环风险
- 改造难度：中
- 是否可以复用 fake_runner.py 的现有代码：部分可复用

## 函数名：_resolve_str_list()
- 位置：`analyze_trace.py` L4627-L4660
- 当前实现方式：
  - 正则识别 loop var / `self.attr` / `UPPER_CASE` 常量。
- 已知局限/bug 场景：
  - 对 `prefix + suffix`、局部变量转存、f-string、dict key 派生场景支持弱。
  -  **TODO（LayerNorm / 动态 ModuleDict 字符串 key 展开）**：当前规划已覆盖“f-string / dynamic key / ModuleDict 展开”大方向，但尚未明确落到 `_loop_var_to_str_items` 的多步推导能力。已知根因是：当 ModuleDict key 需要先由 `loop var -> 中间字符串变量`，再经过 `if/else` 或条件表达式改写成最终 key 时，现有 `_loop_var_to_str_items` 只能看单步映射，导致像并行 LayerNorm 这类节点退化成 `[*]`，无法按实例展开。修复方向：在 Phase B 为 `_loop_var_to_str_items` 新增 AST 驱动的**迭代多步推导**——从 loop var 出发，沿赋值链反复展开（每一步如果结果仍未解析为字面量集合则继续推导），直到收敛到一组具体字符串 key 或确认无法静态枚举为止；不限制推导步数，支持中间经过任意次条件改写、变量转存、f-string 拼接等，并与 `_resolve_str_list()` / `_expand_fstring_template()` 共用 method-scope env。
  -  **TODO（训练图 / 推理图区分）**：当模型代码存在 `if self.training: ... else: ...` 这类分支时，`_loop_var_to_str_items` / `_resolve_str_list()` 在迭代推导过程中可能同时收集到训练路径与推理路径的 key，导致两张图混用同一套节点展开，无法区分。可评估的设计思路是：在 AST 迭代推导启动前，先向 method-scope env 注入“训推区分变量”的提前赋值（例如训练图模式设 `env["self.training"] = True`，推理图模式设 `env["self.training"] = False`）；推导时若遇到 `ast.If`，则依据 env 中的极性裁剪分支，只保留当前图对应的路径，使 key 枚举自然收敛到 train / infer 各自独立的集合。类似 `is_training`、`training_mode`、`phase == "train"` 等常见训推区分变量，也可纳入 env 初始化范围，并复用 `_cond_branch_polarity()` 的现有极性识别逻辑。后续需要评估：① `_cond_branch_polarity()` 现有能力是否足够复用；② 应在 key 枚举阶段直接分叉 train/infer env，还是在 `build_dag_recursive` 级别维护两套独立 env；③ 对无法静态确定训推分支的场景，继续保留现有 union 行为作为兜底。
  
- AST 改造方案：
  - 接受 AST expression 而不是字符串。
  - 支持 `ast.Name`、`ast.Attribute`、`ast.Constant(str)`、必要时受限支持 `ast.JoinedStr` / `ast.BinOp(Add)`。
  - 直接对接 method-scope env 与 file-level constant env。
- 改造后预期输出变化：
  - dynamic `setattr/getattr` 的实名展开率提高。
- 影响的下游规则/字段：
  - `dynamic_setattr_attrs`
  - tensor alias 名称解析
  - `first_call_loc` 和 `dep_edges` 的精度
- 改造难度：中
- 是否可以复用 fake_runner.py 的现有代码：否

## 函数名：_cond_branch_polarity()
- 位置：`analyze_trace.py` L4404-L4458
- 当前实现方式：
  - 关键词启发式：`is_training` / `training` / `is_serving` / `serving`，再配合 `not` 翻转。
- 已知局限/bug 场景：
  - 无法稳健处理 `and/or`、括号、比较表达式、helper bool 变量。
  - 现在的分支识别本质还是文本 polarity 判定。
- AST 改造方案：
  - 解析 `ast.If.test`。
  - 做一个“只识别训练/推理极性”的布尔表达式归一化器。
  - 输出 `train / infer / unknown`，避免复杂场景误判。
- 改造后预期输出变化：
  - `conditional_attrs` 分支归属更稳定。
  - 训练/推理双态模型误连边概率下降。
- 影响的下游规则/字段：
  - `class_conditional_attrs`
  - `attrs`
  - reachable subtree
- 改造难度：中
- 是否可以复用 fake_runner.py 的现有代码：否

## 函数名：_resolve_range_n()
- 位置：`analyze_trace.py` L5176-L5196
- 当前实现方式：
  - 依赖 `_eval_int_atom()` + `self_to_param` + `ctor_kw_int_args`。
- 已知局限/bug 场景：
  - 只能处理很窄的来源链，无法理解更丰富的 int expression。
- AST 改造方案：
  - 作为受限求值器的组合接口，统一支持 `Name` / `self.attr` / const / 简单运算。
  - 输入改为 AST node，避免重复字符串解析。
- 改造后预期输出变化：
  - `range()` 展开命中率提升，wildcard 降低。
- 影响的下游规则/字段：
  - `container_elems`
  - `instance_kw_list_lens`
  - `class_container_kw`
- 改造难度：低
- 是否可以复用 fake_runner.py 的现有代码：否

## 函数名：_expand_fstring_template()
- 位置：`analyze_trace.py` L5379-L5408
- 当前实现方式：
  - 正则抓 `{...}`，只支持单 loop var 和简单 `:02d`。
- 已知局限/bug 场景：
  - 多变量插值、属性访问、字符串拼接都不稳。
  - 这是 Iter15/16/17 多个动态命名补丁的脆弱点之一。
- AST 改造方案：
  - 直接解析 `ast.JoinedStr` + `ast.FormattedValue`。
  - 对 `FormattedValue.value` 做受限求值；保留“无法静态求值就 fallback”的策略。
- 改造后预期输出变化：
  - dynamic setattr 的实名展开更稳定。
  - `attrs`、`dynamic_setattr_attrs`、tensor alias 名称更准。
- 影响的下游规则/字段：
  - `attrs`
  - `dynamic_setattr_attrs`
  - `tensor_alias_producers`
- 改造难度：中
- 是否可以复用 fake_runner.py 的现有代码：否

## 函数名：build_static_module_tree() 中的 first_call_loc 构建段
- 位置：`analyze_trace.py` L5634-L5709
- 当前实现方式：
  - 先从 `dep_edge_locs` 反推；缺失时再用正则扫 `forward()`：`self.(\w+)[(\[]`、循环容器、`getattr(self, 'attr')`。
- 已知局限/bug 场景：
  - “第一次文本出现”不一定是第一次真实调用。
  - 容器元素常借 base container 行号代填。
  - helper method / comprehension / nested call 的定位精度有限。
- AST 改造方案：
  - 在 AST call collector 中对每个模块调用记录 `lineno/end_lineno/col_offset/end_col_offset`。
  - `first_call_loc` 直接取该 attr 的最早 call-site。
  - 容器元素则取对应 `Subscript` call 或 loop body call，而不是 base name 猜测。
- 改造后预期输出变化：
  - `first_call_loc` 行号会更准确，尤其是容器元素与 helper 调用。
- 影响的下游规则/字段：
  - Rule1b
  - Rule1c
  - HTML 高亮 / 源码面板
  - boundary edge evidence
- 改造难度：中
- 是否可以复用 fake_runner.py 的现有代码：部分可复用（节点定位与 class/function AST 可复用，call-site collector 需新增）

## 函数名：_build_data_dependency_edges()
- 位置：`analyze_trace.py` L943-L2411（核心扫描主段集中在 L1176-L2237）
- 当前实现方式：
  - 超大函数，核心依赖正则、逻辑行拼接、变量名文本匹配、最近 producer 猜测。
  - helper-method 可达性也是文本 grep：`self.method\(`。
- 已知局限/bug 场景：
  - 表达式语义、嵌套 call、控制流、tuple unpack、dict slot、tensor alias 都在一个文本状态机里，回归面大。
  - `_rhs_last_called_attr()`、`re.search(var)` 这类启发式容易在复杂表达式失真。
- AST 改造方案：
  - 拆成 AST pass：
    1. `MethodCallGraphCollector`
    2. `ForwardEnvBuilder`
    3. `ExprProducerEvaluator`
    4. `EdgeEvidenceBuilder`
  - 对 `ast.Assign` / `ast.AnnAssign` / `ast.AugAssign` / `ast.Return` / `ast.Call` / `ast.Tuple` / `ast.Dict` / `ast.Subscript` 分别处理。
  - helper reachable methods 用 AST method call graph 递归闭包。
- 改造后预期输出变化：
  - 初期允许 evidence 行号更准，但 `edges` / 统计应尽量保持不变。
  - 后续可逐步减少 wildcard 和“最近 producer”误判。
- 影响的下游规则/字段：
  - `dep_edges`
  - `dep_edge_locs`
  - `split_info`
  - `var_lineage`
  - `first_call_loc`
  - 所有 Rule1/1b/1c/3/5/6/6_out/ProdConsCompleteness
- 改造难度：高
- 是否可以复用 fake_runner.py 的现有代码：否

## 函数名：_collect_dict_reads()
- 位置：`analyze_trace.py` L1008-L1063
- 当前实现方式：
  - 正则识别 `d['k']` 和 `d.get('k')`，key 非字面量时取槽位并集。
- 已知局限/bug 场景：
  - 只能从文本里找 dict 读取。
  - 对复杂 key 表达式、嵌套 subscript、局部 shadowing 没有 AST 语义保护。
- AST 改造方案：
  - 直接解析 `ast.Subscript` 与 `ast.Call(func=Attribute(value=Name(d), attr='get'))`。
  - 用 AST key 节点判断字面量 key / 动态 key。
- 改造后预期输出变化：
  - dict slot producer 归因更准。
- 影响的下游规则/字段：
  - `dict_slots`
  - `dep_edge_locs.var`
  - `var_history`
- 改造难度：中
- 是否可以复用 fake_runner.py 的现有代码：否

## 函数名：_collect_expr_producers()
- 位置：`analyze_trace.py` L1065-L1090
- 当前实现方式：
  - 遍历 `var_producers`，只要变量名在 RHS 文本里出现就算 producer；再叠加 dict reads。
- 已知局限/bug 场景：
  - “变量名出现 = 数据依赖”过粗。
  - 容易受 shadowing、字符串字面量、同名局部变量影响。
- AST 改造方案：
  - 改为 `ExprProducerEvaluator.visit(expr_node)`。
  - `Name`、`Attribute`、`Subscript`、`Call`、`BinOp`、`IfExp` 各自返回 producer set + evidence。
- 改造后预期输出变化：
  - producer 集合更精确；误连边减少。
- 影响的下游规则/字段：
  - `dep_edges`
  - `dep_edge_locs`
  - `var_lineage`
- 改造难度：高
- 是否可以复用 fake_runner.py 的现有代码：否

## 函数名：_parse_dict_literal_items()
- 位置：`analyze_trace.py` L1092-L1145
- 当前实现方式：
  - 手写深度计数器拆 `{k: v, ...}`。
- 已知局限/bug 场景：
  - 本质是在重复 Python parser。
  - 复杂 value、嵌套 dict、字符串边界都容易有维护债。
- AST 改造方案：
  - 直接用 `ast.Dict(keys, values)`。
  - key/value 节点分别走静态求值器和 producer evaluator。
- 改造后预期输出变化：
  - dict 初始化更稳定，dict slot 证据更准。
- 影响的下游规则/字段：
  - `dict_slots`
  - `dep_edge_locs`
- 改造难度：中
- 是否可以复用 fake_runner.py 的现有代码：否

## 函数名：_rhs_last_called_attr()
- 位置：`analyze_trace.py` L1498-L1554（`_build_data_dependency_edges()` 内部）
- 当前实现方式：
  - 在 RHS 文本中找最右侧 `self.xxx(...)` / dict module call / loop var call，猜“最后一个模块调用就是输出 producer”。
- 已知局限/bug 场景：
  - 对嵌套 call、条件表达式、组合表达式、多个并列 call 都不稳。
  - 本质是“最后出现的文本”而不是“最终求值结果”。
- AST 改造方案：
  - 删除该启发式。
  - 改为表达式求值：`ExprProducerEvaluator` 返回“表达式输出 producer set / tail producer / evidence”。
  - 对 `Call` 节点按 callee 和参数语义递归求值。
- 改造后预期输出变化：
  - dict slot 写入和 tuple unpack 的上游行号更准。
- 影响的下游规则/字段：
  - `dict_slots`
  - Rule1c
  - `dep_edge_locs.from_line`
- 改造难度：高
- 是否可以复用 fake_runner.py 的现有代码：否

## 函数名：_build_data_dependency_edges() 中的 tuple unpack 处理
- 位置：`analyze_trace.py` L2040-L2052、L2215-L2237
- 当前实现方式：
  - 正则：`\(?\w+(?:\s*,\s*\w+)+\)?\s*= ...`。
  - 所有 LHS 变量共享同一组 producer。
- 已知局限/bug 场景：
  - 不支持嵌套解包、星号解包、结构化 tuple/list。
  - 无法区分 tuple 不同位置的 producer。
- AST 改造方案：
  - 处理 `ast.Tuple` / `ast.List` target。
  - 对 RHS 为 tuple/list/call/return 值分别建模。
  - 第一阶段可先保持“保守共享 producer”，第二阶段再做按位置绑定。
- 改造后预期输出变化：
  - `var_producers` 与 `var_lineage` 更稳定；复杂解包不再漏。
- 影响的下游规则/字段：
  - `var_lineage`
  - `var_history`
  - `dep_edges`
- 改造难度：中
- 是否可以复用 fake_runner.py 的现有代码：否

## 函数名：_build_data_dependency_edges() 中的 dict slot 处理
- 位置：`analyze_trace.py` L1557-L1615、L2088-L2095
- 当前实现方式：
  - `m_init` / `m_lit` / `m_set` 三组正则维护 `dict_slots`。
  - 读取侧再由 `_collect_dict_reads()` 反查。
- 已知局限/bug 场景：
  - dict 初始化、写入、读取分散在多个文本分支里。
  - 复杂 key/value 表达式会退化成并集。
- AST 改造方案：
  - 用统一 `DictState` 处理 `ast.Assign(target=Subscript(...))`、`ast.Dict`、`ast.Call(...get...)`。
  - key 和 value 都走 AST evaluator。
- 改造后预期输出变化：
  - dict producer 归因和 evidence 会更稳。
- 影响的下游规则/字段：
  - `dict_slots`
  - `dep_edge_locs`
  - Rule1 / ProdConsCompleteness
- 改造难度：高
- 是否可以复用 fake_runner.py 的现有代码：否

## 函数名：_build_data_dependency_edges() 中的 tensor alias 处理
- 位置：`analyze_trace.py` L1673-L1764、L1886-L1987
- 当前实现方式：
  - 正则匹配 `setattr(self, name_expr, src_var)` 与 `getattr(self, name_expr)`。
  - 手工展开 f-string，并用 prefix/suffix 过滤 compatible aliases。
- 已知局限/bug 场景：
  - 复杂 `f-string`、属性访问、字符串拼接、局部变量转存时容易失真。
  - alias 名和值的绑定是补丁式累加出来的，回归风险高。
- AST 改造方案：
  - name_expr 改为 AST 受限求值：支持 `ast.Constant`、`ast.JoinedStr`、`ast.BinOp(Add)`、`ast.Name`。
  - alias registry 记录：`alias_name -> producers + source_line + source_expr_node`。
  - `getattr(self, name_expr)` 时直接对 AST 表达式求值，不再依赖 prefix/suffix 猜测作为主路径；prefix/suffix 只保留 fallback。
- 改造后预期输出变化：
  - tensor alias 命中率提升，误 union 降低。
  - Iter15/16 相关 GroupTower / DenseTower 连接更稳定。
- 影响的下游规则/字段：
  - `tensor_alias_producers`
  - `var_lineage`
  - `dep_edges`
  - Rule3
  - Rule6_out
  - ProdConsCompleteness
- 改造难度：高
- 是否可以复用 fake_runner.py 的现有代码：否

---

## fake_runner.py 可复用代码清单

### 可以直接迁入或升格为公共 AST 基础设施
- `ModuleTreeExtractor._parse_all_files()`：全量 parse `.py` 并收集 `ClassDef`
- `ModuleTreeExtractor._get_base_name()`：提取基类全名
- `ModuleTreeExtractor._is_nn_module_class()` / analyze_trace 内嵌 `_derive_nn_module_set()`：闭包式继承链判断
- `ModuleTreeExtractor._walk_init_body()`：递归遍历 `Assign / Expr(Call) / For / If / With / Try`
- `ModuleTreeExtractor._check_self_assign()`：`self.attr = ...`
- `ModuleTreeExtractor._check_setattr()`：`setattr(self, name, value)`
- `ModuleTreeExtractor._check_container_append()`：`self.container.append(...)`
- `ModuleTreeExtractor._infer_class_from_value()`：识别 `ModuleList / ModuleDict / Sequential / 常规模块 ctor`

### 需要在 fake_runner 基础上新增的 AST 能力
- method call graph（forward -> helper -> helper）
- call-site collector（含 `lineno/end_lineno/col_offset`）
- 受限表达式求值器（int / str / list[str] / f-string）
- forward 变量环境与 producer evaluator
- tuple unpack / dict slot / tensor alias / return lineage 解释器
- edge evidence / var_history 生成器

---

## 推荐实施顺序

1. **第 1 批：** `_build_class_map()` + 统一 AST class/method registry
2. **第 2 批：** `build_static_module_tree()` 的模块注册主干（`self.xxx` / `setattr` / append / container ctor）
3. **第 3 批：** `_extract_kw_*`、`_eval_*`、`_resolve_*` 这一组受限求值器
4. **第 4 批：** `first_call_loc` 改为 AST call-site
5. **第 5 批：** `_build_data_dependency_edges()` 拆小，先 helper reachable + module call collector
6. **第 6 批：** tuple unpack / dict slot / tensor alias / evidence 全量 AST 化

<callout icon="star" bgc="5">  
  **最关键的工程原则：**  
  每次只替换一类语义来源，并保持一个可切换的 fallback 开关。这样每个函数重构完成后，都能直接对比 Iter17 基线输出，快速定位是“识别更准”还是“行为回退”。  
</callout>

## 每批改造后的回归验证清单

- 结构统计：`nodes / groups / top_edges / internal_edges`
- 强规则：Rule1 / 1b / 1c / 2 / 2-strict / 3 / 5 / 6 / 6_out / NoInputPierce / NoDangling / ProdConsCompleteness
- 字段级 diff：
  - `attrs`
  - `container_elems`
  - `dynamic_setattr_attrs`
  - `dep_edges`
  - `dep_edge_locs.file/from_line/to_line`
  - `lineage / var_history`
  - `first_call_loc`
- 用例集：全 7 例 + synthetic cases

如果你下一步愿意，我可以继续把这份计划细化成 **“逐函数实施顺序 + 每一步要写哪些单元测试/快照 diff”** 的执行版 checklist。