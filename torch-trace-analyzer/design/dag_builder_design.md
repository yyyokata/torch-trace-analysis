# DagBuilder 设计文档

## 1. 目标与边界

本文档定义 AST 重构阶段 `dag_builder.py` 的职责边界、调用链、接口形态和 UT 设计入口。目标是把源码静态 DAG 生成从旧的混合解析逻辑中拆出，形成严格串行、模块职责清晰、可单测的后端链路。

核心原则：

1. **常量先决**：`ConstantTable` 与 `ConstantResolver` 在 Attr 构造前完成静态常量折叠，用于 for range 展开、if/else 分支收敛、ModuleList children 展开等静态决策。
2. **Scanner 持有 Resolver**：分支收敛与容器展开都属于 Attr 构造职责，因此 `ConstantResolver` 归属 `AttrScanner`，不下放给 `DagBuilder`。
3. **Builder 只消费已收敛 Attr**：`DagBuilder` 持有 `ASTFrontend` 与 `AttrScanner`，负责把 Scanner 已经收敛好的 Attr 转成 Node、Edge、DAG、DagContext；每个 Attr 的 `class_name` 在进入 Builder 前必须是确定值。
4. **数据流只在类内 forward 中追踪**：本阶段不做跨类数据流穿透，跨类只通过 `inner_dag` 建模。
5. **前端只展示**：所有 trace / source 数据处理仍在后端完成，前端不新增任何分类、聚合、时长统计或 DAG 计算逻辑。

---

## 2. 模块职责一览表

<table header-row="true" header-col="false" col-widths="170,260,360,260">
  <tr>
    <td>模块</td>
    <td>输入</td>
    <td>职责</td>
    <td>输出</td>
  </tr>
  <tr>
    <td>`ASTFrontend`</td>
    <td>`source_files` 或单文件 source/path</td>
    <td>Layer 1 AST 工具层。提供类/方法定位、语句遍历、调用匹配、表达式文本化、简单 alias 匹配等无业务状态工具。</td>
    <td>AST 节点、`CallLoc`、匹配结果、表达式字符串</td>
  </tr>
  <tr>
    <td>`ConstantTable`</td>
    <td>`source_files`、`ASTFrontend` 集合、`nn_module_classes`</td>
    <td>全量预扫描静态事实，构建 file/global/class/instance/method 级常量表。</td>
    <td>只读 `ConstantTable`</td>
  </tr>
  <tr>
    <td>`ConstantResolver`</td>
    <td>`ConstantTable`、`runtime_overrides`</td>
    <td>基于常量表做纯 AST 常量求值。Resolver 在 Scanner 中承担三类静态决策：整数常量折叠用于 for 循环展开；结合 `conditional_mode` 做 if/else 分支收敛；展开 ModuleList children。</td>
    <td>解析后的 int/list/bool/表达式值；无法确定返回 `None`</td>
  </tr>
  <tr>
    <td>`AttrScanner`</td>
    <td>`ASTFrontend`、可选 `ConstantResolver`</td>
    <td>扫描 `__init__` 与 `forward`，构造 6 类 Attr：`InputAttr`、`ModuleAttr`、`ContainerAttr`、`ResultAttr`、`ForwardArgAttr`、`ReturnValAttr`；当 resolver 可用时展开 `ContainerAttr.children/items`。</td>
    <td>`dict[str, list[AttrType]]` 与 `attrs_by_id`</td>
  </tr>
  <tr>
    <td>`DagBuilder`</td>
    <td>`ASTFrontend`、`AttrScanner`</td>
    <td>递归构建单类 DAG：Attr → Node、container 归属边、forward 数据流边、子类 `inner_dag`。</td>
    <td>`DagContext`</td>
  </tr>
  <tr>
    <td>`DagContext`</td>
    <td>全量节点注册表、根 DAG、常量表、Attr 注册表</td>
    <td>顶层上下文容器，为 HTML、测试、后续 timeline 映射提供统一索引。</td>
    <td>`registry`、`root`、`loc_attr_index`、`const_table`、`attr_registry`</td>
  </tr>
</table>

---

## 3. 完整调用链与依赖方向

### 3.1 严格串行调用链

```python
def build_dag(source_files, root_class, runtime_overrides):
    fe = ASTFrontend(source_files)
    table = ConstantTable.build_all(source_files, fe)
    resolver = ConstantResolver(table, runtime_overrides)
    scanner = AttrScanner(resolver)
    builder = DagBuilder(fe, scanner)
    ctx = builder.build(root_class)
    return ctx
```

实际落地时 `ASTFrontend(source_files)` 可以是薄封装，内部按文件构造多个 `ASTFrontend(source=..., path=...)`；本文档只约束逻辑调用顺序和依赖方向。

### 3.2 依赖方向图

```text
source_files
    │
    ▼
ASTFrontend  ───────────────┐
    │                       │
    ▼                       │
ConstantTable.build_all      │
    │                       │
    ▼                       │
ConstantResolver             │
    │                       │
    ▼                       │
AttrScanner(resolver) ◄──────┘
    │
    ▼
DagBuilder(fe, scanner)
    │
    ▼
DagContext(registry, root, const_table, attr_registry)
```

依赖规则：

1. `ConstantResolver` 只归属 `AttrScanner`。
2. `Resolver` 在 `Scanner` 内负责三件事：整数常量折叠（for 循环展开）、if/else 分支收敛（由 `conditional_mode` 决定 infer/train 分支）、ModuleList children 展开。
3. `DagBuilder` 不持有 `ConstantResolver`，只拿已经展开、分支已收敛的 Attr。
4. `ASTFrontend` 是 Layer 1 工具，不持有 `ConstantTable`、`ConstantResolver`、`AttrScanner`、`DagBuilder`。
5. `DagContext` 作为结果容器，可以保存 `const_table` 和 `attr_registry`，但不参与构建逻辑反向调用。

### 3.3 本轮审核决策补充

1. **`conditional_mode` 非法值**：`AttrScanner.__init__` 直接 `raise ValueError`，不做 fallback。
2. **if/else resolver 无法收敛**：`AttrScanner` 直接 `raise ValueError("Cannot resolve branch condition: ...")`，要求调用方提供足够的 resolver 信息；不得保留互斥分支，也不得跳过。
3. **T16 复杂 mock**：复杂 mock 保留在 `test_dag_builder.py` 的 T16 中，不拆新 T17；除容器/嵌套/for/if 外，额外覆盖 global dict 输入读取导致的数据流边。

---

## 4. 各模块接口定义

以下为接口签名和 docstring 设计，不包含实现。

### 4.1 `ASTFrontend` 新增接口

```python
class ASTFrontend:
    def iter_forward_stmts(self, class_name: str) -> Iterator[ast.stmt]:
        """按源码顺序遍历 class_name.forward 的可执行语句。

        语义与 iter_init_stmts 保持一致：
        - 若 class 不存在或没有 forward，返回空迭代器。
        - 展开 if/for/while/with/try 等控制流 body/orelse/finalbody。
        - 跳过 ast.Pass。
        - 不做常量分支裁剪；分支选择由 ConstantResolver/AttrScanner 或后续调用方决定。
        """

    def match_simple_alias(self, stmt: ast.stmt) -> Optional[tuple[str, str]]:
        """识别纯 Name alias 赋值。

        仅匹配如下形式：
            lhs = rhs
        其中 lhs 与 rhs 都必须是 ast.Name。

        返回：
            (lhs_name, rhs_name)

        以下情况必须返回 None：
        - 调用赋值：x = self.fc(y)
        - 属性赋值：self.x = y
        - tuple/list unpack：a, b = x
        - 下标/属性读取：x = obj.y / x = arr[i]
        - 常量或表达式：x = 1 / x = a + b
        """
```

### 4.2 `ConstantTable` 接口

```python
class ConstantTable:
    @classmethod
    def build_all(
        cls,
        source_files: dict[str, list[str]],
        ast_frontends: dict[str, ASTFrontend],
        nn_module_classes: set[str],
    ) -> "ConstantTable":
        """对源码集合执行全量常量预扫描并返回只读常量表。

        构建顺序固定：
        1. file-level 常量、dataclass 默认值
        2. cross-file global int 收敛
        3. class-level 构造链
        4. call-site formal parameter binding

        本表仅记录来自真实语义的常量事实；不得从类型注解猜测值。
        """
```

### 4.3 `ConstantResolver` 接口

```python
class ConstantResolver:
    def __init__(self, table: ConstantTable, runtime_overrides: Optional[dict] = None):
        """创建静态常量解析器。

        table 提供预扫描事实；runtime_overrides 用于测试或运行期显式覆盖。
        无法从真实语义确定的值必须返回 None，不允许 annotation fallback。
        """

    def eval_int(self, node: ast.AST, scope: Scope) -> Optional[int]:
        """解析表达式为 int；失败返回 None。"""

    def eval_list_len(self, node: ast.AST, scope: Scope) -> Optional[int]:
        """解析表达式为列表长度；失败返回 None。"""

    def eval_bool(self, node: ast.AST, scope: Scope) -> Optional[bool]:
        """解析静态 if 条件；失败返回 None。"""
```

### 4.4 `AttrScanner` 扩展接口

```python
class AttrScanner:
    def __init__(self, resolver: Optional[ConstantResolver] = None, conditional_mode: str = "infer"):
        """创建 Attr 扫描器。

        resolver 可为空：
        - 为空时仅扫描直接 Attr，不展开需要常量折叠或分支收敛的结构。
        - 非空时承担三类静态决策：
          1. 整数常量折叠，用于 for 循环展开。
          2. if/else 分支收敛，结合 conditional_mode 选择 infer/train 分支。
          3. ModuleList/ModuleDict/Sequential children 展开。

        conditional_mode 控制 __init__ 中条件分支收敛：
        - "infer"：只扫描推理分支对应 Attr。
        - "train"：只扫描训练分支对应 Attr。

        Scanner 必须保证交给 DagBuilder 的每个 Attr.class_name 已是分支收敛后的确定值。
        """

    def scan_class(self, class_name: str, fe: ASTFrontend) -> dict[str, list[AttrType]]:
        """扫描单个 class 的 Attr。

        返回固定 key：
        - inputs: list[InputAttr]
        - modules: list[ModuleAttr]
        - containers: list[ContainerAttr]
        - results: list[ResultAttr]
        - forward_args: list[ForwardArgAttr]
        - return_vals: list[ReturnValAttr]
        """

    def _scan_container_attrs(self, class_name: str, fe: ASTFrontend) -> list[ContainerAttr]:
        """扫描容器属性，并在 resolver 可用时展开 children/items。

        展开规则由真实静态语义驱动：
        - ModuleList(range(N)) 使用 resolver 解析 N。
        - ModuleDict/Sequential 使用 resolver 解析静态 key 或长度。
        - 无法确定时保留容器本身，不做猜测。
        """
```

### 4.5 `DagBuilder` 新增接口

```python
class DagBuilder:
    def __init__(self, fe: ASTFrontend, scanner: AttrScanner):
        """创建 DAG 构建器。

        Builder 持有 ASTFrontend 与 AttrScanner，不持有 ConstantResolver。
        Builder 只消费 Scanner 已构造、已展开、if/else 分支已收敛的 Attr。
        """

    def build(self, root_class: str) -> DagContext:
        """从 root_class 构建完整 DagContext。

        递归构建规则：
        - root_class 生成 root DAG。
        - ModuleAttr.class_name 可在 source_files 中解析到源码时，递归构建 inner_dag。
        - native 或找不到源码时，inner_dag = None。
        """

    def build_class(self, class_name: str) -> DAG:
        """构建单个 class 的 DAG。

        步骤：
        1. scanner.scan_class(class_name, fe)
        2. Attr → Node
        3. 构造 containment edges
        4. 构造 dataflow edges
        5. 对可解析 ModuleAttr 递归填充 ModuleNode.inner_dag
        """

    def _build_nodes(self, attrs_by_kind: dict[str, list[AttrType]]) -> tuple[list[int], list[int], list[int]]:
        """把 Attr 转换为 DagNode 并注册到 registry。

        映射规则：
        - InputAttr / ForwardArgAttr → InputNode
        - ResultAttr / ReturnValAttr → ResultNode
        - ModuleAttr / ContainerAttr → ModuleNode
        """

    def _build_containment_edges(self, container_attrs: list[ContainerAttr]) -> list[DataFlowEdge]:
        """为 ContainerAttr 与 children 构造归属虚边。

        归属边要求：
        - is_containment=True
        - evidence=[]
        - src_id 为 container 对应 ModuleNode
        - dst_id 为 child 对应 ModuleNode
        """

    def _build_dataflow_edges(
        self,
        class_name: str,
        attrs_by_kind: dict[str, list[AttrType]],
    ) -> list[DataFlowEdge]:
        """遍历 forward 方法体构造真实数据流边。

        两遍扫描：
        1. 建 var_producers：forward 参数、self.attr(...) 左侧变量、纯 alias a=b 多跳。
        2. 建边：当 self.attr(x) 的 x 可追溯到 producer 时，生成 DataFlowEdge + VarEvidence。

        本阶段只做单类 forward 内追踪，不跨类穿透。
        """
```

### 4.6 `DagContext` 扩展接口

```python
@dataclass
class DagContext:
    registry: dict[int, DagNode]
    root: DAG
    loc_attr_index: dict = field(default_factory=dict)
    const_table: Optional[ConstantTable] = None
    attr_registry: dict[int, Attr] = field(default_factory=dict)

    def __post_init__(self):
        """构建 loc_attr_index。

        key: (file: str, line: int, attr_id: int)
        value: node_id

        col 不参与 key，因为 timeline event 不携带 col。
        """
```

---

## 5. 数据流转关系

### 5.1 Attr → Node

<table header-row="true" header-col="false" col-widths="220,220,360">
  <tr>
    <td>Attr 类型</td>
    <td>Node 类型</td>
    <td>说明</td>
  </tr>
  <tr>
    <td>`InputAttr`</td>
    <td>`InputNode`</td>
    <td>Root DAG 输入节点，对应 `LG.*` 输入声明。</td>
  </tr>
  <tr>
    <td>`ForwardArgAttr`</td>
    <td>`InputNode`</td>
    <td>inner_dag 输入节点，对应子类 `forward` 形参。</td>
  </tr>
  <tr>
    <td>`ModuleAttr`</td>
    <td>`ModuleNode`</td>
    <td>普通模块节点。若 class 在 source_files 中定义，递归填充 `inner_dag`。</td>
  </tr>
  <tr>
    <td>`ContainerAttr`</td>
    <td>`ModuleNode`</td>
    <td>容器也作为 ModuleNode 存在；其 children 通过 containment edge 表达归属。</td>
  </tr>
  <tr>
    <td>`ResultAttr`</td>
    <td>`ResultNode`</td>
    <td>Root DAG 结果节点，对应 `LG.result().head(...)`。</td>
  </tr>
  <tr>
    <td>`ReturnValAttr`</td>
    <td>`ResultNode`</td>
    <td>inner_dag 返回值节点，对应子类 `forward` 返回。</td>
  </tr>
</table>

### 5.2 var → Edge

`DagBuilder._build_dataflow_edges` 在单个 class 的 `forward` 内维护 `var_producers`：

```python
var_producers: dict[str, int]
```

含义：变量名 → 产生该变量的 node_id。

第一遍扫描规则：

1. forward 参数：每个 `ForwardArgAttr.attr_name` 映射到对应 `InputNode.node_id`。
2. module 调用赋值：`h = self.fc(x)` 中 `h` 映射到 `self.fc` 对应 `ModuleNode.node_id`。
3. 纯 alias：`a = b` 中若 `b` 已在 `var_producers`，则 `a` 映射到同一个 producer，支持多跳 alias。
4. dict subscript 输入读取：`x = self.inputs["x"]` 或 `x = inputs["x"]` 中，若 key 是字符串字面量且能按 `InputAttr.attr_name` 命中，则 `x` 直接映射到对应 `InputNode.node_id`。
5. dict subscript key 若不是静态字符串字面量（例如变量 key），降级为 unknown producer：不建边、记录 warning、不抛异常。

第二遍扫描规则：

1. 遇到 `self.attr(x)` 调用。
2. 提取输入变量 `x`。
3. 在 `var_producers` 查到 producer node。
4. 生成 `DataFlowEdge(src_id=producer, dst_id=self.attr node, is_containment=False, evidence=[...])`。

边只在当前 class 的 `forward` 内建立，不跨入子类 `inner_dag`。子类内部结构由 `ModuleNode.inner_dag` 表示。

### 5.3 Evidence 结构

```python
@dataclass
class EvidenceStep:
    loc: CallLoc
    role: str  # "producer" / "step" / "consumer"
    var: str

@dataclass
class VarEvidence:
    var: str
    path_id: int
    steps: list[EvidenceStep]

@dataclass
class DataFlowEdge:
    src_id: int
    dst_id: int
    is_containment: bool
    evidence: list[VarEvidence]
```

Evidence 约束：

1. 真实数据流边：`is_containment=False`，`evidence` 至少包含 producer 与 consumer；多跳 alias 以 role=`step` 记录中间变量。
2. 归属虚边：`is_containment=True`，`evidence=[]`。
3. `path_id` 用于区分同一个 src 输出视角下的多条变量追踪链；本阶段可从 0 或自增计数开始，保持单边内唯一即可。
4. 若数据流来自 dict subscript 读取（例如 `inputs["x"]`），evidence chain 中必须保留该 dict 读取步骤，便于 HTML / UT 直接回溯 producer 来源。

### 5.4 T16 复杂 mock 补充要求

T16 使用单个复杂 synthetic source 统一覆盖以下场景：

1. root / branch / container / leaf 多层递归 DAG 构造。
2. `ModuleList([Block(i) for i in range(4)])` 的 containment edge。
3. `if mode == "train"` / `else` 在 `conditional_mode="infer"` 下只收敛到推理分支。
4. global dict 输入读取：`self.inputs = {"x": LG.input(...), "y": LG.input(...)}`，`forward` 中通过 `x = inputs["x"]` 建立 `InputNode("x") -> ModuleNode(fc)`。
5. variable key 读取：`inputs[key]` 只记录 warning，不抛异常。

---

## 6. inner_dag 产生时机

`ModuleNode.inner_dag` 的生成遵循源码可见性：

1. `ModuleAttr.class_name` 能在 `ASTFrontend.class_registry` / source_files 中找到定义：递归调用 `build_class(class_name)`，结果挂到 `ModuleNode.inner_dag`。
2. native 模块，例如 `nn.Linear`、`torch.nn.Linear`，或 `attr.is_native=True`：`inner_dag=None`。
3. 找不到源码的外部类：`inner_dag=None`。

递归需要有 cycle guard。若遇到环形类引用，应停止递归并保守设置当前 `inner_dag=None` 或复用已构建 DAG；具体实现优先采用最小实体方案，避免引入复杂缓存层。

---

## 7. 改动清单与估计行数

<table header-row="true" header-col="false" col-widths="300,520,140">
  <tr>
    <td>文件</td>
    <td>改动内容</td>
    <td>估计行数</td>
  </tr>
  <tr>
    <td>`scripts/ast_frontend.py`</td>
    <td>新增 `iter_forward_stmts`、`match_simple_alias`。实现复用 `iter_init_stmts` 的控制流 body 展开逻辑，避免重复大段代码时可抽小私有 helper。</td>
    <td>+30 ~ +60</td>
  </tr>
  <tr>
    <td>`scripts/attr_scanner.py`</td>
    <td>`__init__` 增加 `resolver: Optional[ConstantResolver] = None` 与 `conditional_mode: str = "infer"`；遍历 `__init__` 中 if/else 时用 resolver + conditional_mode 做分支收敛，确保 Attr.class_name 确定；`_scan_container_attrs` 在 resolver 可用时展开 children/items；保持 scan_class 返回结构不变。</td>
    <td>+70 ~ +160</td>
  </tr>
  <tr>
    <td>`scripts/dag_types.py`</td>
    <td>`DagContext` 增加 `const_table` 与 `attr_registry` 字段；必要时补 TYPE_CHECKING import，避免运行期循环依赖。</td>
    <td>+5 ~ +20</td>
  </tr>
  <tr>
    <td>`scripts/dag_builder.py`</td>
    <td>新增 DagBuilder 主体：build/build_class、Attr→Node、containment edges、dataflow edges、inner_dag 递归。</td>
    <td>+250 ~ +450</td>
  </tr>
  <tr>
    <td>`scripts/analyze_trace.py` 或现有入口胶水层</td>
    <td>把旧 DAG 入口切到新 `build_dag` 串行链路；保持前端 HTML 输入结构兼容或一次性修正调用方。</td>
    <td>+30 ~ +100</td>
  </tr>
  <tr>
    <td>`storage/testset/unit/`</td>
    <td>新增正式 UT。测试代码只落 storage 仓库，不写入研发仓库。</td>
    <td>+250 ~ +450</td>
  </tr>
</table>

注意：研发仓库 `torch-trace-analysis` 禁止放 smoke test、手动验证脚本、硬编码 testset 路径函数。所有新增测试必须落在 storage 仓库 `testset/unit/` 下。

---

## 8. Known Issue

### 8.1 ConstantTable table 11 跨文件 dataclass 链

当前已知限制：`ConstantTable.instance_const_chain` 对跨文件 dataclass 链的覆盖仍不完整。典型场景是 parent class 与 dataclass config 分散在不同文件，且实例字段需要沿 `parent_attr -> param_name -> dataclass.field` 多段传播。

本设计中处理策略：

1. `DagBuilder` 不直接修复 table 11，也不绕过 `ConstantResolver` 做临时 fallback。
2. `AttrScanner` 在 resolver 无法解析 container 展开长度、key 或条件分支时，保守保留可确定部分；若分支无法收敛，不得把互斥分支都当成确定 Attr 交给 Builder。
3. 后续若修复 table 11，应在 `ConstantTable` / `ConstantResolver` 层补真实语义解析，`DagBuilder` 与 `AttrScanner` 接口保持稳定。
4. 严禁使用 annotation fallback；类型注解不能作为运行期值来源。

---

## 9. 本阶段验收标准

1. 新增接口职责符合本文档依赖方向：`Resolver -> Scanner -> Builder`，不得让 Builder 反向持有 Resolver。
2. `DagContext` 能同时提供 `registry`、`root`、`loc_attr_index`、`const_table`、`attr_registry`。
3. 节点构造覆盖 6 类 Attr 到 3 类 Node 的映射。
4. containment edge 使用 `is_containment=True` 与空 evidence。
5. dataflow edge 支持 forward 参数单步输入与纯 Name alias 多跳输入。
6. 子类源码可见时填充 `inner_dag`；native 或源码不可见时为 `None`。
7. 新增测试只在 storage 仓库 `testset/unit/` 落地；编码前需先经 UT 设计方案审核。
