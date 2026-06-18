# dag_normalizer 容器补建性能修复设计

## 1. 背景与目标

本次只出实现方案，不改代码。

已确认的事实来源如下：

- `Attr.parent` 定义在 `scripts/attr_types.py:17-25`，`ContainerAttr.items` / `add_child()` 定义在 `scripts/attr_types.py:76-88`。这说明容器父链和直接子成员关系都挂在 `Attr` / `ContainerAttr` 上。
- `_DAGBuildSession.__enter__()` 在 `storage/mock/dag_session.py:128-130` 构建 `self._container_attr_map = _build_container_attr_map(self._root)`。
- `_create_module_node()` 在 `storage/mock/dag_session.py:773-786`：
  - 优先从 `self._container_attr_map.get(path)` 取 `ContainerAttr`；
  - 若 parent 也是容器，会执行 `parent_container_attr.add_child(key, attr)`，把直接子节点（容器或普通 module）挂进 `ContainerAttr.items`。
- `_build_container_attr_map()` 在 `storage/mock/dag_session_utils.py:92-119`：
  - 先遍历模块树，给所有 native container 预建 `ContainerAttr`；
  - 再根据 path 父子关系补 `parent_container_attr.add_child(key, container_attr)`，因此嵌套容器之间也有完整父链。
- 当前 `finalize_dag_pipeline()` 在 `storage/mock/dag_session_utils.py:580-596` 只把 `dag`、`registry` 传给 `normalize_containers_recursive(dag, registry)`，没有把 `_container_attr_map` 传进去。
- 当前 `normalize_containers_recursive()` / `_normalize_containers_recursive()` / `_ensure_missing_container_nodes()` 在 `scripts/dag_normalizer.py:7-137` 中存在两个明确热点：
  1. 每层递归重复 `_build_attr_to_node_ids(registry)`；
  2. `_ensure_missing_container_nodes()` 通过遍历 `dag.nodes` 后再沿 `node.attr.parent` 一路向上爬，发现缺失容器。

本次设计目标：

1. 把 session 已经构好的 `container_attr_map` 正式接到 normalizer；
2. 把 `attr_to_node_ids` 改成「入口建一次 + 创建新容器节点时增量维护」；
3. 把缺失容器发现逻辑从“子节点向上爬 parent 链”改成“直接枚举 `container_attr_map.values()`”；
4. 不引入 shim / fallback / 兼容层；接口直接升级，所有调用方同步改。

---

## 2. 当前实现现状（按源码展开）

### 2.1 现有 normalizer 调用链

- 外部 pipeline：`storage/mock/dag_session_utils.py:580-596`
  - `validate_dag(dag, registry)`
  - `run_dce(dag, registry)`（可选）
  - `validate_dag(dag, registry)`（DCE 后再次校验）
  - `normalize_containers_recursive(dag, registry)`（当前没有 `container_attr_map`）
- session 组装：`storage/mock/dag_session.py:518-574`
  - `finalize()` 负责返回 `dict(self._node_registry.registry), dag`
  - 当前没有把 `self._container_attr_map` 返回给上游

### 2.2 当前 normalizer 的 3 次 attr 索引重建

`_build_attr_to_node_ids()` 定义在 `scripts/dag_normalizer.py:213-217`，它会完整遍历 `registry.items()` 建立 `id(node.attr) -> [node_id]` 映射。

当前在同一轮 `_normalize_containers_recursive()` 中会触发 3 次：

1. `scripts/dag_normalizer.py:26`
   - 目的：给 `_ensure_missing_container_nodes()` 准备索引。
2. `scripts/dag_normalizer.py:85`
   - 目的：`_ensure_missing_container_nodes()` 内部又重建一次，直接覆盖了传入参数。
3. `scripts/dag_normalizer.py:28`
   - 目的：补建容器节点后，再重建一次给 `_normalize_single_dag()` 使用。

结论：当前每递归处理一个 `DAG` 对象，至少会完整扫描 `registry` 3 次。

### 2.3 当前缺失容器发现逻辑

`scripts/dag_normalizer.py:80-136` 的当前逻辑分两段：

1. `:88-100` 扫 `dag.nodes`，对每个 `node.attr.parent` 持续上爬；
2. 若遇到 `parent_attr` 不在 `attr_to_node_ids`，则放进 `missing_container_attrs`；
3. `:102-136` 再逐个补建容器节点。

这意味着：

- 发现缺失容器依赖“当前 DAG 里先出现某个后代节点”；
- 容器发现成本和 `dag.nodes` 数量、容器嵌套深度强耦合；
- session 阶段已经现成的 `container_attr_map` 完全没被用上。

---

## 3. 接口变更方案

### 3.1 `normalize_containers_recursive` 新签名

当前：

```python
normalize_containers_recursive(dag: DAG, registry: dict[int, DagNode]) -> None
```

改为：

```python
normalize_containers_recursive(
    dag: DAG,
    registry: dict[int, DagNode],
    container_attr_map: dict[str, ContainerAttr],
) -> None
```

说明：

- 直接接 `storage/mock/dag_session_utils.py:_build_container_attr_map()` 的原始返回类型；
- 不新增兼容默认值，不保留旧签名；
- `container_attr_map` 的 key（module path）同时也是合成容器节点 `metadata["module_path"]` 的权威来源。

### 3.2 内部 helper 签名调整

建议改成下面这组关系：

```python
normalize_containers_recursive(dag, registry, container_attr_map) -> None
    # 入口：建 attr 索引 / 建 next_node_id / 先补缺失容器 / 再递归归一化各层 DAG

_normalize_containers_recursive(
    dag,
    registry,
    attr_to_node_ids,
    owner_node_id,
    visited_dag_ids,
) -> None

_ensure_missing_container_nodes(
    root_dag,
    registry,
    container_attr_map,
    attr_to_node_ids,
    next_node_id,
) -> int
```

其中：

- `_normalize_containers_recursive()` 不再自己建 `attr_to_node_ids`；
- `_ensure_missing_container_nodes()` 不再自己建 `attr_to_node_ids`；
- `next_node_id` 由入口一次性初始化，helper 返回更新后的值，避免每建一个容器都 `max(registry)`。

### 3.3 `finalize_dag_pipeline` 调用处修改

当前 `storage/mock/dag_session_utils.py:580-596`：

```python
finalize_dag_pipeline(dag, registry, apply_dce=True, apply_normalize=True)
```

建议改为：

```python
finalize_dag_pipeline(
    dag,
    registry,
    container_attr_map,
    apply_dce: bool = True,
    apply_normalize: bool = True,
) -> list[str]
```

对应 `apply_normalize` 分支改为：

```python
normalize_containers_recursive(dag, registry, container_attr_map)
```

说明：

- `container_attr_map` 不是可选项；
- 因为 normalizer 的新主路径必须依赖它，保留默认值只会形成软兼容路径，不符合当前约束。

### 3.4 `session.finalize()` 与上游传参链修改

当前 `_DAGBuildSession.finalize()` 在 `storage/mock/dag_session.py:518-574` 返回二元组：

```python
tuple[dict[int, DagNode], DAG]
```

建议直接改为三元组：

```python
tuple[dict[int, DagNode], DAG, dict[str, ContainerAttr]]
```

即：

```python
return dict(self._node_registry.registry), dag, dict(self._container_attr_map)
```

说明：

- 直接返回 map，不新增 property / accessor shim；
- 返回副本 `dict(...)` 即可，语义与返回 `registry` 的副本一致，避免 finalize 后上游意外原地改 session 内部对象表结构。

### 3.5 需要同步修改的调用点

#### storage 仓库

1. `storage/testset/unit/dag_session_test_infra.py:223-233`
   - 当前：`registry, dag = session.finalize()`
   - 改后：`registry, dag, container_attr_map = session.finalize()`
   - 再传给：`finalize_dag_pipeline(dag, registry, container_attr_map, ...)`

2. `storage/testset/unit/dag_session_test_infra.py:286-296`
   - `_build_smoke_dag_registry()` 同步接三元组

3. `storage/mock/run_dag_session.py:365-366`
   - 当前：`registry, dag = session.finalize()`
   - 改后：`registry, dag, container_attr_map = session.finalize()`
   - 再传给 `finalize_dag_pipeline(dag, registry, container_attr_map)`

4. `storage/mock/dag_session.py:1319-1332`
   - `_run_forward_with_tracing()` 当前也调用 `session.finalize()`；
   - 要同步接三元组；
   - 若它对外只需要返回 `(registry, dag, records)`，则在函数内部消费 `container_attr_map` 后再决定是否继续向外透传。
   - 这里不建议偷偷丢弃不改签名，而应按实际调用需求同步改干净；避免形成隐藏分叉。

#### torch-trace-analyzer 仓库

- `scripts/dag_normalizer.py` 公有接口变更；
- 如有其他直接调用 `normalize_containers_recursive()` 的地方，必须一起改。当前 grep 到的直接调用主要在 storage test 中；torch 仓库内未见其他调用点。

---

## 4. `_build_attr_to_node_ids` 调用优化设计

### 4.1 当前调用次数

见 `scripts/dag_normalizer.py`：

| 位置 | 当前目的 | 问题 |
|---|---|---|
| `:26` | 进入 `_ensure_missing_container_nodes()` 前建索引 | 每层递归全量扫 `registry` |
| `:85` | `_ensure_missing_container_nodes()` 内部再建一次 | 与传参重复，纯浪费 |
| `:28` | 容器补建后再次建索引 | 只为看到新增节点，但完全可以增量更新 |

### 4.2 改后方案

新流程：

1. `normalize_containers_recursive()` 入口只调用 **1 次** `_build_attr_to_node_ids(registry)`；
2. `_ensure_missing_container_nodes()` 在创建每个新容器节点后，执行：

```python
attr_to_node_ids.setdefault(id(container_attr), []).append(node_id)
```

3. 后续 `_normalize_single_dag()`、`_collect_relevant_container_node_ids()`、`_resolve_direct_member_ids()` 都复用同一个 `attr_to_node_ids` 引用；
4. 递归进入子 `inner_dag` 时，不复制、不重建，直接继续传同一个 dict。

### 4.3 各处改后状态

| 当前位置 | 改后状态 |
|---|---|
| `_normalize_containers_recursive():26` | 删除；入口统一建一次 |
| `_ensure_missing_container_nodes():85` | 删除；使用传入映射 |
| `_normalize_containers_recursive():28` | 删除；新增容器时增量维护 |
| 新增入口调用 | `normalize_containers_recursive()` 最外层建一次 |

---

## 5. `_ensure_missing_container_nodes` 重写方案

### 5.1 重写目标

把当前“从 `dag.nodes` 出发，沿 `node.attr.parent` 向上爬父链找缺失容器”的逻辑，改成“直接枚举 `container_attr_map.values()`”。

这样做的依据是：

- `_build_container_attr_map()` 已经把所有 native container 预建好（`storage/mock/dag_session_utils.py:92-119`）；
- 嵌套容器之间的父子关系也已经在 map 内部通过 `add_child()` 建好；
- `_create_module_node()` 在真实 module / 真实 Sequential 场景下也会把普通 module attr 和 container attr 接到这些 `ContainerAttr.items` 上（`storage/mock/dag_session.py:773-786`）。

因此，缺失容器集合不需要再靠 DAG 反推，可以直接从 map 正向枚举。

### 5.2 新逻辑的处理顺序

**结论：子容器先建，父容器后建。**

原因：

- 父容器的 `items` 里可能直接存的是子容器 `ContainerAttr`；
- 若父先建，父容器会发现自己的直接 child 还没有 node_id；
- 只有先把更深层容器补出来，并把它写回 `attr_to_node_ids`，父容器才能把该子容器识别为自己的直属成员。

建议按 path 深度或 parent 深度做**降序排序**：

- `outer.0.1` 先于 `outer.0`
- `outer.0` 先于 `outer`

### 5.3 新逻辑的具体步骤

建议 `_ensure_missing_container_nodes(root_dag, registry, container_attr_map, attr_to_node_ids, next_node_id)` 执行以下步骤：

1. 取 `container_attr_map.items()`；
2. 按容器深度降序排序；
3. 逐个处理 `(path, container_attr)`：
   - 先检查 `container_attr.class_name in _NATIVE_CONTAINER_KINDS`；否则直接 `raise RuntimeError`；
   - 查看 `existing_ids = attr_to_node_ids.get(id(container_attr), [])`
     - 若 `len(existing_ids) > 1`：`raise RuntimeError`
     - 若 `len(existing_ids) == 1`：说明该容器节点本来就存在（典型是整体调用的 `Sequential`），直接跳过
   - 否则开始收集 `child_ids`
4. 对 `container_attr.items.values()` 中每个 `child_attr`：
   - `matched_ids = attr_to_node_ids.get(id(child_attr), [])`
   - 若为空：表示该 child 在 DCE 后已不存在，或其整个子树无存活 node；跳过，不报错
   - 若长度 > 1：`raise RuntimeError`
   - 若长度 == 1：加入 `child_ids`
5. 若最终 `child_ids` 为空：
   - **不建该容器节点**；
   - 这覆盖两种合法场景：
     1. 整个容器子树都被 DCE 剪掉；
     2. 容器本身原本为空，没有任何可落地成员；
   - 这里不应 raise，因为当前 IR 中没有任何存活成员需要它承载归属关系。
6. 若 `child_ids` 非空：
   - 用当前 `next_node_id` 创建新的 `ModuleNode`；
   - `metadata["module_path"] = path`，不再依赖 `_container_attr_path()` 逆推；
   - `metadata["is_container"] = True`；
   - `attr = container_attr`；
   - `is_native = True`（语义保持和当前补建逻辑一致）
7. `inner_dag` 构建规则保持现语义：
   - `nodes = child_ids.copy()`
   - `direct_nodes = child_ids.copy()`
   - `edges = [edge for edge in root_dag.edges if edge.src_id in child_id_set and edge.dst_id in child_id_set]`
8. 将新节点写入：
   - `registry[node_id] = new_node`
   - `attr_to_node_ids.setdefault(id(container_attr), []).append(node_id)`
   - `next_node_id += 1`
9. 返回更新后的 `next_node_id`

### 5.4 需要覆盖的边界情况

#### A. `container_attr` 对应 node 已存在

典型场景：`Sequential` 被整体调用时，构图阶段已经产生真实节点（见现有 UT `test_n5_nested_sequential_keeps_hooked_container_nodes`）。

处理：

- 若 `existing_ids == [nid]`，直接跳过；
- 不重复补建；
- 后续 `_normalize_single_dag()` 仍会把它当容器节点收纳到相应 `dag.direct_nodes` / `inner_dag.direct_nodes`。

#### B. child attr 在 DCE 后不在 registry

这是合法场景，不是错误。现有代码在两处已经这么处理：

- `scripts/dag_normalizer.py:109-111`
- `scripts/dag_normalizer.py:199-201`

新逻辑继续保持：

- `matched_ids` 为空时直接跳过；
- 但只跳过这个 child，不影响同容器中其他存活 child；
- 如果所有 child 都空，则整个容器不建。

#### C. 嵌套容器顺序

必须**先子后父**。否则父容器的 `child_ids` 会缺失子容器 node。

### 5.5 必须 raise 的情况（禁止静默路径）

以下情况必须明确报错：

1. `container_attr.class_name` 不在 `_NATIVE_CONTAINER_KINDS`
2. 同一个 `container_attr` 在 `attr_to_node_ids` 中映射到多个 node_id
3. 同一个存活 `child_attr` 映射到多个 node_id
4. 已存在的 node_id 对应的 registry 节点不是 `ModuleNode`
5. 已存在的容器节点 `metadata["module_path"]` 与当前 `container_attr_map` key 不一致

以下情况**不是错误**，允许跳过：

1. 某个 child attr 在 DCE 后没有存活节点
2. 整个容器没有任何存活 child，因此本轮无需落地为 node
3. 容器节点本来就已存在（比如真实调用过的 `Sequential`）

---

## 6. `_normalize_containers_recursive` 重写方案

### 6.1 新的入口流程

建议把 public 入口改成两阶段：

```python
normalize_containers_recursive(dag, registry, container_attr_map):
    attr_to_node_ids = _build_attr_to_node_ids(registry)
    next_node_id = max(registry, default=0) + 1
    next_node_id = _ensure_missing_container_nodes(
        root_dag=dag,
        registry=registry,
        container_attr_map=container_attr_map,
        attr_to_node_ids=attr_to_node_ids,
        next_node_id=next_node_id,
    )
    _normalize_containers_recursive(
        dag=dag,
        registry=registry,
        attr_to_node_ids=attr_to_node_ids,
        owner_node_id=None,
        visited_dag_ids=set(),
    )
```

关键点：

- **缺失容器补建只做一次，而且在递归前做**；
- 依赖顶层 `dag.nodes` 的 flatten 语义（`storage/mock/dag_session.py:555-563`），顶层 DAG 已经持有全量 module node id；
- 递归阶段只做“把已有容器节点安放进对应层级 DAG / inner_dag”，不再做容器发现。

### 6.2 递归函数传参方式

建议 `_normalize_containers_recursive()` 改成：

```python
_normalize_containers_recursive(
    dag: DAG,
    registry: dict[int, DagNode],
    attr_to_node_ids: dict[int, list[int]],
    owner_node_id: int | None,
    visited_dag_ids: set[int],
) -> None
```

处理策略：

- `attr_to_node_ids`：**传引用，不复制**
  - 原因：它本质上是 `registry` 的索引视图；
  - 入口补建完容器后，这个映射已经稳定；
  - 递归过程中不再创建新容器节点，因此无须额外 copy。
- `visited_dag_ids`：继续按引用传递同一个 set

### 6.3 `visited_dag_ids` 是否保留

**保留。**

原因：

- 现有 `scripts/dag_normalizer.py:22-25` 已用 `id(dag)` 做重入保护；
- 虽然正常树形 `inner_dag` 不该共享对象，但这是一层低成本安全网；
- 本次性能热点不在这里，删掉收益极低，反而可能引入递归重入风险。

### 6.4 递归阶段保留不变的逻辑

以下逻辑仍可保留：

- `_normalize_single_dag()`：负责把相关容器节点加入 `dag.nodes` / `dag.direct_nodes`
- `_collect_relevant_container_node_ids()`：继续依据当前层节点的 `attr.parent` 收集“本层需要看见哪些容器节点”
- `_resolve_direct_member_ids()`：继续根据 `ContainerAttr.items` 决定 direct members
- `_rebuild_adjacency()`：每层归一化后重建 `dag.in_edges` / `dag.out_edges`

即：本次重写只替换“缺失容器发现和 attr 索引重建”这部分，不额外扩 scope 到 direct member 解析路径。

---

## 7. 复杂度分析

### 7.1 记号

- `N`：DCE 后存活的 module node 数量（normalizer 入口时 registry 中与模块相关的量级）
- `C`：`container_attr_map` 中容器总数
- `M`：其中需要补建的缺失容器数，`M <= C`
- `E`：顶层 `dag.edges` 数量
- `H`：容器嵌套最大深度
- `K`：递归过程中会处理到的 `DAG` 层数（顶层 + 各层 `inner_dag`）
- `P`：所有 `ContainerAttr.items` 的总 child 引用数

### 7.2 改前复杂度

#### A. attr 索引重建

每个递归 DAG 都会扫 3 次 `registry`：

- 单层：`O(3N)`
- 全递归：`O(KN)`

更准确地说，随着补建容器节点，registry 会从 `N` 增长到 `N + M`，所以是：

```text
O(K * (N + M))
```

#### B. 缺失容器发现（子节点向上爬 parent 链）

对每个 DAG：

- 遍历 `dag.nodes`
- 每个 node 最多向上爬 `H` 层 parent

所以单层成本是：

```text
O(|dag.nodes| * H)
```

全递归成本是：

```text
O(Σ_i |dag_i.nodes| * H)
```

由于顶层 `dag.nodes` 是全量 flatten（`storage/mock/dag_session.py:558`），而子层 `inner_dag.nodes` 又会再次遍历直属成员，深层节点会在多个祖先层被重复扫到。最坏情形下，这一项可近似放大到：

```text
O(N * H^2)
```

#### C. 每建一个容器都 `max(registry)`

当前 `scripts/dag_normalizer.py:105`：

```text
每次 O(N + 已建容器数)
```

补 `M` 个容器时总成本：

```text
O(M * (N + M))
```

#### D. 每个容器过滤 inner_edges

当前每补一个容器都扫一次当前 `dag.edges`：

```text
O(M * E)
```

#### 总结

改前主成本可以写成：

```text
O(K * (N + M))
+ O(Σ_i |dag_i.nodes| * H)
+ O(M * (N + M))
+ O(M * E)
```

其中前两项就是这次已确认的主要热点。

### 7.3 改后复杂度

#### A. attr 索引只建一次

入口一次：

```text
O(N)
```

#### B. 枚举 `container_attr_map`

- 排序（按深度降序）：`O(C log C)`
- 遍历所有容器及其直接 child 引用：`O(P)`

```text
O(C log C + P)
```

#### C. 新容器 node_id 递增

入口一次初始化 `next_node_id = max(registry) + 1`：

```text
O(N)
```

之后每补一个容器只做 `next_node_id += 1`：

```text
O(M)
```

#### D. inner_edges 过滤

这一项本轮不变，仍是：

```text
O(M * E)
```

#### E. 递归安放容器节点

`_normalize_single_dag()` / `_collect_relevant_container_node_ids()` 的递归放置成本保留，因此仍有：

```text
O(Σ_i |dag_i.nodes| * H)
```

#### 总结

改后总成本为：

```text
O(N)
+ O(C log C + P)
+ O(M)
+ O(M * E)
+ O(Σ_i |dag_i.nodes| * H)
```

和改前相比，至少去掉了两块确定性的全局重复成本：

1. `O(K * (N + M))` 的重复 attr 索引重建
2. `O(M * (N + M))` 的重复 `max(registry)`

并把“缺失容器发现”从依赖 `dag.nodes × parent depth` 的反向搜索，替换成了对 `container_attr_map` 的一次正向枚举。

---

## 8. 测例设计（UT）

本次只出测例方案，不落代码。

建议继续放在：

- `storage/testset/unit/test_dag_normalizer.py`

并复用已有 infra：

- `run_dag_session`
- `node_id_by_path`
- `_assert_has_edge` / `_assert_no_edge`

若需要新增公共 helper，必须先补到 `testset/unit/dag_session_test_infra.py`，不要在测试文件里重复定义。

### 8.1 Case A：嵌套容器（container 内嵌 container）

#### mock Module 结构

```python
class NestedModuleListModel(nn.Module):
    def __init__(self):
        self.outer = nn.ModuleList([
            nn.ModuleList([
                nn.Linear(4, 4)
            ])
        ])

    def forward(self, x):
        return self.outer[0][0](x)
```

#### 关注点

验证“子容器先建、父容器后建”的方案成立。

#### 断言

1. `outer.0` 和 `outer` 两个容器节点都存在
2. 两者都是 `ModuleNode + ContainerAttr`
3. `outer.inner_dag.direct_nodes` 包含 `outer.0`
4. `outer.0.inner_dag.direct_nodes` 包含 `outer.0.0`
5. 输入边仍然直达叶子 `outer.0.0`，不会改写到容器节点上
6. registry 中 `module_path == "outer"` / `"outer.0"` 各只出现一次，证明没有重复补建

> 现有 `test_n4_nested_modulelist_builds_two_level_container_chain` 已经覆盖结构正确性；实现时可以直接在该 case 上补“无重复补建”断言，或拆新 case。

### 8.2 Case B：DCE 后 child 不存在的容器

#### mock Module 结构

```python
class DeadMemberModuleListModel(nn.Module):
    def __init__(self):
        self.blocks = nn.ModuleList([
            nn.Linear(4, 4),
            nn.Linear(4, 4),
        ])

    def forward(self, x):
        used = self.blocks[0](x)
        _dead = self.blocks[1](x)
        return used
```

#### 关注点

验证“某个 child attr 已被 DCE 剪掉时，normalizer 仍能补建容器，但只收纳存活 child；不把缺失 child 当错误”。

#### 断言

1. `blocks.1` 对应节点在 DCE 后不存在于 registry
2. 容器 `blocks` 仍存在
3. `blocks.inner_dag.direct_nodes == [blocks.0]`（顺序可按现有断言方式写成集合或长度 + 包含）
4. 输入边仍为 `input -> blocks.0`
5. 不存在 `input -> blocks` 数据流边
6. 运行过程中不抛异常

> 现有 `test_n6_normalizer_tolerates_dce_pruned_members` 已经是这个方向；实现时可直接沿用并确认新逻辑仍通过。

### 8.3 Case C：已存在节点不被重复补建

#### mock Module 结构

```python
class SequentialWholeCallModel(nn.Module):
    def __init__(self):
        self.seq = nn.Sequential(nn.Linear(4, 4), nn.ReLU())

    def forward(self, x):
        return self.seq(x)
```

#### 关注点

`Sequential` 整体调用时，构图阶段已经有真实 node（`_create_module_node()` 会直接使用 `self._container_attr_map.get(path)`）。normalizer 新逻辑必须识别“已有节点”，不能再补一个同 attr 的合成节点。

#### 断言

1. `module_path == "seq"` 的节点只有一个
2. 该节点 `is_native is False`（说明保留的是运行期真实节点，不是新增合成节点）
3. 该节点 `metadata["is_container"] is True`
4. 该节点的 direct members 仍是 `seq.0`、`seq.1`
5. 再次调用 `normalize_containers_recursive(...)` 后：
   - `module_path == "seq"` 的节点数量仍为 1
   - `dag.edges` 不变
   - `direct_nodes` 不变

> 现有 `test_n5_nested_sequential_keeps_hooked_container_nodes` + `test_normalize_is_idempotent` 可组合成这个验证目标；如果实现时想把“已有容器节点跳过补建”单独表达，建议补一条更直观的新 case。

---

## 9. 预期改动边界（本次方案不做的事）

本轮方案只覆盖以下范围：

1. normalizer 接 `container_attr_map`
2. attr 索引一次构建 + 增量维护
3. 缺失容器发现逻辑从 parent-chain 反推改为直接枚举 `container_attr_map`
4. session / finalize pipeline 的参数链打通
5. 对应 UT 补强

**不包含**：

- 改写 `_collect_relevant_container_node_ids()` 的 parent 链逻辑
- 改写 container `inner_edges` 的生成策略
- 改动 serializer / frontend / trace 侧逻辑

这样能把本次性能修复范围收敛在已确认的两个热点上，避免额外放大变更面。

---

## 10. 实施顺序建议

1. 改 `_DAGBuildSession.finalize()` 返回三元组
2. 改 `finalize_dag_pipeline()` 新签名并打通所有调用点
3. 改 `normalize_containers_recursive()` 公有签名
4. 入口一次性构建 `attr_to_node_ids` + `next_node_id`
5. 重写 `_ensure_missing_container_nodes()`：
   - 枚举 `container_attr_map`
   - 深度降序
   - 增量更新 `attr_to_node_ids`
6. 精简 `_normalize_containers_recursive()`：只保留递归归一化，不再重建 attr 索引
7. 补 / 调整 `test_dag_normalizer.py` 中上述 3 类 UT

以上顺序的好处是：接口链先打通，再改核心逻辑，最后补回归，定位问题最直接。
