# 5476790 节点数 196 → 229 的修复方案（feat/ast-refactor，AST-only）

> **范围声明**
> 本方案只做代码分析与修复设计，不修改任何源文件。所有解析必须遵循 AST 优先原则，不得引入正则兜底。

---

## 0. 现状

| 项 | 当前 | 目标 | 差距 |
|---|---|---|---|
| 5476790 nodes | **196** | 229 | **-33** |
| Test Suite | 7/7 PASS | 7/7 PASS | — |

实测当前 AST refactor 输出与 iter17 baseline 的精确差异（Sub-Agent 已验证）：

**Missing in current**（14 个直接缺失，加上其下层连锁缺失约 33）：
- `Dense.fc1 / fc2 / gate / key_dense / out_dense / query_dense / value_dense`（7 个）
- `LayerNorm.input_ln / k_norm / ln_1 / ln_2 / q_norm`（5 个）
- `NoShareMultiSlotsEmbedding.agg_ind_debias_emb`、`ShareMultiSlotEmbedding.agg_share_debias_emb`（2 个）

**4 个类完全不在 tree 中**：`AGGDebiasModule` / `TransBlock` / `Attention` / `GatedFFN`

**SeqTrans 只有 1 个 attr**（`input_ln`），缺 `trans_blocks`（ModuleList of TransBlock）

---

## 1. 根因 1 精确定位 ─ AGGDebiasModule / TransBlock attrs 缺失

### 1.1 `AGGDebiasModule` ─ 实际未缺，被 reachability 滤除

**源码注册方式**（`storage_task_clone/testset/extracted/5476790/modelcode/main_model.py`）

```python
【2217】class AGGDebiasModule(torch.nn.Module):
【2218】    def __init__(self, name):
【2219】        super().__init__()
【2220】        self.name = name
【2222】        self.agg_ind_debias_emb = NoShareMultiSlotsEmbedding(...)   # 直接 self.x = Cls(...)
【2223】        self.agg_share_debias_emb = ShareMultiSlotEmbedding(...)    # 直接 self.x = Cls(...)
【2224】        self.agg_debias_tower = DenseTower(...)                     # 直接 self.x = Cls(...)
```

**实例化（构造）位置**：

```python
【3019】 self.agg_modules = torch.nn.ModuleList(
            [AGGDebiasModule(name) for name in ['shopping','pclk','xd','pay','lt_pay']]
        )
```

> 注意：`iter` 是 **字面量列表 `['shopping', ...]`**，不是 `range(N)`。

**当前 `analyze_trace.py` 处理 ListComp 的代码路径**（行 `7986-7997`）：

```python
【7986】 if isinstance(_arg0, ast.ListComp) and isinstance(_arg0.elt, ast.Call):
【7987】     _generators = _arg0.generators or []
【7988】     _gen0 = _generators[0] if len(_generators) == 1 else None
【7989】     _cls_ref = _node_to_text(_arg0.elt.func).split('.')[-1]
【7990】     _n = None
【7991】     if _gen0 and isinstance(_gen0.iter, ast.Call) and (_expr_leaf_name(_gen0.iter.func) == "range") and len(_gen0.iter.args) == 1:
【7992】         _n = _eval_int_node(_gen0.iter.args[0], fname, cname, mname)
【7993】     if _n is not None and _cls_ref in nn_module_classes:
【7994】         attrs.setdefault(cont, _cls_ref)
【7995】         attr_def_loc.setdefault((cname, cont), (fname, phys_lineno))
【7996】         for i in range(_n):
【7997】             _ensure_elem(cont, cname, _elem_attr_list(cont, i), _cls_ref, fname, phys_lineno, attrs)
```

**漏洞**：`_n` 仅当 `iter` 是 `range(N)` 时被赋值。`for name in [...literal_list...]` 模式下 `_n is None`，**整段 ListComp 不被展开**。

但 5476790 的 `agg_modules` 父类（`AdsV3FullRankLayer` 或 `CVRModule`）静态 DAG 仍能记录 `agg_modules = AGGDebiasModule`，所以 AGGDebiasModule 出现在 `nn_module_classes` 但 `tree["agg_modules"]` 没有对应展开成 5 个实例 → **AGGDebiasModule 自身没有被任何 children 引用** → 在 reachability 滤除阶段（`build_static_module_tree` 行 `8793-8806`）被剪除：

```python
【8793】 # Filter tree to only include classes reachable from the primary root.
【8806】 tree = {k: v for k, v in tree.items() if k in reachable}
```

reachability 是从 root 沿 `tree[cname]["children"]` DFS。AGGDebiasModule 不在任何父 children 中 → 被剪。

> 顺带：与之并列的 `agg_ind_debias_emb` / `agg_share_debias_emb` 缺失也是同根 ─ AGGDebiasModule 整棵子树被裁掉时其内部 attrs 自然连带丢失。

### 1.2 `TransBlock` / `Attention` / `GatedFFN` ─ setattr-loop 注册未识别

**源码注册方式**（`trans_refactor.py`）

```python
【200】class TransBlock(nn.Module):
【201】    def __init__(self, config: TransformerConfig, is_last_layer=False, **kwargs):
【214】        self.self_attn = Attention(self.config)         # 直接
【215】        self.ffn = GatedFFN(self.config)                # 直接
【217】        self.ln_1 = LayerNorm(hidden_size=...)          # 直接
【218】        self.ln_2 = LayerNorm(hidden_size=...)          # 直接
【219】        self.tanh = nn.Tanh()
```

```python
【60】class Attention(nn.Module):
【69】        for name in ["query_dense","key_dense","value_dense","out_dense"]:
【70】            setattr(self, name, Dense(...))             # ★ for-loop 内 setattr
【82】        if self.config.use_qk_norm:
【83】            self.q_norm = LayerNorm(...)                # 直接（条件分支）
【84】            self.k_norm = LayerNorm(...)
【86】        self.k_cache = TensorCache(...)
【87】        self.v_cache = TensorCache(...)
```

```python
【139】class GatedFFN(nn.Module):
【145】        for (name, in_dim, out_dim) in [
【146】                ("fc1", ..., ...),
【147】                ("gate", ..., ...),
【148】                ("fc2", ..., ...)
【149】            ]:
【150】            setattr(self, name, Dense(...))             # ★ tuple-unpack for-loop 内 setattr
```

```python
【249】class SeqTrans(nn.Module):
【255】        if self.config.use_input_norm:
【256】            self.input_ln = LayerNorm(...)
【258】        self.trans_blocks = nn.ModuleList([             # ★ ModuleList(ListComp(range))
【259】            TransBlock(config, is_last_layer=(i==(self.config.num_hidden_layers-1)))
【260】            for i in range(self.config.num_hidden_layers)
【261】        ])
```

**实测当前 attrs**：
- `TransBlock.attrs` = `{}` (空)
- `Attention.attrs` = `{}` (空)
- `GatedFFN.attrs` = `{}` (空)
- `SeqTrans.attrs` = `{'input_ln': 'LayerNorm'}`（缺 `trans_blocks`）

**根因细分**：

#### 1.2.1 SeqTrans.trans_blocks 缺失 — ListComp range 但 N 来自 `self.config.num_hidden_layers`

行 `7991-7992` 的 `_eval_int_node` 调用 `_new_eval_resolver.eval_int(...)`，需要从 `TransformerConfig` 类的字段定义解析 `num_hidden_layers=6`。这条 cross-class const evaluation 通路在 ast_refactor 分支可能未实现 ─ 即使 `range(self.config.num_hidden_layers)` 形式上是 `range(N)`，但 `_n` 解析为 `None`。

> 验证方法：抓 `[ERROR] Cannot statically enumerate` 是否覆盖此 case；如果没 ERROR 但 attrs 缺失，则属于"未声明、静默漏掉"。

#### 1.2.2 TransBlock attrs 全空 — TransBlock 类**作为 SeqTrans.trans_blocks ListComp 的 elt** 被丢失

由于 1.2.1，SeqTrans 没有把 trans_blocks 写入 attrs，TransBlock 也就不会作为 `attrs.setdefault(cont, _cls_ref)` 被注册到 SeqTrans 的 children。TransBlock 自身的 `__init__` 直接 attrs（`self.self_attn` 等）应该走另一通路被识别。但实际 attrs 还是空 — 这意味着 TransBlock **整个类**的 init 扫描没跑或被跳过。

进一步推断：TransBlock 在 reachability 滤除阶段（行 8793）被剪 → tree 中消失 → 我们打印时拿到的是被剪后的结果，所以"看起来 attrs=空"实际是"key 不存在"。

#### 1.2.3 Attention / GatedFFN attrs 全空 — for-loop setattr 模式

源码模式：
```python
for name in ["query_dense","key_dense","value_dense","out_dense"]:
    setattr(self, name, Dense(...))
```

`analyze_trace.py` 在 `__init__` 扫描时，`logical_lines` 是逐行遍历，`_loop_var_to_str_items` 跟踪 loop var（行 7807-7888），`_parse_local_setattr_ctor` 解析 setattr（行 7618-7642）。f-string 模式（`setattr(self, f"{name}_x", ...)`）有专门 expander。但 **纯 loop var（`name=='name'`）作为 `name_arg`** 的纯字符串展开路径，需要确认是否落在某条 if 里。

实际验证：上面 1.2.2 的同因 ─ 如果 Attention/GatedFFN **整个类**被 reachability 剪掉（通过 SeqTrans.trans_blocks→TransBlock 链断），那么它们 attrs 也观察不到。

> **根因 1 总结**：核心问题是 **ModuleList ListComp 展开覆盖率不足**：
> - `[Cls(name) for name in ['lit',...]]` 不展开（漏 AGGDebiasModule × 5 实例）
> - `[Cls(...) for i in range(self.config.X)]` cross-class const N 无法解析（漏 SeqTrans→TransBlock × N 实例）
>
> 这两条断链导致 4 个类整体被 reachability 滤除。

---

## 2. 根因 2 精确定位 ─ helper-method 调用导致 dead-child 误裁

### 2.1 helper-method 调用模式

```python
# main_model.py
【1064】 transf_soft_flow_2k, length_info_2k = self.seq_trans.dense_query(   # ★ self.attr.method(...)
            query=trans_query,
            sequence=SequenceTensor.from_masked_tensor(...),
        )
【1154】 trans_soft_flow_ins, length_info_ins = self.ins_trans.dense_query(  # ★ 同模式
            query=ins_trans_query,
            sequence=SequenceTensor.from_masked_tensor(...),
        )
```

`SeqTrans.dense_query` 是 helper method，**不是 `__call__`/`forward`**。所以 `self.ins_trans.dense_query(...)` 形式上是 `func.value = self.ins_trans (ast.Attribute)`，调用的目标是 `dense_query` 而非 `ins_trans`。

### 2.2 first_call_loc 当前覆盖范围（Sub-Agent 已验证）

**`analyze_trace.py` 行 409 `ASTFrontend.get_first_call_loc`** 内调用的 `_extract_called_self_attr`（约行 954）：

```python
if isinstance(func_node, ast.Attribute) \
   and isinstance(func_node.value, ast.Name) \
   and func_node.value.id == "self":
    return func_node.attr           # 只识别 self.attr(...)
```

**支持模式**：
- ✅ `self.attr(...)`
- ✅ `self.attr[idx](...)`（Iter12 Rule6 加的）
- ✅ `getattr(self, "attr")(...)`（字面量）
- ❌ **`self.attr.helper(...)` ─ 不识别为对 attr 的调用**

### 2.3 dead-child 过滤器（Bug D 修复引入）

**`frontend_html.py` 行 2702-2710** （Sub-Agent 已定位）：

```python
【2702】 for attr_name, child_cls in attrs_filtered.items():
【2703】     if attr_name not in seen_attrs:
【2704】         _has_runtime_use = (
【2705】             attr_name in info.get("first_call_loc", {})
【2706】             or attr_name in _dep_attrs
【2707】             or attr_name in _input_consumer_attrs
【2708】         )
【2709】         if not _has_runtime_use:
【2710】             continue                    # ★ 误裁 ins_trans / seq_trans
```

### 2.4 链式效应

1. `InsTrans.forward` 实际调用 `self.ins_trans.dense_query(...)` ─ helper-method 形式。
2. `get_first_call_loc("InsTrans","ins_trans")` 返回 None（因为 2.2 不识别 `self.attr.method`）。
3. dead-child 过滤器（2.3）发现 `ins_trans not in first_call_loc`，且 dep edges 缺，且不消费 Input → 直接 `continue`，**ins_trans 节点连同其子树（SeqTrans/TransBlock/...）一起从 DAG 中删除**。
4. 同理 `C2kTrans.seq_trans` 被删；进一步连带 SeqTrans 自身和 trans_blocks 节点丢失。

> **根因 2 总结**：first_call_loc 当前只识别 1 层 `self.attr(...)`，对 `self.attr.helper_method(...)` 形式失明，导致 dead-child 过滤器把这些通过 helper 调用的子模块全部裁掉。

---

## 3. 修复方案 E1 ─ 扩充 ListComp / 字面量 list 展开

### E1.1 字面量 list 作为 ListComp iter（`for x in ['a','b',...]`）

**位置**：`ast_refactor_workdir/scripts/analyze_trace.py` 行 `7986-7997`（`_ast_container_ctor` 处理 ModuleList ListComp 的分支）

**改动逻辑（伪代码）**：

```python
# 在原有 range(N) 检测之后追加：
if _n is None and _gen0 is not None:
    # 新增：字面量 list iter
    if isinstance(_gen0.iter, ast.List):
        _items = [e for e in _gen0.iter.elts]
        _n = len(_items)         # 直接拿长度
    # 新增：file/global UPPER_CASE 字符串列表常量
    elif isinstance(_gen0.iter, ast.Name):
        _resolved_list = _resolve_str_list(
            ast.unparse(_gen0.iter), fname, cname, _loop_var_to_str_items)
        if _resolved_list:
            _n = len(_resolved_list)
    # 新增：列表长度通过 ConstantResolver 评估
    else:
        _list_len = _eval_list_len_node(_gen0.iter, fname, cname, mname)
        if _list_len is not None:
            _n = _list_len

# 后续 range(_n) 展开逻辑保持不变
```

**预期效果**：
- 行 3019 `[AGGDebiasModule(name) for name in ['shopping','pclk','xd','pay','lt_pay']]` 解析出 `_n=5`，自动展开 `agg_modules[0..4]`
- AGGDebiasModule 进入 reachability，其 attrs（`agg_ind_debias_emb`、`agg_share_debias_emb`、`agg_debias_tower`）随之恢复
- 同理覆盖到 `LogitAdaptive` 中类似 list-literal 模式

**预期恢复节点**：
- `agg_modules['shopping' / 'pclk' / 'xd' / 'pay' / 'lt_pay']` 5 个 AGGDebiasModule 实例
- `agg_ind_debias_emb` × 5（NoShareMultiSlotsEmbedding）
- `agg_share_debias_emb` × 5（ShareMultiSlotEmbedding）
- `agg_debias_tower` × 5（DenseTower 内部 _layers）
- 合计 ~15 个节点恢复

### E1.2 ConstantResolver 跨类常量解析（`range(self.config.num_hidden_layers)`）

**位置**：`ast_refactor_workdir/scripts/analyze_trace.py`
- `_eval_int_node`（行 7054-7065）已经委托给 `ConstantResolver.eval_int`
- `ConstantTable.build_all`（行 7118）/ `ConstantResolver`（行 1566）

**改动逻辑（伪代码）**：

需要让 `ConstantResolver.eval_int` 在 scope 为 `SeqTrans.__init__` 时，能够：
1. 识别 `self.config.num_hidden_layers` 的 receiver 是 `self.config`
2. 通过 `class_init_params["SeqTrans"]` 找到 `__init__(self, config: TransformerConfig, ...)` 的 `config` 参数
3. 通过 `parent_attr` (`InsTrans.ins_trans` / `C2kTrans.seq_trans`) 调用方传入的实参 `TransformerConfig(**self.ins_trans_base_params)` 跨级解析
4. 在 `TransformerConfig.__post_init__` 或 `@dataclass` 字段默认值中拿到 `num_hidden_layers: int = 6`

由于跨级解析复杂度高，**推荐采用降级策略**：

```python
# 在 _eval_int_node 内追加 fallback：
def _eval_int_node(...):
    iv = _new_eval_resolver.eval_int(expr_node, scope)
    if iv is not None:
        return iv.value
    # 新增 fallback：self.<attr>.<sub_attr> 解析为 self.<attr>=Cls(...) 中的 sub_attr
    # 1. 识别 expr_node 是 self.X.Y 形式
    # 2. 在 cname 的 __init__ 中找 self.X = Cls(...) 的 Cls
    # 3. 在 Cls 中找 self.Y = Constant(int) 或 dataclass field default
    return _resolve_self_chain_int(expr_node, fname, cname, ...)
```

**预期效果**：
- `range(self.config.num_hidden_layers)` → `_n = 6`
- `SeqTrans.trans_blocks` 展开为 `trans_blocks[0..5]`，6 个 TransBlock 实例
- TransBlock 进入 reachability

**预期恢复节点**（基于 InsTrans/C2kTrans 各持有 1 个 SeqTrans，每个 SeqTrans 含 6 TransBlock）：
- `trans_blocks[0..5]` × 多个 SeqTrans 实例（约 12+ 节点）
- 每个 TransBlock 内 `self_attn`（Attention）/ `ffn`（GatedFFN）/ `ln_1` / `ln_2` / `tanh`
- 每个 Attention 内 `query_dense / key_dense / value_dense / out_dense`（4 Dense via setattr-loop）
- 每个 GatedFFN 内 `fc1 / gate / fc2`（3 Dense via setattr-loop）

### E1.3 setattr-with-loop-var 的纯字符串展开

**前提**：在 E1.2 让 TransBlock 进入 tree 后，Attention/GatedFFN 的 setattr-loop 解析需要正常工作。

**位置**：`ast_refactor_workdir/scripts/analyze_trace.py` 行 `7807-7888`（`_loop_var_to_str_items` 跟踪 + setattr 解析）

**改动逻辑（伪代码）**：

```python
# 在 _parse_local_setattr_ctor 返回结果后追加：
_setattr_info = _parse_local_setattr_ctor(_local_stmt)
if _setattr_info:
    _name_expr = _setattr_info["name_expr"].strip()
    _cls_full = _setattr_info.get("class_full") or ""
    _cls = _cls_full.split('.')[-1]
    if _cls in nn_module_classes:
        # 新增：纯 loop-var name → 展开为字面量列表中每个 str
        try:
            _name_node = ast.parse(_name_expr, mode="eval").body
        except SyntaxError:
            _name_node = None
        if isinstance(_name_node, ast.Name) and _name_node.id in _loop_var_to_str_items:
            for _real_name in _loop_var_to_str_items[_name_node.id]:
                attrs.setdefault(_real_name, _cls)
                attr_def_loc.setdefault((cname, _real_name), (fname, phys_lineno))
        # 现有 f-string 展开 / 字面量 / 动态分支保留
```

> 同时需要确认 `_loop_var_to_str_items` 在 `for (name, in_dim, out_dim) in [...]` tuple-unpack 形式下也对 `name` 建立条目（GatedFFN 是这种模式，行 7855-7877 的 Rule "for (name, a, b, ...) in [(s, ..),...]" 已经处理 tuple-unpack；需要确认这条规则真的命中 trans_refactor.py 的 GatedFFN）。

**预期效果**：
- Attention.__init__ for-loop 展开 4 个 attrs：`query_dense / key_dense / value_dense / out_dense`（→ Dense）
- GatedFFN.__init__ for-loop 展开 3 个 attrs：`fc1 / gate / fc2`（→ Dense）
- TransBlock.__init__ 直接赋值的 `self_attn / ffn / ln_1 / ln_2` 走原有 direct-assign 通路恢复

**预期恢复节点（合计 E1.1+E1.2+E1.3）**：
- AGGDebiasModule × 5 实例 + 内部子树（~15 节点）
- SeqTrans.trans_blocks 展开（~12 TransBlock 实例）
- TransBlock 子树（self_attn/ffn/ln_1/ln_2 × N 实例）
- Attention.q/k/v/out_dense / q_norm / k_norm / k_cache / v_cache（× N）
- GatedFFN.fc1/gate/fc2（× N）
- 合计预期补回 **18-25 个节点**

---

## 4. 修复方案 E2 ─ first_call_loc 支持 helper-method 链式调用

### 4.1 扩展 `_extract_called_self_attr` / `get_first_call_loc`

**位置**：`ast_refactor_workdir/scripts/analyze_trace.py`
- `ASTFrontend.get_first_call_loc`（行 409+）
- `_extract_called_self_attr`（约行 954，Sub-Agent 定位）

**改动逻辑（伪代码）**：

```python
def _extract_called_self_attr(call_node):
    """识别 call_node 调用的 self.<attr>，包括 helper-method 链式调用。"""
    func = call_node.func
    if not isinstance(func, ast.Attribute):
        return None

    # 现有：self.attr(...)
    if isinstance(func.value, ast.Name) and func.value.id == "self":
        return func.attr

    # 现有：self.attr[idx](...)  -- via ast.Subscript
    if isinstance(func.value, ast.Subscript):
        sub = func.value
        if (isinstance(sub.value, ast.Attribute)
                and isinstance(sub.value.value, ast.Name)
                and sub.value.value.id == "self"):
            return sub.value.attr            # self.<attr>[...].method → still attr

    # ★ 新增：self.<attr>.<helper>(...)  → return attr
    if isinstance(func.value, ast.Attribute) \
       and isinstance(func.value.value, ast.Name) \
       and func.value.value.id == "self":
        return func.value.attr

    # ★ 新增：self.<attr>[idx].<helper>(...) → return attr
    if isinstance(func.value, ast.Attribute) \
       and isinstance(func.value.value, ast.Subscript):
        sub = func.value.value
        if (isinstance(sub.value, ast.Attribute)
                and isinstance(sub.value.value, ast.Name)
                and sub.value.value.id == "self"):
            return sub.value.attr

    return None
```

**关键注意**：本改动不能改变 dep_edge 的 `to_excerpt` 锚定行 ─ first_call_loc 只决定 dead-child 过滤是否保留，dep_edge 的 evidence 仍指向同一行（helper call 的 lineno）。Rule1c 验证时需要把"helper 调用作为 attr 的 first call"加入容忍模式。

### 4.2 dead-child 过滤器收紧（可选 ─ 不强制）

**位置**：`ast_refactor_workdir/scripts/frontend_html.py` 行 `2702-2710`

E2.1 修好后，dead-child 过滤器逻辑无需修改 ─ ins_trans / seq_trans 自然进入 first_call_loc，过滤器条件 `attr_name in info.get("first_call_loc", {})` 满足。

> 不修改 frontend_html.py 的过滤器，避免影响其他场景。

### 4.3 预期效果

- `InsTrans.ins_trans → SeqTrans` chain 恢复
- `C2kTrans.seq_trans → SeqTrans` chain 恢复（5476790 中 SequenceModule 持有多个 C2kTrans 实例：c2k_trans / cpay_trans / cclk_trans）
- 配合 E1.2 / E1.3 ─ `trans_blocks[0..5]` 全部进入 DAG
- Attention / GatedFFN 子节点全部恢复

---

## 5. 测例设计

### 5.1 测例 E1.1 ─ ListComp 字面量 list 展开

**测试文件**：在 `ast_refactor_workdir/testset/synthetic_cases/` 新增 `test_modulelist_listcomp_strlist.py`，并在 `testset/test_dag_rules.py` 中加入对应的入口测例（沿用 `_load_synthetic_case` 模式）。

**Mock 类**：

```python
# synthetic_cases/test_modulelist_listcomp_strlist/main_model.py
import torch
from torch import nn

class Leaf(nn.Module):
    def __init__(self, name):
        super().__init__()
        self.name = name
        self.proj = nn.Linear(8, 8)

    def forward(self, x):
        return self.proj(x)

class Root(nn.Module):
    def __init__(self):
        super().__init__()
        # ★ 字面量 list 驱动 ListComp ─ 期望展开 3 个实例
        self.leaves = nn.ModuleList(
            [Leaf(name) for name in ['a', 'b', 'c']]
        )

    def forward(self, x):
        out = x
        for layer in self.leaves:
            out = layer(out)
        return out
```

**断言**：

```python
def test_modulelist_listcomp_strlist():
    tree, roots, cmap = build_static_module_tree(load(...), conditional_mode='infer')
    assert 'Root' in tree
    assert 'Leaf' in tree
    root_attrs = tree['Root']['attrs']
    # 必须展开 3 个具体 elem
    assert "leaves[0]" in root_attrs and root_attrs["leaves[0]"] == "Leaf"
    assert "leaves[1]" in root_attrs
    assert "leaves[2]" in root_attrs
    # Leaf 自身也必须可达
    leaf_attrs = tree['Leaf']['attrs']
    assert 'proj' in leaf_attrs
```

### 5.2 测例 E1.2 ─ ListComp range(self.config.X) 跨类常量

**Mock 类**：

```python
# synthetic_cases/test_modulelist_listcomp_cross_const/main_model.py
from dataclasses import dataclass
import torch
from torch import nn

@dataclass
class Cfg:
    n_layers: int = 4

class Block(nn.Module):
    def __init__(self):
        super().__init__()
        self.lin = nn.Linear(8, 8)
    def forward(self, x):
        return self.lin(x)

class Stack(nn.Module):
    def __init__(self, cfg: Cfg):
        super().__init__()
        self.config = cfg
        # ★ 跨类常量驱动 range
        self.blocks = nn.ModuleList([Block() for i in range(self.config.n_layers)])

    def forward(self, x):
        for b in self.blocks:
            x = b(x)
        return x

class Root(nn.Module):
    def __init__(self):
        super().__init__()
        self.stack = Stack(Cfg())            # 默认 n_layers=4

    def forward(self, x):
        return self.stack(x)
```

**断言**：

```python
def test_modulelist_listcomp_cross_const():
    tree, _, _ = build_static_module_tree(load(...), conditional_mode='infer')
    assert 'Stack' in tree
    stack_attrs = tree['Stack']['attrs']
    # 跨类常量解析必须给出 4 个具体 elem
    for i in range(4):
        assert f"blocks[{i}]" in stack_attrs
    # Block 子树
    assert 'Block' in tree
    assert 'lin' in tree['Block']['attrs']
```

### 5.3 测例 E1.3 ─ setattr loop-var 纯字符串展开

**Mock 类**：

```python
# synthetic_cases/test_setattr_loop_var/main_model.py
import torch
from torch import nn

class Leaf(nn.Module):
    def __init__(self, n):
        super().__init__()
        self.lin = nn.Linear(n, n)
    def forward(self, x):
        return self.lin(x)

class Holder(nn.Module):
    def __init__(self):
        super().__init__()
        # ★ for name in [字面量列表]: setattr(self, name, Leaf(...))
        for name in ["alpha", "beta", "gamma"]:
            setattr(self, name, Leaf(8))
        # tuple-unpack 形式
        for (n, dim) in [("delta", 4), ("epsilon", 8)]:
            setattr(self, n, Leaf(dim))

    def forward(self, x):
        return self.epsilon(self.delta(self.gamma(self.beta(self.alpha(x)))))
```

**断言**：

```python
def test_setattr_loop_var_strlist():
    tree, _, _ = build_static_module_tree(load(...), conditional_mode='infer')
    holder_attrs = tree['Holder']['attrs']
    for name in ['alpha','beta','gamma','delta','epsilon']:
        assert name in holder_attrs and holder_attrs[name] == 'Leaf'
```

### 5.4 测例 E2 ─ helper-method 调用作为 first_call_loc

**Mock 类**：

```python
# synthetic_cases/test_helper_method_call/main_model.py
import torch
from torch import nn

class Backbone(nn.Module):
    def __init__(self):
        super().__init__()
        self.lin = nn.Linear(8, 8)
    def forward(self, x):
        return self.lin(x)
    def custom_query(self, q):                  # ★ helper method
        return self.forward(q)

class Wrapper(nn.Module):
    def __init__(self):
        super().__init__()
        self.bb = Backbone()                    # ★ 关键 ─ 永远不会被裁剪

    def forward(self, x):
        # ★ 通过 helper-method 调用，不是 self.bb(x)
        return self.bb.custom_query(x)
```

**断言**：

```python
def test_helper_method_first_call_loc():
    tree, _, _ = build_static_module_tree(load(...), conditional_mode='infer')
    # Wrapper.bb 必须保留
    assert 'bb' in tree['Wrapper']['attrs']
    # Backbone 必须可达
    assert 'Backbone' in tree
    # bb 必须有 first_call_loc
    fcl = tree['Wrapper'].get('first_call_loc', {})
    assert 'bb' in fcl, "first_call_loc 必须把 self.bb.custom_query() 识别为对 bb 的调用"
    # 端到端：DAG 不会因为 dead-child 而剪掉 bb
    data = generate_html_flowchart(load(...), _return_data_only=True, conditional_mode='infer')
    cls_name_set = {n.get('class_name') for n in data['nodes']}
    assert 'Backbone' in cls_name_set
```

### 5.5 端到端回归测例（覆盖 5476790 真实场景）

在 `testset/test_dag_rules.py` 的"防回退基准表"中，把 5476790 的目标值上调：

```python
EXPECTED_BASELINE = {
    "5476790": {
        "nodes": 229,           # ← 当前 196，修复后必须 ≥ 229
        "groups": 149,
        "top_edges": 455,       # 与 iter17 baseline 对齐
        "internal_edges": 200,
    },
    # 其他 5 个模型保持不变
}
```

并新增类级别断言函数：

```python
def check_5476790_class_coverage(data):
    """5476790 必须包含以下用户定义类的至少一个实例。"""
    must_have = ['AGGDebiasModule', 'TransBlock', 'Attention', 'GatedFFN',
                 'SeqTrans', 'InsTrans', 'C2kTrans']
    cls_set = {n.get('class_name') for n in data['nodes']}
    missing = [c for c in must_have if c not in cls_set]
    assert not missing, f"5476790 缺少类: {missing}"
```

---

## 6. 验证步骤

### 6.1 任务开始前

按"任务开始前强制检查清单"：
1. 确认工作目录是 `ast_refactor_workdir` 独立 clone：`cd ast_refactor_workdir && git remote -v`
2. 确认在 `feat/ast-refactor` 分支：`git branch --show-current`
3. 拷贝基线：`cp scripts/analyze_trace.py scripts/analyze_trace_5476790_node_gap_BASELINE.bak`

### 6.2 修复实施顺序（建议）

1. **先 E2**（first_call_loc 扩展） ─ 风险最低，单点改动，加 helper-method 4 行 case
2. **再 E1.1**（字面量 list ListComp） ─ 逻辑简单，影响面小
3. **再 E1.3**（setattr loop-var 纯字符串） ─ 已有部分基础设施
4. **最后 E1.2**（跨类常量解析） ─ 最复杂，需要扩 ConstantResolver

每完成一步：
- 跑 `python3 testset/synthetic_cases/run_synthetic_tests.py`
- 跑 `ANALYZE_SCRIPT_OVERRIDE=$(pwd)/scripts/analyze_trace.py python3 testset/test_dag_rules.py` 7/7 PASS
- 看 5476790 nodes 增长趋势：196 → 预期目标 229

### 6.3 最终验证（合入 master 前）

```bash
cd ast_refactor_workdir

# 1. Rebase master
git fetch origin
git rebase origin/master

# 2. 全量测试
ANALYZE_SCRIPT_OVERRIDE=$(pwd)/scripts/analyze_trace.py \
  python3 testset/test_dag_rules.py 2>&1 | tee /tmp/full_test.log

# 3. 检查 5476790 关键指标
grep -E "5476790.*nodes=" /tmp/full_test.log
# 期望输出: 5476790 | PASS | 229 | 149 | 455 | 200 | ...

# 4. 类覆盖断言
python3 -c "
import json, subprocess, os
os.environ['ANALYZE_SCRIPT_OVERRIDE'] = os.path.abspath('scripts/analyze_trace.py')
import sys, importlib.util
sys.path.insert(0, os.path.abspath('scripts'))
spec = importlib.util.spec_from_file_location('m', 'scripts/analyze_trace.py')
m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
src = m.load_model_code('testset/extracted/5476790')
data = m.generate_html_flowchart(src, timing_data=None, meta=None,
                                 output_path='/tmp/_x.html', trace_events=None,
                                 conditional_mode='infer', _return_data_only=True)
nc = {n.get('class_name') for n in data['nodes']}
must = ['AGGDebiasModule','TransBlock','Attention','GatedFFN','SeqTrans','InsTrans','C2kTrans']
missing = [c for c in must if c not in nc]
print('5476790 nodes=', len(data['nodes']))
print('缺失类:', missing)
assert not missing
"
```

### 6.4 合入 master

按"合入 master 规范"：
1. ✅ Rebase master 完成
2. ✅ 7/7 PASS（含新增 4 个合成测例）
3. ✅ 5476790 nodes ≥ 229，类覆盖完整
4. ⏸ **返回测试报告，等待 NaN 明确确认合入分支**
5. ⏸ NaN 确认后才能 push

---

## 7. 风险与回退预案

| 风险 | 触发条件 | 回退动作 |
|---|---|---|
| E1.1 字面量 list 展开过激进，对其他模型产生假节点 | 5547919 / 5698781 nodes 异常增长 | 回退到 BASELINE.bak，缩小 list 类型为"全字符串字面量"白名单 |
| E1.2 跨类常量解析引入误报 | TransformerConfig field 默认值被错误使用 | fallback 仅在 `_n is None and isinstance(_gen0.iter, ast.Call) and func==range` 时生效，且必须显式断言 receiver 是 self 字段 |
| E2 first_call_loc 扩展导致 Rule1c 失败 | helper-method 行号被当作 forward call line 但未指向 module call | 在 dep_edge 生成时仅把 helper-method 行作为 first_call_loc 的备选，dep_edge 优先使用 self.attr(...) 直接调用行 |

每条修复都必须：
- 有自己的 BASELINE.bak（任务开始前 cp）
- 有合成测例（synthetic_cases/）
- 有真实回归（test_dag_rules.py 7/7 PASS）

---

## 8. 不在本次修复范围内（已确认）

- `[WARN] ModuleDict key not enumerable: ...`：5476790 测试日志中大量出现，但属于 Iter13 Step2 已知遗留（动态 key 拼接如 `f'{name}_RecycleNet_{i}'`），与本次 196→229 节点缺失**无直接关联**。
- 5547919 Rule3 假环：MMCN/ResNet ModuleDict 多 key 实例展开问题，独立任务。
- timing 模块：本任务只涉及静态 DAG，不动 timing。

---

**生成者**：Constructor 🕵️‍♂️
**日期**：2026-05-26
**状态**：方案已就绪，等待 NaN 审核确认后再实施修复代码。
