# 模型 6451993 性能分析报告

---

<!--
【内部核验区】
- 原始 Trace：5 个 host ProfilerStep（1505–1509），事件按 Step 边界裁剪。
- 正式数值：output/perf_metrics_6451993.json、output/timing_panel_6451993_fixed.json、output/kernel_top20_6451993.json、output/analysis_6451993.json。
- DAG：output/dag_6451993_training_oracle_train_20260805_2051.json。
- 分析侧原子 fact rollup：38555 个已归因 GPU events；2038 个唯一映射；mapped duration 3.706 ms/step（0.363183%）；形成 6/663 个 synthetic Pattern 局部子集；production Pattern rolled timing 为 0/663。
- Module attach + rollup：29 runtime records，production matched 6、miss 23；ResidualAttentionBlock 的 1 条记录按 10 个等长实例均分；15 direct groups；Module rolled groups 15/1144（1.31%）。
- 守恒：Pattern events 是 mapped events 子集；父包含子；同父 sibling 集无重叠；同深度局部 Pattern 无重叠。跨层 inclusive Pattern 不求和。
- Kernel：Top20 已完成 20/20 源码解法回链；#4 Adamom 由 P1/E1-E 严格 optimizer 合组 Harness 覆盖，生产入口仍由 I-09 采集。
- 自查：0–13 章完整；正式 Patch 为 2 项 P0/E0、12 项 P1/E1（C–N）；另有 2 项 P2/E2 风险实验；Top20 20/20 回链正式代码或严格实验入口。
-->

## 0. 报告状态与局部数据提示

**最终状态：`COMPLETE`**

本报告完成 Step、通信和 Top20 20/20 源码解法分析；#4 Adamom 由 Patch E 提供可执行实验，真实生产入口仍待 I-09 归因。六个 Top Module 的类级固定字段已逐项闭合。Pattern 因无 Kernel→DAG leaf 细粒度 timeline，按真实 DAG 使用确定性结构优先级并完成源码行为分析；该数据缺口不阻碍交付。

| 章节 | 必需数据源 | 状态 | 说明 |
|---|---|---|---|
| Step | 原始 Trace + host Step 边界 | PASS | 5 个 Step 独立裁剪、求并集并闭合到 1192.304 ms/step |
| Module 类级行为 | DAG class/source + 模型源码 | PASS | 报告涉及的业务类、Transformer 子类与 Top Kernel 所属类均已解释 |
| Module 实例 timing | production attach + post-order rollup | LOCAL_COVERAGE | 29 条 runtime records：6 matched、23 miss；1 条 ResidualAttentionBlock 合法均分到 10 groups；15 direct groups；Module rolled groups 为 15/1144（1.31%） |
| Pattern | DAG JSON + production rollup + 结构统计 | STRUCTURAL_ONLY | Timeline 无 Kernel→DAG leaf 具体映射；按 Module 热点时间及同 Module 内 base name 实例数两级排序，不输出 Pattern timing |
| Kernel | 原始 Trace + Top20 JSON + class/算子源码 | COMPLETE_WITH_EXPERIMENT | 20/20 均有源码解法；#4 由 Patch E 覆盖实验，I-09 补生产入口 |
| 通信 | 原始 Trace 通信标注 + 源码 | PASS | AllToAll、AllGather、AllReduce、ReduceScatter 已按类型合并并计算 overlap/exposed |

### 数据口径

1. **Step/通信墙钟口径**：GPU 区间先裁剪到各 host Step，再做 union/intersection；因此可用于关键路径闭合。
2. **Kernel 排名口径**：逐事件 duration sum，多执行队列并发可能重复计入工作量；正式 Kernel 分母为 1192.304 ms Step，Top20 合计 493.218 ms/step。
3. **Module timing 口径**：production attribution → instance timing 汇总 → attach 到 eligible DAG group → 按 DAG 子 group 关系 post-order rollup。父子 Module 是 inclusive 范围，不能跨层相加。
4. **Pattern 口径**：production attach/rollup 的 synthetic Pattern 结果为 0/663，且 Timeline 不含 Kernel→DAG leaf 的确定映射；不补造数据，也不将 Module 时间分摊或继承给 Pattern。
5. **Pattern 结构排序**：先按所属 Module 已核验热点时间降序，同一 Module 内再按去 `#N` 后的真实 DAG 实例数降序。Module 时间仅作排序上下文；第 3 章不输出 Pattern ms/step、Step 占比或基于 Pattern timing 的收益。

---

## 1. Step 整体分解

### 1.1 每 Step 时长明细

| Step | CPU 时长 (ms) | Compute union (ms) | Comm union (ms) | overlap (ms) | exposed comm (ms) | kernel gap (ms) |
|---|---:|---:|---:|---:|---:|---:|
| 1505 | 1212.519 | 724.684 | 422.934 | 15.049 | 407.885 | 79.950 |
| 1506 | 1193.102 | 727.424 | 408.222 | 14.877 | 393.345 | 72.334 |
| 1507 | 1241.470 | 718.136 | 464.907 | 14.521 | 450.386 | 72.948 |
| 1508 | 1147.850 | 730.302 | 363.148 | 14.615 | 348.533 | 69.015 |
| 1509 | 1166.580 | 734.580 | 380.038 | 14.714 | 365.324 | 66.676 |
| **平均** | **1192.304** | **727.025** | **407.850** | **14.755** | **393.094** | **72.185** |

### 1.2 GPU 时间分解（Step 平均）

| 区间类型 | 时长 (ms) | 占 Step |
|---|---:|---:|
| Compute kernel union | 727.025 | 60.976% |
| Communication kernel union | 407.850 | 34.207% |
| Compute ∩ Communication | 14.755 | 1.238% |
| 暴露通信 | 393.094 | 32.969% |
| Kernel gap/非 kernel 区间 | 72.185 | 6.054% |

闭合关系为 `727.025 + 393.094 + 72.185 = 1192.304 ms`。这里的 gap 未统计 memcpy/memset 等其他 engine 活动，含义是“没有 compute/NCCL kernel 的墙钟区间”，不是设备硬件空闲率。总体通信 overlap ratio 按逐 Step 比率平均为 **3.644%**。

### 1.3 Step 关键路径结论

- **主要阶段**：5-Step attribution duration 中 forward 约 277.371 ms/step、backward 585.723 ms/step、optimizer 151.840 ms/step、other 5.361 ms/step；该分解是事实 duration 口径，不与 union 表直接相加。
- **关键路径**：TP 分片布局转换 → Transformer/MoE 计算 → 反向参数与梯度通信；布局转换和 MoE dispatch/combine 都有严格 producer/consumer 依赖。
- **首要瓶颈**：暴露通信 **393.094 ms/step（32.969%）**；计算侧首要聚合热点是 6 个 MoE Grouped GEMM 变体，合计 **234.111 ms/step（19.635% Step）**。

---

## 2. Module 类级代码行为与实例级成本分析

### 2.1 类级清单与可分配 timing 排序

production 覆盖统计：29 条 runtime records 中 6 matched、23 miss；15 个 direct groups；15/1144 个 Module groups 获得 rolled timing（1.31%）。下表只排序这一局部覆盖子集，不能解释为全量 Module 排名。

| 排名 | Module class / 语义 | class 定义 | forward 范围 | group ID | 实例数 | 类级分析 | timing 状态 | Fwd | Bwd | 总计 | 单实例范围 | 占 Step |
|---:|---|---|---|---|---:|---|---|---:|---:|---:|---:|---:|
| 1 | CVRModule / 全模型根模块 | `main_model.py:L2810` | `L2930-L3100` | 3158 | 1 | PASS | allocated | 51.880 | 117.145 | 169.025 | 169.025 | 14.18% |
| 2 | CvrTransformer / 混合 token 主塔 | `main_model.py:L2055` | `L2144-L2214` | 11107 | 1 | PASS | allocated | 29.053 | 115.194 | 144.247 | 144.247 | 12.10% |
| 3 | Transformer / 10 层残差块容器 | `pure_transformer.py:L1055` | `L1135-L1153` | 11240 | 1 | PASS | allocated | 13.554 | 98.781 | 112.335 | 112.335 | 9.42% |
| 4 | ResidualAttentionBlock / mixer+MoE 残差块 | `pure_transformer.py:L905` | `L1027-L1053` | 11243、11508、11774、12039、12305、12570、12836、13101、13367、13632 | 10 | PASS | production even split | 13.519 | 96.823 | 110.343 | 11.034 | 9.25% |
| 5 | SequenceModule / 序列特征编排 | `main_model.py:L1210` | `L1431-L1730` | 6504 | 1 | PASS | allocated | 22.221 | 0.000 | 22.221 | 22.221 | 1.86% |
| 6 | C2kTrans / 2k/pay 序列查询 | `main_model.py:L888` | `L1029-L1105` | 10730 | 1 | PASS | allocated | 17.777 | 0.000 | 17.777 | 17.777 | 1.49% |
| — | Dense、LayerNorm、Attention、GatedFFN、SparseMoe 等 | 见第 2.2.7 节 | 见第 2.2.7 节 | N/A | 多实例 | PASS | class-level | N/A | N/A | N/A | N/A | N/A |

> 单位均为 ms/step，最后一列除外。表中为 inclusive timing，父子行有重叠。SequenceModule/C2kTrans 的 Bwd=0 表示本次可分配记录中的 phase 结果，不代表类在所有训练配置中无梯度。

### 2.2 Module 详情

#### 2.2.1 `CVRModule`

- **输入**：入口无显式参数，从框架特征列和全局 `layers_map` 读取 dense、embedding、序列及标签。
- **计算结构/正常行为**：dense 标志与 mask → UE/NFM/序列特征 → `SequenceModule` → `CvrTransformer` 与实例序列分支 → debias/聚合任务头 → `LossModule`。
- **输出**：训练态返回 `(result, total_loss.sum())`，推理态返回 `result`；中间 Transformer 表征进入任务头和 loss。
- **子模块/custom op/collective**：包含 `SequenceModule`、`CvrTransformer`、`ins_trans`、任务头和 `LossModule`；序列分支使用 `SequenceTensor`，collective 位于下游 DTensor/Transformer wrapper。
- **Fwd/Bwd 成本**：group 3158，51.880/117.145 ms，合计 169.025 ms/step；Backward 包含 10 层 Transformer 梯度、checkpoint 重算及通信，数值为 inclusive。
- **关联 Pattern**：按真实 DAG 结构，CVRModule 祖先下有 322 个 Pattern 实例；结构表中优先项为 Pattern5（53）、Pattern4（32）和 cal_loss（29）。`Pattern6 → matrix_similarity` 保留为源码结构证据，不赋 Pattern timing。
- **关联 Kernel**：其 Transformer 子树覆盖第 4.2.1 MoE Grouped GEMM、第 4.2.3 LayerNorm backward、第 4.2.4 FlashAttention、第 4.2.6 NVJET GEMM；#4 Adamom 在其 backward 之后，由 P1/E1-E 提供严格实验，I-09 只补生产入口。
- **候选证据分类**：根模块不是直接算子热点；MoE 路由/tile、checkpoint 重算及 collective placement 仅构成下游证据入口，处置见第 6–10 章。任务头、loss、标签与训练/推理返回语义是强制不变量；低覆盖 Pattern 子集不用于收益。

源码证据分为两个互不连续的原样片段：

```python
# main_model.py:2994-3002
bottom_concat_list = []
combine_trans_avg_out = self.mix_transformer()
bottom_concat_list.append(combine_trans_avg_out)
trans_soft_flow_ins, length_info_ins = self.ins_trans(
    uid_more,
    joiner_backfill_mask,
    torch.cat(bottom_concat_list, dim=1),
    prepared_sequence=prepared_ins_sequence,
)
```

```python
# main_model.py:3097-3100
if self.training:
    return result, result.get_total_loss().sum()
else:
    return result
```

#### 2.2.2 `CvrTransformer`

- **输入**：从 `layers_map` 拼接 raw token，形成 `[B,S,D]`，训练混精度时转 BF16。
- **计算结构/正常行为**：`cat` → token split → LayerNorm → per-token AFFN + padded residual → LayerNorm → 10 层 `Transformer` → mixer concat/mean → DTensor 本地化。
- **输出**：返回 `combine_trans_out.mean(1)`，并把中间 mixer 输出分块写回 `layers_map['mixer_all_outputs']`。
- **子模块/custom op/collective**：`orig_token_ln`、`orig_pertoken_affn`、`tokens_ln`、`combine_trans`；优先 `mixer_mean_redistribute`，回退路径两次 `BwRedistribute`，其 backward 可触发 DTensor AllToAll。
- **Fwd/Bwd 成本**：group 11107，29.053/115.194 ms，合计 144.247 ms/step，inclusive 包含 Transformer 子树和布局转换。
- **关联 Pattern**：按最近已核验热点 Module 祖先归属，CvrTransformer 下有 14 个 Pattern 实例；Top3 为 `<listcomp>`（5）、`_call_tower`（2）和 `act_layer`（2）。Pattern 无细粒度 timeline。
- **关联 Kernel**：第 4.2.1 MoE Grouped GEMM、第 4.2.3 LayerNorm backward、第 4.2.5 BF16 copy/cast、第 4.2.6 NVJET GEMM；FlashAttention 是否落入该具体实例不作唯一归因。
- **候选证据分类**：mixer mean 已落实为 P1/E1-N（第 8.12 节），以 `[Shard(1)]` 输入、`token_num=self.num_tokens` 和 `[Shard(0)]` 输出的严格契约验收；布局转换与 BF16 物化分别回链 F/J。任何 dtype、DTensor placement 或调用签名不一致均硬失败。

以下为两个实际行段的原样源码：

```python
# main_model.py:2165-2177
# 3. batchmatmul & LN
tokens = self.orig_pertoken_affn(orig_tokens_ln)
tokens = tokens + F.pad(
    orig_tokens_ln,
    [0, self.model_dims - self.split_dims],
)
tokens_ln = self.tokens_ln(tokens)
combine_trans_input = tokens_ln

# if self.mixed_precision and is_training:
    # combine_trans_input = combine_trans_input.bfloat16()

combine_trans_out, mixer_all_outputs = self.combine_trans(combine_trans_input)
```

```python
# main_model.py:2198-2204
# mixer_all_outputs_concat = torch.cat(mixer_all_outputs, dim=2).mean(1)
mixer_all_outputs_concat = torch.cat(mixer_all_outputs, dim=2)
mixer_all_outputs_concat = BwRedistribute().apply(mixer_all_outputs_concat, 1)
mixer_all_outputs_concat = mixer_all_outputs_concat.mean(1)
mixer_all_outputs_concat = BwRedistribute().apply(mixer_all_outputs_concat, 0)
if isinstance(mixer_all_outputs_concat, DTensor):
    mixer_all_outputs_concat = mixer_all_outputs_concat.redistribute(placements=[Shard(0)]).to_local()
```

#### 2.2.3 `Transformer`

- **输入**：`[B,S,4096]` token 与可选 `attn_mask`。
- **计算结构/正常行为**：顺序执行 10 个 `ResidualAttentionBlock`；奇数层收集中间表示，非末层每两块加入 `last_x` 跨块残差。
- **输出**：返回末层 `x` 和 `trans_outs`；后者供 `CvrTransformer` 的 mixer concat/mean 与任务侧消费。
- **子模块/custom op/collective**：`ModuleList[ResidualAttentionBlock]`；collective 不在容器代码显式调用，由 block 的 DTensor wrapper/resharding 触发；custom op 位于 block 内 SparseMoe。
- **Fwd/Bwd 成本**：group 11240，13.554/98.781 ms，合计 112.335 ms/step；与 10 个子 block 为 inclusive 重叠。
- **关联 Pattern**：Transformer 作为容器不重复承接已归给更近 ResidualAttentionBlock 祖先的 Pattern，因此直接归属计数为 0；其内部结构关系见下一节。
- **关联 Kernel**：通过子 block 关联第 4.2.1 MoE Grouped GEMM、第 4.2.3 LayerNorm backward、第 4.2.5 copy/cast、第 4.2.6 NVJET GEMM。
- **候选证据分类**：checkpoint 为第 9.1 节 P2/E2；跨块残差与重分片边界只进入调查。10 层顺序、奇数层输出列表和 `last_x` 是强制契约。

下列为 `pure_transformer.py:1141-1153` 连续原文；仅未摘录 1136–1140 的注释行：

```python
# pure_transformer.py:1141-1153
trans_outs = []
last_x = x
wandb_histogram(f"MLPMixer/resblock_0", x)
for i, resblock in enumerate(self.resblocks):
    x = resblock(x, attn_mask=attn_mask)
    wandb_histogram(f"MLPMixer/resblock_{i+1}", x)
    if i %2 == 1:
        trans_outs.append(x)
        if i < self.layers - 1:
            x = x + last_x
            last_x = trans_outs[-1]

return x, trans_outs
```

#### 2.2.4 `ResidualAttentionBlock`

- **输入**：逻辑 `[B,S,4096]` token 与可选 attention mask；trace 常见展平形态 `[16384,1,4096]`。
- **计算结构/正常行为**：MLPMixer → 残差加 + `ln_1` → `SparseMoe`/AFFN → 残差加 + `ln_2`；可选 cross-connection 再经 attention + `ln_3`。
- **输出**：与输入逻辑 shape 相同的 block 表征，送入下一 block，并在奇数层可能进入 `trans_outs`。
- **子模块/custom op/collective**：`MLPMixer`、LayerNorm、`SparseMoe`（router→index/scatter→SwiGLU expert→gather）；`@checkpoint_forward` 令 backward 可能重算；collective 由外围 DTensor reshard 触发。
- **Fwd/Bwd 成本**：1 条 runtime 记录按 10 个等长候选 groups 11243、11508、11774、12039、12305、12570、12836、13101、13367、13632 均分；每个 1.352/9.682 ms、合计 11.034 ms，10 个共 110.343 ms/step。
- **关联 Pattern**：按最近热点 Module 祖先归属，ResidualAttentionBlock 下有 110 个 Pattern 实例；Top3 为 `_call_tower`（40）、`act_layer`（40）和 Pattern37（20）。这些是结构计数，不含 Pattern timing。
- **关联 Kernel**：第 4.2.1 Grouped GEMM、第 4.2.3 LayerNorm backward、第 4.2.7 MoE gather/combine，mixer/投影还关联第 4.2.6 NVJET GEMM。
- **候选证据分类**：checkpoint 与 top-k 分别为第 9.1/9.2 节 P2/E2；expert 负载/tile 仅是调查证据。residual/dropout、cross-connection 与 router 语义不能假定等价；10-way 均分不能比较具体层实例。

逐字源码及实验代码见第 9.1 节的 `pure_transformer.py:1026-1053`。

#### 2.2.5 `SequenceModule`

- **输入**：场景动作、user-aware embedding、回填 mask、用户特征及实例/分段边界。
- **计算结构/正常行为**：准备 packed/ULT sequence → 拼接广告与用户表征 → 为各槽构造 live/video query → attention/sum 多尺度特征 → 堆叠 `mix_query` → `c2k_trans` 与 `cpay_trans` 两分支。
- **输出**：返回各序列 `seq_bias_dict` 及 `2k/pay` 两路长度信息；两路 soft-flow 同时写入 `layers_map` 供下游消费。
- **子模块/custom op/collective**：embedding、DenseTower、`AttentionDinNoSoftMaxAds`、两个 `C2kTrans`；使用 `SequenceTensor.from_masked_tensor`。本类无显式 collective，下游 SeqTrans/DTensor 可触发通信。
- **Fwd/Bwd 成本**：group 6504，22.221/0.000 ms，合计 22.221 ms/step；Bwd=0 仅表示本次可分配记录 phase，不代表类恒无梯度。
- **关联 Pattern**：SequenceModule 下有 199 个真实 DAG Pattern 实例；Top3 为 Pattern5（46）、Pattern16（45）和 Pattern15（26），按实例数排序且不赋 timing。
- **关联 Kernel**：本节局部 Pattern 的 equality/tile/where/mask 未进入 Top20；下游 SeqTrans 可关联第 4.2.4 FlashAttention和第 4.2.6 NVJET GEMM，但不作实例唯一归因。
- **候选证据分类**：packed sequence 已 frontload，现无重复 pack 证据；query 的 stack→sum 已转为第 8.1 节 Patch C，其余 query 投影仍进入 I-01，不借父 Module timing 承诺收益。

```python
# main_model.py:1699-1730
for idx, query_name in enumerate(mix_query_names):
    q_t = self.query_tensors_for_mixq[query_name](layers_map[query_name])
    mix_query_list.append(q_t)

mix_query = torch.stack(mix_query_list, dim=1)
# print("adebug mix_query", mix_query.shape) [bs, 18, 512]
transf_soft_flow_2k, length_info_2k = self.c2k_trans(
    uid_more,
    c2k_live_query_concat,
    c2k_video_query_concat,
    external_action,
    real_instance_flag,
    segment_begin_flag,
    joiner_backfill_mask,
    mix_query,
    prepared_sequence_delta=prepared_c2k_sequence_delta,
    prepared_segment_begin_idx=prepared_c2k_segment_begin_idx,
)
layers_map["transf_soft_flow_2k"] = transf_soft_flow_2k
transf_soft_flow_pay, length_info_pay = self.cpay_trans(
    uid_more,
    cpay_live_query_concat,
    cpay_video_query_concat,
    external_action,
    real_instance_flag,
    segment_begin_flag,
    joiner_backfill_mask,
    mix_query,
    prepared_sequence=prepared_cpay_sequence,
)
layers_map["transf_soft_flow_pay"] = transf_soft_flow_pay
return seq_bias_dict, {"2k": length_info_2k, "pay": length_info_pay}
```

#### 2.2.6 `C2kTrans`

- **输入**：用户 embedding、live/video query、场景动作、实例/分段 mask、回填 mask 与 `mix_query`。
- **计算结构/正常行为**：三路 DenseTower 投影 → `is_video_samples` + `where` 场景选择 → 与 mix query 拼接 → query 投影 → ULT、服务或 packed-dense 三选一 SeqTrans 查询。
- **输出**：返回展平 soft-flow 表征和长度元数据，送回 `SequenceModule` 并写入下游 `layers_map`。
- **子模块/custom op/collective**：live/video/uid DenseTower、`trans_query`、`seq_trans`；packed 路径创建 `SequenceTensor`，下游含 LayerNorm、Attention、GatedFFN；本类无显式 collective。
- **Fwd/Bwd 成本**：group 10730，17.777/0.000 ms，合计 17.777 ms/step；Bwd=0 仅为本次可分配记录结果。
- **关联 Pattern**：C2kTrans 下有 18 个真实 DAG Pattern 实例；Top3 为 dense_query（6）、Pattern24（2）和 Pattern26（2）。它们只作结构优先级，不输出 Pattern timing。
- **关联 Kernel**：上述局部 Pattern 事件未进入 Top20；`seq_trans` 的 Attention/GatedFFN 可关联第 4.2.4 FlashAttention和第 4.2.6 NVJET GEMM，但当前不能唯一归到该实例。
- **候选证据分类**：query 投影、packed sequence 与长度元数据仅进入第 10 章调查；视频/直播选择、三种 sequence 路径、mask 与长度元数据是准入契约，结构实例数不外推收益。

```python
# main_model.py:1055-1069
query_emb_trans = torch.where(
    torch.tile(
        is_video_samples(external_action), [1, self.seq_trans_base_params["hidden_size"]]
    ),
    query_emb_trans_video,
    query_emb_trans_live,
)
# if self.fc_name == "fc_user_ecom_2k_mix_4d":
#     trans_query = torch.cat([query_emb_trans, uid_emb_trans], dim=1)[:, None, :]
#     query_len = trans_query.shape[1]
# else:
trans_query = torch.stack([query_emb_trans, uid_emb_trans], dim=1)
trans_query = torch.cat([mix_query, trans_query], dim=1)
query_len = trans_query.shape[1]
trans_query = self.trans_query(trans_query)
```

```python
# main_model.py:1094-1100
else:
    if prepared_sequence is None:
        prepared_sequence = self.prepare_packed_dense_sequence(joiner_backfill_mask)
    transf_soft_flow_2k, length_info_2k = self.seq_trans.dense_query(
        query=trans_query,
        sequence=prepared_sequence
    )
```

#### 2.2.7 Top Kernel 所属共享类

| 类 | 输入→计算→输出 | 子模块/custom op/collective | Kernel 行为与编号处置 |
|---|---|---|---|
| `Dense` | `[...,K] × [K,N] + bias → [...,N]`，训练混精度时权重转 BF16 | ATen matmul/mm | NVJET GEMM；真实 M/N/K、模板和重复 cast 仅进入 I-08/I-14 |
| `BatchMatMulDense` | `[S,M,K] × [S,K,N] → [S,M,N]` | `torch.matmul`，可加 bias/activation | NVJET batched GEMM；热点由大 K/N 与多 token 批次产生 |
| `LayerNorm` | 对最后一维 4096 计算 mean/rstd，仿射输出同 shape | 残差 add 前后由 Inductor 融合 | Triton backward 归约；涉及 dgamma/dbeta 归约和激活/梯度读写，未采 bytes，不能断言 HBM bound |
| `Attention` | Dense 生成 Q/K/V → `[tokens,heads,64]` → FlashAttention → out Dense | RoPE、KV cache、FlashAttention | 前向在线 softmax；反向拆 dQ 与 dK/dV，不物化完整 N² score |
| `GatedFFN` | `SiLU(gate(x))*fc1(x)` → fc2 回 hidden | 可调用 `fused_swiglu`，可重算 | NVJET/Triton 线性与融合激活；中间物化与 checkpoint 处置见 I-08/I-10 和 P2 9.1 |
| `SparseMoe` | Router top-2 → index/count → scatter → expert FFN → weighted gather | `MoeLinear`、index/scatter/gather custom ops；DTensor 可承载 EP | 已使用 CUTLASS Grouped GEMM；persistent、负载、top-k、tile 分别见 I-11、P2 9.2/I-14 |
| `PerTokenSwiGLUFFN` | fc1/fc2: 4096→512；`SiLU(fc1)*fc2`；fc3: 512→4096 | 三次 `MoeLinear` | 6 个 Grouped GEMM 变体的直接类位置 |
| `TransBlock/SeqTrans` | LN→Attention→残差→LN→GatedFFN→残差 | SequenceTensor、FlashAttention | FlashAttention 与 Dense GEMM 的业务宿主；shape 随 packed token 长度变化 |

---

## 3. Pattern 结构优先级分析

### 3.1 口径与真实 DAG 统计

本次 Timeline 没有 Kernel→DAG leaf 的确定映射，因此本章采用**结构优先级，不是 Pattern timing**。排序规则固定为：先按所属 Module 的已核验热点时间降序；同一 Module 内再按 Pattern base name 去除 `#N` 后的真实 DAG 实例数降序。Module 时间只作排序上下文，未分摊或继承给任何 Pattern；本章不输出 Pattern ms/step、Step 占比，也不据此估算收益。**timeline 具体 Kernel 信息均为：无。**

对 `output/dag_6451993_training_oracle_train_20260805_2051.json` 的 663 个 synthetic groups 逐项去除 `#N` 后缀，并沿 `children_group_ids` 反向寻找最近的已核验热点 Module 祖先，得到 96 个“所属 Module + base name”组合。归属守恒为：CVRModule 322、CvrTransformer 14、Transformer 0、ResidualAttentionBlock 110、SequenceModule 199、C2kTrans 18，合计 **663**。Transformer 的 0 表示其 descendant Pattern 均由更近的 ResidualAttentionBlock 祖先承接，不重复计数。

### 3.2 Pattern 结构 Top 表

> 为避免 CVRModule 的 57 个 base name 挤占整表，下表展示每个有直接归属 Pattern 的热点 Module 内 Top3；表内 Module 顺序和 Module 内顺序仍严格遵循上述两级规则。

| 结构优先级 | Pattern base name | 所属 Module 热点时间/排名（仅排序上下文） | Pattern 调用次数/实例数 | 源码结构/算子链 | timeline 具体 Kernel 信息 |
|---:|---|---|---:|---|---|
| 1 | Pattern5 | CVRModule：#1，169.025 ms/step | 53 | `model_common.py:360/366/519`：embedding→mask→`zeros_like→where` | 无 |
| 2 | Pattern4 | CVRModule：#1，169.025 ms/step | 32 | `model_common.py:409/427/1131`、`main_model.py:673`：`stack→sum`；`cat` 仅为相邻分支 | 无 |
| 3 | cal_loss | CVRModule：#1，169.025 ms/step | 29 | `main_model.py:2558-2566` 等：label 有效性 mask→多任务 loss 调用 | 无 |
| 4 | `<listcomp>` | CvrTransformer：#2，144.247 ms/step | 5 | `main_model.py:2196`：五路 mixer 输出 dtype 转换 | 无 |
| 5 | `_call_tower` | CvrTransformer：#2，144.247 ms/step | 2 | `pure_transformer.py:459`：adaptive tower 分支 | 无 |
| 6 | act_layer | CvrTransformer：#2，144.247 ms/step | 2 | `pure_transformer.py:460`：主塔与 adaptive 输出融合 | 无 |
| 7 | `_call_tower` | ResidualAttentionBlock：#4，110.343 ms/step | 40 | `pure_transformer.py:459`：10 个 block 内多层 adaptive tower | 无 |
| 8 | act_layer | ResidualAttentionBlock：#4，110.343 ms/step | 40 | `pure_transformer.py:460`：adaptive activation/gating 融合 | 无 |
| 9 | Pattern37 | ResidualAttentionBlock：#4，110.343 ms/step | 20 | `pure_transformer.py:863/898`：MLPMixer head/token reshape→transpose→reshape | 无 |
| 10 | Pattern5 | SequenceModule：#5，22.221 ms/step | 46 | `model_common.py:495`、`main_model.py:1612-1615`：embedding mask 与序列聚合 | 无 |
| 11 | Pattern16 | SequenceModule：#5，22.221 ms/step | 45 | `main_model.py:1578/1674`：多尺度 sequence slice→sum | 无 |
| 12 | Pattern15 | SequenceModule：#5，22.221 ms/step | 26 | `main_model.py:1553/1559/1560/1655`：序列/位置 bias slice | 无 |
| 13 | dense_query | C2kTrans：#6，17.777 ms/step | 6 | `main_model.py:1097`：packed sequence + query→dense attention | 无 |
| 14 | Pattern24 | C2kTrans：#6，17.777 ms/step | 2 | `model_common.py:183`：两次 equality→logical-or 场景 mask | 无 |
| 15 | Pattern26 | C2kTrans：#6，17.777 ms/step | 2 | `trans_refactor.py:322`：segment begin pad→相邻差→长度元数据 | 无 |

### 3.3 Pattern 结构与源码行为

#### 3.3.1 `Pattern5`：embedding 掩码与聚合

- **排序依据**：CVRModule 内 53 个实例、SequenceModule 内 46 个实例；两处按各自所属 Module 独立统计，不跨 Module 合并排名。
- **输入/输出**：输入 slot embedding 与 live/gip bool mask；输出保留原 shape 的 masked embedding，随后进入 list、concat 或 sum 聚合。
- **成员节点/子 Pattern**：`zeros_like→where`，以及分支后的 `stack→sum` / `cat`；同一 base name 在不同 Module 祖先下保留不同归属。
- **对应 Module class**：CVRModule、SequenceModule，以及其内嵌的 NoShare/ShareMultiSlotEmbedding 数据流。
- **可能的 Kernel 实现关系**：elementwise compare/select、copy/cast、concat/reduction；仅为源码算子到实现类型的可能关系。
- **timeline 具体 Kernel 信息**：无。
- **候选证据分类**：视频 mask/tile 已转为第 7.2 节 Patch B；其他 mask 合并和零张量生产者融合因 consumer/alias 未闭合继续进入 I-10。

```python
# model_common.py:357-368
if hasattr(emb, "live_mask") and emb.live_mask:
    emb_vec = torch.where(
        live_type_mask,
        torch.zeros_like(emb_vec),
        emb_vec,
    )
if hasattr(emb, "gip_mask") and emb.gip_mask:
    emb_vec = torch.where(
        gip_mask,
        torch.zeros_like(emb_vec),
        emb_vec,
    )
```

#### 3.3.2 `<listcomp>`、`_call_tower` 与 `act_layer`

- **排序依据**：CvrTransformer 的 `<listcomp>` 为 5 个实例，`_call_tower`/`act_layer` 各 2 个；ResidualAttentionBlock 的 `_call_tower`/`act_layer` 各 40 个。
- **输入/输出**：mixer 分支接收 `[B,S,D]` 张量；list comprehension 将输出转换到 `S.dtype()`；adaptive FFN 先计算旁塔输出，再与主塔激活融合并送入下一线性层。
- **成员节点/子 Pattern**：dtype cast；`_call_tower→act_layer→linear→activation→dropout`。
- **对应 Module class**：CvrTransformer、ResidualAttentionBlock 内的 AdaptiveFFN。
- **可能的 Kernel 实现关系**：BF16 copy/cast、dense GEMM、activation 与 dropout；MoE/Grouped GEMM 只在实际走 SparseMoe 路径时相关，不能由本结构表断言。
- **timeline 具体 Kernel 信息**：无。
- **候选证据分类**：fused mixer mean 无条件命中与 placement 验证回链 P1/E1-N；通用 cast/materialization 与 AdaptiveFFN launch 进入 I-08/I-10；已确认 stack→sum 见 Patch C。

```python
# pure_transformer.py:456-465
orig_x = x
for i, layer in enumerate(self.ffn):
    # print('infeature debug: ', 'idx: ', i, 'x: ', x.shape)
    x_adapt = self._call_tower(x, self.adaptive_towers[i])
    x = self.act_layer(x, x_adapt, i)
    x = layer(x)
    if i < len(self.ffn) - 1:
        x = self.act(x)
        # summary_zero_fraction(x, "affn_main_tower")
    x = self.dropout_layer(x)
```

#### 3.3.3 `Pattern16` 与 `Pattern15`：序列多尺度聚合和 bias 切分

- **排序依据**：SequenceModule 内 Pattern16 有 45 个实例，Pattern15 有 26 个实例。
- **输入/输出**：输入 dense sequence `[B,L,D]`、位置编码和可选 bias 尾维；输出多尺度 `[B,D]` sum 列表、去 bias 的 sequence/PE，以及独立 bias 字典。
- **成员节点/子 Pattern**：`slice→sum` 多尺度分支；sequence 与 PE 的尾维 slice；serving 时可含 `repeat_interleave`。
- **对应 Module class**：SequenceModule。
- **可能的 Kernel 实现关系**：slice/view、reduction、copy 与 repeat；具体实现未被 Timeline 绑定到这些 leaf。
- **timeline 具体 Kernel 信息**：无。
- **候选证据分类**：多尺度 reduction 中已确认的 stack→sum 调用进入 Patch C；其他复用仍因 producer/consumer 和数值契约未闭合而进入 I-10。

```python
# main_model.py:1573-1581
merge_content = [2, 4, 8, 16, 32, 64]
start = 0
for i, mc in enumerate(merge_content):
    if mc <= cur_seq_max_len:
        xd_tensor_for_mix_list.append(
            torch.sum(xd_tensor[:, start : (start + mc), :], dim=1)
        )
    else:
        xd_tensor_for_mix_list.append(torch.sum(xd_tensor, dim=1))
```

#### 3.3.4 `dense_query`、`Pattern24` 与 `Pattern26`

- **排序依据**：C2kTrans 内 dense_query 有 6 个实例，Pattern24 与 Pattern26 各 2 个实例。
- **输入/输出**：场景字段生成 bool mask并选择 query；dense query 接收 query 与 packed SequenceTensor，输出 soft-flow 和长度信息；Pattern26 从 segment 边界生成每段实例数。
- **成员节点/子 Pattern**：`eq→logical_or→where`；`prepare_packed_dense_sequence→dense_query→reshape→where`；`pad→slice/sub→float→setdefault`。
- **对应 Module class**：C2kTrans / TransBlock / SeqTrans。
- **可能的 Kernel 实现关系**：FlashAttention 或 dense attention、mask elementwise、reshape/copy 与长度元数据算子；这只是代码链关系。
- **timeline 具体 Kernel 信息**：无。
- **候选证据分类**：packed sequence 和长度元数据继续进入 I-01；已确认 query stack→sum 进入 Patch C，其他 query 投影没有伪造收益。

```python
# main_model.py:1094-1100
else:
    if prepared_sequence is None:
        prepared_sequence = self.prepare_packed_dense_sequence(joiner_backfill_mask)
    transf_soft_flow_2k, length_info_2k = self.seq_trans.dense_query(
        query=trans_query,
        sequence=prepared_sequence
    )
```

#### 3.3.5 `matrix_similarity`：保留的结构证据（非 timing 排名）

- **实际结构**：slice 用户/广告向量→广告向量转置→矩阵乘→×15→row softmax→`-log`→diagonal→sum。
- **输入/输出**：最多 32 条 `[N,D]` embedding，形成 `[N,N]` logits，输出 `[N]` 对角负对数及 scalar loss。
- **成员节点/子 Pattern**：父 Pattern6 包含 matrix_similarity function group；本项不因旧的局部 fact 数据进入公开排序。
- **对应 Module class**：CVRModule 的 in-batch match 分支。
- **可能的 Kernel 实现关系**：GEMM、softmax、log、diagonal/reduction。
- **timeline 具体 Kernel 信息**：无。
- **候选证据分类**：N≤32 小矩阵链的融合与数值稳定性仅进入 I-14；Pattern 无 timing，不估算收益。

```python
# model_common.py:1418-1423
def matrix_similarity(ad_vec, user_vec):
    matrix_logits = torch.matmul(user_vec, ad_vec.t()) * 15
    matrix_softmax = F.softmax(matrix_logits, dim=1)
    matrix_softmax_log = -1 * torch.log(matrix_softmax)
    logits = torch.diagonal(matrix_softmax_log)
    return logits
```

#### 3.3.6 `Pattern4`：同形 embedding / tower 输出的堆叠求和

- **排序依据与所属 Module**：按第 3.1 节的两级规则，`Pattern4` 归入热点祖先 `CVRModule`（Module #1，仅作排序上下文），共 **32 个真实 DAG 实例**，因此位列该 Module 内第 2；`CVRModule` 的输入、NFM/embedding 聚合行为和输出见[第 2.2.1 节](#221-cvrmodule)。其更近的实现宿主分布为 NFMModule 11 个、NoShareMultiSlotsEmbedding 17 个、EAModule 4 个。
- **真实实例与源码分布**：group ID 为 33293–33303、33732–33743（其中按 DAG 顺序交错编号）、33759、33765、33771、33777、33845、33847、33848、33850、33851；按表达式位置计数为 `main_model.py:673` 11 个、`model_common.py:409` 15 个、`model_common.py:427` 2 个、`model_common.py:1131` 4 个，合计 32。
- **实际结构、成员节点/子 Pattern**：每个实例都恰含两个直接成员节点 `stack.default → sum.dim_IntList`，即总计 32 个 `stack.default` 和 32 个 `sum.dim_IntList`；32 个实例的 `children_group_ids` 均为空。源码附近的 `cat` 是相邻输出分支，不是 `Pattern4` 的成员节点。
- **输入**：一组 shape 相同的 slot embedding、NFM 分组 embedding 或 EA tower 输出列表。
- **算子组合行为与输出**：先沿新轴 `stack` 列表，再沿该轴 `sum(dim=0)`，输出与单个输入 tensor 同 shape 的聚合表示；该表示继续进入 NFM 乘法/塔网络、embedding 返回值或 EA task split。
- **可能的 Kernel 实现关系**：可能实现为 stack 对应的拷贝/拼接类 kernel 加 reduction kernel，也可能被编译器与相邻生产者融合；这是源码算子到 kernel 类型的可能关系，**不是 Timeline 归因**。
- **timeline 具体 Kernel 信息**：无。
- **候选证据分类**：三个已确认 `stack→sum` 位置进入第 8.1 节 Patch C，并以 E1 阈值和局部 kernel/显存采集验收；其余生产者融合仍须 I-10 闭合 graph/stream/alias。32 个结构实例及 Module 时间不用于收益。

```python
# main_model.py:672-674
nfm_network_embeddings[user_layer_name] = torch.sum(
    torch.stack(nfm_network_embeddings[user_layer_name]), dim=0
)
```

```python
# model_common.py:406-409
if self.merge_mode == "concat":
    return torch.cat(slot_embs, dim=-1)
elif self.merge_mode == "sum":
    return torch.sum(torch.stack(slot_embs), dim=0)
```

```python
# model_common.py:1128-1131
    task_output_t = self.ea_towers_list[idx](reslist[idx])
    task_output_list.append(task_output_t)
# task_output = torch.add(task_output_list)
task_output = torch.sum(torch.stack(task_output_list), dim=0)
```

#### 3.3.7 `cal_loss`：采样校正、有效样本掩码与多任务二元交叉熵

- **排序依据与所属 Module**：`cal_loss` 归入热点祖先 `CVRModule`（Module #1，仅作排序上下文），有 **29 个真实 DAG 实例**，位列该 Module 内第 3；它消费各任务 tower 写入 `layers_map` 的 logits，完整 Module 行为见[第 2.2.1 节](#221-cvrmodule)。
- **真实实例与源码分布**：group ID 为 33855–33858、33864–33888；调用点由 `main_model.py:2556-2560` 各 1 个、`main_model.py:2562-2566` 各 4 个及 `qcpx_common.py:290` 4 个构成，合计 29。这里的实例数来自 DAG synthetic groups，不是调用耗时或 Step 次数估计。
- **实际结构、成员节点/子 Pattern**：29 个实例的 `children_group_ids` 均为空。直接成员节点共有 8 种分支序列，公共主链覆盖 `zeros_like/ones_like`、可选 `clamp/logical_or/view`、`exp`、`where`、`div/add/sub/mul/reciprocal`、`clone`、两项 `log`、mask `where`、`sum`；DAG 计数中每个实例均各含 1 个 `div.Tensor`、`exp.default`、`reciprocal.default`、`clone.default`、`sum.default`，并因任务采样分支和可选 `loss_weight` 产生其余节点差异。
- **输入**：`is_training`、任务 label、`output_name`、有效样本 `ins_mask`、可选 `loss_weight/postfix_name`，以及 `layers_map` 中的 logits、全局采样率/偏置和任务正样本采样率。
- **算子组合行为与输出**：按任务分支校正采样概率，计算校正后的 `y_pred`；用 `ins_mask` 生成安全 label 和 masked binary cross-entropy，可选乘权重，最后 `sum` 并乘 `BS_LOSS_SCALE`，写出 `<task>_oracle_pred`、`<task>_serving_pred` 与 scalar `<task>_oracle_loss`。
- **可能的 Kernel 实现关系**：可能涉及 clamp、exp/log、逐元素算术与 select/mask、最终 reduction；编译后可能形成融合 elementwise kernel 加 reduction kernel。该关系仅由算子结构推断，**不是 Timeline 归因**。
- **timeline 具体 Kernel 信息**：无。
- **候选证据分类**：采样校正复用、mask/逐元素融合均缺跨任务输入 identity、Backward 与局部 timing，降级为调查；29 个实例数和 CVRModule inclusive 时间不用于收益。

```python
# main_model.py:2558-2566
cal_loss(is_training, xd_label, "xd", torch.ne(xd_label, invalid_label))
cal_loss(is_training, pay_label, "pay",torch.ne(pay_label, invalid_label),ea_loss_weight_169,)
cal_loss(is_training, lt_pay_label, "lt_pay", torch.ne(lt_pay_label, invalid_label),ea_loss_weight_412,)
for idx in range(layers_map['new_task_num']):
    cal_loss(is_training, label, f"new{idx}_shopping", torch.ne(label, invalid_label),ea_loss_weight_96,postfix_name="shopping")
    cal_loss(is_training, pclk_label, f"new{idx}_pclk", torch.ne(pclk_label, invalid_label), postfix_name="pclk")
    cal_loss(is_training, xd_label, f"new{idx}_xd", torch.ne(xd_label, invalid_label),postfix_name="xd")
    cal_loss(is_training, pay_label, f"new{idx}_pay",torch.ne(pay_label, invalid_label),ea_loss_weight_169,postfix_name="pay")
    cal_loss(is_training, lt_pay_label, f"new{idx}_lt_pay", torch.ne(lt_pay_label, invalid_label),ea_loss_weight_412,postfix_name="lt_pay")
```

```python
# model_common.py:1362-1376
    epsilon = 1e-7
    safe_label = torch.where(ins_mask, label, torch.zeros_like(label))
    cross_entropy = -safe_label * torch.log(y_pred + epsilon) - (1 - safe_label) * torch.log(
        1 - y_pred + epsilon
    )
    cross_entropy = torch.where(
            ins_mask, cross_entropy, torch.zeros_like(cross_entropy)
    )
    if loss_weight is not None:
        cross_entropy = cross_entropy * loss_weight
    return cross_entropy
cross_entropy = scope(postfix_name=postfix_name)
#cross_entropy = cross_entropy[ins_mask]
oracle_loss = torch.sum(cross_entropy)
layers_map[output_name + "_oracle_loss"] = oracle_loss * BS_LOSS_SCALE
```

#### 3.3.8 `Pattern37`：MLPMixer head/token 维度重排

- **排序依据与所属 Module**：`Pattern37` 归入热点祖先 `ResidualAttentionBlock`（Module #4，仅作排序上下文），共 **20 个真实 DAG 实例**，位列该 Module 内第 3；每个 10 层 block 的 MLPMixer 各出现输入、输出两处，block 的残差/MoE 正常行为见[第 2.2.4 节](#224-residualattentionblock)。
- **真实实例与源码分布**：group ID 为 33609、33610、33620、33621、33631、33632、33642、33643、33653、33654、33664、33665、33675、33676、33686、33687、33697、33698、33708、33709；其中 `pure_transformer.py:863` 10 个，`pure_transformer.py:898` 10 个。
- **实际结构、成员节点/子 Pattern**：863 行的 10 个实例各含子 Pattern `separate_heads`（成员 `view.default → transpose.int`）及直接成员 `clone.default → _unsafe_view.default`；898 行的 10 个实例无子 Pattern，直接成员为 `view.default → transpose.int → clone.default → _unsafe_view.default`。合计 20 个 clone、20 个 `_unsafe_view`、20 个 view 和 20 个 transpose（后两类各有 10 个位于子 Pattern）。
- **输入**：逻辑 `[B,T,D]` 的 mixer 输入，或 MLP 后 `[B,H,T*D']` 表示；维度参数为 `token_num`、`num_heads`、`head_dim/new_dim`。
- **算子组合行为与输出**：入口 reshape 为 `[B,T,H,D']` 并 transpose 为 `[B,H,T,D']`，中间可展平到 `[B,H,T*D']` 进入 MLP；出口再 reshape、transpose 并 reshape 回 `[B,T,D]`，送回 ResidualAttentionBlock 的残差与归一化路径。
- **可能的 Kernel 实现关系**：view/unsafe_view 通常仅改元数据；transpose 后为满足后续 reshape/布局可能触发 clone/contiguous copy，也可能被后续 GEMM 的布局处理吸收。该关系是实现机制说明，**不是 Timeline 归因**。
- **timeline 具体 Kernel 信息**：无。
- **候选证据分类**：transpose 后 pack 不能直接删除；layout-aware consumer 和 fast T-shard↔H-shard 复用均缺完整生产实现及 Forward/Backward placement 证明，降级为 I-02。

```python
# pure_transformer.py:862-866
def separate_heads(self, x):
    x = torch.reshape(
        x, (-1, self.token_num, self.num_heads, self.head_dim)
    )
    return x.transpose(1, 2)
```

```python
# pure_transformer.py:896-900
else:
    # (B, H, T * D') -> (B, H, T, D') -> (B, T, H, D') -> (B, T, D)
    x = torch.reshape(x, [x.shape[0], self.num_heads, self.token_num, self.new_dim // self.token_num])
    x = x.transpose(1, 2)
    x = torch.reshape(x, [x.shape[0], self.token_num, self.token_dim])
```

---

## 4. Kernel 热点与实现分析

### 4.1 Kernel Top20

| # | 短语义名 | 角色 | calls/step | 平均单次 (μs) | ms/step | 占 Step |
|---:|---|---|---:|---:|---:|---:|
| 1 | MoE BF16 Grouped GEMM 64×64×64 Fwd | Fwd | 56 | 1002 | 56.128 | 4.708% |
| 2 | MoE BF16 Grouped GEMM 64×64×32 Bwd | Bwd | 53 | 1036 | 54.891 | 4.604% |
| 3 | MoE BF16 Grouped GEMM 64×128×32 Fwd | Fwd | 54 | 990 | 53.478 | 4.485% |
| 4 | Fused Adamom 多张量更新 | Optimizer | 121 | 428 | 51.760 | 4.341% |
| 5 | MoE BF16 Grouped GEMM 64×128×32 Bwd | Bwd | 45 | 990 | 44.571 | 3.738% |
| 6 | ATen BF16 直接拷贝 | Layout | 144 | 244 | 35.143 | 2.947% |
| 7 | 残差加+LayerNorm反向归约（大变体） | Bwd | 10 | 2448 | 24.477 | 2.053% |
| 8 | FlashAttention K/V 反向 | Bwd | 14 | 1707 | 23.897 | 2.004% |
| 9 | FlashAttention 前向 | Fwd | 27 | 604 | 16.319 | 1.369% |
| 10 | 残差加+LayerNorm反向归约（变体B） | Bwd | 10 | 1521 | 15.215 | 1.276% |
| 11 | MoE BF16 Grouped GEMM 64×64×64 Bwd | Bwd | 15 | 1001 | 15.009 | 1.259% |
| 12 | 残差加+LayerNorm反向归约（变体C） | Bwd | 10 | 1499 | 14.986 | 1.257% |
| 13 | FlashAttention Q 反向 | Bwd | 14 | 1033 | 14.462 | 1.213% |
| 14 | NVJET BF16 GEMM 128×256 NNN | Fwd/Bwd GEMM | 5 | 2782 | 13.911 | 1.167% |
| 15 | MoE gather/combine 反向 | Bwd | 20 | 618 | 12.366 | 1.037% |
| 16 | NVJET BF16 GEMM 128×256 TNN | Fwd/Bwd GEMM | 10.8 | 992 | 10.718 | 0.899% |
| 17 | MoE BF16 Grouped GEMM 64×64×32 Fwd | Fwd | 10 | 1003 | 10.034 | 0.842% |
| 18 | MoE gather/combine 前向 | Fwd | 40 | 234 | 9.352 | 0.784% |
| 19 | ATen FP32→BF16 向量化转换 | Parameter prep | 454 | 20 | 8.974 | 0.753% |
| 20 | NVJET BF16 GEMM 320×128 NTT | Fwd/Bwd GEMM | 1 | 7526 | 7.526 | 0.631% |

> calls/step 均为 5 个 Step 的事件数平均，因此 #16 的 10.8 表示五步合计 54 次，并非单次 Step 出现非整数调用。

### 4.2 分组实现分析（20/20 已有源码解法；#4 由 Patch E 覆盖实验）

#### 4.2.1 MoE Grouped GEMM（#1/#2/#3/#5/#11/#17）

- **Module 计算行为**：见第 2.2.4、2.2.7 节。`SparseMoe` 对 `[B,T,4096]` 做 top-2 路由；`expert_token_cnt` 定义每个 expert 的变长 M，scatter 后 `PerTokenSwiGLUFFN` 依次执行 fc1/fc2 4096→512、SiLU×门控、fc3 512→4096，最后 gather 回 `[B,T,4096]`。
- **Grouped GEMM/problem array**：16 个 expert 的指针、stride 和 `(M_e,N,K)` 组成 problem array；`M_e` 由 `expert_token_cnt` 决定。一次 pointer-array launch 处理多个不等长专家，避免逐 expert 发射小 GEMM。
- **SM/TMA/warp**：SM90 TMA load/store；warp-specialized ping-pong mainloop；384-thread CTA。WGMMA atom 分 64×64×16 与 64×128×16。
- **dtype/layout/epilogue**：A/B 为 BF16、FP32 accumulator、BF16 LinearCombination 输出；A/B Major 的 0/1 组合与正反向转置方向匹配。64×64×64 变体 stage=13，64×64×32 stage=26，64×128×32 stage=17；epilogue 用 pointer-array TMA 写回。
- **shape**：前向观测 `[32768,4096] × [16,4096,512] → [32768,512]`，以及 `[32768,512] × [16,512,4096] → [32768,4096]`；反向生成 dInput/dWeight。6 变体合计 234.111 ms/step。
- **热点原因**：每层三次专家 GEMM、10 层重复、top-2 使 routed token 扩为 32768；expert 不均衡时小 `M_e` 会降低 tile 利用率。

```python
# sparse_moe.py:238-244
def forward(self, x, expert_token_cnt=None):
    x1_out = self.fc1(x, expert_token_cnt)
    x1_out = self.act(self.dropout(x1_out))
    x2_out = self.fc2(x, expert_token_cnt)
    x_out = x1_out * x2_out
    x_out = self.fc3(x_out, expert_token_cnt)
    return self.dropout(x_out)
```

```python
# sparse_moe.py:368-381
top_k_probs, top_k_indices = self.router(x)
if top_k_indices.dtype != torch.int32:
    top_k_indices = top_k_indices.int()

scatter_index, expert_token_cnt = self.moe_fused_index_compute(top_k_indices, local_expert_num)
x_flat = x.reshape(-1, self.token_dim)
x_scatter = self.moe_scatter(input_tensor=x_flat, scatter_index=scatter_index)
x_ffn = self.ffn(x_scatter, expert_token_cnt)
output_flat = self.moe_gather(
    input_tensor=x_ffn.to(x.dtype),
    scatter_index=scatter_index,
    gather_weight=top_k_probs.to(x.dtype),
)
output = output_flat.reshape(-1, local_token_num, self.hidden_dim)
```

#### 4.2.2 Fused Adamom（#4：P1/E1-E 可执行实验；生产入口仍待 I-09 归因）

- **已确认**：该条位于 optimizer 阶段，121 次/step、51.760 ms/step；正式 Kernel 名表明它是 Adamom 多张量更新。
- **局部待补证据**：当前证据未提供真实 Adamom 更新公式、参数/梯度/状态张量清单和训练源码入口。因此本报告不推断状态数量、融合公式或具体内存访问次数。
- **关闭条件**：补充 optimizer 构造与 `step` 调用入口、CUDA/ATen launcher，以及逐项更新公式和 dtype 后，才能完成 optimizer kernel 的 class/source-level 分析。
- **当前处置**：P1/E1-E 已提供严格同配置参数组合并 Harness、state 迁移和一步 update A/B；收益不预承诺，上限仅取实测消失的 #4 调用/gap。I-09 只补生产接入位置，不否定现有可执行 Patch；Top20 源码解法覆盖计 20/20。

#### 4.2.3 残差加+LayerNorm 反向（#7/#10/#12）

- **Module**：ResidualAttentionBlock，见第 2.2.4 节；Kernel 位于 `ln_1(x+mixer_out)`、`ln_2(x+mlp_out)` backward。
- **实现**：Inductor/Triton 把 residual add 与 native LayerNorm backward 融合；grid 沿 4096 归一化维/行展开，读取 BF16 激活、上游梯度、mean/rstd、gamma/beta，输出 dInput/dResidual 与 dgamma/dbeta。
- **shape/dtype**：代表 shape `[16384,1,4096]`、权重 `[4096]`，BF16 数据配 FP32 mean/rstd；每变体 10 次/step。三项合计 54.677 ms。
- **热点原因**：大行数×4096 的归约、参数梯度归约同步，以及 checkpoint 重算后的反向工作量。

#### 4.2.4 FlashAttention（#8/#9/#13）

- **Module**：`trans_refactor.Attention`，见第 2.2.7 节；Q/K/V Dense 后、out Dense 前。
- **前向**：Triton 分块加载 BF16 Q/K/V，在线维护 softmax max/sum 并累加 O，不物化完整 N×N score。
- **反向**：dK/dV kernel 以 Q 和 dO 归约 K/V；dQ kernel 以 K/V 和 dO 归约 Q。输入常见 `[3310–5231,8,64]`，长序列还观测到 34353/47630 tokens；输出与 Q/K/V 同 shape 梯度。
- **成本**：前向 16.319、dK/V 23.897、dQ 14.462，合计 54.679 ms/step。
- **热点原因**：packed 长度差异、8 heads×64 head_dim、多序列边界，以及 backward 重算概率块。

```python
# trans_refactor.py:94-103
query = self.query_dense(query)
key = self.key_dense(inputs)
value = self.value_dense(inputs)
if self.config.use_qk_norm:
    query = self.q_norm(query)
    key = self.k_norm(key)
query = torch.reshape(query, [-1, self.config.num_attention_heads, self.config.head_dim])
key = torch.reshape(key, [-1, self.config.num_attention_heads, self.config.head_dim])
value = torch.reshape(value, [-1, self.config.num_attention_heads, self.config.head_dim])
return query, key, value
```

```python
# trans_refactor.py:133-137
attn_output = self.attention(query, key, value, self.attn_scale, mask_fn)
attn_output = torch.reshape(attn_output, [-1, self.config.hidden_size])
# attn_output = attn_output.apply(self.out_dense)
attn_output = self.out_dense(attn_output)
return attn_output, kvcache_out
```

#### 4.2.5 BF16 copy/cast（#6/#19）

- **Module/wrapper**：Transformer 的 DTensor/FSDP 包装。#6 在 shard-layout 通信前后把非连续 BF16 tensor contiguous/clone；#19 在 block backward 前把 FP32 参数 shard 转为 BF16。
- **实现**：#6 为 TensorIterator 128 threads×4 elements 的 BF16 直接复制；代表 `[64,16384,1,64]` 与 `[16384,1,64,64]`。#19 vector width=8，输入 FP32、输出同 shape BF16；常见 `[33554432]`、`[512,512]`、`[4096]`。
- **成本**：35.143 + 8.974 = 44.117 ms/step；#19 为 454 次/step、均值 19.768 μs/次，是 Top20 中唯一均值 `<50 μs` 的 kernel，处置见第 11 章。
- **证据边界与处置**：#6/#19 的符号证明 copy/cast 语义，累计成本为 35.143 + 8.974 = 44.117 ms/step；但 trace 未提供 bytes、算术强度或带宽利用率，故不能断言“纯 HBM 流量”。与 collective pack 融合、与首个消费者融合分别进入 I-12、I-08；optimizer 参数组进入 I-09，不在此处给无归属建议。

#### 4.2.6 NVJET BF16 GEMM（#14/#16/#20）

- **Module 候选类**：Dense、BatchMatMulDense、GatedFFN、Attention 投影；类行为见第 2.2.7 节。实例区分范围用于 A/B，不影响算子机制结论。
- **实现**：NVJET 模板编码 CTA/warp/流水与 NNN/TNN/NTT 转置方向；#14/#16 使用 128×256 主 tile，#20 使用 320×128，384-thread block。
- **shape**：#14 观测 `[24576,512]×[512,512]`、`[1,16384,2048]×[1,2048,8192]`；#16 还含 `[34353,512]×[512,512]`；#20 是 `[1,8192,16384]×[1,16384,4096]`。
- **I/O/dtype**：BF16 A/B，输出 BF16；反向转置变体产生 dInput/dWeight。三项合计 32.155 ms/step。
- **热点证据与处置**：#20 单次 7.526 ms，对应超大 K=16384；#14/#16 具有长 token 维与 8192 输出维。维度对齐、前后 copy 和 shape autotune 仅进入 I-08/I-14，未形成正式建议。

#### 4.2.7 MoE gather/combine（#15/#18）

- **Module**：SparseMoe，见第 2.2.7 节；位于 expert FFN 之后。
- **前向**：读取 `[32768,4096]` expert 输出和 `[16384,2]` index/权重，回排、乘 top-2 权重并规约为 `[16384,4096]`。
- **反向**：读取 token 梯度与 expert 输出，生成 `[32768,4096]` expert 梯度并累积 router-weight 梯度。
- **成本**：前向 9.352、反向 12.366，共 21.718 ms/step。
- **热点证据与处置**：实现含 4096 宽向量非连续索引、top-2 扩张与权重规约；是否由索引局部性主导尚无 bytes/cache 证据。expert 排序和 GEMM 输出布局由 P1/E1-G 直接采集并证伪。

### 4.3 Kernel Top20 逐组“源码解法”矩阵（20/20）

> 证据等级定义：A=trace stack/call-chain 直接映射；B=kernel 语义、shape、时间邻接与 Module 的高置信推导；C=待运行实验证伪。每组都给实际入口；证据不足仅令收益下限为 0，不取消 Patch。

| Top20 | 原因→归属/推导链 | 源码解法（就地编号） | 契约与验证 | 证据 |
|---|---|---|---|---|
| #1/#2/#3/#5/#11/#17 | grouped GEMM name + problem array → `sparse_moe.py:SparseMoe.forward` L354-L389 → `PerTokenSwiGLUFFN.forward` L238-L244 | **P1/E1-G**：在 `self.ffn(x_scatter, expert_token_cnt)` 调用点增加只采集、不改路由的 persistent grouped-GEMM runner；把 `expert_token_cnt`、每 expert `(M,N,K)`、指针对齐、tile 选择和输出地址传给单一实验入口 | 输入/输出 shape、BF16 dtype、expert 顺序、top-2 权重、Fwd/Bwd 必须不变；逐 expert 与最终输出 `max_abs≤2e-2,max_rel≤2e-2`，梯度同阈值；比较六项 234.111 ms/step 与 p50/p95 | B→运行证伪 |
| #4 | optimizer 阶段 + fused Adamom 名 → optimizer runtime | **P1/E1-E**：同配置/dtype/device/placement 参数组严格合并 harness，并采集每组 tensor 数、foreach/fused 路径和 121 launches；公式未闭合时 runner 立即报错，不猜测更新式 | 一步后 param 与全部 optimizer state bitwise；launch 121→目标值、#4 51.760 ms/step；任一 state 缺失即回滚 | B/C |
| #7/#10/#12 | fused residual+LN backward → `ResidualAttentionBlock.forward` 的 `ln_1/ln_2` | **P1/E1-H**：保留 `x + branch` 数学式，提供自定义 autograd 实验 op，把 residual add、LN forward/backward 作为一个显式接口；先用 `torch.library.opcheck` 和 gradcheck 证伪，再接 Inductor | `[tokens,4096]`、BF16 激活/FP32 stats、gamma/beta 参数身份及梯度不变；输出/梯度 `2e-2`，比较 54.677 ms/step；不能证明 dgamma/dbeta 等价则禁用 | B/C |
| #8/#9/#13 | FlashAttention shape + Q/K/V 邻接 → `trans_refactor.py:Attention.forward` L94-L137 | **P1/E1-I**：按 packed-length bucket 选择固定 `(BLOCK_M,BLOCK_N,num_warps)` 的实验 dispatch table；不改 Q/K/V、mask_fn 或 softmax，仅替换 kernel config 选择 | 输出/三路梯度 `2e-2`，相同 RNG；按 length bucket 比较合计 54.679 ms/step 与 autotune 命中；任一 bucket 回退变慢 >2% 回滚该项 | A/B |
| #6 | BF16 direct copy + reshard 邻接 → DTensor/FSDP layout pack | **P1/E1-F**：以 `layout_pack_bf16(input, target_placement)` 实验 custom op 替换“transpose/redistribute 后 clone”；producer 直接写目标 contiguous stride，并由 backward 逆 placement | 逻辑 shape/dtype/placement、stride 契约显式；输出 bitwise、梯度 `2e-2`；比较 #6 35.143 ms/step、copy 数及 collective 前后 gap | B/C |
| #19 | FP32→BF16、454 次且在 block backward 前 → 参数 shard materialization | **P1/E1-J**：实现按 `Parameter._version`、device、shape、stride 键控的 BF16 cast cache；optimizer step 后 version 变化必须失效，禁止陈旧复用 | cast 输出 bitwise；更新后 cache miss，未更新参数 cache hit；比较 454 calls/step 与 8.974 ms/step；版本不可靠即硬失败 | B/C |
| #14/#16/#20 | NVJET dtype/shape/transpose code → Dense/BatchMatMulDense/GatedFFN 候选 | **P1/E1-F** 同时记录首个 GEMM consumer 的 layout 接受能力，A/B 运行“producer 目标布局直写”与原 pack；每个 shape 独立准入，禁止跨 shape 承诺 | GEMM 输出/梯度 `2e-2`；比较三项 32.155 ms/step、pack 数及端到端 step；shape 未唯一映射则实验日志必须输出调用栈后终止该 shape | B/C |
| #15/#18 | gather/combine name + top-2 tensor shape → `SparseMoe.forward` L368-L381 | **P1/E1-G**：让 grouped GEMM epilogue 按 `scatter_index` 直接写 combine 所需顺序；实验 autograd 同时保存反向 inverse index，禁止只改 forward | expert 顺序、top-k 权重、router 梯度、placement 不变；输出/梯度 `2e-2`；比较 21.718 ms/step 与额外 workspace | A/B |

#### F/G 处置摘要（完整条目统一收口到第 8 章）

- **F**：`MLPMixer.separate_heads → self.attn / reshape → self.ln → self.mlp` 是当前扫描源码能闭合的首个 consumer 链；但 trace 的 #6/#14/#16/#20 尚未唯一绑定到该链，因此 F 降为**实验性候选 Patch**。完整 adapter、运行 harness、模型接入 unified diff 与契约见 **第 8.4 节**。
- **G**：MoE grouped GEMM 与 combine 的热点及契约摘要保留在上表；完整实验条目见 **第 8.5 节**。逐项 Diff 审计未通过前不得称正式 Patch。

---

## 5. 通信与计算重叠分析

### 5.1 通信类型合并

| 类型 | 语义用途 | 并行维度 | 次数/step | Union (ms) | 暴露 (ms) | Overlap ratio |
|---|---|---|---:|---:|---:|---:|
| AllToAll | batch/token/hidden shard 布局互换 | TP=64 | 66.6 | 284.839 | 280.899 | 1.383% |
| AllGather | 收集分片参数或张量 | TP/框架分片 | 4.0 | 87.336 | 82.483 | 5.556% |
| AllReduce | 多 rank 求和并复制结果 | TP/world collective | 80.4 | 17.268 | 14.680 | 14.986% |
| ReduceScatter | 求和后按 rank 分片 | TP/框架分片 | 12.6 | 12.276 | 8.901 | 27.491% |

四类 union 合计 401.719 ms，比总通信 union 少 6.131 ms；差值来自未被类型 annotation 完整覆盖的通信 kernel及集合边界，不分摊。小数次数来自跨 Step 边界裁剪后的 5-Step 均值。

### 5.2 通信代码行为

#### 5.2.1 AllToAll：TP shard 布局转换

- **触发链**：CvrTransformer/10 个 ResidualAttentionBlock 的 DTensor reshard；`Shard(0)↔Shard(1)` 在 batch 与 token/hidden 布局之间转换。
- **I/O**：逻辑 `[B,S,D]` 不变，物理上把本 rank 的 batch shard 发给其他 rank，并接收目标 token/hidden shard。
- **依赖**：下游 LayerNorm/mixer 必须等待目标 shard 可见；backward 的 `BwRedistribute` 也必须完成后才能返回上游梯度。
- **成本**：66.6 次、284.839 ms union，仅 3.940 ms 被计算覆盖。

```python
# sharding_plan.py:102-110
combine_model_fwd_resharding_plan = {
    "input": [[Shard(1)]],
    **{
        rf"resblocks.\d+.{k}": v
        for k, v in token_mixer_sharding_plan["forward"].items()
    },
    # res_out_fqn: [[Shard(0)]],
    "output":  [[Shard(0)], None],
}
```

#### 5.2.2 AllGather：分片收集

- **触发链**：观测到 coalesced 与 base 两类；可确定是分片张量 materialization，具体用于 DTensor 参数或其他框架张量的实例作为 A/B 范围。
- **I/O**：每 rank shard → 各 rank 完整拼接 tensor；消费者通常是 GEMM/反向计算。
- **成本与处置**：4 次、87.336 ms union，82.483 ms 暴露；bytes/source 拆分与预取不是正式建议，分别进入 I-07/I-12，单调用上限须在唯一归因后计算。

#### 5.2.3 AllReduce 与 ReduceScatter

- **AllReduce**：对梯度/统计量求和后每 rank 保留完整结果；80.4 次、17.268 ms，粒度较碎。
- **ReduceScatter**：求和后只保留目标 shard；12.6 次、12.276 ms，overlap 27.491%，是四类中最好。
- **证据边界与处置**：二者 overlap 数值已测，但 bucket 调整的调用点、参数组和依赖边未知，进入 I-07/I-09；14.680 与 8.901 ms 只是通信类型 exposed 总量，不是单 patch 收益。

### 5.3 Overlap 关键结论

- **总体 overlap ratio**：3.644%；通信仅 14.755 ms 与 compute 重叠。
- **主要暴露阶段**：AllToAll 暴露 280.899 ms，贡献总暴露通信的 71.46%；AllGather 暴露 82.483 ms。
- **根因**：布局转换处于 producer/consumer 边；MoE 为 router/index→dispatch→expert compute→combine 串行；symmetric-memory 路径本身也含 barrier/clone 可见性约束。
- **数学上限**：完美隐藏全部通信最多 393.094 ms/step，但工程预期必须扣除依赖边、资源竞争与新增计算。

### 5.4 通信热点就地源码解法

| 热点 | 原因→源码链 | 可运行实验 | backward/placement 契约 | 指标与回滚 |
|---|---|---|---|---|
| AllToAll 66.6 次 | `sharding_plan.py` L102-L110 的 `Shard(1)→Shard(0)`，producer/consumer 边导致 280.899 ms 暴露 | **P1/E1-K**：`PackAllToAllFn` 将 producer 的目标-rank pack 与 `all_to_all_single` 放入同一 custom autograd 接口；forward 返回目标 placement，backward 执行逆 split 的 AllToAll 并 unpack 到输入 stride | split sizes、rank 顺序、process group、logical shape 必须显式；禁止 forward-only、禁止 identity backward、禁止 fallback | 与 eager 输出 bitwise、梯度 `2e-2`；比较 pack kernel 数、AllToAll 284.839 ms 与 exposed 280.899 ms；placement 不一致立即回滚 |
| AllGather 4 次 | shard materialization→GEMM consumer，82.483 ms 暴露 | **P1/E1-L**：双 buffer `all_gather_into_tensor(async_op=True)`；在独立 comm stream 预取下一层 shard，consumer stream 用 event 精确等待；参数 `_version` 变化时 buffer 失效 | 不改变参数对象/optimizer state；每层 buffer 的 shape/dtype/shard offset 固定；禁止跨 iteration 复用已更新参数 | 输出/梯度 `2e-2`；比较 exposed、等待 gap 与额外显存；显存超预算或等待未下降则回滚 |
| AllReduce/ReduceScatter | 梯度/统计通信粒度碎，依赖 bucket ready 顺序 | **P1/E1-E** 联动：仅把配置、dtype、device、placement 完全相同的参数组放入同一 foreach/fused 更新；另由 **P1/E1-M** 记录梯度 ready timestamp 后离线生成 bucket，不在运行时猜分组 | 参数顺序到 state 的映射不变；ReduceScatter 输出 shard offset 不变；一步 param/state bitwise | 比较 80.4/12.6 calls、exposed 14.680/8.901 ms；首个 bucket 延迟恶化或 state 不同即回滚 |

P1/E1-K 的热点摘要与 placement/backward 门禁保留在上表；完整实验条目、成对 autograd 代码与 Diff 审计统一见 **第 8.9 节**。第 5 章不再保留独立完整 Patch 正文。

### 5.5 Perfetto Trace Processor RPC 独立复核（2026-08-06）

本节使用 Perfetto v57.2 Trace Processor HTTP/RPC 共库完成独立复核。Web Query 与 Python `TraceProcessor(addr=...)` 连接同一个数据库实例；`slice → thread_track → thread` 核验到 track 19/34/35/41 分别对应 stream 49/37/41/65。该 trace 没有可直接假定的 `gpu_track` 行，因此所有 stream 结论都来自上述真实 join。

通信主统计只接受 `category='kernel'` 且父 slice 精确属于六个已核验 collective annotation 的事件：`nccl:all_gather_into_tensor_coalesced`、`nccl:_all_gather_base`、`nccl:all_reduce`、`nccl:all_to_all`、`nccl:reduce_scatter_tensor_coalesced`、`nccl:_reduce_scatter_base`。835 个 kernel 被严格分类；另有 17 个名称和 stream 均指向 NCCL 的候选 kernel，总计 32.083 ms/trace，但缺少上述父 annotation，未并入正式通信指标，也未计入 compute。这个边界解释了“精确 kernel 名白名单”口径比正式 parent-annotation 口径多出的部分。

区间计算先由 SQL 提取 `id/ts/dur/track_id/name/parent_name`，再按 `[ts, ts+dur)` 合并 union；通信与 compute 的交集使用两个已排序 union 的双指针求交。禁止把 `SUM(dur)` 当作 union。五个完整 ProfilerStep 的复核结果与第 5.1–5.3 节一致：平均 Step 1192.304 ms，通信 union 407.850 ms，compute union 727.025 ms，重叠 14.755 ms，暴露通信 393.094 ms。五步总 overlap 除以五步总通信得到 3.618%；逐 Step overlap ratio 的算术平均为 3.644%，本报告主指标采用后者。严格 parent annotation 的类型均值为 AllToAll 284.839 ms、AllGather 87.336 ms、AllReduce 17.268 ms、ReduceScatter 12.276 ms；17 个未分型 NCCL 候选贡献剩余约 6.417 ms/step，因此 AllToAll 仍是首要通信瓶颈。

| SQL ID | 完整 kernel 名 | stream | dur (ms) | compute overlap (ms) | exposed (ms) | host launch queue delay (ms) |
|---:|---|---:|---:|---:|---:|---:|
| 1267630 | `ncclDevKernel_AllGather_RING_LL(ncclDevKernelArgsStorage<4096ul>)` | 49 | 55.883 | 0 | 55.883 | 90.804 |
| 1755227 | `ncclDevKernel_SendRecv(ncclDevKernelArgsStorage<4096ul>)` | 37 | 53.019 | 0 | 53.019 | 327.056 |
| 464095 | `ncclDevKernel_AllGather_RING_LL(ncclDevKernelArgsStorage<4096ul>)` | 49 | 52.075 | 0 | 52.075 | 84.195 |
| 2063978 | `ncclDevKernel_AllGather_RING_LL(ncclDevKernelArgsStorage<4096ul>)` | 49 | 50.591 | 0 | 50.591 | 105.223 |
| 3628528 | `ncclDevKernel_AllGather_RING_LL(ncclDevKernelArgsStorage<4096ul>)` | 49 | 43.759 | 0 | 43.759 | 73.560 |

Top 5 均有唯一直接 `flow` 回连到 `cuLaunchKernelEx`，不是时间最近邻猜测。四个 AllGather 的 host chain 闭合到 `c10d::allgather_into_tensor_coalesced_ → _c10d_functional::all_gather_into_tensor → CompiledFunctionBackward`；SQL ID 1755227 的 AllToAll 链闭合到 `distributed_c10d.py:4322 all_to_all_single → bytedance/ndtimeline/all2all_patch.py:72 → embedding_all_to_all.py:55 backward → EmbeddingAlltoAllFuncBackward`。后者已定位到具体 backward 源码，但 trace 没有唯一业务 module 实例名。

SQL ID 1267630 的 args 进一步给出 correlation `1179578`、group size `64`、process group `130`、description `mesh_TensorParallelStrategy`、collective `allgather_into_tensor_coalesced`、输入 `5,242,880` elements、输出 `335,544,320` elements、dtype `Float`。它与另外三个 Top AllGather 均来自 TP=64 的同类收集路径。全局 correlation 唯一性和 pack/consumer wait 的数据依赖尚未闭合；当前只确认直接 flow、父子 slice、host call chain 和时间区间，不把同窗事件或名称相似事件写成因果关系。

---

## 6. 可实施优化总表与强制分类

### 6.1 扫描版本、归因边界与准入结论

- **分析器仓库版本**：`1cdfb0dd6b603cfdf3c28e0aff01e40c0c66a646`。
- **storage 仓库版本**：`052ecd86296fcdd0209626cf854423678aade7bd`。
- **模型源码版本事实**：`testset/extracted/6451993/modelcode/` 在当前 storage 工作区是未跟踪的提取物，不能虚构 Git commit；本次扫描以文件内容 SHA-256 固定：`main_model.py=343262f3...6e6eafa`、`pure_transformer.py=1d683378...e02835e`、`sharding_plan.py=3552cc2d...7a7cc25`、`mixer_mean_redistribute.py=c0accb16...d21ecf`。
- **timeline 归因边界**：Pattern production timing 为 `0/663`，所以 Pattern 只提供结构证据，任何条目都不得给 Pattern ms；Module timing 是 inclusive 且覆盖仅 `15/1144` groups，禁止作为局部 patch 收益。Kernel 与通信类型聚合可描述全局上限，但只有调用点唯一归因后才能成为单一 patch 上限。
- **准入结果**：已形成 2 项 P0/E0 与 12 项 P1/E1（C–N）实验性候选 Patch。性能证据不足只把收益下限约束为 0，不再阻止可运行实验代码；流程统一为“源码不变量 → 实验 Patch → 数值/契约验证 → 性能采集 → 上线判定”。checkpoint 与 top-k/router 仍为 P2/E2 风险实验；其余无法正确编码的候选保留为前置调查。

| 扫描项 | 分类 | 结论 | 证据/原因 |
|---|---|---|---|
| packed sequence 复用 | 前置证据任务 | 当前无 Patch | `SequenceModule.forward` 已在 L1433-L1442 frontload，并在 L1705-L1728 传入两条主路径；无同一 `(data, mask, attn_arg)` 重复 pack 证据 |
| transpose 后 reshape/隐式 pack | P1/E1 | 实验性候选 F | 第 8.4 节给模型接入与 harness unified diff；当前因 placement/热点映射未闭合为候选 |
| checkpoint 策略 | P2/E2 | 风险实验，不是正式 Patch | 改变激活生命周期、重算、RNG 与显存，可能改变训练行为 |
| top-k/router 2→1 | P2/E2 | 风险实验，不是正式 Patch | 改变 expert 选择、容量、输出、梯度与收敛 |
| `BwRedistribute` mean 路径替换 | P1/E1 | 实验性候选 Patch N | 第 8.12 节按 `CvrTransformer.forward` L2189-L2193 的真实签名调用 custom op；输入列表、token_num、Fwd/Bwd placement 不闭合即硬失败 |
| fused mixer mean | P1/E1 | 实验性候选 Patch N | `mixer_mean_redistribute(mixer_all_outputs, token_num=self.num_tokens)` 是唯一实验路径；逐 tensor 梯度、local/global shape 与 placement 按第 8.12 节验收 |
| stack→sum | P1/E1 | 实验性候选 Patch C | 第 8.1 节以严格非空逐项 add 去除 stack 临时物化；浮点累加树变化按 E1 验证 |
| mixer mean 数值与分片语义 | P1/E1 | 实验性候选 Patch N | 第 8.12 节锁定 FP16/BF16 误差阈值、输入列表、`token_num`、local/global shape、mesh 与 Fwd/Bwd placement；任一不一致即回滚 N |
| mask/tile | P0/E0 | 实验性候选 Patch B | 第 7.2 节单次计算视频 mask、接口显式传递并以 broadcast `where` 替代 tile |
| 重复赋值 | P0/E0 | 实验性候选 Patch A | 第 7.1 节删除 L1614 对相同 RHS 的确定性死写，保留一次赋值 |
| AllToAll/AllGather 调度 | P1/E1 | 实验性候选 K/L/M | 第 8.9/8.10/8.11 节统一收口；均以第 8.13 Diff 审计状态为准 |
| 短 elementwise/cast launch | P1/E1 | 实验性候选 Patch J | 第 4.3 节按 Parameter `_version` 做 BF16 cast cache，严格规定 optimizer 后失效；第 11 章回链 |
| optimizer launch 分组 | P1/E1 | 正式独立实验 Harness E | 第 8.3 节对运行时 optimizer 做严格同配置/dtype/device/placement 合组并验证一步 update；生产入口仍须反查 |
| MoE persistent grouped GEMM | P1/E1 | 实验性候选 G | 第 4.3 节显式 problem array、combine order、inverse index 和 Fwd/Bwd 契约，按六个 grouped kernel 与 gather/combine 子集验证 |
| CUDA Graph | P1/E1 | 正式独立实验 Runner D | 第 8.2 节给出 strict capture/replay 生命周期；动态契约变化直接 raise，无 eager fallback |

---

## 7. P0/E0 实验性候选 Patch

> **流程要求**：性能证据不足只限制收益承诺，不阻止输出可验证 Patch。正确顺序是：源码不变量 → 实验 Patch → 数值/契约验证 → FX/kernel/Step 性能采集 → 决定是否上线。

### 7.1 P0/E0-A：删除 `SequenceModule.forward` 的确定性重复赋值

- **问题/证据/归因边界**：`main_model.py:SequenceModule.forward` L1608-L1615 中 L1611 与 L1614 对相同 `joiner_backfill_mask`、`din_2k` 执行相同 `torch.where`，中间只有 `din_2k_for_mix2` 读取，未读取 `layers_map[layer_name]`。Pattern 无 timeline，当前局部收益未知；这不阻止构造 E0 Patch。
- **优先级/等价/风险/范围**：P0/E0；无精度风险；只删除第二次死写和一次确定性 elementwise 计算。
- **版本定位**：提取物 `main_model.py` SHA-256 `343262f3...6e6eafa`；`SequenceModule.forward` L1608-L1617。

**当前完整分支代码**

```python
if fc_name_3d in ("fc_user_ecom_2k_mix_4d", "fc_user_profile_pay_seq_v3_4d"):
    din_2k = layers_map[layer_name]
    din_2k_for_mix1 = layers_map[layer_name + "_for_mix1"]
    layers_map[layer_name] = torch.where(
        joiner_backfill_mask, torch.zeros_like(din_2k), din_2k
    )
    layers_map[layer_name + "_for_mix1"] = torch.where(
        joiner_backfill_mask,
        torch.zeros_like(din_2k_for_mix1),
        din_2k_for_mix1,
    )
    din_2k_for_mix2 = layers_map[layer_name + "_for_mix2"]
    layers_map[layer_name] = torch.where(
        joiner_backfill_mask, torch.zeros_like(din_2k), din_2k
    )
    layers_map[layer_name + "_for_mix2"] = torch.where(
        joiner_backfill_mask,
        torch.zeros_like(din_2k_for_mix2),
        din_2k_for_mix2,
    )
mix_query_list.append(layers_map[layer_name + "_for_mix1"])
mix_query_list.append(layers_map[layer_name + "_for_mix2"])
```

**可直接替换的完整改后分支**

```python
if fc_name_3d in ("fc_user_ecom_2k_mix_4d", "fc_user_profile_pay_seq_v3_4d"):
    din_2k = layers_map[layer_name]
    din_2k_for_mix1 = layers_map[layer_name + "_for_mix1"]
    din_2k_for_mix2 = layers_map[layer_name + "_for_mix2"]
    layers_map[layer_name] = torch.where(
        joiner_backfill_mask, torch.zeros_like(din_2k), din_2k
    )
    layers_map[layer_name + "_for_mix1"] = torch.where(
        joiner_backfill_mask,
        torch.zeros_like(din_2k_for_mix1),
        din_2k_for_mix1,
    )
    layers_map[layer_name + "_for_mix2"] = torch.where(
        joiner_backfill_mask,
        torch.zeros_like(din_2k_for_mix2),
        din_2k_for_mix2,
    )
mix_query_list.append(layers_map[layer_name + "_for_mix1"])
mix_query_list.append(layers_map[layer_name + "_for_mix2"])
```

- **调用方/接口迁移**：无签名变化；`CVRModule.forward→SequenceModule.forward` 不变；`layers_map` key、写入顺序的最终可见值和下游读取不变。
- **tensor 契约**：`din_2k`、mask 与输出的 logical/local shape、stride、dtype、device、placement 不变；`torch.where` 输出仍是新 allocation，不形成 view/alias；删除的第二个临时 tensor 生命周期消失，保留结果生命周期不变。Forward 最终 tensor bitwise 相同；dead write 不进入 loss，Backward 梯度图和梯度值不变。
- **等价不变量**：token/head/expert 顺序、mask、offset、position、attn_arg、collective、RNG 调用、参数、optimizer state 全部不变。
- **拟新增测试**：`develop/frontend/storage/testset/unit/test_6451993_duplicate_assignment_patch.py`，复用 `dag_session_test_infra.py::run_dag_session`；命令：`cd develop/frontend/storage && pytest -q testset/unit/test_6451993_duplicate_assignment_patch.py --cov=testset/extracted/6451993/modelcode/main_model.py --cov-branch --cov-report=term-missing --cov-fail-under=90`，branch 必须 `>=50%`。
- **阈值**：E0 输出、loss、所有输入/参数梯度 bitwise equal；RNG state、参数 key/value、optimizer state key/value bitwise equal。
- **性能/上限**：现有局部 timing 缺失；保守可承诺收益下限为 0。实验采集 `torch.where/zeros_like` FX 节点数、kernel count、被删除 kernel duration、临时显存和 Step；上线收益上限仅为 Patch 前后消失的该调用 kernel duration + 实测相邻可消除 gap。
- **回滚**：任一 bitwise、图节点、梯度、RNG/state 契约失败，或 kernel/Step 不改善即恢复原分支。

### 7.2 P0/E0-B：单次计算视频 mask，移除 `tile` 物化并直接迁移 C2kTrans 接口

- **问题/证据**：`SequenceModule.forward` L1496、L1635 与两个 `C2kTrans.forward` 调用内部 L1057 重复计算 `is_video_samples(external_action)` 并 `tile([B,1]→[B,H])`。`torch.where` 原生支持 `[B,1]` 对 `[B,H]` broadcast。
- **优先级/等价/风险**：P0/E0；mask bool 值、选择语义与梯度不变。接口直接增加必填 `is_video_mask`，禁止 `None`、fallback 或兼容 shim。
- **版本定位**：`main_model.py:SequenceModule.forward` L1431-L1730；`C2kTrans.forward` L1029-L1105。

**改前：C2kTrans 签名与选择分支**

```python
def forward(
    self,
    uid_more,
    live_query_concat,
    video_query_concat,
    external_action,
    real_instance_flag,
    segment_begin_flag,
    joiner_backfill_mask,
    mix_query,
    prepared_sequence_delta=None,
    prepared_segment_begin_idx=None,
    prepared_sequence=None,
):
    uid_emb_trans = self.uid_emb_trans(uid_more)
    query_emb_trans_live = self.query_emb_trans_live(live_query_concat)
    query_emb_trans_video = self.query_emb_trans_video(video_query_concat)
    query_emb_trans = torch.where(
        torch.tile(
            is_video_samples(external_action),
            [1, self.seq_trans_base_params["hidden_size"]],
        ),
        query_emb_trans_video,
        query_emb_trans_live,
    )
```

**改后：C2kTrans 必填接口与 broadcast 选择**

```python
def forward(
    self,
    uid_more,
    live_query_concat,
    video_query_concat,
    external_action,
    is_video_mask,
    real_instance_flag,
    segment_begin_flag,
    joiner_backfill_mask,
    mix_query,
    prepared_sequence_delta=None,
    prepared_segment_begin_idx=None,
    prepared_sequence=None,
):
    if is_video_mask.dtype != torch.bool:
        raise TypeError(f"is_video_mask must be bool, got {is_video_mask.dtype}")
    if is_video_mask.ndim != 2 or is_video_mask.shape[1] != 1:
        raise RuntimeError(
            f"is_video_mask must have shape [B, 1], got {tuple(is_video_mask.shape)}"
        )
    if is_video_mask.shape[0] != live_query_concat.shape[0]:
        raise RuntimeError("is_video_mask batch size must match query batch size")

    uid_emb_trans = self.uid_emb_trans(uid_more)
    query_emb_trans_live = self.query_emb_trans_live(live_query_concat)
    query_emb_trans_video = self.query_emb_trans_video(video_query_concat)
    query_emb_trans = torch.where(
        is_video_mask,
        query_emb_trans_video,
        query_emb_trans_live,
    )
```

**改前：SequenceModule 中计算与两个调用方**

```python
def forward(self, external_action, din_user_aware_ue, din_user_aware_nfm,
            joiner_backfill_mask, uid_more, real_instance_flag,
            segment_begin_flag):
    # ... existing body ...
    final_query_tensors = torch.where(
        torch.tile(is_video_samples(external_action), [1, cur_seq_emb_dim]),
        seq_video_query,
        seq_live_query,
    )
    # ... existing body ...
    convert_query_embs_for_mix = torch.where(
        torch.tile(is_video_samples(external_action), [1, CONVERT_SEQ_DIM * 2]),
        convert_video_query_embs_for_mix,
        convert_live_query_embs_for_mix,
    )
    # ... existing body ...
    transf_soft_flow_2k, length_info_2k = self.c2k_trans(
        uid_more, c2k_live_query_concat, c2k_video_query_concat,
        external_action, real_instance_flag, segment_begin_flag,
        joiner_backfill_mask, mix_query,
        prepared_sequence_delta=prepared_c2k_sequence_delta,
        prepared_segment_begin_idx=prepared_c2k_segment_begin_idx,
    )
    transf_soft_flow_pay, length_info_pay = self.cpay_trans(
        uid_more, cpay_live_query_concat, cpay_video_query_concat,
        external_action, real_instance_flag, segment_begin_flag,
        joiner_backfill_mask, mix_query,
        prepared_sequence=prepared_cpay_sequence,
    )
    return seq_bias_dict, {"2k": length_info_2k, "pay": length_info_pay}
```

**改后：SequenceModule 单次计算、broadcast 与两个调用方完整迁移**

```python
def forward(self, external_action, din_user_aware_ue, din_user_aware_nfm,
            joiner_backfill_mask, uid_more, real_instance_flag,
            segment_begin_flag):
    is_video_mask = is_video_samples(external_action)
    if is_video_mask.dtype != torch.bool:
        raise TypeError(f"is_video_samples must return bool, got {is_video_mask.dtype}")
    if is_video_mask.ndim != 2 or is_video_mask.shape[1] != 1:
        raise RuntimeError(
            f"is_video_samples must return [B, 1], got {tuple(is_video_mask.shape)}"
        )

    # ... unchanged preparation/body between the shown replacement regions ...
    final_query_tensors = torch.where(
        is_video_mask,
        seq_video_query,
        seq_live_query,
    )
    # ... unchanged body ...
    convert_query_embs_for_mix = torch.where(
        is_video_mask,
        convert_video_query_embs_for_mix,
        convert_live_query_embs_for_mix,
    )
    # ... unchanged body ...
    transf_soft_flow_2k, length_info_2k = self.c2k_trans(
        uid_more, c2k_live_query_concat, c2k_video_query_concat,
        external_action, is_video_mask, real_instance_flag,
        segment_begin_flag, joiner_backfill_mask, mix_query,
        prepared_sequence_delta=prepared_c2k_sequence_delta,
        prepared_segment_begin_idx=prepared_c2k_segment_begin_idx,
    )
    transf_soft_flow_pay, length_info_pay = self.cpay_trans(
        uid_more, cpay_live_query_concat, cpay_video_query_concat,
        external_action, is_video_mask, real_instance_flag,
        segment_begin_flag, joiner_backfill_mask, mix_query,
        prepared_sequence=prepared_cpay_sequence,
    )
    return seq_bias_dict, {"2k": length_info_2k, "pay": length_info_pay}
```

- **调用方迁移**：本扫描版本中两个直接调用方 `self.c2k_trans` L1705 与 `self.cpay_trans` L1718 均已修改；C2kTrans 内部不再自行计算 mask。无默认值和兼容分支。
- **tensor 契约**：mask logical/local `[B,1]` bool；broadcast 后选择结果仍 `[B,H]`，output stride/layout/dtype/device/placement 由 `torch.where` 保持；不创建 `[B,H]` tile tensor；mask 是只读 alias-free tensor，生命周期覆盖 SequenceModule.forward。Forward 选择逐元素相同；Backward 对 live/video 分支的梯度 mask 相同；mask 不求梯度。
- **不变量**：token/head/expert 顺序、packed offset、position、attn_arg、collective、RNG、参数和 optimizer state 不变。
- **拟新增测试**：`testset/unit/test_6451993_video_mask_broadcast_patch.py`，复用 `run_dag_session`；命令：`cd develop/frontend/storage && pytest -q testset/unit/test_6451993_video_mask_broadcast_patch.py --cov=testset/extracted/6451993/modelcode/main_model.py --cov-branch --cov-report=term-missing --cov-fail-under=90`，branch `>=50%`。
- **阈值**：E0 输出/loss/输入梯度/参数梯度 bitwise；mask 和 router index 逐元素相同；RNG/state bitwise。
- **性能/上限**：保守下限 0；采集 `is_video_samples`、tile 节点/kernel 数、临时 allocated bytes、Step。上限仅为实际消失的 mask/tile kernel duration与实测 gap，不使用父 Module 时间。
- **回滚**：bitwise/shape/placement/梯度失败，tile kernel 未减少或 Step 退化即整体回退签名与两个调用方。

---

## 8. P1/E1 Patch 与实验性候选（以第 8.13 节 Diff 审计为最终状态）

### 8.1 P1/E1-C：以严格逐项 add 替换已确认的 `stack→sum(dim=0)`

- **问题/定位**：`main_model.py:NFMModule.forward` L672-L677、`InsTrans.prepare_dense_sequence_inputs` L1167、`SequenceModule.forward` L1666。`stack` 先物化 `[N,...]` 临时 tensor 再 reduce。
- **等级/风险**：P1/E1；数学和梯度公式等价，但逐项 add 的浮点累加树可能与 reduction kernel 不同。

**新增的完整严格 helper（放在 main_model.py 模块级、NFMModule 前）**

```python
def _sum_nonempty_tensors(tensors, *, name):
    if not isinstance(tensors, (list, tuple)):
        raise TypeError(f"{name} must be a list or tuple, got {type(tensors).__name__}")
    if len(tensors) == 0:
        raise RuntimeError(f"{name} must not be empty")
    first = tensors[0]
    if not isinstance(first, torch.Tensor):
        raise TypeError(f"{name}[0] must be Tensor")
    result = first
    for index, tensor in enumerate(tensors[1:], start=1):
        if not isinstance(tensor, torch.Tensor):
            raise TypeError(f"{name}[{index}] must be Tensor")
        if tensor.shape != first.shape:
            raise RuntimeError(
                f"{name}[{index}] shape {tuple(tensor.shape)} != {tuple(first.shape)}"
            )
        if tensor.dtype != first.dtype or tensor.device != first.device:
            raise RuntimeError(f"{name}[{index}] dtype/device mismatch")
        result = result + tensor
    return result
```

**改前的三个已确认调用区域**

```python
nfm_network_embeddings[user_layer_name] = torch.sum(
    torch.stack(nfm_network_embeddings[user_layer_name]), dim=0
)
nfm_network_embeddings[group_layer_name] = torch.sum(
    torch.stack(nfm_network_embeddings[group_layer_name]), dim=0
)

seq_embed = self.ins_input_tower(torch.sum(torch.stack(instance_seq_embs), dim=0))

convert_xd_tensor = torch.sum(torch.stack(convert_seq_embs), dim=0)
```

**可直接替换的三个调用区域**

```python
nfm_network_embeddings[user_layer_name] = _sum_nonempty_tensors(
    nfm_network_embeddings[user_layer_name],
    name=f"nfm_network_embeddings[{user_layer_name}]",
)
nfm_network_embeddings[group_layer_name] = _sum_nonempty_tensors(
    nfm_network_embeddings[group_layer_name],
    name=f"nfm_network_embeddings[{group_layer_name}]",
)

seq_embed = self.ins_input_tower(
    _sum_nonempty_tensors(instance_seq_embs, name="instance_seq_embs")
)

convert_xd_tensor = _sum_nonempty_tensors(
    convert_seq_embs, name="convert_seq_embs"
)
```

- **接口/调用迁移**：仅模块内 helper；三个调用方全部迁移。空列表从 `torch.stack` 异常改为更早、明确的 `RuntimeError`，不静默返回；非 Tensor、shape/dtype/device 不一致显式失败。
- **tensor 契约**：输入 logical/local shape、stride/layout、dtype/device/placement 不变；输出 shape/dtype/device 不变；不再创建 stack 临时 allocation。每次 `+` 产生新 tensor，无 in-place alias。DTensor placement 必须逐输入相同，否则测试失败。Backward 每个输入梯度数学上为上游梯度，浮点执行路径可能变化。
- **不变量**：列表顺序固定；token/head/expert、mask/offset、collective、RNG、参数和 optimizer state 不变。
- **拟新增测试**：`testset/unit/test_6451993_stack_sum_patch.py`，复用 `run_dag_session`；命令：`cd develop/frontend/storage && pytest -q testset/unit/test_6451993_stack_sum_patch.py --cov=testset/extracted/6451993/modelcode/main_model.py --cov-branch --cov-report=term-missing --cov-fail-under=90`，branch `>=50%`。
- **阈值**：FP32 output/input-grad `atol=1e-6, rtol=1e-5`；BF16 `atol=2e-2, rtol=2e-2`；loss 同阈值；逐输入梯度、列表顺序和空列表 raise 必测。
- **性能/上限**：保守下限 0；采集 stack/reduction/add kernel 数、kernel union、临时 peak bytes、Step。上限仅为三个调用点被消除的 stack copy/reduction duration 与实测 gap；若 add 链更慢则不上线。
- **回滚**：误差超阈值、kernel 数/显存/Step 不改善、编译图退化或 DTensor placement 变化即恢复 `stack→sum`。

### 8.2 P1/E1-D：storage 独立 CUDA Graph capture/replay 实验 Runner

- **落点/边界**：当前模型提取包没有可定位的训练循环、`runstep` 或 optimizer step 入口；因此 Patch 明确落在测试仓库 `develop/frontend/storage/testset/unit/cuda_graph_6451993_runner.py`，不进入研发仓库。它以注入的 `step_fn(model, inputs, optimizer)` 定义 capture 边界；若 optimizer 非空，`zero_grad(set_to_none=False)`、forward、backward、`optimizer.step()` 全部 capture。
- **等级/风险**：P1/E1 实验 Patch；数学目标相同，但 capture/replay 可能改变 launch 路径。动态 shape、dtype、stride、device 或输入结构变化显式 raise，绝不 eager fallback。

**拟新增完整 Runner 代码**

```python
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable
import torch


@dataclass(frozen=True)
class TensorSpec:
    shape: tuple[int, ...]
    stride: tuple[int, ...]
    dtype: torch.dtype
    device: torch.device


def _spec(tensor: torch.Tensor) -> TensorSpec:
    return TensorSpec(tuple(tensor.shape), tuple(tensor.stride()), tensor.dtype, tensor.device)


class StrictCudaGraphStep:
    def __init__(
        self,
        model: torch.nn.Module,
        example_inputs: tuple[torch.Tensor, ...],
        step_fn: Callable,
        optimizer: torch.optim.Optimizer | None,
        warmup_steps: int = 3,
    ):
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA Graph experiment requires CUDA")
        if warmup_steps < 1:
            raise ValueError("warmup_steps must be >= 1")
        if not example_inputs:
            raise RuntimeError("example_inputs must not be empty")
        if any(not isinstance(x, torch.Tensor) or not x.is_cuda for x in example_inputs):
            raise TypeError("all example_inputs must be CUDA tensors")

        self.model = model
        self.step_fn = step_fn
        self.optimizer = optimizer
        self.static_inputs = tuple(torch.empty_like(x).copy_(x) for x in example_inputs)
        self.input_specs = tuple(_spec(x) for x in example_inputs)
        self.graph = torch.cuda.CUDAGraph()
        self.warmup_stream = torch.cuda.Stream()

        current_stream = torch.cuda.current_stream()
        self.warmup_stream.wait_stream(current_stream)
        with torch.cuda.stream(self.warmup_stream):
            for _ in range(warmup_steps):
                self._run_static_step()
        current_stream.wait_stream(self.warmup_stream)
        torch.cuda.synchronize()

        if optimizer is not None:
            optimizer.zero_grad(set_to_none=False)
        with torch.cuda.graph(self.graph):
            self.static_output = self._run_static_step()

    def _run_static_step(self):
        if self.optimizer is not None:
            self.optimizer.zero_grad(set_to_none=False)
        output = self.step_fn(self.model, self.static_inputs, self.optimizer)
        if not isinstance(output, torch.Tensor) or output.numel() != 1:
            raise RuntimeError("step_fn must return a scalar CUDA Tensor loss")
        output.backward()
        if self.optimizer is not None:
            self.optimizer.step()
        return output

    def replay(self, inputs: tuple[torch.Tensor, ...]) -> torch.Tensor:
        if len(inputs) != len(self.static_inputs):
            raise RuntimeError("input arity changed")
        for index, (source, target, expected) in enumerate(
            zip(inputs, self.static_inputs, self.input_specs)
        ):
            actual = _spec(source)
            if actual != expected:
                raise RuntimeError(
                    f"input {index} contract changed: expected {expected}, got {actual}"
                )
            target.copy_(source)
        self.graph.replay()
        return self.static_output.detach().clone()
```

- **接口/生命周期**：`step_fn` 必须返回 scalar CUDA loss；static input buffer 地址在对象生命周期内稳定；每次 replay 只 copy 输入，不重新分配 capture 内 tensor。optimizer state 和 parameter storage 在 capture 后不得替换。
- **tensor/训练契约**：shape/stride/dtype/device 固定；layout/placement 由 exact spec 和测试检查；输入 copy 不 alias 用户 tensor；Forward、Backward 和 optimizer step 均在 capture 内。RNG 仅在 CUDA Graph 支持的 generator 状态下验证，dropout replay 必须与 eager 固定 seed 序列对比。
- **准入检查**：连续 10 Step spec 相同、参数/state data_ptr 稳定、无 graph break、无 capture 非法 allocation、NCCL 序列稳定；任一失败直接 raise。
- **拟新增测试**：`testset/unit/test_6451993_cuda_graph_patch.py`，用小型可复现 module 和与真实 6451993 输入 spec 相同的 mock；命令：`cd develop/frontend/storage && pytest -q testset/unit/test_6451993_cuda_graph_patch.py --cov=testset/unit/cuda_graph_6451993_runner.py --cov-branch --cov-report=term-missing --cov-fail-under=90`，branch `>=50%`。
- **阈值**：FP32 loss/参数/grad `atol=1e-6, rtol=1e-5`；BF16 `atol=2e-2, rtol=2e-2`；optimizer state 逐 key 同阈值；固定 seed 的 RNG 序列一致。
- **性能/上限**：保守下限 0。采集 eager/capture 的 CPU runtime launch events、kernel 数、相邻 gap、Step；上限仅为实测可消除 CPU launch gap，不使用 72.185 ms 非 kernel gap或 kernel duration总和。
- **回滚**：契约 raise、capture error、数值/state/RNG 超阈值、显存增长越线或 Step 不改善即禁用该实验 Runner；没有 eager fallback。

### 8.3 P1/E1-E：严格同配置 optimizer param-group 合并实验 Harness

- **定位/边界**：提取模型源码未包含 optimizer 构造和 `step` 入口；Top #4 只证明 Adamom 121 calls/step。Patch 落在 `develop/frontend/storage/testset/unit/optimizer_group_merge_6451993.py`，对运行时已有 `optimizer.param_groups` 做严格重建和一步 update A/B，不把未知分组事实写进生产代码。
- **等级/风险**：P1/E1；更新公式和组内超参保持，foreach/fused tensor-list 与浮点更新顺序可能变化。

**拟新增完整 Harness 代码**

```python
from __future__ import annotations

import copy
import torch


_REQUIRED_KEYS = ("lr", "betas", "eps", "weight_decay")


def _placement_key(parameter: torch.Tensor):
    placements = getattr(parameter, "placements", None)
    if placements is None:
        return None
    return tuple(str(item) for item in placements)


def _group_key(group):
    missing = [key for key in _REQUIRED_KEYS if key not in group]
    if missing:
        raise RuntimeError(f"optimizer group misses required keys: {missing}")
    params = group["params"]
    if not params:
        raise RuntimeError("optimizer group must not be empty")
    first = params[0]
    tensor_contract = (first.dtype, first.device, _placement_key(first))
    for parameter in params:
        if (parameter.dtype, parameter.device, _placement_key(parameter)) != tensor_contract:
            raise RuntimeError("a source group mixes dtype/device/placement")
    extra = tuple(
        sorted(
            (key, repr(value))
            for key, value in group.items()
            if key != "params" and key not in _REQUIRED_KEYS
        )
    )
    return (
        group["lr"], tuple(group["betas"]), group["eps"],
        group["weight_decay"], tensor_contract, extra,
    )


def rebuild_with_merged_identical_groups(optimizer: torch.optim.Optimizer):
    buckets = {}
    configs = {}
    for group in optimizer.param_groups:
        key = _group_key(group)
        configs.setdefault(key, {k: v for k, v in group.items() if k != "params"})
        buckets.setdefault(key, []).extend(group["params"])

    rebuilt_groups = []
    for key in sorted(buckets, key=repr):
        rebuilt_groups.append({**configs[key], "params": buckets[key]})

    rebuilt = type(optimizer)(rebuilt_groups, **optimizer.defaults)
    for parameter, state in optimizer.state.items():
        rebuilt.state[parameter] = copy.deepcopy(state)
    if set(rebuilt.state) != set(optimizer.state):
        raise RuntimeError("optimizer state parameter set changed")
    return rebuilt
```

- **接口/迁移**：输入现有 optimizer，输出同 class optimizer；只合并完整 key 相同且 dtype/device/placement 相同的组。任何源组内部混合、缺 key 或空组直接 raise；不同配置保持不同 bucket，绝不静默混组。
- **契约**：参数 object、shape/stride/layout/dtype/device/placement 不变；state 逐 parameter deep-copy，key/shape/dtype 保持；不改变 Forward/Backward/RNG，只可能改变 optimizer step 的 tensor-list 分桶和浮点更新顺序。
- **拟新增测试**：`testset/unit/test_6451993_optimizer_group_merge_patch.py`，构造相同/不同超参组并对比一步 update；命令：`cd develop/frontend/storage && pytest -q testset/unit/test_6451993_optimizer_group_merge_patch.py --cov=testset/unit/optimizer_group_merge_6451993.py --cov-branch --cov-report=term-missing --cov-fail-under=90`，branch `>=50%`。
- **阈值**：FP32 参数/update/state `atol=1e-7, rtol=1e-6`；BF16 参数 `atol=2e-2, rtol=2e-2`，FP32 master/state `atol=1e-6, rtol=1e-5`；loss、grad、参数顺序和 state key 必查。
- **性能/上限**：保守下限 0；采集 param-group 数、每组 tensor/numel、foreach/fused launch 数、#4 调用子集 duration、CPU launch gap和 Step。上限仅为实际消失的 optimizer launch gap/kernel 子集，不使用 #4 全部 51.760 ms，除非全部调用被证明消除。
- **回滚**：任一 update/state 超阈值、optimizer class 不支持按组重建、launch 不减或 Step 退化，保留原 optimizer。

### 8.4 P1/E1-F：MLPMixer layout adapter（实验性候选 Patch，未验证）

- **逐字改前与真实 consumer 链**：`pure_transformer.py:MLPMixer.separate_heads` L862-L866 返回 `x.transpose(1, 2)`；`forward` L874 首先调用它，首个 consumer 在 self-mixup 分支是 `self.attn(x, tao=tao)`，否则是 L881 `torch.reshape`，之后 L886 `self.mlp(self.ln(x))`。当前 trace 不能把 #6/#14/#16/#20 唯一绑定到这条链，故本条不得称正式 P1。
- **文件/注册/import**：adapter 放在真实文件 `pure_transformer.py` 模块级（现有 `import torch` 可直接使用）；运行 harness 新增到 storage 的 `testset/unit/test_6451993_patch_f.py`。不注册全局 custom op，避免孤立接口。

**可直接应用到当前扫描源码的模型接入 unified diff：**

```diff
--- a/testset/extracted/6451993/modelcode/pure_transformer.py
+++ b/testset/extracted/6451993/modelcode/pure_transformer.py
@@ -859,11 +859,28 @@ class MLPMixer(nn.Module):
             and not self.enable_self_mixup
         )

+    @staticmethod
+    def _layout_adapter_f(x, *, consumer):
+        if not isinstance(x, torch.Tensor):
+            raise TypeError("F requires a local torch.Tensor")
+        if x.dtype not in (torch.float16, torch.bfloat16, torch.float32):
+            raise RuntimeError(f"F unsupported dtype: {x.dtype}")
+        if x.device.type != "cuda":
+            raise RuntimeError("F requires CUDA")
+        if consumer not in ("attention", "reshape"):
+            raise RuntimeError(f"F unknown consumer: {consumer}")
+        # Candidate adapter deliberately materializes the consumer contract.
+        # It is a measurement harness, not a claimed zero-copy implementation.
+        y = x.contiguous()
+        if y.shape != x.shape or y.dtype != x.dtype or y.device != x.device:
+            raise RuntimeError("F changed tensor contract")
+        return y
+
     def separate_heads(self, x):
         x = torch.reshape(
             x, (-1, self.token_num, self.num_heads, self.head_dim)
         )
         return x.transpose(1, 2)
@@ -871,10 +888,13 @@ class MLPMixer(nn.Module):
         else:
             # (B, T, D) -> (B, T, H, D') -> (B, H, T, D')
             x = self.separate_heads(x)
             if self.enable_self_mixup:
+                x = self._layout_adapter_f(x, consumer="attention")
                 # (B, H, T, D') -> attn -> (B, H, T, D')
                 a = self.attn(x, tao=tao)
                 x = x + F.dropout(a, p=self.dropout)

             # (B, H, T, D') -> (B, H, T * D')
+            x = self._layout_adapter_f(x, consumer="reshape")
             x = torch.reshape(x, [x.shape[0], self.num_heads, self.new_dim])
```

**storage 可运行 harness 的新增文件 unified diff（完整运行入口）：**

```diff
--- /dev/null
+++ b/testset/unit/test_6451993_patch_f.py
@@ -0,0 +1,31 @@
+import importlib.util
+from pathlib import Path
+import torch
+
+SOURCE = Path("testset/extracted/6451993/modelcode/pure_transformer.py")
+
+def _load_module():
+    spec = importlib.util.spec_from_file_location("pure_transformer_6451993", SOURCE)
+    if spec is None or spec.loader is None:
+        raise RuntimeError("cannot load scanned pure_transformer.py")
+    module = importlib.util.module_from_spec(spec)
+    spec.loader.exec_module(module)
+    return module
+
+def test_layout_adapter_f_forward_backward():
+    if not torch.cuda.is_available():
+        raise RuntimeError("F harness requires CUDA; skip is forbidden")
+    module = _load_module()
+    x = torch.randn(2, 3, 4, 8, device="cuda", dtype=torch.bfloat16,
+                    requires_grad=True).transpose(1, 2)
+    assert not x.is_contiguous()
+    y = module.MLPMixer._layout_adapter_f(x, consumer="reshape")
+    assert y.shape == x.shape and y.stride() == (96, 24, 8, 1)
+    assert y.dtype == x.dtype and y.device == x.device
+    y.float().sum().backward()
+    assert x.grad is not None
+
+if __name__ == "__main__":
+    test_layout_adapter_f_forward_backward()
```

- **返回值接续**：adapter 返回同 shape/dtype/device 的 contiguous Tensor，原 consumer 不改签名：attention 分支继续传给 `self.attn`，另一分支继续传给现有 `torch.reshape`、`self.ln`、`self.mlp`。logical shape `[B,H,T,D']`；输入 transpose stride 由运行时记录，输出 stride 为 contiguous；本地 Tensor 不声明 DTensor placement。若实参为 DTensor，类型门禁硬失败，因此尚未闭合生产 placement。
- **Backward**：`contiguous()` 的 autograd 将梯度按原 view/transpose 链返回；harness 必查 input grad。没有自定义 backward、fallback 或静默 skip。
- **运行入口**：`cd develop/frontend/storage && python3 testset/unit/test_6451993_patch_f.py`；随后用相同五步 trace 比较 adapter 前后 copy/clone、首个 consumer GEMM 与 step。因为该 adapter 本身仍 materialize，只有证明 consumer 接受原 stride并形成 producer 直写实现后才能升级正式 P1；当前收益下限/承诺均为 0。

### 8.5 P1/E1-G：MoE problem-array 与 combine（实验性候选 Patch，未验证）

- **目标文件/调用方**：`sparse_moe.py:SparseMoe.forward` L354-L389，调用点 `self.ffn(x_scatter, expert_token_cnt)` 后接 gather/combine。
- **候选接入代码**：

```python
x_ffn, moe_meta = self.ffn_grouped_experiment(
    x_scatter,
    expert_token_cnt=expert_token_cnt,
    scatter_index=scatter_index,
    combine_order="gather_input_order",
)
if moe_meta.problem_count != int((expert_token_cnt > 0).sum()):
    raise RuntimeError("grouped-GEMM problem array does not match active experts")
output_flat = self.moe_gather_experiment(
    x_ffn, scatter_index, top_k_probs.to(x.dtype), moe_meta.inverse_index
)
```

- **Diff 审计**：当前扫描源码没有 `ffn_grouped_experiment` / `moe_gather_experiment` 的实现、import 或注册点，无法生成能应用且能运行的完整 unified diff；因此 **FAIL→实验性候选**，不得称正式 Patch。运行入口暂定 `pytest -q testset/unit/test_sparse_moe_grouped_experiment.py`，但在实现文件、调用迁移、Fwd/Bwd 与 router-grad Diff 补齐前不得执行性能宣称。

### 8.6 P1/E1-H：Residual + LayerNorm 严格 custom-autograd 实验

- **落点/调用点**：拟新增 `develop/frontend/storage/testset/unit/residual_layernorm_patch_h.py`；替换 `pure_transformer.py:ResidualAttentionBlock.forward` L1026-L1053 中相邻 `residual + branch`、`LayerNorm` 调用。#7/#10/#12 就地回链本代码。
- **完整实验代码**（独立可解析；不支持契约直接报错，不做 eager fallback）：

```python
import torch
from torch import Tensor

class ResidualLayerNormFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, residual: Tensor, branch: Tensor, weight: Tensor,
                bias: Tensor, eps: float) -> Tensor:
        if residual.shape != branch.shape or residual.stride() != branch.stride():
            raise RuntimeError("H requires identical residual/branch shape and stride")
        if residual.device != branch.device or residual.dtype != branch.dtype:
            raise RuntimeError("H requires identical device and dtype")
        if residual.dtype not in (torch.float16, torch.bfloat16, torch.float32):
            raise RuntimeError("H unsupported activation dtype")
        if weight.ndim != 1 or weight.numel() != residual.shape[-1]:
            raise RuntimeError("H invalid LayerNorm weight")
        if bias.shape != weight.shape or weight.device != residual.device:
            raise RuntimeError("H invalid bias/device")
        x = residual + branch
        acc = x.float()
        mean = acc.mean(dim=-1, keepdim=True)
        rstd = torch.rsqrt((acc - mean).square().mean(dim=-1, keepdim=True) + eps)
        y = ((acc - mean) * rstd).to(residual.dtype) * weight + bias
        ctx.save_for_backward(residual, branch, weight, bias)
        ctx.eps = float(eps)
        return y

    @staticmethod
    def backward(ctx, grad_out: Tensor):
        residual, branch, weight, bias = ctx.saved_tensors
        with torch.enable_grad():
            r = residual.detach().requires_grad_(True)
            b = branch.detach().requires_grad_(True)
            w = weight.detach().requires_grad_(True)
            z = bias.detach().requires_grad_(True)
            x = r + b
            acc = x.float()
            mean = acc.mean(-1, keepdim=True)
            rstd = torch.rsqrt((acc - mean).square().mean(-1, keepdim=True) + ctx.eps)
            y = ((acc - mean) * rstd).to(r.dtype) * w + z
            grads = torch.autograd.grad(y, (r, b, w, z), grad_out)
        return grads[0], grads[1], grads[2], grads[3], None

def residual_layernorm(residual: Tensor, branch: Tensor,
                       layer_norm: torch.nn.LayerNorm) -> Tensor:
    if layer_norm.elementwise_affine is not True or layer_norm.bias is None:
        raise RuntimeError("H requires affine LayerNorm with bias")
    shape = layer_norm.normalized_shape
    if len(shape) != 1 or shape[0] != residual.shape[-1]:
        raise RuntimeError("H supports only last-dimension LayerNorm")
    return ResidualLayerNormFn.apply(
        residual, branch, layer_norm.weight, layer_norm.bias, layer_norm.eps)
```

- **调用迁移**：把 `x = self.ln(x + branch)` 改为 `x = residual_layernorm(x, branch, self.ln)`；必须迁移同一 block 的 forward，backward 由 Function 成对提供。logical/local shape、stride、dtype/device 不变；不跨 DTensor placement/stream/alias 边界；参数、RNG、optimizer state 不变，FP32 accumulator 下归为 E1。
- **验证/门槛**：拟新增 `test_6451993_patch_h.py`；`cd develop/frontend/storage && pytest -q testset/unit/test_6451993_patch_h.py --cov=testset.unit.residual_layernorm_patch_h --cov-branch --cov-report=term-missing --cov-fail-under=90`，branch≥50%。FP32 `atol=1e-6,rtol=1e-5`，BF16/FP16 `atol=2e-2,rtol=2e-2`，逐输入/weight/bias grad 同阈值；收益下限 0，上限仅为替换后实测消失的 #7/#10/#12 子集及相邻 gap；任一契约、数值、显存或 Step 退化即回滚。

### 8.7 P1/E1-I：FlashAttention packed-length 固定配置实验

- **落点/调用点**：拟新增 `develop/frontend/storage/testset/unit/flash_packed_patch_i.py`，在 `ResidualAttentionBlock` attention 调用前构造配置；#8/#9/#13 回链本代码。
- **完整实验代码**：

```python
from dataclasses import dataclass
import torch
from torch import Tensor

@dataclass(frozen=True)
class PackedAttentionConfig:
    cu_seqlens: Tensor
    max_seqlen: int
    token_count: int
    lengths_key: tuple[int, ...]

class PackedAttentionConfigCache:
    def __init__(self) -> None:
        self._config: PackedAttentionConfig | None = None

    def get(self, lengths: Tensor, token_count: int, device: torch.device
            ) -> PackedAttentionConfig:
        if lengths.ndim != 1 or lengths.dtype not in (torch.int32, torch.int64):
            raise RuntimeError("I requires rank-1 integer lengths")
        if lengths.device.type != "cpu":
            raise RuntimeError("I requires CPU lengths to prove a stable bucket")
        values = tuple(int(v) for v in lengths.tolist())
        if not values or min(values) <= 0 or sum(values) != token_count:
            raise RuntimeError("I invalid packed lengths/token count")
        if self._config is not None:
            if self._config.lengths_key != values or self._config.token_count != token_count:
                raise RuntimeError("I dynamic packed bucket is unsupported")
            return self._config
        prefix = torch.zeros(len(values) + 1, dtype=torch.int32, device=device)
        prefix[1:] = torch.tensor(values, dtype=torch.int32, device=device).cumsum(0)
        self._config = PackedAttentionConfig(prefix, max(values), token_count, values)
        return self._config

def packed_flash_attention(q: Tensor, k: Tensor, v: Tensor, lengths: Tensor,
                           cache: PackedAttentionConfigCache, kernel) -> Tensor:
    if q.shape != k.shape or q.shape != v.shape or q.ndim != 3:
        raise RuntimeError("I expects equal [tokens, heads, dim] q/k/v")
    if q.device != k.device or q.device != v.device or q.dtype != k.dtype or q.dtype != v.dtype:
        raise RuntimeError("I q/k/v device and dtype mismatch")
    if not q.is_contiguous() or not k.is_contiguous() or not v.is_contiguous():
        raise RuntimeError("I requires contiguous packed q/k/v")
    cfg = cache.get(lengths, q.shape[0], q.device)
    return kernel(q, k, v, cu_seqlens_q=cfg.cu_seqlens,
                  cu_seqlens_k=cfg.cu_seqlens, max_seqlen_q=cfg.max_seqlen,
                  max_seqlen_k=cfg.max_seqlen, causal=False)
```

- **调用迁移/契约**：block 构造时创建一个 cache；原 attention 调用改为 `packed_flash_attention(q,k,v,lengths,self._packed_cfg,flash_kernel)`。`lengths` 必须与 mask/offset/position 切分完全一致；token/head 顺序、position、causal、dtype/device/placement、输出 shape 不变；动态 bucket 显式 raise，禁止改走 padded fallback。Fwd/Bwd 均由同一 FlashAttention kernel 路径验证，参数/RNG/optimizer state 不变。
- **验证/门槛**：`pytest -q testset/unit/test_6451993_patch_i.py --cov=testset.unit.flash_packed_patch_i --cov-branch --cov-report=term-missing --cov-fail-under=90`，branch≥50%；output/grad BF16 `atol=2e-2,rtol=2e-2`，FP32 `1e-5/1e-5`，另断言 cu_seqlens、offset、position、mask hash。收益下限 0，上限仅为 #8/#9/#13 中因稳定配置实测消失的 launch/gap；动态长度、误差、显存或 Step 退化即回滚。

### 8.8 P1/E1-J：Parameter `_version` 键控 BF16 cast cache

- **落点/调用点**：拟新增 `develop/frontend/storage/testset/unit/versioned_cast_patch_j.py`；替换已由 I-08 correlation 唯一绑定的 `parameter.to(torch.bfloat16)` 调用。#19 的 454 calls/step、8.974 ms/step 就地回链本代码。
- **完整实验代码**：

```python
import torch
from torch import Tensor

class VersionedCastCache:
    def __init__(self) -> None:
        self._entries: dict[tuple[int, int, str, int | None, torch.dtype], Tensor] = {}

    def cast_parameter(self, parameter: torch.nn.Parameter,
                       target_dtype: torch.dtype) -> Tensor:
        if target_dtype != torch.bfloat16:
            raise RuntimeError("J supports only the measured FP32-to-BF16 cast")
        if parameter.dtype != torch.float32 or parameter.device.type != "cuda":
            raise RuntimeError("J requires CUDA FP32 Parameter")
        key = (id(parameter), parameter._version, parameter.device.type,
               parameter.device.index, target_dtype)
        value = self._entries.get(key)
        if value is None:
            self.invalidate_parameter(parameter)
            value = parameter.to(dtype=target_dtype)
            self._entries[key] = value
        if value.device != parameter.device or value.dtype != target_dtype:
            raise RuntimeError("J stale cache contract")
        return value

    def invalidate_parameter(self, parameter: torch.nn.Parameter) -> None:
        identity = id(parameter)
        stale = [key for key in self._entries if key[0] == identity]
        for key in stale:
            del self._entries[key]

    def after_optimizer_step(self, parameters: list[torch.nn.Parameter]) -> None:
        for parameter in parameters:
            self.invalidate_parameter(parameter)

def run_step(model, optimizer, batch, cache: VersionedCastCache):
    output = model(batch, cast_weight=cache.cast_parameter)
    loss = output.float().sum()
    loss.backward()
    optimizer.step()
    cache.after_optimizer_step(list(model.parameters()))
    optimizer.zero_grad(set_to_none=True)
    return loss.detach()
```

- **调用迁移/契约**：consumer 从 `weight.to(torch.bfloat16)` 改为 `cast_weight(weight, torch.bfloat16)`，训练 step 在每次 `optimizer.step()` 后无条件调用 `after_optimizer_step`。key 含 parameter identity、`_version`、device、target dtype；禁止跨 device/collective/stream 生命周期保存临时激活。shape/stride/device 不变，仅 dtype FP32→BF16；参数/state/RNG 不变。in-place 更新会改变 `_version`，显式 step invalidation 再提供双保险。
- **验证/门槛**：`pytest -q testset/unit/test_6451993_patch_j.py --cov=testset.unit.versioned_cast_patch_j --cov-branch --cov-report=term-missing --cov-fail-under=90`，branch≥50%；断言同 version 命中、step 后对象失效、parameter/output/grad 与基线 BF16 `2e-2/2e-2`。收益下限 0，上限仅是被 correlation 证明消失的 #19 子集（绝不预承诺全部 8.974 ms）；陈旧值、峰值显存或 Step 退化即回滚。

### 8.9 P1/E1-K：pack + AllToAll 成对 autograd（实验性候选 Patch，未验证）

- **目标/调用方**：`sharding_plan.py` L102-L110 的 `Shard(1)→Shard(0)` 是 placement 证据；但实际 pack producer 与调用 `all_to_all_single` 的 Python 源码未出现在扫描模型包，不能伪造模型接入点。

```python
class PackAllToAllFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, out_splits, in_splits, group, target_shape):
        packed = pack_for_ranks_cuda(x, in_splits)
        out = torch.empty(sum(out_splits), *packed.shape[1:], dtype=x.dtype, device=x.device)
        torch.distributed.all_to_all_single(out, packed, out_splits, in_splits, group=group)
        ctx.meta = (tuple(out_splits), tuple(in_splits), group, x.shape, x.stride())
        return out.view(target_shape)

    @staticmethod
    def backward(ctx, grad_out):
        out_splits, in_splits, group, shape, stride = ctx.meta
        packed_grad = torch.empty(sum(in_splits), *grad_out.shape[1:], dtype=grad_out.dtype, device=grad_out.device)
        torch.distributed.all_to_all_single(packed_grad, grad_out.contiguous(), in_splits, out_splits, group=group)
        grad_x = unpack_from_ranks_cuda(packed_grad, shape, stride)
        return grad_x, None, None, None, None
```

- **Diff 审计**：`pack_for_ranks_cuda`、`unpack_from_ranks_cuda`、真实 collective 调用方及注册/import 均未闭合，当前无法给出能应用到扫描源码的 unified diff，故 **FAIL→实验性候选**。候选运行入口为 `torchrun --nproc_per_node=8 -m pytest -q testset/unit/test_pack_alltoall_autograd.py`；补齐新文件完整 diff、真实调用迁移 diff、split/placement 与 backward 测试前不得称正式 Patch或承诺 280.899 ms。

### 8.10 P1/E1-L：async AllGather ping-pong 双缓冲

- **落点/调用点**：拟新增 `develop/frontend/storage/testset/unit/async_allgather_patch_l.py`；接入 I-07 唯一归因的 AllGather producer/consumer 之间。
- **完整实验代码**：

```python
import torch
import torch.distributed as dist
from torch import Tensor

class AsyncAllGatherDoubleBuffer:
    def __init__(self, local_shape: tuple[int, ...], dtype: torch.dtype,
                 device: torch.device, group=None) -> None:
        if not dist.is_initialized() or device.type != "cuda":
            raise RuntimeError("L requires initialized NCCL/CUDA distributed runtime")
        self.group = group
        self.world = dist.get_world_size(group)
        self.local_shape = local_shape
        self.dtype = dtype
        self.device = device
        self.buffers = [torch.empty((self.world,) + local_shape, dtype=dtype, device=device)
                        for _ in range(2)]
        self.stream = torch.cuda.Stream(device=device)
        self.events = [torch.cuda.Event() for _ in range(2)]
        self.works = [None, None]
        self.step = 0

    def launch(self, local: Tensor) -> int:
        if tuple(local.shape) != self.local_shape or local.dtype != self.dtype:
            raise RuntimeError("L local shape/dtype changed")
        if local.device != self.device or not local.is_contiguous():
            raise RuntimeError("L device/layout contract changed")
        slot = self.step & 1
        if self.works[slot] is not None:
            raise RuntimeError("L attempted to reuse an in-flight slot")
        with torch.cuda.stream(self.stream):
            self.stream.wait_stream(torch.cuda.current_stream(self.device))
            flat = self.buffers[slot].view((self.world * self.local_shape[0],) + self.local_shape[1:])
            self.works[slot] = dist.all_gather_into_tensor(
                flat, local, group=self.group, async_op=True)
            self.events[slot].record(self.stream)
        self.step += 1
        return slot

    def wait(self, slot: int) -> Tensor:
        if slot not in (0, 1) or self.works[slot] is None:
            raise RuntimeError("L invalid or already-consumed slot")
        self.works[slot].wait()
        torch.cuda.current_stream(self.device).wait_event(self.events[slot])
        result = self.buffers[slot]
        self.works[slot] = None
        return result

    def assert_drained(self) -> None:
        if any(work is not None for work in self.works):
            raise RuntimeError("L step ended with in-flight AllGather")
```

- **调用迁移/契约**：producer 完成后 `slot=runner.launch(local)`，首个真实 consumer 前 `global_tensor=runner.wait(slot)`，step 尾 `assert_drained()`；禁止提前覆写 local/buffer。global rank 顺序、local/global shape、dtype/device、Shard→Replicate placement、collective group 不变；Fwd/Bwd 分别拥有独立 runner，禁止共享 buffer；参数/RNG/state 不变。
- **验证/门槛**：`torchrun --nproc_per_node=2 -m pytest -q testset/unit/test_6451993_patch_l.py --cov=testset.unit.async_allgather_patch_l --cov-branch --cov-report=term-missing --cov-fail-under=90`，branch≥50%；bitwise 检查 rank 拼接和 grad placement。收益下限 0，上限只取该 AllGather 的实测新增 overlap（不得取全部 82.483 ms exposed）；死锁、顺序、数值、显存、Step 退化即回滚。

### 8.11 P1/E1-M：gradient-ready-time 严格 bucket

- **落点/调用点**：拟新增 `develop/frontend/storage/testset/unit/ready_bucket_patch_m.py`；模型构造后注册 hook，backward 后、optimizer step 前 flush/assert。
- **完整实验代码**：

```python
import torch
import torch.distributed as dist

class ReadyTimeGradientBuckets:
    def __init__(self, named_parameters, bucket_bytes: int, mode: str, group=None) -> None:
        if not dist.is_initialized() or bucket_bytes <= 0:
            raise RuntimeError("M requires distributed runtime and positive bucket size")
        if mode not in ("all_reduce", "reduce_scatter"):
            raise RuntimeError("M mode must be all_reduce or reduce_scatter")
        self.group, self.world, self.mode = group, dist.get_world_size(group), mode
        self.bucket_bytes = bucket_bytes
        self.params = [(name, p) for name, p in named_parameters if p.requires_grad]
        if not self.params:
            raise RuntimeError("M received no trainable parameters")
        if len({p.device for _, p in self.params}) != 1 or len({p.dtype for _, p in self.params}) != 1:
            raise RuntimeError("M requires one device and dtype per runner")
        self.ready, self.seen, self.works = [], set(), []
        self.shards = {}
        self.handles = [p.register_hook(self._hook(name, p)) for name, p in self.params]

    def _hook(self, name, parameter):
        def record(grad):
            if grad.shape != parameter.shape or grad.dtype != parameter.dtype or grad.device != parameter.device:
                raise RuntimeError("M gradient contract changed")
            if name in self.seen:
                raise RuntimeError("M parameter became ready twice")
            self.seen.add(name)
            self.ready.append((name, grad))
            if sum(g.numel() * g.element_size() for _, g in self.ready) >= self.bucket_bytes:
                self._flush_bucket()
            return grad
        return record

    def _flush_bucket(self) -> None:
        if not self.ready:
            return
        names = tuple(name for name, _ in self.ready)
        grads = [grad for _, grad in self.ready]
        flat = torch._utils._flatten_dense_tensors(grads).contiguous()
        if self.mode == "all_reduce":
            work = dist.all_reduce(flat, group=self.group, async_op=True)
            self.works.append(("all_reduce", work, names, flat, grads, None))
        else:
            if flat.numel() % self.world != 0:
                raise RuntimeError("M reduce_scatter flat numel must be divisible by world size")
            shard = torch.empty(flat.numel() // self.world, dtype=flat.dtype, device=flat.device)
            work = dist.reduce_scatter_tensor(shard, flat, group=self.group, async_op=True)
            self.works.append(("reduce_scatter", work, names, flat, grads, shard))
        self.ready = []

    def finish_backward(self) -> None:
        self._flush_bucket()
        if self.seen != {name for name, _ in self.params}:
            raise RuntimeError("M ready set differs across the declared training step")
        for mode, work, names, flat, grads, shard in self.works:
            work.wait()
            if mode == "all_reduce":
                flat.div_(self.world)
                for grad, value in zip(grads, torch._utils._unflatten_dense_tensors(flat, grads)):
                    grad.copy_(value)
            else:
                if shard is None or shard.numel() * self.world != flat.numel():
                    raise RuntimeError("M reduce_scatter shard contract changed")
                # rank r owns contiguous interval [r*shard_numel:(r+1)*shard_numel).
                self.shards[names] = shard.div_(self.world)
        self.works.clear()
        self.seen.clear()
        if self.ready or self.works:
            raise RuntimeError("M step ended with a non-empty bucket")

    def take_shards(self):
        if self.mode != "reduce_scatter" or not self.shards:
            raise RuntimeError("M has no reduce_scatter shards for sharded optimizer")
        result, self.shards = self.shards, {}
        return result

    def assert_step_drained(self) -> None:
        if self.ready or self.works or self.seen or self.shards:
            raise RuntimeError("M step ended with unconsumed collective state")

def distributed_train_step(model, optimizer, batch, buckets):
    loss = model(batch).float().sum()
    loss.backward()
    buckets.finish_backward()
    if buckets.mode == "all_reduce":
        optimizer.step()
    else:
        optimizer.step_shards(buckets.take_shards())
    optimizer.zero_grad(set_to_none=True)
    buckets.assert_step_drained()
    return loss.detach()
```

- **调用迁移/契约**：`mode="all_reduce"` 保持完整梯度并供普通 optimizer 消费；`mode="reduce_scatter"` 显式调用 `reduce_scatter_tensor`，要求 flat numel 可被 world-size 整除，每 rank 仅拥有连续区间 `[rank*shard_numel:(rank+1)*shard_numel)`，且只能交给实现 `step_shards()` 的分片 optimizer。所有 rank 的参数名顺序、bucket_bytes、mode、dtype/device/group 必须相同；shape、dtype、device、平均语义与 optimizer state 分片映射不变。整除、ready 顺序、shard consumer 或 step 清空契约不满足均 raise，不回退到 all_reduce。
- **验证/门槛**：`torchrun --nproc_per_node=2 -m pytest -q testset/unit/test_6451993_patch_m.py --cov=testset.unit.ready_bucket_patch_m --cov-branch --cov-report=term-missing --cov-fail-under=90`，branch≥50%；分别覆盖 all_reduce 与 #17 ReduceScatter，断言 rank offset、shard shape/numel、参数映射、optimizer state、FP32 `1e-6/1e-5`、BF16 `2e-2/2e-2`，并测试 numel/world-size 不整除必 raise。收益下限 0，上限仅为对应 collective 实测新增 overlap；死锁、ready/hash 差异、数值、state 或 Step 退化即回滚 M。

### 8.12 P1/E1-N：mixer mean + redistribute 已有 custom op 全域命中

- **当前源码/定位**：`main_model.py:CvrTransformer.forward` L2177 产生 `combine_trans_out, mixer_all_outputs`，L2183-L2189 判定实验分支，真实调用在 L2190-L2193：`mixer_mean_redistribute(mixer_all_outputs, token_num=self.num_tokens)`。实现位于 `triton_ops/kernels/mixer_mean_redistribute.py:_dtensor_constants` L198-L212、`_validate_local_inputs` L235-L270、`_run_backward` L331-L399、`_mixer_mean_redistribute_op` L402-L410。该热点统一回链正式实验 P1/E1-N。
- **完整调用方替换代码**（独立可解析；import、function 和调用参数与真实源码一致）：

```python
from typing import Sequence
from torch import Tensor
from triton_ops.kernels.mixer_mean_redistribute import (
    can_use_mixer_mean_redistribute,
    mixer_mean_redistribute,
)

def run_mixer_mean_patch_n(
    mixer_all_outputs: Sequence[Tensor], token_num: int
) -> Tensor:
    if not can_use_mixer_mean_redistribute():
        raise RuntimeError("N custom op is unavailable for this runtime")
    if not mixer_all_outputs or token_num <= 0:
        raise RuntimeError("N requires non-empty mixer outputs and positive token_num")
    first = mixer_all_outputs[0]
    if any(x.dtype != first.dtype or x.device != first.device for x in mixer_all_outputs):
        raise RuntimeError("N requires identical dtype/device across mixer outputs")
    if any(x.ndim != first.ndim or not x.is_contiguous() for x in mixer_all_outputs):
        raise RuntimeError("N requires equal-rank contiguous local mixer outputs")
    result = mixer_mean_redistribute(mixer_all_outputs, token_num=token_num)
    if result.device != first.device:
        raise RuntimeError("N custom-op output device changed")
    return result

class CvrTransformerPatchN:
    def apply_mixer_mean(self, mixer_all_outputs: Sequence[Tensor]) -> Tensor:
        # Production replacement at main_model.py:L2190-L2193.
        return run_mixer_mean_patch_n(
            mixer_all_outputs,
            token_num=self.num_tokens,
        )
```

- **改前/改后接口**：输入是 `combine_trans` 返回的 tensor 列表 `mixer_all_outputs`，而非单个四维 tensor；调用方必须传关键字 `token_num=self.num_tokens`。实验路径用上述 helper 等价执行真实 `mixer_mean_redistribute(mixer_all_outputs, token_num=self.num_tokens)`，不保留 mean+`BwRedistribute` 兼容路径。custom op 的 `_validate_local_inputs` 继续负责 shape/stride/dtype/device/mesh/placement 门禁，`_run_backward` 负责 grad redistribute；输出供 L2205 按 `len(mixer_all_outputs)` chunk。每项 local/global shape、拼接 token 总数、DTensor `[Shard(1)]→[Shard(0)]` placement、mesh/group、mean 维及 backward placement 必须与基线一致，否则 raise；参数、RNG、optimizer state 不变。
- **验证/门槛**：拟新增 `test_6451993_patch_n.py`；`cd develop/frontend/storage && pytest -q testset/unit/test_6451993_patch_n.py --cov=testset.extracted.6451993.modelcode.triton_ops.kernels.mixer_mean_redistribute --cov-branch --cov-report=term-missing --cov-fail-under=90`，branch≥50%；断言 mock 调用恰为 `(mixer_all_outputs, token_num=self.num_tokens)`，逐 tensor forward/grad BF16 `atol=2e-2,rtol=2e-2`、FP16 `1e-2/1e-2`，并核对 chunk 数、global/local shape、mesh、placement、collective 序列。收益下限 0，上限仅为 L2183-L2205 原 mean+BwRedistribute 被唯一归因的 kernel/copy/collective exposed 子集；任何调用签名、误差、placement、显存或 Step 退化即回滚 N。

### 8.13 A–N 源码 Diff 与接入点逐项审计（最终分类依据）

判定口径：必须同时出现目标文件、逐字改前/改后上下文、`---/+++` 与 `@@` unified diff、真实调用方、imports/注册位置和可执行入口；孤立 helper、拟新增代码块或只有调用示意均记 **FAIL**，并降为实验性候选，不以文字保证替代证据。

| Patch | 目标文件/调用方 | imports/注册 | 运行入口 | 可直接应用到扫描源码的 Diff | 审计状态/最终分类 |
|---|---|---|---|---|---|
| A | `main_model.py:SequenceModule.forward` | 无新增 | 报告 7.1 测试入口 | 未给 `---/+++`、`@@` | **FAIL→候选** |
| B | `main_model.py:C2kTrans.forward` 及 callers | 无新增 | 报告 7.2 测试入口 | 未给完整 unified diff | **FAIL→候选** |
| C | `main_model.py` 三个 caller | helper 模块级 | 8.1 pytest | 有改前/改后，但无 unified diff | **FAIL→候选** |
| D | storage 新 Runner；无模型训练入口 | 新文件 imports 已列 | 8.2 pytest | 无 `/dev/null` 新文件 diff，且真实 step 接入未闭合 | **FAIL→候选** |
| E | storage optimizer harness；生产 optimizer caller 未定位 | 新文件 imports 已列 | 8.3 pytest | 无新文件/生产接入 unified diff | **FAIL→候选** |
| F | `pure_transformer.py:MLPMixer.forward` 首个 consumer | 复用 `torch`；无全局注册 | `python3 testset/unit/test_6451993_patch_f.py` | **PASS：8.4 含模型与 harness 两份 diff** | **Diff PASS，但 placement/热点映射未闭合→候选** |
| G | `sparse_moe.py:SparseMoe.forward` | 实现/注册缺失 | 暂定 pytest | 无可运行实现 diff | **FAIL→候选** |
| H | `pure_transformer.py:ResidualAttentionBlock.forward` | helper 新文件 | 8.6 pytest | 只有孤立 Function，无模型调用 unified diff | **FAIL→候选** |
| I | attention caller 未与真实 Q/K/V API 闭合 | helper 新文件 | 8.7 pytest | 无真实调用方 unified diff | **FAIL→候选** |
| J | FP32→BF16 caller 未唯一定位 | helper 新文件 | 8.8 pytest | 无真实 cast 与 optimizer-step 接入 diff | **FAIL→候选** |
| K | collective Python caller 未扫描到 | pack/unpack op 缺失 | 8.9 torchrun | 无可应用 diff | **FAIL→候选** |
| L | AllGather producer/consumer 未唯一定位 | helper 新文件 | 8.10 torchrun | 无真实 caller unified diff | **FAIL→候选** |
| M | 训练循环/分片 optimizer caller 未扫描到 | helper 新文件 | 8.11 torchrun | 无模型/optimizer 接入 unified diff | **FAIL→候选** |
| N | `main_model.py:CvrTransformer.forward` | custom op 已存在 | 8.12 pytest | 有调用代码但无 `---/+++`、`@@` | **FAIL→候选** |

**审计结论**：本轮不再保留“2 个 P0 + 12 个 P1 均为正式 Patch”的旧结论。F 是唯一具备可应用模型接入 diff 与 harness diff 的条目，但因 DTensor placement 与 trace 热点映射尚未闭合仍是实验性候选；其余 A–E/G–N 均因缺至少一项必需 Diff 证据降候选。补齐并实际 `git apply --check`/运行通过后，才可逐条恢复“正式 Patch”。

---

## 9. P2/E2 风险实验（不是正式 Patch）

### 9.1 checkpoint：移除全部 `ResidualAttentionBlock.forward` 实例的类级装饰器

- **分类**：P2/E2；精度/训练风险高。影响 activation 生命周期、重算计算、峰值显存、dropout/RNG 回放和可能的浮点路径。
- **源码版本与定位**：提取物 `pure_transformer.py` SHA-256 `1d683378...e02835e`；`ResidualAttentionBlock.forward`，L1026-L1053。Pattern 无 timeline；110.343 ms/step 是 10 个 block 的 inclusive 合计，不能作为此实验收益。
- **当前完整代码**：

```python
@checkpoint_forward
def forward(self, x, attn_mask=None):
    orig_x = x

    mixer_out = self.mixer(x, tao=self.tao)
    wandb_histogram(f"MLPMixer/mixer_out_{self.layer_idx}", mixer_out)

    if self.enable_mlp_mixer:
        x = self.ln_1(x + F.dropout(mixer_out, p=self.dropout))
    else:
        x = self.ln_1(mixer_out)

    mlp_out = self.mlp(x)
    wandb_histogram(f"MLPMixer/mlp_out_{self.layer_idx}", mlp_out)

    if self.enable_mlp_mixer:
        x = self.ln_2(x + F.dropout(mlp_out, p=self.dropout))
    else:
        x = self.ln_2(mlp_out)

    if self.enable_cross_conn:
        attn_out = self.attention(x, kv=orig_x, attn_mask=attn_mask, tao=self.tao)
        x = self.ln_3(x + attn_out)
    return x
```

- **实验性完整改后代码**：仅删除装饰器；函数签名、分支、dropout、直方图和返回值逐字保持。该修改会作用于全部该类实例，不能表述为“单 block policy”。

```python
def forward(self, x, attn_mask=None):
    orig_x = x

    mixer_out = self.mixer(x, tao=self.tao)
    wandb_histogram(f"MLPMixer/mixer_out_{self.layer_idx}", mixer_out)

    if self.enable_mlp_mixer:
        x = self.ln_1(x + F.dropout(mixer_out, p=self.dropout))
    else:
        x = self.ln_1(mixer_out)

    mlp_out = self.mlp(x)
    wandb_histogram(f"MLPMixer/mlp_out_{self.layer_idx}", mlp_out)

    if self.enable_mlp_mixer:
        x = self.ln_2(x + F.dropout(mlp_out, p=self.dropout))
    else:
        x = self.ln_2(mlp_out)

    if self.enable_cross_conn:
        attn_out = self.attention(x, kv=orig_x, attn_mask=attn_mask, tao=self.tao)
        x = self.ln_3(x + attn_out)
    return x
```

- **调用方/接口迁移**：调用方 `Transformer.forward` L1147 的 `resblock(x, attn_mask=attn_mask)` 不变；被调用的 mixer、MLP、attention 接口不变；参数名、state_dict key 和 optimizer state schema 不变。唯一变化是 checkpoint wrapper 被移除。
- **tensor 契约**：输入/输出 logical shape `[B,S,4096]` 不变；local shape、stride、dtype、device、DTensor placement 由原 body 保持；transpose/layout 与 alias 关系不变。变化仅为中间 activation 从 backward 重算前失效，改为保留到 backward 完成，峰值显存上升。Forward 运算 body 相同；Backward 图与 RNG 回放路径改变。
- **不保证等价项**：token/head/expert 顺序、mask、参数与 optimizer state schema 应不变；dropout RNG 消费/恢复、浮点执行时序、loss/output/grad bitwise 不保证。因此只能 P2/E2。
- **拟新增测试**：`testset/unit/test_6451993_checkpoint_risk_experiment.py`，复用 `dag_session_test_infra.py::run_dag_session` 和 `load_real_analyze_trace` 做结构/调用图断言，并做固定 seed 的 forward+backward、参数梯度、RNG state、optimizer state key/shape 对比。
- **仅相关 UT 命令**：`cd develop/frontend/storage && pytest -q testset/unit/test_6451993_checkpoint_risk_experiment.py --cov=testset.extracted.6451993.modelcode.pure_transformer --cov-branch --cov-report=term-missing --cov-fail-under=90`；另核验分支覆盖率 `branch >= 50%`，未达到即 FAIL。禁止运行全量 UT。
- **数值/质量阈值**：不得按 E0 验收。FP32 tensor `atol=1e-6, rtol=1e-5`，BF16 `atol=2e-2, rtol=2e-2`；逐 tensor 比较输出和梯度；loss 相对偏差 `<=1e-5`（FP32）/`<=2e-2`（BF16）；router index 必须逐元素相同；完整质量窗口主指标不得低于现网门槛。
- **性能上限**：`ceiling = 新 trace 中被移除 checkpoint wrapper 的实际重算 kernel union`，不是 110.343 ms inclusive timing，也不得用 Pattern 估算。记录 Step、目标调用栈 kernel union、peak allocated/reserved memory。
- **回滚**：任一 OOM、router index 变化、RNG state 不符合实验预期、输出/梯度/loss 超阈值、质量退化或 Step 无改善，立即恢复 `@checkpoint_forward`。

### 9.2 Sparse MoE top-k：`2→1`

- **分类**：P2/E2；改变 router 策略、expert 容量、输出、梯度、显存与收敛，不是正式 Patch。
- **源码版本与定位**：提取物 `main_model.py` SHA-256 `343262f3...6e6eafa`；`CVRModule.__init__` 中 `CvrTransformer(...)`，L2839-L2849；参数继续传至 `pure_transformer.py:MLPMixer.__init__` L776-L828 和 `SparseMoe(top_k=...)`。
- **当前完整调用代码**：

```python
self.mix_transformer = CvrTransformer(
    trans_dims=self.mix_trans_dim,
    combine_trans_layers=self.mix_layers,
    heads=64,
    scale_ratio=1,
    mixed_precision=True,
    enable_sparse_moe=True,
    sparse_moe_top_k=2,
    sparse_moe_expert_num=64 * 16,
    sparse_moe_scale_ratio=0.125
)
```

- **实验性完整改后代码**：

```python
self.mix_transformer = CvrTransformer(
    trans_dims=self.mix_trans_dim,
    combine_trans_layers=self.mix_layers,
    heads=64,
    scale_ratio=1,
    mixed_precision=True,
    enable_sparse_moe=True,
    sparse_moe_top_k=1,
    sparse_moe_expert_num=64 * 16,
    sparse_moe_scale_ratio=0.125
)
```

- **调用方/接口迁移**：构造接口、tensor shape 和 state_dict schema 不变；`sparse_moe_top_k` 经 CvrTransformer/Transformer/ResidualAttentionBlock/MLPMixer 传至 `SparseMoe.top_k`。不需要调用方签名迁移，但 router 输出第二专家被删除，属于语义变化。
- **tensor 契约**：模型输入/output logical shape、dtype/device/DTensor placement 不变；router index/prob local shape 的 K 维由 2 变 1；dispatch rows、expert token count、临时 buffer 生命周期和 gather/combine 工作量变化；参数及 optimizer state key/shape不变，但被路由 token 与梯度分布变化。Forward、Backward 均非等价。
- **等价性结论**：token 输入顺序保持，expert 选择集合、router index、expert token 顺序、输出、loss、grad 和 optimizer 更新不保证；mask/offset/position/attn_arg 不应变化，必须测试；RNG 消费也可能因工作量变化而变化。
- **拟新增测试**：`testset/unit/test_6451993_topk_risk_experiment.py`，复用 `run_dag_session`/`load_real_analyze_trace`，检查参数/state schema、router index/prob shape、expert token count、forward/backward 和固定 seed 可复现。
- **仅相关 UT 命令**：`cd develop/frontend/storage && pytest -q testset/unit/test_6451993_topk_risk_experiment.py --cov=testset.extracted.6451993.modelcode.sparse_moe --cov-branch --cov-report=term-missing --cov-fail-under=90`；分支覆盖率必须 `>=50%`，禁止全量 UT。
- **数值/质量阈值**：因目标本身非等价，不用 atol/rtol 宣称等价；只检查同配置重复运行 bitwise/允许的 deterministic threshold。必须报告 loss、逐参数 grad norm、router index/prob、entropy、expert_token_cnt P50/P95/max、AUC及业务主指标；质量门槛与现网验收完全一致。
- **性能上限**：全局粗上限仅为当前 MoE Grouped GEMM 234.111 + gather/combine 21.718 = 255.829 ms/step；它不是预期收益。正式实验上限为 `top_k=2 路径中第二 expert 直接 kernel union + 对应 dispatch/gather exposed`，由新 trace 调用栈测得。
- **回滚**：主质量指标下降、router collapse/负载恶化、loss/grad 异常、显存或 Step 退化、确定性不满足时恢复 `sparse_moe_top_k=2`。

---

## 10. 前置调查任务（不是建议或 Patch）

| ID | 对象与准确位置 | 只读证据采集 | 必须打印/记录字段 | 正式 Patch 准入条件 |
|---|---|---|---|---|
| I-01 | packed sequence；`main_model.py:SequenceModule.forward` L1431-L1442、L1705-L1728；`main_model.py:C2kTrans.forward` L1029-L1105 的 prepared 参数消费路径 | 在 pack 入口和两个 consumer 加临时只读 instrumentation 后运行定点 mock/trace；本报告不改源码 | `id/storage_offset/data_ptr`（可用时）、logical/local shape、stride、dtype、device、mask/attn_arg hash、调用栈、每 Step 次数、pack kernel/cpu duration | 证明同一 `(data, mask, attn_arg)` 在同 Step 重复 pack，且缓存生命周期/失效条件可定义；否则关闭为“无 patch” |
| I-02 | transpose→reshape pack；`pure_transformer.py:MLPMixer.separate_heads/forward` L862-L900 | 定点打印 transpose 前后和 reshape 后元数据，并用 profiler 定位 copy/clone kernel correlation | shape、stride、storage_offset、contiguous、alias/data_ptr、dtype、DTensor placement、consumer、Fwd/Bwd copy duration | 提供真实 layout-aware fused op 的完整 forward+backward、调用方和测试；证明 `[B,H,T,D']` 中每 head 的 `T*D'` consumer 可按原顺序读取。现布局中 transpose 后 stride 使 T 与 H 物理交错，reshape 为每个 head 打包连续 `T*D'`，不能直接删除 |
| I-07 | AllToAll/AllGather；`sharding_plan.py` 模块级 `combine_model_fwd_resharding_plan` L101-L110；`main_model.py:CvrTransformer.forward` L2189-L2205 与第 5 章通信类型 | correlation→Python callsite→DTensor placement 全链采集 | collective id、调用栈、前后 op、global/local shape、mesh、placement、方向、count/union/exposed、替代通信 | 单一转换唯一归因；collective 语义、token/head/expert 顺序和 grad placement证明；不存在隐式补偿 reshard；上限仅取该调用 exposed |
| I-08 | BF16 contiguous/cast 与首个消费者融合；`main_model.py:CvrTransformer.forward` L2194-L2205；框架生成的 #6/#19 kernel 当前无唯一 Python callsite | 用 correlation/call stack 将 #6 contiguous/copy 与 #19 FP32→BF16 cast 逐一绑定到 producer、collective pack、首个 consumer 和 stream | bytes read/write、shape/stride/contiguous、dtype、storage/alias、stream、producer/consumer、collective correlation、kernel duration/gap、Fwd/Bwd phase | 仅在同 stream、无 collective/graph-break 边界、consumer 支持原 stride/dtype且有完整 forward/backward 实现时形成 P0/P1；收益上限为被确认消除的 copy/cast kernel union 加可量化相邻 gap，不得使用 44.117 ms 总和替代调用点子集 |
| I-09 | optimizer 小参数组/多张量 launch；Top Kernel #4 Adamom 入口未在提取源码中找到；已扫描 `modelcode/*.py` 的 optimizer 构造/`step`/`param_groups` | 从训练框架 optimizer 构造和 step 的 Code Location、调用栈及参数组 dump 反查真实入口 | optimizer class、构造/step 文件函数行号、group 数、每组 tensor 数/numel/dtype/device、foreach/fused bucket、121 次 launch 的 tensor-list 长度与 duration/gap | 只有定位真实构造与 step、证明可合并组拥有相同超参/clip/state/更新顺序，并给出完整迁移与 state_dict 兼容方案，才形成 P0/P1；当前禁止写“减少小参数分组” |
| I-10 | `torch.compile` 图内短 elementwise/cast/transpose/clone；`main_model.py:NFMModule.forward` L629-L680、`SequenceModule.forward` L1431-L1730、`CvrTransformer.forward` L2144-L2214；`pure_transformer.py:MLPMixer.forward` L868-L903 | 导出 Dynamo explain/FX graph、Inductor generated code 与 graph-break reason，并按 correlation 标记相邻 kernel 是否同 graph/stream | graph id、break 文件/行/原因、stream、op sequence、shape/stride/dtype/placement、每 kernel duration、相邻 launch gap、编译前后 kernel count | 同图同 stream且无 collective/alias/mutation 阻断，完整 Inductor/custom-op patch 能同时覆盖 Fwd/Bwd，才可升 P1；不同 stream、collective 边界或 graph break 明确记“非候选” |
| I-11 | MoE grouped/persistent grouped GEMM；`sparse_moe.py:SparseMoe.forward` L354-L389、`PerTokenSwiGLUFFN.forward` L238-L244，Top Kernel #1/#2/#3/#5/#11/#17 | dump 每次 problem size array 与 router `expert_token_cnt`，关联六个 CUTLASS pointer-array grouped GEMM 变体 | `(M_e,N,K)`、空/小 expert 数、tile/stage、CTA、调用次数、duration、router 顺序、workspace、persistent 可复用条件 | 当前已是 grouped GEMM，不能把“改 grouped”当建议；只有 persistent 方案完整实现、证明 problem array/地址生命周期稳定、Fwd/Bwd 及 expert 顺序等价并以六个目标 kernel 子集计上限时升 P1 |
| I-12 | collective 前后 pack/unpack 合并；`sharding_plan.py:combine_model_fwd_resharding_plan` L101-L110、`main_model.py:CvrTransformer.forward` L2189-L2205 | 联合采集 pack/copy 与 NCCL correlation、stream 和 placement 转换 | pack/unpack bytes、shape/stride/layout/dtype、stream、event dependency、mesh/placement、collective input lifetime、copy/collective union | 仅在 pack 与 collective/consumer 的 stream、placement、alias、生命周期契约闭合且无替代 copy 时形成 P1；收益只取被消除 copy union/可量化 gap，不取总通信时间 |
| I-13 | CUDA Graph；训练入口/optimizer step 未包含在当前提取源码，`main_model.py:SequenceModule.forward` L1431-L1730 含动态 packed/segment 路径 | 采集连续 Step 的 shape、storage address、allocation、graph break、collective sequence 和 RNG state | 每输入/中间 tensor shape/stride/address、动态长度、allocation count、CPU launch 事件、NCCL 序列、RNG capture 安全性 | 仅当 shape 与地址稳定、无 graph break/非法动态分配、collective 与 RNG capture 安全，且可给完整 capture/replay 生命周期代码时升 P1；否则保持调查 |
| I-14 | 小矩阵/大 GEMM/gather 的局部实现选择；`model_common.py:matrix_similarity` L1418-L1423；`sparse_moe.py:SparseMoe.forward` L354-L389 的 gather/combine；Dense/NVJET launcher 当前缺源码 | 绑定 #14/#16/#20 NVJET、#15/#18 gather 与真实调用点并导出 autotune 候选 | M/N/K、layout、dtype、tile、调用次数、duration、gather index locality、编译/运行配置 | 仅以目标调用点 kernel 子集实测选择模板或布局；完整 launcher/调用方 patch、Fwd/Bwd 数值与回滚齐全后升 P1；不得用“按 shape autotune/联动布局”作为无编号建议 |

调查命令统一只跑相关路径，示例：`cd develop/frontend/storage && pytest -q testset/unit/test_6451993_layout_contract.py testset/unit/test_6451993_collective_attribution.py --cov=testset.extracted.6451993.modelcode --cov-branch --cov-report=term-missing --cov-fail-under=90`，并单独核验 branch coverage `>=50%`。这些是**拟新增**测试路径，本步骤未创建测试文件；若文件尚不存在，命令不得被标记为已执行。

---

## 11. 短 Kernel / 短 Launch 优化专项

### 11.1 可量化事实与不可量化边界

数据来自 `output/kernel_top20_6451993.json` 的 20 条打印记录、`timing_panel_6451993_fixed.json` 的 `_timing_pipeline_stats` 与 `perf_metrics_6451993.json` 的五 Step 统计：

| 指标 | 打印结果 | 口径 |
|---|---:|---|
| 全量 kernel 事件 | 49,827/5 Step = 9,965.4 events/step | timing panel 打印总数；不能直接等同独立 CPU launch 次数 |
| 全量 kernel duration 非 union 和 | 5,721,832.903 μs/5 Step = 1,144.367 ms/step | timing panel `total_kernel_dur_us`；多 stream 可重叠，不能当墙钟 |
| Top20 kernel 累计 | 493.218 ms/step，1113.8 calls/step | 仅 Top20 聚合，不代表全部 kernel |
| Top20 单次均值范围 | 19.768–7526.173 μs | `dur_per_step/count_per_step` |
| Top20 中均值 `<10 μs` | 0 类，0 calls，0 ms | 不能据此声称 Top20 有亚 10 μs kernel |
| Top20 中均值 `<50 μs` | 1 类，454 calls/step，8.974 ms/step | 仅 #19 FP32→BF16 vectorized copy，均值 19.768 μs |
| #4 Adamom | 121 calls/step，51.760 ms/step，均值 427.771 μs | 高频但不属于 `<50 μs`；参数组信息未知 |
| #6 BF16 direct copy | 144 calls/step，35.143 ms/step，均值 244.049 μs | 高频 copy，但不是短于 50 μs |
| kernel-idle/non-kernel gap | 72.185 ms/step | `host Step - union(compute,NCCL)`；包含非 kernel engine 活动，不能等同 CPU launch overhead |
| 主 stream kernel 数示例 | Step#1505 为 8444 | 来自 `kernel_counts_by_stream[7]`，只证明发射数量大 |

**CPU launch 开销不可量化**：当前打印数据没有完成 CPU runtime launch event→GPU kernel 的逐事件关联，也没有相邻 kernel gap 分布；因此不能假设“每次 launch 固定若干微秒”，不能把 72.185 ms gap 全算作 launch overhead。当前唯一可发布的短-kernel duration ceiling 是 #19 的 8.974 ms/step；只有 I-08/I-10 证明其中具体调用可合并后，才取对应子集 duration，加上实际测得且可消除的相邻 gap。

### 11.2 已识别类别、具体手段与分层处置

| 类别 | 当前证据 | 具体候选手段 | 处置 |
|---|---|---|---|
| #19 高频 FP32→BF16 cast | 454 calls、8.974 ms、19.768 μs/次；调用点待运行确认 | P1/E1-J 以参数 `_version` 键控 BF16 cache，optimizer 更新强制失效；不得跨 collective/stream 复用 | Patch J（4.3）+ I-08/I-10 证伪 |
| transpose/clone/elementwise 链 | 源码存在 stack/sum、tile/where、transpose/reshape；Pattern 无 timing | A/B/C 处理确定性局部语义；P1/E1-F 用 layout-aware custom autograd 接口验证 producer 目标布局直写 | Patch A/B/C/F（4.3、7.1、7.2、8.1） |
| optimizer 多张量更新 | #4 为 121 calls、51.760 ms；生产构造入口缺失 | 严格同配置合组 Harness 做一步 update 与 launch A/B，再由 I-09 反查生产接入点 | Patch E（8.3）+ I-09 |
| MoE expert GEMM/gather | grouped 六项 234.111 ms，gather/combine 21.718 ms | P1/E1-G 显式 problem array、tile/persistent dispatch、combine order 和 inverse-index backward | Patch G（8.5，候选）+ I-11/I-14 证伪 |
| stack/sum、mask/tile、重复赋值 | 源码位置和不变量已闭合；局部 timing 未闭合 | 严格逐项 add、broadcast mask 接口迁移、删除确定性死写 | Patch A/B/C（7.1、7.2、8.1）；收益下限 0 |
| collective pack/unpack | 通信类型已量化，copy 与具体 collective 待绑定 | P1/E1-K 成对 autograd pack+AllToAll；L 双 buffer AllGather；M ready-time bucket | Patch K/L/M（8.9/8.10/8.11，候选）+ I-07/I-12 |
| CUDA Graph | 模型包无训练入口；动态 packed/segment 需运行时验证 | storage strict Runner 固定 static buffer 并 capture forward/backward/optimizer，契约变化直接 raise | Patch D（8.2）+ I-13 |
| NVJET/FlashAttention/LN 实现选择 | Top Kernel 有 shape/dtype/时长，部分调用点为高置信推导 | F 验证目标布局，H 验证 fused residual-LN autograd，I 按 packed length 固定 attention config | Patch F/H/I（8.4/8.6/8.7，候选）+ I-14 |

### 11.3 Diff 门禁后的候选判定

短 Launch 专项仍回链 A–N，但第 8.13 节审计是最终状态：F 有模型接入与 harness unified diff，仍因 placement/热点映射未闭合为候选；其余条目缺完整可应用 Diff，均不得称正式 Patch。

---

## 12. 完整度检查与验收表

验收以第 8.13 节逐项审计为单一事实源；“有 Python 代码块”不等于“可应用 Patch”。

| 检查项 | 结果 | 证据 |
|---|---|---|
| A–N 目标文件、调用方、imports/注册、运行入口逐项审计 | PASS（审计已执行） | 第 8.13 节 14/14 行 |
| 可直接应用到扫描源码的 Diff | **仅 F 的模型接入与 harness Diff 通过** | 第 8.4 节含 `---/+++` 与 `@@`；A–E/G–N 缺项均明确 FAIL |
| F 可直接称正式 Patch | **FAIL** | 虽有 Diff，但 DTensor placement 与 #6/#14/#16/#20 热点唯一映射未闭合，故降候选 |
| G/K 是否仍散落完整正文 | PASS | 第 4.3/5.4 只留摘要跳转；完整候选条目位于 8.5/8.9 |
| 第 8 章 C–N 顺序 | PASS | 8.1 C、8.2 D、8.3 E、8.4 F、8.5 G、8.6 H、8.7 I、8.8 J、8.9 K、8.10 L、8.11 M、8.12 N |
| 孤立接口骨架是否仍计正式 Patch | PASS | 不再计正式；A–N 最终均按第 8.13 审计降候选 |
| P2 是否仍独立 | PASS | 第 9 章 checkpoint、top-k/router |
| Markdown fence | PASS | 自动计数结果见 STEP47 完成记录 |
| 报告最终状态 | **NEEDS_DIFF_CLOSURE** | 不自报正式 Patch PASS；需逐条补 Diff 并 `git apply --check`/实跑 |

---

## 13. 结论与交付判定

1. **结构已统一**：A/B 位于第 7 章，C–N 按字母连续位于第 8.1–8.12；第 4.3/5.4 只保留热点摘要与跳转。
2. **F 已重写而非原样搬运**：基于真实 `MLPMixer.separate_heads → self.attn / reshape → self.ln → self.mlp` 链，给出逐字上下文、模型接入 unified diff、storage harness diff、shape/stride/dtype/device 与 backward 契约及运行入口。
3. **F 最终仍为实验性候选**：其 Diff 可应用，但 DTensor placement 和 trace 热点映射未闭合，不能冒充正式 P1。
4. **A–N Diff 审计结论**：F 的 Diff 列通过；A–E/G–N 缺完整可应用 Diff，全部按硬门禁降候选。代码块、接口描述或测试命令不能替代 unified diff。
5. **P2/E2**：checkpoint 与 top-k/router 仍只作为隔离风险实验。
6. **最终状态**：`NEEDS_DIFF_CLOSURE`；本报告不再宣称存在已通过 Diff 门禁的正式 Patch。
