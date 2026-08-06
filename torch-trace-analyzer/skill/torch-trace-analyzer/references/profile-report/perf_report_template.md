# 模型 {MODEL_ID} 性能分析报告

---

<!-- ============================================================
【内部检查与原始证据区】以下内容仅供分析人员使用，不得出现在对外报告中
============================================================

## [内部] A. 前置核验与分析元数据

| 检查项 | 记录 | 通过标准 |
|---|---|---|
| Trace 文件 | `{TRACE_PATH}` | 文件可读，包含原始 `traceEvents` |
| Trace 进程/线程 | pid=`{PID}`；tid=`{TIDS}` | 与目标训练任务一致，无跨模型混入 |
| ProfilerStep 边界 | `{STEP_BOUNDARY_EVIDENCE}` | 每个 Step 的起止时间和裁剪规则已核验 |
| 模型源码版本（内部元数据） | commit=`{CODE_COMMIT}` | 仅用于复现和定位，不构成完成状态门禁 |
| Trace 关联版本（内部元数据） | commit=`{TRACE_COMMIT}` | 用户提供的 Trace↔源码关联是任务前提；版本信息不影响 `COMPLETE` |
| 分析工具版本 | commit=`{ANALYZER_COMMIT}` | 可复现 |
| 并行配置 | TP=`{TP}` / DP=`{DP}` / PP=`{PP}` / EP=`{EP}` / rank=`{RANK}` | 与 Trace 和源码配置一致 |
| 增强 Trace | `{YES_NO}` | Code Location、stack_traces 等字段完整性已记录 |

## [内部] B. 生产 timing 流程与局部数据检查

> Pattern timing 首选 production 流程：`kernel/cpu attribution` → `rollup_instance_timing` → `attach_timing_to_dag_groups` 为 eligible 非 synthetic group 写入 `direct_timing` → 沿 DAG 子 group 关系做 post-order rollup。若 Timeline 不含 Kernel→DAG leaf 的确定映射，不要求补造不存在的数据，也不得将 Module 时间分摊或继承给 Pattern。此时对外 Pattern 采用**结构优先级（不是 Pattern timing）**：先按所属 Module 的已核验热点时间降序，再在同一 Module 内按 Pattern base name 去除 `#N` 后的 DAG 实例数降序；表中只展示 Module 排序上下文、Pattern 实例数、源码结构/算子链，并将 timeline 具体 Kernel 信息标为“无”。分析侧原子 fact rollup 只能放在内部证据区，不得作为公开 Pattern 排名、占比或优化收益依据。AllToAll / AllGather 仅是通信类型，不得登记为 Pattern。

| 章节 | 必需数据 | 强制处理 | 内部核验结果 | 状态 |
|---|---|---|---|---|
| Step | 原始 Trace + ProfilerStep/训练 Step 边界 | 按每个 Step 边界裁剪事件后求多步平均 | `{STEP_GATE_EVIDENCE}` | PASS / LOCAL_MISSING |
| Module 类级行为 | 可唯一定位的 class/source | class/source 唯一即分析计算结构、执行路径和源码；只记录可优化证据，不在本节形成修改建议 | `{MODULE_CLASS_EVIDENCE}` | PASS / LOCAL_UNEXPLAINED |
| Module 实例 timing | DAG groups + attribution + attach + rollup 结果 | 仅在 timing 可合法分配时填写；production 多个等长候选均分属于合法实例分配 | `{MODULE_TIMING_EVIDENCE}` | PASS / LOCAL_TIMING_UNAVAILABLE |
| Pattern | DAG JSON + production rollup；若无细粒度 timeline 则使用 DAG 结构统计 | 有 production timing 时按全部实例总时长排序；否则按 Module 热点时间、同 Module 内 Pattern base name 实例数两级排序，不输出 Pattern ms/Step 或占比 | `{PATTERN_GATE_EVIDENCE}` | PASS / STRUCTURAL_ONLY / LOCAL_MISSING / ROLLUP_FAILED |
| Kernel | 原始 Trace + Kernel 聚合 + class/候选 class 映射 | 只要可确定 Module class 或候选 class，就分析类代码及 Kernel 在其中的位置；实例不唯一只标 timing 分配不确定 | `{KERNEL_GATE_EVIDENCE}` | PASS / LOCAL_UNEXPLAINED |
| 通信 | 原始 Trace + stream 分类 + 通信触发源码 | 由 stream/tid 分类并核验 collective、并行维度和触发代码 | `{COMM_GATE_EVIDENCE}` | PASS / LOCAL_MISSING |

### [内部] 完成状态决策

- 若 Step、Module、Kernel、通信证据足以形成完整分析，且 Pattern 在无细粒度 timeline 时已按确定性结构优先级完成源码与算子链分析，使用 `COMPLETE`。
- Pattern 缺少 Kernel→DAG leaf 细粒度映射本身不阻碍交付；必须在数据口径中明确“Pattern 无细粒度 timeline”，并禁止输出 Pattern ms/Step、占比或基于其计算收益。
- 实例不唯一不阻止类级代码行为、计算结构和源码分析，也不导致报告整体 `INCOMPLETE`；修改项仍须单独通过第 6 章 patch 准入。
- production 多等长候选均分是合法实例分配；须披露均分规则，但不能据此降低完成状态。
- 缺少 commit 或版本信息差异不影响上述完成状态；版本只保留为内部复现元数据。
- 只有真实数据缺失导致核心报告无法形成时，才使用 `INCOMPLETE: {LOCAL_REASON}`。

## [内部] C. timing、编译与归因数据质量检查

> 本节检查数据质量。Module 必须同时公开 runtime records 的 matched/miss 分解，以及 attach + rollup 后 Module groups 的覆盖（`rolled/total` 与百分比）；不得把 runtime record 匹配率冒充 group 覆盖率。编译和归因覆盖率不得作为降低类级源码解释深度的理由。

| 指标 | 值 | 处置 |
|---|---|---|
| DAG group 总数 | `{N}` | 与 DAG JSON 核对 |
| `compute_timing_coverage_summary` | `{SUMMARY}` | 输入必须是 `attach_timing_to_dag_groups` 且 post-order rollup 完成后的 DAG groups |
| direct timing group | `{N} ({PCT}%)` | 仅统计 eligible 非 synthetic group；Pattern 无 direct timing 正常 |
| rollup-only group | `{N} ({PCT}%)` | 核验沿 `children_group_ids` 的 post-order 聚合路径，避免重复计时 |
| 无 timing group | `{N} ({PCT}%)` | 只对真实缺数据或 rollup 失败的局部项提示 |
| 多等长候选分配 | `{N}` | production 均分为合法实例分配，记录候选和分配规则 |
| torch.compile/图编译对 Module 边界记录的影响 | `{INTERNAL_FINDING}` | 仅用于定位采集问题，不得削弱类级分析 |
| call_chain / Code Location 覆盖率 | `{PCT}%` | 实例不唯一时保留 class/候选 class 级分析，并标注 timing 分配不确定 |
| 行号越界或文件不存在 | `{N}` | 尽可能定位 class 实现；确实无法解释时仅提示对应局部项 |

## [内部] D. Kernel 原始符号、时间定位与 demangle 证据

> 下表可包含完整原始技术标识；这些字段严禁复制到对外正文。对外只使用短而精确的语义名称，例如“MoE BF16 Grouped GEMM（64×128 tile）”“AdamW 参数更新”“Triton RMSNorm 反向”。

| 对外语义名 | 完整 mangled name | 完整 demangled name | SQL ID | stream | ts | correlation | demangle 命令 | 原始 Trace/聚合证据 |
|---|---|---|---|---|---|---|---|---|
| `{SHORT_SEMANTIC_NAME}` | `{FULL_MANGLED_NAME}` | `{FULL_DEMANGLED_NAME}` | `{SQL_ID}` | `{STREAM}` | `{TS}` | `{CORRELATION_ID}` | `{DEMANGLE_COMMAND}` | `{RAW_EVIDENCE}` |

### [内部] demangle 到实现分析的强制转译

每个 Top Kernel 必须把符号信息转译为正文中的实现行为，不得停留在符号释义：

- **CUTLASS**：逐项核验并解释 Grouped GEMM/problem array、SM 架构与 TMA、warp specialization、MMA tile、输入 dtype 与 accumulator、A/B/C layout、pipeline stage、epilogue、不同 shape/tile 变体的实际含义；必须回连模型源码中的 dispatch、`expert_token_cnt`、`MoeLinear`、SwiGLU、combine 等真实调用与张量流。
- **PyTorch / ATen**：解释算子语义、dispatch 路径、输入输出 tensor、shape/dtype、Fwd/Bwd 角色及热点原因，并回连真实模型调用代码。
- **Triton**：解释 program/grid、block/tile、数据搬运、融合算子、输入输出、shape/dtype、Fwd/Bwd 角色及热点原因，并回连 launcher/wrapper 与模型调用代码。
- **NVJet / FasterTransformer / custom op**：解释注册与 wrapper、核心 CUDA 实现行为、输入输出、shape、模型中的调用链及热点原因；不得以二进制实现不可见结束分析。
- **optimizer kernel**：解释参数/梯度/状态读写、融合更新公式、dtype、tensor 分组、带宽或 launch 特征及其训练代码入口。
- 若具体实例不能唯一确定：列出 Module class 或候选 class，继续完成类代码行为、Kernel 位置和热点原因分析；仅把实例 timing 标为“分配不确定”。只有连 class 实现都无法解释时，才把对应 Kernel 作为局部待补证据项。

## [内部] E. 待核验项

| ID | 对象 | 已有证据 | 候选源码关系 | 缺失证据 | 关闭条件 | 状态 |
|---|---|---|---|---|---|---|
| `{VERIFY_ID}` | `{GROUP_OR_KERNEL}` | `{EVIDENCE}` | `{CANDIDATES}` | `{MISSING_EVIDENCE}` | `{CLOSE_CRITERIA}` | OPEN / CLOSED |

## [内部] F. 源码解释完成度检查

| 对象 | 必须完成的解释 | 结果 |
|---|---|---|
| 每个 Module Top-N 类级行为 | class/source 唯一时必须完成正常代码行为、输入、计算结构、执行路径、输出、子模块/custom op/collective、Pattern/Kernel 关联、真实文件/行号；父 Module 源码摘录最多 70 行，超长时用白话完整解释计算结构 | PASS / LOCAL_UNEXPLAINED |
| 每个 Module Top-N 实例 timing | timing 可合法分配时填写实例 groups、Fwd/Bwd 成本；多等长候选均分合法，实例不唯一只标 timing 不确定，不阻断类级分析 | PASS / LOCAL_TIMING_UNAVAILABLE |
| 每个 Pattern Top-N | 实际结构、去 `#N` 后的 base name 与真实实例数、所属已核验热点 Module、源码表达式、输入输出、算子组合、成员节点/子 Pattern、可能的 Kernel 实现关系与结构价值；无细粒度 timeline 时明确写“timeline 具体 Kernel 信息：无”，且不输出 Pattern timing；不得在本节形成修改建议 | PASS / LOCAL_MISSING |
| 每个 Kernel Top-N | 短语义名、实现行为、可确定的 Module class/候选 class 代码、Kernel 在类计算中的位置、输入输出、shape/dtype、Fwd/Bwd、热点原因、Module/Pattern 关系，以及所属 Module 的计算行为；若第 2 章已完整解释，必须给出章节跳转 | PASS / LOCAL_UNEXPLAINED |
| 每种通信类型 | stream 分类证据、collective 语义、调用次数、并行维度、触发源码、输入输出与 overlap | PASS / LOCAL_MISSING |
| 每条正式 P0/P1 patch | Module→Pattern→Kernel→源码证据链完整，且满足第 6 章全部 patch 字段；任一字段缺失即不得进入正式建议 | PASS / FAIL |
| 每条实验性候选 Patch | 能写出不违反已知契约的完整可运行代码；明确未验证项、证据等级、验证闭环及升格/退回条件 | PASS / FAIL |
| 每条 P2 风险实验 | 明确 E2 风险、变量、完整代码/配置修改、隔离验证与回滚；不得进入主要优化建议 | PASS / FAIL |
| 每条前置调查任务 | 仅限无法写出不违反 shape/layout/placement/API 契约代码的阻断项；关闭条件必须能解除代码构造阻断或证明不可行 | PASS / FAIL |

局部分析项不通过时只标记该项及影响范围；不得因实例不唯一、缺少 commit、缺少性能 timing 或缺少细粒度 timeline 把整份报告判为 `INCOMPLETE`。修改项按第 6 章分层：完整等价与源码契约闭合进入正式 Patch；可写完整运行代码但等价或性能映射待验证进入实验性候选 Patch；改变训练目标、数值、收敛或并行策略进入 P2/E2；只有无法写出不违反已知 shape/layout/placement/API 契约的代码时才进入前置调查。禁止 fallback、软化错误、静默 `continue`/`skip`、兼容 shim 或占位建议。

## [内部] G. 报告完成后强制自查

| 自查项 | 通过标准 | 结果 |
|---|---|---|
| 对外无编译/Module 边界免责声明 | `torch.compile`、编译边界覆盖率等只存在本 HTML comment | PASS / FAIL |
| 对外无完整长符号和内部定位字段 | 完整 mangled/demangled、SQL ID、stream、ts、correlation、demangle 命令只存在本 HTML comment | PASS / FAIL |
| Pattern 路径标注正确 | 无细粒度 timeline 时使用确定性结构优先级，不以分析侧原子 fact rollup 作为公开排名，不输出 Pattern ms/Step、占比或收益 | PASS / FAIL |
| Pattern synthetic 语义正确 | base name 通过去除 `#N` 聚合；实例数来自真实 DAG，所属 Module 由 DAG 祖先关系确定，Module 时间仅作排序上下文、不分摊给 Pattern | PASS / FAIL |
| Module 固定结构完整 | 输入/计算结构/路径/输出、子模块/custom op/collective、Fwd/Bwd、Pattern/Kernel、源码位置和必要核心代码齐全；过长父 Module 已限制摘录长度并完成白话行为解释 | PASS / FAIL |
| Pattern 固定结构完整 | 实际结构、实例数、源码表达式、I/O、成员节点/子 Pattern、算子组合、所属 Module、可能的 Kernel 实现关系与结构价值齐全；无细粒度 timeline 时已明确标“无”；未在本节夹带修改建议 | PASS / FAIL |
| Kernel 固定结构完整 | 实现行为、调用代码、I/O、shape/dtype、热点原因、跨层关系、所属 Module 计算行为或明确章节跳转齐全 | PASS / FAIL |
| Pattern 排序/命名正确 | 无细粒度 timeline 时先按所属 Module 已核验热点时间降序，同一 Module 内再按去 `#N` 后的实例数降序；明确标注“结构优先级，不是 Pattern timing” | PASS / FAIL |
| Module 覆盖披露完整 | 同时报告 runtime records matched/miss 与 attach+rollup 后 Module groups rolled/total，未混淆两种分母 | PASS / FAIL |
| 类级与实例级解耦 | class/source 唯一即完成类级分析；实例 timing 仅在可合法分配时填写，多等长候选均分视为合法 | PASS / FAIL |
| Kernel 候选类继续分析 | 能确定 Module class/候选 class 时已分析类代码和 Kernel 位置；实例不唯一仅标 timing 分配不确定 | PASS / FAIL |
| 对外 Kernel 名称简短准确 | 只保留可读语义名和必要变体信息 | PASS / FAIL |
| 源码证据真实 | 所有文件/函数/当前扫描版本行号存在且相互一致；当前代码逐字、完整摘录，无伪代码冒充当前代码 | PASS / FAIL |
| 正式 patch 字段完整 | 每条正式 P0/P1 均含优先级/等价等级、精度风险、准确定位、当前完整代码、完整改后代码、调用方/接口迁移、tensor 契约、等价不变量、可运行验证脚本与命令、数值阈值、性能指标、收益上限、回滚条件 | PASS / FAIL |
| 实验候选字段完整 | 每条候选均含 Candidate ID、瓶颈/标准手段、优先级/预计等价等级、证据等级、完整当前/改后代码、调用迁移、契约、验证、收益上限、回滚、升格/退回条件，并明确“未验证，需实跑收集证据” | PASS / FAIL |
| 修改项均有完整代码方案 | 任何检查项、建议项、优化项、修复项或实验候选，只要涉及代码/配置/接口/布局/通信调度/训练策略修改，均给出可直接替换的完整方案；只有文字方向、调查措辞或伪概念即硬 FAIL | PASS / FAIL |
| 建议分层合法 | 正式 P0/P1、实验性候选 Patch、P2 风险实验、前置调查严格分区；可执行实验项未降级为纯调查，P2 未成为主要优化建议 | PASS / FAIL |
| 等价等级合法 | P0/E0 保证数学、输出、梯度、训练语义不变；P1/E1 仅浮点路径可变；checkpoint/top-k/router/并行/分片等风险变更只能为 P2/E2 | PASS / FAIL |
| Layout/placement 证明完整 | 删除或改变 transpose/reshape/redistribute 前，已逐项证明 shape/stride/layout/dtype/device/placement/alias/生命周期契约 | PASS / FAIL |
| 收益归因合法 | 未用父 Module inclusive timing 推导局部 patch 收益；收益上限来自 patch 直接可消除成本并列明假设；候选缺 timing 时明确“不承诺收益”，但不因此省略代码 | PASS / FAIL |
| Forward/Backward 覆盖完整 | patch 及验证覆盖 Forward、Backward、梯度、RNG、参数和 optimizer state；缺任一项即硬 FAIL | PASS / FAIL |
| 无降级逃逸路径 | 无 fallback、静默 `continue`/`skip`、软化错误或兼容 shim；性能 timing 缺失未被用作不给实验候选代码的理由 | PASS / FAIL |
| 禁用空泛措辞 | “复用/删 clone/删通信/让后续消费/融合”等方向均已展开为准确定位、完整代码、契约与验证；任何一句话方向即硬 FAIL | PASS / FAIL |
| 无 timeline 推导完成 | 已用 kernel name/dtype/shape/problem array/stream/ts、Module/call_chain、时间邻接和源码控制流推导并标证据等级；未因无 timeline 停止分析或拒绝候选 Patch | PASS / FAIL |
| 手段覆盖度矩阵完整 | 每个已识别瓶颈均逐项核对对应类别标准手段；适用项均落为正式/候选/P2 Patch，不适用项均有源码或契约理由 | PASS / FAIL |
| 热点到源码解法闭环 | 每个热点原因均有源码级完整解法或明确的契约阻断；只有原因无解法即硬 FAIL | PASS / FAIL |
| 正文与自查一致 | Patch 分类、ID、状态、手段覆盖和计数与正文逐项一致；自查声称 PASS 但正文缺项即硬 FAIL | PASS / FAIL |
| 最终状态合法 | 核心分析完整且 Pattern 结构降级口径完整时用 COMPLETE；缺性能证据不阻止候选 Patch；仅核心报告因真实数据缺失无法形成时用 INCOMPLETE | PASS / FAIL |

============================================================ -->

## 0. 报告状态与局部数据提示

**最终状态：`{COMPLETE_OR_INCOMPLETE_STATUS}`**

> Step、Module、Kernel、通信及 Pattern 结构分析足以形成完整报告时使用 `COMPLETE`。Pattern 无 Kernel→DAG leaf 细粒度 timeline 时，采用确定性结构优先级并明确披露，不输出 Pattern ms/Step 或占比；该缺口不阻碍交付。仅当真实数据缺失使核心报告无法形成时使用 `INCOMPLETE: {LOCAL_REASON}`。实例不唯一、合法均分、缺少 commit 或版本信息差异不降低类级分析完成度。

| 章节 | 必需数据源 | 状态 | 说明 |
|---|---|---|---|
| Step | 原始 Trace + Step 边界 | `{PASS_OR_LOCAL_NOTICE}` | `{EVIDENCE_SUMMARY}` |
| Module 类级行为 | 可唯一定位的 class/source | `{PASS_OR_LOCAL_NOTICE}` | class/source 唯一即完成，不受实例唯一性影响 |
| Module 实例 timing | attach + post-order rollup 后的 DAG groups | `{PASS_OR_LOCAL_NOTICE}` | 公开 runtime records matched/miss 与 Module groups rolled/total；二者分列，不把 record 匹配率当 group 覆盖率 |
| Pattern | DAG JSON + production rollup；无细粒度 timeline 时使用 DAG 结构统计 | `{PASS_OR_STRUCTURAL_ONLY_OR_LOCAL_NOTICE}` | 结构降级时按 Module 热点时间、同 Module 内实例数两级排序；Module 时间仅作上下文，timeline 具体 Kernel 信息写“无” |
| Kernel | 原始 Trace + 聚合 + class/候选 class 映射 | `{PASS_OR_LOCAL_NOTICE}` | 实例不唯一仅标 timing 分配不确定，继续分析类代码及 Kernel 位置 |
| 通信 | 原始 Trace + 通信事件分类 + 源码 | `{PASS_OR_LOCAL_NOTICE}` | AllToAll/AllGather 仅作为通信类型 |

---

## 1. Step 整体分解

> 数据来源：原始 Trace 中的 ProfilerStep/训练 Step 事件；所有事件按 Step 边界裁剪后计算，多步结果取平均。

### 1.1 每 Step 时长明细

| Step # | CPU 时长 (ms) | GPU 活跃区间 (ms) | 备注 |
|---|---:|---:|---|
| `#{N}` | `{XX}` | `{XX}` | `{NOTE}` |
| **平均** | **`{XX}`** | **`{XX}`** | `{STEP_COUNT}` 个 Step |

### 1.2 GPU 时间分解（Step 平均）

| 区间类型 | 时长 (ms) | 占 Step |
|---|---:|---:|
| Compute kernel union | `{XX}` | `{XX%}` |
| Communication kernel union | `{XX}` | `{XX%}` |
| Compute ∩ Communication overlap | `{XX}` | `{XX%}` |
| 暴露通信 | `{XX}` | `{XX%}` |
| GPU 空闲 | `{XX}` | `{XX%}` |

### 1.3 Step 关键路径结论

- **主要阶段**：`{FORWARD_BACKWARD_OPTIMIZE_OTHER_BREAKDOWN}`
- **关键路径**：`{CRITICAL_PATH_IN_SEMANTIC_TERMS}`
- **首要瓶颈**：`{BOTTLENECK_AND_NUMERIC_EVIDENCE}`

---

## 2. Module 类级代码行为与实例级成本分析

> Module 分析严格拆成两层。**类级代码行为**：只要 `class/source` 可唯一定位，就必须分析计算结构、输入输出、执行路径、源码、Kernel/Pattern 位置和优化机会，不得因具体实例不唯一而停止。**实例级 timing**：仅在 attribution + attach + post-order rollup 后可合法分配时填写；production 多等长候选均分是合法分配，须披露规则。成本排序同时给出全部实例总量和单实例范围。父 Module 源码摘录最多 70 行；实现过长时只摘关键分派/调用片段，但必须用白话完整解释计算结构、数据流和子模块职责。

### 2.1 Module 类级分析清单与 timing 排序

| 排名 | Module class / 语义名称 | class 定义 | forward 范围 | 类级分析 | group ID | 实例数 | timing 分配状态 | Fwd 总计 | Bwd 总计 | 总计 | 单实例范围 | 占 Step |
|---:|---|---|---|---|---|---:|---|---:|---:|---:|---:|---:|
| 1 | `{MODULE_CLASS_AND_SEMANTIC_NAME}` | `{SOURCE_FILE}:L{CLASS_LINE}` | `L{FORWARD_START}-L{FORWARD_END}` | `{COMPLETE_OR_LOCAL_UNEXPLAINED}` | `{GROUP_IDS_OR_UNCERTAIN}` | `{N_OR_UNKNOWN}` | `{ALLOCATED/EVEN_SPLIT/UNCERTAIN/UNAVAILABLE}` | `{XX_OR_NA}` | `{XX_OR_NA}` | `{XX_OR_NA}` | `{MIN–MAX_OR_NA}` | `{XX%_OR_NA}` |

> 有合法 timing 的 Module 按全部实例总耗时占 Step 降序；没有可分配 timing 但 class/source 唯一的 Module 仍须进入类级分析，不伪造数值，并在 timing 列填 `N/A`。

### 2.2 Module 详情（每个目标 Module 均按以下固定结构填写）

#### 2.2.{N} `{MODULE_SEMANTIC_NAME}`

**A. 类级代码行为（class/source 唯一时必填）**

- **Module class / 源码位置**：`{SOURCE_FILE}:L{START}-L{END}`，`{CLASS}.{METHOD}`
- **输入**：`{INPUT_TENSORS_SHAPES_DTYPES_AND_SEMANTICS}`
- **执行路径**：`{INPUT → DISPATCH/BRANCH → SUBMODULE_OR_CUSTOM_OP → COLLECTIVE_IF_ANY → OUTPUT}`
- **输出**：`{OUTPUT_TENSORS_SHAPES_DTYPES_AND_DOWNSTREAM_USE}`
- **正常代码行为**：`{EXPLAIN_WHAT_THE_CODE_COMPUTES_AND_WHY}`
- **子模块 / custom op / collective**：`{CHILD_MODULES_CUSTOM_OPS_COLLECTIVES_AND_ROLES}`
- **关联 Pattern**：`{PATTERN_SEMANTIC_NAMES_AND_GROUPS}`，对应第 3 章 `{SECTION_REFS}`
- **关联 Kernel**：`{SHORT_KERNEL_SEMANTIC_NAMES}`，对应第 4 章 `{SECTION_REFS}`
- **可优化证据（不是建议）**：`{OBSERVED_LOCAL_EVIDENCE_AND_PATCH_PREREQUISITES}`；任何修改只能在第 6 章以正式 Patch、实验性候选 Patch 或 P2 风险实验出现；仅存在代码构造契约阻断时列前置调查

**B. 实例级 timing（仅可合法分配时填写）**

- **实例 group**：`{GROUP_IDS_OR_CANDIDATES}`；实例数 `{N_OR_UNKNOWN}`
- **分配状态与规则**：`{ALLOCATED / PRODUCTION_EVEN_SPLIT_AMONG_EQUAL_LENGTH_CANDIDATES / UNCERTAIN / UNAVAILABLE}`
- **Forward / Backward 成本**：Fwd `{XX_OR_NA}` ms/step；Bwd `{XX_OR_NA}` ms/step；合计 `{XX_OR_NA}` ms/step，占 Step `{XX%_OR_NA}`；`{EXPLAIN_COST_ASYMMETRY_OR_UNCERTAINTY}`
- **热点原因**：`{CONNECT_SHAPE_CALL_COUNT_DATA_MOVEMENT_AND_IMPLEMENTATION_OR_CLASS_LEVEL_REASON}`

**必要核心源码证据（`{SOURCE_FILE}:L{START}-L{END}`，最多 70 行；父 Module 过长时仅摘录关键分派/调用片段）**

```python
{VERBATIM_CORE_SOURCE_CODE}
```

> 若未粘贴父 Module 全量源码，必须在“正常代码行为”和“执行路径”中用白话解释完整计算结构，不能只给类名或子模块清单。

---

## 3. Pattern 结构优先级分析

> Pattern 是 DAG 中识别出的 synthetic 算子组合结构，不是 named Module，也不是通信类型。若 Timeline 没有 Kernel→DAG leaf 的具体映射，本章采用**结构优先级，不是 Pattern timing**：先按所属 Module 的已核验热点时间降序，再在同一 Module 内按 Pattern base name 去除 `#N` 后的真实 DAG 实例数降序。Module 时间只作排序上下文，绝不分摊或继承给 Pattern；不得输出 Pattern ms/Step、Step 占比或据此估算收益。分析侧原子 fact rollup 如需保留，只放内部证据区。

### 3.1 Pattern 结构 Top 表

| 结构优先级 | Pattern base name | 所属 Module 热点时间/排名（仅排序上下文） | Pattern 调用次数/实例数 | 源码结构/算子链 | timeline 具体 Kernel 信息 |
|---:|---|---|---:|---|---|
| 1 | `{PATTERN_BASE_NAME_WITHOUT_INSTANCE_SUFFIX}` | `{MODULE_NAME}: {RANK}, {MS_PER_STEP} ms/step` | `{DAG_INSTANCE_COUNT}` | `{SOURCE_LOCATION_AND_OPERATOR_CHAIN}` | `{无_OR_EXACT_TIMELINE_KERNEL}` |

### 3.2 Pattern 详情（每个 Top-N Pattern 均按以下固定结构填写）

#### 3.2.{N} `{PATTERN_SEMANTIC_NAME}`

- **排序依据**：所属 Module `{MODULE_NAME}` 排名 `{RANK}`、热点时间 `{XX}` ms/step（仅排序上下文）；Pattern base name 去 `#N` 后有 `{N}` 个真实 DAG 实例。
- **实际结构**：`{OP_A → OP_B → OP_C / BRANCH_AND_MERGE_STRUCTURE}`
- **成员节点 / 子 Pattern**：`{MEMBER_NODES_AND_CHILD_PATTERNS}`
- **源码位置**：`{SOURCE_FILE}:L{START}-L{END}`，`{CLASS}.{METHOD}`
- **源码表达式**：`{EXACT_SOURCE_EXPRESSION}`
- **输入**：`{INPUT_TENSORS_SHAPES_DTYPES_AND_SEMANTICS}`
- **输出**：`{OUTPUT_TENSORS_SHAPES_DTYPES_AND_SEMANTICS}`
- **算子组合行为**：`{EXPLAIN_EACH_OPERATOR_DATA_DEPENDENCY_FUSION_OR_MATERIALIZATION}`
- **所属 Module**：`{MODULE_SEMANTIC_NAME_AND_GROUP}`，对应第 2 章 `{SECTION_REF}`
- **可能的 Kernel 实现关系**：`{POSSIBLE_SHORT_KERNEL_SEMANTIC_NAMES_AND_ROLES}`；只表示源码算子到实现类型的可能关系，不冒充 timeline 归因。
- **timeline 具体 Kernel 信息**：`{无_OR_EXACT_EVIDENCE}`
- **结构价值（不是建议）**：`{STRUCTURAL_VALUE_AND_EVIDENCE_BOUNDARY}`；不得在本节提出修改，修改项必须进入第 6 章并通过完整 patch 门禁

**原样源码证据（`{SOURCE_FILE}:L{START}-L{END}`）**

```python
{VERBATIM_SOURCE_CODE}
```

---

## 4. Kernel 热点与实现分析

> 对外仅使用短而精确的语义名称。只要能确定 Module class 或候选 class，每个 Top Kernel 就必须继续分析对应类代码、实际实现行为、Kernel 在该类计算结构中的位置、输入输出、shape/dtype、Forward/Backward 角色和热点原因，并与 Module、Pattern 建立证据链。具体实例不唯一时仅标注“timing 分配不确定”，不得写“Kernel 源码关系未核验”并停止分析。Kernel 章节必须解释所属 Module 的白话计算行为；若第 2 章已解释，必须跳转对应 Module 小节并说明本 Kernel 的具体位置和作用。

### 4.1 Kernel Top-N（按累计时长降序）

| 排名 | Kernel 语义名称 | 角色 | 调用次数/step | 单次平均 (μs) | 累计 (ms/step) | 占 GPU 活跃时长 |
|---:|---|---|---:|---:|---:|---:|
| 1 | `{SHORT_PRECISE_KERNEL_NAME}` | `{FWD_OR_BWD_AND_FUNCTION}` | `{N}` | `{XX}` | `{XX}` | `{XX%}` |

### 4.2 Kernel 详情（每个 Top-N Kernel 均按以下固定结构填写）

#### 4.2.{N} `{SHORT_PRECISE_KERNEL_NAME}`

- **实现类别与行为**：`{CUTLASS/PYTORCH/TRITON/NVJET/FT/OPTIMIZER/CUSTOM_OP}`；`{WHAT_THE_KERNEL_ACTUALLY_DOES}`
- **Module class / 候选 class**：`{MODULE_CLASS_OR_CANDIDATE_CLASSES}`；`{SOURCE_FILE}:L{START}-L{END}`，`{CLASS}.{METHOD}`
- **实例 timing 分配**：`{ALLOCATED / PRODUCTION_EVEN_SPLIT / UNCERTAIN}`；若实例不唯一，列出候选实例并继续以下全部类代码与 Kernel 位置分析
- **模型调用代码**：`{EXPLAIN_DISPATCH_OR_WRAPPER_AND_ARGUMENTS}`
- **Kernel 在类代码中的位置**：`{CALLSITE_OR_DISPATCH_TO_KERNEL_POSITION}`
- **所属 Module 的计算行为**：`{PLAIN_LANGUAGE_EXPLANATION_OF_MODULE_COMPUTATION_STRUCTURE}`；若第 2 章已完整解释，必须写“见第 2 章 `{MODULE_SECTION_REF}`”，并补充本 Kernel 在该计算结构中的具体位置和作用
- **输入**：`{INPUT_TENSORS_SHAPES_DTYPES_LAYOUTS_AND_SEMANTICS}`
- **输出**：`{OUTPUT_TENSORS_SHAPES_DTYPES_LAYOUTS_AND_CONSUMERS}`
- **实现参数解释**：`{ARCHITECTURE_TILE_WARP_PIPELINE_LAYOUT_EPILOGUE_OR_EQUIVALENT_DETAILS}`
- **shape 变体含义**：`{WHY_VARIANTS_EXIST_AND_WHICH_MODEL_SHAPES/EXPERT_LOADS_THEY_SERVE}`
- **Forward / Backward 角色**：`{ROLE_AND_GRADIENT_RELATION}`
- **调用与成本**：`{N}` 次/step；累计 `{XX}` ms/step；占 GPU 活跃时长 `{XX%}`
- **所属 Module / Pattern**：`{MODULE_REF}` → `{PATTERN_REF}` → 本 Kernel
- **热点原因**：`{CONNECT_CALL_COUNT_SHAPE_EXPERT_IMBALANCE_TILE_EFFICIENCY_BANDWIDTH_OR_LAUNCH_OVERHEAD}`
- **源码级解法入口**：`{EXACT_SOURCE_SOLUTION_OR_CONTRACT_BLOCKER}`；若可构造运行代码，必须在第 6 章落为正式或实验性候选 Patch，不得停留在一句话方向

**原样源码证据（`{SOURCE_FILE}:L{START}-L{END}`）**

```python
{VERBATIM_SOURCE_CODE}
```

#### CUTLASS Grouped GEMM 类 Kernel 的附加必填项

- **Grouped GEMM / problem array**：`{HOW_EXPERT_PROBLEMS_AND_POINTERS/SHAPES_ARE_BUILT}`
- **模型回连**：dispatch `{SOURCE_REF}` → `expert_token_cnt` `{SOURCE_REF}` → `MoeLinear` `{SOURCE_REF}` → SwiGLU `{SOURCE_REF}` → combine `{SOURCE_REF}`
- **SM / TMA / warp specialization**：`{ARCH_AND_SCHEDULING_BEHAVIOR}`
- **MMA tile / dtype / accumulator**：`{M_N_K_TILE}`；输入 `{DTYPE}`；累加 `{ACCUMULATOR}`
- **layout / stage / epilogue**：`{A_B_C_LAYOUTS}`；`{STAGES}`；`{EPILOGUE_BEHAVIOR}`
- **变体对比**：`{COMPARE_TILE_OR_SHAPE_VARIANTS_WITH_CALLS_TIME_AND_EXPERT_TOKEN_DISTRIBUTION}`

#### 其他实现类别的附加必填项

- **PyTorch / ATen**：`{OP_SEMANTICS_DISPATCH_TENSOR_SHAPES_AND_HOTSPOT_CAUSE}`
- **Triton**：`{GRID_PROGRAM_BLOCK_TILE_LOAD_STORE_FUSION_AND_HOTSPOT_CAUSE}`
- **NVJet / FasterTransformer / custom op**：`{REGISTRATION_WRAPPER_CORE_BEHAVIOR_IO_SHAPES_AND_MODEL_CALL}`
- **Optimizer**：`{UPDATE_FORMULA_PARAMETER_GRAD_STATE_IO_DTYPE_GROUPING_AND_BANDWIDTH}`

### 4.3 无细粒度 timeline 的归因推导规则

> 缺少 Kernel→DAG leaf 或 runtime launch→GPU kernel 的直接关联时，不得停止源码分析，也不得以此为由拒绝实验性候选 Patch。必须联合使用 kernel name、dtype、shape/problem array、stream、ts 与时间邻接、Module/call_chain、源码控制流和调用次数构建候选映射，并逐项排除冲突候选。

| 证据等级 | 判定要求 | 允许结论 |
|---|---|---|
| 直接映射 | correlation/stack trace/Code Location/唯一调用点闭合 | 可用于正式 Patch 的直接证据链与收益基线 |
| 高置信推导 | 多个独立字段一致，源码控制流唯一或候选已排除，但无直接 correlation | 可形成实验性候选 Patch；必须披露推导链并实跑证伪 |
| 实验候选待证伪 | 方向与源码契约合理，存在多个未排除映射或缺性能 timing | 仍须给完整候选代码；不得承诺收益，实跑采集决定升格/退回 |

- 不确定性只限制收益承诺和上线判断，不阻止给出完整实验代码。
- kernel-idle/non-kernel gap 在无 launch 关联时不得直接宣称为 CPU launch overhead；可把短 Kernel 调用密度、源码循环/分组和时间邻接作为候选证据，并用候选 Patch 的 before/after launch 数与 Step 实测闭环。
- 每条推导必须记录：使用字段、排除的候选、剩余歧义、证伪采集项、对应第 6 章 Patch/Candidate ID。

---

## 5. 通信与计算重叠分析

> 通信事件必须由原始 Trace 的通信事件分类与语义标注识别，并回连触发源码；不得只按 Kernel 名字符串猜测。AllToAll、AllGather、AllReduce、ReduceScatter、SendRecv 均是通信类型，不是 Pattern。

### 5.1 通信来源明细

| 通信类型 | 语义用途 | 触发源码位置 | 并行维度 | 调用次数/step | Union (ms/step) | 暴露时长 (ms/step) | Overlap ratio |
|---|---|---|---|---:|---:|---:|---:|
| `{COLLECTIVE_TYPE}` | `{INPUT_TO_OUTPUT_SEMANTICS}` | `{SOURCE_FILE}:L{LINE}` | `{TP/DP/PP/EP/FSDP}` | `{N}` | `{XX}` | `{XX}` | `{XX%}` |

### 5.2 每类通信的代码行为（固定结构）

#### 5.2.{N} `{COLLECTIVE_TYPE_AND_PURPOSE}`

- **触发 Module / 方法**：`{MODULE}` / `{METHOD}`
- **源码位置**：`{SOURCE_FILE}:L{START}-L{END}`
- **输入与输出**：`{INPUT_OUTPUT_SHAPES_DTYPES_AND_RANK_SEMANTICS}`
- **通信行为**：`{HOW_DATA_IS_PARTITIONED_GATHERED_SCATTERED_OR_EXCHANGED}`
- **并行维度**：`{TP/DP/PP/EP/FSDP_AND_WORLD_SIZE}`
- **调用与成本**：`{N}` 次/step；Union `{XX}` ms；暴露 `{XX}` ms；Overlap `{XX%}`
- **关联 Module / Pattern / Kernel**：`{CROSS_SECTION_EVIDENCE_CHAIN}`
- **源码级解法入口**：`{EXACT_SOURCE_SOLUTION_OR_CONTRACT_BLOCKER}`；可构造代码时必须对应第 6 章正式/候选/P2 项

**原样源码证据（`{SOURCE_FILE}:L{START}-L{END}`）**

```python
{VERBATIM_SOURCE_CODE}
```

### 5.3 Overlap 关键结论

- **总体 overlap ratio**：`{XX%}`
- **主要暴露阶段**：`{SEMANTIC_STAGE_AND_DURATION}`
- **形成原因**：`{DEPENDENCY_BUCKET_SCHEDULING_OR_LOAD_IMBALANCE_EXPLANATION}`
- **可优化边界**：`{NUMERIC_UPPER_BOUND_AND_ASSUMPTIONS}`

---

## 6. 修改项准入、标准手段覆盖与分层 Patch

### 6.1 强制准入与四层决策规则

> **NaN 硬门禁**：任何检查项、建议项、优化项、修复项或实验候选，只要涉及代码、配置、接口、tensor 布局、通信调度或训练策略修改，就必须给出可直接实施和验证的完整代码修改方案。只有文字方向、调查措辞、伪概念描述，视为未完成。
>
> **四层决策只按契约与风险分流，不按“有没有性能 timing”分流**：
> 1. 完整数学等价、源码/API/tensor 契约闭合且证据链可归因：进入 **正式 P0/E0 或 P1/E1 Patch**。
> 2. 方向合理、能写出完整可运行代码，且不违反已知 shape/layout/placement/API 契约，但等价证明或性能映射尚未闭合：进入 **实验性候选 Patch**，必须标注“**未验证，需实跑收集证据**”。缺性能 timing 不能单独成为不给 Patch 的理由。
> 3. 修改训练目标、数值、收敛、RNG、checkpoint、top-k/router、并行或分片策略：进入 **P2/E2 风险实验**，不得作为主要优化建议。
> 4. **只有无法写出不违反已知 shape/layout/placement/API 契约的代码时**才进入前置调查；必须写明具体代码构造阻断，不能仅写“缺 timeline/缺 timing/映射待确认”。
>
> “复用”“删 clone”“删通信”“让后续直接消费”“融合”等方向可以提出，但必须展开为准确版本与位置、完整当前代码、可直接替换的完整改后代码、全部调用迁移、tensor/placement 契约、Forward/Backward/RNG/state 验证、性能采集和回滚。一句话方向视为未完成。不得引入 fallback、软化错误、静默 `continue`/`skip`、`try/except`、`hasattr` 或兼容 shim。

### 6.2 按瓶颈类别的标准手段核对表（每个已识别瓶颈必填）

| 瓶颈类别 | 必核对标准手段 | 适用性与源码/契约依据 | 对应正式 Patch / Candidate / P2 ID | 不适用理由 |
|---|---|---|---|---|
| Host launch / 短 Kernel | 批处理/分组放大；算子融合；CUDA Graph；移除 Python/host 循环；合并 optimizer 参数组；减少同步与逐 tensor dispatch | `{APPLICABILITY_AND_SOURCE_EVIDENCE}` | `{IDS_OR_NONE}` | `{REQUIRED_IF_NOT_APPLICABLE}` |
| layout / copy / cast / HBM | 删除冗余 clone/contiguous/cast；view/stride 保持；生产者或消费者融合；通信 pack 融合；延迟 materialization；复用 buffer/避免往返转换 | `{APPLICABILITY_AND_SOURCE_EVIDENCE}` | `{IDS_OR_NONE}` | `{REQUIRED_IF_NOT_APPLICABLE}` |
| GEMM / MoE / Attention | shape/tile/dispatch 选择；Grouped GEMM problem array；expert token 分桶与负载均衡；SwiGLU/epilogue 融合；QKV/attention 融合；padding/bucketing；dtype/accumulator 核对 | `{APPLICABILITY_AND_SOURCE_EVIDENCE}` | `{IDS_OR_NONE}` | `{REQUIRED_IF_NOT_APPLICABLE}` |
| 通信 / 暴露通信 | collective 合并/删除等价冗余；ReduceScatter/AllGather 替代链；bucket 调整；异步发起与依赖重排；计算通信 overlap；直接消费 sharded output；EP/TP/FSDP 调度核对 | `{APPLICABILITY_AND_SOURCE_EVIDENCE}` | `{IDS_OR_NONE}` | `{REQUIRED_IF_NOT_APPLICABLE}` |
| 显存 / 重算 | activation checkpoint 粒度；saved tensor 生命周期；重算与通信交叠；buffer 复用；in-place/alias 安全；offload；峰值显存与 OOM 余量 | `{APPLICABILITY_AND_SOURCE_EVIDENCE}` | `{IDS_OR_NONE}` | `{REQUIRED_IF_NOT_APPLICABLE}` |

> **手段覆盖度硬门禁**：每个瓶颈必须归入至少一类并逐项核对。适用手段若没有对应正式 Patch、实验性候选 Patch 或 P2/E2 风险实验，硬 FAIL；不适用却没有源码/shape/layout/placement/API/数值契约理由，硬 FAIL。表内 ID 必须与正文一致。

### 6.3 正式 P0/P1 Patch（每条逐字段完整填写）

#### 正式 Patch 6.{N} `{ACTIONABLE_TITLE}`

- **优先级 / 等价等级**：`{P0/E0_OR_P1/E1}`；**精度风险**：`{NONE_OR_QUANTIFIED_FLOAT_PATH_RISK}`
- **问题、标准手段与直接证据**：`{QUANTIFIED_LOCAL_BOTTLENECK_AND_STANDARD_METHOD}`；证据链：Module `{REF}` → Pattern `{REF_OR_NONE}` → Kernel/通信 `{REF}`
- **准确定位（当前扫描版本）**：commit `{SCANNED_COMMIT}`；`{SOURCE_FILE}:L{START}-L{END}`；`{CLASS_OR_FUNCTION}`。行号和代码必须与该 commit 一致。
- **收益基线与上限**：直接可消除成本 `{DIRECT_COST}`；理论收益上限 `{NUMERIC_CEILING}`，计算式 `{FORMULA_AND_ASSUMPTIONS}`。禁止用父 Module inclusive timing 推导局部 patch 收益。

**当前完整代码（准确覆盖替换边界，不得省略）**

```python
{VERBATIM_COMPLETE_CURRENT_CODE}
```

**可直接替换的完整改后代码（不得使用伪代码、diff 省略号或 TODO）**

```python
{COMPLETE_REPLACEMENT_CODE}
```

- **调用方与接口迁移**：`{ALL_CALLERS_FILES_FUNCTIONS_SIGNATURE_CONFIG_AND_CHECKPOINT_MIGRATION}`；无迁移时写明逐项扫描结果和“无需迁移”的证据。
- **Tensor 契约（改前→改后逐项对照）**：shape `{...}`；stride `{...}`；layout `{...}`；dtype `{...}`；device `{...}`；placement/sharding `{...}`；alias/view/in-place `{...}`；生命周期/所有权 `{...}`。删除或改变 `transpose`/`reshape`/`redistribute` 前必须证明每项等价。
- **等价不变量**：Forward 输出与数学 `{...}`；Backward/梯度 `{...}`；RNG 调用次数、顺序、seed/state `{...}`；参数名/shape/dtype/更新 `{...}`；optimizer state 的 key/shape/dtype/生命周期 `{...}`；训练语义、checkpoint/save-load `{...}`。
- **可直接运行的验证脚本**：路径 `{REPO_RELATIVE_VALIDATION_SCRIPT}`；命令：`{COPY_PASTE_RUN_COMMAND}`。脚本必须同时覆盖 Forward 与 Backward，并检查 RNG、参数和 optimizer state。
- **数值阈值**：输出 `max_abs≤{X}`、`max_rel≤{Y}`；梯度 `max_abs≤{X}`、`max_rel≤{Y}`；loss/训练轨迹 `{THRESHOLD}`；P0/E0 默认要求 bitwise 或明确证明的零差异，P1/E1 必须给 dtype/硬件适配阈值。
- **性能指标与通过标准**：固定 `{CONTROL_VARIABLES}`，warmup `{N}`、采样不少于 `{N>=5}` Step；比较 `{STEP/PATCH_LOCAL_KERNEL/COMM/MEMORY_METRICS}`；通过阈值 `{PERF_THRESHOLDS}`。
- **回滚条件与方法**：数值、梯度、RNG、状态、OOM、收敛或性能任一触发 `{ROLLBACK_THRESHOLDS}` 即回滚；执行 `{EXACT_ROLLBACK_COMMAND_OR_CONFIG}`。

> 上述任一字段为空、不准确或不可执行：本条不得保留为正式 Patch。若仍能写出不违反已知契约的完整运行代码，必须转入 6.4 实验性候选 Patch；只有代码构造被契约阻断时才转入 6.6 前置调查。

### 6.4 实验性候选 Patch（完整代码，未验证）

> 本节不是一句话建议，也不是调查任务。每条均必须明确标注：**未验证，需实跑收集证据**。候选代码必须可直接运行且对未知契约显式 `raise`，禁止 fallback 或静默跳过。

#### Candidate `{CANDIDATE_ID}` `{ACTIONABLE_TITLE}`

- **状态**：**未验证，需实跑收集证据**
- **瓶颈 / 标准手段**：`{BOTTLENECK_CATEGORY_AND_STANDARD_METHOD}`；对应 6.2 核对项 `{ROW_OR_ITEM}`
- **优先级 / 预计等价等级**：`{P0/E0_OR_P1/E1_EXPECTED}`；尚未闭合项 `{EQUIVALENCE_OR_MAPPING_GAP}`
- **证据等级与推导链**：`{DIRECT_MAPPING/HIGH_CONFIDENCE_INFERENCE/CANDIDATE_TO_FALSIFY}`；使用 kernel name/dtype/shape/problem array/stream/ts、Module/call_chain、时间邻接、源码控制流中的 `{FIELDS}`；剩余歧义 `{AMBIGUITY}`
- **准确定位**：commit `{SCANNED_COMMIT}`；`{SOURCE_FILE}:L{START}-L{END}`；`{CLASS_OR_FUNCTION}`

**当前完整代码（准确覆盖替换边界，不得省略）**

```python
{VERBATIM_COMPLETE_CURRENT_CODE}
```

**可直接运行的候选改后代码（不得使用伪代码、diff 省略号或 TODO）**

```python
{COMPLETE_RUNNABLE_CANDIDATE_CODE_WITH_EXPLICIT_CONTRACT_ASSERTIONS}
```

- **调用方 / 接口 / 配置迁移**：`{ALL_CALLERS_AND_MIGRATION}`
- **已知 Tensor/API 契约**：shape `{...}`；stride/layout `{...}`；dtype/device `{...}`；placement/sharding `{...}`；alias/lifetime `{...}`；API/返回值 `{...}`；代码如何显式拒绝不支持输入 `{EXPLICIT_RAISE}`
- **待验证不变量**：Forward `{...}`；Backward/梯度 `{...}`；RNG `{...}`；参数/optimizer state `{...}`；checkpoint/save-load `{...}`
- **可直接运行的正确性验证**：脚本 `{PATH}`；命令 `{COMMAND}`；输出/梯度/loss 阈值 `{NUMERIC_THRESHOLDS}`；shape/layout/placement/API 断言 `{CONTRACT_ASSERTIONS}`
- **可直接运行的性能采集**：脚本/命令 `{PATH_AND_COMMAND}`；固定变量 `{CONTROL_VARIABLES}`；warmup/Step `{N}`；采集 `{LAUNCH_COUNT/KERNEL/COPY/COMM/STEP/MEMORY}`
- **收益上限**：已有可归因成本 `{DIRECT_COST_OR_NA}`；上限 `{NUMERIC_CEILING_OR_NOT_YET_QUANTIFIABLE}`；若无 timing 必须写“不承诺收益，实跑后计算”，不得借用父 Module inclusive timing
- **回滚条件与方法**：`{NUMERIC_CONTRACT_PERF_OOM_CONVERGENCE_THRESHOLDS_AND_EXACT_ROLLBACK}`
- **升格条件**：等价/精度阈值通过、映射由采集闭合、性能达到 `{THRESHOLD}` 后升为正式 `{P0/E0_OR_P1/E1}`
- **退回条件**：契约或数值失败则删除候选；若失败证明无法构造不违反契约的代码，转 6.6 并记录具体阻断；仅性能无收益则关闭候选，不转调查

### 6.5 P2/E2 风险实验（不得列入主要优化建议）

#### 风险实验 6.{N} `{EXPERIMENT_TITLE}`

- **风险分类**：`P2/E2`；涉及 `{CHECKPOINT/TOP-K/ROUTER/PARALLELISM/SHARDING/NUMERICS/RNG/MEMORY/CONVERGENCE/POLICY}`
- **瓶颈 / 标准手段**：`{BOTTLENECK_CATEGORY_AND_STANDARD_METHOD}`
- **完整修改方案**：准确版本、文件/函数/行号、当前完整代码、可直接替换的完整改后代码、调用方/接口迁移和 tensor 契约均按 6.3 填写，不允许只有实验方向。
- **非等价项与影响面**：`{EXACTLY_WHAT_CAN_CHANGE_AND_WHY}`
- **隔离验证**：脚本 `{PATH}`；命令 `{COMMAND}`；数值、训练、收敛、显存、性能阈值 `{THRESHOLDS}`
- **停止/回滚条件**：`{STOP_AND_ROLLBACK_CRITERIA}`

### 6.6 前置调查任务（仅限代码构造契约阻断）

> 本节只允许“当前无法写出不违反已知 shape/layout/placement/API 契约的代码”的任务。每项必须指出冲突契约和无法构造代码的原因。缺性能 timing、缺细粒度 timeline、映射未验证或收益未知不能单独成为调查理由；只要可写完整运行代码，就必须进入 6.4。

| 调查 ID | 阻断的 shape/layout/placement/API 契约 | 为何无法构造合法代码 | 只读采集脚本/命令 | 预期产物 | 关闭条件 | 关闭后处置 |
|---|---|---|---|---|---|---|
| `{INVESTIGATION_ID}` | `{EXACT_CONTRACT_BLOCKER}` | `{CODE_CONSTRUCTION_IMPOSSIBILITY}` | `{READ_ONLY_SCRIPT_AND_COMMAND}` | `{ARTIFACT_AND_FIELDS}` | `{OBJECTIVE_CLOSE_CRITERIA}` | 解除阻断后进入 6.3/6.4/6.5，或证明无合法方案 |

---

## 7. 通信与 Overlap 修改专项约束

> 通信调度修改同样受第 6 章硬门禁约束。必须定位具体 collective 调用点和依赖边，提供完整改前/改后代码、接口迁移、placement/sharding 契约及 Forward/Backward 验证。不得把通信类型平均耗时当单次调用收益，不得只写“删除一次通信”“增加 overlap”或“让后续直接消费”。若可构造不违反已知契约的运行代码但等价或性能映射待验证，进入实验性候选 Patch；只有 placement/sharding/API 契约使合法代码无法构造时才进入前置调查。

| Patch/Candidate ID | 层级 | collective 与调用点 | 依赖/调度改动 | placement 契约/待验证项 | 暴露通信直接上限 | 验证脚本与命令 | 状态 |
|---|---|---|---|---|---:|---|---|
| `{ID}` | `{FORMAL/CANDIDATE/P2}` | `{TYPE_AND_FILE_FUNCTION_LINE}` | `{EXACT_CODE_CHANGE_REF}` | `{PLACEMENT_SHARDING_LIFETIME_PROOF_OR_GAP}` | `{MS_PER_STEP_OR_NA}` | `{PATH_AND_COMMAND}` | `{PASS/UNVERIFIED/FAIL}` |

---

## 8. 结论与交付判定

### 8.1 关键结论

1. **Step**：`{STEP_CONCLUSION_WITH_NUMBERS}`
2. **Module**：`{MODULE_CONCLUSION_WITH_SOURCE_BEHAVIOR_AND_COST}`
3. **Pattern**：`{PATTERN_STRUCTURAL_PRIORITY_CONCLUSION_WITHOUT_PATTERN_TIMING}`
4. **Kernel**：`{KERNEL_IMPLEMENTATION_CONCLUSION_AND_MODEL_SOURCE_RELATION}`
5. **通信**：`{COMMUNICATION_CONCLUSION_WITH_EXPOSED_COST_AND_SOURCE}`
6. **正式 P0/P1 Patch**：`{PATCH_IDS_OR_NONE}`
7. **实验性候选 Patch**：`{CANDIDATE_IDS_OR_NONE}`
8. **P2 风险实验**：`{EXPERIMENT_IDS_OR_NONE}`
9. **前置调查任务**：`{INVESTIGATION_IDS_OR_NONE}`
10. **标准手段覆盖**：`{FIVE_CATEGORY_COVERAGE_AND_ANY_NOT_APPLICABLE_REASONS}`

### 8.2 最终状态

**`{COMPLETE_OR_INCOMPLETE_STATUS}`**

- **数据与流程依据**：`{STEP_MODULE_PATTERN_KERNEL_COMM_SUMMARY}`
- **分层准入结果**：正式 `{FORMAL_IDS_OR_NONE}`；候选 `{CANDIDATE_IDS_OR_NONE}`；P2 `{P2_IDS_OR_NONE}`；调查 `{INVESTIGATION_IDS_OR_NONE}`
- **手段覆盖度**：`{ALL_IDENTIFIED_BOTTLENECKS_MAPPED_TO_STANDARD_METHODS_AND_IDS}`
- **局部待补证据**：`{NONE_OR_LOCAL_MISSING_DATA_OR_CONTRACT_BLOCKERS}`
- **硬 FAIL 审计**：`{NONE_OR_EXACT_FAILED_CHECKS}`
- **交付判定**：`{COMPLETE_OR_CORE_REPORT_BLOCKED_BY_REAL_MISSING_DATA}`
