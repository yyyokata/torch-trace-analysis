# 分析检查清单与证据契约

## 1. 加载前预检

对运行时 trace 参数逐项记录并验证：

- 路径存在、可读，字节数大于 0；记录原始文件大小。
- 依据 magic bytes 判断 gzip/未压缩，禁止仅凭扩展名；压缩流损坏立即 raise。
- 解压后用 JSON 解析器解析，禁止 regex 修复或提取。
- 接受前先验证 Chrome Trace 顶层数组，或包含事件数组的标准对象结构；记录采用的结构。
- 每个参与分析的事件必须具备其计算所需字段，例如 `name`、`ph`、`ts`、`dur`、`pid`、`tid`、`cat`、`args`。缺字段时报告事件索引/ID 和字段名。
- 记录事件数、时间单位、最小/最大时间戳；单位不明确则停止量化。

## 2. 加载后轨道核验

依次展开并记录实际 track 名称：

1. CPU main
2. autograd worker
3. CUDA runtime
4. GPU compute stream
5. comm stream

任何轨道缺失都要列为“缺失证据”，不能解释为该活动不存在或开销为零。

## 3. 初筛输出

### Top GPU idle

按 GPU compute stream 计算同流相邻 kernel 间隙，按 gap 降序输出。不得跨 stream 拼接事件。

### Exposed collective

筛选 NCCL/collective 活动，确认通信区间未被有效 compute 覆盖；同时保留覆盖计算作为反证。必须回看 collective 前 pack 和后 consumer wait。

### Short-kernel burst

以明确阈值和连续性规则分组短 kernel；输出每组 kernel 数、总 busy、组跨度、同流 gap 分布。阈值必须随结果记录，不得隐藏默认值。

### Host launch

关联 CUDA launch API 与 GPU kernel，输出 host API duration 及 launch end 到 kernel start 的 queue delay。缺少唯一 correlation 时停止该项，不做最近邻猜测。

## 4. 强制策略审计

### 并行与通信

- world_size、TP、FSDP、HSDP 的实际配置和预期关系。
- process group 的成员、backend 与用途。
- 跨机拓扑：node/rank/device 映射及可能跨机链路。
- fast-path guard：条件、是否命中、未命中原因及源码位置。

### GPU idle 同窗反查

在每个重要 idle 窗口内或其因果前驱中检查：

- `.item()`
- `aten::nonzero`
- D2H copy
- CUDA/device/stream/event 同步 API
- graph break

不存在命中时也要提供所用窗口和 SQL 作为反证，不能只写“未发现”。

### Optimizer

统计 `multi_tensor_apply` 次数，并按 optimizer step 给出 param-group、bucket、tensor-size 的数量与分布。缺少映射字段时明确报告无法完成的统计维度。

## 5. 源码回连

1. 使用 trace correlation、call_chain、Module 信息建立候选调用路径。
2. 对 Python 源码使用 AST 解析定位函数、调用和源码行；禁止 regex fallback。
3. 将证据关联到模型源码行和并行策略配置。
4. 若 source map、call_chain、module 或仓库版本缺失，报告缺口并降低证据等级，不得虚构定位。

## 6. 单项证据契约

每个发现必须包含：

```text
标题：
可复现时间窗：ts/dur 或 visStart/visEnd（注明单位）
track/stream：
Perfetto SQL：完整可执行查询
事件证据：完整 kernel name + stream + ts，或稳定 SQL ID
call_chain/module：
源码与配置：仓库版本、文件、行号、相关配置
证据等级：A/B/C
结论：
反证：
缺口或阻断：
```

证据等级：

- **A**：SQL、事件、call_chain/module、源码与配置完整且相互一致。
- **B**：时间与事件证据完整，但源码或策略配置存在明确缺口。
- **C**：仅候选相关性；不得写成确定根因。

截图只能作为上述证据的附件，不能单独提升等级或支撑结论。

## 7. 固化与测试

仅将稳定、可复现且 schema 已验证的发现固化到目标项目 `analyze_trace.py`：

- 数据提取函数对缺字段、无 correlation、schema 不兼容使用显式异常。
- 禁止返回 `None`、空列表、隐式 default 或静默 `continue` 掩盖失败。
- 禁止 regex fallback；源码相关提取使用 AST-first。
- storage UT 覆盖正常样本、缺字段、schema 变化、correlation 缺失、压缩损坏和时间单位不明。
- 断言完整证据字段，不只断言结果数量。
