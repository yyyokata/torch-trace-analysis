# Trace Processor Agent 工作流与已验证 PoC

## 1. 环境确认门禁

任何虚拟环境、依赖安装、证书生成、文件服务或 RPC 服务启动之前，必须先向用户确认实际运行环境。至少确认操作系统或容器类型、Python 版本与包管理器、允许创建文件和安装依赖的位置、浏览器所在环境、本地端口是否可达，以及用户禁止修改的环境范围。

不得默认用户使用 WSL，也不得把 `venv`、`pip`、`apt`、Homebrew、Conda 或特定路径写成唯一方案。确认环境后再把下面的逻辑步骤转换成该环境适用的命令。环境未知、权限范围不明确或浏览器到服务的网络路径不清楚时，停止并询问，禁止试装或猜测。

## 2. 环境无关架构

稳定架构由三个独立入口组成。Trace 文件服务通过浏览器可访问的 HTTPS URL 提供运行时传入的 trace。Trace Processor HTTP/RPC Server 使用与 Perfetto Web UI 相同的版本，并由 Web UI 加载 trace。Python API 通过 RPC 地址连接同一个 Trace Processor 实例并执行 PerfettoSQL。

Web UI 负责轨道展示、时间窗定位、SQL 结果交互和截图复核。Python/RPC 负责 schema 核验、批量 SQL、SQL ID、track、stream、时间戳和 duration 的结构化输出。禁止仅靠 Canvas 坐标或截图得出结论。

## 3. 版本门禁

Python 包版本与它默认下载的 Trace Processor 二进制版本不一定一致。必须分别打印并记录 Python 包版本、Trace Processor `--version` 和 Web UI 页面版本。

HTTP/RPC Server 的 Trace Processor 版本必须与 Web UI 版本一致。版本不一致时，WebSocket 可能建立后关闭，页面可能显示 `WebSocket error`、`Websocket closed (1006)` 或停留在加载状态。遇到这种情况必须停止并报告实际版本，不得静默切回另一个引擎。

已验证实例中，Python 包 `perfetto 0.57.2` 默认取得的固定二进制为 Trace Processor v56.1，而 Web UI 为 v57.2。使用 Python API 的最新二进制获取能力取得 v57.2 后，Web UI、RPC 和 Python SQL 才能稳定共享同一实例。这个版本差异是实例证据，不得泛化为所有环境的固定版本关系。

## 4. 加载顺序

先启动不预加载 trace 的空 HTTP/RPC Server，再让 Perfetto Web UI 通过 HTTPS `?url=` Deep Link 加载 trace，最后用 Python API 的 `addr` 模式连接该 RPC 实例。

不要先让 RPC Server 预加载 trace，再让 Web UI 向同一实例重复加载 URL trace。已验证实例中，该顺序会让 Web 长时间停留在 `Opening trace`。正确顺序下，Web UI 日志应明确出现 `Opening trace using native accelerator over HTTP+RPC`，页面状态应显示 `RPC`，Python 查询和 Web Query 应返回同一 SQL ID、名称、时间戳与 duration。

## 5. Trace 与 HTTPS 预检

Trace 路径仅作为运行时参数传入。先以 magic bytes 判断 gzip 状态，再用 JSON 解析器完整解析，核验 Chrome Trace 顶层结构、事件数量、时间戳范围和时间单位。压缩损坏、JSON 解析失败、必要结构缺失或单位不明时立即报错。

HTTPS 文件服务必须返回 `200`、正确 `Content-Length`、浏览器可接受的 TLS 连接和允许 Perfetto 来源读取的 CORS 响应头。本地自签名证书可能需要浏览器会话先接受；如果用户要求无人值守，应先与用户确认允许采用的可信 HTTPS 暴露方式，不得擅自修改系统证书库。

## 6. Python/RPC 验收

Python API 连接后，首先查询 `sqlite_master` 和所用表的 `PRAGMA table_info`。然后至少核验 `slice` 数量、`track` 数量、thread/process/stream 关系，以及一条已知真实事件。

Web Query 使用同一条 SQL 再查一次。只有 Python 与 Web 返回相同的 SQL ID、完整事件名、track、timestamp 和 duration，且当前页面错误列表为空，才判定共库成功。

已验证实例的共库事件为 SQL ID `1267630`，完整名称为 `ncclDevKernel_AllGather_RING_LL(ncclDevKernelArgsStorage<4096ul>)`，timestamp 为 `2879362288577765 ns`，duration 为 `55883193 ns`，track ID 为 `19`，对应 stream `49`。该事件只用于证明本次 PoC 的端到端一致性，后续运行必须重新查询，不得写成其他 trace 的固定事实。

## 7. Web 定位策略

侧栏、Query 页面、Run Query 按钮、轨道组展开按钮优先使用语义 ref 或 selector。Perfetto Timeline 主体是 Canvas，子轨道和 slice 往往无法通过 accessibility snapshot 完整读取；这时只能用截图和 box 辅助定位，但最终事件必须由 SQL ID、完整名称、track、timestamp 和 duration 回连。

Query 页面稳定流程是进入 `#!/query`，重新 snapshot，向编辑器 textbox 写入完整 SQL，重新 snapshot 后点击 `Run Query`，再读取结果文本。直接执行 DOM `button.click()` 在已验证实例中不总能触发查询，不能把调用返回成功当作 SQL 已执行；必须看到 `Returned N rows` 和结果单元格。

## 8. 已验证实例时间线

本次 PoC 先复现隐藏文件输入上传的 `TraceFileStream.readChunk` 错误，以及 iframe/postMessage 与 popup/opener 无法形成可核验加载的阻断。随后在用户明确指定的隔离环境中启动 HTTPS 文件服务，使用 `?url=` 成功加载约 85 MB 的 gzip Chrome Trace，并看到实际 PyTorch 轨道。

之后安装官方 Python 包，分别验证直接文件模式和 RPC `addr` 模式。发现默认 Trace Processor v56.1 与 Web UI v57.2 不一致后，切换为版本一致的 v57.2 二进制。最终由空 RPC Server 接收 Web URL trace，Python 连接同一实例，Web Query 和 Python SQL 对同一 NCCL kernel 返回一致结果，当前页面 errors 为空。

## 9. 错误策略

环境、版本、schema、字段、correlation 或加载状态不符合门禁时，必须显式报错并停止当前分析项。禁止返回空结果冒充“没有事件”，禁止静默跳过，禁止用最近邻时间匹配替代 correlation，禁止把 WebSocket 曾经连接、PONG、进度条、页面变化或截图当作加载成功。
