---
name: perfetto-trace-interaction
description: 为 PyTorch Chrome Trace 建立 Perfetto 可视化与程序化分析流程，覆盖 trace 预检、HTTPS URL 加载、Trace Processor Python/RPC 共库、PerfettoSQL、网页定位、源码与并行策略回连及 storage UT 固化。适用于分析 PyTorch Profiler Chrome Trace、GPU idle、collective 暴露、短 kernel burst、host launch、同步与通信瓶颈。
author: linnan.5931
---

# PyTorch Trace → Perfetto 前置流程

## 硬门禁

- 把 trace 路径仅作为运行时参数接收，禁止写死本地路径。
- 创建虚拟环境、安装依赖或启动服务前，必须先询问并确认用户的实际运行环境、可用 Python、允许修改的范围和服务可达方式。禁止把 WSL、Linux、macOS、容器或特定包管理器写成固定前提；本次已验证的 WSL 流程仅作为实例。
- 先检查格式、字节数和压缩状态；用 JSON 解析器验证 Chrome Trace 结构。解析失败、压缩识别失败或字段不完整时立即 raise/报告。
- 当前 `https://ui.perfetto.dev` v57.2 的 `input.trace_file` 自动上传路径已被真实 PoC 证明阻断。出现 `TraceFileStream.readChunk` 的 `undefined.size` TypeError 时，记录完整错误并停止；禁止 fallback、静默跳过或声称 trace 已加载。
- iframe + `postMessage` loader 与 `window.open` 通道均未验证成功，不得作为自动化主路径。复现与判定见 [poc-loading-blocker.md](references/poc-loading-blocker.md)。
- 任何 SQL schema 不匹配、缺 correlation、缺源码、字段不完整都立即 raise/报告；禁止返回 `None`、空列表、default 或静默 `continue`，也禁止 regex fallback。源码定位采用 AST-first。
- 截图只能辅助定位。每项结论必须提供可查询、可回连的证据，禁止只凭截图下结论。

## 主流程

1. **确认环境**：询问并记录用户的实际运行环境、Python/包管理器、网络边界、允许安装与启动服务的位置，以及浏览器访问本地服务的方式；未确认前不得创建虚拟环境或服务。
2. **预检输入**：记录绝对路径、实际字节数、压缩类型、解压后格式、顶层结构、事件数和必要字段。执行完整门禁见 [analysis-checklist.md](references/analysis-checklist.md)。
3. **建立程序化入口**：优先使用官方 Trace Processor Python API + 版本匹配的 HTTP/RPC Server，让 Python SQL 与 Perfetto Web 共享同一数据库实例。环境无关流程、版本门禁和已验证实例见 [trace-processor-agent-workflow.md](references/trace-processor-agent-workflow.md)。
4. **加载 Perfetto**：优先通过 HTTPS `?url=` Deep Link 加载 trace；使用本地 `browser` skill 完成 `get_tabs → navigate → snapshot → wait → errors → console → screenshot` 循环。页面变化后重新 snapshot；Canvas 语义不可用时才使用 box。
5. **判断加载结果**：只有页面出现实际轨道、Python/RPC SQL 可查询到已知事件，且事件名称、SQL ID、时间戳与输入 trace 一致，才进入分析。否则输出阻断报告并停止。
6. **展开关键轨道**：依次核验 CPU main、autograd worker、CUDA runtime、GPU compute stream、comm stream；缺轨道必须报告，不能把缺失当作零开销。
7. **SQL 初筛与量化**：筛出 Top GPU idle、exposed collective、short-kernel burst、host launch，保留候选时间窗、track/stream、SQL ID、原始名称与完整查询；先验证 schema 和 correlation，再使用 [perfetto-sql.md](references/perfetto-sql.md) 中的模板。
8. **证据回连**：将可视化与 SQL 证据回连到 Module、call_chain、模型源码行和并行策略配置；源码定位 AST-first，定位不完整即报告。
9. **策略审计**：强制检查 world_size/TP/FSDP/HSDP、process group、跨机拓扑、fast-path guard；对 GPU idle 同窗反查 `.item()`、`aten::nonzero`、D2H、同步 API、graph break；统计 optimizer `multi_tensor_apply` 次数及 param-group/bucket/tensor-size 分布。
10. **固化与输出**：稳定发现写回目标项目并新增/更新 storage UT；测试失败或证据夹具不完整时明确报错。逐项按 [analysis-checklist.md](references/analysis-checklist.md) 的证据契约输出结论与反证。

## 浏览器异常处理

加载失败、选择器变化、Canvas/iframe 不可观测、popup/opener 异常时，读取 [browser-diagnostics.md](references/browser-diagnostics.md)。该文档用于诊断和报告，不授权绕过加载门禁。
