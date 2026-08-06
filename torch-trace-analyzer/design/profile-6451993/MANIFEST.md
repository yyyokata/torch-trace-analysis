# Profile 6451993 归档清单

本目录归档模型 6451993 本次性能分析使用的通用报告模板和已填充报告。Perfetto 交互与 Trace Processor SQL 工作流位于项目根目录 `skill/perfetto-trace-interaction/`；离线 Trace 分析与 DAG/Swimlane 工具说明位于 `skill/torch-trace-analyzer/`。

`perf_report_template.md` 是可复用的 Profile 报告模板。`perf_report_6451993.md` 是基于五个完整 ProfilerStep、源码、DAG timing 与 Perfetto RPC 独立复核形成的模型报告。

本归档不包含原始 trace、模型源码、HTML、JSON 中间结果、调试脚本、日志、缓存或临时备份。报告中的证据路径用于说明本次分析来源，复跑时必须由用户提供实际环境和运行时路径，禁止写死 WSL 或其他固定环境。
