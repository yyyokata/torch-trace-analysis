---
name: torch-trace-analyzer
description: 分析 PyTorch profiler Chrome Tracing JSON 训练 trace，生成模块耗时、线程时间线、GPU 热点、源码映射和交互式 HTML DAG / Swimlane 可视化。适用于排查训练性能、核对 nn.Module 数据依赖、从 trace 反查模型源码热点，或基于源码与 trace 生成流程图报告。
---

# Torch Trace Analyzer

分析 PyTorch profiler 输出的 Chrome Tracing JSON 训练 trace 文件（支持多 ProfilerStep），提供以下能力：

1. Trace 概览：设备信息、分布式配置、ProfilerStep 数量、平均 Step 耗时、Step 分解（forward_backward / optimize / other）
2. 线程分析：主线程、autograd 线程、worker 线程角色和负载
3. 多线程时间线分析：按线程独立分析和标注
4. Module 耗时分析：按层级展示 `nn.Module` 的 inclusive/exclusive 耗时与线程归属
5. GPU 时间线分析：kernel 热点、kernel 间隙、GPU 利用率、热点 kernel 到 host Module 的映射
6. Kernel 热点调用栈（增强 trace）：展示从 Python 用户代码到 kernel 的调用路径
7. Device/Host 耗时总览：GPU kernel/memcpy/memset 与 CPU 算子每步平均耗时对比
8. 通信/计算 Overlap 分析：识别 NCCL 通信流和 Memcpy 流，计算 overlap / exposed 开销
9. 源码热点分析（增强 trace）：从 kernel 的 `stack_traces` 映射回用户模型源码，区分 Forward/Backward 耗时并精确到 Class.Method
10. 交互式 HTML DAG：基于源码构建 Module 数据依赖图，并在有 trace 时叠加运行时和 timing 数据
11. Wrapper Swimlane：独立输出框架 Wrapper Module 并发泳道图

## 多步平均与 Step 边界对齐

当 trace 中包含多个 `ProfilerStep` 时，脚本自动：
- 检测所有 ProfilerStep 的起止时间边界
- 将每个 Module/Kernel 事件的 duration 裁剪到所属 Step 的起止范围内
- 对所有步的累加值除以步数，得到每步平均耗时
- 对 Step 分解（forward_backward / optimize）取所有步的平均值

## 适用范围

- 当前仅支持训练 trace（含正向+反向）
- 支持 `.json` 和 `.json.gz` 输入
- 增强 trace（含 `Code Location` 和 kernel `stack_traces`）必须同时提供模型源码

## 输入要求

- trace 文件：Chrome Tracing JSON / JSON.GZ，包含 `traceEvents`
- 模型源码：源码目录或 `.tar.gz` 压缩包（增强 trace 必须）

## 使用方法

运行主分析脚本：

```bash
python3 scripts/analyze_trace.py [trace文件路径] [选项]
```

| 参数 | 说明 |
|------|------|
| `trace_file` | trace JSON 文件路径（可选，若仅生成源码流程图则不需要） |
| `--code-path PATH` | 模型源码路径（目录或 .tar.gz），增强 trace 必须提供 |
| `--max-level N` | 限制 Module 层级显示深度 |
| `--output FILE` / `-o FILE` | 输出 Markdown 报告 |
| `--json-output FILE` | 输出 JSON 分析结果 |
| `--no-tree` | 不输出 Module 层级树 |
| `--screenshot` | 生成 Chrome Tracing 可视化截图（需 playwright） |
| `--html-flowchart PATH` | 生成 HTML 模块执行流程图 |

常见用法：

```bash
python3 scripts/analyze_trace.py trace.json
python3 scripts/analyze_trace.py trace.json.gz -o report.md
python3 scripts/analyze_trace.py timeline.json --code-path code_commit.tar.gz
python3 scripts/analyze_trace.py timeline.json --code-path code_commit.tar.gz -o report.md --screenshot
python3 scripts/analyze_trace.py --code-path code_commit.tar.gz --html-flowchart flowchart.html
python3 scripts/analyze_trace.py trace.json --code-path code_commit.tar.gz --html-flowchart flowchart.html
```

运行 Wrapper Swimlane：

```bash
python3 scripts/wrapper_swimlane.py <trace.json|.gz> -o swimlane.html
```

## HTML 模块流程图

通过 `--html-flowchart` 生成交互式 HTML DAG 可视化，展示模型 Module 层级结构：
- 基于源码解析 `__init__()` 中的 `self.xxx = Module()` 定义
- 基于 `forward()` 中的 tensor 数据流建立数据依赖边
- 支持容器展开/折叠、节点源码侧边栏、边 evidence 侧边栏、Fit to View、Hover 提示
- 当提供 trace 时叠加节点 timing、线程角色和调用统计

### AST-first 解析约定（当前默认实现）

- `scripts/analyze_trace.py` 当前主路径已切换为 **AST-first** 解析。
- 正则表达式（Regex）路径仅保留为 **最终 fallback**，用于兜底未覆盖语法，不能再作为常规主路径依赖。
- 若运行日志出现 `Regex fallback triggered: ...`，表示命中了 AST 盲点，应视为 **待修复 bug signal**，后续迭代要优先补 AST 能力，而不是长期接受该 fallback。
- 做 AST 改造或回归验证时，优先使用：
  - `testset/export_ast_refactor_current.py`
  - `testset/compare_ast_refactor.py`
  - `testset/test_dag_rules.py`
- 基线要求：对 6 个源码 case + 1 个 trace case 做全量验证，保证 compare 无 suspect regression，规则集全 PASS。

## Wrapper Swimlane

使用 scripts/wrapper_swimlane.py 基于 trace 生成框架 Wrapper Module 并发泳道图：
- 不依赖用户模型源码
- 识别框架 Wrapper 类 nn.Module
- 按 `(pid, tid)` 拆分多条泳道
- 展示执行时间段、热点 Top-N、线程信息和依赖关系
- 输出单文件交互式 HTML

## 增强 trace 规则

自动检测 trace 是否包含 `Code Location` 线程和 kernel `stack_traces`：
- 含以上字段：视为增强 trace，必须提供 `--code-path`
- 不含以上字段：按普通 trace 分析

## 模型 mock 适配指南

在进行动态分析或 Mock Run 时，必须确保模型入口文件（如 `main_model_mock.py`）是 FakeTensor-safe 的，以避免 OOM 或环境依赖问题。

### 模型 mock 适配检查清单

1. **环境门控检查**：
   - 检查 `main_model_mock.py` 顶部是否强制设置以下环境变量：
     ```python
     os.environ["MOCK_RUN_DISABLE_FAKE_TENSOR"] = "0"
     os.environ["FORGE_COMPILE"] = "1"
     ```
   - 确保 `main_model.py` 内部没有反向设置 `MOCK_RUN_DISABLE_FAKE_TENSOR=1` 的逻辑，如有冲突必须统一改为 `0`。

2. **Mock Run 入口规范（强制）**：
   - 执行入口必须是 `runstep.mock_run()`，禁止使用 `runstep.func(...)`。
   - `runstep.func(...)` 会绕过 FakeTensorMode，直接走真实 CPU forward，是 bug。
   - `mock_runtime.scope()` 只负责 runtime 注入，不激活 FakeTensorMode。
   - `MOCK_RUN_DISABLE_FAKE_TENSOR="0"` 只在 `runstep.mock_run()` 内部生效。
   - FakeTensorMode 由框架 `RunStep.mock_run()` 负责激活，不需要手工 wrap。

3. **设备元数据化（Meta Device）**：
   - 检查模型构建是否在 `with torch.device("meta"):` 上下文中进行。
   - 目的：在 import 或模型初始化阶段避免 materialize 真实参数，防止在 CPU/Mock 环境下触发 OOM。

4. **算子注册补全**：
   - 对于 triton / lego_ops 或自定义算子（custom op），必须补齐 `register_fake` + CPU fallback 逻辑。
   - 参考基准：`testset/extracted/5698781/modelcode/main_model_mock.py` 中的 patch 结构，通过对 driver/autotune 核心方法打桩来规避 GPU 依赖。

5. **验证执行**：
   - 运行 `python3 main_model_mock.py` 进行验证。
   - 预期结果：脚本正常退出，且不出现 `exit 137` (OOM kill) 报错。

## Mock Run 执行规范

规则如下：
- **超时限制**：每个 mock run 的 bash 执行必须设置 `timeout 300`（5分钟），不得省略
- **禁止嵌套子任务**：mock run 验证必须直接用 bash 执行，严禁派发 subagent/aime 子任务来"帮忙跑"mock run，否则会形成等待链条永远卡住
- **验证命令标准格式**：`cd <modelcode_dir> && timeout 300 python3 main_model_mock.py 2>&1 | grep -E "mock-runner|mock-fake-tensor|RuntimeError|Traceback|Error|WARNING.*cpu"; echo "EXIT: $?"`

## Mock Run 全量通过记录

6个 testcase mock run 全部通过（exit 0，FakeTensorMode 验证，无 CPU 真实执行）：

| testcase | 状态 | 关键 patch |
|---|---|---|
| 5698781 | ✅ | 基准 |
| 5476790 | ✅ | FakeTensor 入口强制 + DenseTower meta device |
| 5800920 | ✅ | 移除 MOCK_RUN_DISABLE_FAKE_TENSOR=1 |
| 5820572 | ✅ | triton CPU stub + kernel_dump stub + lego_ops.tf_training stub |
| 5547919 | ✅ | TensorCache stub + fake CUDA properties + triton CPU stub |
| 5758056 | ✅ | triton CPU stub + kernel_dump stub |

所有 testcase 均满足：
- 入口为 `runstep.mock_run()`
- 顶部强制 `MOCK_RUN_DISABLE_FAKE_TENSOR=0`
- FakeTensor pre-hook 检查已注册，无 real CPU tensor 进入 forward

## 参考文档

- Trace 字段与事件格式：references/trace-format.md

## 操作说明

Skill 资源位于 user_skills/torch-trace-analyzer。

相关资源路径：
- scripts/analyze_trace.py
- scripts/wrapper_swimlane.py
- references/trace-format.md

执行脚本时：
- 先 cd user_skills/torch-trace-analyzer
- 运行 python3 scripts/analyze_trace.py
- 或运行 python3 scripts/wrapper_swimlane.py

读取参考文档时，优先使用：
- view_skill user_skills/torch-trace-analyzer/references/trace-format.md
