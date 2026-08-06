# 浏览器加载与故障诊断

## 标准循环

使用本地 `browser` skill 操作已连接的 Chrome/Aime Browser：

1. `get_tabs`：复用或识别目标 tab。
2. `navigate`：打开 `https://ui.perfetto.dev`。
3. `snapshot`：页面稳定后获取新快照，禁止复用页面变化前的定位信息。
4. `upload`：只向当前快照确认的 `input.trace_file` 上传运行时 trace 参数。
5. `wait`：等待加载状态变化。
6. 页面变化后重新 `snapshot`。
7. 采集 `errors`、`console`、`screenshot`，并与时间点对应保存。
8. 只有语义定位无法覆盖 Canvas/iframe 时才使用 box；使用 box 后，页面变化仍需重新 snapshot。

## 成功判定

以下条件必须同时成立：

- 页面不再停留在 `Opening trace`，且无加载异常。
- CPU/GPU 等实际轨道可见并可展开。
- 至少一个原 trace 中已知事件能通过 Perfetto SQL 查到。
- 事件名称和时间戳与输入 trace 一致。

PONG、上传命令返回成功、ArrayBuffer 已发送、窗口已打开、页面视觉变化都不是充分条件。

## 已知阻断

Perfetto v57.2 的隐藏输入为 `input.trace_file`。直接使用 `aime-browser upload` 后页面进入 `Opening trace`，继而出现：

```text
TypeError: Cannot read properties of undefined (reading 'size')
    at TraceFileStream.readChunk
```

命中后立即停止并输出阻断报告，不尝试未验证 fallback。完整 PoC 见 [poc-loading-blocker.md](poc-loading-blocker.md)。

## 诊断记录最小集合

- Perfetto URL 与可见版本。
- trace 运行时路径、原始大小、压缩状态；路径只用于当次报告，不固化进 Skill。
- selector 与上传动作结果。
- 状态变化前后的 snapshot 标识。
- errors 和 console 的完整报错与 stack。
- screenshot 路径及截图时间点。
- tiny trace 与目标 trace 是否同样复现。
- iframe/postMessage 或 popup/opener 是否被试验；若试验，只能标记“当前未验证”，不得升级为主路径。

## 分支处理

- **selector 不存在**：报告页面版本、最新 snapshot 和 selector 变化，停止。
- **upload 命令失败**：报告工具错误、输入路径与文件预检结果，停止。
- **进入 Opening trace 后报错**：报告完整 stack，标记加载阻断，停止。
- **无报错但无轨道**：按成功判定执行已知事件 SQL；查不到即标记未加载，停止。
- **SQL schema 不匹配**：报告缺少的表/列和 `sqlite_master`/`PRAGMA table_info` 证据，停止。
- **Canvas/iframe 语义不可用**：可用 box 收集可视化证据，但 box 不能绕过 File 对象、消息通道或 SQL 验证门禁。
