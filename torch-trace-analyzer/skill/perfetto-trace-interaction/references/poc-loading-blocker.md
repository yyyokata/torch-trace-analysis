# 真实 PoC：Perfetto 自动加载阻断

## 已验证环境与范围

- 本地 Chrome/Aime Browser 已连接，可打开 `https://ui.perfetto.dev`。
- 验证页面版本：Perfetto v57.2。
- 本地 trace 路径仅是每次运行传入的参数；Skill 不保存固定路径。
- 页面隐藏文件输入为 `input.trace_file`。

## 最小复现

1. 使用本地 browser skill 获取 tabs，导航到 `https://ui.perfetto.dev`。
2. 页面稳定后 snapshot，定位隐藏的 `input.trace_file`。
3. 使用 `aime-browser upload` 向该 input 上传运行时给定 trace。
4. wait 后重新 snapshot，并采集 errors、console、screenshot。
5. 页面进入 `Opening trace`，随后抛出：

```text
TypeError: Cannot read properties of undefined (reading 'size')
    at TraceFileStream.readChunk
```

6. 使用 192-byte tiny Chrome trace 重复步骤 1–5，得到同类失败。大型样本为 84,113,465 bytes；最小样本仍复现，证明“大 trace 或 gzip 大小”不是该问题的唯一原因。

## 其他 loader 路径的实测状态

### iframe + postMessage

自建本地 loader 的 iframe 能收到 PONG，也能发送 `ArrayBuffer`，但 Perfetto iframe 未加载 trace。消息可达不等于 trace 被消费，不能把 PONG 或发送成功当作加载成功。

### window.open

该路径受 Aime Browser popup interception/opener 影响，未建立可靠消息通道。不能把窗口打开或脚本执行完成当作加载成功。

## 判定与错误策略

当前自动化加载状态为 **阻断且未验证成功**：

- `input.trace_file` 上传触发的对象没有满足 Perfetto `TraceFileStream` 对 File/Blob 大小语义的预期。
- iframe/postMessage 与 window.open 路径均未形成可核验的已加载轨道。
- 命中上述错误时必须 raise/报告：页面版本、输入 selector、文件大小、压缩状态、完整错误、console、截图路径和复现步骤。
- 禁止 fallback 到未验证 loader，禁止静默跳过，禁止继续输出假定已加载后的 SQL 或性能结论。

## 宣称自动化成功前的必要条件

满足以下任一项并完成端到端验证后，才能更新本判定：

1. 修复 Aime Browser upload，使页面收到具有正确 `name`、`size`、`type`、内容和 File/Blob 行为的文件对象；或
2. 采用 Perfetto 官方、可验证的 loader，并证明指定 trace 已生成可查询轨道。

端到端验证至少包含：页面无加载错误、目标轨道可见、已知事件可通过 Perfetto SQL 查到且时间戳/名称与原 trace 一致。仅有 PONG、ArrayBuffer 已发送、页面显示 `Opening trace` 或截图变化均不算成功。
