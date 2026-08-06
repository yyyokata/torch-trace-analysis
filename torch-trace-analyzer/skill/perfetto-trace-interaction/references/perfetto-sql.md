# Perfetto SQL 模板

## 使用门禁

这些查询是结构模板，不保证适配所有 PyTorch/Perfetto schema。执行前先在 Perfetto 中检查实际表、列、时间单位和 slice 语义，并把模板中的占位符替换为已验证的实际字段。表或列不存在、关联不唯一、单位不明时立即 raise/报告，禁止改用 regex、最近邻猜测或空结果兜底。

建议先核验：

```sql
SELECT name, type FROM sqlite_master
WHERE type IN ('table', 'view')
ORDER BY name;
```

再用 `PRAGMA table_info(<table>)` 核验所用列。所有最终 SQL 必须随证据完整保存。

## Launch API duration

```sql
SELECT
  s.id AS launch_slice_id,
  s.ts AS launch_ts,
  s.dur AS launch_dur,
  s.name AS launch_api,
  t.name AS host_track
FROM slice s
JOIN track t ON t.id = s.track_id
WHERE s.name IN (
  'cudaLaunchKernel',
  'cudaLaunchKernelExC',
  'cuLaunchKernel',
  'cuLaunchKernelEx'
)
ORDER BY s.dur DESC;
```

若实际名称不同，先从 schema/数据枚举并记录精确名称；禁止用 regex fallback 隐藏名称变化。

## Launch end → kernel start queue delay

必须使用 trace 中已验证的唯一 correlation 映射。`<correlation_table>` 与字段均为占位符：

```sql
SELECT
  l.id AS launch_slice_id,
  k.id AS kernel_slice_id,
  l.ts + l.dur AS launch_end_ts,
  k.ts AS kernel_start_ts,
  k.ts - (l.ts + l.dur) AS queue_delay,
  k.name AS kernel_name,
  kt.name AS kernel_stream
FROM slice l
JOIN <correlation_table> c ON c.launch_slice_id = l.id
JOIN slice k ON k.id = c.kernel_slice_id
JOIN track kt ON kt.id = k.track_id
WHERE k.ts >= l.ts + l.dur
ORDER BY queue_delay DESC;
```

一对多关系必须先解释语义；无 correlation 时停止，不做时间最近邻关联。

## Same-stream gap

```sql
WITH kernels AS (
  SELECT
    s.id,
    s.track_id,
    t.name AS stream,
    s.name,
    s.ts,
    s.dur,
    LEAD(s.id) OVER (PARTITION BY s.track_id ORDER BY s.ts) AS next_id,
    LEAD(s.name) OVER (PARTITION BY s.track_id ORDER BY s.ts) AS next_name,
    LEAD(s.ts) OVER (PARTITION BY s.track_id ORDER BY s.ts) AS next_ts
  FROM slice s
  JOIN track t ON t.id = s.track_id
  WHERE <verified_gpu_kernel_predicate>
)
SELECT
  id,
  name,
  stream,
  ts,
  dur,
  next_id,
  next_name,
  next_ts - (ts + dur) AS same_stream_gap
FROM kernels
WHERE next_ts > ts + dur
ORDER BY same_stream_gap DESC;
```

`<verified_gpu_kernel_predicate>` 必须基于已验证 track/type 元数据，不得用未经说明的名称模糊匹配。

## Sync API

```sql
SELECT
  s.id,
  s.ts,
  s.dur,
  s.name,
  t.name AS track
FROM slice s
JOIN track t ON t.id = s.track_id
WHERE s.name IN (
  'cudaDeviceSynchronize',
  'cudaStreamSynchronize',
  'cudaEventSynchronize',
  'cuCtxSynchronize',
  'cuStreamSynchronize',
  'cuEventSynchronize'
)
ORDER BY s.dur DESC;
```

先枚举实际 API 名称；列表未命中不能推出“无同步”。

## Collective 前 pack / 后 consumer wait

先以已验证 collective slice 为锚点，显式给定窗口参数：

```sql
WITH collective AS (
  SELECT id, track_id, ts, dur, name
  FROM slice
  WHERE id = <collective_slice_id>
), windowed AS (
  SELECT
    s.id,
    s.track_id,
    t.name AS track,
    s.ts,
    s.dur,
    s.name,
    CASE
      WHEN s.ts + s.dur <= c.ts THEN 'before'
      WHEN s.ts >= c.ts + c.dur THEN 'after'
      ELSE 'overlap'
    END AS relation
  FROM collective c
  JOIN slice s
    ON s.ts < c.ts + c.dur + <post_window>
   AND s.ts + s.dur > c.ts - <pre_window>
  JOIN track t ON t.id = s.track_id
)
SELECT * FROM windowed
ORDER BY ts, track_id;
```

用完整事件名、track、call_chain/module 和源码确认哪些事件是 pack 或 consumer wait。名称相似不能代替因果证据。
