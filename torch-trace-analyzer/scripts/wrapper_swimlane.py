#!/usr/bin/env python3
"""
wrapper_swimlane.py

独立脚本：仅基于 PyTorch profiler trace（JSON / JSON.GZ）生成
"框架 Wrapper Module 并发泳道图"（Swimlane / Timeline 形式）。

特点：
- 不依赖用户模型源码。
- 仅识别框架级 Wrapper 类 nn.Module（如 DDP / FSDP / Runstep / Sync / Async / Sequential / Stream / Pipeline 等）。
- 按 (pid, tid) 拆成多条泳道，展示每个 Wrapper Module 的执行时间段。
- 计算每个 Wrapper Module 的总耗时、占比，并标注热点（Top-N 自动加 🔥）。
- 输出单文件交互式 HTML（缩放、拖动、悬停 tooltip、热点高亮、热点列表）。

使用：
    python3 scripts/wrapper_swimlane.py <trace.json|.gz> -o swimlane.html
        [--step N]       # 只渲染第 N 个 ProfilerStep（默认渲染全部步并选最具代表性的一步）
        [--top-k K]      # 热点 Top-K 数量（默认 8）
        [--min-dur US]   # 过滤短事件，单位微秒（默认 50us）
        [--include-all]  # 不只是 wrapper，连普通 nn.Module 也一起画（默认 False）
"""

import argparse
import gzip
import json
import os
import re
import sys
from collections import defaultdict, Counter


# 框架 Wrapper Module 关键词（匹配 nn.Module 名）。
# 这些是 lagrange_torch / DDP / FSDP / 训练 runner 中常见的运行时 wrapper。
WRAPPER_KEYWORDS = [
    "DDP", "FSDP", "Runstep", "RunStep", "Sync", "Async",
    "Sequential", "Stream", "Pipeline", "Composition",
    "Wrapper", "Hang", "Read", "Recv", "Lookup", "FeedRuntime",
    "GlobalStep", "SetInput", "SetOutput", "Module:Main",
    "EmbeddingAllToAll", "FidAllToAll", "AllToAll",
    "AllReduce", "ReduceScatter", "AllGather",
    "SampleRate", "SizeTensor", "Dense", "Sparse",
    "Pool", "Split", "Multi", "PerStep",
]

# 这些是用户业务 nn.Module 的提示词，识别后默认排除（除非 --include-all）
BUSINESS_KEYWORDS = [
    "Transformer", "Attention", "Encoder", "Decoder",
    "MLP", "Linear", "Conv", "BatchNorm", "LayerNorm",
    "Embedding", "Block", "Head", "Loss", "Activation",
    "Perceiver", "Bottom", "Tower",
]


def load_trace(path):
    if path.endswith(".gz"):
        with gzip.open(path, "rt", encoding="utf-8") as f:
            return json.load(f)
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def parse_module_event(ev):
    """从 'nn.Module: XXX_0, callsite: 30' 中提取干净名字。"""
    name = ev.get("name", "")
    if not name.startswith("nn.Module:"):
        return None
    raw = name[len("nn.Module:"):].strip()
    # 去掉 ', callsite: NN'
    raw = re.split(r",\s*callsite", raw)[0].strip()
    return raw


def is_wrapper_name(clean_name):
    # 必须有 _<idx> 后缀（这是 PyTorch Profiler 的实例编号）
    base = re.sub(r"_\d+$", "", clean_name)
    # 排除业务关键字（这些通常是真正的模型层）
    for bk in BUSINESS_KEYWORDS:
        if bk.lower() in base.lower():
            return False
    for kw in WRAPPER_KEYWORDS:
        if kw.lower() in base.lower():
            return True
    # 名字结尾是 Module 也算 wrapper
    if base.endswith("Module"):
        return True
    return False


def extract_thread_names(events):
    thread_names = {}
    process_labels = {}
    for e in events:
        if e.get("ph") != "M":
            continue
        if e.get("name") == "thread_name":
            thread_names[(e.get("pid"), e.get("tid"))] = e.get("args", {}).get("name", "")
        elif e.get("name") == "process_labels":
            process_labels[e.get("pid")] = e.get("args", {}).get("labels", "")
    return thread_names, process_labels


def find_profiler_steps(events):
    steps = []
    for e in events:
        if e.get("cat") == "user_annotation" and "ProfilerStep" in e.get("name", ""):
            steps.append({
                "name": e.get("name"),
                "ts": e.get("ts"),
                "dur": e.get("dur", 0),
                "tid": e.get("tid"),
                "pid": e.get("pid"),
            })
    steps.sort(key=lambda s: s["ts"])
    return steps


def collect_module_events(events, include_all=False):
    """返回 [{name, tid, pid, ts, dur, callsite, parent_pyid, py_id}]"""
    out = []
    for e in events:
        if e.get("cat") != "python_function":
            continue
        clean = parse_module_event(e)
        if clean is None:
            continue
        if not include_all and not is_wrapper_name(clean):
            continue
        args = e.get("args", {})
        out.append({
            "name": clean,
            "tid": e.get("tid"),
            "pid": e.get("pid"),
            "ts": e.get("ts"),
            "dur": e.get("dur", 0.0),
            "callsite": args.get("CallFrom", ""),
            "parent_pyid": args.get("Python parent id"),
            "py_id": args.get("Python id"),
        })
    return out


def select_step(steps, step_idx=None):
    """选择一个 ProfilerStep 作为渲染窗口。默认选 dur 最接近中位数的一步。"""
    if not steps:
        return None
    if step_idx is not None:
        if 0 <= step_idx < len(steps):
            return steps[step_idx]
        raise IndexError(f"--step {step_idx} 越界，共 {len(steps)} 个 step")
    # 选中位 dur 的步（更稳态）
    sorted_steps = sorted(steps, key=lambda s: s["dur"])
    return sorted_steps[len(sorted_steps) // 2]


def filter_in_window(mod_events, t_start, t_end):
    """裁剪事件到窗口范围内。"""
    out = []
    for m in mod_events:
        s = m["ts"]
        e = s + m["dur"]
        if e <= t_start or s >= t_end:
            continue
        cs = max(s, t_start)
        ce = min(e, t_end)
        if ce - cs <= 0:
            continue
        nm = dict(m)
        nm["ts"] = cs
        nm["dur"] = ce - cs
        out.append(nm)
    return out


def aggregate_module_stats(mod_events):
    """按 module_name 聚合：累计耗时、调用次数、所在线程。"""
    agg = defaultdict(lambda: {"dur": 0.0, "count": 0, "threads": Counter()})
    for m in mod_events:
        a = agg[m["name"]]
        a["dur"] += m["dur"]
        a["count"] += 1
        a["threads"][m["tid"]] += 1
    return agg


def assign_sub_rows(mod_events):
    """为每个事件分配同泳道内的 sub_row（行内子轨道），避免视觉重叠/拥挤。

    策略：按 (pid, tid) 分组，同组内按 (ts 升序, dur 降序) 排序，模拟 Perfetto
    的"层叠"算法：维护若干"轨道"的当前结束时间；新事件从上到下找第一条
    "结束时间 ≤ 当前事件起始 - GAP" 的轨道放入；找不到就新开一条。
    GAP 为可视化间隔（μs），保证两个相邻事件之间始终有视觉空隙。

    返回：原 list 直接附加 'sub_row' 字段，并返回每个 lane_key 的最大 sub_row 数。
    """
    GAP_RATIO = 0.0015  # 间隔占整个 step 的比例（动态在外部传入），此处先按事件群计算
    per_lane = defaultdict(list)
    for m in mod_events:
        per_lane[(m["pid"], m["tid"])].append(m)

    lane_subrow_count = {}
    for key, lst in per_lane.items():
        if not lst:
            lane_subrow_count[key] = 1
            continue
        # 估计动态 gap：取该 lane 所有事件 dur 的中位数 * 0.05，并设最小/最大值
        durs = sorted(m["dur"] for m in lst)
        med = durs[len(durs) // 2]
        gap = max(20.0, min(med * 0.1, 500.0))  # 20μs ~ 500μs

        lst.sort(key=lambda m: (m["ts"], -m["dur"]))
        track_ends = []  # 每条轨道当前结束时间 + gap
        for m in lst:
            placed = False
            for i, end in enumerate(track_ends):
                if m["ts"] >= end:
                    m["sub_row"] = i
                    track_ends[i] = m["ts"] + m["dur"] + gap
                    placed = True
                    break
            if not placed:
                m["sub_row"] = len(track_ends)
                track_ends.append(m["ts"] + m["dur"] + gap)
        lane_subrow_count[key] = max(len(track_ends), 1)
    return lane_subrow_count



def build_dependencies(mod_events):
    """推断 wrapper 间的父子调用关系。
    优先使用 Python parent id <-> Python id 链；当中间存在非 wrapper 帧导致断链时，
    退化为同线程的时间嵌套（containment）：A 事件完全包含 B 事件 → A 是 B 的祖先；
    再在所有祖先里取"时间最近"（即最深嵌套）的那个作为 B 的父 wrapper。
    """
    by_pyid = {m["py_id"]: m for m in mod_events if m.get("py_id") is not None}
    edges = Counter()

    # 第一种：直接 pyid 链
    direct_hit = 0
    for m in mod_events:
        ppid = m.get("parent_pyid")
        if ppid in by_pyid:
            parent = by_pyid[ppid]
            if parent["name"] != m["name"]:
                edges[(parent["name"], m["name"])] += 1
                direct_hit += 1

    # 第二种：基于时间嵌套（同线程内）
    if direct_hit == 0:
        per_thread = defaultdict(list)
        for m in mod_events:
            per_thread[(m["pid"], m["tid"])].append(m)
        for tid_key, lst in per_thread.items():
            # 按 ts 升序、dur 降序（外层先开始且最长）
            lst_sorted = sorted(lst, key=lambda m: (m["ts"], -m["dur"]))
            stack = []  # 当前打开的祖先（按嵌套）
            for m in lst_sorted:
                start, end = m["ts"], m["ts"] + m["dur"]
                # 弹出已结束的祖先
                while stack and stack[-1]["end"] <= start:
                    stack.pop()
                if stack and stack[-1]["m"]["name"] != m["name"]:
                    parent = stack[-1]["m"]
                    edges[(parent["name"], m["name"])] += 1
                stack.append({"m": m, "end": end})

    return [
        {"source": s, "target": t, "count": c}
        for (s, t), c in edges.most_common()
    ]


# ---- HTML 生成 ----

HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="zh">
<head>
<meta charset="utf-8">
<title>Wrapper Module Swimlane</title>
<style>
  :root {
    --bg: #0d1117;
    --panel: #161b22;
    --border: #30363d;
    --text: #e6edf3;
    --muted: #8b949e;
    --accent: #58a6ff;
    --hot: #f85149;
    --warn: #d29922;
    --grid: #21262d;
  }
  * { box-sizing: border-box; }
  html, body { margin: 0; padding: 0; background: var(--bg); color: var(--text); font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", "PingFang SC", "Microsoft YaHei", sans-serif; }
  header { padding: 12px 20px; border-bottom: 1px solid var(--border); display: flex; align-items: center; justify-content: space-between; flex-wrap: wrap; gap: 12px; }
  header h1 { font-size: 16px; margin: 0; font-weight: 600; }
  header .meta { color: var(--muted); font-size: 12px; }
  .layout { display: flex; height: calc(100vh - 56px); }
  .side { width: 420px; border-right: 1px solid var(--border); overflow-y: auto; padding: 12px; background: var(--panel); }
  .main { flex: 1; position: relative; overflow: hidden; }
  .toolbar { padding: 8px 12px; border-bottom: 1px solid var(--border); display: flex; gap: 8px; align-items: center; background: var(--panel); }
  .toolbar button { background: transparent; color: var(--text); border: 1px solid var(--border); padding: 4px 10px; border-radius: 4px; cursor: pointer; font-size: 12px; }
  .toolbar button:hover { background: var(--grid); }
  .toolbar input[type=range] { width: 200px; }
  .toolbar .hint { color: var(--muted); font-size: 11px; margin-left: auto; }
  .swimlane-wrap { position: absolute; top: 38px; left: 0; right: 0; bottom: 0; overflow: auto; }
  svg { display: block; user-select: none; }

  .lane-bg { fill: var(--panel); }
  .lane-bg.alt { fill: #11161d; }
  .lane-label { fill: var(--text); font-size: 12px; }
  .lane-sub { fill: var(--muted); font-size: 10px; }
  .grid-line { stroke: var(--grid); stroke-width: 1; }
  .axis-text { fill: var(--muted); font-size: 10px; }
  .sub-row-divider { stroke: var(--grid); stroke-width: 0.5; stroke-dasharray: 3 3; opacity: 0.6; }

  .ev-rect { stroke: rgba(0,0,0,0.4); stroke-width: 0.5; cursor: pointer; }
  .ev-rect.hot { stroke: var(--hot); stroke-width: 1.5; }
  .ev-rect.dim { opacity: 0.25; }
  .ev-text { fill: #fff; font-size: 10px; pointer-events: none; }

  .tooltip { position: fixed; background: rgba(20, 24, 30, 0.95); border: 1px solid var(--border); padding: 8px 10px; border-radius: 6px; pointer-events: none; font-size: 12px; max-width: 360px; z-index: 999; box-shadow: 0 6px 16px rgba(0,0,0,0.4); display: none; }
  .tooltip b { color: var(--accent); }
  .tooltip .row { margin: 2px 0; color: var(--text); }
  .tooltip .muted { color: var(--muted); }

  table.stats { width: 100%; border-collapse: collapse; font-size: 12px; table-layout: fixed; }
  table.stats th { text-align: left; color: var(--muted); font-weight: 500; padding: 6px 4px; border-bottom: 1px solid var(--border); white-space: nowrap; }
  table.stats td { padding: 4px; border-bottom: 1px solid var(--grid); white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
  table.stats td:first-child { white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
  #hotspot-table col.c-name, #full-table col.c-name { width: 48%; }
  #hotspot-table col.c-dur { width: 22%; }
  #hotspot-table col.c-pct { width: 20%; }
  #hotspot-table col.c-cnt { width: 10%; }
  #full-table col.c-dur { width: 26%; }
  #full-table col.c-pct { width: 26%; }
  table.stats tr.hot td:first-child::before { content: "🔥 "; }
  table.stats tr:hover { background: var(--grid); cursor: pointer; }
  .bar { display: inline-block; height: 6px; background: var(--accent); border-radius: 2px; vertical-align: middle; margin-left: 4px; }
  h2 { font-size: 13px; margin: 16px 0 6px; color: var(--accent); }
  h2:first-child { margin-top: 0; }
  .pill { display: inline-block; padding: 1px 6px; border-radius: 8px; background: var(--grid); color: var(--muted); font-size: 10px; margin-left: 4px; }
</style>
</head>
<body>
<header>
  <div>
    <h1>🏊 框架 Wrapper Module 并发泳道图 <span class="pill">{step_name}</span></h1>
    <div class="meta">
      Device: {device} · Rank {rank}/{world_size} · Step Duration: {step_dur} ·
      Wrapper Events: {n_events} · Threads: {n_threads}
    </div>
  </div>
</header>

<div class="layout">
  <aside class="side">
    <h2>🔥 热点 Wrapper Top {top_k}</h2>
    <table class="stats" id="hotspot-table">
      <colgroup><col class="c-name"><col class="c-dur"><col class="c-pct"><col class="c-cnt"></colgroup>
      <thead><tr><th>Module</th><th>耗时</th><th>占比</th><th>次数</th></tr></thead>
      <tbody>{hotspot_rows}</tbody>
    </table>

    <h2>📊 全部 Wrapper 占比</h2>
    <table class="stats" id="full-table">
      <colgroup><col class="c-name"><col class="c-dur"><col class="c-pct"></colgroup>
      <thead><tr><th>Module</th><th>耗时</th><th>占比</th></tr></thead>
      <tbody>{full_rows}</tbody>
    </table>

    <h2>🧵 线程信息</h2>
    <table class="stats">
      <thead><tr><th>Lane</th><th>Thread Name</th><th>事件数</th></tr></thead>
      <tbody>{thread_rows}</tbody>
    </table>

    <h2>🔗 Wrapper 调用依赖（Top）</h2>
    <table class="stats">
      <thead><tr><th>Parent → Child</th><th>次数</th></tr></thead>
      <tbody>{dep_rows}</tbody>
    </table>
  </aside>

  <div class="main">
    <div class="toolbar">
      <button id="btn-fit">Fit</button>
      <button id="btn-zoomin">＋</button>
      <button id="btn-zoomout">－</button>
      <label style="font-size:12px;color:var(--muted);">Zoom</label>
      <input type="range" id="zoom-slider" min="1" max="200" value="1" step="0.1">
      <button id="btn-only-hot">仅显示热点</button>
      <button id="btn-reset">重置</button>
      <span class="hint">滚轮缩放 · 拖动平移 · Hover 查看详情</span>
    </div>
    <div class="swimlane-wrap" id="canvas-wrap">
      <svg id="canvas" xmlns="http://www.w3.org/2000/svg"></svg>
    </div>
  </div>
</div>

<div class="tooltip" id="tooltip"></div>

<script>
const DATA = {data_json};

const SVG_NS = "http://www.w3.org/2000/svg";
const SUB_ROW_H = 22;          // 每个 sub_row 的高度
const SUB_ROW_GAP = 4;         // 同 lane 内 sub_row 之间的垂直间隔
const LANE_PAD = 8;            // 每个 lane 上下内边距
const LANE_DIVIDER = 6;        // 相邻 lane 之间的额外分隔
const TOP_PAD = 28;
const LEFT_PAD = 220;
const RIGHT_PAD = 20;
const TIME_AXIS_H = 22;
const EV_PAD_PX = 1.5;         // 每个事件矩形左右各留出的视觉间隙

const svg = document.getElementById('canvas');
const wrap = document.getElementById('canvas-wrap');
const tooltip = document.getElementById('tooltip');

let viewState = {
  zoom: 1.0,
  panX: 0,
  onlyHot: false,
};

const lanes = DATA.lanes; // [{lane_id, label, sub, count}]
const events = DATA.events; // [{lane, name, ts_rel, dur, hot, color, callsite, full_thread_name}]
const stepDur = DATA.step_dur;
const hotspots = new Set(DATA.hotspots);
const colorMap = DATA.color_map; // name -> color

function laneIndexById(id) {
  return lanes.findIndex(l => l.lane_id === id);
}

// 计算每个 lane 的 y 起点和高度（基于其 sub_rows 数量）
function computeLaneLayout() {
  const layout = [];
  let cursor = TOP_PAD + TIME_AXIS_H;
  lanes.forEach(lane => {
    const sr = Math.max(1, lane.sub_rows || 1);
    const innerH = sr * SUB_ROW_H + (sr - 1) * SUB_ROW_GAP;
    const totalH = innerH + LANE_PAD * 2;
    layout.push({ y: cursor, height: totalH, innerY: cursor + LANE_PAD });
    cursor += totalH + LANE_DIVIDER;
  });
  return { layout, totalH: cursor };
}

function fmtUs(us) {
  if (us >= 1e6) return (us/1e6).toFixed(3) + ' s';
  if (us >= 1e3) return (us/1e3).toFixed(2) + ' ms';
  return us.toFixed(1) + ' μs';
}

function render() {
  // 清空
  while (svg.firstChild) svg.removeChild(svg.firstChild);

  const { layout: laneLayout, totalH: laneAreaEnd } = computeLaneLayout();
  const totalHeight = laneAreaEnd + 20;
  const baseWidth = wrap.clientWidth - 4;
  const innerW = Math.max(800, baseWidth - LEFT_PAD - RIGHT_PAD);
  const drawW = innerW * viewState.zoom;
  const totalW = LEFT_PAD + drawW + RIGHT_PAD;

  svg.setAttribute('width', totalW);
  svg.setAttribute('height', totalHeight);

  // Time axis
  for (let i = 0; i <= 10; i++) {
    const x = LEFT_PAD + (drawW * i / 10);
    const t = stepDur * i / 10;
    const line = document.createElementNS(SVG_NS, 'line');
    line.setAttribute('x1', x); line.setAttribute('x2', x);
    line.setAttribute('y1', TOP_PAD); line.setAttribute('y2', totalHeight - 10);
    line.setAttribute('class', 'grid-line');
    svg.appendChild(line);
    const txt = document.createElementNS(SVG_NS, 'text');
    txt.setAttribute('x', x + 2);
    txt.setAttribute('y', TOP_PAD + 12);
    txt.setAttribute('class', 'axis-text');
    txt.textContent = fmtUs(t);
    svg.appendChild(txt);
  }

  // 泳道背景 + 标签
  lanes.forEach((lane, i) => {
    const lay = laneLayout[i];
    const bg = document.createElementNS(SVG_NS, 'rect');
    bg.setAttribute('x', 0); bg.setAttribute('y', lay.y);
    bg.setAttribute('width', totalW); bg.setAttribute('height', lay.height);
    bg.setAttribute('class', 'lane-bg' + (i % 2 ? ' alt' : ''));
    svg.appendChild(bg);

    // 内部 sub_row 分隔线（柔和虚线）
    for (let r = 1; r < (lane.sub_rows || 1); r++) {
      const sy = lay.innerY + r * SUB_ROW_H + (r - 1) * SUB_ROW_GAP + SUB_ROW_GAP / 2;
      const ln = document.createElementNS(SVG_NS, 'line');
      ln.setAttribute('x1', LEFT_PAD); ln.setAttribute('x2', totalW);
      ln.setAttribute('y1', sy); ln.setAttribute('y2', sy);
      ln.setAttribute('class', 'sub-row-divider');
      svg.appendChild(ln);
    }

    const lbl = document.createElementNS(SVG_NS, 'text');
    lbl.setAttribute('x', 8);
    lbl.setAttribute('y', lay.y + 16);
    lbl.setAttribute('class', 'lane-label');
    lbl.textContent = lane.label + ' (' + (lane.sub_rows || 1) + ' tracks)';
    svg.appendChild(lbl);

    const sub = document.createElementNS(SVG_NS, 'text');
    sub.setAttribute('x', 8);
    sub.setAttribute('y', lay.y + 30);
    sub.setAttribute('class', 'lane-sub');
    sub.textContent = lane.sub + ' · ' + lane.count + ' events';
    svg.appendChild(sub);
  });

  // 事件
  events.forEach(ev => {
    if (viewState.onlyHot && !ev.hot) return;
    const li = laneIndexById(ev.lane);
    if (li < 0) return;
    const lay = laneLayout[li];
    const sr = ev.sub_row || 0;
    const y = lay.innerY + sr * (SUB_ROW_H + SUB_ROW_GAP);
    const h = SUB_ROW_H - 4;

    let xLeft = LEFT_PAD + (ev.ts_rel / stepDur) * drawW;
    let wRaw = Math.max(1.5, (ev.dur / stepDur) * drawW);
    // 视觉左右各留 EV_PAD_PX，避免相邻事件贴在一起
    const x = xLeft + EV_PAD_PX;
    const w = Math.max(1.0, wRaw - EV_PAD_PX * 2);

    const rect = document.createElementNS(SVG_NS, 'rect');
    rect.setAttribute('x', x); rect.setAttribute('y', y + 2);
    rect.setAttribute('width', w); rect.setAttribute('height', h);
    rect.setAttribute('rx', 3); rect.setAttribute('ry', 3);
    rect.setAttribute('fill', colorMap[ev.name] || '#6e7681');
    rect.setAttribute('class', 'ev-rect' + (ev.hot ? ' hot' : ''));
    rect.setAttribute('data-module', ev.name);
    rect.addEventListener('mouseenter', e => showTip(e, ev));
    rect.addEventListener('mousemove', e => moveTip(e));
    rect.addEventListener('mouseleave', hideTip);
    svg.appendChild(rect);

    if (w > 60) {
      const charPx = 7;
      const maxChars = Math.max(0, Math.floor((w - 10) / charPx));
      const fullLabel = (ev.hot ? '🔥 ' : '') + ev.name;
      let label = '';
      if (maxChars >= fullLabel.length) {
        label = fullLabel;
      } else if (maxChars >= 4) {
        label = fullLabel.slice(0, maxChars - 1) + '…';
      }
      if (label) {
        const t = document.createElementNS(SVG_NS, 'text');
        t.setAttribute('x', x + 4); t.setAttribute('y', y + 2 + h/2 + 3);
        t.setAttribute('class', 'ev-text');
        t.textContent = label;
        svg.appendChild(t);
      }
    }
  });
}

function showTip(e, ev) {
  tooltip.innerHTML = `
    <div class="row"><b>${ev.hot ? '🔥 ' : ''}${ev.name}</b></div>
    <div class="row">⏱ 起始：${fmtUs(ev.ts_rel)} 内 · 持续 ${fmtUs(ev.dur)}</div>
    <div class="row">📊 占 step：${(ev.dur / stepDur * 100).toFixed(2)}%</div>
    <div class="row">🧵 线程：${ev.full_thread_name}</div>
    <div class="row muted">📍 ${ev.callsite || '(no callsite)'}</div>
  `;
  tooltip.style.display = 'block';
  moveTip(e);
}
function moveTip(e) {
  const pad = 14;
  let x = e.clientX + pad;
  let y = e.clientY + pad;
  const tw = tooltip.offsetWidth;
  const th = tooltip.offsetHeight;
  if (x + tw > window.innerWidth) x = e.clientX - tw - pad;
  if (y + th > window.innerHeight) y = e.clientY - th - pad;
  tooltip.style.left = x + 'px';
  tooltip.style.top = y + 'px';
}
function hideTip() { tooltip.style.display = 'none'; }

document.getElementById('btn-fit').onclick = () => { viewState.zoom = 1; document.getElementById('zoom-slider').value = 1; render(); };
document.getElementById('btn-zoomin').onclick = () => { viewState.zoom = Math.min(200, viewState.zoom * 1.5); document.getElementById('zoom-slider').value = viewState.zoom; render(); };
document.getElementById('btn-zoomout').onclick = () => { viewState.zoom = Math.max(1, viewState.zoom / 1.5); document.getElementById('zoom-slider').value = viewState.zoom; render(); };
document.getElementById('btn-only-hot').onclick = (e) => { viewState.onlyHot = !viewState.onlyHot; e.target.style.background = viewState.onlyHot ? 'var(--hot)' : 'transparent'; render(); };
document.getElementById('btn-reset').onclick = () => { viewState.zoom = 1; viewState.onlyHot = false; document.getElementById('zoom-slider').value = 1; document.getElementById('btn-only-hot').style.background = 'transparent'; render(); };

document.getElementById('zoom-slider').addEventListener('input', e => {
  viewState.zoom = parseFloat(e.target.value);
  render();
});

wrap.addEventListener('wheel', (e) => {
  if (!e.ctrlKey && !e.metaKey) return;
  e.preventDefault();
  const factor = e.deltaY < 0 ? 1.2 : 0.83;
  viewState.zoom = Math.max(1, Math.min(200, viewState.zoom * factor));
  document.getElementById('zoom-slider').value = viewState.zoom;
  render();
}, { passive: false });

// 表格点击 → 高亮 module
document.querySelectorAll('table.stats tr[data-module]').forEach(tr => {
  tr.addEventListener('click', () => {
    const name = tr.getAttribute('data-module');
    document.querySelectorAll('rect.ev-rect').forEach(r => r.classList.add('dim'));
    document.querySelectorAll(`rect.ev-rect[data-module="${name}"]`).forEach(r => r.classList.remove('dim'));
    setTimeout(() => document.querySelectorAll('rect.ev-rect').forEach(r => r.classList.remove('dim')), 2500);
  });
});

window.addEventListener('resize', render);
render();
</script>
</body>
</html>
"""


def make_color(name, hot=False):
    """根据名称生成稳定的颜色（HSL）。热点偏红橙，普通走蓝绿色。"""
    h = abs(hash(name)) % 360
    if hot:
        # 偏暖色
        h = 0 + (abs(hash(name)) % 50)  # 0-50 红橙
        s, l = 70, 55
    else:
        s, l = 55, 45
    return f"hsl({h}, {s}%, {l}%)"


def build_html(trace_data, mod_events, step, top_k, out_path, include_all):
    events_in = filter_in_window(mod_events, step["ts"], step["ts"] + step["dur"])
    if not events_in:
        print("[!] 选定 Step 内没有 wrapper module 事件，退出。", file=sys.stderr)
        sys.exit(2)

    thread_names, process_labels = extract_thread_names(trace_data.get("traceEvents", []))
    distributed = trace_data.get("distributedInfo", {}) or {}
    device_info = (trace_data.get("deviceProperties") or [{}])[0]

    # 按 (pid,tid) 拆泳道，每条泳道 = 一个线程
    lane_keys = sorted({(m["pid"], m["tid"]) for m in events_in})

    # 为每条泳道内的事件分配 sub_row（避免重叠/拥挤）
    lane_subrow_count = assign_sub_rows(events_in)

    lanes = []
    for (pid, tid) in lane_keys:
        full = thread_names.get((pid, tid), f"pid={pid} tid={tid}")
        plabel = process_labels.get(pid, "")
        # 主线程标记
        sub = f"pid {pid} · tid {tid}"
        if plabel:
            sub = f"{plabel} · {sub}"
        is_main = (tid == pid)
        label = f"{'[MAIN] ' if is_main else ''}thread {tid}"
        cnt = sum(1 for m in events_in if m["pid"] == pid and m["tid"] == tid)
        sub_rows = lane_subrow_count.get((pid, tid), 1)
        lanes.append({
            "lane_id": f"{pid}_{tid}",
            "label": label,
            "sub": (full or sub),
            "count": cnt,
            "full": full,
            "sub_rows": sub_rows,
        })

    # 聚合统计 + 热点
    stats = aggregate_module_stats(events_in)
    sorted_modules = sorted(stats.items(), key=lambda kv: -kv[1]["dur"])
    hotspot_names = [n for n, _ in sorted_modules[:top_k]]
    hotspot_set = set(hotspot_names)

    step_dur = step["dur"]
    color_map = {}
    for name, _ in sorted_modules:
        color_map[name] = make_color(name, hot=(name in hotspot_set))

    # 构造 events 数据
    js_events = []
    for m in events_in:
        js_events.append({
            "lane": f"{m['pid']}_{m['tid']}",
            "name": m["name"],
            "ts_rel": m["ts"] - step["ts"],
            "dur": m["dur"],
            "hot": m["name"] in hotspot_set,
            "callsite": m["callsite"],
            "full_thread_name": thread_names.get((m["pid"], m["tid"]), f"pid={m['pid']} tid={m['tid']}"),
            "sub_row": m.get("sub_row", 0),
        })

    # 依赖
    deps = build_dependencies(events_in)[:15]

    # ---- HTML rows ----
    def row_hot(name, st):
        pct = st["dur"] / step_dur * 100 if step_dur else 0
        bar_w = max(2, min(40, int(pct * 0.4)))
        return (
            f'<tr class="hot" data-module="{name}">'
            f'<td title="{name}">{name}</td>'
            f'<td>{format_dur(st["dur"])}</td>'
            f'<td>{pct:.1f}%<span class="bar" style="width:{bar_w}px"></span></td>'
            f'<td>{st["count"]}</td>'
            f'</tr>'
        )

    def row_full(name, st):
        pct = st["dur"] / step_dur * 100 if step_dur else 0
        bar_w = max(2, min(50, int(pct * 0.5)))
        return (
            f'<tr data-module="{name}">'
            f'<td title="{name}">{name}</td>'
            f'<td>{format_dur(st["dur"])}</td>'
            f'<td>{pct:.2f}%<span class="bar" style="width:{bar_w}px"></span></td>'
            f'</tr>'
        )

    hotspot_rows = "\n".join(row_hot(n, stats[n]) for n in hotspot_names)
    full_rows = "\n".join(row_full(n, stats[n]) for n, _ in sorted_modules)

    thread_rows = "\n".join(
        f'<tr><td>{l["label"]}</td><td>{l["full"]}</td><td>{l["count"]}</td></tr>'
        for l in lanes
    )
    dep_rows = "\n".join(
        f'<tr><td>{d["source"]} → {d["target"]}</td><td>{d["count"]}</td></tr>'
        for d in deps
    ) or '<tr><td class="muted" colspan="2">无可推断的 wrapper-wrapper 依赖</td></tr>'

    payload = {
        "lanes": lanes,
        "events": js_events,
        "step_dur": step_dur,
        "hotspots": hotspot_names,
        "color_map": color_map,
    }

    repl = {
        "{step_name}": str(step["name"]),
        "{device}": str(device_info.get("name", "Unknown")),
        "{rank}": str(distributed.get("rank", "?")),
        "{world_size}": str(distributed.get("world_size", "?")),
        "{step_dur}": format_dur(step_dur),
        "{n_events}": str(len(events_in)),
        "{n_threads}": str(len(lanes)),
        "{top_k}": str(top_k),
        "{hotspot_rows}": hotspot_rows,
        "{full_rows}": full_rows,
        "{thread_rows}": thread_rows,
        "{dep_rows}": dep_rows,
        "{data_json}": json.dumps(payload, ensure_ascii=False),
    }
    html = HTML_TEMPLATE
    for k, v in repl.items():
        html = html.replace(k, v)

    with open(out_path, "w", encoding="utf-8") as f:
        f.write(html)
    print(f"[✓] 已生成 {out_path}（{len(events_in)} 个 wrapper 事件，{len(lanes)} 条泳道，热点 {len(hotspot_names)} 个）")


def format_dur(us):
    if us >= 1e6:
        return f"{us/1e6:.3f} s"
    if us >= 1e3:
        return f"{us/1e3:.2f} ms"
    return f"{us:.1f} μs"


def main():
    ap = argparse.ArgumentParser(description="Wrapper Module Concurrency Swimlane Generator")
    ap.add_argument("trace_file", help="trace JSON 或 JSON.GZ 路径")
    ap.add_argument("-o", "--output", default="swimlane.html", help="输出 HTML 路径")
    ap.add_argument("--step", type=int, default=None, help="渲染第 N 个 ProfilerStep（0-indexed），默认中位")
    ap.add_argument("--top-k", type=int, default=8, help="热点 Top-K 数（默认 8）")
    ap.add_argument("--min-dur", type=float, default=0.0, help="过滤短事件（μs）")
    ap.add_argument("--include-all", action="store_true", help="也渲染普通 nn.Module（默认仅 wrapper）")
    args = ap.parse_args()

    if not os.path.exists(args.trace_file):
        print(f"[!] trace 文件不存在: {args.trace_file}", file=sys.stderr)
        sys.exit(1)

    print(f"[*] 加载 trace: {args.trace_file}")
    data = load_trace(args.trace_file)
    events = data.get("traceEvents", [])
    print(f"[*] 共 {len(events)} 个 trace 事件")

    steps = find_profiler_steps(events)
    if not steps:
        print("[!] 未发现 ProfilerStep，无法对齐窗口。", file=sys.stderr)
        sys.exit(2)
    print(f"[*] 发现 {len(steps)} 个 ProfilerStep")

    step = select_step(steps, args.step)
    print(f"[*] 选择 step: {step['name']} (dur={format_dur(step['dur'])})")

    mod_events = collect_module_events(events, include_all=args.include_all)
    if args.min_dur > 0:
        mod_events = [m for m in mod_events if m["dur"] >= args.min_dur]
    print(f"[*] 收集 wrapper 事件: {len(mod_events)} 个（include_all={args.include_all}）")

    build_html(data, mod_events, step, args.top_k, args.output, args.include_all)


if __name__ == "__main__":
    main()
