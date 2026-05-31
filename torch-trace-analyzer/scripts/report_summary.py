#!/usr/bin/env python3

import io
import os
import sys

from common_utils import format_duration, pct_str
from source_hotspot import LAGRANGE_TORCH_REPO


def print_header(title):
    print("=" * 90)
    print(f"  {title}")
    print("=" * 90)


def print_trace_overview(meta, trace_type, step_dur_us, thread_info, step_decomp, num_steps=1):
    print_header("Trace 概览")
    print(f"  Trace 类型:       {'训练 (Training)' if trace_type == 'training' else '推理 (Inference)'}")
    print(f"  设备:             {meta.get('device_name', 'N/A')}")
    if meta.get("world_size", 1) > 1:
        print(f"  分布式:           rank {meta.get('rank', 0)} / world_size {meta.get('world_size', 1)}")
    print(f"  ProfilerStep 数:  {num_steps}")
    print(f"  平均 Step 耗时:   {format_duration(step_dur_us)} (所有数据已按 {num_steps} 步平均)")
    if step_decomp:
        if "forward_backward_dur" in step_decomp:
            print(f"  ├─ forward_backward: {format_duration(step_decomp['forward_backward_dur'])}  ({pct_str(step_decomp['forward_backward_dur'], step_decomp.get('step_dur', step_dur_us))})")
        if "optimize_dur" in step_decomp:
            print(f"  ├─ optimize:         {format_duration(step_decomp['optimize_dur'])}  ({pct_str(step_decomp['optimize_dur'], step_decomp.get('step_dur', step_dur_us))})")
        if "other_dur" in step_decomp:
            print(f"  └─ other:            {format_duration(step_decomp['other_dur'])}  ({pct_str(step_decomp['other_dur'], step_decomp.get('step_dur', step_dur_us))})")
    print()

    # Thread summary
    roles = thread_info["thread_roles"]
    main_count = sum(1 for r in roles.values() if r["role"] == "main_thread")
    worker_count = sum(1 for r in roles.values() if r["role"] == "worker_thread")
    autograd_count = sum(1 for r in roles.values() if r["role"] == "autograd_thread")
    print(f"  CPU 线程:  main={main_count}, worker={worker_count}, autograd={autograd_count}")
    gpu_streams = len(thread_info["gpu_stream_stats"])
    print(f"  GPU 流:    {gpu_streams} 个活跃流 (主流: stream {thread_info['primary_gpu_stream'][1] if thread_info['primary_gpu_stream'] else 'N/A'})")
    print()


def print_thread_overview(thread_info, meta, workers):
    print_header("线程分析")
    thread_names = meta["thread_names"]
    roles = thread_info["thread_roles"]

    # Main thread
    print("  ── 主线程 (forward_backward 执行线程) ──")
    for (pid, tid), info in sorted(roles.items(), key=lambda x: -x[1]["dur"]):
        if info["role"] != "main_thread":
            continue
        print(f"    [{pid},{tid}] {info['name']}  events={info['count']}, dur={format_duration(info['dur'])}")
    print()

    # Autograd thread
    print("  ── Autograd 线程 (反向传播引擎) ──")
    for (pid, tid), info in sorted(roles.items(), key=lambda x: -x[1]["dur"]):
        if info["role"] != "autograd_thread":
            continue
        print(f"    [{pid},{tid}] {info['name']}  events={info['count']}, dur={format_duration(info['dur'])}")
    print()

    # Worker threads
    print("  ── Worker 线程 (并发 plugin/pipeline) ──")
    for w in workers:
        if w["role"] != "worker_thread":
            continue
        mod_str = ", ".join(w["modules"][:5]) if w["modules"] else "无 Module"
        print(f"    [{w['pid']},{w['tid']}] {w['name']}")
        print(f"      events={w['event_count']}, dur={format_duration(w['total_dur'])}")
        print(f"      modules: {mod_str}")
        if w["top_funcs"]:
            for f in w["top_funcs"][:3]:
                print(f"      top_func: {f['name'][:80]} ({format_duration(f['dur'])})")
        print()


def print_module_tree(mod_info, parent="ROOT", indent=0, step_dur_us=1.0, thread_info_map=None):
    children = mod_info["module_children"].get(parent, [])
    children_sorted = sorted(children, key=lambda n: mod_info["module_durations"].get(n, 0), reverse=True)
    for name in children_sorted:
        inc_dur = mod_info["module_durations"].get(name, 0)
        pct = inc_dur / step_dur_us * 100 if step_dur_us > 0 else 0
        short = name.replace("nn.Module: ", "")
        prefix = "  " + "│  " * indent + "├─ "
        # Add thread role indicator
        thread_tag = ""
        if thread_info_map:
            ti = thread_info_map.get(name, {})
            roles = ti.get("roles", set())
            if "worker_thread" in roles and "main_thread" not in roles:
                thread_tag = " [worker]"
            elif "autograd_thread" in roles:
                thread_tag = " [autograd]"
        print(f"{prefix}{short}  [{format_duration(inc_dur)}, {pct:.1f}%]{thread_tag}")
        print_module_tree(mod_info, name, indent + 1, step_dur_us, thread_info_map)


def print_module_report(mod_info, step_dur_us, max_level=None):
    print_header("Module 耗时分析 (Host 侧, 每步平均)")

    print(f"  主线程 Module: {mod_info['main_thread_count']} 个事件")
    print(f"  Worker 线程 Module: {mod_info['worker_thread_count']} 个事件")
    print(f"  注意: Worker 线程 Module 与主线程并发执行，不一定构成阻塞")
    if mod_info.get("num_steps", 1) > 1:
        print(f"  数据已按 {mod_info['num_steps']} 步平均")
    print()

    depth_map = mod_info["depth_map"]
    max_depth = mod_info["max_depth"]
    if max_level is not None:
        max_depth = min(max_depth, max_level)

    for level in range(max_depth + 1):
        level_modules = [name for name, d in depth_map.items() if d == level]
        if not level_modules:
            continue
        level_modules.sort(key=lambda n: mod_info["module_durations"].get(n, 0), reverse=True)

        print(f"  ── Level {level} Modules ──")
        print(f"  {'Module Name':<45} {'Thread':>8} {'Inclusive':>12} {'Exclusive':>12} {'Incl%':>8} {'Excl%':>8} {'Calls':>6}")
        print(f"  {'─'*45} {'─'*8} {'─'*12} {'─'*12} {'─'*8} {'─'*8} {'─'*6}")

        for name in level_modules:
            inc_dur = mod_info["module_durations"].get(name, 0)
            exc_dur = mod_info["module_exclusive"].get(name, 0)
            calls = mod_info["module_call_counts"].get(name, 0)
            inc_pct = (inc_dur / step_dur_us * 100) if step_dur_us > 0 else 0
            exc_pct = (exc_dur / step_dur_us * 100) if step_dur_us > 0 else 0
            short_name = name.replace("nn.Module: ", "")
            ti = mod_info["module_thread_info"].get(name, {})
            roles = ti.get("roles", set())
            if "worker_thread" in roles and "main_thread" not in roles:
                thread_tag = "worker"
            elif "main_thread" in roles:
                thread_tag = "main"
            elif "autograd_thread" in roles:
                thread_tag = "autograd"
            else:
                thread_tag = "other"
            print(
                f"  {short_name:<45} {thread_tag:>8} {format_duration(inc_dur):>12} {format_duration(exc_dur):>12} {inc_pct:>7.2f}% {exc_pct:>7.2f}% {calls:>6}"
            )
        print()


def print_gpu_timeline_report(gpu_info, step_dur_us):
    if not gpu_info:
        print("  GPU 时间线分析不可用")
        return

    print_header("GPU 时间线分析 (Device 侧, 主流 stream {})".format(gpu_info["primary_stream"][1]))

    n_gpu_steps = gpu_info.get("num_gpu_steps", 1)
    print(f"  GPU ProfilerStep 数:  {n_gpu_steps}")
    print(f"  平均 GPU Step 耗时:   {format_duration(gpu_info['gpu_step_dur'])}")
    print(f"  平均 Kernel 活跃时间: {format_duration(gpu_info['total_kernel_dur'])}  ({pct_str(gpu_info['total_kernel_dur'], gpu_info['gpu_step_dur'])})")
    print(f"  平均 Kernel 间隙时间: {format_duration(gpu_info['total_gap'])}  ({pct_str(gpu_info['total_gap'], gpu_info['gpu_step_dur'])})")
    print(f"  GPU 利用率:           {gpu_info['gpu_utilization']:.1f}%")
    print(f"  平均每步 Kernel 数:   {gpu_info['num_kernels']}")
    print()

    # Top kernels with host module mapping
    print("  ── Top 30 热点 Kernel (按总耗时) ──")
    print(f"  {'Kernel 名称':<65} {'耗时':>12} {'占比':>7} {'次数':>5}  Host Module")
    print(f"  {'─'*65} {'─'*12} {'─'*7} {'─'*5}  {'─'*30}")
    for kname, info in gpu_info["top_kernels"]:
        pct = info["dur"] / gpu_info["gpu_step_dur"] * 100 if gpu_info["gpu_step_dur"] > 0 else 0
        modules = gpu_info["top_kernel_modules"].get(kname, [])
        mod_str = modules[0][0] if modules else "N/A"
        print(f"  {kname[:65]:<65} {format_duration(info['dur']):>12} {pct:>6.2f}% {info['count']:>5}  {mod_str}")
    print()

    # Top kernel gaps
    print("  ── Top 15 Kernel 间隙 (GPU idle) ──")
    print(f"  {'间隙时长':>12}  {'前一个 Kernel':<45}  {'后一个 Kernel':<45}")
    print(f"  {'─'*12}  {'─'*45}  {'─'*45}")
    for g in gpu_info["top_gaps"]:
        print(f"  {format_duration(g['gap_us']):>12}  {g['before_name'][:45]:<45}  {g['after_name'][:45]:<45}")
    print()


def print_device_host_report(dh_info):
    print_header("Device/Host 耗时总览 (每步平均)")
    step_dur = dh_info["step_dur_us"]
    host = dh_info["host_total_us"]
    device = dh_info["device_total_us"]

    print(f"  平均 Step 耗时:          {format_duration(step_dur)}")
    print(f"  平均 Host (CPU) 耗时:    {format_duration(host)}  ({pct_str(host, step_dur)})")
    print(f"  平均 Device (GPU) 耗时:  {format_duration(device)}  ({pct_str(device, step_dur)})")
    print(f"  (Host/Device 可能超过 Step 耗时，因为多线程/多流并行)")
    print()

    print("  ── Device 耗时分解 ──")
    for cat, dur in sorted(dh_info["device_breakdown"].items(), key=lambda x: -x[1]):
        print(f"    {cat:<20} {format_duration(dur):>12}  ({pct_str(dur, step_dur)})")
    print()

    print("  ── Top 20 Device 算子 ──")
    print(f"  {'算子名称':<70} {'耗时':>12} {'占比':>8}")
    print(f"  {'─'*70} {'─'*12} {'─'*8}")
    for name, dur in dh_info["top_device_ops"]:
        print(f"  {name[:70]:<70} {format_duration(dur):>12} {pct_str(dur, step_dur):>8}")
    print()

    print("  ── Top 20 Host 算子 ──")
    print(f"  {'算子名称':<70} {'耗时':>12} {'占比':>8}")
    print(f"  {'─'*70} {'─'*12} {'─'*8}")
    for name, dur in dh_info["top_host_ops"]:
        print(f"  {name[:70]:<70} {format_duration(dur):>12} {pct_str(dur, step_dur):>8}")
    print()


def print_comm_compute_overlap(overlap_info, gpu_info):
    if not overlap_info:
        return
    print_header("通信/计算 Overlap 分析 (每步平均)")
    step_dur = overlap_info["step_dur"]
    print(f"  分析步数:           {overlap_info['num_steps']}")
    print(f"  通信流 (NCCL):      {overlap_info['num_comm_streams']} 条")
    print(f"  Memcpy 流:          {overlap_info['num_memcpy_streams']} 条")
    print()

    print("  ── 通信 (NCCL) vs 计算 ──")
    print(f"  通信总时间:         {format_duration(overlap_info['comm_total'])}")
    print(f"  与计算重叠时间:     {format_duration(overlap_info['comm_overlap'])}  ({overlap_info['comm_overlap_pct']:.1f}% 被重叠)")
    print(f"  暴露时间 (开销):    {format_duration(overlap_info['comm_exposed'])}  ({overlap_info['comm_exposed_pct_of_step']:.2f}% of step)")
    print()

    if overlap_info["memcpy_total"] > 0:
        print("  ── Memcpy vs 计算 ──")
        print(f"  Memcpy 总时间:      {format_duration(overlap_info['memcpy_total'])}")
        print(f"  与计算重叠时间:     {format_duration(overlap_info['memcpy_overlap'])}  ({overlap_info['memcpy_overlap_pct']:.1f}% 被重叠)")
        print(f"  暴露时间 (开销):    {format_duration(overlap_info['memcpy_exposed'])}  ({overlap_info['memcpy_exposed_pct_of_step']:.2f}% of step)")
        print()

    total_exposed = overlap_info["comm_exposed"] + overlap_info["memcpy_exposed"]
    total_exposed_pct = total_exposed / step_dur * 100 if step_dur > 0 else 0
    compute_total = overlap_info["compute_total"]
    print("  ── 综合 ──")
    print(f"  主流计算时间:       {format_duration(compute_total)}")
    print(f"  总暴露开销:         {format_duration(total_exposed)}  ({total_exposed_pct:.2f}% of step)")
    if gpu_info and gpu_info.get("gpu_step_dur", 0) > 0:
        effective_busy = compute_total + total_exposed
        print(f"  有效 GPU 忙碌时间:  {format_duration(effective_busy)}  (计算 + 暴露开销)")
    print()

    if overlap_info["top_comm_kernels"]:
        print("  ── Top 通信 Kernel (按耗时) ──")
        print(f"  {'Kernel 名称':<75} {'耗时':>12}")
        print(f"  {'─'*75} {'─'*12}")
        for kname, dur in overlap_info["top_comm_kernels"]:
            print(f"  {kname[:75]:<75} {format_duration(dur):>12}")
        print()


def save_comm_compute_overlap_md(overlap_info, gpu_info, L):
    if not overlap_info:
        return
    L.append("## 通信/计算 Overlap 分析\n")
    step_dur = overlap_info["step_dur"]

    L.append("### 通信 (NCCL) vs 计算\n")
    L.append("| 指标 | 数值 | 说明 |")
    L.append("|------|------|------|")
    L.append(f"| 通信总时间 | {format_duration(overlap_info['comm_total'])} | 所有 NCCL kernel 在通信流上的累计时间 |")
    L.append(f"| 与计算重叠 | {format_duration(overlap_info['comm_overlap'])} | 通信与主流计算同时进行的时间 ({overlap_info['comm_overlap_pct']:.1f}%) |")
    L.append(f"| **暴露时间** | **{format_duration(overlap_info['comm_exposed'])}** | **实际通信开销 ({overlap_info['comm_exposed_pct_of_step']:.2f}% of step)** |")
    L.append("")

    if overlap_info["memcpy_total"] > 0:
        L.append("### Memcpy vs 计算\n")
        L.append("| 指标 | 数值 | 说明 |")
        L.append("|------|------|------|")
        L.append(f"| Memcpy 总时间 | {format_duration(overlap_info['memcpy_total'])} | 非主流上的 memcpy 累计时间 |")
        L.append(f"| 与计算重叠 | {format_duration(overlap_info['memcpy_overlap'])} | ({overlap_info['memcpy_overlap_pct']:.1f}%) |")
        L.append(f"| **暴露时间** | **{format_duration(overlap_info['memcpy_exposed'])}** | **({overlap_info['memcpy_exposed_pct_of_step']:.2f}% of step)** |")
        L.append("")

    total_exposed = overlap_info["comm_exposed"] + overlap_info["memcpy_exposed"]
    total_exposed_pct = total_exposed / step_dur * 100 if step_dur > 0 else 0
    L.append("### 综合\n")
    L.append("| 指标 | 数值 |")
    L.append("|------|------|")
    L.append(f"| 主流计算时间 | {format_duration(overlap_info['compute_total'])} |")
    L.append(f"| **总暴露开销** | **{format_duration(total_exposed)} ({total_exposed_pct:.2f}% of step)** |")
    L.append("")

    if overlap_info["top_comm_kernels"]:
        L.append("### Top 通信 Kernel\n")
        L.append("| Kernel | 每步平均耗时 |")
        L.append("|--------|-------------|")
        for kname, dur in overlap_info["top_comm_kernels"]:
            L.append(f"| {kname[:80]} | {format_duration(dur)} |")
        L.append("")


def print_source_hotspot_report(src_info, source_files):
    if not src_info:
        return
    print_header("源码热点分析 (Class/Module 粒度, 正向/反向分离)")
    total = src_info["total_kernel_dur"]
    print(f"  Kernel 总耗时 (带 stack_traces): {format_duration(total)}")
    print()

    # Per-class with fwd/bwd
    print("  ── Class (nn.Module) 级别耗时 ──")
    print(f"  {'Class':<45} {'Forward':>12} {'Backward':>12} {'Total':>12} {'占比':>8}")
    print(f"  {'─'*45} {'─'*12} {'─'*12} {'─'*12} {'─'*8}")
    for ckey, phases in src_info["sorted_classes"][:20]:
        fwd, bwd = phases["fwd"], phases["bwd"]
        t = fwd + bwd
        print(f"  {ckey:<45} {format_duration(fwd):>12} {format_duration(bwd):>12} {format_duration(t):>12} {pct_str(t, total):>8}")
    print()

    # Per-class.method with fwd/bwd
    print("  ── Class.Method 级别耗时 ──")
    print(f"  {'Class.Method':<55} {'Forward':>12} {'Backward':>12} {'Total':>12} {'占比':>8}")
    print(f"  {'─'*55} {'─'*12} {'─'*12} {'─'*12} {'─'*8}")
    for cmkey, phases in src_info["sorted_methods"][:25]:
        fwd, bwd = phases["fwd"], phases["bwd"]
        t = fwd + bwd
        print(f"  {cmkey:<55} {format_duration(fwd):>12} {format_duration(bwd):>12} {format_duration(t):>12} {pct_str(t, total):>8}")
    print()

    # Top classes with annotated hot code
    for ckey, phases in src_info["sorted_classes"][:5]:
        fwd, bwd = phases["fwd"], phases["bwd"]
        t = fwd + bwd
        if t < 1000:
            continue
        hot_lines = src_info["class_hot_lines"].get(ckey, [])[:10]
        if not hot_lines:
            continue
        fname = ckey.split(":")[0]
        cname = ckey.split(":")[1]
        print(f"  ── {ckey} (fwd={format_duration(fwd)}, bwd={format_duration(bwd)}) ──")
        lines = source_files.get(fname, [])
        for lineno, dur_total, dur_fwd, dur_bwd, mname in hot_lines:
            content = lines[lineno - 1].rstrip()[:80] if 0 < lineno <= len(lines) else ""
            method_tag = f"[{mname}]" if mname else ""
            phase_tag = ""
            if dur_fwd > 0 and dur_bwd > 0:
                phase_tag = f"fwd={format_duration(dur_fwd)},bwd={format_duration(dur_bwd)}"
            elif dur_fwd > 0:
                phase_tag = f"fwd={format_duration(dur_fwd)}"
            else:
                phase_tag = f"bwd={format_duration(dur_bwd)}"
            print(f"    🔥 {format_duration(dur_total):>10} L{lineno:<5} {method_tag:<12} {content}")
        print()


def save_source_hotspot_markdown(src_info, source_files, L):
    if not src_info:
        return
    total = src_info["total_kernel_dur"]
    L.append("## 源码热点分析 (Class/Module 粒度)\n")
    L.append(f"Kernel 总耗时 (带 stack_traces): {format_duration(total)}\n")

    L.append("### Class (nn.Module) 级别耗时\n")
    L.append("| Class | Forward | Backward | Total | 占比 |")
    L.append("|-------|---------|----------|-------|------|")
    for ckey, phases in src_info["sorted_classes"][:20]:
        fwd, bwd = phases["fwd"], phases["bwd"]
        t = fwd + bwd
        L.append(f"| {ckey} | {format_duration(fwd)} | {format_duration(bwd)} | {format_duration(t)} | {pct_str(t, total)} |")
    L.append("")

    L.append("### Class.Method 级别耗时\n")
    L.append("| Class.Method | Forward | Backward | Total | 占比 |")
    L.append("|--------------|---------|----------|-------|------|")
    for cmkey, phases in src_info["sorted_methods"][:25]:
        fwd, bwd = phases["fwd"], phases["bwd"]
        t = fwd + bwd
        L.append(f"| {cmkey} | {format_duration(fwd)} | {format_duration(bwd)} | {format_duration(t)} | {pct_str(t, total)} |")
    L.append("")

    L.append("### 热点代码段\n")
    for ckey, phases in src_info["sorted_classes"][:5]:
        fwd, bwd = phases["fwd"], phases["bwd"]
        t = fwd + bwd
        if t < 1000:
            continue
        hot_lines = src_info["class_hot_lines"].get(ckey, [])[:10]
        if not hot_lines:
            continue
        fname = ckey.split(":")[0]
        L.append(f"**{ckey}** (fwd={format_duration(fwd)}, bwd={format_duration(bwd)})\n")
        L.append("```python")
        lines = source_files.get(fname, [])
        for lineno, dur_total, dur_fwd, dur_bwd, mname in hot_lines:
            content = lines[lineno - 1].rstrip()[:90] if 0 < lineno <= len(lines) else ""
            L.append(f"# 🔥 {format_duration(dur_total)} (L{lineno}) [{mname or ''}]")
            L.append(f"{content}")
        L.append("```\n")
    L.append("")


def print_per_thread_timeline(timelines):
    print_header("多线程时间线分析 (按线程独立标注)")
    for tkey, tinfo in timelines.items():
        label = tkey.replace("_", " ").title()
        tid = tinfo["tid"]
        tname = tinfo.get("name", "")
        role = tinfo.get("role", tkey)
        print(f"  ── {label}: {tname} [pid={tid[0]}, tid={tid[1]}] ──")
        print(f"    事件总数: {tinfo['total_events']}")
        print(f"    {'事件名称':<65} {'类别':>12} {'耗时':>12}")
        print(f"    {'─'*65} {'─'*12} {'─'*12}")
        for ev in tinfo["top_events"]:
            print(f"    {ev['name'][:65]:<65} {ev['cat']:>12} {format_duration(ev['dur']):>12}")
        print()


def save_per_thread_timeline_md(timelines, L):
    L.append("## 多线程时间线分析\n")
    L.append("> 各线程独立分析，以主线程和 GPU ProfilerStep 为主。\n")
    for tkey, tinfo in timelines.items():
        label = tkey.replace("_", " ").title()
        tid = tinfo["tid"]
        tname = tinfo.get("name", "")
        L.append(f"### {label}: {tname} [pid={tid[0]}, tid={tid[1]}]\n")
        L.append(f"事件总数: {tinfo['total_events']}\n")
        L.append("| 事件名称 | 类别 | 耗时 |")
        L.append("|----------|------|------|")
        for ev in tinfo["top_events"]:
            L.append(f"| {ev['name'][:80]} | {ev['cat']} | {format_duration(ev['dur'])} |")
        L.append("")


def print_kernel_call_stacks(kernel_stacks, gpu_info):
    if not kernel_stacks:
        return
    print_header("Kernel 热点调用栈分析")
    for kname, info in list(kernel_stacks.items())[:10]:
        dur = info["total_dur"]
        pct = dur / gpu_info["gpu_step_dur"] * 100 if gpu_info.get("gpu_step_dur", 0) > 0 else 0
        print(f"  ── {kname[:80]} ({format_duration(dur)}, {pct:.2f}%, count={info['count']}) ──")
        stack = info["stack"]
        for line in stack.strip().split("\n")[:12]:
            print(f"    {line.strip()}")
        print()


def save_kernel_call_stacks_md(kernel_stacks, gpu_info, L):
    if not kernel_stacks:
        return
    L.append("## Kernel 热点调用栈\n")
    for kname, info in list(kernel_stacks.items())[:10]:
        dur = info["total_dur"]
        pct = dur / gpu_info["gpu_step_dur"] * 100 if gpu_info.get("gpu_step_dur", 0) > 0 else 0
        L.append(f"### {kname[:80]}\n")
        L.append(f"- 耗时: {format_duration(dur)} ({pct:.2f}%), 次数: {info['count']}\n")
        L.append("```")
        stack = info["stack"]
        for line in stack.strip().split("\n")[:15]:
            L.append(line.strip())
        L.append("```\n")


def print_lagrange_refs(lagrange_refs):
    if not lagrange_refs:
        return
    print_header("lagrange_torch 框架引用")
    print(f"  仓库地址: {LAGRANGE_TORCH_REPO}\n")
    for ref in lagrange_refs:
        print(f"  Kernel: {ref['kernel_name'][:80]}")
        print(f"    耗时: {format_duration(ref['dur'])}, 次数: {ref['count']}")
        if ref["op_name"]:
            print(f"    算子: {ref['op_name']}")
            if ref["search_url"]:
                print(f"    源码搜索: {ref['search_url']}")
        print()


def save_lagrange_refs_md(lagrange_refs, L):
    if not lagrange_refs:
        return
    L.append("## lagrange_torch 框架引用\n")
    L.append(f"仓库地址: [{LAGRANGE_TORCH_REPO}]({LAGRANGE_TORCH_REPO})\n")
    L.append("| Kernel | 耗时 | 次数 | 算子 | 源码链接 |")
    L.append("|--------|------|------|------|----------|")
    for ref in lagrange_refs:
        op = ref["op_name"] or "-"
        link = f"[搜索]({ref['search_url']})" if ref["search_url"] else "-"
        L.append(f"| {ref['kernel_name'][:70]} | {format_duration(ref['dur'])} | {ref['count']} | {op} | {link} |")
    L.append("")


def save_markdown_report(meta, trace_type, step_dur_us, step_decomp, thread_info, workers,
                         mod_info, gpu_info, dh_info, output_path, max_level=None,
                         src_info=None, source_files=None, timelines=None,
                         kernel_stacks=None, lagrange_refs=None, screenshot_path=None,
                         num_steps=1, overlap_info=None):
    L = []
    L.append("# Torch Trace 训练分析报告\n")

    # Overview
    L.append("## 概览\n")
    L.append(f"- **Trace 类型**: {'训练' if trace_type == 'training' else '推理'}")
    L.append(f"- **设备**: {meta.get('device_name', 'N/A')}")
    if meta.get("world_size", 1) > 1:
        L.append(f"- **分布式**: rank {meta.get('rank', 0)} / world_size {meta.get('world_size', 1)}")
    L.append(f"- **ProfilerStep 数**: {num_steps}")
    L.append(f"- **平均 Step 耗时**: {format_duration(step_dur_us)} (所有数据已按 {num_steps} 步平均)")
    L.append("")

    if step_decomp:
        L.append("### Step 分解\n")
        L.append("| 阶段 | 耗时 | 占比 |")
        L.append("|------|------|------|")
        sdur = step_decomp.get("step_dur", step_dur_us)
        if "forward_backward_dur" in step_decomp:
            L.append(f"| forward_backward | {format_duration(step_decomp['forward_backward_dur'])} | {pct_str(step_decomp['forward_backward_dur'], sdur)} |")
        if "optimize_dur" in step_decomp:
            L.append(f"| optimize | {format_duration(step_decomp['optimize_dur'])} | {pct_str(step_decomp['optimize_dur'], sdur)} |")
        if "other_dur" in step_decomp:
            L.append(f"| other | {format_duration(step_decomp['other_dur'])} | {pct_str(step_decomp['other_dur'], sdur)} |")
        L.append("")

    # Thread overview
    L.append("## 线程分析\n")
    L.append("| 线程 | 角色 | 事件数 | 总耗时 | Modules |")
    L.append("|------|------|--------|--------|---------|")
    for w in workers:
        mod_str = ", ".join(w["modules"][:3]) if w["modules"] else "-"
        L.append(f"| {w['name']} | {w['role']} | {w['event_count']} | {format_duration(w['total_dur'])} | {mod_str} |")
    L.append("")

    # Module tree
    L.append("## Module 耗时分析\n")
    L.append("### Module 层级树\n")
    L.append("```")
    old_stdout = sys.stdout
    sys.stdout = buf = io.StringIO()
    print_module_tree(mod_info, "ROOT", 0, step_dur_us, mod_info.get("module_thread_info"))
    tree_str = buf.getvalue()
    sys.stdout = old_stdout
    L.append(tree_str)
    L.append("```\n")

    depth_map = mod_info["depth_map"]
    max_depth = mod_info["max_depth"]
    if max_level is not None:
        max_depth = min(max_depth, max_level)

    for level in range(max_depth + 1):
        level_modules = [n for n, d in depth_map.items() if d == level]
        if not level_modules:
            continue
        level_modules.sort(key=lambda n: mod_info["module_durations"].get(n, 0), reverse=True)
        L.append(f"### Level {level} Modules\n")
        L.append("| Module | Thread | Inclusive | Exclusive | Incl% | Excl% | Calls |")
        L.append("|--------|--------|----------|-----------|-------|-------|-------|")
        for name in level_modules:
            inc = mod_info["module_durations"].get(name, 0)
            exc = mod_info["module_exclusive"].get(name, 0)
            calls = mod_info["module_call_counts"].get(name, 0)
            short = name.replace("nn.Module: ", "")
            ti = mod_info["module_thread_info"].get(name, {})
            roles = ti.get("roles", set())
            tag = "worker" if ("worker_thread" in roles and "main_thread" not in roles) else "main" if "main_thread" in roles else "other"
            L.append(f"| {short} | {tag} | {format_duration(inc)} | {format_duration(exc)} | {pct_str(inc, step_dur_us)} | {pct_str(exc, step_dur_us)} | {calls} |")
        L.append("")

    # GPU timeline
    if gpu_info:
        L.append("## GPU 时间线分析\n")
        L.append(f"- **GPU 利用率**: {gpu_info['gpu_utilization']:.1f}%")
        L.append(f"- **Kernel 活跃时间**: {format_duration(gpu_info['total_kernel_dur'])}")
        L.append(f"- **Kernel 间隙时间**: {format_duration(gpu_info['total_gap'])}")
        L.append("")

        L.append("### Top 热点 Kernel\n")
        L.append("| Kernel | 耗时 | 占比 | 次数 | Host Module |")
        L.append("|--------|------|------|------|-------------|")
        for kname, info in gpu_info["top_kernels"]:
            pct = pct_str(info["dur"], gpu_info["gpu_step_dur"])
            modules = gpu_info["top_kernel_modules"].get(kname, [])
            mod_str = modules[0][0] if modules else "N/A"
            L.append(f"| {kname[:80]} | {format_duration(info['dur'])} | {pct} | {info['count']} | {mod_str} |")
        L.append("")

        L.append("### Top Kernel 间隙\n")
        L.append("| 间隙时长 | 前一个 Kernel | 后一个 Kernel |")
        L.append("|----------|---------------|---------------|")
        for g in gpu_info["top_gaps"][:15]:
            L.append(f"| {format_duration(g['gap_us'])} | {g['before_name'][:50]} | {g['after_name'][:50]} |")
        L.append("")

    # Device/Host
    L.append("## Device/Host 耗时总览\n")
    host = dh_info["host_total_us"]
    device = dh_info["device_total_us"]
    L.append("| 类别 | 耗时 | 占比 |")
    L.append("|------|------|------|")
    L.append(f"| Host (CPU) | {format_duration(host)} | {pct_str(host, step_dur_us)} |")
    L.append(f"| Device (GPU) | {format_duration(device)} | {pct_str(device, step_dur_us)} |")
    L.append("")

    L.append("### Device 耗时分解\n")
    L.append("| 类别 | 耗时 | 占比 |")
    L.append("|------|------|------|")
    for cat, dur in sorted(dh_info["device_breakdown"].items(), key=lambda x: -x[1]):
        L.append(f"| {cat} | {format_duration(dur)} | {pct_str(dur, step_dur_us)} |")
    L.append("")

    # Comm/compute overlap
    if overlap_info:
        save_comm_compute_overlap_md(overlap_info, gpu_info, L)

    # Source hotspot
    if src_info and source_files:
        save_source_hotspot_markdown(src_info, source_files, L)

    # Per-thread timeline
    if timelines:
        save_per_thread_timeline_md(timelines, L)

    # Kernel call stacks
    if kernel_stacks and gpu_info:
        save_kernel_call_stacks_md(kernel_stacks, gpu_info, L)

    # lagrange_torch references
    if lagrange_refs:
        save_lagrange_refs_md(lagrange_refs, L)

    # Screenshot
    if screenshot_path:
        rel_path = os.path.basename(screenshot_path)
        L.append("## Trace 可视化\n")
        L.append(f"![Trace Screenshot]({rel_path})\n")

    with open(output_path, "w") as f:
        f.write("\n".join(L))
    print(f"  Markdown 报告已保存到: {output_path}")
