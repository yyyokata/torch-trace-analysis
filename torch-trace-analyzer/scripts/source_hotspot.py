#!/usr/bin/env python3

import os
import re
from collections import defaultdict

from source_index import _build_class_map, _find_class_for_line


def analyze_source_hotspots(events, source_files):
    # Build class/method line range map
    class_map = _build_class_map(source_files)

    line_dur = defaultdict(lambda: {"fwd": 0.0, "bwd": 0.0})
    class_dur = defaultdict(lambda: {"fwd": 0.0, "bwd": 0.0})
    class_method_dur = defaultdict(lambda: {"fwd": 0.0, "bwd": 0.0})
    total_kernel_dur = 0.0

    for e in events:
        if e.get("cat") != "kernel":
            continue
        dur = e.get("dur", 0)
        total_kernel_dur += dur
        st = e.get("args", {}).get("stack", {})
        traces = st.get("stack_traces", [])
        if not traces:
            continue

        is_bwd = any("Gradient" in t or "Backward" in t or "backward" in t for t in traces)
        phase = "bwd" if is_bwd else "fwd"
        n_traces = max(1, len(traces))
        per_trace_dur = dur / n_traces

        seen_classes = set()
        seen_lines = set()
        for trace in traces:
            for tl in trace.split("\n"):
                m = re.search(r'File "([^"]+)", line (\d+), in (\w+)', tl)
                if not m:
                    continue
                fpath, lineno_s, func = m.groups()
                fname = os.path.basename(fpath)
                lineno = int(lineno_s)
                loc = f"{fname}:{lineno_s}"

                if loc not in seen_lines:
                    seen_lines.add(loc)
                    line_dur[loc][phase] += per_trace_dur

                cname, mname = _find_class_for_line(fname, lineno, class_map)
                if cname:
                    ckey = f"{fname}:{cname}"
                    if ckey not in seen_classes:
                        seen_classes.add(ckey)
                        class_dur[ckey][phase] += per_trace_dur
                    if mname:
                        cmkey = f"{ckey}.{mname}"
                        if cmkey not in seen_classes:
                            seen_classes.add(cmkey)
                            class_method_dur[cmkey][phase] += per_trace_dur

    # Sort classes by total time
    sorted_classes = sorted(class_dur.items(), key=lambda x: -(x[1]["fwd"] + x[1]["bwd"]))
    sorted_methods = sorted(class_method_dur.items(), key=lambda x: -(x[1]["fwd"] + x[1]["bwd"]))

    # For top classes, collect hot lines
    class_hot_lines = defaultdict(list)
    for loc, phases in line_dur.items():
        parts = loc.split(":")
        fname, lineno = parts[0], int(parts[1])
        cname, mname = _find_class_for_line(fname, lineno, class_map)
        if cname:
            total = phases["fwd"] + phases["bwd"]
            class_hot_lines[f"{fname}:{cname}"].append((lineno, total, phases["fwd"], phases["bwd"], mname))

    for k in class_hot_lines:
        class_hot_lines[k].sort(key=lambda x: -x[1])

    return {
        "class_dur": dict(class_dur),
        "class_method_dur": dict(class_method_dur),
        "line_dur": {k: dict(v) for k, v in line_dur.items()},
        "total_kernel_dur": total_kernel_dur,
        "sorted_classes": sorted_classes,
        "sorted_methods": sorted_methods,
        "class_hot_lines": dict(class_hot_lines),
        "class_map": class_map,
    }


def enrich_kernel_modules_with_source(events, gpu_info, src_info):
    """When source code is available, map top kernel HostModules to actual nn.Module class names
    by using stack_traces to find the deepest user-code class in the call chain."""
    if not gpu_info or not src_info:
        return
    class_map = src_info.get("class_map", {})
    if not class_map:
        return

    # Build kernel -> source class mapping using stack_traces
    kernel_source_class = defaultdict(lambda: defaultdict(float))
    for e in events:
        if e.get("cat") != "kernel":
            continue
        kname = e.get("name", "")
        dur = e.get("dur", 0)
        st = e.get("args", {}).get("stack", {})
        traces = st.get("stack_traces", [])
        if not traces:
            continue
        n_traces = max(1, len(traces))
        per_dur = dur / n_traces
        for trace in traces:
            # Find the most specific (innermost) user source class in the stack
            # Stack traces go from innermost (top) to outermost (bottom) typically,
            # but in PyTorch traces they go outermost first. We want the leaf user class.
            best_class = None
            for tl in trace.split("\n"):
                m = re.search(r'File "([^"]+)", line (\d+), in (\w+)', tl)
                if not m:
                    continue
                fpath, lineno_s, func = m.groups()
                if "site-packages" in fpath or "/usr/" in fpath:
                    continue
                fname = os.path.basename(fpath)
                lineno = int(lineno_s)
                cname, mname = _find_class_for_line(fname, lineno, class_map)
                if cname:
                    best_class = cname  # keep overwriting to get the last (leaf) match
            if best_class:
                kernel_source_class[kname][best_class] += per_dur

    # Enrich top_kernel_modules: replace runtime names with source class names where possible
    enriched = {}
    for kname, runtime_modules in gpu_info["top_kernel_modules"].items():
        source_classes = kernel_source_class.get(kname, {})
        if source_classes:
            sorted_src = sorted(source_classes.items(), key=lambda x: -x[1])
            enriched[kname] = sorted_src
        elif runtime_modules:
            enriched[kname] = runtime_modules
        else:
            enriched[kname] = []
    gpu_info["top_kernel_modules"] = enriched


def analyze_kernel_call_stacks(events, gpu_info):
    if not gpu_info:
        return {}
    kernel_stacks = defaultdict(lambda: {"stacks": [], "total_dur": 0.0, "count": 0})
    for e in events:
        if e.get("cat") != "kernel":
            continue
        name = e.get("name", "")
        dur = e.get("dur", 0)
        st = e.get("args", {}).get("stack", {})
        traces = st.get("stack_traces", [])
        kernel_stacks[name]["total_dur"] += dur
        kernel_stacks[name]["count"] += 1
        if traces and len(kernel_stacks[name]["stacks"]) < 3:
            kernel_stacks[name]["stacks"].append(traces[0])

    top_names = [kname for kname, _ in gpu_info["top_kernels"][:20]]
    result = {}
    for kname in top_names:
        info = kernel_stacks.get(kname)
        if info and info["stacks"]:
            result[kname] = {
                "stack": info["stacks"][0],
                "total_dur": info["total_dur"],
                "count": info["count"],
            }
    return result


LAGRANGE_TORCH_REPO = "https://code.byted.org/lagrange/torch"


def extract_lagrange_refs(events, gpu_info, kernel_call_stacks):
    lagrange_kernels = []
    if not gpu_info:
        return lagrange_kernels
    for kname, info in gpu_info["top_kernels"]:
        is_lagrange = "lagrange" in kname.lower()
        stack_info = kernel_call_stacks.get(kname, {})
        stack_text = stack_info.get("stack", "")
        if is_lagrange or (stack_text and "lagrange" in stack_text.lower()):
            op_name = None
            if "lagrange_torch::" in kname or "lagrange::" in kname:
                m = re.search(r"lagrange(?:_torch)?::(\w+)", kname)
                if m:
                    op_name = m.group(1)
            lagrange_kernels.append({
                "kernel_name": kname,
                "dur": info["dur"],
                "count": info["count"],
                "op_name": op_name,
                "repo_url": LAGRANGE_TORCH_REPO,
                "search_url": f"{LAGRANGE_TORCH_REPO}/search?search={op_name}&type=code" if op_name else None,
            })
    return lagrange_kernels
