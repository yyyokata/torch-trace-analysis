#!/usr/bin/env python3
import argparse
import json
import logging
import os
import re
import sys
from collections import defaultdict

LOG = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Module identity registration.
#
# ``frontend_html.py`` imports ``_validate_timeline_modules`` from this module.
# Some entry points load this file directly via ``importlib.util.spec_from_
# file_location("analyze_trace_runtime", ...)`` (see
# ``testset/test_dag_rules.py::_load_analyze_module``).  In that scenario
# ``__name__`` is NOT ``"analyze_trace"`` and a downstream
# ``from analyze_trace import ...`` would otherwise re-execute this file via
# the normal import machinery, producing two parallel copies of every class.
#
# Registering the half-built module under the canonical name ``analyze_trace``
# (without overwriting an existing entry) makes the partial module visible
# to ``frontend_html`` so imports resolve to *this* instance regardless of
# how it was loaded.
#
# Three loading modes need to work:
#
#   1. Subprocess CLI: ``python3 scripts/analyze_trace.py ...``
#      ``__name__`` is ``"__main__"`` and the module is in ``sys.modules``.
#
#   2. Normal import: ``import analyze_trace``
#      ``__name__`` is ``"analyze_trace"`` so the registration is a no-op.
#
#   3. importlib.util.spec_from_file_location: ``test_dag_rules.py``
#      uses this with module name ``"analyze_trace_runtime"`` and does NOT
#      pre-register the module in ``sys.modules``.  We therefore look the
#      module up via the frame globals (``globals()``), which always points
#      at the currently-executing module dict regardless of how it was
#      created.
# ---------------------------------------------------------------------------
def _self_register_as_analyze_trace():
    """Make the currently-executing module reachable as ``analyze_trace``."""
    self_mod_name = globals().get("__name__")
    self_mod = sys.modules.get(self_mod_name) if self_mod_name else None
    if self_mod is None:
        # importlib loaders sometimes defer sys.modules registration.  Build
        # a lightweight proxy whose ``__getattr__`` delegates to our LIVE
        # frame globals so any name bound later in this file becomes
        # immediately visible through ``analyze_trace.<name>``.  Using
        # ``__getattr__`` (rather than copying ``__dict__``) avoids the
        # read-only ``Module.__dict__`` restriction in CPython.
        import types as _types

        _live_globals = globals()

        class _AnalyzeTraceProxy(_types.ModuleType):
            __slots__ = ()

            def __getattr__(self, name):
                try:
                    return _live_globals[name]
                except KeyError:
                    raise AttributeError(
                        f"module 'analyze_trace' has no attribute {name!r}"
                    ) from None

        self_mod = _AnalyzeTraceProxy("analyze_trace")
    sys.modules.setdefault("analyze_trace", self_mod)


_self_register_as_analyze_trace()
del _self_register_as_analyze_trace

# Ensure this scripts/ directory is on sys.path so the bottom-of-file
# ``from frontend_html import ...`` works even when analyze_trace.py is
# loaded by absolute path (e.g. via importlib.util.spec_from_file_location).
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if _SCRIPT_DIR not in sys.path:
    sys.path.insert(0, _SCRIPT_DIR)


from trace_io import load_model_code, load_trace
from source_hotspot import (
    analyze_kernel_call_stacks,
    analyze_source_hotspots,
    enrich_kernel_modules_with_source,
    extract_lagrange_refs,
)
from report_summary import (
    print_comm_compute_overlap,
    print_device_host_report,
    print_gpu_timeline_report,
    print_header,
    print_kernel_call_stacks,
    print_lagrange_refs,
    print_module_report,
    print_module_tree,
    print_per_thread_timeline,
    print_source_hotspot_report,
    print_thread_overview,
    print_trace_overview,
    save_markdown_report,
)
from common_utils import format_duration
from trace_analysis import (
    add_func_call_parent,
    analyze_comm_compute_overlap,
    analyze_device_host,
    analyze_gpu_timeline,
    analyze_modules_by_thread,
    analyze_per_thread_timeline,
    analyze_step_decomposition,
    analyze_worker_threads,
    build_parent_index,
    classify_threads,
    detect_enhanced_trace,
    detect_trace_type,
    extract_metadata,
    generate_trace_screenshot,
)
from legacy_static_tree import build_static_module_tree


def _normalize_runtime_module_name(mod_name):
    short = (mod_name or "").replace("nn.Module: ", "").replace("nn.Module:", "")
    m_callsite = re.search(r',\s*callsite:\s*(\d+)', short)
    callsite = int(m_callsite.group(1)) if m_callsite else None
    short = re.sub(r',\s*callsite:\s*\d+', '', short).strip()
    m_index = re.search(r'_(\d+)$', short)
    runtime_index = int(m_index.group(1)) if m_index else None
    base = re.sub(r'_\d+$', '', short)
    stripped = re.sub(r'^(FSDP|DDP|DistributedDataParallel|Wrapped)', '', base)
    return {
        "runtime_name": short,
        "class_name": stripped,
        "runtime_index": runtime_index,
        "callsite": callsite,
    }


def _build_external_id_to_module_map(events):
    by_tid = defaultdict(list)
    for e in events:
        if e.get("ph") != "X" or e.get("dur") is None:
            continue
        if e.get("cat") == "kernel":
            continue
        by_tid[e.get("tid")].append(e)

    ext_to_module = {}
    for tid, tid_events in by_tid.items():
        tid_events.sort(key=lambda e: (e["ts"], -(e.get("dur") or 0), 0 if e.get("cat") == "python_function" else 1))
        module_stack = []
        for e in tid_events:
            start = e["ts"]
            end = start + (e.get("dur") or 0)
            while module_stack and module_stack[-1][1] <= start:
                module_stack.pop()

            is_module = e.get("cat") == "python_function" and str(e.get("name", "")).startswith("nn.Module:")
            if is_module:
                module_stack.append((e, end))

            ext_id = e.get("args", {}).get("External id")
            if ext_id is not None and module_stack:
                ext_to_module.setdefault(ext_id, module_stack[-1][0].get("name"))

    return ext_to_module


# ==========================================================================
# Timing Pipeline Redesign — 4-step instance-level attribution (cherry-picked)




def _build_runtime_module_hierarchy(events):
    """Build module containment hierarchy from trace events.
    Returns: {class_name: {"children": [child_names], "parents": set(parent_names)}}
    """
    mod_events = []
    for e in events:
        name = e.get('name', '')
        if not name.startswith('nn.Module:') or e.get('ph') != 'X':
            continue
        short = name.replace('nn.Module: ', '').replace('nn.Module:', '')
        short = re.sub(r',\s*callsite:\s*\d+', '', short).strip()
        base = re.sub(r'_\d+$', '', short)
        mod_events.append({
            'name': base,
            'ts': e['ts'],
            'dur': e.get('dur', 0),
            'end': e['ts'] + e.get('dur', 0),
            'tid': e.get('tid'),
        })

    parent_child = defaultdict(set)
    child_parent = defaultdict(set)
    all_tids = set(e['tid'] for e in mod_events)
    top_level = set()

    for tid in all_tids:
        tid_events = [e for e in mod_events if e['tid'] == tid]
        tid_events.sort(key=lambda x: (x['ts'], -x['dur']))
        stack = []
        for e in tid_events:
            while stack and stack[-1]['end'] <= e['ts']:
                stack.pop()
            if stack:
                parent_child[stack[-1]['name']].add(e['name'])
                child_parent[e['name']].add(stack[-1]['name'])
            else:
                top_level.add(e['name'])
            stack.append(e)

    hierarchy = {}
    all_names = set(e['name'] for e in mod_events)
    for name in all_names:
        children = sorted(parent_child.get(name, set()))
        parents = child_parent.get(name, set())
        hierarchy[name] = {"children": children, "parents": parents}

    return hierarchy, top_level


def _merge_runtime_into_tree(tree, roots, runtime_hierarchy, runtime_top_level, class_durations):
    """Merge runtime-only modules into the static tree so all timeline modules are connected.
    Runtime modules not in source tree get virtual entries connected via containment edges.
    DDPRootModule -> RootModule connection is established by stripping DDP prefix.
    """
    static_classes = set(tree.keys())
    runtime_classes = set(runtime_hierarchy.keys())
    runtime_only = runtime_classes - static_classes

    if not runtime_only:
        return tree, roots

    # Establish mapping from runtime wrapper to source root
    # DDPRootModule/FSDPRootModule -> RootModule
    wrapper_to_source = {}
    for rmod in runtime_only:
        stripped = re.sub(r'^(FSDP|DDP|DistributedDataParallel|Wrapped)', '', rmod)
        if stripped in static_classes and stripped != rmod:
            wrapper_to_source[rmod] = stripped

    # Add runtime-only modules to tree
    for rmod in runtime_only:
        rinfo = runtime_hierarchy[rmod]
        # Children: only include those that are also runtime-only OR are wrapper targets
        children = []
        attrs = {}
        for child in rinfo["children"]:
            if child in runtime_only or child in wrapper_to_source:
                attr_name = child[0].lower() + child[1:]
                children.append(child)
                attrs[attr_name] = child
            elif child in static_classes:
                attr_name = child[0].lower() + child[1:]
                children.append(child)
                attrs[attr_name] = child

        # If this runtime module wraps a source module, add that as child
        if rmod in wrapper_to_source:
            src_target = wrapper_to_source[rmod]
            if src_target not in children:
                attr_name = src_target[0].lower() + src_target[1:]
                children.append(src_target)
                attrs[attr_name] = src_target

        # Build sequential dep edges (runtime modules are pipeline-sequential)
        dep_edges = []
        attr_names_ordered = list(attrs.keys())
        for i in range(len(attr_names_ordered) - 1):
            dep_edges.append((attr_names_ordered[i], attr_names_ordered[i + 1]))

        tree[rmod] = {"children": children, "attrs": attrs, "dep_edges": dep_edges,
                       "dep_edge_locs": {}, "attr_def_loc": {}}

    # Determine new roots: runtime top-level modules that contain everything
    # Prefer a single root that encompasses all others
    new_roots = []
    # Find the broadest runtime top-level module (most descendants)
    def count_rt_descendants(name, visited=None):
        if visited is None:
            visited = set()
        if name in visited or name not in tree:
            return 0
        visited.add(name)
        total = len(tree[name]["children"])
        for child in tree[name]["children"]:
            total += count_rt_descendants(child, visited)
        return total

    # Runtime top-level modules become candidate roots
    rt_roots = [r for r in runtime_top_level if r in tree]
    if rt_roots:
        rt_roots_sorted = sorted(rt_roots, key=lambda r: -count_rt_descendants(r))
        # All runtime top-level modules are roots (they are independent execution units)
        new_roots = list(rt_roots_sorted)
        # Check reachability from all runtime roots
        reachable = set()
        def collect_reachable(name, visited):
            if name in visited or name not in tree:
                return
            visited.add(name)
            reachable.add(name)
            for child in tree[name]["children"]:
                collect_reachable(child, visited)
        for rr in rt_roots:
            collect_reachable(rr, set())

        # Add source roots that are NOT reachable from any runtime root
        for sr in roots:
            if sr not in reachable:
                new_roots.append(sr)
    else:
        new_roots = roots

    return tree, new_roots


def _validate_timeline_modules(trace_events, tree):
    """Validate that all user model nn.Module classes in timeline are present in the DAG tree.
    Runtime wrapper modules (not in user source) are expected to be absent.
    Returns validation info dict.
    """
    if not trace_events:
        return None

    timeline_modules = set()
    for e in trace_events:
        name = e.get('name', '')
        if not name.startswith('nn.Module:') or e.get('ph') != 'X':
            continue
        short = name.replace('nn.Module: ', '').replace('nn.Module:', '')
        short = re.sub(r',\s*callsite:\s*\d+', '', short).strip()
        base = re.sub(r'_\d+$', '', short)
        timeline_modules.add(base)

    if not timeline_modules:
        return None

    tree_classes = set(tree.keys())
    # Timeline modules that ARE in the source tree (user model modules)
    in_tree = timeline_modules & tree_classes
    # Timeline modules that are NOT in source tree (runtime wrappers)
    runtime_only = timeline_modules - tree_classes

    # Check connectivity: all tree classes should be reachable from root via dep_edges or children
    connected = set()
    def _check_connected(cname, visited=None):
        if visited is None:
            visited = set()
        if cname in visited or cname not in tree:
            return
        visited.add(cname)
        connected.add(cname)
        for child in tree[cname].get("children", []):
            _check_connected(child, visited)
    for root in tree_classes:
        if root not in [c for info in tree.values() for c in info.get("children", [])]:
            _check_connected(root)

    validation = {
        "timeline_total": len(timeline_modules),
        "in_source_tree": sorted(in_tree),
        "runtime_wrappers": sorted(runtime_only),
        "tree_classes": sorted(tree_classes),
        "connected_classes": sorted(connected),
        "all_connected": tree_classes == connected,
    }

    # Print validation summary
    print("  📋 Timeline nn.Module 验证:")
    print(f"     Timeline 中共 {len(timeline_modules)} 个 Module 类")
    if in_tree:
        print(f"     ✅ 用户模型 Module (在 DAG 中): {sorted(in_tree)}")
    if runtime_only:
        print(f"     ℹ️  运行时 wrapper Module (不在用户源码中, 无需加入 DAG): {len(runtime_only)} 个")
    disconnected = tree_classes - connected
    if disconnected:
        print(f"     ⚠️  DAG 中未连接的类: {sorted(disconnected)}")
    else:
        print(f"     ✅ DAG 中所有 {len(tree_classes)} 个源码 Module 类均已正确连接")

    # Cross-check: any timeline module that has the same name as a source nn.Module
    # MUST also exist in the DAG tree. If a class is defined in user source but
    # filtered out of the tree (e.g. because the wrong root was picked), surface it
    # as a validation failure so the caller can iterate.
    # We re-load the full nn_module_classes list from class_map by inspecting the
    # tree's union with whatever we know. For a stricter check the caller should
    # have already used preferred_root logic.
    # Validation output above is the authoritative pass/fail summary.

    return validation


# ---------------------------------------------------------------------------
# HTML flowchart rendering lives in frontend_html.py.
#
# ``main()`` performs a function-scope import to avoid circular imports.  This
# file intentionally does not re-export frontend_html entry points.
# ---------------------------------------------------------------------------


def main():
    # Lazy, function-scope import of the HTML rendering entry points keeps the
    # frontend dependency acyclic while still letting the CLI use them.
    from frontend_html import (
        generate_html_flowchart_dual,
        build_timing_data_from_trace,
    )

    parser = argparse.ArgumentParser(description="分析 PyTorch 训练 trace JSON 文件")
    parser.add_argument("trace_file", nargs="?", default=None, help="trace JSON 文件路径 (可选，若仅生成源码流程图则不需要)")
    parser.add_argument("--max-level", type=int, default=None, help="最大显示的 Module 层级深度")
    parser.add_argument("--output", "-o", type=str, default=None, help="输出 Markdown 报告文件路径")
    parser.add_argument("--no-tree", action="store_true", help="不输出 Module 层级树")
    parser.add_argument("--json-output", type=str, default=None, help="输出 JSON 格式分析结果")
    parser.add_argument("--code-path", type=str, default=None, help="模型源码路径（目录或 .tar.gz）")
    parser.add_argument("--screenshot", action="store_true", help="生成 Chrome Tracing 可视化截图")
    parser.add_argument("--html-flowchart", type=str, default=None, help="生成 HTML 模块流程图路径")
    args = parser.parse_args()

    # Mode 1: Source-code only flowchart (no trace required)
    if args.html_flowchart and args.code_path and not args.trace_file:
        print(f"  正在加载模型源码: {args.code_path}")
        source_files = load_model_code(args.code_path)
        print(f"  加载了 {len(source_files)} 个源文件: {', '.join(sorted(source_files.keys())[:10])}")
        print("  正在生成 HTML 模块流程图 (纯源码静态结构, 双 Tab: 训练/推理)...")
        result = generate_html_flowchart_dual(source_files, timing_data=None, meta=None, output_path=args.html_flowchart)
        if result:
            print(f"  HTML 流程图已保存到: {args.html_flowchart}")
        return

    # Mode 2: Full trace analysis (trace_file required)
    if not args.trace_file:
        print("  ❌ 错误: 需要提供 trace 文件路径，或同时指定 --code-path 和 --html-flowchart 生成纯源码流程图")
        sys.exit(1)

    print(f"  正在加载 trace 文件: {args.trace_file}")
    data = load_trace(args.trace_file)
    events = data.get("traceEvents", [])
    print(f"  共 {len(events)} 个事件")

    # Metadata
    meta = extract_metadata(data, events)

    # Detect trace type
    trace_type = detect_trace_type(events)
    if trace_type != "training":
        print(f"  ⚠️ 检测到这是 {trace_type} trace，当前 skill 仅支持训练 trace 分析")
        print("  如确认是训练 trace，将继续分析")

    # Detect enhanced trace (Code Location + stack_traces)
    has_code_loc, has_stack_traces = detect_enhanced_trace(events)
    is_enhanced = has_code_loc and has_stack_traces
    if is_enhanced:
        print("  检测到增强 trace（含 Code Location 和 stack_traces）")
        if not args.code_path:
            print("  ❌ 错误: 增强 trace 需要提供模型源码才能进行源码级分析")
            print("  请通过 --code-path 参数提供模型源码目录或 tar.gz 压缩包")
            print("  示例: python3 scripts/analyze_trace.py trace.json --code-path code_commit.tar.gz")
            sys.exit(1)

    # Load model source code
    source_files = {}
    if args.code_path:
        print(f"  正在加载模型源码: {args.code_path}")
        source_files = load_model_code(args.code_path)
        print(f"  加载了 {len(source_files)} 个源文件: {', '.join(sorted(source_files.keys())[:10])}")
        if len(source_files) > 10:
            print(f"    ... 等共 {len(source_files)} 个文件")

    # Step duration: collect ALL ProfilerStep events and compute average
    profiler_steps = []
    for e in events:
        if e.get("cat") == "user_annotation" and "ProfilerStep" in e.get("name", ""):
            profiler_steps.append({"ts": e["ts"], "dur": e.get("dur", 0), "end": e["ts"] + e.get("dur", 0)})
    profiler_steps.sort(key=lambda x: x["ts"])
    num_steps = len(profiler_steps)

    if num_steps > 0:
        step_dur_us = sum(s["dur"] for s in profiler_steps) / num_steps
    else:
        step_dur_us = 0
        for e in events:
            if e.get("cat") == "Trace":
                step_dur_us = e.get("dur", 0)
                break
        if step_dur_us == 0:
            ts_vals = [e.get("ts", 0) for e in events if e.get("ph") == "X"]
            end_vals = [e.get("ts", 0) + e.get("dur", 0) for e in events if e.get("ph") == "X"]
            if ts_vals and end_vals:
                step_dur_us = max(end_vals) - min(ts_vals)

    print(f"  ProfilerStep 数量: {num_steps}")
    print(f"  平均 Step 耗时: {format_duration(step_dur_us)}")

    # Build parent relationships
    print("  正在构建事件调用关系...")
    build_parent_index(events)
    add_func_call_parent(events)

    # Classify threads
    print("  正在分析线程结构...")
    thread_info = classify_threads(events, meta)

    # Step decomposition
    step_decomp = analyze_step_decomposition(events, thread_info, profiler_steps)

    # Module analysis
    print("  正在分析 Module 耗时...")
    mod_info = analyze_modules_by_thread(events, thread_info, step_dur_us, profiler_steps)
    print(f"  发现 {len(mod_info['depth_map'])} 种 Module 类型")

    # GPU timeline analysis
    print("  正在分析 GPU 时间线...")
    gpu_info = analyze_gpu_timeline(events, thread_info, step_dur_us, profiler_steps)

    # Device/Host analysis
    print("  正在分析 Device/Host 耗时...")
    dh_info = analyze_device_host(events, step_dur_us, thread_info, profiler_steps)

    # Worker thread analysis
    workers = analyze_worker_threads(events, thread_info, meta)

    # Source code hotspot analysis
    src_info = None
    if source_files and has_stack_traces:
        print("  正在分析源码热点...")
        src_info = analyze_source_hotspots(events, source_files)
        print(f"  映射到 {len(src_info['sorted_classes'])} 个 Class, {len(src_info['sorted_methods'])} 个 Method")
        # Enrich top kernel HostModule mapping with source-code nn.Module class names
        if gpu_info:
            print("  正在将 TopKernel HostModule 映射到源码 nn.Module...")
            enrich_kernel_modules_with_source(events, gpu_info, src_info)

    # Per-thread timeline analysis
    print("  正在分析多线程时间线...")
    timelines = analyze_per_thread_timeline(events, thread_info, meta)
    print(f"  分析了 {len(timelines)} 个线程时间线")

    # Kernel call stack analysis
    kernel_stacks = {}
    if gpu_info and is_enhanced:
        print("  正在提取 Kernel 调用栈...")
        kernel_stacks = analyze_kernel_call_stacks(events, gpu_info)
        print(f"  提取了 {len(kernel_stacks)} 个 Kernel 的调用栈")

    # Communication / Compute overlap analysis
    print("  正在分析通信/计算 Overlap...")
    overlap_info = analyze_comm_compute_overlap(events, thread_info, profiler_steps)
    if overlap_info:
        print(f"  通信流: {overlap_info['num_comm_streams']} 条, Memcpy 流: {overlap_info['num_memcpy_streams']} 条")
        if overlap_info["comm_total"] > 0:
            print(f"  通信 overlap: {overlap_info['comm_overlap_pct']:.1f}%, 暴露开销: {format_duration(overlap_info['comm_exposed'])}")

    # lagrange_torch references
    lagrange_refs = extract_lagrange_refs(events, gpu_info, kernel_stacks)
    if lagrange_refs:
        print(f"  发现 {len(lagrange_refs)} 个 lagrange_torch 相关 Kernel")

    # Screenshot
    screenshot_path = None
    if args.screenshot:
        print("  正在生成 Trace 可视化截图...")
        output_dir = os.path.dirname(args.output) if args.output else "."
        if not output_dir:
            output_dir = "."
        screenshot_path, err = generate_trace_screenshot(args.trace_file, output_dir)
        if screenshot_path:
            print(f"  截图已保存到: {screenshot_path}")
        else:
            print(f"  ⚠️ 截图生成失败: {err}")

    print()

    # === Print Reports ===
    print_trace_overview(meta, trace_type, step_dur_us, thread_info, step_decomp, num_steps)
    print_thread_overview(thread_info, meta, workers)

    # Per-thread timeline
    print_per_thread_timeline(timelines)

    if not args.no_tree:
        print_header("Module 层级结构")
        print_module_tree(mod_info, "ROOT", 0, step_dur_us, mod_info.get("module_thread_info"))
        print()

    print_module_report(mod_info, step_dur_us, args.max_level)
    print_gpu_timeline_report(gpu_info, step_dur_us)

    # Kernel call stacks
    if kernel_stacks:
        print_kernel_call_stacks(kernel_stacks, gpu_info)

    print_device_host_report(dh_info)

    # Comm/compute overlap
    if overlap_info:
        print_comm_compute_overlap(overlap_info, gpu_info)

    if src_info:
        print_source_hotspot_report(src_info, source_files)

    # lagrange_torch references
    if lagrange_refs:
        print_lagrange_refs(lagrange_refs)

    # Save markdown
    if args.output:
        save_markdown_report(meta, trace_type, step_dur_us, step_decomp, thread_info, workers,
                             mod_info, gpu_info, dh_info, args.output, args.max_level,
                             src_info=src_info, source_files=source_files,
                             timelines=timelines, kernel_stacks=kernel_stacks,
                             lagrange_refs=lagrange_refs, screenshot_path=screenshot_path,
                             num_steps=num_steps, overlap_info=overlap_info)

    # Save JSON
    if args.json_output:
        result = {
            "trace_type": trace_type,
            "device": meta.get("device_name", "N/A"),
            "step_duration_us": step_dur_us,
            "step_decomposition": step_decomp,
            "modules": {
                name.replace("nn.Module: ", ""): {
                    "inclusive_us": mod_info["module_durations"].get(name, 0),
                    "exclusive_us": mod_info["module_exclusive"].get(name, 0),
                    "inclusive_pct": mod_info["module_durations"].get(name, 0) / step_dur_us * 100 if step_dur_us > 0 else 0,
                    "exclusive_pct": mod_info["module_exclusive"].get(name, 0) / step_dur_us * 100 if step_dur_us > 0 else 0,
                    "calls": mod_info["module_call_counts"].get(name, 0),
                    "depth": mod_info["depth_map"].get(name, 0),
                    "parent": mod_info["module_parent_map"].get(name, "ROOT").replace("nn.Module: ", ""),
                    "thread": list(mod_info["module_thread_info"].get(name, {}).get("roles", set())),
                }
                for name in mod_info["module_durations"]
            },
            "gpu_timeline": {
                "gpu_utilization": gpu_info["gpu_utilization"] if gpu_info else 0,
                "total_kernel_dur_us": gpu_info["total_kernel_dur"] if gpu_info else 0,
                "total_gap_us": gpu_info["total_gap"] if gpu_info else 0,
                "num_kernels": gpu_info["num_kernels"] if gpu_info else 0,
                "top_kernels": [
                    {
                        "name": kname,
                        "dur_us": info["dur"],
                        "count": info["count"],
                        "host_module": (gpu_info["top_kernel_modules"].get(kname, []) or [("N/A",)])[0][0],
                    }
                    for kname, info in (gpu_info["top_kernels"] if gpu_info else [])
                ],
            } if gpu_info else None,
            "device_host": {
                "host_total_us": dh_info["host_total_us"],
                "device_total_us": dh_info["device_total_us"],
                "host_pct": dh_info["host_total_us"] / step_dur_us * 100 if step_dur_us > 0 else 0,
                "device_pct": dh_info["device_total_us"] / step_dur_us * 100 if step_dur_us > 0 else 0,
                "device_breakdown": {
                    cat: {"dur_us": dur, "pct": dur / step_dur_us * 100 if step_dur_us > 0 else 0}
                    for cat, dur in dh_info["device_breakdown"].items()
                },
            },
            "comm_compute_overlap": {
                "comm_total_us": overlap_info["comm_total"],
                "comm_overlap_us": overlap_info["comm_overlap"],
                "comm_exposed_us": overlap_info["comm_exposed"],
                "comm_overlap_pct": overlap_info["comm_overlap_pct"],
                "memcpy_total_us": overlap_info["memcpy_total"],
                "memcpy_overlap_us": overlap_info["memcpy_overlap"],
                "memcpy_exposed_us": overlap_info["memcpy_exposed"],
                "memcpy_overlap_pct": overlap_info["memcpy_overlap_pct"],
                "compute_total_us": overlap_info["compute_total"],
            } if overlap_info else None,
        }
        with open(args.json_output, "w") as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
        print(f"  JSON 结果已保存到: {args.json_output}")

    # HTML Flowchart (source-code first, timing overlay optional)
    if args.html_flowchart:
        if not source_files:
            print("  ⚠️ 跳过 HTML 流程图生成: 需要提供源码 (--code-path)")
            print("  流程图基于源码静态结构生成，无源码时无法确定模块层级关系")
        else:
            # Extract roots for timing rollup (Step 3 orphan aggregation)
            static_module_tree, roots, _ = build_static_module_tree(source_files, conditional_mode="train")
            # Build timing data from trace to overlay on static structure
            timing_data = build_timing_data_from_trace(events, source_files, mod_info, step_dur_us, profiler_steps, src_info=src_info, roots=roots, static_module_tree=static_module_tree)
            print("  正在生成 HTML 模块流程图 (源码结构 + 运行时层级 + 时间填充, 双 Tab: 训练/推理)...")
            result = generate_html_flowchart_dual(source_files, timing_data=timing_data, meta=meta, output_path=args.html_flowchart, trace_events=events)
            if result:
                print(f"  HTML 流程图已保存到: {args.html_flowchart}")

if __name__ == "__main__":
    main()
