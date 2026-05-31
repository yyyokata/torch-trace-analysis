#!/usr/bin/env python3

import os
import re
from collections import defaultdict, deque


def extract_metadata(data, events):
    meta = {}
    # Device info
    dp = data.get("deviceProperties", [])
    if dp:
        meta["device_name"] = dp[0].get("name", "Unknown")
        meta["num_sms"] = dp[0].get("numSms", 0)
    # Distributed info
    di = data.get("distributedInfo", {})
    if di:
        meta["rank"] = di.get("rank", 0)
        meta["world_size"] = di.get("world_size", 1)
    # Thread names & process labels
    thread_names = {}
    process_labels = {}
    for e in events:
        if e.get("ph") != "M":
            continue
        if e.get("name") == "thread_name":
            thread_names[(e.get("pid"), e.get("tid"))] = e.get("args", {}).get("name", "")
        elif e.get("name") == "process_labels":
            process_labels[e.get("pid")] = e.get("args", {}).get("labels", "")
    meta["thread_names"] = thread_names
    meta["process_labels"] = process_labels
    return meta


def detect_trace_type(events):
    has_fwdbwd = any(e.get("cat") == "fwdbwd" for e in events)
    has_backward = any("backward" in e.get("name", "").lower() or "Backward" in e.get("name", "")
                       for e in events if e.get("cat") in ("cpu_op", "python_function"))
    if has_fwdbwd or has_backward:
        return "training"
    return "inference"


def detect_enhanced_trace(events):
    has_code_location = False
    has_stack_traces = False
    for e in events:
        if e.get("name") == "thread_name" and e.get("ph") == "M":
            if "Code Location" in e.get("args", {}).get("name", ""):
                has_code_location = True
        if has_code_location and has_stack_traces:
            break
    for e in events:
        if e.get("cat") == "kernel":
            st = e.get("args", {}).get("stack", {})
            if "stack_traces" in st:
                has_stack_traces = True
                break
    return has_code_location, has_stack_traces


def build_main_thread_hierarchy(events):
    profiler_steps = [
        e for e in events
        if str(e.get("name", "")).startswith("ProfilerStep")
        and e.get("ph") == "X"
        and e.get("cat") == "user_annotation"
    ]
    if not profiler_steps:
        return None, [], defaultdict(list)
    main_tid = max(
        {e["tid"] for e in profiler_steps},
        key=lambda tid: sum(1 for e in profiler_steps if e["tid"] == tid),
    )
    main_events = sorted(
        [
            e for e in events
            if e.get("tid") == main_tid and e.get("ph") == "X" and e.get("dur") is not None
        ],
        key=lambda e: (e["ts"], -(e.get("dur") or 0)),
    )

    children = defaultdict(list)
    stack = []
    for e in main_events:
        start = e["ts"]
        end = start + (e.get("dur") or 0)
        while stack and stack[-1][1] <= start:
            stack.pop()
        if stack:
            children[id(stack[-1][0])].append(e)
        stack.append((e, end))

    return main_tid, main_events, children


def phase_name_from_label(name):
    lower = str(name).lower()
    if "optimize" in lower or "optimizer.step" in lower:
        return "optimize"
    if re.search(r"(^|[^a-z])backward([^a-z]|$)", lower):
        return "backward"
    if re.search(r"(^|[^a-z])forward([^a-z]|$)", lower):
        return "forward"
    return None


def extract_step_phase_intervals(main_events, children):
    profiler_steps = [
        e for e in main_events
        if str(e.get("name", "")).startswith("ProfilerStep") and e.get("cat") == "user_annotation"
    ]
    step_infos = []
    for profiler_step in profiler_steps:
        direct_children = children[id(profiler_step)]
        if not direct_children:
            continue
        step_event = max(direct_children, key=lambda e: e.get("dur", 0))
        info = {
            "profiler_name": profiler_step["name"],
            "step_interval": (profiler_step["ts"], profiler_step["ts"] + profiler_step["dur"]),
            "forward": [],
            "backward": [],
            "optimize": [],
        }
        for level2 in children[id(step_event)]:
            level2_name = str(level2.get("name", ""))
            level2_name_lower = level2_name.lower()
            if "forward_backward" in level2_name_lower:
                for level3 in children[id(level2)]:
                    phase = phase_name_from_label(level3.get("name", ""))
                    if phase in ("forward", "backward"):
                        info[phase].append((level3["ts"], level3["ts"] + level3["dur"], level3.get("name", "")))
        optimize_intervals = []
        seen_opt = set()
        for desc in _iter_descendants_within_step(step_event, children):
            phase = phase_name_from_label(desc.get("name", ""))
            if phase != "optimize":
                continue
            ts = float(desc.get("ts") or 0.0)
            end = ts + float(desc.get("dur") or 0.0)
            opt_key = (ts, end, str(desc.get("name", "")))
            if opt_key in seen_opt:
                continue
            seen_opt.add(opt_key)
            optimize_intervals.append((ts, end, desc.get("name", "")))
        info["optimize"] = _merge_phase_intervals(optimize_intervals)
        step_infos.append(info)
    return step_infos


def overlap(a_start, a_end, b_start, b_end):
    return max(0.0, min(a_end, b_end) - max(a_start, b_start))


def classify_kernel_phase(kernel_start, kernel_end, step_infos):
    matched_step = None
    for info in step_infos:
        step_start, step_end = info["step_interval"]
        if kernel_end >= step_start and kernel_start <= step_end:
            matched_step = info
            break
    if matched_step is None:
        return "other"

    phase_overlap = {"forward": 0.0, "backward": 0.0, "optimize": 0.0}
    for phase in ("forward", "backward", "optimize"):
        for start, end, _label in matched_step[phase]:
            phase_overlap[phase] += overlap(kernel_start, kernel_end, start, end)

    phase_priority = {"forward": 0, "backward": 1, "optimize": 2}
    ranked = sorted(phase_overlap.items(), key=lambda item: (item[1], phase_priority[item[0]]), reverse=True)
    best_phase, best_overlap = ranked[0]
    return best_phase if best_overlap > 0 else "other"


def _iter_descendants_within_step(step_event, children_map):
    if not step_event or not children_map:
        return []
    step_start = float(step_event.get("ts") or 0.0)
    step_end = step_start + float(step_event.get("dur") or 0.0)
    out = []
    queue = deque(children_map.get(id(step_event), []) or [])
    while queue:
        ev = queue.popleft()
        ts = float(ev.get("ts") or 0.0)
        dur = float(ev.get("dur") or 0.0)
        end = ts + dur
        if ts < step_start or end > step_end:
            continue
        out.append(ev)
        for child in children_map.get(id(ev), []) or []:
            queue.append(child)
    return out


def _merge_phase_intervals(intervals):
    normalized = []
    for item in intervals or []:
        if not item or len(item) < 2:
            continue
        start = float(item[0])
        end = float(item[1])
        label = item[2] if len(item) >= 3 else ""
        if end < start:
            start, end = end, start
        normalized.append((start, end, label))
    if not normalized:
        return []
    normalized.sort(key=lambda x: (x[0], x[1]))
    merged = [normalized[0]]
    for start, end, label in normalized[1:]:
        last_start, last_end, last_label = merged[-1]
        if start <= last_end:
            merged[-1] = (last_start, max(last_end, end), last_label)
        else:
            merged.append((start, end, label))
    return merged


def analyze_per_thread_timeline(events, thread_info, meta):
    thread_names = meta["thread_names"]
    cpu_pid = thread_info["cpu_pid"]
    gpu_pids = thread_info["gpu_pids"]
    thread_roles = thread_info["thread_roles"]

    timelines = {}

    # Main thread timeline
    main_tid = thread_info["main_fb_thread"] or thread_info["step_thread"]
    if main_tid:
        main_events = []
        for e in events:
            if e.get("pid") == main_tid[0] and e.get("tid") == main_tid[1] and e.get("ph") == "X":
                if e.get("cat") in ("python_function", "cpu_op", "user_annotation") and e.get("dur", 0) > 500:
                    p_idx = e.get("args", {}).get("_P", -1)
                    is_top = p_idx == -1 or e.get("dur", 0) > 10000
                    main_events.append({
                        "name": e.get("name", ""),
                        "cat": e.get("cat", ""),
                        "ts": e.get("ts", 0),
                        "dur": e.get("dur", 0),
                        "is_top": is_top,
                    })
        main_events.sort(key=lambda x: x["ts"])
        top_events = sorted(main_events, key=lambda x: -x["dur"])[:20]
        tname = thread_names.get(main_tid, "main")
        timelines["main_thread"] = {
            "tid": main_tid,
            "name": tname,
            "total_events": len(main_events),
            "top_events": top_events,
        }

    # GPU ProfilerStep timeline (primary stream)
    primary_stream = thread_info["primary_gpu_stream"]
    if primary_stream:
        gpu_timeline_events = []
        for e in events:
            if e.get("pid") == primary_stream[0] and e.get("tid") == primary_stream[1] and e.get("ph") == "X":
                if e.get("cat") in ("kernel", "gpu_memcpy", "gpu_memset", "gpu_user_annotation"):
                    gpu_timeline_events.append({
                        "name": e.get("name", ""),
                        "cat": e.get("cat", ""),
                        "ts": e.get("ts", 0),
                        "dur": e.get("dur", 0),
                    })
        gpu_timeline_events.sort(key=lambda x: x["ts"])
        top_gpu_events = sorted(gpu_timeline_events, key=lambda x: -x["dur"])[:20]
        sname = thread_names.get(primary_stream, f"stream {primary_stream[1]}")
        timelines["gpu_primary_stream"] = {
            "tid": primary_stream,
            "name": sname,
            "total_events": len(gpu_timeline_events),
            "top_events": top_gpu_events,
        }

    # Other GPU streams with significant activity
    for (pid, tid), n_kernels in thread_info["gpu_stream_stats"].items():
        if (pid, tid) == primary_stream:
            continue
        if n_kernels < 10:
            continue
        stream_events = []
        for e in events:
            if e.get("pid") == pid and e.get("tid") == tid and e.get("ph") == "X":
                if e.get("cat") in ("kernel", "gpu_memcpy", "gpu_memset"):
                    stream_events.append({
                        "name": e.get("name", ""),
                        "cat": e.get("cat", ""),
                        "ts": e.get("ts", 0),
                        "dur": e.get("dur", 0),
                    })
        if stream_events:
            stream_events.sort(key=lambda x: x["ts"])
            top_stream = sorted(stream_events, key=lambda x: -x["dur"])[:10]
            sname = thread_names.get((pid, tid), f"stream {tid}")
            timelines[f"gpu_stream_{tid}"] = {
                "tid": (pid, tid),
                "name": sname,
                "total_events": len(stream_events),
                "top_events": top_stream,
            }

    # Worker threads
    for (pid, tid), role_info in thread_roles.items():
        if role_info["role"] not in ("worker_thread", "autograd_thread"):
            continue
        worker_events = []
        for e in events:
            if e.get("pid") == pid and e.get("tid") == tid and e.get("ph") == "X":
                if e.get("dur", 0) > 500:
                    worker_events.append({
                        "name": e.get("name", ""),
                        "cat": e.get("cat", ""),
                        "ts": e.get("ts", 0),
                        "dur": e.get("dur", 0),
                    })
        if worker_events:
            worker_events.sort(key=lambda x: x["ts"])
            top_w = sorted(worker_events, key=lambda x: -x["dur"])[:10]
            tname = thread_names.get((pid, tid), f"thread {tid}")
            role = role_info["role"]
            timelines[f"{role}_{tid}"] = {
                "tid": (pid, tid),
                "name": tname,
                "role": role,
                "total_events": len(worker_events),
                "top_events": top_w,
            }

    return timelines


def generate_trace_screenshot(trace_file, output_dir):
    screenshot_path = os.path.join(output_dir, "trace_screenshot.png")
    try:
        from playwright.sync_api import sync_playwright
    except ImportError:
        return None, "playwright not installed"

    abs_trace = os.path.abspath(trace_file)
    trace_size_mb = os.path.getsize(abs_trace) / (1024 * 1024)
    if trace_size_mb > 200:
        return None, f"trace file too large ({trace_size_mb:.0f}MB) for browser visualization"

    try:
        with sync_playwright() as p:
            browser = p.chromium.launch(headless=True, args=["--no-sandbox", "--disable-gpu"])
            page = browser.new_page(viewport={"width": 1920, "height": 1080})
            page.goto("https://ui.perfetto.dev/", timeout=30000)
            page.wait_for_load_state("networkidle", timeout=30000)
            page.set_input_files('input[type="file"]', abs_trace, timeout=10000)
            page.wait_for_timeout(8000)
            page.screenshot(path=screenshot_path, full_page=False)
            browser.close()
        if os.path.exists(screenshot_path):
            return screenshot_path, None
    except Exception as ex:
        return None, str(ex)
    return None, "screenshot failed"


def build_parent_index(events):
    for i, event in enumerate(events):
        if "args" not in event:
            event["args"] = {}
        event["args"]["_IDX"] = i


def add_func_call_parent(events):
    thread_indices = defaultdict(list)
    for event in events:
        pid = event.get("pid", -1)
        tid = event.get("tid", -1)
        if pid != -1 and tid != -1 and event.get("ph") == "X":
            thread_indices[(pid, tid)].append(event["args"]["_IDX"])

    for key in thread_indices:
        thread_indices[key].sort(key=lambda i: events[i]["ts"])

    for _, indices in thread_indices.items():
        stack = [(-1, -1e50, 1e50)]
        for idx in indices:
            while stack[-1][0] != -1 and events[idx]["ts"] > stack[-1][2]:
                stack.pop()
            events[idx]["args"]["_P"] = stack[-1][0]
            stack.append(
                (idx, events[idx]["ts"], events[idx]["ts"] + events[idx].get("dur", 0))
            )
    return thread_indices


def classify_threads(events, meta):
    thread_names = meta["thread_names"]
    process_labels = meta["process_labels"]

    cpu_pid = None
    gpu_pids = set()
    for pid, label in process_labels.items():
        if "CPU" in label:
            cpu_pid = pid
        elif "GPU" in label or "MLU" in label:
            gpu_pids.add(pid)

    # Collect per-thread event stats
    thread_stats = defaultdict(lambda: {"count": 0, "dur": 0.0, "cats": defaultdict(int), "has_modules": False})
    for e in events:
        if e.get("ph") != "X":
            continue
        key = (e.get("pid", -1), e.get("tid", -1))
        ts = thread_stats[key]
        ts["count"] += 1
        ts["dur"] += e.get("dur", 0)
        ts["cats"][e.get("cat", "")] += 1
        if e.get("cat") == "python_function" and e.get("name", "").startswith("nn.Module:"):
            ts["has_modules"] = True

    # Identify main forward/backward thread
    main_fb_thread = None
    for e in events:
        name = e.get("name", "")
        if "forward_backward" in name and e.get("cat") == "python_function":
            main_fb_thread = (e.get("pid"), e.get("tid"))
            break

    # Identify step thread (train_runner step)
    step_thread = None
    for e in events:
        name = e.get("name", "")
        if name.startswith("train_runner.py") and "step" in name and e.get("dur", 0) > 100000:
            step_thread = (e.get("pid"), e.get("tid"))
            break

    # Classify each CPU thread
    thread_roles = {}
    if cpu_pid is not None:
        for (pid, tid), stats in thread_stats.items():
            if pid != cpu_pid:
                continue
            tname = thread_names.get((pid, tid), "")
            role = "unknown"
            if (pid, tid) == main_fb_thread or (pid, tid) == step_thread:
                role = "main_thread"
            elif "pt_autograd" in tname:
                role = "autograd_thread"
            elif stats["count"] > 100 and stats["has_modules"]:
                role = "worker_thread"
            elif stats["count"] > 10:
                role = "worker_thread"
            else:
                role = "other"
            thread_roles[(pid, tid)] = {
                "name": tname,
                "role": role,
                "count": stats["count"],
                "dur": stats["dur"],
            }

    # Find the primary GPU stream (most kernels)
    gpu_stream_stats = {}
    for (pid, tid), stats in thread_stats.items():
        if pid not in gpu_pids:
            continue
        n_kernels = stats["cats"].get("kernel", 0) + stats["cats"].get("gpu_memcpy", 0) + stats["cats"].get("gpu_memset", 0)
        if n_kernels > 0:
            gpu_stream_stats[(pid, tid)] = n_kernels

    primary_gpu_stream = max(gpu_stream_stats, key=gpu_stream_stats.get) if gpu_stream_stats else None

    return {
        "cpu_pid": cpu_pid,
        "gpu_pids": gpu_pids,
        "main_fb_thread": main_fb_thread,
        "step_thread": step_thread,
        "thread_roles": thread_roles,
        "primary_gpu_stream": primary_gpu_stream,
        "gpu_stream_stats": gpu_stream_stats,
    }


def analyze_step_decomposition(events, thread_info, profiler_steps=None):
    main_tid = thread_info["main_fb_thread"] or thread_info["step_thread"]
    if not main_tid:
        return None

    num_steps = len(profiler_steps) if profiler_steps else 1

    # Collect all step/fb/optimize events for multi-step averaging
    step_events = []
    fb_events = []
    optimize_events = []
    for e in events:
        if e.get("pid") != main_tid[0] or e.get("tid") != main_tid[1]:
            continue
        if e.get("cat") != "python_function":
            continue
        name = e.get("name", "")
        if name.startswith("train_runner.py") and "step" in name and e.get("dur", 0) > 100000:
            step_events.append(e)
        if "forward_backward" in name:
            fb_events.append(e)
        if name.startswith("train_runner.py") and "optimize" in name and e.get("dur", 0) > 1000:
            optimize_events.append(e)

    result = {}
    if step_events:
        result["step_dur"] = sum(e.get("dur", 0) for e in step_events) / max(len(step_events), 1)
    if fb_events:
        result["forward_backward_dur"] = sum(e.get("dur", 0) for e in fb_events) / max(len(fb_events), 1)
    if optimize_events:
        result["optimize_dur"] = sum(e.get("dur", 0) for e in optimize_events) / max(len(optimize_events), 1)

    # Compute other time (step - fb - optimize)
    if all(k in result for k in ["step_dur", "forward_backward_dur", "optimize_dur"]):
        result["other_dur"] = result["step_dur"] - result["forward_backward_dur"] - result["optimize_dur"]

    return result


def find_module_parent(event, events):
    p_idx = event["args"].get("_P", -1)
    while 0 <= p_idx < len(events):
        parent = events[p_idx]
        pname = parent.get("name", "")
        if parent.get("cat") == "python_function" and pname.startswith("nn.Module:"):
            return pname, p_idx
        p_idx = parent.get("args", {}).get("_P", -1)
    return None, -1


def _clip_dur_to_step(event_ts, event_dur, step_start, step_end):
    ev_start = event_ts
    ev_end = event_ts + event_dur
    clipped_start = max(ev_start, step_start)
    clipped_end = min(ev_end, step_end)
    if clipped_end <= clipped_start:
        return 0.0
    return clipped_end - clipped_start


def _find_step_for_event(event_ts, profiler_steps):
    for s in profiler_steps:
        if s["ts"] <= event_ts < s["end"]:
            return s
    return None


def analyze_modules_by_thread(events, thread_info, step_dur_us, profiler_steps=None):
    cpu_pid = thread_info["cpu_pid"]
    thread_roles = thread_info["thread_roles"]
    num_steps = len(profiler_steps) if profiler_steps else 1

    per_thread_modules = defaultdict(list)
    all_modules = []

    for e in events:
        if e.get("cat") != "python_function" or not e.get("name", "").startswith("nn.Module:"):
            continue
        pid, tid = e.get("pid", -1), e.get("tid", -1)
        if pid != cpu_pid:
            continue
        key = (pid, tid)
        per_thread_modules[key].append(e)
        all_modules.append(e)

    # Classify modules by thread role
    main_thread_modules = []
    worker_modules = defaultdict(list)

    for (pid, tid), mods in per_thread_modules.items():
        role_info = thread_roles.get((pid, tid), {})
        role = role_info.get("role", "unknown")
        if role == "main_thread":
            main_thread_modules.extend(mods)
        elif role in ("worker_thread",):
            worker_modules[(pid, tid)].extend(mods)

    # Build global module tree from all modules
    # Duration is clipped to step boundaries and averaged across steps
    module_children = defaultdict(set)
    module_parent_map = {}
    module_durations = defaultdict(float)  # sum of clipped durations across all steps
    module_call_counts = defaultdict(int)
    module_thread_map = defaultdict(set)

    for m in all_modules:
        name = m.get("name", "")
        raw_dur = m.get("dur", 0)
        m_ts = m.get("ts", 0)

        # Clip to step boundary if profiler_steps provided
        if profiler_steps:
            step = _find_step_for_event(m_ts, profiler_steps)
            if step:
                clipped = _clip_dur_to_step(m_ts, raw_dur, step["ts"], step["end"])
            else:
                clipped = 0.0
        else:
            clipped = raw_dur

        module_durations[name] += clipped
        module_call_counts[name] += 1
        module_thread_map[name].add((m.get("pid"), m.get("tid")))
        parent_name, _ = find_module_parent(m, events)
        if parent_name:
            module_children[parent_name].add(name)
            module_parent_map[name] = parent_name
        else:
            module_parent_map[name] = "ROOT"
            module_children["ROOT"].add(name)

    # Average across steps
    if num_steps > 1:
        for name in module_durations:
            module_durations[name] /= num_steps

    depth_map = {}
    def get_depth(name):
        if name in depth_map:
            return depth_map[name]
        parent = module_parent_map.get(name, "ROOT")
        if parent == "ROOT":
            depth_map[name] = 0
        else:
            depth_map[name] = get_depth(parent) + 1
        return depth_map[name]

    for name in module_parent_map:
        get_depth(name)

    max_depth = max(depth_map.values()) if depth_map else 0

    module_exclusive = {}
    for name in module_durations:
        children_dur = sum(module_durations.get(c, 0) for c in module_children.get(name, set()))
        module_exclusive[name] = max(0, module_durations[name] - children_dur)

    # Thread info for each module
    module_thread_info = {}
    for name in module_durations:
        threads = module_thread_map[name]
        roles = set()
        for t in threads:
            ri = thread_roles.get(t, {})
            roles.add(ri.get("role", "unknown"))
        module_thread_info[name] = {"threads": threads, "roles": roles}

    return {
        "module_durations": dict(module_durations),
        "module_exclusive": module_exclusive,
        "module_call_counts": dict(module_call_counts),
        "module_children": {k: list(v) for k, v in module_children.items()},
        "module_parent_map": module_parent_map,
        "depth_map": depth_map,
        "max_depth": max_depth,
        "module_thread_info": module_thread_info,
        "main_thread_count": len(main_thread_modules),
        "worker_thread_count": sum(len(v) for v in worker_modules.values()),
        "num_steps": num_steps,
    }


def analyze_gpu_timeline(events, thread_info, step_dur_us, profiler_steps=None):
    primary_stream = thread_info["primary_gpu_stream"]
    if not primary_stream:
        return None

    num_steps = len(profiler_steps) if profiler_steps else 1

    # Get GPU ProfilerStep events for boundary alignment
    gpu_profiler_steps = []
    for e in events:
        if (e.get("cat") == "gpu_user_annotation" and "ProfilerStep" in e.get("name", "")
                and e.get("pid") == primary_stream[0] and e.get("tid") == primary_stream[1]):
            gpu_profiler_steps.append({"ts": e["ts"], "dur": e.get("dur", 0), "end": e["ts"] + e.get("dur", 0)})
    gpu_profiler_steps.sort(key=lambda x: x["ts"])

    # Get GPU events on primary stream, sorted by time
    gpu_events = []
    for e in events:
        if e.get("pid") == primary_stream[0] and e.get("tid") == primary_stream[1]:
            if e.get("ph") == "X" and e.get("cat") in ("kernel", "gpu_memcpy", "gpu_memset"):
                gpu_events.append(e)
    gpu_events.sort(key=lambda x: x.get("ts", 0))

    if not gpu_events:
        return None

    # Total active time on this stream (clip to GPU step boundaries, average)
    total_kernel_dur = 0.0
    for e in gpu_events:
        dur = e.get("dur", 0)
        if gpu_profiler_steps:
            step = _find_step_for_event(e.get("ts", 0), gpu_profiler_steps)
            if step:
                dur = _clip_dur_to_step(e["ts"], dur, step["ts"], step["end"])
            else:
                dur = 0.0
        total_kernel_dur += dur
    if num_steps > 1:
        total_kernel_dur /= num_steps

    # GPU Step duration (average)
    if gpu_profiler_steps:
        gpu_step_dur = sum(s["dur"] for s in gpu_profiler_steps) / len(gpu_profiler_steps)
    else:
        gpu_step_dur = 0
        for e in events:
            if (e.get("cat") == "gpu_user_annotation" and "ProfilerStep" in e.get("name", "")
                    and e.get("pid") == primary_stream[0] and e.get("tid") == primary_stream[1]):
                gpu_step_dur = e.get("dur", 0)
                break

    # Compute inter-kernel gaps (average over steps)
    gaps = []
    for i in range(1, len(gpu_events)):
        end_prev = gpu_events[i-1]["ts"] + gpu_events[i-1].get("dur", 0)
        start_curr = gpu_events[i]["ts"]
        gap = start_curr - end_prev
        if gap > 0:
            gaps.append({
                "gap_us": gap,
                "before_name": gpu_events[i-1].get("name", ""),
                "before_dur": gpu_events[i-1].get("dur", 0),
                "after_name": gpu_events[i].get("name", ""),
                "after_dur": gpu_events[i].get("dur", 0),
            })
    total_gap = sum(g["gap_us"] for g in gaps)
    if num_steps > 1:
        total_gap /= num_steps
    gaps.sort(key=lambda x: -x["gap_us"])

    # Kernel hotspot: aggregate by name (clip and average)
    kernel_agg = defaultdict(lambda: {"dur": 0.0, "count": 0})
    for e in gpu_events:
        dur = e.get("dur", 0)
        if gpu_profiler_steps:
            step = _find_step_for_event(e.get("ts", 0), gpu_profiler_steps)
            if step:
                dur = _clip_dur_to_step(e["ts"], dur, step["ts"], step["end"])
            else:
                dur = 0.0
        kernel_agg[e.get("name", "")]["dur"] += dur
        kernel_agg[e.get("name", "")]["count"] += 1
    if num_steps > 1:
        for kname in kernel_agg:
            kernel_agg[kname]["dur"] /= num_steps
            kernel_agg[kname]["count"] = kernel_agg[kname]["count"] // num_steps or kernel_agg[kname]["count"]
    top_kernels = sorted(kernel_agg.items(), key=lambda x: -x[1]["dur"])[:30]

    # Map top kernels to host modules via External id
    ext_id_to_cpu_event = {}
    for e in events:
        ext_id = e.get("args", {}).get("External id")
        if ext_id is not None and e.get("cat") == "cpu_op" and e.get("ph") == "X":
            ext_id_to_cpu_event[ext_id] = e

    kernel_host_map = defaultdict(lambda: defaultdict(float))
    for e in gpu_events:
        ext_id = e.get("args", {}).get("External id")
        host_op = ext_id_to_cpu_event.get(ext_id)
        if not host_op:
            continue
        # Trace up to nn.Module
        module_chain = []
        p = host_op["args"].get("_P", -1)
        while 0 <= p < len(events):
            parent = events[p]
            if parent.get("cat") == "python_function" and parent.get("name", "").startswith("nn.Module:"):
                module_chain.append(parent["name"].replace("nn.Module: ", ""))
            p = parent.get("args", {}).get("_P", -1)
        if module_chain:
            leaf_module = module_chain[0]  # nearest module
            kernel_host_map[e.get("name", "")][leaf_module] += e.get("dur", 0)

    # Build top kernel -> host module mapping
    top_kernel_modules = {}
    for kname, _ in top_kernels:
        module_durs = kernel_host_map.get(kname, {})
        if module_durs:
            sorted_modules = sorted(module_durs.items(), key=lambda x: -x[1])
            top_kernel_modules[kname] = sorted_modules
        else:
            top_kernel_modules[kname] = []

    return {
        "primary_stream": primary_stream,
        "gpu_step_dur": gpu_step_dur,
        "total_kernel_dur": total_kernel_dur,
        "total_gap": total_gap,
        "num_kernels": len(gpu_events) // num_steps if num_steps > 1 else len(gpu_events),
        "gpu_utilization": total_kernel_dur / gpu_step_dur * 100 if gpu_step_dur > 0 else 0,
        "top_gaps": gaps[:15],
        "top_kernels": top_kernels,
        "top_kernel_modules": top_kernel_modules,
        "num_gpu_steps": len(gpu_profiler_steps),
    }


def analyze_device_host(events, step_dur_us, thread_info, profiler_steps=None):
    cpu_pid = thread_info["cpu_pid"]
    main_thread = thread_info["main_fb_thread"] or thread_info["step_thread"]
    num_steps = len(profiler_steps) if profiler_steps else 1

    host_total = 0.0
    device_total = 0.0
    device_breakdown = defaultdict(float)
    device_op_dur = defaultdict(float)
    host_op_dur = defaultdict(float)

    for e in events:
        cat = e.get("cat", "")
        dur = e.get("dur", 0)
        name = e.get("name", "")

        if cat in ("kernel", "gpu_memcpy", "gpu_memset"):
            device_total += dur
            device_breakdown[cat] += dur
            device_op_dur[name] += dur

        if cat == "cpu_op":
            host_total += dur
            host_op_dur[name] += dur

    # Average across steps
    if num_steps > 1:
        host_total /= num_steps
        device_total /= num_steps
        for k in device_breakdown:
            device_breakdown[k] /= num_steps
        for k in device_op_dur:
            device_op_dur[k] /= num_steps
        for k in host_op_dur:
            host_op_dur[k] /= num_steps

    return {
        "host_total_us": host_total,
        "device_total_us": device_total,
        "device_breakdown": dict(device_breakdown),
        "step_dur_us": step_dur_us,
        "top_device_ops": sorted(device_op_dur.items(), key=lambda x: -x[1])[:20],
        "top_host_ops": sorted(host_op_dur.items(), key=lambda x: -x[1])[:20],
    }


def analyze_worker_threads(events, thread_info, meta):
    cpu_pid = thread_info["cpu_pid"]
    thread_roles = thread_info["thread_roles"]
    thread_names = meta["thread_names"]

    workers = []
    for (pid, tid), role_info in thread_roles.items():
        if role_info["role"] not in ("worker_thread", "autograd_thread"):
            continue
        # Get top-level events on this thread
        thread_events = [e for e in events
                         if e.get("pid") == pid and e.get("tid") == tid
                         and e.get("ph") == "X" and e.get("dur", 0) > 1000]
        thread_events.sort(key=lambda x: -x.get("dur", 0))

        # Find nn.Module events
        modules = [e for e in events
                   if e.get("pid") == pid and e.get("tid") == tid
                   and e.get("cat") == "python_function"
                   and e.get("name", "").startswith("nn.Module:")]
        module_names = sorted(set(e.get("name", "").replace("nn.Module: ", "") for e in modules))

        # Find top python_function calls (not nn.Module)
        top_funcs = []
        for e in thread_events[:10]:
            name = e.get("name", "")
            if e.get("cat") == "python_function" and not name.startswith("nn.Module:"):
                # Skip threading boilerplate
                if any(skip in name for skip in ["threading.py", "concurrent/futures"]):
                    continue
                top_funcs.append({"name": name, "dur": e.get("dur", 0)})

        workers.append({
            "pid": pid,
            "tid": tid,
            "name": thread_names.get((pid, tid), ""),
            "role": role_info["role"],
            "event_count": role_info["count"],
            "total_dur": role_info["dur"],
            "modules": module_names,
            "top_funcs": top_funcs[:5],
        })

    workers.sort(key=lambda x: -x["total_dur"])
    return workers


def _merge_intervals(intervals):
    if not intervals:
        return []
    intervals.sort()
    merged = [intervals[0]]
    for s, e in intervals[1:]:
        if s <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], e))
        else:
            merged.append((s, e))
    return merged


def _intervals_total(merged):
    return sum(e - s for s, e in merged)


def _intervals_overlap(a_merged, b_merged):
    total = 0.0
    j = 0
    for a_s, a_e in a_merged:
        while j < len(b_merged) and b_merged[j][1] <= a_s:
            j += 1
        k = j
        while k < len(b_merged) and b_merged[k][0] < a_e:
            ov_s = max(a_s, b_merged[k][0])
            ov_e = min(a_e, b_merged[k][1])
            if ov_e > ov_s:
                total += ov_e - ov_s
            k += 1
    return total


def analyze_comm_compute_overlap(events, thread_info, profiler_steps=None):
    primary_stream = thread_info["primary_gpu_stream"]
    gpu_pids = thread_info["gpu_pids"]
    if not primary_stream or not gpu_pids:
        return None

    num_steps = len(profiler_steps) if profiler_steps else 1

    # Collect GPU ProfilerStep boundaries on the primary stream
    gpu_profiler_steps = []
    for e in events:
        if (e.get("cat") == "gpu_user_annotation" and "ProfilerStep" in e.get("name", "")
                and e.get("pid") == primary_stream[0] and e.get("tid") == primary_stream[1]):
            gpu_profiler_steps.append({"ts": e["ts"], "dur": e.get("dur", 0), "end": e["ts"] + e.get("dur", 0)})
    gpu_profiler_steps.sort(key=lambda x: x["ts"])

    # Determine time bounds per step: use CPU profiler_steps as the universal time window
    if profiler_steps:
        step_bounds = [(s["ts"], s["end"]) for s in profiler_steps]
    elif gpu_profiler_steps:
        step_bounds = [(s["ts"], s["end"]) for s in gpu_profiler_steps]
    else:
        return None

    # Classify GPU streams
    primary_key = (primary_stream[0], primary_stream[1])
    comm_stream_keys = set()
    memcpy_stream_keys = set()
    for (pid, tid), n in thread_info["gpu_stream_stats"].items():
        if (pid, tid) == primary_key:
            continue
        if pid not in gpu_pids:
            continue
        has_nccl = False
        has_memcpy = False
        for e in events:
            if e.get("pid") != pid or e.get("tid") != tid or e.get("ph") != "X":
                continue
            if e.get("cat") == "kernel" and "nccl" in e.get("name", "").lower():
                has_nccl = True
            if e.get("cat") == "gpu_memcpy":
                has_memcpy = True
            if has_nccl:
                break
        if has_nccl:
            comm_stream_keys.add((pid, tid))
        elif has_memcpy:
            memcpy_stream_keys.add((pid, tid))

    # Pre-collect relevant events once (avoid iterating all events per step)
    compute_events = []
    comm_events = []
    memcpy_events = []
    for e in events:
        if e.get("ph") != "X":
            continue
        pid, tid = e.get("pid", -1), e.get("tid", -1)
        cat = e.get("cat", "")
        if (pid, tid) == primary_key and cat in ("kernel", "gpu_memcpy", "gpu_memset"):
            compute_events.append(e)
        elif (pid, tid) in comm_stream_keys and cat == "kernel":
            comm_events.append(e)
        elif (pid, tid) in memcpy_stream_keys and cat in ("gpu_memcpy", "gpu_memset"):
            memcpy_events.append(e)

    compute_events.sort(key=lambda e: e["ts"])
    comm_events.sort(key=lambda e: e["ts"])
    memcpy_events.sort(key=lambda e: e["ts"])

    # Per-step analysis
    per_step_results = []
    for step_start, step_end in step_bounds:
        def _clip_events_to_step(ev_list):
            ivs = []
            for e in ev_list:
                ts = e["ts"]
                if ts >= step_end:
                    break
                dur = e.get("dur", 0)
                cs = max(ts, step_start)
                ce = min(ts + dur, step_end)
                if ce > cs:
                    ivs.append((cs, ce))
            return ivs

        compute_ivs = _clip_events_to_step(compute_events)
        compute_merged = _merge_intervals(compute_ivs)

        comm_ivs = []
        comm_kernel_names = defaultdict(float)
        for e in comm_events:
            ts = e["ts"]
            if ts >= step_end:
                break
            dur = e.get("dur", 0)
            cs = max(ts, step_start)
            ce = min(ts + dur, step_end)
            if ce > cs:
                comm_ivs.append((cs, ce))
                comm_kernel_names[e.get("name", "")] += ce - cs
        comm_merged = _merge_intervals(comm_ivs)

        memcpy_ivs = _clip_events_to_step(memcpy_events)
        memcpy_merged = _merge_intervals(memcpy_ivs)

        compute_total = _intervals_total(compute_merged)
        comm_total = _intervals_total(comm_merged)
        memcpy_total = _intervals_total(memcpy_merged)

        comm_overlap = _intervals_overlap(comm_merged, compute_merged)
        memcpy_overlap = _intervals_overlap(memcpy_merged, compute_merged)

        comm_exposed = comm_total - comm_overlap
        memcpy_exposed = memcpy_total - memcpy_overlap

        per_step_results.append({
            "step_dur": step_end - step_start,
            "compute_total": compute_total,
            "comm_total": comm_total,
            "comm_overlap": comm_overlap,
            "comm_exposed": comm_exposed,
            "memcpy_total": memcpy_total,
            "memcpy_overlap": memcpy_overlap,
            "memcpy_exposed": memcpy_exposed,
            "comm_kernel_names": dict(comm_kernel_names),
        })

    # Average across steps
    n = max(len(per_step_results), 1)
    avg = {}
    for key in ("step_dur", "compute_total", "comm_total", "comm_overlap", "comm_exposed",
                "memcpy_total", "memcpy_overlap", "memcpy_exposed"):
        avg[key] = sum(r[key] for r in per_step_results) / n

    # Aggregate comm kernel names across all steps
    agg_comm_names = defaultdict(float)
    for r in per_step_results:
        for kn, dur in r["comm_kernel_names"].items():
            agg_comm_names[kn] += dur
    for kn in agg_comm_names:
        agg_comm_names[kn] /= n
    top_comm_kernels = sorted(agg_comm_names.items(), key=lambda x: -x[1])[:15]

    avg["comm_overlap_pct"] = avg["comm_overlap"] / avg["comm_total"] * 100 if avg["comm_total"] > 0 else 0
    avg["memcpy_overlap_pct"] = avg["memcpy_overlap"] / avg["memcpy_total"] * 100 if avg["memcpy_total"] > 0 else 0
    avg["comm_exposed_pct_of_step"] = avg["comm_exposed"] / avg["step_dur"] * 100 if avg["step_dur"] > 0 else 0
    avg["memcpy_exposed_pct_of_step"] = avg["memcpy_exposed"] / avg["step_dur"] * 100 if avg["step_dur"] > 0 else 0
    avg["num_steps"] = n
    avg["num_comm_streams"] = len(comm_stream_keys)
    avg["num_memcpy_streams"] = len(memcpy_stream_keys)
    avg["top_comm_kernels"] = top_comm_kernels

    return avg

