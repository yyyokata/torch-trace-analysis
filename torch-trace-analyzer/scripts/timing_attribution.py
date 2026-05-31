#!/usr/bin/env python3

import logging
import os
import re
from dataclasses import dataclass
from collections import defaultdict

from common_utils import (
    _dedup_consecutive_frames,
    _frame_get,
    _is_user_source_frame,
    _normalize_parallel_wrapper_frame,
)
from source_index import _find_class_for_line
from trace_analysis import classify_kernel_phase, find_module_parent


LOG = logging.getLogger(__name__)


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


# Restored from 9a103e3 after accidental deletion in 08cb9e3 (HTML cleanup).
# --------------------------------------------------------------------------
# Step 1: build_fwdbwd_flow_index           — pair fwdbwd flow events
# Step 2: build_kernel_attribution_table    — kernel → InstanceKey weights
# Step 3: rollup_instance_timing            — bottom-up self/inclusive sum
# Step 4: build_timing_panel_data           — instance + class output
# Top-level entry: build_instance_timing_pipeline
#
# Design doc: timing_fix_workdir/output/timing_redesign.md
# ==========================================================================


_STACK_FRAME_RE = re.compile(r'File "([^"]+)", line (\d+), in (\w+)')
_INSTANCE_SUFFIX_RE = re.compile(r'_(\d+)$')
_LOC_INDEX_SELF_FALLBACK_WARNED_CLASSES = set()


@dataclass(frozen=True)
class StackFrame:
    file: str
    line: int
    func: str = ""


@dataclass(frozen=True)
class InstanceKey:
    class_name: str
    instance_suffix: str = ""


@dataclass(frozen=True)
class KernelChain:
    kernel_name: str
    duration_us: float
    frames: list
    instance_suffix: str = ""


@dataclass(frozen=True)
class MatchResult:
    instance_key: object
    reason: str
    matched_frame: object
    candidates: list


def extract_instance_suffix(event_name: str) -> str:
    """Extract an explicit trailing ModuleList instance suffix.

    Only a final ``_N`` token is accepted, e.g. ``FSDPBlock_3`` -> ``_3``.
    Missing or non-final numeric tokens return ``""``; no weak inference is
    performed from annotations or source context.
    """
    m = _INSTANCE_SUFFIX_RE.search(str(event_name or ""))
    return f"_{m.group(1)}" if m else ""


def _frame_file_line(frame):
    file_path = _frame_get(frame, 0, "file", "")
    line = _frame_get(frame, 1, "line", None)
    try:
        line = int(line)
    except (TypeError, ValueError):
        line = None
    return file_path, line


def _source_loc_file_line(loc):
    if not loc:
        return None
    if isinstance(loc, dict):
        file_path = loc.get("file") or loc.get("fname") or loc.get("path")
        line = loc.get("line") or loc.get("lineno") or loc.get("to_line") or loc.get("from_line")
    elif isinstance(loc, (tuple, list)) and len(loc) >= 2:
        file_path, line = loc[0], loc[1]
    else:
        file_path = getattr(loc, "file", None) or getattr(loc, "fname", None) or getattr(loc, "path", None)
        line = getattr(loc, "line", None) or getattr(loc, "lineno", None)
    if not file_path or line is None:
        return None
    try:
        return os.path.basename(str(file_path)), int(line)
    except (TypeError, ValueError):
        return None


def _lookup_from_dag(dag, method_name, *args):
    method = getattr(dag, method_name, None)
    if callable(method):
        return method(*args)
    if isinstance(dag, dict):
        table = dag.get(method_name)
        if callable(table):
            return table(*args)
        if isinstance(table, dict):
            return table.get(tuple(args))
    return None


def _fallback_parent_key_from_attr(class_name, attr_name):
    base = str(attr_name or "").split("[", 1)[0]
    if class_name and base and base != attr_name:
        return InstanceKey(class_name, "")
    return InstanceKey(class_name or "", "")


def _fallback_instance_key_from_attr(class_name, attr_name):
    m = re.search(r'\[(\d+)\]$', str(attr_name or ""))
    if m:
        return InstanceKey(class_name or "", f"_{m.group(1)}")
    return InstanceKey(class_name or "", "")


def _lookup_parent_instance_key(dag, class_name, attr_name):
    key = _lookup_from_dag(dag, "lookup_parent_instance_key", class_name, attr_name)
    if key is not None:
        return key
    if isinstance(dag, dict):
        parent_map = dag.get("parent_instance_keys") or dag.get("parent_keys") or {}
        key = parent_map.get((class_name, attr_name)) or parent_map.get(attr_name)
        if key is not None:
            return key
    return _fallback_parent_key_from_attr(class_name, attr_name)


def _lookup_instance_key(dag, class_name, attr_name):
    key = _lookup_from_dag(dag, "lookup_instance_key", class_name, attr_name)
    if key is not None:
        return key
    if isinstance(dag, dict):
        instance_map = dag.get("instance_keys") or dag.get("instances") or {}
        key = instance_map.get((class_name, attr_name)) or instance_map.get(attr_name)
        if key is not None:
            return key
        if instance_map:
            return None
    if re.search(r'\[(\d+)\]$', str(attr_name or "")):
        return None
    return _fallback_instance_key_from_attr(class_name, attr_name)


def _make_indexed_attr_name(attr_name, index):
    text = str(attr_name or "")
    base = text.split("[", 1)[0]
    return f"{base}[{index}]"


def _parse_instance_suffix(instance_suffix):
    m = _INSTANCE_SUFFIX_RE.fullmatch(str(instance_suffix or ""))
    return int(m.group(1)) if m else None


def _warn_loc_index_self_fallback_once(class_name):
    class_name = str(class_name or "")
    if not class_name or class_name in _LOC_INDEX_SELF_FALLBACK_WARNED_CLASSES:
        return
    _LOC_INDEX_SELF_FALLBACK_WARNED_CLASSES.add(class_name)
    LOG.warning(
        "[build_loc_index] class=%s triggered <self> fallback — no structural attrs found; verify caller passed static_module_tree not class_map",
        class_name,
    )


def _normalize_tree_class_name(class_key):
    if isinstance(class_key, tuple) and len(class_key) >= 2:
        return class_key[1]
    return class_key


def _iter_static_tree_items(dag):
    if not isinstance(dag, dict):
        return []
    items = []
    for class_key, info in dag.items():
        if not isinstance(info, dict):
            continue
        class_name = _normalize_tree_class_name(class_key)
        items.append((class_key, class_name, info))
    return items


def _find_static_tree_info(dag, class_name):
    class_name = str(class_name or "")
    if not class_name or not isinstance(dag, dict):
        return None
    for _class_key, normalized_name, info in _iter_static_tree_items(dag):
        if str(normalized_name or "") == class_name:
            return info
    return None


def _lookup_child_class_from_static_tree(dag, class_name, attr_name):
    info = _find_static_tree_info(dag, class_name)
    if not info:
        return None
    attrs = info.get("attrs", {}) or {}
    child_class = attrs.get(attr_name)
    if child_class:
        return child_class
    attr_text = str(attr_name or "")
    base_attr = attr_text.split("[", 1)[0]
    if base_attr and base_attr != attr_text:
        child_class = attrs.get(base_attr)
        if child_class:
            return child_class
    return None


def _normalize_ancestor_entry(entry):
    if isinstance(entry, (tuple, list)) and len(entry) >= 2:
        return (os.path.basename(str(entry[0] or "")), int(entry[1]))
    return None


def _build_ancestors_from_matched_frames(frames, matched_depth):
    ancestors = []
    for frame in (frames or [])[:matched_depth]:
        file_path, line = _frame_file_line(frame)
        if line is None:
            continue
        ancestors.append((os.path.basename(str(file_path or "")), int(line)))
    return tuple(ancestors)


def _resolve_attr_instance_key(dag, class_name, attr_name, frame, chain, matched_depth):
    file_path, line = _frame_file_line(frame)
    callsite_file = os.path.basename(str(file_path or ""))
    callsite_line = int(line or 0)
    ancestors = _build_ancestors_from_matched_frames(getattr(chain, "frames", []) or [], matched_depth)
    child_class = _lookup_child_class_from_static_tree(dag, class_name, attr_name)
    resolved_class = child_class or class_name
    return (resolved_class, callsite_file, callsite_line, ancestors)






def build_loc_index(dag):
    """Build source-location inverted index for Step 2 matching.

    key: ``(basename(file), line)``; value: ``[(class_name, attr_name), ...]``.
    ``first_call_loc`` / ``call_loc`` are preferred over declaration locations.
    The function accepts the analyzer's static tree dict and lightweight mock
    DAG objects used by tests.
    """
    loc_index = defaultdict(list)

    def add(loc, class_name, attr_name):
        key = _source_loc_file_line(loc)
        if not key or not class_name or not attr_name:
            return
        candidate = (class_name, attr_name)
        if candidate not in loc_index[key]:
            loc_index[key].append(candidate)

    if isinstance(dag, dict) and "classes" not in dag:
        for class_key, info in dag.items():
            if not isinstance(info, dict):
                continue
            class_name = class_key[1] if isinstance(class_key, tuple) and len(class_key) >= 2 else class_key
            attrs = info.get("attrs", {}) or {}
            first_call = info.get("first_call_loc", {}) or {}
            attr_def = info.get("attr_def_loc", {}) or {}
            dep_locs = info.get("dep_edge_locs", {}) or {}
            for attr_name in attrs:
                add(first_call.get(attr_name), class_name, attr_name)
                if attr_name not in first_call:
                    add(attr_def.get(attr_name), class_name, attr_name)
            if not attrs and not first_call and not attr_def and not dep_locs:
                _warn_loc_index_self_fallback_once(class_name)
                file_path = class_key[0] if isinstance(class_key, tuple) and len(class_key) >= 2 else info.get("file")
                for method_name, (start, end) in (info.get("methods") or {}).items():
                    if method_name == "forward":
                        for line in range(int(start), int(end) + 1):
                            add((file_path, line), class_name, "<self>")
            for (from_attr, to_attr), ev in dep_locs.items():
                if isinstance(ev, dict):
                    add((ev.get("file"), ev.get("from_line")), class_name, from_attr)
                    add((ev.get("file"), ev.get("to_line")), class_name, to_attr)
        return dict(loc_index)

    class_nodes = getattr(dag, "classes", None)
    if class_nodes is None and isinstance(dag, dict):
        class_nodes = dag.get("classes", [])
    for class_node in class_nodes or []:
        class_name = getattr(class_node, "class_name", None) or getattr(class_node, "name", None)
        attrs = getattr(class_node, "attrs", None) or []
        if isinstance(class_node, dict):
            class_name = class_node.get("class_name") or class_node.get("name")
            attrs = class_node.get("attrs", [])
        if isinstance(attrs, dict):
            attrs = attrs.values()
        for attr in attrs:
            attr_name = getattr(attr, "attr_name", None) or getattr(attr, "name", None)
            loc = getattr(attr, "call_loc", None) or getattr(attr, "forward_loc", None) or getattr(attr, "source_loc", None)
            if isinstance(attr, dict):
                attr_name = attr.get("attr_name") or attr.get("name") or attr.get("attr")
                loc = attr.get("call_loc") or attr.get("forward_loc") or attr.get("source_loc") or attr.get("loc")
            add(loc, class_name, attr_name)
    return dict(loc_index)


def match_kernel_to_instance(chain, loc_index, dag) -> MatchResult:
    matched = []
    for depth, frame in enumerate(chain.frames or []):
        file_path, line = _frame_file_line(frame)
        if line is None:
            continue
        key = (os.path.basename(str(file_path or "")), line)
        candidates = loc_index.get(key)
        if candidates:
            matched.append((depth, frame, candidates))

    if not matched:
        return MatchResult("__unattributed__", "no_stack_frame_hit_loc_index", None, [])

    # Step 1 stores frames in outer→inner order; therefore the largest enumerate
    # depth is the deepest / innermost matched frame.
    depth, frame, candidates = max(matched, key=lambda item: item[0])

    for class_name, attr_name in candidates:
        if attr_name == "<self>":
            file_path, line = _frame_file_line(frame)
            return MatchResult((class_name, "<self>", line or 0, ()), "matched_self_forward_frame", frame, candidates)
        parent_key = _resolve_attr_instance_key(dag, class_name, attr_name, frame, chain, depth)
        instance_suffix = getattr(chain, "instance_suffix", "") or ""
        if instance_suffix == "":
            return MatchResult(parent_key, "matched_without_instance_suffix", frame, candidates)

        index = _parse_instance_suffix(instance_suffix)
        if index is None:
            return MatchResult(parent_key, "instance_suffix_index_missing_fallback_to_parent_group", frame, candidates)
        child_attr_name = _make_indexed_attr_name(attr_name, index)
        child_key = _resolve_attr_instance_key(dag, class_name, child_attr_name, frame, chain, depth)
        if child_key[0] != class_name or _lookup_child_class_from_static_tree(dag, class_name, child_attr_name):
            return MatchResult(child_key, "matched_with_instance_suffix", frame, candidates)

        file_path, line = _frame_file_line(frame)
        LOG.error(
            "ModuleList index mismatch: kernel=%s suffix=%s class=%s attr=%s indexed_attr=%s frame=%s:%s; "
            "runtime trace has this instance but static DAG does not. Attribute cost to parent group.",
            getattr(chain, "kernel_name", ""),
            instance_suffix,
            class_name,
            attr_name,
            child_attr_name,
            os.path.basename(str(file_path or "")),
            line,
        )
        return MatchResult(parent_key, "instance_suffix_index_missing_fallback_to_parent_group", frame, candidates)

    return MatchResult("__unattributed__", "matched_frame_but_no_resolvable_candidate", frame, candidates)


def _parse_user_frames(traces, source_files):
    """Parse a list of trace strings into per-trace user-code frame chains.

    Returns a list of frame chains, one per input trace, preserving the
    original outer-first ordering inside each chain. Frames whose file is
    not in source_files (third-party libs, torch internals) are dropped.
    Empty chains (traces with no user frames) are filtered out.

    Contract change (P0-A): previously this function returned a single flat
    list and stopped at the first non-empty trace, which silently dropped
    every additional trace recorded for a kernel and prevented multi-leaf
    weight accumulation.  We now return all per-trace chains so downstream
    can attribute weight across distinct leaf InstanceKeys (and accumulate
    repeats when multiple traces resolve to the same key).

    Returns:
        list[list[(fname_basename, lineno, func_name)]]
    """
    if not traces or not source_files:
        return []
    chains = []
    for trace in traces:
        chain = []
        for ln in trace.split("\n"):
            m = _STACK_FRAME_RE.search(ln)
            if not m:
                continue
            fpath, lineno_s, func = m.groups()
            fname = os.path.basename(fpath)
            if fname not in source_files:
                continue
            chain.append((fname, int(lineno_s), func))
        if chain:
            chains.append(chain)
    return chains


def _frame_class(fname, lineno, class_map):
    cls, _ = _find_class_for_line(fname, lineno, class_map)
    return cls


def build_fwdbwd_flow_index(events):
    """
    构建 fwdbwd_index，供 build_kernel_stack_cost_table 使用。
    返回结构：
    {
        "corr_to_launch":       {correlation: launch_event},
        "fwdbwd_f_by_tid_ts":   {(tid, ts): event},
        "fwdbwd_s_by_id":       {id: event},
        "tid_cpuop_sorted":     {tid: [events sorted by ts]},
        "tid_nn_module_sorted": {tid: [events sorted by ts]},
        "main_thread_tid":      int,   # nn.Module python_function 最多的 tid
    }

    兼容旧的 timing pipeline：同时保留 "by_bwd_tid" / "all" 字段。
    """
    corr_to_launch = {}
    fwdbwd_f_by_tid_ts = {}
    fwdbwd_s_by_id = {}
    tid_cpuop_sorted = defaultdict(list)
    tid_nn_module_sorted = defaultdict(list)
    nn_module_count_by_tid = defaultdict(int)
    pairs = {}

    for e in events or []:
        cat = e.get("cat")
        tid = e.get("tid")
        args = e.get("args") or {}

        if cat == "cuda_runtime" and isinstance(args, dict):
            corr = args.get("correlation")
            if corr is not None:
                prev = corr_to_launch.get(corr)
                name = str(e.get("name", ""))
                if prev is None or any(x in name for x in ("cudaLaunchKernel", "cuLaunchKernel", "cudaGraphLaunch")):
                    corr_to_launch[corr] = e

        if cat == "cpu_op" and tid is not None:
            tid_cpuop_sorted[tid].append(e)

        if cat == "python_function" and tid is not None and "nn.Module" in str(e.get("name", "")):
            tid_nn_module_sorted[tid].append(e)
            nn_module_count_by_tid[tid] += 1

        if cat == "fwdbwd":
            ph = e.get("ph")
            fid = e.get("id")
            if fid is None or ph not in ("s", "f"):
                continue
            slot = pairs.setdefault(fid, {})
            slot[ph] = e
            if ph == "f":
                ts = e.get("ts")
                if tid is not None and ts is not None:
                    fwdbwd_f_by_tid_ts[(tid, ts)] = e
            elif ph == "s":
                fwdbwd_s_by_id[fid] = e

    for bucket in tid_cpuop_sorted.values():
        bucket.sort(key=lambda x: float(x.get("ts") or 0.0))
    for bucket in tid_nn_module_sorted.values():
        bucket.sort(key=lambda x: float(x.get("ts") or 0.0))

    main_thread_tid = None
    if nn_module_count_by_tid:
        main_thread_tid = max(nn_module_count_by_tid.items(), key=lambda kv: (kv[1], -int(kv[0]) if isinstance(kv[0], int) else 0))[0]
    else:
        launch_counts_by_tid = defaultdict(int)
        for launch in corr_to_launch.values():
            if launch.get("tid") is not None:
                launch_counts_by_tid[launch.get("tid")] += 1
        if launch_counts_by_tid:
            max_count = max(launch_counts_by_tid.values())
            candidate_tids = [tid for tid, cnt in launch_counts_by_tid.items() if cnt == max_count]
            if len(candidate_tids) == 1 and len(corr_to_launch) > 1:
                main_thread_tid = candidate_tids[0]

    # Backward-compatible fwdbwd flow entries for existing attribution code.
    entries = []
    for fid, pp in pairs.items():
        s_ev = pp.get("s")
        f_ev = pp.get("f")
        if not s_ev or not f_ev:
            continue
        try:
            entries.append({
                "flow_id": fid,
                "fwd_ts": float(s_ev.get("ts", 0.0)),
                "fwd_tid": s_ev.get("tid"),
                "bwd_ts": float(f_ev.get("ts", 0.0)),
                "bwd_tid": f_ev.get("tid"),
            })
        except Exception:
            continue
    entries.sort(key=lambda x: x["bwd_ts"])
    by_tid = defaultdict(list)
    for ent in entries:
        by_tid[ent["bwd_tid"]].append(ent)

    return {
        "corr_to_launch": dict(corr_to_launch),
        "fwdbwd_f_by_tid_ts": dict(fwdbwd_f_by_tid_ts),
        "fwdbwd_s_by_id": dict(fwdbwd_s_by_id),
        "tid_cpuop_sorted": dict(tid_cpuop_sorted),
        "tid_nn_module_sorted": dict(tid_nn_module_sorted),
        "main_thread_tid": main_thread_tid,
        "by_bwd_tid": dict(by_tid),
        "all": entries,
    }


def _find_enclosing_op(events_by_tid_sorted, tid, ts):
    """Find the smallest cpu_op event on `tid` whose [ts, ts+dur] contains
    the timestamp. Returns event dict or None.

    `events_by_tid_sorted`: {tid: [event]} sorted by ts ascending.
    """
    cands = events_by_tid_sorted.get(tid, [])
    if not cands:
        return None
    # Linear scan with early exit (events count per tid is bounded; OK).
    best = None
    best_dur = float("inf")
    for ev in cands:
        ev_ts = ev.get("ts", 0.0)
        if ev_ts > ts:
            break
        dur = ev.get("dur") or 0.0
        if ev_ts <= ts <= ev_ts + dur and dur < best_dur:
            best = ev
            best_dur = dur
    return best


def _build_cpu_op_index_by_tid(events):
    """Group cpu_op events by tid and sort by ts. Used for fwdbwd flow
    interval resolution.
    """
    by_tid = defaultdict(list)
    for e in events:
        if e.get("cat") != "cpu_op":
            continue
        if e.get("dur") is None:
            continue
        by_tid[e.get("tid")].append(e)
    for tid in by_tid:
        by_tid[tid].sort(key=lambda x: x.get("ts", 0.0))
    return dict(by_tid)


def _resolve_fwdbwd_scope(fwdbwd_index, cpu_op_by_tid, bwd_kernel_ts, bwd_kernel_tid):
    """For a backward kernel at (ts, tid), find the corresponding forward
    scope time range.

    Returns (fwd_start, fwd_end, fwd_tid) or None.
    """
    cands = fwdbwd_index["by_bwd_tid"].get(bwd_kernel_tid, [])
    if not cands:
        return None
    # Find the fwdbwd entry whose enclosing backward op contains bwd_kernel_ts.
    # Strategy: pick the entry with the largest bwd_ts <= kernel_ts whose
    # enclosing bwd op covers the kernel.
    chosen = None
    chosen_op = None
    for ent in cands:
        if ent["bwd_ts"] > bwd_kernel_ts:
            continue
        bwd_op = _find_enclosing_op(cpu_op_by_tid, ent["bwd_tid"], ent["bwd_ts"])
        if bwd_op is None:
            continue
        bwd_start = bwd_op.get("ts", 0.0)
        bwd_end = bwd_start + (bwd_op.get("dur") or 0.0)
        if bwd_start <= bwd_kernel_ts <= bwd_end:
            chosen = ent
            chosen_op = bwd_op
            # Don't break — later entries (same op) override since they
            # represent finer-grained scope.
    if chosen is None:
        return None
    fwd_op = _find_enclosing_op(cpu_op_by_tid, chosen["fwd_tid"], chosen["fwd_ts"])
    if fwd_op is None:
        return None
    fwd_start = fwd_op.get("ts", 0.0)
    fwd_end = fwd_start + (fwd_op.get("dur") or 0.0)
    return (fwd_start, fwd_end, chosen["fwd_tid"])


# --------------------------------------------------------------------------
# Step 2: kernel attribution to InstanceKey
# --------------------------------------------------------------------------

def _extract_instance_keys_from_stack(frames, class_map):
    """From a list of (fname, lineno, func_name) frames (outer→inner),
    walk class boundaries and produce a list of InstanceKey ordered from
    outermost to deepest.

    InstanceKey = (class_name, callsite_file, callsite_line, ancestors_tuple)
      - callsite_file/line: parent frame's (file, line) — i.e. where the
        parent class's code calls into this child class.
      - ancestors_tuple: ((file, line), ...) of all preceding callsites
        outermost→inner, NOT including self.

    Single-frame fallback: when only one user frame is present and it maps
    to a class, emit a weak key for that class (callsite_file=<self>,
    callsite_line=that frame's line, ancestors=()).  This lets us attribute
    e.g. autograd-leaf kernels whose stack has only the leaf class's forward
    method.

    Returns [] only if no frame maps to any user class.
    """
    if not frames or not class_map:
        return []
    # Compute class for each frame.
    frame_classes = []
    for f in frames:
        cls = _frame_class(f[0], f[1], class_map)
        frame_classes.append(cls)
    keys = []
    # Find the first frame that maps to a class.
    first_idx = -1
    for i, cls in enumerate(frame_classes):
        if cls:
            first_idx = i
            break

    if first_idx != -1:
        # Outermost class attribution. If it was called from a user frame
        # that doesn't map to a class (e.g. main.py), use that as callsite.
        # If it's the very first frame in the stack, use a <root> anchor.
        cls = frame_classes[first_idx]
        if first_idx > 0:
            cf, cl = frames[first_idx-1][0], frames[first_idx-1][1]
        else:
            # Use the class's start line as a stable anchor for the root.
            cf, cl = "<root>", class_map.get((frames[0][0], cls), {}).get("start", 0)
        
        root_key = (cls, cf, cl, ())
        keys.append(root_key)
        
        ancestors = [(cf, cl)]
        last_class = cls
        last_frame = frames[first_idx]
        for i in range(first_idx + 1, len(frames)):
            cur_class = frame_classes[i]
            if cur_class and last_class and cur_class != last_class:
                key = (cur_class, last_frame[0], last_frame[1], tuple(ancestors))
                keys.append(key)
                ancestors.append((last_frame[0], last_frame[1]))
            if cur_class:
                last_class = cur_class
            last_frame = frames[i]

    if not keys:
        # No class boundary detected. Emit a weak key for the deepest
        # frame whose class is known so the kernel still gets attributed.
        for i in range(len(frames) - 1, -1, -1):
            cls = frame_classes[i]
            if cls:
                key = (cls, "<self>", frames[i][1], ())
                keys.append(key)
                break
    return keys


def _is_backward_trace(traces):
    if not traces:
        return False
    for t in traces:
        if "Gradient" in t or "Backward" in t or "backward" in t:
            return True
    return False


def filter_kernel_stack_chains(rows, source_files, step_infos):
    stats = {
        "total": 0,
        "phase_fwd": 0,
        "phase_bwd": 0,
        "phase_other": 0,
        "frames_in": 0,
        "frames_kept_user": 0,
        "frames_dropped_framework": 0,
        "unmatched_after_filter": 0,
        "non_bwd_overlaps_backward": 0,
    }

    class FilteredKernelStackRows(list):
        def __init__(self):
            super().__init__()
            self.by_event_idx = {}
            self.step1b_stats = stats

        def add(self, event_idx, row):
            if event_idx is not None:
                self.by_event_idx[event_idx] = row
            self.append(row)

        def items(self):
            return self.by_event_idx.items() if self.by_event_idx else enumerate(self)

        def keys(self):
            return self.by_event_idx.keys() if self.by_event_idx else range(len(self))

        def values(self):
            return self.by_event_idx.values() if self.by_event_idx else list(self)

        def get(self, key, default=None):
            if self.by_event_idx:
                return self.by_event_idx.get(key, default)
            try:
                return self[key]
            except (TypeError, IndexError):
                return default

        def __getitem__(self, key):
            if isinstance(key, int) and key in self.by_event_idx:
                return self.by_event_idx[key]
            return super().__getitem__(key)

    def _row_get(row, key, default=None):
        if isinstance(row, dict):
            return row.get(key, default)
        return getattr(row, key, default)

    def _row_set(row, key, value):
        if isinstance(row, dict):
            row[key] = value
        elif hasattr(row, key):
            try:
                setattr(row, key, value)
            except Exception:
                pass

    def _copy_row(row):
        if isinstance(row, dict):
            return dict(row)
        return row

    def _refine_phase(row):
        phase = _row_get(row, "phase", "non-bwd")
        if phase == "bwd":
            return "bwd"
        ts = float(_row_get(row, "ts", 0.0) or 0.0)
        dur = float(_row_get(row, "dur_us", 0.0) or 0.0)
        window_phase = classify_kernel_phase(ts, ts + dur, step_infos or [])
        if window_phase == "forward":
            return "fwd"
        if window_phase == "backward":
            stats["non_bwd_overlaps_backward"] += 1
        return "other"

    out = FilteredKernelStackRows()
    for input_idx, row in enumerate(rows or []):
        stats["total"] += 1
        filtered = []
        for frame in _row_get(row, "chains", []) or []:
            stats["frames_in"] += 1
            normalized_frame = _normalize_parallel_wrapper_frame(frame)
            if normalized_frame is None:
                stats["frames_dropped_framework"] += 1
                continue
            if _is_user_source_frame(normalized_frame, source_files):
                filtered.append(normalized_frame)
                stats["frames_kept_user"] += 1
            else:
                stats["frames_dropped_framework"] += 1
        filtered = _dedup_consecutive_frames(filtered)
        refined_phase = _refine_phase(row)
        new_row = _copy_row(row)
        _row_set(new_row, "phase", refined_phase)
        _row_set(new_row, "chains", filtered)
        _row_set(new_row, "unmatched", len(filtered) == 0)
        _row_set(new_row, "is_bwd", refined_phase == "bwd")
        _row_set(new_row, "legacy_chains", [filtered] if filtered else [])
        stats[f"phase_{refined_phase}"] += 1
        if len(filtered) == 0:
            stats["unmatched_after_filter"] += 1
        event_idx = _row_get(new_row, "event_idx", input_idx)
        out.add(event_idx, new_row)

    return out


def build_kernel_stack_cost_table(events, source_files, fwdbwd_index):
    """
    返回 list[dict]，每条 kernel 对应一条，dict 字段：
      name, dur_us, phase, has_stack_traces, chains, unmatched
    chains 元素为 (file, line, func) tuple，innermost first。
    phase: "bwd"(launch.tid != main_thread_tid) | "non-bwd"

    兼容旧的 Step 2：额外返回 event/ts/tid/traces/is_bwd/mod_name 字段，且
    KernelStackRows 支持旧代码按原 events 下标取值和按 kernel 顺序迭代。
    """
    class KernelStackRows(list):
        def __init__(self):
            super().__init__()
            self.by_event_idx = {}

        def add(self, event_idx, row):
            self.by_event_idx[event_idx] = row
            self.append(row)

        def items(self):
            return self.by_event_idx.items()

        def keys(self):
            return self.by_event_idx.keys()

        def values(self):
            return self.by_event_idx.values()

        def get(self, key, default=None):
            return self.by_event_idx.get(key, default)

        def __getitem__(self, key):
            if isinstance(key, int) and key in self.by_event_idx:
                return self.by_event_idx[key]
            return super().__getitem__(key)

    if fwdbwd_index is None:
        fwdbwd_index = build_fwdbwd_flow_index(events)

    main_thread_tid = fwdbwd_index.get("main_thread_tid")
    corr_to_launch = fwdbwd_index.get("corr_to_launch", {})
    fwdbwd_f_by_tid_ts = fwdbwd_index.get("fwdbwd_f_by_tid_ts", {})
    fwdbwd_s_by_id = fwdbwd_index.get("fwdbwd_s_by_id", {})
    tid_cpuop_sorted = fwdbwd_index.get("tid_cpuop_sorted", {})
    tid_nn_module_sorted = fwdbwd_index.get("tid_nn_module_sorted", {})
    ext_to_module = _build_external_id_to_module_map(events or [])

    def _ts_list(sorted_events):
        return [float(e.get("ts") or 0.0) for e in sorted_events]

    ts_cache = {}

    def _find_covering(sorted_events, target_ts):
        from bisect import bisect_right
        if not sorted_events or target_ts is None:
            return []
        key = id(sorted_events)
        ts_list = ts_cache.get(key)
        if ts_list is None:
            ts_list = _ts_list(sorted_events)
            ts_cache[key] = ts_list
        target_ts = float(target_ts)
        idx = bisect_right(ts_list, target_ts) - 1
        result = []
        while idx >= 0:
            e = sorted_events[idx]
            e_ts = float(e.get("ts") or 0.0)
            dur = float(e.get("dur") or 0.0)
            if e_ts <= target_ts <= e_ts + dur:
                result.append((dur, e))
            idx -= 1
        result.sort(key=lambda x: x[0])
        return result

    def _covering_cpu_ops(tid, ts):
        return [e for _dur, e in _find_covering(tid_cpuop_sorted.get(tid, []), ts)]

    def _covering_nn_modules(tid, ts):
        return [e for _dur, e in _find_covering(tid_nn_module_sorted.get(tid, []), ts)]

    def _infer_bwd_from_cpu_chain(launch_ev):
        if not launch_ev:
            return False
        launch_tid = launch_ev.get("tid")
        launch_ts = launch_ev.get("ts")
        cpu_chain = _covering_cpu_ops(launch_tid, launch_ts)
        for op in cpu_chain:
            name = str(op.get("name", ""))
            lname = name.lower()
            if name.startswith("autograd::engine::evaluate_function:"):
                return True
            if "backward" in lname or "gradient" in lname:
                return True
        return False

    def _parse_stack_frames(traces):
        frames = []
        for trace in traces or []:
            for line in str(trace).splitlines():
                m = re.search(r'File "([^"]+)", line (\d+), in (.+)', line)
                if not m:
                    continue
                frames.append((m.group(1), int(m.group(2)), m.group(3)))
        return frames

    def _module_event_to_frame(ev):
        name_str = str(ev.get("name", ""))
        callsite_match = re.search(r'callsite:\s*(\d+)', name_str)
        callsite_line = int(callsite_match.group(1)) if callsite_match else 0
        call_from = (ev.get("args") or {}).get("CallFrom", "")
        if ":" in str(call_from):
            file_path, line_s = str(call_from).rsplit(":", 1)
            try:
                line = int(line_s)
            except ValueError:
                line = callsite_line
        else:
            file_path = str(call_from)
            line = callsite_line
        return (file_path, line, name_str)

    def _upper_frames_for_non_bwd(launch):
        if launch is None:
            return []
        launch_ts = launch.get("ts")
        if launch_ts is None:
            return []
        tid = main_thread_tid if main_thread_tid is not None else launch.get("tid")
        return [_module_event_to_frame(e) for e in _covering_nn_modules(tid, launch_ts)]

    def _upper_frames_for_bwd(launch):
        if launch is None:
            return []
        bwd_tid = launch.get("tid")
        launch_ts = launch.get("ts")
        cpu_chain = _covering_cpu_ops(bwd_tid, launch_ts)
        f_ev = None
        for cpu_op in cpu_chain:
            op_ts = cpu_op.get("ts")
            if op_ts is None:
                continue
            f_ev = fwdbwd_f_by_tid_ts.get((bwd_tid, op_ts))
            if f_ev is not None:
                break
        if f_ev is None:
            return []
        s_ev = fwdbwd_s_by_id.get(f_ev.get("id"))
        if s_ev is None:
            return []
        fwd_tid = s_ev.get("tid")
        fwd_ts = s_ev.get("ts")
        return [_module_event_to_frame(e) for e in _covering_nn_modules(fwd_tid, fwd_ts)]

    def _legacy_chains_from_frames(frames):
        return [list(frames)] if frames else []

    def _row(idx, ev, ts, tid, phase, has_stack_traces, chains, unmatched, traces=None, mod_name=None):
        normalized = _normalize_runtime_module_name(mod_name) if mod_name else None
        class_name = normalized.get("class_name") if normalized else None
        is_bwd = phase == "bwd"
        return {
            "name": ev.get("name", ""),
            "dur_us": float(ev.get("dur") or 0.0),
            "phase": phase,
            "has_stack_traces": bool(has_stack_traces),
            "chains": chains,
            "instance_suffix": extract_instance_suffix(ev.get("name", "")),
            "unmatched": unmatched,
            # Legacy compatibility for build_kernel_attribution_table.
            "event": ev,
            "ts": ts,
            "tid": tid,
            "traces": traces or [],
            "is_bwd": is_bwd,
            "mod_name": class_name,
            "legacy_chains": _legacy_chains_from_frames(chains),
            "event_idx": idx,
        }

    rows = KernelStackRows()
    for idx, kernel in enumerate(events or []):
        if kernel.get("cat") != "kernel":
            continue
        dur = float(kernel.get("dur") or 0.0)
        if dur <= 0:
            continue
        args = kernel.get("args") or {}
        ts = float(kernel.get("ts") or 0.0)
        tid = kernel.get("tid")
        corr = args.get("correlation") if isinstance(args, dict) else None
        launch = corr_to_launch.get(corr)
        st = args.get("stack", {}) if isinstance(args, dict) else {}
        traces = st.get("stack_traces", []) if isinstance(st, dict) else []

        if main_thread_tid is None:
            phase = "bwd" if _infer_bwd_from_cpu_chain(launch) else "non-bwd"
        elif not tid_nn_module_sorted and len(corr_to_launch) == 1 and launch and launch.get("tid") != main_thread_tid:
            phase = "bwd"
        elif tid_cpuop_sorted and launch and launch.get("tid") in tid_cpuop_sorted and not tid_nn_module_sorted:
            phase = "bwd"
        else:
            phase = "bwd" if (launch and launch.get("tid") != main_thread_tid) else "non-bwd"

        base_chains = _parse_stack_frames(traces)
        has_stack_traces = bool(base_chains)
        upper_frames = _upper_frames_for_bwd(launch) if phase == "bwd" else _upper_frames_for_non_bwd(launch)
        chains = base_chains + upper_frames
        unmatched = not bool(chains)
        ext_id = args.get("External id") if isinstance(args, dict) else None
        mod_name = ext_to_module.get(ext_id)
        if not mod_name:
            mod_name, _pidx = find_module_parent(kernel, events or [])
        if phase == "non-bwd" and not chains and mod_name is not None:
            unmatched = False
        rows.add(idx, _row(idx, kernel, ts, tid, phase, has_stack_traces, chains, unmatched, traces=traces, mod_name=mod_name))

    return rows


def build_kernel_attribution_table(events, source_files, class_map, step_infos, fwdbwd_index, static_module_tree=None):
    """Step 2: for every kernel, decide which InstanceKey(s) it belongs to
    and with what weight (weights sum to 1 per kernel).

    Returns (attribution, debug_stats):
      attribution: {kernel_idx: {InstanceKey: weight}}
      debug_stats: counts dict
    """
    if not source_files or not class_map:
        return {}, {"total_kernels": 0, "fwd_attributed": 0, "bwd_attributed": 0,
                    "wrapped_fallback": 0, "fwd_unattributed": 0, "bwd_unattributed": 0,
                    "bwd_via_flow_narrowed": 0}

    ext_to_module = _build_external_id_to_module_map(events)
    loc_index = build_loc_index(static_module_tree or class_map)
    kernel_rows = build_kernel_stack_cost_table(events, source_files, fwdbwd_index)
    kernel_rows = filter_kernel_stack_chains(kernel_rows, source_files, step_infos)

    attribution = {}
    stats = {
        "total_kernels": 0, "fwd_attributed": 0, "bwd_attributed": 0,
        "wrapped_fallback": 0, "fwd_unattributed": 0, "bwd_unattributed": 0,
        "bwd_via_flow_narrowed": 0,
        "total_kernel_dur_us": 0.0,
    }

    def _wrapped_fallback(idx, mod_name):
        info = _normalize_runtime_module_name(mod_name)
        cls = info.get("class_name")
        if not cls:
            return False
        # Synthesize a weak InstanceKey: ancestors empty, callsite_line None.
        rt_idx = info.get("runtime_index")
        callsite = info.get("callsite")
        # Use callsite line if available; otherwise use runtime_index packed as -idx
        cs_line = callsite if callsite is not None else (-(rt_idx or 0))
        key = (cls, "<wrapped>", int(cs_line) if cs_line is not None else 0, ())
        attribution[idx] = {key: 1.0}
        return True

    for idx, meta in kernel_rows.items():
        stats["total_kernels"] += 1
        stats["total_kernel_dur_us"] += meta.get("dur_us", 0.0)
        event = meta.get("event") or events[idx]
        traces = meta.get("traces") or []
        is_bwd = meta.get("is_bwd", False)
        if not traces and meta.get("chains"):
            filtered_chain = []
            for frame in meta.get("chains") or []:
                fname = os.path.basename(frame[0]) if frame and frame[0] else ""
                if fname in source_files:
                    filtered_chain.append((fname, frame[1], frame[2]))
            chains_from_row = [filtered_chain] if filtered_chain else []
        else:
            chains_from_row = []
        if not traces and not chains_from_row:
            mod_name = meta.get("mod_name") or ext_to_module.get((event.get("args") or {}).get("External id"))
            if not mod_name:
                # Second-level fallback: walk parent chain via _P pointers.
                mod_name, _pidx = find_module_parent(event, events)
            if mod_name and _wrapped_fallback(idx, mod_name):
                stats["wrapped_fallback"] += 1
                continue
            if is_bwd:
                stats["bwd_unattributed"] += 1
            else:
                stats["fwd_unattributed"] += 1
            continue

        chains = chains_from_row or _parse_user_frames(traces, source_files)
        if not chains:
            # Try wrapped fallback
            mod_name = meta.get("mod_name") or ext_to_module.get((event.get("args") or {}).get("External id"))
            if not mod_name:
                mod_name, _pidx = find_module_parent(event, events)
            if mod_name and _wrapped_fallback(idx, mod_name):
                stats["wrapped_fallback"] += 1
                continue
            if is_bwd:
                stats["bwd_unattributed"] += 1
            else:
                stats["fwd_unattributed"] += 1
            continue

        # Accumulate leaf-InstanceKey hits across ALL traces of this kernel.
        # Contract (P0-A, no-dedup):
        #   * Every non-empty trace contributes +1 weight to its leaf key.
        #   * If two traces resolve to the same leaf key, that key gets +2.
        #   * Final weight = hit_count / total_hits (sum normalized to 1.0).
        # This makes 1 kernel × N traces equivalent to N kernels each
        # attributed once, as required by the timing contract.
        hit_counts = {}
        total_hits = 0
        for chain in chains:
            kernel_chain = KernelChain(
                kernel_name=meta.get("name") or event.get("name", ""),
                duration_us=meta.get("dur_us", 0.0),
                frames=chain,
                instance_suffix=meta.get("instance_suffix", ""),
            )
            result = match_kernel_to_instance(kernel_chain, loc_index, static_module_tree or class_map)
            if result.instance_key == "__unattributed__":
                continue
            hit_counts[result.instance_key] = hit_counts.get(result.instance_key, 0) + 1
            total_hits += 1

        if total_hits == 0:
            mod_name = meta.get("mod_name") or ext_to_module.get((event.get("args") or {}).get("External id"))
            if not mod_name:
                mod_name, _pidx = find_module_parent(event, events)
            if mod_name and _wrapped_fallback(idx, mod_name):
                stats["wrapped_fallback"] += 1
                continue
            if is_bwd:
                stats["bwd_unattributed"] += 1
            else:
                stats["fwd_unattributed"] += 1
            continue

        if is_bwd and fwdbwd_index and fwdbwd_index.get("all"):
            # Try to narrow via fwdbwd flow: find forward scope, then look
            # for forward kernels whose stack reproduces the leaf keys —
            # if there's a unique match, we keep the current weights.
            # Flow-based narrowing is already reflected by Step 1a Path B.
            # Keep this counter informational for backward-compatible stats.
            stats["bwd_via_flow_narrowed"] += 1
            # Backward stack_traces already encode forward callsite chains
            # (recorded by autograd), so the leaf keys are normally already
            # correct. Flow-based narrowing is informational; we keep the
            # accumulated weights.

        # Normalize hit counts to weights summing to 1.0.
        attribution[idx] = {k: cnt / total_hits for k, cnt in hit_counts.items()}
        if is_bwd:
            stats["bwd_attributed"] += 1
        else:
            stats["fwd_attributed"] += 1

    return attribution, stats


# --------------------------------------------------------------------------
# Step 3: bottom-up self/inclusive rollup
# --------------------------------------------------------------------------

def rollup_instance_timing(kernel_attribution, events, step_infos, class_map, roots=None):
    """Step 3: aggregate kernel durations into per-InstanceKey self_us, then
    compute inclusive_us by walking the InstanceKey ancestors chain.

    Returns: {InstanceKey: instance_record}
    instance_record = {
        "class_name", "callsite_file", "callsite_line", "ancestors",
        "self_us": {"forward","backward","optimize","other"},
        "inclusive_us": {"forward","backward","optimize","other"},
        "child_keys": set,
    }
    """
    num_steps = max(1, len(step_infos))
    timings = {}

    def _ensure(key):
        if key not in timings:
            class_name, csf, csl, ancestors = key
            timings[key] = {
                "class_name": class_name,
                "callsite_file": csf,
                "callsite_line": csl,
                "ancestors": ancestors,
                "self_us": {"forward": 0.0, "backward": 0.0, "optimize": 0.0, "other": 0.0},
                "inclusive_us": {"forward": 0.0, "backward": 0.0, "optimize": 0.0, "other": 0.0},
                "child_keys": set(),
            }
        return timings[key]

    # Phase A: self_us
    for idx, weights in kernel_attribution.items():
        e = events[idx]
        dur = float(e.get("dur") or 0.0)
        ts = float(e.get("ts") or 0.0)
        phase = classify_kernel_phase(ts, ts + dur, step_infos)
        if phase not in ("forward", "backward", "optimize", "other"):
            phase = "other"
        for key, w in weights.items():
            rec = _ensure(key)
            rec["self_us"][phase] += dur * w

    # Divide by num_steps for per-step average.
    for rec in timings.values():
        for ph in rec["self_us"]:
            rec["self_us"][ph] /= num_steps

    # Phase B: parent-child relationships from ancestors
    # parent of (cls, csf, csl, ancestors) = (parent_cls, parent_csf, parent_csl, ancestors[:-1])
    # where parent_cls is the class containing (csf, csl).
    for key in list(timings.keys()):
        class_name, csf, csl, ancestors = key
        if not ancestors:
            continue  # top-level / weak key
        # Find parent's class via class_map
        parent_cls = _frame_class(csf, csl, class_map) if csf != "<wrapped>" else None
        if not parent_cls:
            continue
        parent_csf, parent_csl = ancestors[-1]
        parent_ancestors = ancestors[:-1]
        parent_key = (parent_cls, parent_csf, parent_csl, parent_ancestors)
        if parent_key not in timings:
            # Synthesize parent record so its inclusive_us aggregates
            # children even when no kernel directly attributed to parent.
            _ensure(parent_key)
        timings[parent_key]["child_keys"].add(key)

    # Phase C: bottom-up sum (topological order — leaves first).
    # Compute depth = len(ancestors); larger depth processed first.
    def _depth(k):
        return len(k[3])
    sorted_keys = sorted(timings.keys(), key=_depth, reverse=True)
    # Initialize inclusive = self
    for k in timings:
        for ph, v in timings[k]["self_us"].items():
            timings[k]["inclusive_us"][ph] = v
    # Add children inclusive
    for k in sorted_keys:
        rec = timings[k]
        for ck in rec["child_keys"]:
            child = timings.get(ck)
            if not child:
                continue
            for ph, v in child["inclusive_us"].items():
                rec["inclusive_us"][ph] += v

    # Phase D: orphan rollup to model roots
    if roots:
        # Find a primary root instance to receive orphans
        primary_root_key = None
        for k in timings:
            if k[0] in roots and k[1] == "<wrapped>":
                primary_root_key = k
                break
        if not primary_root_key:
            for k in timings:
                if k[0] in roots:
                    primary_root_key = k
                    break
        
        if primary_root_key:
            # Find all orphans (nodes with no parent)
            all_children = set()
            for rec in timings.values():
                all_children.update(rec["child_keys"])
            
            for k, rec in timings.items():
                if k != primary_root_key and k not in all_children:
                    # This is an orphan! Add its inclusive time to primary root.
                    # This ensures model-level total inclusive time is preserved.
                    for ph, v in rec["inclusive_us"].items():
                        timings[primary_root_key]["inclusive_us"][ph] += v

    return timings


# --------------------------------------------------------------------------
# Step 4: instance + class panel data
# --------------------------------------------------------------------------

def _compute_step_phase_totals(events, step_infos):
    totals = {"forward": 0.0, "backward": 0.0, "optimize": 0.0, "other": 0.0}
    num_steps = max(1, len(step_infos or []))
    for event in events or []:
        if event.get("cat") != "kernel":
            continue
        ts = float(event.get("ts") or 0.0)
        dur = float(event.get("dur") or 0.0)
        phase = classify_kernel_phase(ts, ts + dur, step_infos or [])
        if phase not in totals:
            phase = "other"
        totals[phase] += dur
    for phase in totals:
        totals[phase] /= num_steps
    return totals


def build_timing_panel_data(instance_timing, class_map, step_dur_us, phase_totals=None):
    """Step 4: convert instance_timing → timing_data fields:
      - runtime_instance_timings_by_class (instance level)
      - class_durations / class_durations_fwd / class_durations_bwd
        (class level = sum of all instances' inclusive)
    """
    by_class = defaultdict(list)
    root_step_phase_str = None
    root_step_phase_total = -1.0
    root_inclusive_forward_us = 0.0
    root_inclusive_backward_us = 0.0
    root_inclusive_optimize_us = 0.0
    root_inclusive_other_us = 0.0
    if phase_totals:
        root_inclusive_forward_us = float(phase_totals.get("forward", 0.0) or 0.0)
        root_inclusive_backward_us = float(phase_totals.get("backward", 0.0) or 0.0)
        root_inclusive_optimize_us = float(phase_totals.get("optimize", 0.0) or 0.0)
        root_inclusive_other_us = float(phase_totals.get("other", 0.0) or 0.0)
        host_us = max(
            0.0,
            float(step_dur_us or 0.0)
            - root_inclusive_forward_us
            - root_inclusive_backward_us
            - root_inclusive_optimize_us
            - root_inclusive_other_us,
        )
        root_step_phase_str = (
            f"fwd={root_inclusive_forward_us / 1000.0:.3f} + "
            f"bwd={root_inclusive_backward_us / 1000.0:.3f} + "
            f"opt={root_inclusive_optimize_us / 1000.0:.3f} + "
            f"host={host_us / 1000.0:.3f} = {float(step_dur_us or 0.0) / 1000.0:.3f} ms"
        )
    for key, rec in instance_timing.items():
        cls = rec["class_name"]
        csf = rec["callsite_file"]
        csl = rec["callsite_line"]
        anc = rec["ancestors"]
        sus_fwd = rec["self_us"]["forward"]
        sus_bwd = rec["self_us"]["backward"]
        sus_opt = rec["self_us"]["optimize"]
        sus_oth = rec["self_us"]["other"]
        inc_fwd = rec["inclusive_us"]["forward"]
        inc_bwd = rec["inclusive_us"]["backward"]
        inc_opt = rec["inclusive_us"]["optimize"]
        inc_oth = rec["inclusive_us"]["other"]
        self_total = sus_fwd + sus_bwd + sus_opt + sus_oth
        inc_total = inc_fwd + inc_bwd + inc_opt + inc_oth
        runtime_name = (f"{cls}@{os.path.basename(csf)}:{csl}"
                        if csf and csf != "<wrapped>" and csl and csl > 0
                        else cls)
        item = {
            "runtime_name": runtime_name,
            "runtime_index": rec.get("runtime_index"),
            "callsite": csl if (csl and csl > 0) else None,
            "callsite_file": csf,
            "callsite_line": csl if (csl and csl > 0) else None,
            "ancestors": [list(a) for a in anc],
            # Backwards-compat: forward_us/backward_us/total_us are EXCLUSIVE (self)
            # values, matching the prior contract under the bottom-up rollup.
            "forward_us": sus_fwd,
            "backward_us": sus_bwd,
            "optimize_us": sus_opt,
            "other_us": sus_oth,
            "total_us": self_total,
            "self_us": self_total,
            # New: inclusive values for downstream UI.
            "inclusive_us": inc_total,
            "inclusive_forward_us": inc_fwd,
            "inclusive_backward_us": inc_bwd,
            "inclusive_optimize_us": inc_opt,
            "inclusive_other_us": inc_oth,
        }
        if (not phase_totals) and (not anc) and inc_total > root_step_phase_total:
            root_step_phase_total = inc_total
            root_inclusive_forward_us = inc_fwd
            root_inclusive_backward_us = inc_bwd
            root_inclusive_optimize_us = inc_opt
            root_inclusive_other_us = inc_oth
            host_us = max(0.0, float(step_dur_us or 0.0) - inc_fwd - inc_bwd - inc_opt)
            root_step_phase_str = (
                f"fwd={inc_fwd / 1000.0:.3f} + "
                f"bwd={inc_bwd / 1000.0:.3f} + "
                f"opt={inc_opt / 1000.0:.3f} + "
                f"host={host_us / 1000.0:.3f} = {float(step_dur_us or 0.0) / 1000.0:.3f} ms"
            )
        by_class[cls].append(item)

    # Sort each class's instances deterministically.
    for cn in by_class:
        by_class[cn].sort(key=lambda x: (
            x.get("callsite_file") or "",
            x.get("callsite_line") or 0,
            x.get("runtime_name") or "",
        ))

    # Build class-level totals from instance inclusive.
    class_durations = {}
    class_durations_fwd = {}
    class_durations_bwd = {}
    for cls, items in by_class.items():
        cls_fwd = sum(it.get("inclusive_forward_us", 0.0) for it in items)
        cls_bwd = sum(it.get("inclusive_backward_us", 0.0) for it in items)
        cls_total = sum(it.get("inclusive_us", 0.0) for it in items)
        class_durations_fwd[cls] = cls_fwd
        class_durations_bwd[cls] = cls_bwd
        class_durations[cls] = cls_total

    return {
        "runtime_instance_timings_by_class": dict(by_class),
        "class_durations": class_durations,
        "class_durations_fwd": class_durations_fwd,
        "class_durations_bwd": class_durations_bwd,
        "step_phase_str": root_step_phase_str,
        "inclusive_forward_us": root_inclusive_forward_us,
        "inclusive_backward_us": root_inclusive_backward_us,
        "inclusive_optimize_us": root_inclusive_optimize_us,
        "inclusive_other_us": root_inclusive_other_us,
    }


def build_instance_timing_pipeline(events, source_files, class_map, step_infos, step_dur_us, roots=None, static_module_tree=None):
    """Top-level entry: runs Step 1→4 and returns timing_data fragment.

    Output keys:
      - runtime_instance_timings_by_class
      - class_durations / class_durations_fwd / class_durations_bwd
      - _timing_pipeline_stats (debug)
    """
    fwdbwd_index = build_fwdbwd_flow_index(events)
    attribution, stats = build_kernel_attribution_table(
        events, source_files, class_map, step_infos, fwdbwd_index, static_module_tree=static_module_tree,
    )
    instance_timing = rollup_instance_timing(attribution, events, step_infos, class_map, roots=roots)
    phase_totals = _compute_step_phase_totals(events, step_infos)
    panel = build_timing_panel_data(instance_timing, class_map, step_dur_us, phase_totals=phase_totals)
    panel["_timing_pipeline_stats"] = stats
    num_steps = max(1, len(step_infos))
    panel["step_kernel_us"] = stats.get("total_kernel_dur_us", 0) / num_steps
    
    total_attributed_us = sum(
        rec["self_us"]["forward"] + rec["self_us"]["backward"] + 
        rec["self_us"]["optimize"] + rec["self_us"]["other"] 
        for rec in instance_timing.values()
    )
    panel["unattributed_kernel_us"] = max(0, panel["step_kernel_us"] - total_attributed_us)
    return panel


# --------------------------------------------------------------------------
# Coverage summary — diagnostics over DAG groups
# --------------------------------------------------------------------------

def compute_timing_coverage_summary(groups):
    """Compute timing coverage statistics over DAG sub-groups.

    Walks the DAG ``groups`` list (as produced by ``generate_html_flowchart``
    with ``_return_data_only=True``) and reports how many sub-group instances
    have non-zero timing attributed to them.

    Statistics rules:
        * Sub-groups only — top-level / root groups (``depth <= 0``) are
          excluded so the figure reflects user-facing modules rather than the
          synthetic RootModule wrapper.
        * An instance is considered "timed" when ``kernel_us > 0`` OR
          ``total_us > 0`` (whichever field is present on the group dict).
        * Each list entry in the ``groups`` argument is treated as one
          instance — the caller is expected to pass the flat ``dag_groups``
          list which already contains one entry per instance occurrence.

    Args:
        groups: Iterable of group dicts. Each dict is expected to expose at
            least ``depth``; ``label``/``class_name`` and one of
            ``attr_name``/``id`` are used to populate the missed-instance
            list. ``kernel_us`` and/or ``total_us`` (in microseconds) are
            consulted to decide whether an instance is timed. Missing fields
            are treated as zero.

    Returns:
        dict with keys::

            {
              "total_instances": int,
              "timed_instances": int,
              "coverage_rate": float,           # in [0.0, 1.0]
              "no_timing_instances": [
                  {"class_name": str, "instance_key": str,
                   "kernel_us": float, "total_us": float},
                  ...
              ]
            }

        ``coverage_rate`` is ``0.0`` when ``total_instances == 0`` to avoid
        ``ZeroDivisionError``; callers can therefore safely format the value
        without extra guards.
    """
    total = 0
    timed = 0
    no_timing = []

    if not groups:
        return {
            "total_instances": 0,
            "timed_instances": 0,
            "coverage_rate": 0.0,
            "no_timing_instances": [],
        }

    for g in groups:
        if not isinstance(g, dict):
            continue
        # Skip the synthetic RootModule wrapper / any depth==0 entries; we only
        # want user-facing sub-groups in this report.
        depth = g.get("depth", 0)
        try:
            depth_i = int(depth) if depth is not None else 0
        except (TypeError, ValueError):
            depth_i = 0
        if depth_i <= 0:
            continue

        total += 1

        kernel_us = g.get("kernel_us", 0) or 0
        total_us = g.get("total_us", 0) or 0
        try:
            kernel_us = float(kernel_us)
        except (TypeError, ValueError):
            kernel_us = 0.0
        try:
            total_us = float(total_us)
        except (TypeError, ValueError):
            total_us = 0.0

        if kernel_us > 0 or total_us > 0:
            timed += 1
        else:
            class_name = g.get("class_name") or g.get("label") or ""
            instance_key = (
                g.get("attr_name")
                or g.get("id")
                or ""
            )
            no_timing.append({
                "class_name": class_name,
                "instance_key": instance_key,
                "kernel_us": kernel_us,
                "total_us": total_us,
            })

    coverage_rate = (timed / total) if total > 0 else 0.0
    return {
        "total_instances": total,
        "timed_instances": timed,
        "coverage_rate": coverage_rate,
        "no_timing_instances": no_timing,
    }
