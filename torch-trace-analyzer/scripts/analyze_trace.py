#!/usr/bin/env python3
import ast
import json
import gzip
import sys
import io
import os
import re
import copy
import tarfile
import argparse
import subprocess
import textwrap
import tokenize
from collections import Counter, defaultdict

# ---------------------------------------------------------------------------
# Module identity registration.
#
# `frontend_html.py` (sibling module in this scripts/ directory) does a normal
# ``from analyze_trace import ASTFrontend, ...`` at module load time so the
# split between DAG core and HTML rendering is a clean, explicit dependency
# (no globals().update / no spec_from_file_location dynamic loading).
#
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
# how it was loaded.  All names referenced by frontend_html (ASTFrontend,
# _build_class_map, _strip_inline_comment, build_static_module_tree, ...) are
# defined further down in this file BEFORE ``from frontend_html import ...``
# at the bottom triggers the cross-module import.
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


# ---------------------------------------------------------------------------
# PR3 — ConstantResolver is now the only production evaluator path.
# Keep the module-level switch for compatibility with older tests / scripts,
# but default it to ``True`` and no longer maintain a legacy fallback path.
# ---------------------------------------------------------------------------
USE_NEW_EVAL: bool = True

# Retain the report hook for diagnostics compatibility. PR3 no longer records
# legacy/new diffs, so the log stays empty unless future instrumentation adds
# explicit entries.
_AB_EVAL_DIFFS: list = []

import torch.nn as _torch_nn

# PyTorch 内置容器类（无独立 forward 数据流，不作为 leaf stub）
_NATIVE_CONTAINER_NAMES: frozenset = frozenset({
    'ModuleDict', 'ModuleList', 'Sequential',
    'ParameterDict', 'ParameterList',
})

def _is_nn_leaf_stub(class_name: str) -> bool:
    """True iff nn.<class_name> 是叶子计算 Module（非容器、且是 nn.Module 子类）。
    用于动态判断 native stub 是否应加入 torch_native_module_classes。
    """
    if class_name in _NATIVE_CONTAINER_NAMES:
        return False
    cls = getattr(_torch_nn, class_name, None)
    if cls is None:
        return False
    try:
        return issubclass(cls, _torch_nn.Module)
    except TypeError:
        return False


def _strip_inline_comment(code_line: str) -> str:
    """剥离行内 # 注释（尽量不影响字符串中的 #）。

    目的：用于括号计数/静态解析时避免注释中的 " :( " 等非平衡括号破坏 open_count。
    """
    # tokenize.untokenize 会保留大部分原始空白/结构，比简单 split('#') 更安全
    try:
        toks = list(tokenize.generate_tokens(io.StringIO(code_line).readline))
        toks = [t for t in toks if t.type != tokenize.COMMENT]
        return tokenize.untokenize(toks)
    except Exception:
        # 兜底：不处理引号场景，仅用于防止解析直接崩
        return code_line.split('#', 1)[0]


def _join_logical_lines(raw_lines, base_lineno):
    """Join multi-line statements by tracking open brackets/parens.

    Returns list of (start_lineno, joined_text) tuples (start_lineno is the
    absolute line number in the source file of the FIRST physical line that
    contributed to the joined logical line).
    """
    logical = []
    buf = ""
    buf_start = None
    open_count = 0
    for offset, line in enumerate(raw_lines):
        phys_lineno = base_lineno + offset
        stripped = line.strip()
        if not stripped or stripped.startswith('#'):
            if not buf:
                logical.append((phys_lineno, line))
            continue
        if not buf:
            buf_start = phys_lineno
        stripped_nc = _strip_inline_comment(stripped).strip()
        if not stripped_nc:
            continue
        buf += (" " if buf else "") + stripped_nc
        open_count += stripped_nc.count('(') + stripped_nc.count('[') + stripped_nc.count('{')
        open_count -= stripped_nc.count(')') + stripped_nc.count(']') + stripped_nc.count('}')
        if open_count <= 0:
            logical.append((buf_start, buf))
            buf = ""
            buf_start = None
            open_count = 0
    if buf:
        logical.append((buf_start, buf))
    return logical


def load_trace(filepath):
    if filepath.endswith(".gz"):
        with gzip.open(filepath, "rt", encoding="utf-8") as f:
            data = json.load(f)
    else:
        with open(filepath, "r") as f:
            data = json.load(f)
    return data


def format_duration(us):
    if us >= 1e6:
        return f"{us/1e6:.3f} s"
    elif us >= 1e3:
        return f"{us/1e3:.3f} ms"
    else:
        return f"{us:.2f} us"


def pct_str(part, total):
    return f"{part / total * 100:.2f}%" if total > 0 else "N/A"


# ---------------------------------------------------------------------------
# Phase 0: Metadata extraction
# ---------------------------------------------------------------------------

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


def load_model_code(code_path):
    if code_path is None:
        return {}
    source_files = {}
    if code_path.endswith(".tar.gz") or code_path.endswith(".tgz"):
        with tarfile.open(code_path, "r:gz") as tar:
            for member in tar.getmembers():
                if member.isfile() and member.name.endswith(".py"):
                    content = tar.extractfile(member).read().decode("utf-8", errors="replace")
                    basename = os.path.basename(member.name)
                    source_files[basename] = content.split("\n")
    elif os.path.isdir(code_path):
        for root, _, files in os.walk(code_path):
            for f in files:
                if f.endswith(".py"):
                    fpath = os.path.join(root, f)
                    with open(fpath, "r", errors="replace") as fp:
                        source_files[f] = fp.read().split("\n")
    return source_files


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
            else:
                phase = phase_name_from_label(level2_name)
                if phase == "optimize":
                    info["optimize"].append((level2["ts"], level2["ts"] + level2["dur"], level2.get("name", "")))
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

    best_phase = max(phase_overlap, key=phase_overlap.get)
    return best_phase if phase_overlap[best_phase] > 0 else "other"


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
    """Step 1: pair fwdbwd flow events (ph='s' on forward tid, ph='f' on
    backward tid) and resolve them to forward/backward op time ranges.

    Returns: {
        "by_bwd_tid": {tid: [entry, ...]},   # sorted by bwd_start_us
        "all": [entry, ...],                 # sorted by bwd_start_us
    }
    where each entry = {
        "flow_id": int,
        "fwd_ts": float, "fwd_tid": int,
        "bwd_ts": float, "bwd_tid": int,
    }
    The `_ts` values are the flow source/dest timestamps (μs).  Callers can
    look up the enclosing forward/backward op by interval-containment.
    """
    if not events:
        return {"by_bwd_tid": {}, "all": []}
    pairs = {}
    for e in events:
        if e.get("cat") != "fwdbwd":
            continue
        ph = e.get("ph")
        fid = e.get("id")
        if fid is None or ph not in ("s", "f"):
            continue
        slot = pairs.setdefault(fid, {})
        slot[ph] = e
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
    return {"by_bwd_tid": dict(by_tid), "all": entries}


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


def build_kernel_stack_cost_table(events, source_files, fwdbwd_index):
    """Step 1a: 将所有 kernel 对齐到"堆栈-开销"记录。

    Returns:
        dict[kernel_idx, {
            "dur_us":    float,
            "phase":     "fwd" | "bwd" | "other",
            "chains":    list[list[(fname, lineno, func_name)]],
            "mod_name":  str | None,
            "unmatched": bool,
        }]
    """
    if not events:
        return {}

    cpu_op_by_tid = _build_cpu_op_index_by_tid(events) if fwdbwd_index else {}
    ext_to_module = _build_external_id_to_module_map(events)
    bwd_tids = set((fwdbwd_index or {}).get("by_bwd_tid", {}).keys())

    # Forward kernels with usable user-frame chains; used by Path B to borrow
    # the nearest forward stack within the matched fwdbwd scope.
    fwd_kernel_buckets = defaultdict(list)  # tid -> [(ts, idx, chains)]
    kernel_rows = {}

    for idx, e in enumerate(events):
        if e.get("cat") != "kernel":
            continue
        dur = float(e.get("dur") or 0.0)
        if dur <= 0:
            continue
        ts = float(e.get("ts") or 0.0)
        tid = e.get("tid")
        traces = e.get("args", {}).get("stack", {}).get("stack_traces", [])
        is_bwd = (tid in bwd_tids) or _is_backward_trace(traces)
        chains = _parse_user_frames(traces, source_files)
        kernel_rows[idx] = {
            "event": e,
            "ts": ts,
            "tid": tid,
            "dur_us": dur,
            "traces": traces,
            "chains": chains,
            "is_bwd": is_bwd,
        }
        if (not is_bwd) and chains:
            fwd_kernel_buckets[tid].append((ts, idx, chains))

    for tid in fwd_kernel_buckets:
        fwd_kernel_buckets[tid].sort(key=lambda x: x[0])

    table = {}
    for idx, meta in kernel_rows.items():
        phase = "bwd" if meta["is_bwd"] else "fwd"
        if meta["traces"]:
            table[idx] = {
                "dur_us": meta["dur_us"],
                "phase": "bwd" if _is_backward_trace(meta["traces"]) else "fwd",
                "chains": meta["chains"],
                "mod_name": None,
                "unmatched": False,
            }
            continue

        borrowed_chains = []
        if meta["is_bwd"] and fwdbwd_index:
            scope = _resolve_fwdbwd_scope(fwdbwd_index, cpu_op_by_tid, meta["ts"], meta["tid"])
            if scope is not None:
                fwd_start, fwd_end, fwd_tid = scope
                nearest = None
                nearest_dist = None
                for cand_ts, cand_idx, cand_chains in fwd_kernel_buckets.get(fwd_tid, []):
                    if not (fwd_start <= cand_ts <= fwd_end):
                        continue
                    dist = abs(cand_ts - meta["ts"])
                    if nearest is None or dist < nearest_dist:
                        nearest = cand_chains
                        nearest_dist = dist
                if nearest:
                    borrowed_chains = nearest

        if borrowed_chains:
            table[idx] = {
                "dur_us": meta["dur_us"],
                "phase": "bwd",
                "chains": borrowed_chains,
                "mod_name": None,
                "unmatched": False,
            }
            continue

        ext_id = meta["event"].get("args", {}).get("External id")
        mod_name = ext_to_module.get(ext_id)
        if not mod_name:
            mod_name, _pidx = find_module_parent(meta["event"], events)
        normalized = _normalize_runtime_module_name(mod_name) if mod_name else None
        class_name = normalized.get("class_name") if normalized else None
        table[idx] = {
            "dur_us": meta["dur_us"],
            "phase": phase,
            "chains": [],
            "mod_name": class_name,
            "unmatched": class_name is None,
        }

    return table


def build_kernel_attribution_table(events, source_files, class_map, step_infos, fwdbwd_index):
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

    cpu_op_by_tid = _build_cpu_op_index_by_tid(events) if fwdbwd_index else {}
    ext_to_module = _build_external_id_to_module_map(events)

    # Index forward kernels by enclosing forward scope (fwd_tid, [fwd_start,fwd_end])
    # — actually we don't index globally; instead, for each backward kernel
    # we find its fwd scope and re-scan forward kernels within that range.
    # To make that efficient, pre-bucket forward kernels (with stack_traces)
    # by tid, sorted by ts, and only those classified as 'forward' phase.
    fwd_kernel_buckets = defaultdict(list)  # tid -> [(ts, ts+dur, kernel_idx, frames)]
    kernel_meta = {}  # idx -> dict
    for idx, e in enumerate(events):
        if e.get("cat") != "kernel":
            continue
        dur = float(e.get("dur") or 0.0)
        if dur <= 0:
            continue
        ts = float(e.get("ts") or 0.0)
        traces = e.get("args", {}).get("stack", {}).get("stack_traces", [])
        is_bwd = _is_backward_trace(traces)
        kernel_meta[idx] = {
            "ts": ts, "dur": dur, "tid": e.get("tid"),
            "is_bwd": is_bwd, "traces": traces, "ext_id": e.get("args", {}).get("External id"),
        }
        if not is_bwd and traces:
            chains = _parse_user_frames(traces, source_files)
            # The fwd_kernel_buckets entry is currently informational
            # (no downstream consumer yet); record the first non-empty
            # chain so the bucket schema stays stable.
            if chains:
                fwd_kernel_buckets[e.get("tid")].append((ts, ts + dur, idx, chains[0]))
    for tid in fwd_kernel_buckets:
        fwd_kernel_buckets[tid].sort(key=lambda x: x[0])

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

    for idx, meta in kernel_meta.items():
        stats["total_kernels"] += 1
        stats["total_kernel_dur_us"] += meta["dur"]
        traces = meta["traces"]
        is_bwd = meta["is_bwd"]
        if not traces:
            mod_name = ext_to_module.get(meta["ext_id"])
            if not mod_name:
                # Second-level fallback: walk parent chain via _P pointers.
                mod_name, _pidx = find_module_parent(events[idx], events)
            if mod_name and _wrapped_fallback(idx, mod_name):
                stats["wrapped_fallback"] += 1
                continue
            if is_bwd:
                stats["bwd_unattributed"] += 1
            else:
                stats["fwd_unattributed"] += 1
            continue

        chains = _parse_user_frames(traces, source_files)
        if not chains:
            # Try wrapped fallback
            mod_name = ext_to_module.get(meta["ext_id"])
            if not mod_name:
                mod_name, _pidx = find_module_parent(events[idx], events)
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
            chain_keys = _extract_instance_keys_from_stack(chain, class_map)
            if not chain_keys:
                continue
            leaf = chain_keys[-1]
            hit_counts[leaf] = hit_counts.get(leaf, 0) + 1
            total_hits += 1

        if total_hits == 0:
            mod_name = ext_to_module.get(meta["ext_id"])
            if not mod_name:
                mod_name, _pidx = find_module_parent(events[idx], events)
            if mod_name and _wrapped_fallback(idx, mod_name):
                stats["wrapped_fallback"] += 1
                continue
            if is_bwd:
                stats["bwd_unattributed"] += 1
            else:
                stats["fwd_unattributed"] += 1
            continue

        if is_bwd and fwdbwd_index and fwdbwd_index["all"]:
            # Try to narrow via fwdbwd flow: find forward scope, then look
            # for forward kernels whose stack reproduces the leaf keys —
            # if there's a unique match, we keep the current weights.
            scope = _resolve_fwdbwd_scope(fwdbwd_index, cpu_op_by_tid, meta["ts"], meta["tid"])
            if scope is not None:
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
            if k[0] in roots and k[1] == "<root>":
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

def build_timing_panel_data(instance_timing, class_map, step_dur_us):
    """Step 4: convert instance_timing → timing_data fields:
      - runtime_instance_timings_by_class (instance level)
      - class_durations / class_durations_fwd / class_durations_bwd
        (class level = sum of all instances' inclusive)
    """
    by_class = defaultdict(list)
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
        }
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
    }


def build_instance_timing_pipeline(events, source_files, class_map, step_infos, step_dur_us, roots=None):
    """Top-level entry: runs Step 1→4 and returns timing_data fragment.

    Output keys:
      - runtime_instance_timings_by_class
      - class_durations / class_durations_fwd / class_durations_bwd
      - _timing_pipeline_stats (debug)
    """
    fwdbwd_index = build_fwdbwd_flow_index(events)
    attribution, stats = build_kernel_attribution_table(
        events, source_files, class_map, step_infos, fwdbwd_index,
    )
    instance_timing = rollup_instance_timing(attribution, events, step_infos, class_map, roots=roots)
    panel = build_timing_panel_data(instance_timing, class_map, step_dur_us)
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



def _build_class_map(source_files, ast_frontends=None):
    ast_map, failed_files = _build_class_map_ast(source_files, ast_frontends=ast_frontends)
    if failed_files:
        print(f"[WARN] AST parse failed for {len(failed_files)} file(s): {sorted(failed_files)}", file=sys.stderr)
    return ast_map


def _find_class_for_line(fname, lineno, class_map):
    for (f, cname), info in class_map.items():
        if f == fname and info["start"] <= lineno <= info["end"]:
            for mname, (ms, me) in info["methods"].items():
                if ms <= lineno <= me:
                    return cname, mname
            return cname, None
    return None, None


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


# ---------------------------------------------------------------------------
# Kernel call stack analysis
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# lagrange_torch source code reference
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Per-thread timeline analysis
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Chrome tracing visualization
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Phase 1: Build parent-child relationships
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Phase 2: Identify threads and their roles
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Phase 3: Training step decomposition
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Phase 4: Module analysis (per-thread aware)
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Phase 5: GPU timeline analysis (kernel hotspot + gaps + host mapping)
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Phase 6: Device/Host overview
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Phase 7: Worker thread analysis
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Phase 8: Communication/Compute overlap analysis
# ---------------------------------------------------------------------------

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


# ===========================================================================
# Printing / Output
# ===========================================================================

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


# ===========================================================================
# Source code hotspot report
# ===========================================================================

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


# ===========================================================================
# Per-thread timeline report
# ===========================================================================

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


# ===========================================================================
# Kernel call stack report
# ===========================================================================

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


# ===========================================================================
# lagrange_torch reference report
# ===========================================================================

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


# ===========================================================================
# Markdown report
# ===========================================================================

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


# ===========================================================================
# HTML Flowchart Generation (Source-Code First)
# ===========================================================================


def _resolve_str_list_iterative(
    expr: str,
    fname: str,
    cname: str,
    source_lines,
    loop_var_to_str_items: dict,
    class_str_list_attrs: dict,
    file_str_list_globals: dict,
    cond_branch_stack: list,
    conditional_mode: str = "default",
    _global_str_lists_unique: dict = None,
):
    """Phase B iterative multi-step string-list resolver.

    Resolves a dynamic key expression (used in ModuleDict subscript or setattr
    name_expr) to a concrete list[str] of keys via iterative derivation.

    Returns list[str] on success, None when resolution fails.
    """
    if _global_str_lists_unique is None:
        _global_str_lists_unique = {}
    if expr is None:
        return None
    e = expr.strip()
    if not e:
        return None

    # Step 1: direct loop variable lookup
    if e in loop_var_to_str_items:
        items = list(loop_var_to_str_items[e])
        # If the expression itself is a plain loop var, return its items
        return items

    # Parse the expression as AST
    try:
        _node = ast.parse(e, mode='eval').body
    except SyntaxError:
        return None

    # Step 2: f-string expansion
    # e.g. f"{name}_tower" with name ∈ ['relation', 'gift', 'stay']
    # → ['relation_tower', 'gift_tower', 'stay_tower']
    if isinstance(_node, ast.JoinedStr):
        return _expand_fstring_with_loop_vars(_node, loop_var_to_str_items)

    # Step 3: self.<attr> referencing a known class str-list attr
    if (isinstance(_node, ast.Attribute)
            and isinstance(_node.value, ast.Name)
            and _node.value.id == 'self'):
        items = (class_str_list_attrs.get(cname) or {}).get(_node.attr)
        if items:
            return list(items)

    # Step 4: UPPER_CASE file/global constant
    _const_name = None
    if isinstance(_node, ast.Name):
        _const_name = _node.id
    elif isinstance(_node, ast.Attribute):
        _const_name = _node.attr
    if (_const_name and _const_name.isupper() and _const_name[:1].isalpha()
            and all(c.isalnum() or c == '_' for c in _const_name)):
        file_slots = file_str_list_globals.get(fname, {})
        if _const_name in file_slots:
            return list(file_slots[_const_name])
        if _const_name in _global_str_lists_unique:
            return list(_global_str_lists_unique[_const_name])

    # Step 5: bare Name that is NOT a loop var and NOT UPPER_CASE —
    # indicates a runtime/parameter variable we cannot statically resolve.
    if isinstance(_node, ast.Name):
        # Check if it might be resolvable via dotted Cfg.KEYS (already handled)
        # or if it's truly opaque (e.g. cfg.dynamic_keys).
        return None

    # Step 6: dotted attribute that is NOT self.* (e.g. cfg.dynamic_keys)
    if isinstance(_node, ast.Attribute):
        return None

    return None


def _expand_fstring_with_loop_vars(joined_str_node, loop_var_to_str_items: dict):
    """Expand an f-string AST node by substituting loop variables.

    For each FormattedValue in the JoinedStr, if the referenced variable is a
    known loop variable with string items, produce the cartesian expansion.
    Only single-variable f-strings are supported (one FormattedValue referencing
    one loop var); multi-variable f-strings return None.

    Returns list[str] on success, None on failure.
    """
    # Collect parts: each is either a constant string or a variable reference
    parts = []  # list of (kind, data) where kind='const' data=str | kind='var' data=items_list
    for value in joined_str_node.values:
        if isinstance(value, ast.Constant) and isinstance(value.value, str):
            parts.append(('const', value.value))
        elif isinstance(value, ast.FormattedValue):
            # Only support simple Name references
            inner = value.value
            if isinstance(inner, ast.Name):
                var_name = inner.id
                if var_name in loop_var_to_str_items:
                    parts.append(('var', list(loop_var_to_str_items[var_name])))
                else:
                    return None
            else:
                return None
        else:
            return None

    if not parts:
        return None

    # Check we have at least one variable part
    has_var = any(k == 'var' for k, _ in parts)
    if not has_var:
        # Pure constant f-string (no interpolation) — just concat
        result = ''.join(d for _, d in parts)
        return [result]

    # Cartesian expansion: for each variable part, iterate its items;
    # for constant parts, keep as-is.
    # Only support single variable dimension for now (avoid combinatorial explosion).
    var_parts = [(i, items) for i, (k, items) in enumerate(parts) if k == 'var']
    if len(var_parts) > 1:
        # Multiple independent variables in one f-string — check if they all
        # reference the same loop var (same items list).
        first_items = var_parts[0][1]
        if all(items == first_items for _, items in var_parts):
            # Same variable repeated — expand once
            pass
        else:
            return None

    # Single variable (or same variable repeated): expand
    items_list = var_parts[0][1]
    results = []
    for item in items_list:
        built = []
        for kind, data in parts:
            if kind == 'const':
                built.append(data)
            else:
                built.append(str(item))
        results.append(''.join(built))
    return results


def __getattr__(name):  # noqa: D401 (PEP 562 module-level __getattr__)
    """Lazily re-export ``frontend_html`` entry points.

    Triggered only when ``analyze_trace.<name>`` is accessed for one of the
    HTML rendering functions.  By the time any consumer reaches this lookup
    both modules have finished loading, so we can safely import the symbol
    without re-entering a half-initialised ``frontend_html``.
    """
    if name in _FRONTEND_HTML_REEXPORT_NAMES:
        import frontend_html as _fh
        value = getattr(_fh, name)
        # Cache on this module so repeat lookups are O(1) and so that the
        # importlib proxy used in the ``analyze_trace_runtime`` mode also
        # picks up the name through its live globals delegation.
        globals()[name] = value
        return value
    raise AttributeError(f"module 'analyze_trace' has no attribute {name!r}")


def main():
    parser = argparse.ArgumentParser(description="分析 PyTorch 训练 trace JSON 文件")
    parser.add_argument("trace_file", nargs="?", default=None, help="trace JSON 文件路径 (可选，若仅生成源码流程图则不需要)")
    parser.add_argument("--max-level", type=int, default=None, help="最大显示的 Module 层级深度")
    parser.add_argument("--output", "-o", type=str, default=None, help="输出 Markdown 报告文件路径")
    parser.add_argument("--no-tree", action="store_true", help="不输出 Module 层级树")
    parser.add_argument("--json-output", type=str, default=None, help="输出 JSON 格式分析结果")
    parser.add_argument("--code-path", type=str, default=None, help="模型源码路径（目录或 .tar.gz）")
    parser.add_argument("--screenshot", action="store_true", help="生成 Chrome Tracing 可视化截图")
    parser.add_argument("--html-flowchart", type=str, default=None, help="生成 HTML 模块流程图路径")
    parser.add_argument("--use-new-eval", action="store_true",
                        help="兼容旧脚本参数；静态分析代码已移除，该参数仅保留 CLI 兼容性")
    parser.add_argument("--ab-eval-report", type=str, default=None,
                        help="写入当前求值器状态摘要 JSON（静态分析代码清理后仅保留 CLI 兼容性）")
    args = parser.parse_args()

    global USE_NEW_EVAL
    USE_NEW_EVAL = True
    if args.use_new_eval:
        print("  [cleanup] --use-new-eval 已保留为兼容参数；当前不再包含静态分析求值器")

    if args.html_flowchart:
        raise NotImplementedError(
            "static HTML flowchart generation has been removed in feat/static-cleanup"
        )

    if not args.trace_file:
        print("  ❌ 错误: 需要提供 trace 文件路径")
        sys.exit(1)

    print(f"  正在加载 trace 文件: {args.trace_file}")
    data = load_trace(args.trace_file)
    events = data.get("traceEvents", [])
    print(f"  共 {len(events)} 个事件")

    meta = extract_metadata(data, events)

    trace_type = detect_trace_type(events)
    if trace_type != "training":
        print(f"  ⚠️ 检测到这是 {trace_type} trace，当前 skill 仅支持训练 trace 分析")
        print("  如确认是训练 trace，将继续分析")

    has_code_loc, has_stack_traces = detect_enhanced_trace(events)
    is_enhanced = has_code_loc and has_stack_traces
    if is_enhanced:
        print("  检测到增强 trace（含 Code Location 和 stack_traces）")
        if not args.code_path:
            print("  ❌ 错误: 增强 trace 需要提供模型源码才能进行源码级分析")
            print("  请通过 --code-path 参数提供模型源码目录或 tar.gz 压缩包")
            print("  示例: python3 scripts/analyze_trace.py trace.json --code-path code_commit.tar.gz")
            sys.exit(1)

    source_files = {}
    if args.code_path:
        print(f"  正在加载模型源码: {args.code_path}")
        source_files = load_model_code(args.code_path)
        print(f"  加载了 {len(source_files)} 个源文件: {', '.join(sorted(source_files.keys())[:10])}")
        if len(source_files) > 10:
            print(f"    ... 等共 {len(source_files)} 个文件")

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

    print("  正在构建事件调用关系...")
    build_parent_index(events)
    thread_indices = add_func_call_parent(events)

    print("  正在分析线程结构...")
    thread_info = classify_threads(events, meta)

    step_decomp = analyze_step_decomposition(events, thread_info, profiler_steps)

    if source_files and is_enhanced:
        print("  正在分析源码热点...")
        src_info = analyze_source_hotspots(events, source_files, profiler_steps)
    else:
        src_info = None

    print("  正在构建 Module 层级...")
    module_tree, root_modules, mod_info = build_module_hierarchy(events, source_files)

    if args.output:
        print(f"  正在生成 Markdown 报告: {args.output}")
        report = generate_markdown_report(
            meta, thread_info, module_tree, mod_info, step_dur_us,
            step_decomp=step_decomp, source_hotspots=src_info
        )
        with open(args.output, "w", encoding="utf-8") as f:
            f.write(report)
        print(f"  Markdown 报告已保存到: {args.output}")
    else:
        print_summary(meta, thread_info, module_tree, mod_info, step_dur_us,
                      max_level=args.max_level, show_tree=(not args.no_tree),
                      step_decomp=step_decomp, source_hotspots=src_info)

    if args.json_output:
        result = {
            "metadata": meta,
            "step_dur_us": step_dur_us,
            "step_decomposition": step_decomp,
            "source_hotspots": src_info,
            "module_tree": module_tree,
            "module_info": mod_info,
        }
        with open(args.json_output, "w") as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
        print(f"  JSON 结果已保存到: {args.json_output}")

    _emit_ab_summary_if_enabled(args)


def _emit_ab_summary_if_enabled(args):
    """PR3: optionally persist the current evaluator status for debugging."""
    report = getattr(args, "ab_eval_report", None)
    if not report:
        return
    payload = {
        "mode": "constant_resolver_only",
        "use_new_eval": True,
        "total": len(_AB_EVAL_DIFFS),
        "matches": 0,
        "mismatches": 0,
        "all": _AB_EVAL_DIFFS,
    }
    try:
        with open(report, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
        print(f"  [PR3] evaluator status saved to: {report}")
    except Exception as e:
        print(f"  ⚠️ failed to write evaluator status report: {e}")


