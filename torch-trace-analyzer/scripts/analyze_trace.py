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
import logging
from dataclasses import dataclass
from collections import Counter, defaultdict, deque

LOG = logging.getLogger(__name__)

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


try:
    from scripts.ast_types import Scope, IntValue, ListValue, _RECURSING
    from scripts.ast_constants import ConstantTable
    from scripts.ast_resolver import ConstantResolver
    from scripts.ast_frontend import ASTFrontend
except ModuleNotFoundError:
    from ast_types import Scope, IntValue, ListValue, _RECURSING
    from ast_constants import ConstantTable
    from ast_resolver import ConstantResolver
    from ast_frontend import ASTFrontend

try:
    from scripts.trace_io import load_model_code, load_trace
except ModuleNotFoundError:
    from trace_io import load_model_code, load_trace

try:
    from scripts.source_index import (
        _build_ast_frontends,
        _build_class_map,
        _build_class_map_ast,
        _find_class_for_line,
        build_module_like_set,
    )
except ModuleNotFoundError:
    from source_index import (
        _build_ast_frontends,
        _build_class_map,
        _build_class_map_ast,
        _find_class_for_line,
        build_module_like_set,
    )

try:
    from scripts.source_hotspot import (
        LAGRANGE_TORCH_REPO,
        analyze_kernel_call_stacks,
        analyze_source_hotspots,
        enrich_kernel_modules_with_source,
        extract_lagrange_refs,
    )
except ModuleNotFoundError:
    from source_hotspot import (
        LAGRANGE_TORCH_REPO,
        analyze_kernel_call_stacks,
        analyze_source_hotspots,
        enrich_kernel_modules_with_source,
        extract_lagrange_refs,
    )

try:
    from scripts.report_summary import (
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
except ModuleNotFoundError:
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

try:
    from scripts.common_utils import (
        _dedup_consecutive_frames,
        _extract_frame_class_name,
        _frame_get,
        _frame_with_func,
        _is_user_source_frame,
        _join_logical_lines,
        _normalize_parallel_wrapper_frame,
        _strip_inline_comment,
        format_duration,
        pct_str,
    )
except ModuleNotFoundError:
    from common_utils import (
        _dedup_consecutive_frames,
        _extract_frame_class_name,
        _frame_get,
        _frame_with_func,
        _is_user_source_frame,
        _join_logical_lines,
        _normalize_parallel_wrapper_frame,
        _strip_inline_comment,
        format_duration,
        pct_str,
    )


try:
    from scripts.trace_analysis import (
        _clip_dur_to_step,
        _find_step_for_event,
        _intervals_overlap,
        _intervals_total,
        _merge_intervals,
        _merge_phase_intervals,
        _iter_descendants_within_step,
        add_func_call_parent,
        analyze_comm_compute_overlap,
        analyze_device_host,
        analyze_gpu_timeline,
        analyze_modules_by_thread,
        analyze_per_thread_timeline,
        analyze_step_decomposition,
        analyze_worker_threads,
        build_main_thread_hierarchy,
        build_parent_index,
        classify_kernel_phase,
        classify_threads,
        detect_enhanced_trace,
        detect_trace_type,
        extract_metadata,
        extract_step_phase_intervals,
        find_module_parent,
        generate_trace_screenshot,
        overlap,
        phase_name_from_label,
    )
except ModuleNotFoundError:
    from trace_analysis import (
        _clip_dur_to_step,
        _find_step_for_event,
        _intervals_overlap,
        _intervals_total,
        _merge_intervals,
        _merge_phase_intervals,
        _iter_descendants_within_step,
        add_func_call_parent,
        analyze_comm_compute_overlap,
        analyze_device_host,
        analyze_gpu_timeline,
        analyze_modules_by_thread,
        analyze_per_thread_timeline,
        analyze_step_decomposition,
        analyze_worker_threads,
        build_main_thread_hierarchy,
        build_parent_index,
        classify_kernel_phase,
        classify_threads,
        detect_enhanced_trace,
        detect_trace_type,
        extract_metadata,
        extract_step_phase_intervals,
        find_module_parent,
        generate_trace_screenshot,
        overlap,
        phase_name_from_label,
    )


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


# ===========================================================================
# End of PR1 ConstantTable + ConstantResolver skeleton
# ===========================================================================


# ---------------------------------------------------------------------------
# Phase 0: Metadata extraction
# ---------------------------------------------------------------------------







# ---------------------------------------------------------------------------
# Iter18: AST-based static module-tree extractor (inlined from fake_runner.py
# Phase 1).
#
# This is a *ground-truth lookup* that runs in-process during static analysis.
# It does not replace the existing string/regex-based logic in
# build_static_module_tree(); instead, its output is exposed on each
# tree[cname] entry as ``_ast_*`` keys so downstream consumers (and the
# existing edge/expansion code) can consult an authoritative AST view when
# their heuristics are uncertain (e.g. ModuleList element type for a
# DenseTower built with ``[Dense(...) for _ in output_dims]``, dynamic
# ``setattr(self, f"group_tower_{i}", GroupTower(...))`` inside a ``for i in
# range(N)`` loop, etc.).
#
# The class is a near-verbatim port of fake_runner.py's ``ModuleTreeExtractor``
# adapted to consume ``source_files`` (a dict of ``{filename: [lines]}``)
# directly rather than reading from disk.  Behaviour and outputs are
# semantically identical to the standalone tool whose Phase 1 has been
# verified end-to-end against ``fake_runner_output/module_tree.json``.
# ---------------------------------------------------------------------------

class _AST_ModuleTreeExtractor:
    """Inlined fake_runner.py Phase 1 ``ModuleTreeExtractor``.

    Walks ``__init__`` of every ``nn.Module`` subclass and records:
      • ``self.attr = SomeModule(...)``               → direct child
      • ``setattr(self, name, SomeModule(...))``      → dynamic / literal child
      • ``self.container.append(SomeModule(...))``    → container element
      • Local-var tracking: ``v = SomeClass(...); self.x.append(v)``
                           ``v = SomeClass(...); setattr(self, n, v)``
      • For-loop / if / with bodies are walked (no condition evaluation).

    Exposes:
      • ``classes``                  : {cname: {file, node, bases}}
      • ``nn_module_classes``        : set[cname]
      • ``class_assignments[cname]`` : list[dict] (raw __init__ assigns)
      • ``ast_class_attrs[cname]``   : {attr -> child_class}     (literal)
      • ``ast_container_elems[cname][container] -> [child_class, ...]``
      • ``ast_container_kinds[cname][container] -> "ModuleList"|"ModuleDict"|"Sequential"``
      • ``ast_dynamic_attrs[cname]`` : list[child_class]   (setattr w/ non-literal name)
      • ``module_tree``              : list[dict]  (recursive tree from build_tree)
    """

    _NON_MODULE_TYPES = {
        "OrderedDict", "dict", "list", "set", "tuple",
        "defaultdict", "Counter", "deque",
    }

    def __init__(self, source_files, ast_frontends=None):
        # source_files: {fname: [lines]}
        self.source_files = source_files
        self.ast_frontends = ast_frontends or {}
        self.classes = {}            # cname -> {file, node, bases}
        self.class_assignments = {}  # cname -> list of dicts
        self.ast_class_attrs = defaultdict(dict)
        self.ast_container_elems = defaultdict(lambda: defaultdict(list))
        self.ast_container_kinds = defaultdict(dict)
        self.ast_dynamic_attrs = defaultdict(list)
        self.nn_module_classes = set()
        self.module_tree = []
        self._parse_all_files()
        self._derive_nn_module_set()
        self._build_class_assignments()

    # ---- Parsing -----------------------------------------------------

    def _parse_all_files(self):
        for fname, lines in self.source_files.items():
            tree = None
            fe = self.ast_frontends.get(fname)
            if fe is not None:
                tree = fe.tree
            else:
                try:
                    source = "\n".join(lines)
                    tree = ast.parse(source, filename=fname)
                except SyntaxError:
                    tree = None
            if tree is None:
                continue
            for node in ast.walk(tree):
                if isinstance(node, ast.ClassDef):
                    # Last-write-wins on duplicate class names (matches
                    # _build_class_map's behaviour).
                    self.classes[node.name] = {
                        "file": fname,
                        "node": node,
                        "bases": [self._get_base_name(b) for b in node.bases],
                    }

    def _get_base_name(self, node) -> str:
        if isinstance(node, ast.Name):
            return node.id
        elif isinstance(node, ast.Attribute):
            return f"{self._get_base_name(node.value)}.{node.attr}"
        return ""

    def _derive_nn_module_set(self):
        # Iteratively close over: any class whose any base is nn.Module/Module
        # or another already-known nn.Module subclass.
        known = set()
        # Direct hits
        for cname, info in self.classes.items():
            for b in info["bases"]:
                tail = b.split(".")[-1]
                if tail in ("Module", "nn.Module"):
                    known.add(cname)
                    break
                if b in ("nn.Module", "torch.nn.Module"):
                    known.add(cname)
                    break
        # Closure
        changed = True
        while changed:
            changed = False
            for cname, info in self.classes.items():
                if cname in known:
                    continue
                for b in info["bases"]:
                    tail = b.split(".")[-1]
                    if tail in known or b in known:
                        known.add(cname)
                        changed = True
                        break
        self.nn_module_classes = known

    # ---- __init__ walking -------------------------------------------

    def _build_class_assignments(self):
        for cname in self.classes:
            assigns = self._find_init_assignments(cname)
            self.class_assignments[cname] = assigns
            # Materialise lookups
            for a in assigns:
                attr = a["attr"]
                child = a["class"]
                kind = a.get("container_type", "")
                if a.get("_is_container_element"):
                    cont = a.get("_container_attr") or attr.split("[")[0]
                    if child:
                        self.ast_container_elems[cname][cont].append(child)
                    continue
                if a.get("dynamic"):
                    if child:
                        self.ast_dynamic_attrs[cname].append(child)
                    continue
                if a.get("is_container") and kind:
                    self.ast_container_kinds[cname][attr] = kind
                    # The container itself has no immediate child class —
                    # elements are filled by .append() patterns.
                    continue
                if child and attr and "<dynamic>" not in attr:
                    # First write wins — mirrors how the regex pass uses
                    # ``attrs.setdefault``.
                    self.ast_class_attrs[cname].setdefault(attr, child)

    def _find_init_assignments(self, class_name):
        if class_name not in self.classes:
            return []
        cls_node = self.classes[class_name]["node"]
        assignments = []
        for item in cls_node.body:
            if isinstance(item, ast.FunctionDef) and item.name == "__init__":
                self._walk_init_body(item.body, assignments, class_name)
        return assignments

    def _walk_init_body(self, stmts, assignments, parent_class, local_vars=None):
        if local_vars is None:
            local_vars = {}
        for stmt in stmts:
            # local_var = SomeClass(...) and self.attr = SomeClass(...)
            if isinstance(stmt, ast.Assign):
                for target in stmt.targets:
                    self._check_self_assign(target, stmt.value, assignments)
                    if isinstance(target, ast.Name):
                        cls, _, _ = self._infer_class_from_value(stmt.value)
                        if cls:
                            local_vars[target.id] = cls
            # call expressions (setattr / append)
            elif isinstance(stmt, ast.Expr) and isinstance(stmt.value, ast.Call):
                call = stmt.value
                if isinstance(call.func, ast.Name) and call.func.id == "setattr":
                    self._check_setattr(call, assignments, local_vars)
                elif isinstance(call.func, ast.Attribute) and call.func.attr == "append":
                    self._check_container_append(call, assignments, local_vars)
            # Recurse into compound statements
            elif isinstance(stmt, ast.For):
                self._walk_init_body(stmt.body, assignments, parent_class, local_vars)
                self._walk_init_body(stmt.orelse, assignments, parent_class, local_vars)
            elif isinstance(stmt, (ast.If, ast.With)):
                body = getattr(stmt, "body", []) or []
                orelse = getattr(stmt, "orelse", []) or []
                self._walk_init_body(body, assignments, parent_class, local_vars)
                self._walk_init_body(orelse, assignments, parent_class, local_vars)
            elif isinstance(stmt, ast.Try):
                self._walk_init_body(stmt.body, assignments, parent_class, local_vars)
                for h in stmt.handlers:
                    self._walk_init_body(h.body, assignments, parent_class, local_vars)
                self._walk_init_body(stmt.orelse, assignments, parent_class, local_vars)
                self._walk_init_body(stmt.finalbody, assignments, parent_class, local_vars)

    def _check_self_assign(self, target, value, assignments):
        if not isinstance(target, ast.Attribute):
            return
        if not (isinstance(target.value, ast.Name) and target.value.id == "self"):
            return
        attr_name = target.attr
        class_name, is_container, container_type = self._infer_class_from_value(value)
        if class_name or is_container:
            assignments.append({
                "attr": attr_name,
                "class": class_name,
                "is_container": is_container,
                "container_type": container_type,
                "line": getattr(target, "lineno", 0),
            })

    def _check_setattr(self, call_node, assignments, local_vars):
        if len(call_node.args) < 3:
            return
        obj, name_arg, value_arg = call_node.args[0], call_node.args[1], call_node.args[2]
        if not (isinstance(obj, ast.Name) and obj.id == "self"):
            return
        if isinstance(name_arg, ast.Constant) and isinstance(name_arg.value, str):
            attr_name = name_arg.value
            dynamic = False
        else:
            attr_name = "<dynamic>"
            dynamic = True
        class_name, is_container, container_type = self._infer_class_from_value(value_arg)
        # Resolve `setattr(self, n, var)` where var is a local-var ctor.
        if not class_name and isinstance(value_arg, ast.Name) and local_vars:
            class_name = local_vars.get(value_arg.id, "")
        if class_name or is_container:
            assignments.append({
                "attr": attr_name,
                "class": class_name,
                "is_container": is_container,
                "container_type": container_type,
                "line": getattr(call_node, "lineno", 0),
                "dynamic": dynamic,
            })

    def _check_container_append(self, call_node, assignments, local_vars):
        func = call_node.func
        if not isinstance(func, ast.Attribute):
            return
        if not isinstance(func.value, ast.Attribute):
            return
        if not (isinstance(func.value.value, ast.Name) and func.value.value.id == "self"):
            return
        container_attr = func.value.attr
        if not call_node.args:
            return
        element_value = call_node.args[0]
        class_name, _, _ = self._infer_class_from_value(element_value)
        if not class_name and isinstance(element_value, ast.Name) and local_vars:
            class_name = local_vars.get(element_value.id, "")
        if class_name:
            assignments.append({
                "attr": f"{container_attr}[*]",
                "class": class_name,
                "is_container": False,
                "container_type": "",
                "line": getattr(call_node, "lineno", 0),
                "dynamic": False,
                "_is_container_element": True,
                "_container_attr": container_attr,
            })

    def _infer_class_from_value(self, value):
        """Return (class_name, is_container, container_type)."""
        if isinstance(value, ast.Call):
            func = value.func
            func_name = ""
            if isinstance(func, ast.Name):
                func_name = func.id
            elif isinstance(func, ast.Attribute):
                func_name = func.attr
                if isinstance(func.value, ast.Name) and func.value.id == "nn":
                    func_name = f"nn.{func.attr}"
            if func_name in ("ModuleList", "nn.ModuleList"):
                return ("", True, "ModuleList")
            elif func_name in ("ModuleDict", "nn.ModuleDict"):
                return ("", True, "ModuleDict")
            elif func_name in ("Sequential", "nn.Sequential"):
                return ("", True, "Sequential")
            if func_name in self._NON_MODULE_TYPES:
                return ("", False, "")
            if func_name and func_name[0:1].isupper():
                return (func_name, False, "")
            if isinstance(func, ast.Attribute) and func.attr[0:1].isupper():
                return (func.attr, False, "")
        return ("", False, "")

    # ---- Tree builder ------------------------------------------------

    def build_tree(self, root_class, root_path=""):
        self.module_tree = []
        if root_class and root_class in self.classes:
            self._build_recursive(root_class, root_path, depth=0,
                                  ancestor_classes=set())
        return self.module_tree

    def _build_recursive(self, class_name, prefix, depth, ancestor_classes):
        if class_name in ancestor_classes:
            return
        new_ancestors = ancestor_classes | {class_name}
        for assign in self.class_assignments.get(class_name, []):
            attr = assign["attr"]
            path = f"{prefix}.{attr}" if prefix else attr
            child_class = assign["class"]
            entry = {
                "attr_path": path,
                "class": child_class if child_class else assign.get("container_type", "Unknown"),
                "depth": depth + 1,
                "is_container": assign.get("is_container", False),
                "container_type": assign.get("container_type", ""),
                "parent_class": class_name,
                "file": self.classes.get(class_name, {}).get("file", ""),
                "line": assign.get("line", 0),
                "dynamic": assign.get("dynamic", False),
            }
            self.module_tree.append(entry)
            if (child_class and child_class in self.classes
                    and child_class in self.nn_module_classes):
                self._build_recursive(child_class, path, depth + 1, new_ancestors)












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




def _build_source_dependency_order(source_files, class_map, ast_frontends=None):
    """Parse forward() methods to extract the order in which child modules are called.
    Returns a dict: {ClassName: [child_attr_name_in_order, ...]}
    Also returns attr_to_class: {(file, attr): ClassName} from __init__ self.xxx = SomeClass(...)
    """
    module_call_order = {}
    attr_to_class = {}  # (fname, attr_name) -> class_name
    if ast_frontends is None:
        ast_frontends = _build_ast_frontends(source_files)

    def _get_ast_frontend(fname):
        return ast_frontends.get(fname)

    def _norm_index(idx_expr: str):
        idx_expr = (idx_expr or '').strip()
        if not idx_expr:
            return None
        # AST-only path: parse the textual index as a Python expression and
        # only accept integer or string Constant nodes.
        try:
            tree = ast.parse(idx_expr, mode='eval')
        except SyntaxError:
            return None
        body = tree.body
        if isinstance(body, ast.Constant):
            if isinstance(body.value, int):
                return str(body.value)
            if isinstance(body.value, str):
                return "'" + body.value.replace("'", "\\'") + "'"
        return None

    for fname, lines in source_files.items():
        fe = _get_ast_frontend(fname)
        for (f, cname), info in class_map.items():
            if f != fname:
                continue
            # Parse __init__ to find self.attr = ClassName(...) mappings
            if fe:
                for item in fe.get_init_assignments_ast(cname):
                    attr_name = item.get("attr")
                    cls_ref = (item.get("class") or "").split('.')[-1]
                    if attr_name and cls_ref:
                        attr_to_class[(fname, attr_name)] = cls_ref

            # Parse forward() reachable helpers to find self.xxx(...) and self.xxx[idx](...) call order
            call_order = []
            if fe and fe.get_method_lines(cname, "forward"):
                reachable = ["forward"] + sorted(fe.get_reachable_helpers(cname, "forward"))
                for method_name in reachable:
                    for call in fe.get_module_calls(cname, method_name):
                        attr_name = call.get("attr")
                        if attr_name and attr_name not in call_order and attr_name != "forward":
                            call_order.append(attr_name)
            module_call_order[cname] = call_order

    return module_call_order, attr_to_class


def _build_data_dependency_edges(source_files, class_map, module_attrs, class_str_list_attrs=None, dynamic_attrs_per_class=None, ast_frontends=None):
    """Parse forward() methods to extract tensor data dependency edges between child modules.

    For each class, analyzes variable assignments from self.xxx() calls and variable
    usage in subsequent self.yyy() calls to determine actual data flow edges.
    Also handles loop iteration patterns (for x in self.attr) and indexed access
    (self.attr[idx](...)).

    Args:
        source_files: {filename: [lines]}
        class_map: from _build_class_map
        module_attrs: {class_name: {attr_name: class_ref}} - known nn.Module attrs

    Returns:
        Tuple (edges_dict, edge_locs_dict, split_info):
        - edges_dict: {class_name: [(from_attr, to_attr), ...]} - data dependency edges per class
        - edge_locs_dict: {class_name: {(from_attr, to_attr): {"file", "from_line", "to_line", "var"}}}
            evidence for each edge: variable name carrying the data, the line that
            *produced* the variable (LHS = self.from_attr(...)) and the line that
            *consumed* it (self.to_attr(..., var, ...)).
        - split_info: {class_name: {original_attr: [split_name_0, split_name_1, ...]}}
            maps original attr names to their split node names for attrs called >1 time
    """
    class_dep_edges = {}
    class_edge_locs = {}
    class_split_info = {}
    if ast_frontends is None:
        ast_frontends = _build_ast_frontends(source_files)

    def _get_ast_frontend(fname):
        return ast_frontends.get(fname)

    if class_str_list_attrs is None:
        class_str_list_attrs = {}

    def _extract_all_names(node: ast.AST) -> list[str]:
        """从 AST node walk 出所有 Name.id（去重，保序，过滤 self/cls）"""
        seen, result = set(), []
        if node is None:
            return result
        for n in ast.walk(node):
            if isinstance(n, ast.Name) and n.id not in seen and n.id not in ("self", "cls"):
                seen.add(n.id)
                result.append(n.id)
        return result

    # -------------------------------------------------------------------
    # dict 数据流追踪（v7 根因修复核心）：
    # - dict 槽位写入：d['k'] = v
    # - dict 槽位读取：x = d['k'] / d.get('k')
    # - 字面量 dict：d = {'k': v, ...}
    # - key 非字面量时：使用该 dict 当前所有槽位 producers 的并集（保守上界，保证不漏边）
    #
    # 说明：这里追踪的是“张量数据流”，不是 dict 这个对象本身。
    # 因此我们不把 dict 作为 var_producers 的普通变量，而是单独维护槽位。
    # -------------------------------------------------------------------

    # 全局（跨 class）共享 dict 槽位：{(fname, dict_name): {key: (producers, from_line, var_evidence)}}
    shared_dict_slots = defaultdict(dict)

    def _as_str_lit(s: str):
        """AST-only string-literal extraction.  Returns the literal value
        for ``"abc"`` / ``'abc'``, otherwise ``None``."""
        s = (s or '').strip()
        if not s:
            return None
        try:
            tree = ast.parse(s, mode='eval')
        except SyntaxError:
            return None
        body = tree.body
        if isinstance(body, ast.Constant) and isinstance(body.value, str):
            return body.value
        return None

    def _dict_union_slots(dslots: dict):
        """将 dict 的所有槽位 producers 合并（key 非字面量时使用）。"""
        merged = set()
        best_line = -1
        best_var = None
        for _k, (ps, pl, pv) in dslots.items():
            if ps:
                merged.update(ps)
                if pl is not None and pl > best_line:
                    best_line = pl
                    best_var = pv
        return merged, best_line if best_line >= 0 else None, best_var

    def _collect_dict_reads(expr: str, dict_slots: dict, fname: str):
        """从表达式中收集 dict 读取产生的 producers。

        AST-only walk.  Recognises:
          * ``d['k']`` / ``d[var]``  → ast.Subscript with ast.Name owner
          * ``d.get('k', ...)``      → ast.Call with ast.Attribute func of '.get'

        返回：
          producers_set, evidence_map(producer_attr -> (var_evidence, from_line))
        """
        prod = set()
        ev_map = {}
        if not expr:
            return prod, ev_map
        # Parse as expression; tolerate top-level commas via tuple wrap.
        try:
            tree = ast.parse(expr, mode='eval')
        except SyntaxError:
            try:
                tree = ast.parse("(" + expr + ")", mode='eval')
            except SyntaxError:
                return prod, ev_map

        def _idx_text(idx_node):
            if isinstance(idx_node, ast.Index):  # py<3.9
                idx_node = idx_node.value
            try:
                return ast.unparse(idx_node).strip()
            except Exception:
                return ""

        for sub in ast.walk(tree):
            # d[k] / d['k']
            if isinstance(sub, ast.Subscript) and isinstance(sub.value, ast.Name):
                dname = sub.value.id
                idx_node = sub.slice
                if isinstance(idx_node, ast.Index):
                    idx_node = idx_node.value
                key_expr = _idx_text(idx_node)
                dslots = dict_slots.get(dname) or shared_dict_slots.get((fname, dname))
                if not dslots:
                    continue
                if isinstance(idx_node, ast.Constant) and isinstance(idx_node.value, str):
                    key_lit = idx_node.value
                else:
                    key_lit = _as_str_lit(key_expr)
                if key_lit is not None and key_lit in dslots:
                    ps, pl, pv = dslots[key_lit]
                    prod.update(ps)
                    for p in ps:
                        prev = ev_map.get(p)
                        if prev is None or (pl is not None and pl > prev[1]):
                            ev_map[p] = (f"{dname}['{key_lit}']", pl)
                else:
                    ps, pl, pv = _dict_union_slots(dslots)
                    prod.update(ps)
                    for p in ps:
                        prev = ev_map.get(p)
                        if prev is None or (pl is not None and pl > prev[1]):
                            ev_map[p] = (f"{dname}[{key_expr}]", pl)
            # d.get('k', ...)
            elif (
                isinstance(sub, ast.Call)
                and isinstance(sub.func, ast.Attribute)
                and sub.func.attr == "get"
                and isinstance(sub.func.value, ast.Name)
                and sub.args
            ):
                dname = sub.func.value.id
                key_node = sub.args[0]
                key_expr = _idx_text(key_node)
                dslots = dict_slots.get(dname) or shared_dict_slots.get((fname, dname))
                if not dslots:
                    continue
                if isinstance(key_node, ast.Constant) and isinstance(key_node.value, str):
                    key_lit = key_node.value
                else:
                    key_lit = _as_str_lit(key_expr)
                if key_lit is not None and key_lit in dslots:
                    ps, pl, pv = dslots[key_lit]
                    prod.update(ps)
                    for p in ps:
                        prev = ev_map.get(p)
                        if prev is None or (pl is not None and pl > prev[1]):
                            ev_map[p] = (f"{dname}.get('{key_lit}')", pl)
                else:
                    ps, pl, pv = _dict_union_slots(dslots)
                    prod.update(ps)
                    for p in ps:
                        prev = ev_map.get(p)
                        if prev is None or (pl is not None and pl > prev[1]):
                            ev_map[p] = (f"{dname}.get({key_expr})", pl)

        return prod, ev_map

    def _collect_expr_producers(expr: str, var_producers: dict, dict_slots: dict, fname: str):
        """统一收集一个表达式的 producers（普通变量 + dict reads）。

        AST-only walk over ``ast.Name`` nodes for variable references,
        plus delegation to ``_collect_dict_reads`` for dict-based flow.
        """
        producers = set()
        best_line = None
        best_var = None

        # 普通变量引用：parse expression and walk Name nodes.
        if expr and var_producers:
            tree = None
            try:
                tree = ast.parse(expr, mode='eval')
            except SyntaxError:
                try:
                    tree = ast.parse("(" + expr + ")", mode='eval')
                except SyntaxError:
                    tree = None
            if tree is not None:
                seen_vars = set()
                for sub in ast.walk(tree):
                    if isinstance(sub, ast.Name):
                        var = sub.id
                        if var in seen_vars:
                            continue
                        seen_vars.add(var)
                        info = var_producers.get(var)
                        if not info:
                            continue
                        ps, ploc = info
                        if ps:
                            producers.update(ps)
                            if ploc is not None and (best_line is None or ploc > best_line):
                                best_line = ploc
                                best_var = var

        # dict 读取
        dps, dev = _collect_dict_reads(expr, dict_slots, fname)
        if dps:
            producers.update(dps)
            # 证据取最新的一条
            for _p, (vname, pl) in dev.items():
                if pl is not None and (best_line is None or pl > best_line):
                    best_line = pl
                    best_var = vname

        return producers, best_line, best_var

    def _parse_dict_literal_items(content: str):
        """解析 {k: v, ...} 的顶层 key/value 列表（只做静态近似，足以用于 DAG 依赖）。"""
        items = []
        s = content.strip()
        if not s:
            return items

        parts = []
        buf = []
        depth = 0
        in_str = None
        esc = False
        for ch in s:
            if in_str:
                buf.append(ch)
                if esc:
                    esc = False
                elif ch == '\\':
                    esc = True
                elif ch == in_str:
                    in_str = None
                continue
            if ch in ("'", '"'):
                in_str = ch
                buf.append(ch)
                continue
            if ch in "([{":
                depth += 1
            elif ch in ")]}":
                depth = max(0, depth - 1)
            if ch == ',' and depth == 0:
                part = ''.join(buf).strip()
                if part:
                    parts.append(part)
                buf = []
            else:
                buf.append(ch)
        last = ''.join(buf).strip()
        if last:
            parts.append(last)

        for p in parts:
            # key : value （只按顶层第一个冒号切分）
            depth = 0
            in_str = None
            esc = False
            split_i = None
            for i, ch in enumerate(p):
                if in_str:
                    if esc:
                        esc = False
                    elif ch == '\\':
                        esc = True
                    elif ch == in_str:
                        in_str = None
                    continue
                if ch in ("'", '"'):
                    in_str = ch
                    continue
                if ch in "([{":
                    depth += 1
                elif ch in ")]}":
                    depth = max(0, depth - 1)
                if ch == ':' and depth == 0:
                    split_i = i
                    break
            if split_i is None:
                continue
            k = p[:split_i].strip()
            v = p[split_i+1:].strip()
            items.append((k, v))
        return items

    for fname, lines in source_files.items():
        for (f, cname), info in class_map.items():
            if f != fname:
                continue
            if cname not in module_attrs:
                continue
            attrs = module_attrs[cname]
            fwd_range = info["methods"].get("forward")
            if not fwd_range:
                continue

            # var_producers: {var_name: (set_of_attr_names_that_produce_it, lineno_of_last_production)}
            var_producers = {}
            # Iter15 tensor-alias: tracks tensor aliases registered via
            # ``setattr(self, <name_expr>, <tensor_var>)`` where ``tensor_var``
            # is an existing tracked variable (not a Module constructor call).
            # The alias name (after f-string expansion) acts like a virtual
            # variable: subsequent ``getattr(self, <name_expr>)`` should resolve
            # to the same producer set as the tensor that was stored.
            #   {alias_name_str: (producer_set, registration_lineno)}
            tensor_alias_producers = {}
            # var_lineage: {var_name: [{var, file, line, text}, ...]}
            # Iter11 varlineage: assignment chain for each tracked variable in
            # forward(). Each step records the variable name, the line of the
            # logical statement, and the raw line text. When a variable is
            # produced from a previous variable (e.g. `b = a + bias`), the
            # lineage of `a` is inherited and the new step appended. This is
            # later attached to dep edges so the front-end can render the
            # complete history (a = ..., b = a + ..., c = self.X(b)).
            var_lineage = {}

            def _line_text(ln):
                if not ln or ln <= 0 or ln > len(lines):
                    return ""
                return lines[ln - 1].rstrip("\n")

            def _record_lineage(target_var, parent_vars, lineno):
                """Append a step (target_var produced at lineno) to the chain.
                Inherit chains from parent_vars (deduped, preserving order).
                """
                if not target_var or target_var == '_':
                    return
                base = []
                seen = set()
                if parent_vars:
                    for pv in parent_vars:
                        if not pv or pv == target_var:
                            continue
                        for step in var_lineage.get(pv, []) or []:
                            key = (step.get("var"), step.get("line"))
                            if key in seen:
                                continue
                            seen.add(key)
                            base.append(step)
                step_text = _line_text(lineno)
                key = (target_var, lineno)
                if key not in seen and step_text:
                    _carriers = []
                    if isinstance(target_var, str) and target_var.isidentifier():
                        _carriers = [target_var]
                    base.append({"var": target_var, "file": fname,
                                 "line": lineno, "text": step_text,
                                 "carriers": _carriers})
                if base:
                    var_lineage[target_var] = base

            def _vars_in(expr_text):
                """Return tracked var names that textually appear in expr_text
                (AST walk, ``ast.Name`` only), preserving expression order so
                the most relevant carrier (last) is at the end."""
                if not expr_text:
                    return []
                hits = []
                seen = set()
                # Search both in var_lineage (rich chain) and var_producers
                # (module-rooted tracking) so we don't miss variables that have
                # producers but no recorded lineage step yet.
                _candidates = set(var_lineage.keys()) | set(var_producers.keys())
                if not _candidates:
                    return []
                # AST-only path: parse the expression and walk Name nodes in
                # source order.  ``ast.parse`` requires a complete expression;
                # wrap as ``(expr,)`` to tolerate top-level comma-separated
                # arg-strings without changing Name semantics.
                try:
                    tree = ast.parse(expr_text, mode='eval')
                except SyntaxError:
                    try:
                        tree = ast.parse("(" + expr_text + ")", mode='eval')
                    except SyntaxError:
                        return []
                for sub in ast.walk(tree):
                    if isinstance(sub, ast.Name):
                        v = sub.id
                        if not v or v == '_' or v not in _candidates:
                            continue
                        if v not in seen:
                            seen.add(v)
                            hits.append(v)
                return hits

            def _collect_ast_expr_consumers(expr_node, current_attr):
                """AST-first consumer extraction for a call argument expression.

                Returns:
                  (producer_set, evidence_map)
                where evidence_map is producer_attr -> (var_evidence, from_line).
                """
                producers = set()
                ev_map = {}
                if expr_node is None:
                    return producers, ev_map

                def _merge(_ps, _ev):
                    if not _ps:
                        return
                    producers.update(_ps)
                    for _p, _meta in (_ev or {}).items():
                        if _p == current_attr:
                            continue
                        _prev = ev_map.get(_p)
                        _line = _meta[1] if _meta else None
                        if _prev is None or (_line is not None and (_prev[1] is None or _line > _prev[1])):
                            ev_map[_p] = _meta

                for sub in ast.walk(expr_node):
                    if isinstance(sub, ast.Name):
                        _v = sub.id
                        if _v in var_producers:
                            _ps, _pl = var_producers[_v]
                            if _ps:
                                _merge(_ps, {p: (_v, _pl) for p in _ps})
                    elif isinstance(sub, ast.Subscript) and isinstance(sub.value, ast.Name):
                        _dname = sub.value.id
                        _idx_node = sub.slice
                        if isinstance(_idx_node, ast.Index):
                            _idx_node = _idx_node.value
                        _key_expr = _ast_fe_dd._node_to_text(_idx_node).strip() if _ast_fe_dd else ""
                        dslots = dict_slots.get(_dname) or shared_dict_slots.get((fname, _dname))
                        if dslots:
                            _key_lit = _as_str_lit(_key_expr)
                            if _key_lit is not None and _key_lit in dslots:
                                _ps, _pl, _pv = dslots[_key_lit]
                                _merge(_ps, {p: (f"{_dname}['{_key_lit}']", _pl) for p in _ps})
                            else:
                                _ps, _pl, _pv = _dict_union_slots(dslots)
                                _merge(_ps, {p: (f"{_dname}[{_key_expr}]", _pl) for p in _ps})
                    elif isinstance(sub, ast.Call):
                        _nested_attr = _ast_fe_dd._extract_called_self_attr(sub.func) if _ast_fe_dd else None
                        if _nested_attr and _nested_attr in known_attrs and _nested_attr != current_attr:
                            _merge({_nested_attr}, {_nested_attr: ("(nested call)", phys_lineno)})
                            continue
                        if isinstance(sub.func, ast.Attribute):
                            _get_attr = sub.func
                            if isinstance(_get_attr.value, ast.Name) and _get_attr.attr == "get":
                                _dname = _get_attr.value.id
                                _key_expr = _ast_fe_dd._node_to_text(sub.args[0]).strip() if (_ast_fe_dd and sub.args) else ""
                                dslots = dict_slots.get(_dname) or shared_dict_slots.get((fname, _dname))
                                if dslots:
                                    _key_lit = _as_str_lit(_key_expr)
                                    if _key_lit is not None and _key_lit in dslots:
                                        _ps, _pl, _pv = dslots[_key_lit]
                                        _merge(_ps, {p: (f"{_dname}.get('{_key_lit}')", _pl) for p in _ps})
                                    else:
                                        _ps, _pl, _pv = _dict_union_slots(dslots)
                                        _merge(_ps, {p: (f"{_dname}.get({_key_expr})", _pl) for p in _ps})
                        if isinstance(sub.func, ast.Call) and isinstance(sub.func.func, ast.Name) and sub.func.func.id == "getattr":
                            _gargs = sub.func.args
                            if len(_gargs) >= 2 and isinstance(_gargs[0], ast.Name) and _gargs[0].id == "self":
                                _name_node = _gargs[1]
                                if isinstance(_name_node, ast.Constant) and isinstance(_name_node.value, str):
                                    _nested_attr = _name_node.value
                                    if _nested_attr in known_attrs and _nested_attr != current_attr:
                                        _merge({_nested_attr}, {_nested_attr: ("(nested getattr call)", phys_lineno)})
                                else:
                                    _name_arg = _ast_fe_dd._node_to_text(_name_node).strip() if _ast_fe_dd else ""
                                    if _name_arg in _loop_var_to_real_attrs:
                                        for _nested_attr in _loop_var_to_real_attrs[_name_arg]:
                                            if _nested_attr != current_attr:
                                                _merge({_nested_attr}, {_nested_attr: ("(nested getattr call)", phys_lineno)})
                                    elif dynamic_setattr_attrs:
                                        for _nested_attr in dynamic_setattr_attrs:
                                            if _nested_attr != current_attr:
                                                _merge({_nested_attr}, {_nested_attr: ("(nested getattr call)", phys_lineno)})

                return producers, ev_map
            # dict_slots: {dict_name: {key: (producers_set, from_line, var_evidence)}}
            # 仅用于追踪“经过 dict 槽位传递”的张量数据流。
            dict_slots = defaultdict(dict)
            # edges: set of (from_attr, to_attr) tuples
            edges = set()
            # edge -> evidence dict
            edge_locs = {}
            # Iter11: when `for name in self.xxx_names:` iterates over a known literal
            # string list, the loop var carries an *attribute name* (string), not a
            # tensor. So `x = getattr(self, name)` resolves to the union of real
            # per-element attrs (e.g. {relation_layer, gift_layer, stay_layer, pos_layer, other_layer}).
            # {loop_var_name: [real_attr_str, ...]}
            _loop_var_to_real_attrs = {}
            # Iter15 tensor-alias: parallel tracker that stores the *raw* string
            # items the loop variable iterates over (before filtering to known
            # Module attrs). Used to expand f-string templates like
            # ``f"{group_name}_out"`` into concrete alias names. Distinct from
            # ``_loop_var_to_real_attrs`` because tensor alias names typically
            # are NOT registered Module attrs (e.g. ``gift_layer_out``).
            _loop_var_to_str_items_local = {}
            _str_list_attrs_for_class = (class_str_list_attrs.get(cname) or {})
            # Track which attrs appear in forward
            known_attrs = set(attrs.keys())
            # Iter11: dynamic attrs (synthesised when setattr name is non-literal) are
            # tracked by the upstream scanner via `dynamic_attrs_per_class[cname]`.
            # The DAG node name is now just the class name (e.g. `GroupTower`), with
            # no `__setattr_` prefix.
            dynamic_setattr_attrs = sorted(
                [a for a in (dynamic_attrs_per_class.get(cname) or set()) if a in known_attrs]
            )

            # container -> [elem_attr,...] (ModuleList/ModuleDict 展开后的子节点)
            # 约定：elem_attr 形如 "layers[0]" / "loss_fns['ctr0']" / "blocks[*]"
            container_to_elems = defaultdict(list)

            def _split_indexed_attr(attr_name):
                """Pure-string split of ``container[idx]`` style attr names.
                Returns (container_name, idx_text) or ``None`` when the attr
                does not look like an indexed reference.  No regex is used —
                only matching brackets at the very end of the string and an
                identifier head for the container."""
                if not attr_name or not attr_name.endswith(']'):
                    return None
                lb = attr_name.find('[')
                if lb <= 0 or lb >= len(attr_name) - 1:
                    return None
                head = attr_name[:lb]
                if not (head.isidentifier()):
                    return None
                inner = attr_name[lb + 1:-1]
                return head, inner

            for a in known_attrs:
                split = _split_indexed_attr(a)
                if not split:
                    continue
                container_to_elems[split[0]].append(a)
            # 稳定排序：优先按 [数字] 排序，否则按字典序
            for cont in list(container_to_elems.keys()):
                elems = container_to_elems[cont]
                num_elems = []
                other = []
                for ea in elems:
                    sp = _split_indexed_attr(ea)
                    if sp and sp[0] == cont and sp[1].isdigit():
                        num_elems.append((int(sp[1]), ea))
                    else:
                        other.append(ea)
                if num_elems and not other:
                    container_to_elems[cont] = [ea for _, ea in sorted(num_elems)]
                else:
                    container_to_elems[cont] = sorted(elems)

            def _norm_index_expr(idx_expr: str):
                idx_expr = (idx_expr or '').strip()
                if not idx_expr:
                    return None
                # AST-only path: only int / str Constant nodes are accepted.
                try:
                    tree = ast.parse(idx_expr, mode='eval')
                except SyntaxError:
                    return None
                body = tree.body
                if isinstance(body, ast.Constant):
                    if isinstance(body.value, int):
                        return str(body.value)
                    if isinstance(body.value, str):
                        return "'" + body.value.replace("'", "\\'") + "'"
                return None

            def _resolve_indexed_attr(container: str, idx_expr: str):
                norm = _norm_index_expr(idx_expr)
                if norm is not None:
                    cand = f"{container}[{norm}]"
                    if cand in known_attrs:
                        return [cand]
                # 动态 index：保守连接到 container 的所有展开子元素；若无展开则退回 container 本身
                if container in container_to_elems and container_to_elems[container]:
                    return list(container_to_elems[container])
                return [container]

            # Collect lines from forward() AND helper methods called from forward()
            # that themselves invoke known attrs (nn.Module submodules).
            # This handles patterns like MultiHeadAttention.forward() calling
            # self.forward_dense() which contains the actual module calls.
            _all_methods = info["methods"]
            _helper_method_ranges = []

            # Iter16: transitive helper-method discovery.
            # forward() -> helper1 -> helper2 -> ... where any node in the chain
            # invokes known attrs is included. This is required to support
            # dispatch patterns like
            #     def _compute_task_tower_outputs(self):
            #         if cond: return self._compute_task_tower_outputs_grouped()
            #         return self._compute_task_tower_outputs_old()
            # where the leaf method is what actually consumes setattr-registered
            # modules. Without transitivity those consumers are invisible and
            # Rule2 fires "缺少输出连线".
            _fwd_text = "\n".join(lines[fwd_range[0] - 1: min(fwd_range[1], len(lines))])
            _method_text_cache = {"forward": _fwd_text}
            for _mname, _mrange in _all_methods.items():
                if _mname in ("forward", "__init__"):
                    continue
                _method_text_cache[_mname] = "\n".join(
                    lines[_mrange[0] - 1: min(_mrange[1], len(lines))])

            # AST cache for helper-method discovery: parse each method body once
            # and reuse for both attr-call detection and self-method-call lookup.
            _method_ast_cache = {}

            def _parse_method_ast(_text: str):
                """Parse method body text via ast.parse; returns the function-body
                AST list or None on failure.  Uses leading-whitespace stripping
                so that a top-level ``def`` parses correctly."""
                if not _text:
                    return None
                if _text in _method_ast_cache:
                    return _method_ast_cache[_text]
                stripped = textwrap.dedent(_text)
                try:
                    tree = ast.parse(stripped)
                except SyntaxError:
                    _method_ast_cache[_text] = None
                    return None
                # Find the FunctionDef body
                body = None
                for n in tree.body:
                    if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)):
                        body = n.body
                        break
                if body is None:
                    body = tree.body
                _method_ast_cache[_text] = body
                return body

            def _method_has_attr_call(_mtext: str) -> bool:
                """AST-only: does this method body contain a ``self.<attr>(...)`` or
                ``self.<attr>[...]`` reference for any known attr?  Walks both
                ``ast.Call`` (with self-attr func) and ``ast.Subscript`` whose
                owner is ``self.<attr>``."""
                body = _parse_method_ast(_mtext)
                if body is None:
                    return False
                for top in body:
                    for sub in ast.walk(top):
                        # self.<attr>(...)  : Call → Attribute(value=Name('self'))
                        if isinstance(sub, ast.Call):
                            f = sub.func
                            if (
                                isinstance(f, ast.Attribute)
                                and isinstance(f.value, ast.Name)
                                and f.value.id == "self"
                                and f.attr in attrs
                            ):
                                return True
                        # self.<attr>[...]  : Subscript whose value is Attribute(self, attr)
                        if isinstance(sub, ast.Subscript):
                            owner = sub.value
                            if (
                                isinstance(owner, ast.Attribute)
                                and isinstance(owner.value, ast.Name)
                                and owner.value.id == "self"
                                and owner.attr in attrs
                            ):
                                return True
                return False

            def _method_calls_self_method(_text: str, _target_name: str) -> bool:
                """AST-only: does this method body contain a ``self.<_target_name>(...)`` call?"""
                body = _parse_method_ast(_text)
                if body is None:
                    return False
                for top in body:
                    for sub in ast.walk(top):
                        if isinstance(sub, ast.Call):
                            f = sub.func
                            if (
                                isinstance(f, ast.Attribute)
                                and isinstance(f.value, ast.Name)
                                and f.value.id == "self"
                                and f.attr == _target_name
                            ):
                                return True
                return False

            _reachable_helpers: "set[str]" = set()
            _seen_methods: "set[str]" = {"forward"}
            _frontier = ["forward"]
            while _frontier:
                _next = []
                for _src in _frontier:
                    _src_text = _method_text_cache.get(_src, "")
                    if not _src_text:
                        continue
                    for _mname in _all_methods:
                        if _mname in _seen_methods or _mname in ("__init__",):
                            continue
                        if not _method_calls_self_method(_src_text, _mname):
                            continue
                        _seen_methods.add(_mname)
                        _next.append(_mname)
                        # Include the helper if it (transitively) contains attr calls.
                        if _method_has_attr_call(_method_text_cache.get(_mname, "")):
                            _reachable_helpers.add(_mname)
                _frontier = _next

            # Second pass: also include intermediate dispatchers that don't
            # directly call attrs but route into a helper that does. Without
            # this, var/loop-var producers set inside the dispatcher are lost.
            # Conservative: include ANY transitively reached helper.
            for _mname in _seen_methods - {"forward"}:
                _reachable_helpers.add(_mname)

            for _mname in sorted(_reachable_helpers):
                _mrange = _all_methods.get(_mname)
                if _mrange is not None:
                    _helper_method_ranges.append(_mrange)

            fwd_lines = lines[fwd_range[0] - 1: min(fwd_range[1], len(lines))]
            logical_lines = _join_logical_lines(fwd_lines, fwd_range[0])
            # Append helper method lines after forward lines
            for _hrange in _helper_method_ranges:
                _hlines = lines[_hrange[0] - 1: min(_hrange[1], len(lines))]
                logical_lines.extend(_join_logical_lines(_hlines, _hrange[0]))

            _ast_stmt_info_by_line = {}
            _ast_var_env = {}
            _ast_alias_env = {}
            _ast_dict_slot_env = {}
            _ast_calls_by_line = {}
            _ast_fe_dd = _get_ast_frontend(fname)
            if _ast_fe_dd:
                _ast_method_names = ["forward"] + sorted(_reachable_helpers)
                for _mn in _ast_method_names:
                    for _stmt_info in (_ast_fe_dd._get_stmt_infos(cname, _mn) or []):
                        _ln = _stmt_info.get("line")
                        if _ln is not None and _ln not in _ast_stmt_info_by_line:
                            _ast_stmt_info_by_line[_ln] = _stmt_info
                    for _call_info in (_ast_fe_dd.get_module_calls(cname, _mn) or []):
                        _ln = _call_info.get("line")
                        if _ln is None:
                            continue
                        _ast_calls_by_line.setdefault(_ln, []).append(_call_info)
                    _mn_var_env = _ast_fe_dd.build_var_env(cname, _mn)
                    for _var_name, _chain in (_mn_var_env or {}).items():
                        if not _chain:
                            continue
                        _ast_var_env.setdefault(_var_name, []).extend(list(_chain))
                    _ast_alias_env.update(_ast_fe_dd.build_alias_env(cname, _mn))
                    _ast_dict_slot_env.update(_ast_fe_dd.build_dict_slot_env(cname, _mn))
                for _ln, _calls in _ast_calls_by_line.items():
                    _calls.sort(key=lambda item: (item.get("col") is None, item.get("col") or 0, item.get("attr") or item.get("name") or ""))

            # --- Call-site splitting pre-pass ---
            # Strategy: only split attrs that would create cycles if unsplit.
            # 1. Do a quick pre-scan to count calls per attr within each method
            # 2. Only mark attrs for splitting if they are called >1 in ANY single method
            # 3. After edge building, validate no cycles; if cycles remain, warn.
            #
            # Additionally, to avoid splitting attrs in mutually exclusive branches
            # (if/else), we only split attrs that actually cause cycles. We detect
            # this by checking if an attr is both a source AND a target in the edge set.
            _method_ranges_for_counting = [(fwd_range[0], fwd_range[1])]
            for _hr in _helper_method_ranges:
                _method_ranges_for_counting.append((_hr[0], _hr[1]))

            _attr_max_calls_in_single_method = defaultdict(int)
            for _mr_start, _mr_end in _method_ranges_for_counting:
                _method_lines = lines[_mr_start - 1: min(_mr_end, len(lines))]
                _method_text = "\n".join(_method_lines)
                _method_call_count = defaultdict(int)

                # AST-only: walk every Call/Subscript/For node in the method body.
                _body = _parse_method_ast(_method_text)
                if not _body:
                    # If method body fails to parse, skip; do NOT fall back to regex.
                    for _ca, _cnt in _method_call_count.items():
                        if _cnt > _attr_max_calls_in_single_method[_ca]:
                            _attr_max_calls_in_single_method[_ca] = _cnt
                    continue

                def _is_self_attr(node, attr_name=None):
                    if not (isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name) and node.value.id == "self"):
                        return False
                    return attr_name is None or node.attr == attr_name

                # ``for`` loops over self.<attr> introduce a loop variable that
                # binds to a Module-instance taken from self.<attr>.  We track
                # only existence of these loop vars (the call-counting prepass
                # below does not need to know loop var names because it counts
                # syntactic call sites, not run-time invocations).
                for top in _body:
                    for sub in ast.walk(top):
                        if isinstance(sub, ast.Call):
                            _func = sub.func
                            # 1) self.<attr>(...)
                            if _is_self_attr(_func) and _func.attr in known_attrs:
                                _method_call_count[_func.attr] += 1
                                continue
                            # 2) getattr(self, '<lit>')(...)
                            if (
                                isinstance(_func, ast.Call)
                                and isinstance(_func.func, ast.Name)
                                and _func.func.id == "getattr"
                                and len(_func.args) >= 2
                                and isinstance(_func.args[0], ast.Name)
                                and _func.args[0].id == "self"
                                and isinstance(_func.args[1], ast.Constant)
                                and isinstance(_func.args[1].value, str)
                                and _func.args[1].value in known_attrs
                            ):
                                _method_call_count[_func.args[1].value] += 1
                                continue
                            # 3) self.<container>[<idx>](...)
                            if (
                                isinstance(_func, ast.Subscript)
                                and isinstance(_func.value, ast.Attribute)
                                and isinstance(_func.value.value, ast.Name)
                                and _func.value.value.id == "self"
                            ):
                                _cont = _func.value.attr
                                if _cont not in known_attrs:
                                    continue
                                idx_node = _func.slice
                                if isinstance(idx_node, ast.Index):  # py<3.9
                                    idx_node = idx_node.value
                                # Literal index?
                                _norm = None
                                if isinstance(idx_node, ast.Constant):
                                    if isinstance(idx_node.value, int):
                                        _norm = str(idx_node.value)
                                    elif isinstance(idx_node.value, str):
                                        _norm = "'" + idx_node.value.replace("'", "\\'") + "'"
                                if _norm is not None:
                                    _cand = f"{_cont}[{_norm}]"
                                    if _cand in known_attrs:
                                        _method_call_count[_cand] += 1
                                    else:
                                        _wild = f"{_cont}[*]"
                                        if _wild in known_attrs:
                                            _method_call_count[_wild] += 1
                                        else:
                                            _method_call_count[_cont] += 1
                                else:
                                    _wild = f"{_cont}[*]"
                                    if _wild in known_attrs:
                                        _method_call_count[_wild] += 1
                                    elif _cont in container_to_elems and container_to_elems[_cont]:
                                        _method_call_count[_cont + "[*]"] += 1
                                    else:
                                        _method_call_count[_cont] += 1
                # Track the max across all methods
                for _ca, _cnt in _method_call_count.items():
                    if _cnt > _attr_max_calls_in_single_method[_ca]:
                        _attr_max_calls_in_single_method[_ca] = _cnt

            # Only attrs called >1 in a single method are CANDIDATES for splitting.
            # We only split attrs that actually participate in cycles (detected after
            # a trial edge-build without splitting). This avoids unnecessary splits
            # for attrs in mutually exclusive if/else branches.
            _multi_call_candidates = {a for a, cnt in _attr_max_calls_in_single_method.items() if cnt > 1}

            # split_node_map populated post-hoc if cycles are detected
            split_node_map = {}

            for phys_lineno, line in logical_lines:
                # v7 修复：剥离行内注释（避免注释中的非平衡括号/字符干扰解析）
                line = _strip_inline_comment(line).strip()
                stripped = line
                if not stripped or stripped.startswith('#'):
                    continue

                _ast_stmt = _ast_stmt_info_by_line.get(phys_lineno)

                def _rhs_last_called_attr(expr: str, expr_node=None):
                    """近似判断 expr 的“输出 producer”：优先取 AST 顶层调用 attr。

                    仅用于 dict 槽位赋值/字面量场景，避免把“输入 producer”误当成“输出 producer”。
                    """
                    def _resolve_call_attrs(_call_node):
                        if not isinstance(_call_node, ast.Call):
                            return []
                        _func = _call_node.func
                        _direct_attr = _ast_fe_dd._extract_called_self_attr(_func) if _ast_fe_dd else None
                        if _direct_attr:
                            if _direct_attr in known_attrs:
                                return [_direct_attr]
                            if "[" in _direct_attr:
                                _base, _idx = _direct_attr.split("[", 1)
                                _idx = _idx.rstrip("]")
                                return [a for a in _resolve_indexed_attr(_base, _idx) if a in known_attrs]
                        if isinstance(_func, ast.Call) and isinstance(_func.func, ast.Name) and _func.func.id == "getattr":
                            if len(_func.args) >= 2 and isinstance(_func.args[0], ast.Name) and _func.args[0].id == "self":
                                _name_node = _func.args[1]
                                if isinstance(_name_node, ast.Constant) and isinstance(_name_node.value, str):
                                    return [_name_node.value] if _name_node.value in known_attrs else []
                                _name_arg = _ast_fe_dd._node_to_text(_name_node).strip() if _ast_fe_dd else ""
                                if _name_arg in _loop_var_to_real_attrs:
                                    return [a for a in _loop_var_to_real_attrs[_name_arg] if a in known_attrs]
                                if dynamic_setattr_attrs:
                                    return [a for a in dynamic_setattr_attrs if a in known_attrs]
                        if isinstance(_func, ast.Name):
                            _name = _func.id
                            if _name in var_producers:
                                _ps, _pl = var_producers[_name]
                                return [a for a in _ps if a in known_attrs]
                        return []

                    _ast_attrs = []
                    if expr_node is None and expr:
                        try:
                            _parsed = ast.parse(expr, mode="eval")
                            expr_node = _parsed.body
                        except Exception:
                            expr_node = None
                    if isinstance(expr_node, ast.Call):
                        _ast_attrs = _resolve_call_attrs(expr_node)
                        if _ast_attrs:
                            if len(_ast_attrs) > 1:
                                _tail = None
                                for _cont, _elems in container_to_elems.items():
                                    for _ea in reversed(_elems):
                                        if _ea in _ast_attrs:
                                            _tail = _ea
                                            break
                                    if _tail:
                                        break
                                return _tail or _ast_attrs[-1]
                            return _ast_attrs[-1]

                    # AST is the authoritative path for ``_rhs_last_called_attr``.
                    # If the expression failed to parse or the AST root is not a
                    # Call, we cannot reliably identify the producer; return None
                    # rather than fall back to regex (per AST-only directive).
                    return None

                # ------------------------------
                # dict 数据流：初始化 / 字面量 / 槽位写入  (AST-only)
                # ------------------------------
                # Parse the joined logical line as a standalone Python module.
                # Failure → skip the dict-flow scan for this line (the regex
                # path used to gracefully no-op too).
                _line_mod = None
                try:
                    _line_mod = ast.parse(line)
                except SyntaxError:
                    _line_mod = None
                _line_stmt = (_line_mod.body[0]
                              if _line_mod and len(_line_mod.body) == 1
                              else None)

                # Pattern: ``var = {}`` or ``var = dict()``  →  initialise empty dict slot.
                if (
                    isinstance(_line_stmt, ast.Assign)
                    and len(_line_stmt.targets) == 1
                    and isinstance(_line_stmt.targets[0], ast.Name)
                ):
                    _val = _line_stmt.value
                    _is_empty_dict = (
                        (isinstance(_val, ast.Dict) and not _val.keys and not _val.values)
                        or (
                            isinstance(_val, ast.Call)
                            and isinstance(_val.func, ast.Name)
                            and _val.func.id == "dict"
                            and not _val.args
                            and not _val.keywords
                        )
                    )
                    if _is_empty_dict:
                        dname = _line_stmt.targets[0].id
                        dict_slots[dname] = {}
                        shared_dict_slots[(fname, dname)] = dict_slots[dname]
                        continue

                # Pattern: ``var = { 'k': expr, ... }``  →  literal dict assignment.
                if (
                    isinstance(_line_stmt, ast.Assign)
                    and len(_line_stmt.targets) == 1
                    and isinstance(_line_stmt.targets[0], ast.Name)
                    and isinstance(_line_stmt.value, ast.Dict)
                    and _line_stmt.value.keys
                ):
                    dname = _line_stmt.targets[0].id
                    for _k_node, _v_node in zip(_line_stmt.value.keys, _line_stmt.value.values):
                        if _k_node is None or _v_node is None:
                            continue
                        # key must be a literal string
                        if not (isinstance(_k_node, ast.Constant) and isinstance(_k_node.value, str)):
                            continue
                        k_lit = _k_node.value
                        try:
                            v_expr = ast.unparse(_v_node)
                        except Exception:
                            v_expr = ""
                        last_attr = _rhs_last_called_attr(v_expr, expr_node=_v_node)
                        if last_attr is not None:
                            dict_slots[dname][k_lit] = ({last_attr}, phys_lineno, f"self.{last_attr}(...)")
                            shared_dict_slots[(fname, dname)] = dict_slots[dname]
                            continue
                        # module ref: {'k': self.xxx}
                        if (
                            isinstance(_v_node, ast.Attribute)
                            and isinstance(_v_node.value, ast.Name)
                            and _v_node.value.id == "self"
                            and _v_node.attr in known_attrs
                        ):
                            a = _v_node.attr
                            dict_slots[dname][k_lit] = ({a}, phys_lineno, f"self.{a}")
                            shared_dict_slots[(fname, dname)] = dict_slots[dname]
                            continue
                        ps, pl, pv = _collect_expr_producers(v_expr, var_producers, dict_slots, fname)
                        if ps:
                            dict_slots[dname][k_lit] = (ps, pl if pl is not None else phys_lineno, pv or v_expr)
                            shared_dict_slots[(fname, dname)] = dict_slots[dname]
                    # value 里可能包含 self.xxx()，这里不 continue

                _ast_dict_write = _ast_stmt if _ast_stmt and _ast_stmt.get("kind") == "dict_write" else None
                _ast_subscript_assign = None
                if _ast_dict_write:
                    dname = (_ast_dict_write.get("dict_var") or "").strip()
                    slot_key = _ast_dict_write.get("dict_key")
                    slot_key = slot_key if slot_key is not None else '*'
                    rhs = (_ast_dict_write.get("rhs_text") or "").strip()
                    key_expr = repr(slot_key) if slot_key != '*' else '*'
                    _v_rhs_node = _ast_dict_write.get("rhs_node")
                else:
                    # AST-only fallback: ``dname[key_expr] = rhs`` form.
                    if (
                        isinstance(_line_stmt, ast.Assign)
                        and len(_line_stmt.targets) == 1
                        and isinstance(_line_stmt.targets[0], ast.Subscript)
                        and isinstance(_line_stmt.targets[0].value, ast.Name)
                    ):
                        _ast_subscript_assign = _line_stmt
                        dname = _line_stmt.targets[0].value.id
                        _slice = _line_stmt.targets[0].slice
                        try:
                            key_expr = ast.unparse(_slice)
                        except Exception:
                            key_expr = ""
                        try:
                            rhs = ast.unparse(_line_stmt.value)
                        except Exception:
                            rhs = ""
                        if isinstance(_slice, ast.Constant) and isinstance(_slice.value, str):
                            slot_key = _slice.value
                        else:
                            slot_key = '*'
                        _v_rhs_node = _line_stmt.value
                if _ast_dict_write or _ast_subscript_assign is not None:
                    _ast_slot = _ast_dict_slot_env.get((dname, slot_key))
                    if _ast_slot:
                        _prod_attr, _prod_line = _ast_slot
                        if _prod_attr:
                            dict_slots[dname][slot_key] = ({_prod_attr}, _prod_line if _prod_line is not None else phys_lineno, f"self.{_prod_attr}(...)" )
                            shared_dict_slots[(fname, dname)] = dict_slots[dname]
                    else:
                        last_attr = _rhs_last_called_attr(rhs, expr_node=_v_rhs_node)
                        if last_attr is not None:
                            dict_slots[dname][slot_key] = ({last_attr}, phys_lineno, f"self.{last_attr}(...)" )
                            shared_dict_slots[(fname, dname)] = dict_slots[dname]
                        else:
                            # module ref: d['k'] = self.xxx  (AST check)
                            if (
                                isinstance(_v_rhs_node, ast.Attribute)
                                and isinstance(_v_rhs_node.value, ast.Name)
                                and _v_rhs_node.value.id == "self"
                                and _v_rhs_node.attr in known_attrs
                            ):
                                a = _v_rhs_node.attr
                                dict_slots[dname][slot_key] = ({a}, phys_lineno, f"self.{a}")
                                shared_dict_slots[(fname, dname)] = dict_slots[dname]
                            else:
                                ps, pl, pv = _collect_expr_producers(rhs, var_producers, dict_slots, fname)
                                if ps:
                                    dict_slots[dname][slot_key] = (ps, pl if pl is not None else phys_lineno, pv or rhs)
                                    shared_dict_slots[(fname, dname)] = dict_slots[dname]
                    # RHS 可能包含 self.xxx()，这里不 continue

                # Detect loop iteration over ModuleList/ModuleDict values:
                #   for x in self.attr:
                #   for _, x in enumerate(self.attr):
                #   for k, v in self.dict.items():
                #   for v in self.dict.values():
                # AST-only: the joined logical line by itself is not a complete
                # statement (no body), so we wrap it as ``<for-line>\n    pass``
                # before parsing.
                loop_var = None
                loop_attr = None
                loop_items = None
                _for_stmt = None
                _stripped_line = line.lstrip()
                if _stripped_line.startswith("for ") or _stripped_line.startswith("for\t"):
                    try:
                        _for_mod = ast.parse(_stripped_line.rstrip() + "\n    pass\n")
                        if (
                            _for_mod.body
                            and isinstance(_for_mod.body[0], ast.For)
                        ):
                            _for_stmt = _for_mod.body[0]
                    except SyntaxError:
                        _for_stmt = None
                if _for_stmt is not None:
                    _it = _for_stmt.iter
                    _tgt = _for_stmt.target
                    # Determine the loop variable name (Name or Tuple).
                    _loop_var_candidate = None
                    if isinstance(_tgt, ast.Name):
                        _loop_var_candidate = _tgt.id
                    elif isinstance(_tgt, ast.Tuple) and _tgt.elts:
                        # for k, v in ...  -> the SECOND element (when 2-tuple)
                        # for i, x in enumerate(...) -> SECOND element
                        # The original regex picked group(1)=last var of any
                        # leading "<x>," prefix, then last positional name.
                        _last = _tgt.elts[-1]
                        if isinstance(_last, ast.Name):
                            _loop_var_candidate = _last.id
                    if _loop_var_candidate is not None:
                        # Variant 1: ``for x in self.<attr>:`` (also matches
                        # ``for _, x in enumerate(self.<attr>):``)
                        _attr_node = None
                        if (
                            isinstance(_it, ast.Attribute)
                            and isinstance(_it.value, ast.Name)
                            and _it.value.id == "self"
                        ):
                            _attr_node = _it
                        elif (
                            isinstance(_it, ast.Call)
                            and isinstance(_it.func, ast.Name)
                            and _it.func.id == "enumerate"
                            and _it.args
                            and isinstance(_it.args[0], ast.Attribute)
                            and isinstance(_it.args[0].value, ast.Name)
                            and _it.args[0].value.id == "self"
                        ):
                            _attr_node = _it.args[0]
                        if _attr_node is not None:
                            loop_var = _loop_var_candidate
                            loop_attr = _attr_node.attr
                        # Variant 2: ``for k, v in self.<attr>.items():`` /
                        #            ``for v in self.<attr>.values():``
                        if (
                            loop_attr is None
                            and isinstance(_it, ast.Call)
                            and isinstance(_it.func, ast.Attribute)
                            and _it.func.attr in ("items", "values")
                            and isinstance(_it.func.value, ast.Attribute)
                            and isinstance(_it.func.value.value, ast.Name)
                            and _it.func.value.value.id == "self"
                            and not _it.args
                            and not _it.keywords
                        ):
                            loop_var = _loop_var_candidate
                            loop_attr = _it.func.value.attr
                if loop_var and loop_attr and loop_attr in known_attrs:
                    # 若该容器已展开，则 loop_var 视为可能调用任一 elem；并补一条 elem 顺序链
                    if loop_attr in container_to_elems and container_to_elems[loop_attr]:
                        elems = container_to_elems[loop_attr]
                        var_producers[loop_var] = (set(elems), phys_lineno)
                        _record_lineage(loop_var, [], phys_lineno)
                        # 顺序链：e0 -> e1 -> ...（for 循环隐含的顺序数据依赖）
                        if len(elems) > 1:
                            for ii in range(len(elems) - 1):
                                edges.add((elems[ii], elems[ii + 1]))
                                edge_locs.setdefault((elems[ii], elems[ii + 1]), {
                                    "file": fname,
                                    "var": f"{loop_var} (sequential iteration over self.{loop_attr})",
                                    "from_line": phys_lineno,
                                    "to_line": phys_lineno,
                                })
                    else:
                        var_producers[loop_var] = ({loop_attr}, phys_lineno)
                        _record_lineage(loop_var, [], phys_lineno)
                    continue

                # Iter11: detect `for name in self.xxx_names:` where xxx_names is a
                # known literal string list (NOT a Module attr). The loop var carries
                # an attribute *name* (string). Real per-element attrs registered via
                # setattr expansion are looked up later via `getattr(self, name)`.
                if loop_var and loop_attr and loop_attr in _str_list_attrs_for_class:
                    real_items = _str_list_attrs_for_class[loop_attr]
                    # Iter15: always remember the raw string items for f-string
                    # alias name expansion, even if some items aren't Module attrs.
                    _loop_var_to_str_items_local[loop_var] = list(real_items)
                    real_module_attrs = [a for a in real_items if a in known_attrs]
                    if real_module_attrs:
                        _loop_var_to_real_attrs[loop_var] = real_module_attrs
                    continue

                # ------------------------------------------------------------
                # Iter15 tensor-alias: detect `setattr(self, <name_expr>, <tensor_var>)`
                # where the value is an existing tracked variable (NOT a class
                # constructor like ``Foo(...)``). This registers a tensor alias
                # mapping the (resolved) name string to the same producers as
                # the source variable. f-string templates referencing the
                # current loop variable get unrolled into per-iteration alias
                # names. Common pattern (cvr.py:206):
                #     for group_name in self.group_tower_names:
                #         group_tower = getattr(self, group_name)
                #         group_tower_out = group_tower(combine_out)
                #         setattr(self, f"{group_name}_out", group_tower_out)
                # We resolve `f"{group_name}_out"` against the literal string
                # list bound to `group_name` and register tensor aliases
                # `relation_layer_out`, `gift_layer_out`, ... → producer set
                # of `group_tower_out` (which is the union of GroupTower attrs).
                # ------------------------------------------------------------
                m_setattr_tensor = None
                _setattr_call = None
                _setattr_name_node = None
                _setattr_src_var = None
                # AST-only: detect ``setattr(self, <name_expr>, <Name>)``
                # statements where the value is a bare local Name.
                if (
                    _line_stmt is not None
                    and isinstance(_line_stmt, ast.Expr)
                    and isinstance(_line_stmt.value, ast.Call)
                    and isinstance(_line_stmt.value.func, ast.Name)
                    and _line_stmt.value.func.id == "setattr"
                    and len(_line_stmt.value.args) == 3
                    and isinstance(_line_stmt.value.args[0], ast.Name)
                    and _line_stmt.value.args[0].id == "self"
                    and isinstance(_line_stmt.value.args[2], ast.Name)
                ):
                    _setattr_call = _line_stmt.value
                    _setattr_name_node = _setattr_call.args[1]
                    _setattr_src_var = _setattr_call.args[2].id
                    m_setattr_tensor = True
                if m_setattr_tensor:
                    src_var = _setattr_src_var
                    # Only proceed if the value isn't a constructor call
                    # (already handled by the Module setattr scanner).
                    if src_var in var_producers:
                        src_producers, _src_pl = var_producers[src_var]
                        # Resolve the name expression to one or more concrete
                        # alias name strings.
                        # alias_name_to_producers: {alias_name: producer_set}
                        # Default: every alias maps to the full producer set of
                        # the source variable; but f-string templates that are
                        # bound 1:1 to a loop variable iterating over Module
                        # attrs allow per-iteration refinement (each alias maps
                        # to ONLY the corresponding Module).
                        alias_name_to_producers = {}
                        if (
                            isinstance(_setattr_name_node, ast.Constant)
                            and isinstance(_setattr_name_node.value, str)
                        ):
                            alias_name_to_producers[_setattr_name_node.value] = set(src_producers)
                        elif isinstance(_setattr_name_node, ast.JoinedStr):
                            # f-string template: extract referenced variable
                            # names from the FormattedValue parts and the
                            # literal-only template (with each FormattedValue
                            # collapsed to a single placeholder for sub).
                            ref_names = set()
                            for _part in _setattr_name_node.values:
                                if isinstance(_part, ast.FormattedValue):
                                    _vname = None
                                    if isinstance(_part.value, ast.Name):
                                        _vname = _part.value.id
                                    if _vname:
                                        ref_names.add(_vname)
                            resolvable = (
                                len(ref_names) == 1
                                and next(iter(ref_names)) in _loop_var_to_str_items_local
                            )
                            if resolvable:
                                lv = next(iter(ref_names))
                                items = _loop_var_to_str_items_local[lv]
                                # Per-iteration refinement: if every item
                                # is itself a known Module attr AND the
                                # source variable's producers are exactly
                                # the union of those items, bind each
                                # alias to the single corresponding item.
                                items_set = set(items)
                                can_refine = (
                                    all(it in known_attrs for it in items)
                                    and items_set & set(src_producers) == items_set & set(known_attrs)
                                )
                                for it in items:
                                    # Expand the JoinedStr node by substituting
                                    # the loop variable's value for every
                                    # FormattedValue and concatenating the
                                    # literal parts.
                                    _expanded_parts = []
                                    for _part in _setattr_name_node.values:
                                        if isinstance(_part, ast.Constant) and isinstance(_part.value, str):
                                            _expanded_parts.append(_part.value)
                                        elif isinstance(_part, ast.FormattedValue):
                                            # Refining only handles the simple
                                            # ``{loop_var}`` case (with optional
                                            # !s/!r/!a or :spec).  Skip format
                                            # specs and conversions and just
                                            # str() the value.
                                            _expanded_parts.append(str(it))
                                        else:
                                            _expanded_parts.append("")
                                    expanded = "".join(_expanded_parts)
                                    if not expanded:
                                        continue
                                    if can_refine and it in src_producers:
                                        alias_name_to_producers[expanded] = {it}
                                    else:
                                        alias_name_to_producers[expanded] = set(src_producers)
                        else:
                            # Bare Name expression: try to resolve via
                            # _loop_var_to_str_items_local.
                            if isinstance(_setattr_name_node, ast.Name):
                                _name_arg = _setattr_name_node.id
                                if _name_arg in _loop_var_to_str_items_local:
                                    for it in _loop_var_to_str_items_local[_name_arg]:
                                        alias_name_to_producers[it] = set(src_producers)
                        # Register the aliases (union with any existing).
                        # Use the source variable's production line as the
                        # alias production line, so downstream getattr resolves
                        # back to the original Module call site (e.g. line 205
                        # ``group_tower_out = group_tower(combine_out)``)
                        # rather than the setattr line.
                        for an, prods in alias_name_to_producers.items():
                            existing = tensor_alias_producers.get(
                                an, (set(), _src_pl))[0]
                            tensor_alias_producers[an] = (
                                set(existing) | set(prods), _src_pl)
                    # The setattr line itself does not produce or consume a
                    # tensor variable directly; fall through so other passes
                    # (Module setattr) still see it if relevant.

                # Detect module calls (AST-first, regex fallback during migration)
                called_attrs = []
                _ast_line_calls = list(_ast_calls_by_line.get(phys_lineno, [])) if _ast_calls_by_line else []
                if _ast_line_calls:
                    for _call in _ast_line_calls:
                        _kind = _call.get("kind")
                        _start = _call.get("col")
                        _end = _call.get("func_end_col")
                        if _start is None:
                            _start = 0
                        if _end is None:
                            _end = _start
                        if _kind in ("self", "getattr_literal"):
                            _attr = _call.get("attr")
                            if _attr in known_attrs:
                                called_attrs.append((_attr, _start, _end, _call.get("node")))
                        elif _kind == "self_indexed":
                            _cont = _call.get("container_attr")
                            _idx_expr = (_call.get("index_expr") or "").strip()
                            if _cont in known_attrs:
                                for _attr in _resolve_indexed_attr(_cont, _idx_expr):
                                    if _attr in known_attrs:
                                        called_attrs.append((_attr, _start, _end, _call.get("node")))
                        elif _kind == "dict_index":
                            _dname = _call.get("dict_name")
                            _key_expr = (_call.get("index_expr") or "").strip()
                            dslots = dict_slots.get(_dname) or shared_dict_slots.get((fname, _dname))
                            if dslots:
                                _key_lit = _as_str_lit(_key_expr)
                                if _key_lit is not None and _key_lit in dslots:
                                    _ps, _pl, _pv = dslots[_key_lit]
                                else:
                                    _ps, _pl, _pv = _dict_union_slots(dslots)
                                for _attr in _ps:
                                    if _attr in known_attrs:
                                        called_attrs.append((_attr, _start, _end, _call.get("node")))
                        elif _kind == "dict_get":
                            _dname = _call.get("dict_name")
                            _key_expr = (_call.get("index_expr") or "").strip()
                            dslots = dict_slots.get(_dname) or shared_dict_slots.get((fname, _dname))
                            if dslots:
                                _key_lit = _as_str_lit(_key_expr)
                                if _key_lit is not None and _key_lit in dslots:
                                    _ps, _pl, _pv = dslots[_key_lit]
                                else:
                                    _ps, _pl, _pv = _dict_union_slots(dslots)
                                for _attr in _ps:
                                    if _attr in known_attrs:
                                        called_attrs.append((_attr, _start, _end, _call.get("node")))
                        elif _kind == "getattr_dynamic":
                            _name_arg = (_call.get("name_expr") or "").strip()
                            _expr_node = None
                            _call_node = _call.get("node")
                            if (
                                _ast_fe_dd
                                and isinstance(_call_node, ast.Call)
                                and _ast_fe_dd._is_self_getattr_call(_call_node.func)
                            ):
                                _expr_node = _call_node.func.args[1]
                            if _name_arg in _loop_var_to_real_attrs:
                                for _attr in _loop_var_to_real_attrs[_name_arg]:
                                    if _attr in known_attrs:
                                        called_attrs.append((_attr, _start, _end, _call.get("node")))
                            elif _expr_node is not None:
                                _matched_attrs = [
                                    _attr for _attr in known_attrs
                                    if _ast_fe_dd._dynamic_attr_expr_matches(_expr_node, _attr)
                                ]
                                for _attr in sorted(_matched_attrs):
                                    called_attrs.append((_attr, _start, _end, _call.get("node")))
                            elif dynamic_setattr_attrs:
                                for _attr in dynamic_setattr_attrs:
                                    called_attrs.append((_attr, _start, _end, _call.get("node")))
                        elif _kind == "name":
                            _name = _call.get("name")
                            if _name and _name not in known_attrs and _name in var_producers:
                                _producers, _ploc = var_producers[_name]
                                for _attr in _producers:
                                    if _attr in known_attrs:
                                        called_attrs.append((_attr, _start, _end, _call.get("node")))

                if not called_attrs:
                    # AST path is authoritative for module-call detection.
                    # No regex fallback — if the AST scan didn't find any calls
                    # we treat this as a non-call line.  The downstream
                    # ``if not called_attrs:`` block below still handles
                    # variable-flow tracking through non-module ops.
                    pass

                if not called_attrs:
                    # No module calls - track variable flow through non-module ops
                    _ast_stmt = _ast_stmt_info_by_line.get(phys_lineno)
                    _ast_lhs = None
                    _ast_rhs = None
                    if _ast_stmt and _ast_stmt.get("kind") in ("assign", "augassign"):
                        _targets = [t for t in (_ast_stmt.get("targets") or []) if t and t != '_']
                        if len(_targets) == 1:
                            _ast_lhs = _targets[0]
                            _ast_rhs = (_ast_stmt.get("rhs_text") or "").strip()
                    if _ast_lhs is not None:
                        lhs = _ast_lhs
                        rhs = _ast_rhs
                    elif (
                        isinstance(_line_stmt, ast.Assign)
                        and len(_line_stmt.targets) == 1
                        and isinstance(_line_stmt.targets[0], ast.Name)
                    ):
                        # AST-only fallback for ``var = expr`` parsed from the
                        # joined logical line (used when _ast_stmt did not
                        # provide an upstream record).
                        lhs = _line_stmt.targets[0].id
                        try:
                            rhs = ast.unparse(_line_stmt.value)
                        except Exception:
                            rhs = ""
                    else:
                        lhs = None
                        rhs = None
                    # Resolve the rhs AST node for downstream attribute checks.
                    _rhs_ast = None
                    if _ast_stmt:
                        _rhs_ast = _ast_stmt.get("rhs_node")
                    if _rhs_ast is None and isinstance(_line_stmt, (ast.Assign, ast.AugAssign)):
                        _rhs_ast = _line_stmt.value
                    if lhs is not None and rhs is not None:

                        # Special case: ModuleDict/ModuleList module reference
                        #   x = self.container[key]
                        if (
                            isinstance(_rhs_ast, ast.Subscript)
                            and isinstance(_rhs_ast.value, ast.Attribute)
                            and isinstance(_rhs_ast.value.value, ast.Name)
                            and _rhs_ast.value.value.id == "self"
                        ):
                            cont = _rhs_ast.value.attr
                            try:
                                idx_expr = ast.unparse(_rhs_ast.slice)
                            except Exception:
                                idx_expr = ""
                            if cont in known_attrs:
                                resolved = _resolve_indexed_attr(cont, idx_expr)
                                # 这里 resolved 可能是多个 elem（动态 key/index），保守合并
                                var_producers[lhs] = (set([a for a in resolved if a in known_attrs]), phys_lineno)
                                _record_lineage(lhs, _vars_in(rhs), phys_lineno)
                                continue

                        # Special case: module reference assignment
                        #   x = getattr(self, 'attr')
                        #   x = getattr(self, name_var)
                        # This is common with setattr-registered modules:
                        #   group_tower = getattr(self, group_name)
                        #   group_tower_out = group_tower(tensor)
                        if (
                            isinstance(_rhs_ast, ast.Call)
                            and isinstance(_rhs_ast.func, ast.Name)
                            and _rhs_ast.func.id == "getattr"
                            and len(_rhs_ast.args) >= 2
                            and isinstance(_rhs_ast.args[0], ast.Name)
                            and _rhs_ast.args[0].id == "self"
                            and isinstance(_rhs_ast.args[1], ast.Constant)
                            and isinstance(_rhs_ast.args[1].value, str)
                        ):
                            a = _rhs_ast.args[1].value
                            if a in known_attrs:
                                var_producers[lhs] = ({a}, phys_lineno)
                                _record_lineage(lhs, _vars_in(rhs), phys_lineno)
                                continue
                            # Iter15 tensor-alias: literal name resolves to a
                            # previously-registered tensor alias. The lhs
                            # variable inherits the same producer set as the
                            # tensor stored under that alias, and crucially
                            # the *production line* (so Rule1c sees the real
                            # Module-call site, not the setattr/getattr line).
                            if a in tensor_alias_producers:
                                _alias_prods, _alias_pl = tensor_alias_producers[a]
                                if _alias_prods:
                                    var_producers[lhs] = (set(_alias_prods), _alias_pl)
                                    _record_lineage(lhs, _vars_in(rhs), phys_lineno)
                                    continue
                        if (
                            isinstance(_rhs_ast, ast.Call)
                            and isinstance(_rhs_ast.func, ast.Name)
                            and _rhs_ast.func.id == "getattr"
                            and len(_rhs_ast.args) >= 2
                            and isinstance(_rhs_ast.args[0], ast.Name)
                            and _rhs_ast.args[0].id == "self"
                            and not (
                                isinstance(_rhs_ast.args[1], ast.Constant)
                                and isinstance(_rhs_ast.args[1].value, str)
                            )
                        ):
                            try:
                                _name_arg = ast.unparse(_rhs_ast.args[1]).strip()
                            except Exception:
                                _name_arg = ""
                            _ast_rhs_node = _rhs_ast
                            # Iter11: prefer precise loop-var resolution to literal str list
                            if _name_arg in _loop_var_to_real_attrs:
                                _resolved = [a for a in _loop_var_to_real_attrs[_name_arg] if a in known_attrs]
                                if _resolved:
                                    var_producers[lhs] = (set(_resolved), phys_lineno)
                                    _record_lineage(lhs, _vars_in(rhs), phys_lineno)
                                    continue
                            if (
                                _ast_fe_dd
                                and isinstance(_ast_rhs_node, ast.Call)
                                and _ast_fe_dd._is_self_getattr_call(_ast_rhs_node)
                            ):
                                _resolved = [
                                    a for a in known_attrs
                                    if _ast_fe_dd._dynamic_attr_expr_matches(_ast_rhs_node.args[1], a)
                                ]
                                if _resolved:
                                    var_producers[lhs] = (set(_resolved), phys_lineno)
                                    _record_lineage(lhs, _vars_in(rhs), phys_lineno)
                                    continue
                            # Iter15 tensor-alias: dynamic name bound to a
                            # known literal string list — union the producers
                            # of every alias name that's been registered.
                            if _name_arg in _loop_var_to_str_items_local:
                                _items = _loop_var_to_str_items_local[_name_arg]
                                _alias_union = set()
                                _alias_max_line = None
                                for _it in _items:
                                    if _it in tensor_alias_producers:
                                        _ap, _apl = tensor_alias_producers[_it]
                                        _alias_union |= _ap
                                        if _apl is not None and (_alias_max_line is None or _apl > _alias_max_line):
                                            _alias_max_line = _apl
                                if _alias_union:
                                    var_producers[lhs] = (_alias_union, _alias_max_line or phys_lineno)
                                    _record_lineage(lhs, _vars_in(rhs), phys_lineno)
                                    continue
                            # Iter15 tensor-alias: dynamic name expression that
                            # references an attribute on a non-tensor object,
                            # e.g. ``getattr(self, f"{cvr_task_conf.group_key}")``
                            # or ``getattr(self, group_key)`` where ``group_key``
                            # iterates over a dict whose keys are the same as
                            # the alias name set. Fall back to unioning ALL
                            # registered tensor aliases when no precise binding
                            # is available.
                            #
                            # Iter16: gate the union by f-string template prefix /
                            # suffix.  ``f"combine_tower_mha_{x.task_name}"`` has a
                            # literal prefix ``combine_tower_mha_`` that does NOT
                            # match any alias name (e.g. ``gift_layer_out``), so
                            # we MUST avoid the all-aliases union for that case.
                            # Without this gate, the producer of ``combine_tower``
                            # in ``_compute_task_tower_outputs_old`` is the union
                            # of all 5 GroupTowers, and ``combine_tower(group_tower_out_fp32)``
                            # creates a fully-connected dependency mesh among the
                            # 5 GroupTowers (Rule3 cycle).  Filter aliases by
                            # plausible-name match (literal prefix and suffix).
                            #
                            # Compatible aliases must:
                            #   - start with the literal prefix
                            #   - end with the literal suffix
                            # When the template is purely ``f"{x}"`` (no literal
                            # prefix/suffix), every alias is compatible (the old
                            # behaviour, which is still correct for genuine alias
                            # lookups like ``getattr(self, group_key)``).
                            _compat_aliases = None
                            if tensor_alias_producers:
                                _name_lit_prefix = ""
                                _name_lit_suffix = ""
                                # AST-only: derive prefix/suffix from the
                                # JoinedStr (f-string) AST node when the
                                # name expression is an f-string.  For pure
                                # ``f"literal"`` (a JoinedStr containing only a
                                # single Constant) we treat the entire literal
                                # as the prefix.  For an unparseable or non-
                                # f-string node, prefix/suffix stay empty.
                                _name_node_for_fstr = (
                                    _rhs_ast.args[1] if _rhs_ast is not None and len(_rhs_ast.args) >= 2 else None
                                )
                                if isinstance(_name_node_for_fstr, ast.JoinedStr):
                                    _values = _name_node_for_fstr.values
                                    if all(
                                        isinstance(_p, ast.Constant) and isinstance(_p.value, str)
                                        for _p in _values
                                    ):
                                        _name_lit_prefix = "".join(_p.value for _p in _values)
                                    else:
                                        _prefix_parts = []
                                        for _p in _values:
                                            if isinstance(_p, ast.Constant) and isinstance(_p.value, str):
                                                _prefix_parts.append(_p.value)
                                            else:
                                                break
                                        _suffix_parts = []
                                        for _p in reversed(_values):
                                            if isinstance(_p, ast.Constant) and isinstance(_p.value, str):
                                                _suffix_parts.append(_p.value)
                                            else:
                                                break
                                        _name_lit_prefix = "".join(_prefix_parts)
                                        _name_lit_suffix = "".join(reversed(_suffix_parts))
                                _compat_aliases = []
                                for _alias_name in tensor_alias_producers:
                                    if _name_lit_prefix and not _alias_name.startswith(_name_lit_prefix):
                                        continue
                                    if _name_lit_suffix and not _alias_name.endswith(_name_lit_suffix):
                                        continue
                                    _compat_aliases.append(_alias_name)
                            if _compat_aliases:
                                _alias_union = set()
                                _alias_max_line = None
                                for _alias_name in _compat_aliases:
                                    _prods, _pl = tensor_alias_producers[_alias_name]
                                    _alias_union |= _prods
                                    if _pl is not None and (_alias_max_line is None or _pl > _alias_max_line):
                                        _alias_max_line = _pl
                                if _alias_union:
                                    var_producers[lhs] = (_alias_union, _alias_max_line or phys_lineno)
                                    _record_lineage(lhs, _vars_in(rhs), phys_lineno)
                                    continue
                            if dynamic_setattr_attrs:
                                var_producers[lhs] = (set(dynamic_setattr_attrs), phys_lineno)
                                _record_lineage(lhs, _vars_in(rhs), phys_lineno)
                                continue

                        rhs_producers, _pl, _pv = _collect_expr_producers(rhs, var_producers, dict_slots, fname)
                        if rhs_producers:
                            # Iter15: when the rhs is a passthrough/transform
                            # of an existing tracked variable (no Module call
                            # on this line), keep the upstream production line
                            # so edge evidence anchors at the original Module
                            # call site rather than the intermediate line.
                            # Only use the upstream line if there's no direct
                            # ``self.<attr>(...)`` call on this rhs.
                            _has_self_call = False
                            if isinstance(_rhs_ast, ast.AST):
                                # AST-only check for ``self.<attr>(...)`` /
                                # ``self.<attr>[...]`` access or
                                # ``getattr(self, <expr>)(...)`` calls
                                # anywhere within the rhs.
                                for _sub in ast.walk(_rhs_ast):
                                    if (
                                        isinstance(_sub, ast.Attribute)
                                        and isinstance(_sub.value, ast.Name)
                                        and _sub.value.id == "self"
                                    ):
                                        _has_self_call = True
                                        break
                                    if (
                                        isinstance(_sub, ast.Call)
                                        and isinstance(_sub.func, ast.Name)
                                        and _sub.func.id == "getattr"
                                        and len(_sub.args) >= 2
                                        and isinstance(_sub.args[0], ast.Name)
                                        and _sub.args[0].id == "self"
                                    ):
                                        _has_self_call = True
                                        break
                            _line_for_lhs = phys_lineno
                            if (not _has_self_call) and _pl is not None:
                                _line_for_lhs = _pl
                            var_producers[lhs] = (rhs_producers, _line_for_lhs)
                            _record_lineage(lhs, _vars_in(rhs), phys_lineno)
                        else:
                            # No tracked module-producer in rhs, but still record
                            # an intermediate variable assignment for lineage if
                            # any tracked var appears on rhs.  This captures
                            # transformations like `b = a + bias` so the front-end
                            # can show the full chain.
                            _parents = _vars_in(rhs)
                            if _parents:
                                _record_lineage(lhs, _parents, phys_lineno)
                    _ast_append = _ast_stmt if _ast_stmt and _ast_stmt.get("kind") == "append" else None
                    if _ast_append:
                        list_var = (_ast_append.get("target_var") or "").strip()
                        rhs = (_ast_append.get("rhs_text") or "").strip()
                    elif (
                        isinstance(_line_stmt, ast.Expr)
                        and isinstance(_line_stmt.value, ast.Call)
                        and isinstance(_line_stmt.value.func, ast.Attribute)
                        and _line_stmt.value.func.attr == "append"
                        and isinstance(_line_stmt.value.func.value, ast.Name)
                        and len(_line_stmt.value.args) == 1
                    ):
                        # AST-only fallback: ``var.append(rhs)``
                        list_var = _line_stmt.value.func.value.id
                        try:
                            rhs = ast.unparse(_line_stmt.value.args[0])
                        except Exception:
                            rhs = ""
                    else:
                        list_var = None
                        rhs = None
                    if list_var and rhs is not None:
                        rhs_producers, _pl, _pv = _collect_expr_producers(rhs, var_producers, dict_slots, fname)
                        if rhs_producers:
                            existing = var_producers.get(list_var, (set(), phys_lineno))[0]
                            var_producers[list_var] = (existing | rhs_producers, phys_lineno)
                            _record_lineage(list_var, _vars_in(rhs), phys_lineno)
                    _ast_aug = _ast_stmt if _ast_stmt and _ast_stmt.get("kind") == "augassign" else None
                    if _ast_aug:
                        _aug_targets = [t for t in (_ast_aug.get("targets") or []) if t and t != '_']
                        list_var = _aug_targets[0] if len(_aug_targets) == 1 else None
                        rhs = (_ast_aug.get("rhs_text") or "").strip()
                    elif (
                        isinstance(_line_stmt, ast.AugAssign)
                        and isinstance(_line_stmt.op, ast.Add)
                        and isinstance(_line_stmt.target, ast.Name)
                    ):
                        # AST-only fallback: ``var += rhs``
                        list_var = _line_stmt.target.id
                        try:
                            rhs = ast.unparse(_line_stmt.value)
                        except Exception:
                            rhs = ""
                    else:
                        list_var = None
                        rhs = None
                    if list_var and rhs is not None:
                        rhs_producers, _pl, _pv = _collect_expr_producers(rhs, var_producers, dict_slots, fname)
                        if rhs_producers:
                            existing = var_producers.get(list_var, (set(), phys_lineno))[0]
                            var_producers[list_var] = (existing | rhs_producers, phys_lineno)
                            _record_lineage(list_var, [list_var] + _vars_in(rhs), phys_lineno)
                    # Handle tuple unpacking: (a, b) = ... or a, b = ...
                    _ast_tuple_targets = []
                    if _ast_stmt and _ast_stmt.get("kind") == "assign":
                        _targets = [v for v in (_ast_stmt.get("targets") or []) if v and v != '_']
                        if len(_targets) > 1:
                            _ast_tuple_targets = list(_targets)
                    # AST-only fallback: detect ``(a, b) = expr`` / ``a, b = expr``
                    _line_tuple_targets = []
                    _line_tuple_rhs = None
                    if (
                        not _ast_tuple_targets
                        and isinstance(_line_stmt, ast.Assign)
                        and len(_line_stmt.targets) == 1
                        and isinstance(_line_stmt.targets[0], ast.Tuple)
                    ):
                        _line_tuple_targets = [
                            elt.id for elt in _line_stmt.targets[0].elts
                            if isinstance(elt, ast.Name)
                        ]
                        if len(_line_tuple_targets) >= 2:
                            try:
                                _line_tuple_rhs = ast.unparse(_line_stmt.value)
                            except Exception:
                                _line_tuple_rhs = ""
                        else:
                            _line_tuple_targets = []
                    if _ast_tuple_targets or _line_tuple_targets:
                        lhs_vars = _ast_tuple_targets or _line_tuple_targets
                        if _ast_tuple_targets:
                            rhs = (_ast_stmt.get("rhs_text") or "")
                        else:
                            rhs = _line_tuple_rhs or ""
                        rhs_producers, _pl, _pv = _collect_expr_producers(rhs, var_producers, dict_slots, fname)
                        if rhs_producers:
                            for v in lhs_vars:
                                v = v.strip()
                                if v and v != '_':
                                    _line_for_v = phys_lineno
                                    _ast_var_entry = None
                                    if _ast_fe_dd is not None:
                                        _ast_var_entry = _ast_fe_dd._lookup_var_env_entry(_ast_var_env, v, phys_lineno)
                                    elif _ast_var_env.get(v):
                                        _chain = _ast_var_env.get(v) or []
                                        if not isinstance(_chain, list):
                                            _chain = [_chain]
                                        for _item in _chain:
                                            if _item and len(_item) > 1 and _item[1] is not None and _item[1] <= phys_lineno:
                                                _ast_var_entry = _item
                                    if _ast_var_entry is not None and len(_ast_var_entry) > 1 and _ast_var_entry[1] is not None:
                                        _line_for_v = _ast_var_entry[1]
                                    var_producers[v] = (rhs_producers, _line_for_v)
                                    _record_lineage(v, _vars_in(rhs), phys_lineno)
                    continue

                if called_attrs:
                    _ast_append = _ast_stmt if _ast_stmt and _ast_stmt.get("kind") == "append" else None
                    if _ast_append:
                        _list_var = (_ast_append.get("target_var") or "").strip()
                        _rhs_text = (_ast_append.get("rhs_text") or "").strip()
                        if _list_var:
                            _existing = var_producers.get(_list_var, (set(), phys_lineno))[0]
                            _call_producers = {
                                _attr for _attr, _call_start, _call_end, _call_node in called_attrs
                                if _attr in known_attrs
                            }
                            if _call_producers:
                                var_producers[_list_var] = (_existing | _call_producers, phys_lineno)
                                _record_lineage(_list_var, _vars_in(_rhs_text), phys_lineno)

                # For each module call, determine consumed variables
                for attr, call_start, call_end, call_node in called_attrs:
                    consumed_producers = set()
                    consumed_var_for_producer = {}  # producer_attr -> (var_name, prod_line)

                    if call_node is not None:
                        _arg_nodes = list(getattr(call_node, "args", []) or [])
                        _arg_nodes.extend([kw.value for kw in (getattr(call_node, "keywords", []) or []) if getattr(kw, "value", None) is not None])
                        for _arg_node in _arg_nodes:
                            _ps, _ev = _collect_ast_expr_consumers(_arg_node, attr)
                            if _ps:
                                consumed_producers.update(_ps)
                            for p, meta in (_ev or {}).items():
                                prev = consumed_var_for_producer.get(p)
                                if prev is None or (meta[1] is not None and (prev[1] is None or meta[1] > prev[1])):
                                    consumed_var_for_producer[p] = meta
                    # Note: a regex-based fallback path used to handle
                    # ``call_node is None`` here; that branch has been retired
                    # because every call now originates from the AST scan and
                    # always carries a real ``ast.Call`` node.

                    def _filter_chain_by_producer(chain, producer, var_producers):
                        result = []
                        for step in chain or []:
                            producers_of_step = var_producers.get(step.get("var", ""), (set(), None))[0]
                            if not producers_of_step or producer in producers_of_step:
                                result.append(step)
                        return result

                    for producer in consumed_producers:
                        if producer != attr:
                            edges.add((producer, attr))
                            ev = consumed_var_for_producer.get(producer, ("?", phys_lineno))
                            # Iter11 varlineage: gather the assignment chain that
                            # carried `producer`'s output into the consumer line.
                            # Steps come from var_lineage[var] for the variable
                            # name we identified. We append the consumer step at
                            # the end so the front-end shows a complete history
                            # ending at the actual call site.
                            _carrier = ev[0]
                            _chain = []
                            if isinstance(_carrier, str) and _carrier in var_lineage:
                                _chain = list(var_lineage.get(_carrier) or [])
                                _chain = _filter_chain_by_producer(_chain, producer, var_producers)
                            _consumer_text = _line_text(phys_lineno)
                            if _consumer_text:
                                _consumer_carriers = _extract_all_names(call_node)
                                _consumer_step = {"var": _carrier if isinstance(_carrier, str) else "",
                                                  "file": fname,
                                                  "line": phys_lineno,
                                                  "text": _consumer_text,
                                                  "role": "consumer",
                                                  "carriers": _consumer_carriers}
                                # Avoid duplicating if last chain entry is the
                                # consumer line already.
                                if not _chain or _chain[-1].get("line") != phys_lineno:
                                    _chain.append(_consumer_step)
                            # Store evidence: file, line where var was produced (defines from_attr's output)
                            # and line where it was consumed (this attr's call line)
                            edge_locs.setdefault((producer, attr), {
                                "file": fname,
                                "var": ev[0],
                                "from_line": ev[1],
                                "to_line": phys_lineno,
                                "lineage": _chain,
                            })

                # Handle LHS: what variables this line produces
                # Prefer AST targets / producer_attr when available, with regex retained as fallback.
                _ast_assign_like = _ast_stmt if _ast_stmt and _ast_stmt.get("kind") in ("assign", "augassign") else None
                _ast_lhs_targets = []
                if _ast_assign_like:
                    _ast_lhs_targets = [v for v in (_ast_assign_like.get("targets") or []) if v and v != '_']
                _handled_lhs = False
                if _ast_lhs_targets and called_attrs:
                    _ast_producer_attr = (_ast_assign_like.get("producer_attr") or "").strip() if _ast_assign_like else ""
                    last_calls = set()
                    if _ast_producer_attr:
                        if _ast_producer_attr in known_attrs:
                            last_calls = {_ast_producer_attr}
                        elif "[" in _ast_producer_attr:
                            _base, _idx = _ast_producer_attr.split("[", 1)
                            _idx = _idx.rstrip("]")
                            last_calls = set(a for a in _resolve_indexed_attr(_base, _idx) if a in known_attrs)
                    if not last_calls:
                        last_s, last_e = called_attrs[-1][1], called_attrs[-1][2]
                        last_calls = {a for (a, s, e, _node) in called_attrs if s == last_s and e == last_e}
                        last_calls = last_calls or {called_attrs[-1][0]}
                    if len(last_calls) > 1:
                        tail = None
                        for cont, elems in container_to_elems.items():
                            for ea in reversed(elems):
                                if ea in last_calls:
                                    tail = ea
                                    break
                            if tail:
                                break
                        if tail:
                            last_calls = {tail}
                    for v_clean in _ast_lhs_targets:
                        var_producers[v_clean] = (set(last_calls), phys_lineno)
                        _record_lineage(v_clean, _vars_in(line), phys_lineno)
                        if v_clean in var_lineage and var_lineage[v_clean]:
                            var_lineage[v_clean][-1]["carriers"] = list(_ast_lhs_targets)
                            if _ast_assign_like and _ast_assign_like.get("rhs_node") is not None:
                                var_lineage[v_clean][-1]["arg_carriers"] = _extract_all_names(_ast_assign_like.get("rhs_node"))
                    _handled_lhs = True
                if not _handled_lhs:
                    # AST-only LHS handling.
                    # Three target shapes are recognised:
                    #   1) ``var = getattr(self, 'attr')(...)``
                    #   2) ``var = self.attr(...)`` / ``var = self.attr[idx](...)``
                    #   3) ``var = <expr>`` / ``a, b = <expr>`` (general)
                    # In each case, the LHS targets and (for shapes 1/2) the
                    # producer attr come straight from the AST stmt.
                    _ast_assign_stmt = None
                    if isinstance(_line_stmt, ast.Assign):
                        _ast_assign_stmt = _line_stmt
                    _lhs_targets = []
                    _lhs_is_tuple = False
                    if _ast_assign_stmt and len(_ast_assign_stmt.targets) == 1:
                        _t0 = _ast_assign_stmt.targets[0]
                        if isinstance(_t0, ast.Name):
                            _lhs_targets = [_t0.id]
                        elif isinstance(_t0, ast.Tuple):
                            _lhs_targets = [
                                e.id for e in _t0.elts
                                if isinstance(e, ast.Name) and e.id != "_"
                            ]
                            _lhs_is_tuple = True

                    # Shape 1: ``... = getattr(self, 'attr')(...)``
                    _shape_handled = False
                    if (
                        _ast_assign_stmt is not None
                        and isinstance(_ast_assign_stmt.value, ast.Call)
                        and isinstance(_ast_assign_stmt.value.func, ast.Call)
                        and isinstance(_ast_assign_stmt.value.func.func, ast.Name)
                        and _ast_assign_stmt.value.func.func.id == "getattr"
                        and len(_ast_assign_stmt.value.func.args) >= 2
                        and isinstance(_ast_assign_stmt.value.func.args[0], ast.Name)
                        and _ast_assign_stmt.value.func.args[0].id == "self"
                        and isinstance(_ast_assign_stmt.value.func.args[1], ast.Constant)
                        and isinstance(_ast_assign_stmt.value.func.args[1].value, str)
                    ):
                        producing_attr = _ast_assign_stmt.value.func.args[1].value
                        if producing_attr in known_attrs:
                            for v_clean in _lhs_targets:
                                if v_clean and v_clean != "_" and v_clean.isidentifier():
                                    var_producers[v_clean] = ({producing_attr}, phys_lineno)
                                    _record_lineage(v_clean, _vars_in(line), phys_lineno)
                        _shape_handled = True
                        continue

                    # Shape 2: ``... = self.attr(...)`` or
                    #         ``... = self.attr[idx](...)``
                    if (
                        _ast_assign_stmt is not None
                        and isinstance(_ast_assign_stmt.value, ast.Call)
                    ):
                        _call_func = _ast_assign_stmt.value.func
                        _producing_attr_name = None
                        _producing_idx_expr = None
                        if (
                            isinstance(_call_func, ast.Attribute)
                            and isinstance(_call_func.value, ast.Name)
                            and _call_func.value.id == "self"
                        ):
                            _producing_attr_name = _call_func.attr
                        elif (
                            isinstance(_call_func, ast.Subscript)
                            and isinstance(_call_func.value, ast.Attribute)
                            and isinstance(_call_func.value.value, ast.Name)
                            and _call_func.value.value.id == "self"
                        ):
                            _producing_attr_name = _call_func.value.attr
                            try:
                                _producing_idx_expr = ast.unparse(_call_func.slice)
                            except Exception:
                                _producing_idx_expr = ""
                        if _producing_attr_name is not None:
                            if _producing_idx_expr is not None:
                                producing_attrs = _resolve_indexed_attr(
                                    _producing_attr_name, _producing_idx_expr)
                            else:
                                producing_attrs = [_producing_attr_name]
                            producing_attrs = [a for a in producing_attrs if a in known_attrs]
                            if producing_attrs:
                                for v_clean in _lhs_targets:
                                    if v_clean and v_clean != "_" and v_clean.isidentifier():
                                        var_producers[v_clean] = (set(producing_attrs), phys_lineno)
                                        _record_lineage(v_clean, _vars_in(line), phys_lineno)
                            _shape_handled = True

                    # Shape 3: general ``var = ...`` / ``a, b = ...``
                    # (only when shape 2 didn't match).  When ``called_attrs``
                    # is non-empty, the producer is the last-call attr.
                    if not _shape_handled and called_attrs:
                        last_s, last_e = called_attrs[-1][1], called_attrs[-1][2]
                        last_calls = {a for (a, s, e, _node) in called_attrs if s == last_s and e == last_e}
                        last_calls = last_calls or {called_attrs[-1][0]}
                        # 若 last_calls 是 container 的元素集合（例如 loop_var），取"链尾"作为更贴近真实的 producer
                        if len(last_calls) > 1:
                            tail = None
                            for cont, elems in container_to_elems.items():
                                for ea in reversed(elems):
                                    if ea in last_calls:
                                        tail = ea
                                        break
                                if tail:
                                    break
                            if tail:
                                last_calls = {tail}
                        if not _lhs_is_tuple and len(_lhs_targets) == 1:
                            lhs = _lhs_targets[0]
                            if lhs and lhs != "_" and lhs.isidentifier():
                                var_producers[lhs] = (set(last_calls), phys_lineno)
                                _record_lineage(lhs, _vars_in(line), phys_lineno)
                        elif _lhs_is_tuple and _lhs_targets:
                            for v in _lhs_targets:
                                if v and v != "_" and v.isidentifier():
                                    var_producers[v] = (set(last_calls), phys_lineno)
                                    _record_lineage(v, _vars_in(line), phys_lineno)

            # --- Cycle validation (Rule 3): DAG must be acyclic ---
            # Detect cycles using Kahn's algorithm and return cycle-involved nodes.
            def _detect_cycle_nodes(edge_set):
                if not edge_set:
                    return set()
                in_deg = defaultdict(int)
                adj = defaultdict(list)
                all_nodes = set()
                for (u, v) in edge_set:
                    adj[u].append(v)
                    in_deg[v] += 1
                    all_nodes.add(u)
                    all_nodes.add(v)
                queue = [n for n in all_nodes if in_deg[n] == 0]
                visited = 0
                while queue:
                    node = queue.pop()
                    visited += 1
                    for nb in adj[node]:
                        in_deg[nb] -= 1
                        if in_deg[nb] == 0:
                            queue.append(nb)
                if visited < len(all_nodes):
                    return {n for n in all_nodes if in_deg[n] > 0}
                return set()

            def _wildcard_base(attr_name):
                _m = re.match(r'^(\w+)(?:\[\*\])$', attr_name or '')
                if _m:
                    return _m.group(1)
                return attr_name

            _wildcard_bases = {
                _wildcard_base(a) for a in known_attrs
                if isinstance(a, str) and a.endswith('[*]')
            }
            if edges and _wildcard_bases:
                _drop_edges = set()
                _edges_by_base = defaultdict(list)
                for (u, v) in edges:
                    _edges_by_base[(_wildcard_base(u), _wildcard_base(v))].append((u, v))
                for (_ub, _vb), _uv_edges in list(_edges_by_base.items()):
                    if _ub == _vb:
                        continue
                    if _ub not in _wildcard_bases or _vb not in _wildcard_bases:
                        continue
                    _vu_edges = _edges_by_base.get((_vb, _ub), [])
                    if not _vu_edges:
                        continue
                    _best_uv = min(
                        _uv_edges,
                        key=lambda _e: (edge_locs.get(_e, {}) or {}).get('to_line') or (edge_locs.get(_e, {}) or {}).get('from_line') or 10**9,
                    )
                    _best_vu = min(
                        _vu_edges,
                        key=lambda _e: (edge_locs.get(_e, {}) or {}).get('to_line') or (edge_locs.get(_e, {}) or {}).get('from_line') or 10**9,
                    )
                    _uv_line = (edge_locs.get(_best_uv, {}) or {}).get('to_line') or (edge_locs.get(_best_uv, {}) or {}).get('from_line') or 10**9
                    _vu_line = (edge_locs.get(_best_vu, {}) or {}).get('to_line') or (edge_locs.get(_best_vu, {}) or {}).get('from_line') or 10**9
                    if _uv_line <= _vu_line:
                        _drop_edges.update(_vu_edges)
                    else:
                        _drop_edges.update(_uv_edges)
                if _drop_edges:
                    edges = {e for e in edges if e not in _drop_edges}
                    for _e in _drop_edges:
                        edge_locs.pop(_e, None)

            cycle_nodes = _detect_cycle_nodes(edges)
            if cycle_nodes and _multi_call_candidates:
                # Identify which multi-call candidates participate in cycles
                _attrs_to_split = _multi_call_candidates & cycle_nodes
                if _attrs_to_split:
                    # Build split_node_map for cycle-involved attrs
                    split_node_map = {}
                    for a in _attrs_to_split:
                        cnt = _attr_max_calls_in_single_method[a]
                        split_names = [f"{a}#{i}" for i in range(cnt)]
                        split_node_map[a] = split_names

                    # Determine occurrence order for each split attr by collecting
                    # all line numbers where it appears (as source or target) from edge evidence
                    _attr_line_numbers = defaultdict(set)
                    for (u, v), ev in edge_locs.items():
                        if not isinstance(ev, dict):
                            continue
                        if u in _attrs_to_split:
                            _attr_line_numbers[u].add(ev.get("from_line", 0))
                        if v in _attrs_to_split:
                            _attr_line_numbers[v].add(ev.get("to_line", 0))
                    _attr_sorted_lines = {a: sorted(lns) for a, lns in _attr_line_numbers.items()}

                    def _line_to_occurrence(attr, lineno):
                        sorted_lns = _attr_sorted_lines.get(attr, [])
                        if not sorted_lns:
                            return 0
                        cnt = _attr_max_calls_in_single_method[attr]
                        # Map line numbers to occurrence indices by position in sorted order
                        idx = 0
                        for i, ln in enumerate(sorted_lns):
                            if lineno >= ln:
                                idx = i
                        return min(idx, cnt - 1)

                    # Remap edges: split attrs get #N suffix based on line evidence
                    _new_edges = set()
                    _new_edge_locs = {}
                    for (u, v) in edges:
                        ev = edge_locs.get((u, v), {})
                        new_u = u
                        new_v = v
                        if u in _attrs_to_split and isinstance(ev, dict):
                            fl = ev.get("from_line", 0)
                            idx = _line_to_occurrence(u, fl)
                            new_u = f"{u}#{idx}"
                        if v in _attrs_to_split and isinstance(ev, dict):
                            tl = ev.get("to_line", 0)
                            idx = _line_to_occurrence(v, tl)
                            new_v = f"{v}#{idx}"
                        _new_edges.add((new_u, new_v))
                        _new_edge_locs[(new_u, new_v)] = ev
                    edges = _new_edges
                    edge_locs = _new_edge_locs

                    # Re-validate after splitting
                    cycle_nodes_2 = _detect_cycle_nodes(edges)
                    if cycle_nodes_2:
                        print(f"  [WARNING] Cycle still present in {cname} after splitting: {cycle_nodes_2}")
            elif cycle_nodes:
                print(f"  [WARNING] Cycle detected in class {cname}: nodes = {list(cycle_nodes)}")

            class_dep_edges[cname] = list(edges)
            class_edge_locs[cname] = edge_locs
            if split_node_map:
                class_split_info[cname] = dict(split_node_map)

    return class_dep_edges, class_edge_locs, class_split_info


# ---------------------------------------------------------------------------


# Per-thread timeline analysis


# ---------------------------------------------------------------------------






# ---------------------------------------------------------------------------
# Chrome tracing visualization
# ---------------------------------------------------------------------------



# ---------------------------------------------------------------------------
# Phase 1: Build parent-child relationships
# ---------------------------------------------------------------------------





# ---------------------------------------------------------------------------
# Phase 2: Identify threads and their roles
# ---------------------------------------------------------------------------



# ---------------------------------------------------------------------------
# Phase 3: Training step decomposition
# ---------------------------------------------------------------------------



# ---------------------------------------------------------------------------
# Phase 4: Module analysis (per-thread aware)
# ---------------------------------------------------------------------------









# ---------------------------------------------------------------------------
# Phase 5: GPU timeline analysis (kernel hotspot + gaps + host mapping)
# ---------------------------------------------------------------------------



# ---------------------------------------------------------------------------
# Phase 6: Device/Host overview
# ---------------------------------------------------------------------------



# ---------------------------------------------------------------------------
# Phase 7: Worker thread analysis
# ---------------------------------------------------------------------------



# ---------------------------------------------------------------------------
# Phase 8: Communication/Compute overlap analysis
# ---------------------------------------------------------------------------









# ===========================================================================
# Printing / Output
# ===========================================================================

# ===========================================================================
# Source code hotspot report
# ===========================================================================

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


def build_static_module_tree(source_files, preferred_root=None, conditional_mode="infer", runtime_overrides=None):
    """Build module hierarchy tree purely from source code.
    Args:
        source_files: dict of {filename: [lines]}
        preferred_root: optional class name to use as primary root if present in tree.
            Useful when timeline indicates a specific top-level wrapper (e.g. DDPAwemeLiveCVR
            implies AwemeLiveCVR is the actual model entry-point).
        conditional_mode: one of "infer" / "train" / "default".
            Iter13 Step1: inside ``__init__``, when we encounter::
                if is_training: self.attr = A(...)
                else:           self.attr = B(...)
            (or any equivalent variant whose condition references
            ``is_training`` / ``is_serving`` / ``training`` / ``serving``),
            we follow ONLY the matching branch:
              - ``infer`` → take the inference branch (default)
              - ``train`` → take the training branch
              - ``default`` → fall back to last-write-wins (legacy behaviour,
                kept for tests / backwards compatibility)
            The discovered ``class_conditional_attrs`` mapping is also stashed
            on every tree[cname] as ``conditional_attrs`` so the caller can
            decide whether to render train/infer tabs.
    Returns: tree dict, root list, class_map
    """
    ast_frontends = _build_ast_frontends(source_files)

    def _get_ast_frontend(fname):
        return ast_frontends.get(fname)

    class_map = _build_class_map(source_files, ast_frontends=ast_frontends)
    module_call_order, _ = _build_source_dependency_order(source_files, class_map, ast_frontends=ast_frontends)

    # Iter18: Build the AST module tree IN-PROCESS (no JSON intermediate).
    # The extractor runs once per build_static_module_tree() invocation and
    # produces a parallel "ground-truth" view of class_attrs / container
    # elements / dynamic setattrs that downstream consumers can consult when
    # the existing regex pass is uncertain.  The data is stashed on each
    # tree[cname] entry as ``_ast_*`` keys near the end of this function.
    _ast_extractor = _AST_ModuleTreeExtractor(source_files, ast_frontends=ast_frontends)

    # ------------------------------------------------------------------
    # 轻量常量表：用于 ModuleList/ModuleDict 展开时解析 range(N) / layers=CONST
    # 仅解析“文件级”形如 `NAME = 6` 的 int 常量。
    # AST-only:  parse each source file once and inspect top-level
    # ``ast.Assign`` statements where the target is an UPPER_SNAKE Name
    # and the value is a non-negative integer Constant.
    # ------------------------------------------------------------------
    file_int_consts = {}
    file_str_list_globals_raw = {}  # {fname: {VARNAME: raw_inner_string}}

    def _is_upper_snake_name(_s: str) -> bool:
        return bool(_s) and _s[0:1].isupper() and all(
            c.isupper() or c.isdigit() or c == "_" for c in _s
        )

    nn_import_aliases = {}  # {alias_name: short_name}, e.g. {"TorchLinear": "Linear"}

    for _fname, _lines in source_files.items():
        int_consts = {}
        str_list_consts_raw = {}
        _fe = _get_ast_frontend(_fname)
        _file_tree = _fe.tree if _fe is not None else None
        if _file_tree is not None:
            for _node in _file_tree.body:
                if isinstance(_node, ast.ImportFrom) and _node.module == "torch.nn":
                    for _alias in _node.names:
                        if _alias.name == "*":
                            continue
                        _alias_name = _alias.asname or _alias.name
                        nn_import_aliases[_alias_name] = _alias.name
                    continue
                if not isinstance(_node, ast.Assign) or len(_node.targets) != 1:
                    continue
                _t = _node.targets[0]
                if not (isinstance(_t, ast.Name) and _is_upper_snake_name(_t.id)):
                    continue
                _v = _node.value
                # int constant: ``NAME = 42``
                if isinstance(_v, ast.Constant) and isinstance(_v.value, int) and not isinstance(_v.value, bool):
                    if _v.value >= 0:
                        int_consts[_t.id] = _v.value
                # list literal: ``NAME = ['a', 'b']`` — store the unparse of
                # the inner element list so the existing
                # ``_parse_str_literal_list`` helper can consume it later.
                elif isinstance(_v, ast.List):
                    try:
                        _inner_text = ", ".join(ast.unparse(_e) for _e in _v.elts)
                    except Exception:
                        _inner_text = ""
                    str_list_consts_raw[_t.id] = _inner_text
        file_int_consts[_fname] = int_consts
        file_str_list_globals_raw[_fname] = str_list_consts_raw

    def _extract_nn_short(cls_ref_full, nn_import_aliases=None):
        """Normalize a constructor ref into _is_nn_leaf_stub's short name.

        Supports nn.Linear, torch.nn.Linear, and from-torch.nn import aliases.
        Returns (short_name, canonical_ref) or (None, None).
        """
        cls_ref_full = (cls_ref_full or "").strip()
        if cls_ref_full.startswith("nn."):
            short = cls_ref_full.split(".", 1)[1]
            return short, cls_ref_full
        if cls_ref_full.startswith("torch.nn."):
            short = cls_ref_full.split(".", 2)[2]
            return short, "nn." + short
        if nn_import_aliases and cls_ref_full in nn_import_aliases:
            short = nn_import_aliases[cls_ref_full]
            return short, "nn." + short
        return None, None

    # 全局常量：同名常量在多个文件出现时，若值唯一则可跨文件解析
    global_int_const_values = defaultdict(set)
    for _fname, _cs in file_int_consts.items():
        for _k, _v in _cs.items():
            global_int_const_values[_k].add(_v)

    # Iter13 Step2: file-level string-literal list globals, e.g.
    #   COMMON_KEYS = ['a', 'b']
    # Used to resolve ModuleDict keys driven by such constants.
    # NOTE: ``file_str_list_globals_raw`` was populated above (AST-driven).
    # Realised lookup: parsed lists, populated below once
    # ``_parse_str_literal_list`` is available.  We keep it as a placeholder dict
    # and fill it lazily in the per-class scanning loop.
    file_str_list_globals = {}  # {fname: {VARNAME: [str, ...]}}

    # ctor_kw_int_args[ClassName][kw] = {1,2,...}
    ctor_kw_int_args = defaultdict(lambda: defaultdict(set))

    # Iter17: ctor_kw_list_lens[ClassName][kw] = {len1, len2, ...}
    # Records the LENGTH (int) of list literals passed as kw= at construction
    # sites of ClassName.  Used to expand `for i, x in enumerate(<param>):`
    # inside ClassName.__init__ when <param> is bound to a kw with known
    # list-length.
    ctor_kw_list_lens = defaultdict(lambda: defaultdict(set))

    # Iter17: per-instance ctor kw list lengths.
    # instance_kw_list_lens[(parent_cname, attr_name)][kw] = N
    # Used in build_dag_recursive to PRUNE per-instance children counts when
    # the class internally expands containers via `for ... in enumerate(<kw>)`.
    instance_kw_list_lens = defaultdict(dict)

    def _eval_expr_node(expr_text: str):
        expr_text = (expr_text or '').strip()
        if not expr_text:
            return None
        try:
            return ast.parse(expr_text, mode="eval").body
        except Exception:
            return None

    def _resolver_scope(fname: str, cname: str = None, mname: str = None,
                        parent_cls: str = None, parent_attr: str = None):
        return Scope(file=fname, cls=cname, method=mname,
                     parent_cls=parent_cls, parent_attr=parent_attr)

    def _eval_int_node(expr_node, fname: str, cname: str = None, mname: str = None,
                       parent_cls: str = None, parent_attr: str = None):
        if expr_node is None or _new_eval_resolver is None:
            return None
        try:
            iv = _new_eval_resolver.eval_int(
                expr_node,
                _resolver_scope(fname, cname, mname, parent_cls, parent_attr),
            )
        except Exception:
            return None
        return iv.value if iv is not None else None

    def _eval_list_len_node(expr_node, fname: str, cname: str = None, mname: str = None,
                            parent_cls: str = None, parent_attr: str = None):
        if expr_node is None or _new_eval_resolver is None:
            return None
        try:
            return _new_eval_resolver.eval_list_len(
                expr_node,
                _resolver_scope(fname, cname, mname, parent_cls, parent_attr),
            )
        except Exception:
            return None

    # Build inheritance info - detect which classes extend nn.Module (or other user classes)
    nn_module_classes = set()
    for fname, lines in source_files.items():
        _fe = _get_ast_frontend(fname)
        if _fe:
            for _ast_cname in _fe.class_registry.keys():
                if _fe.is_nn_module(_ast_cname):
                    nn_module_classes.add(_ast_cname)
        _file_tree_for_classes = _fe.tree if _fe is not None else None
        if _file_tree_for_classes is not None:
            for _node in ast.walk(_file_tree_for_classes):
                if not isinstance(_node, ast.ClassDef) or not _node.bases:
                    continue
                cname = _node.name
                for _base_node in _node.bases:
                    try:
                        _b = ast.unparse(_base_node)
                    except Exception:
                        _b = ""
                    b_short = _b.split(".")[-1] if _b else ""
                    if b_short in ("Module", "nn.Module") or b_short in nn_module_classes:
                        nn_module_classes.add(cname)
                        break
    # Also add classes found in class_map that have forward() method
    for (fname, cname), info in class_map.items():
        if "forward" in info["methods"]:
            nn_module_classes.add(cname)

    # ------------------------------------------------------------------
    # PR3 — build the ConstantTable + ConstantResolver once and use it as the
    # only evaluator path.
    # ------------------------------------------------------------------
    _new_eval_table = None
    _new_eval_resolver = None
    _new_eval_diagnostics = []
    _new_eval_ast_frontends = {}
    for _fname in source_files.keys():
        _fe = _get_ast_frontend(_fname)
        if _fe is not None:
            _new_eval_ast_frontends[_fname] = _fe
    _new_eval_table = ConstantTable.build_all(
        source_files=source_files,
        ast_frontends=_new_eval_ast_frontends,
        nn_module_classes=set(nn_module_classes),
    )
    _new_eval_resolver = ConstantResolver(
        _new_eval_table,
        runtime_overrides=runtime_overrides,
        diagnostics=_new_eval_diagnostics,
    )


    # Per-class attr->class mapping. We scan ALL methods (not just __init__) because
    # some codebases lazily build child sub-modules in helper methods (e.g.
    # ``self.revenue_perceiver_module = SeqPerceiver(...)`` inside ``gen_lsf_fc``).
    # We also capture ``setattr(self, name, ClassName(...))`` patterns where the
    # attribute name comes from a runtime variable — such usages cannot be resolved
    # to a literal attr name, so we synthesise one from the class to keep the child
    # connected (group_<ClassName>).
    class_attrs = {}  # {class_name: {attr_name: class_ref}}
    torch_native_module_classes = set()  # {"nn.LayerNorm", "nn.Linear", ...}
    # Per-(class, attr) source location: {(cname, attr_name): (fname, lineno)}
    attr_def_loc = {}
    # Track attrs that are LG input sources (LG.feature_column/dense_feature/get_sample_rate/get_bias)
    input_source_attrs = defaultdict(set)  # {class_name: {attr_name, ...}}
    # Iter11: literal string-list attrs, e.g. self.group_tower_names = ['relation_layer', ...]
    # {class_name: {attr_name: [str, str, ...]}}
    # Used to drive dynamic setattr/getattr expansion in for-loops over literal str lists.
    class_str_list_attrs = defaultdict(dict)
    # Iter11: track which attrs were registered via dynamic setattr (non-literal name,
    # not driven by a known str-list loop). DAG node name uses just the class name
    # (e.g. `MultiHeadAttention`); the per-class set lets the data-flow analyzer know
    # to wildcard-route dynamic getattr calls to these attrs.
    dynamic_attrs_per_class = defaultdict(set)  # {class_name: {attr_name, ...}}

    # Iter13 Step1: per-class conditional attr mapping discovered while
    # scanning ``__init__`` for ``if is_training: ...; else: ...`` patterns.
    # Structure:
    #   class_conditional_attrs[cname][attr_name] = {
    #       "train_class": Cls or None,
    #       "infer_class": Cls or None,
    #       "train_loc":  (file, lineno) or None,
    #       "infer_loc":  (file, lineno) or None,
    #   }
    # Independent of ``conditional_mode`` — we always populate this so that
    # the caller can decide whether to render a train / infer tab.
    class_conditional_attrs = defaultdict(dict)

    # Iter13 Step2: unresolved ModuleDict/ModuleList expansions.
    # Each entry: {"class": cname, "attr": attr_name, "expr": raw_expr,
    #              "reason": str, "file": fname, "line": lineno}
    # Surfaced in stdout as `[ERROR] Cannot statically enumerate ...` and
    # also stashed on the tree (tree[cname]["unresolved_moduledict"]) for
    # downstream summary printing in generate_html_flowchart().
    unresolved_moduledict = []

    def _cond_branch_polarity(cond_text: str):
        """Iter13 Step1: classify whether a Python ``if <cond>:`` header
        targets the **training** branch or the **inference** branch.

        Returns one of:
          - "train"   → cond evaluates True only when training
          - "infer"   → cond evaluates True only when serving/inference
          - None      → cond does not reference is_training / is_serving keywords

        Heuristic only — we don't try to fully evaluate complex Python
        expressions.  We look for whole-word occurrences of the four
        canonical names (case-insensitive, including ``self.is_training``
        / ``common.is_training`` style attribute access) and figure out
        the polarity from the surrounding ``not``.
        """
        if not cond_text:
            return None
        text = cond_text.strip()
        # Strip a trailing colon if any (defensive — _cond_if_re keeps it out).
        text = text.rstrip(':').strip()
        # AST-only: parse the cond as a Python expression, then walk Name/
        # Attribute nodes to collect tokens; track UnaryOp(Not, ...) clauses
        # to detect polarity flips around the canonical names.
        try:
            _cond_node = ast.parse(text, mode="eval").body
        except SyntaxError:
            return None
        train_words = ("is_training", "training")
        serve_words = ("is_serving", "serving")

        def _leaf_token(_node):
            """Render the rightmost identifier of a Name/Attribute chain
            (lowercased). Returns None for anything else."""
            if isinstance(_node, ast.Name):
                return _node.id.lower()
            if isinstance(_node, ast.Attribute):
                return _node.attr.lower()
            return None

        # Collect every leaf token anywhere in the expression so we can
        # decide whether a train / serve canonical name appears at all.
        tokens = []
        for _sub in ast.walk(_cond_node):
            if isinstance(_sub, (ast.Name, ast.Attribute)):
                _tok = _leaf_token(_sub)
                if _tok:
                    tokens.append(_tok)
        has_train = any(t in train_words for t in tokens)
        has_serve = any(t in serve_words for t in tokens)
        if not (has_train or has_serve):
            return None
        # Detect a UnaryOp(Not, <name/attr>) whose leaf token is a canonical
        # train/serve word — that flips polarity.
        flipped = False
        for _sub in ast.walk(_cond_node):
            if isinstance(_sub, ast.UnaryOp) and isinstance(_sub.op, ast.Not):
                _tail = _leaf_token(_sub.operand)
                if _tail and (_tail in train_words or _tail in serve_words):
                    flipped = True
                    break
        # Compute base polarity from which keyword family appeared.
        if has_train and not has_serve:
            base = "train"
        elif has_serve and not has_train:
            base = "infer"
        else:
            # Mixed expression like ``is_training and is_serving`` is nonsense; skip.
            return None
        if flipped:
            return "infer" if base == "train" else "train"
        return base

    # ModuleList/ModuleDict 子模块展开：
    # - container_elems[cname][container_attr] = ["layers[0]", "layers[1]", ...] / ["loss_fns['ctr0']", ...]
    # - 这些 elem_attr 会额外写入 attrs 中，保证 DAG 里出现真实子模块节点。
    container_elems = defaultdict(lambda: defaultdict(list))
    # Iter14 container-group: record the *kind* (declared container class) of
    # each container attr so the DAG renderer can wrap expanded children in a
    # synthetic container parent group labelled e.g. ``ModuleDict[name][N]`` or
    # ``ModuleList[blocks][4]``. Keys: (cname, container_attr) -> kind string,
    # one of {"ModuleDict", "ModuleList", "dict", "list"}.
    container_kinds = {}

    def _record_container_kind(cname: str, container_attr: str, kind: str):
        """Record (or upgrade) the kind for cname.container_attr.

        Once recorded as a Module-typed container ("ModuleDict"/"ModuleList"),
        a later assignment (e.g. ``self.xx[k] = SomeModule(...)``) must not
        downgrade the kind to ``dict``/``list``.
        """
        key = (cname, container_attr)
        existing = container_kinds.get(key)
        priority = {"ModuleDict": 4, "ModuleList": 3, "dict": 2, "list": 1}
        if existing is None or priority.get(kind, 0) > priority.get(existing, 0):
            container_kinds[key] = kind

    def _normalize_str_key(expr: str):
        expr = (expr or '').strip()
        if not expr:
            return None
        try:
            node = ast.parse(expr, mode='eval').body
        except SyntaxError:
            return None
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            return node.value
        return None

    def _elem_attr_list(container: str, idx: int) -> str:
        return f"{container}[{idx}]"

    def _elem_attr_dict(container: str, key: str) -> str:
        # 统一使用单引号，便于稳定匹配
        key = key.replace("'", "\\'")
        return f"{container}['{key}']"

    def _ensure_elem(container: str, cname: str, elem_attr: str, cls_ref: str, fname: str, lineno: int, attrs: dict):
        if elem_attr not in attrs:
            attrs[elem_attr] = cls_ref
            attr_def_loc.setdefault((cname, elem_attr), (fname, lineno))
        if elem_attr not in container_elems[cname][container]:
            container_elems[cname][container].append(elem_attr)

    def _split_top_level_commas(s: str):
        """把 `a(b, c), d, {'k': e}` 按顶层逗号切分。"""
        parts = []
        buf = []
        depth = 0
        in_str = None
        esc = False
        for ch in s:
            if in_str:
                buf.append(ch)
                if esc:
                    esc = False
                elif ch == '\\':
                    esc = True
                elif ch == in_str:
                    in_str = None
                continue
            if ch in ("'", '"'):
                in_str = ch
                buf.append(ch)
                continue
            if ch in '([{':
                depth += 1
            elif ch in ')]}':
                depth = max(0, depth - 1)
            if ch == ',' and depth == 0:
                p = ''.join(buf).strip()
                if p:
                    parts.append(p)
                buf = []
            else:
                buf.append(ch)
        last = ''.join(buf).strip()
        if last:
            parts.append(last)
        return parts

    # Iter11: parse a literal list of string literals.
    # e.g. "['relation_layer', 'gift_layer', \"stay_layer\"]"
    # Returns list of strings, or None if any non-string-literal element exists.
    def _parse_str_literal_list(list_inner: str):
        try:
            node = ast.parse('[' + list_inner + ']', mode='eval').body
        except SyntaxError:
            return None
        if not isinstance(node, ast.List):
            return None
        result = []
        for elt in node.elts:
            if isinstance(elt, ast.Constant) and isinstance(elt.value, str):
                result.append(elt.value)
            else:
                return None
        return result if result else None

    # Iter13 Step2: realise the str-list globals lookup now that
    # ``_parse_str_literal_list`` is available.
    for _fname, _slots in file_str_list_globals_raw.items():
        parsed = {}
        for _name, _raw in _slots.items():
            _items = _parse_str_literal_list(_raw)
            if _items:
                parsed[_name] = _items
        file_str_list_globals[_fname] = parsed
    # Cross-file fallback: when a global str-list is unique by name, allow
    # lookup without specifying the file.
    _global_str_lists_unique = {}
    _name_to_lists = defaultdict(set)
    for _fname, _slots in file_str_list_globals.items():
        for _name, _items in _slots.items():
            _name_to_lists[_name].add(tuple(_items))
    for _name, _lists in _name_to_lists.items():
        if len(_lists) == 1:
            _global_str_lists_unique[_name] = list(next(iter(_lists)))

    _str_list_resolve_cache = {}

    def _resolve_str_list(expr: str, fname: str, cname: str,
                          loop_var_to_str_items: dict):
        """Iter13 Step2 + Phase B: iterative multi-step string-list resolver.
        Delegates to the module-level _resolve_str_list_iterative which handles:
          1. direct loop variable lookup
          2. f-string expansion with loop variable substitution
          3. self.<attr> referring to a known class str-list attr
          4. file-level / global UPPER_CASE str-list constant
          5. bare Name / dotted attribute → None (unresolvable)
        """
        _loop_key = tuple(
            sorted((str(k), tuple(v or ())) for k, v in (loop_var_to_str_items or {}).items())
        )
        _cache_key = (expr, fname, cname, _loop_key, conditional_mode)
        if _cache_key in _str_list_resolve_cache:
            _cached_keys = _str_list_resolve_cache[_cache_key]
            return list(_cached_keys) if _cached_keys is not None else None
        _resolved = _resolve_str_list_iterative(
            expr=expr,
            fname=fname,
            cname=cname,
            source_lines=source_files.get(fname, []),
            loop_var_to_str_items=loop_var_to_str_items,
            class_str_list_attrs=class_str_list_attrs,
            file_str_list_globals=file_str_list_globals,
            cond_branch_stack=[],
            conditional_mode=conditional_mode,
            _global_str_lists_unique=_global_str_lists_unique,
        )
        _str_list_resolve_cache[_cache_key] = tuple(_resolved) if _resolved is not None else None
        return list(_resolved) if _resolved is not None else None

    for (fname, cname), info in class_map.items():
        if cname not in nn_module_classes:
            continue
        lines = source_files.get(fname, [])
        _fe = _get_ast_frontend(fname)
        attrs = {}

        # Scan every method body for module assignments
        for mname, (ms, me) in info["methods"].items():
            _ast_method_range = _fe.get_method_lines(cname, mname) if _fe else None
            if _ast_method_range:
                ms, me = _ast_method_range
            method_lines = lines[ms - 1: min(me, len(lines))]
            # Track absolute line numbers per joined logical line by storing the
            # index of the first physical line that contributed to it.
            logical_lines = []
            buf = ""
            buf_start_lineno = None
            open_count = 0
            for offset, line in enumerate(method_lines):
                phys_lineno = ms + offset
                stripped = line.strip()
                if not stripped or stripped.startswith('#'):
                    if not buf:
                        logical_lines.append((phys_lineno, line))
                    continue
                if not buf:
                    buf_start_lineno = phys_lineno
                stripped_nc = _strip_inline_comment(stripped).strip()
                if not stripped_nc:
                    # 这一行去掉注释后为空
                    continue
                buf += (" " if buf else "") + stripped_nc
                open_count += stripped_nc.count('(') + stripped_nc.count('[') + stripped_nc.count('{')
                open_count -= stripped_nc.count(')') + stripped_nc.count(']') + stripped_nc.count('}')
                if open_count <= 0:
                    logical_lines.append((buf_start_lineno, buf))
                    buf = ""
                    buf_start_lineno = None
                    open_count = 0
            if buf:
                logical_lines.append((buf_start_lineno, buf))

            _ast_init_assignments_by_line = {}
            if mname == "__init__" and _fe:
                for _ast_item in (_fe.get_init_assignments_ast(cname) or []):
                    _lineno = _ast_item.get("lineno")
                    if _lineno is not None:
                        _ast_init_assignments_by_line[_lineno] = _ast_item

            # Iter11: track loop variables that iterate over a literal string list,
            # so `for name in self.xxx_names: setattr(self, name, Cls(...))` expands
            # the setattr into real per-element attrs ['relation_layer', 'gift_layer', ...].
            # Scope: simple per-method, no nested-loop complexity. Reset per method.
            _loop_var_to_str_items = {}  # {loop_var_name: [str_item, ...]}
            _loop_indent_stack = []  # [(indent, loop_var)] tracking active loops

            # Iter13 Step1: track if/else conditional branches whose header
            # references is_training / is_serving / training / serving.
            # Stack entries: (indent_of_if_header, current_branch_polarity)
            #   current_branch_polarity ∈ {"train", "infer"}.
            # On ``else:`` at the SAME indent as the matching ``if``, we flip
            # the top-of-stack polarity. On dedent past the if-header indent,
            # we pop. Only top-level (single-frame) conditionals are honoured —
            # nested is_training/is_serving conditionals stack as expected.
            _cond_branch_stack = []  # [(if_indent, branch_polarity)]

            # Iter16: per-method local-variable constructor table.
            # Tracks `var = ClassName(...)` where ClassName is a known
            # nn.Module class. Used to resolve the two-step setattr pattern:
            #     combine_tower = DenseTower(...)
            #     setattr(self, f"combine_tower_mha_{name}", combine_tower)
            # so that `setattr(self, name_expr, var)` can be treated as an
            # inline `setattr(self, name_expr, ClassName(...))`.
            #   {var_name: (cls_ref, fname, ctor_lineno)}
            _local_var_ctor: dict = {}

            # Phase B: track local variable → string list assignments for
            # iterative multi-step resolution. Handles patterns like:
            #     ks0 = ['a', 'b', 'c']
            #     ks1 = ks0
            #     keys = ks1
            #     for k in keys: ...
            # {var_name: list[str]}
            _local_var_str_list: dict = {}

            def _self_attr_from_target(_target):
                return _fe._extract_self_attr_name(_target) if _fe else None

            def _node_to_text(_node):
                return _fe._local_node_to_text(_node) if _fe else ""

            for phys_lineno, line in logical_lines:
                # Iter11: track indent of this logical line (use raw source line)
                _raw_line_for_indent = lines[phys_lineno - 1] if 0 < phys_lineno <= len(lines) else ""
                _cur_indent = len(_raw_line_for_indent) - len(_raw_line_for_indent.lstrip())
                # Iter12-final-fix: skip pure comment lines.  `_join_logical_lines`
                # preserves blank/comment lines (with the leading ``#``) when no
                # multi-line buffer is in flight, so without this guard a
                # commented-out ``# self.dense_towers.append(DenseTower(`` would
                # be matched by the AST setattr/append helpers below and pollute
                # attr_def_loc / attr enumeration with attrs that have no real
                # call site.
                if line.lstrip().startswith('#'):
                    continue
                _local_stmt = _fe.parse_local_stmt(line) if _fe else None
                # Pop loops whose indent is >= current indent (we left their body)
                while _loop_indent_stack and _cur_indent <= _loop_indent_stack[-1][0]:
                    _, _popped_var = _loop_indent_stack.pop()
                    _loop_var_to_str_items.pop(_popped_var, None)

                # Iter13 Step1: pop conditional frames whose if-header indent is
                # >= current indent (we left their body) — but only when the
                # current line is NOT the matching ``else:`` (which lives at the
                # same indent as the if header).
                # Detect ``if/elif/else:`` headers via AST instead of regex.
                # Strategy: try parsing ``<line>\n    pass`` so a header line
                # becomes a syntactically complete If statement.
                _hdr_node = None
                _stripped_line = line.rstrip()
                if _stripped_line.endswith(':'):
                    _stripped_left = _stripped_line.lstrip()
                    if _stripped_left.startswith(('if ', 'elif ', 'else:', 'if(', 'elif(')) \
                            or _stripped_left == 'else:':
                        try:
                            _hdr_mod = ast.parse(_stripped_line + "\n    pass")
                        except SyntaxError:
                            _hdr_mod = None
                        if _hdr_mod and len(_hdr_mod.body) == 1 \
                                and isinstance(_hdr_mod.body[0], ast.If):
                            _hdr_node = _hdr_mod.body[0]
                _is_else_line = (_hdr_node is not None
                                 and _stripped_line.lstrip().startswith('else'))
                _is_elif_line = (_hdr_node is not None
                                 and _stripped_line.lstrip().startswith('elif'))
                _is_if_line = (_hdr_node is not None and not _is_else_line and not _is_elif_line)
                while _cond_branch_stack:
                    _if_ind, _polarity = _cond_branch_stack[-1]
                    if _cur_indent < _if_ind:
                        _cond_branch_stack.pop()
                    elif _cur_indent == _if_ind and (_is_else_line or _is_elif_line):
                        # ``else:`` at the matching indent → flip polarity.
                        _cond_branch_stack[-1] = (_if_ind,
                                                  "infer" if _polarity == "train" else "train")
                        break
                    elif _cur_indent == _if_ind:
                        # Iter14 fix: same indent as the ``if`` header but the
                        # current line is *not* an ``else``/``elif`` — the
                        # conditional block has ended and execution returned to
                        # the surrounding scope. Pop the frame so subsequent
                        # ``self.xxx = ...`` lines are treated as unconditional
                        # (shared between train/infer). Without this, e.g.
                        # CTRModule.__init__'s ``if not is_training: self.df=...``
                        # line was leaving the stack populated, causing every
                        # later ``self.ue = UniversalEmbedding()`` etc. to be
                        # filtered out in train mode.
                        _cond_branch_stack.pop()
                    else:
                        break

                # Detect new ``if <cond>:`` whose cond polarity is train/infer.
                # We treat a fresh ``if`` (not elif) as opening a new frame.
                if _is_if_line and _hdr_node is not None:
                    try:
                        _cond_text = ast.unparse(_hdr_node.test)
                    except Exception:
                        _cond_text = ""
                    _pol = _cond_branch_polarity(_cond_text)
                    if _pol is not None:
                        _cond_branch_stack.append((_cur_indent, _pol))

                # Compute the active branch polarity (top of stack) — None when
                # we are not inside any train/infer conditional.
                _active_branch = _cond_branch_stack[-1][1] if _cond_branch_stack else None

                # Iter11: detect literal string-list assignments:
                #   self.xxx_names = ['a', 'b', ...]
                # Tracked separately from `attrs` because the elements are not nn.Modules
                # by themselves; they are *names* used to drive setattr/getattr.
                if (isinstance(_local_stmt, ast.Assign)
                        and len(_local_stmt.targets) == 1
                        and isinstance(_local_stmt.value, ast.List)):
                    _slt_attr = _self_attr_from_target(_local_stmt.targets[0])
                    if _slt_attr:
                        _slt_items = []
                        _slt_ok = True
                        for _e in _local_stmt.value.elts:
                            if isinstance(_e, ast.Constant) and isinstance(_e.value, str):
                                _slt_items.append(_e.value)
                            else:
                                _slt_ok = False
                                break
                        if _slt_ok and _slt_items:
                            class_str_list_attrs[cname][_slt_attr] = list(_slt_items)

                # Phase B: track local variable string-list assignments
                # Patterns: var = ['a', 'b'] | var = other_known_var | var = self.attr
                if (isinstance(_local_stmt, ast.Assign)
                        and len(_local_stmt.targets) == 1
                        and isinstance(_local_stmt.targets[0], ast.Name)):
                    _lv_name = _local_stmt.targets[0].id
                    _lv_val = _local_stmt.value
                    if isinstance(_lv_val, ast.List):
                        _lv_items = []
                        _lv_ok = True
                        for _e in _lv_val.elts:
                            if isinstance(_e, ast.Constant) and isinstance(_e.value, str):
                                _lv_items.append(_e.value)
                            else:
                                _lv_ok = False
                                break
                        if _lv_ok and _lv_items:
                            _local_var_str_list[_lv_name] = list(_lv_items)
                    elif isinstance(_lv_val, ast.Name) and _lv_val.id in _local_var_str_list:
                        _local_var_str_list[_lv_name] = list(_local_var_str_list[_lv_val.id])
                    elif (isinstance(_lv_val, ast.Attribute)
                          and isinstance(_lv_val.value, ast.Name)
                          and _lv_val.value.id == 'self'):
                        _ref_items = class_str_list_attrs.get(cname, {}).get(_lv_val.attr)
                        if _ref_items:
                            _local_var_str_list[_lv_name] = list(_ref_items)

                # Iter11: detect `for [i,] name in [enumerate(]self.xxx_names[)]:`
                # If self.xxx_names is a known literal string list, record the loop var
                # so subsequent setattr(self, name, Cls(...)) can be expanded.
                # AST-based: parse via "<line>\n    pass" so we get a real ast.For.
                _for_node = None
                _stripped_for_line = line.rstrip()
                if _stripped_for_line.lstrip().startswith('for ') and _stripped_for_line.endswith(':'):
                    try:
                        _for_mod = ast.parse(_stripped_for_line + "\n    pass")
                    except SyntaxError:
                        _for_mod = None
                    if (_for_mod and len(_for_mod.body) == 1
                            and isinstance(_for_mod.body[0], ast.For)):
                        _for_node = _for_mod.body[0]

                def _self_attr_simple(_n):
                    if (isinstance(_n, ast.Attribute)
                            and isinstance(_n.value, ast.Name)
                            and _n.value.id == 'self'):
                        return _n.attr
                    return None

                def _str_list_from_node(_n):
                    if not isinstance(_n, ast.List):
                        return None
                    out = []
                    for _e in _n.elts:
                        if isinstance(_e, ast.Constant) and isinstance(_e.value, str):
                            out.append(_e.value)
                        else:
                            return None
                    return out if out else None

                _m_for_use_hit = False
                _m_for_lit_hit = False
                _m_for_tup_hit = False
                if _for_node is not None:
                    _iter = _for_node.iter
                    _tgt = _for_node.target
                    # Determine "second variable" (the name var) for both
                    # Name and Tuple targets.
                    _name_var = None
                    if isinstance(_tgt, ast.Name):
                        _name_var = _tgt.id
                    elif isinstance(_tgt, ast.Tuple) and len(_tgt.elts) >= 2 \
                            and all(isinstance(e, ast.Name) for e in _tgt.elts):
                        # `for i, name in enumerate(...)` → name var = last
                        _name_var = _tgt.elts[-1].id

                    # Strip outer enumerate(...) wrapper.
                    _enum_inner = None
                    if (isinstance(_iter, ast.Call)
                            and isinstance(_iter.func, ast.Name)
                            and _iter.func.id == 'enumerate'
                            and len(_iter.args) >= 1):
                        _enum_inner = _iter.args[0]

                    # Rule: for x in self.xxx: / for i, x in enumerate(self.xxx):
                    _src_self = _self_attr_simple(_iter) or (
                        _self_attr_simple(_enum_inner) if _enum_inner is not None else None)
                    if _src_self and _name_var and (
                            isinstance(_tgt, ast.Name)
                            or (_enum_inner is not None and isinstance(_tgt, ast.Tuple))):
                        _items = class_str_list_attrs.get(cname, {}).get(_src_self)
                        if _items:
                            _loop_var_to_str_items[_name_var] = list(_items)
                            _loop_indent_stack.append((_cur_indent, _name_var))
                            _m_for_use_hit = True

                    # Rule: for x in [literals]: / for i, x in enumerate([literals]):
                    if not _m_for_use_hit and _name_var:
                        _lit_node = _iter if isinstance(_iter, ast.List) else (
                            _enum_inner if _enum_inner is not None else None)
                        _items = _str_list_from_node(_lit_node) if _lit_node is not None else None
                        if _items and (
                                isinstance(_tgt, ast.Name)
                                or (_enum_inner is not None and isinstance(_tgt, ast.Tuple))):
                            _loop_var_to_str_items[_name_var] = list(_items)
                            _loop_indent_stack.append((_cur_indent, _name_var))
                            _m_for_lit_hit = True

                    # Rule: for (name, a, b, ...) in [(s, ..), (s, ..), ...]:
                    if (not _m_for_use_hit and not _m_for_lit_hit
                            and isinstance(_tgt, ast.Tuple)
                            and len(_tgt.elts) >= 2
                            and all(isinstance(e, ast.Name) for e in _tgt.elts)
                            and isinstance(_iter, ast.List)):
                        _first_var = _tgt.elts[0].id
                        _str_items = []
                        _ok = True
                        for _outer_elt in _iter.elts:
                            if not isinstance(_outer_elt, ast.Tuple) or not _outer_elt.elts:
                                _ok = False
                                break
                            _first_inner = _outer_elt.elts[0]
                            if not (isinstance(_first_inner, ast.Constant)
                                    and isinstance(_first_inner.value, str)):
                                _ok = False
                                break
                            _str_items.append(_first_inner.value)
                        if _ok and _str_items:
                            _loop_var_to_str_items[_first_var] = _str_items
                            _loop_indent_stack.append((_cur_indent, _first_var))
                            _m_for_tup_hit = True

                    # Phase B Rule 4: for x in local_var / for i, x in enumerate(local_var)
                    # where local_var is a tracked string-list local variable.
                    if (not _m_for_use_hit and not _m_for_lit_hit and not _m_for_tup_hit
                            and _name_var):
                        _local_iter_node = _iter if _enum_inner is None else _enum_inner
                        if isinstance(_local_iter_node, ast.Name):
                            _local_items = _local_var_str_list.get(_local_iter_node.id)
                            if _local_items:
                                _loop_var_to_str_items[_name_var] = list(_local_items)
                                _loop_indent_stack.append((_cur_indent, _name_var))

                _ast_init_item = _ast_init_assignments_by_line.get(phys_lineno)

                _ast_ctor_assign = _fe.parse_local_ctor_assign(_local_stmt) if _fe else None
                if _ast_ctor_assign:
                    _ctor_attr = _ast_ctor_assign.get("attr")
                    _cls_full = _ast_ctor_assign.get("class_full") or ""
                    _cls = _cls_full.split('.')[-1]
                    if _cls:
                        _kw_lens = {}
                        for k, v_expr in (_ast_ctor_assign.get("kwargs") or {}).items():
                            v_node = _eval_expr_node(v_expr)
                            v = _eval_int_node(v_node, fname, cname, mname)
                            if v is not None:
                                ctor_kw_int_args[_cls][k].add(v)
                            n = _eval_list_len_node(v_node, fname, cname, mname)
                            if n is not None:
                                ctor_kw_list_lens[_cls][k].add(n)
                                _kw_lens[k] = n
                        if _kw_lens and _ctor_attr:
                            instance_kw_list_lens[(cname, _ctor_attr)] = dict(_kw_lens)
                elif _ast_init_item:
                    _ctor_attr = _ast_init_item.get("attr")
                    _cls_full = _ast_init_item.get("class") or ""
                    _cls = _cls_full.split('.')[-1]
                    if _cls:
                        _kw_lens = {}
                        for k, v_expr in (_ast_init_item.get("kwargs") or {}).items():
                            v_node = _eval_expr_node(v_expr)
                            v = _eval_int_node(v_node, fname, cname, mname)
                            if v is not None:
                                ctor_kw_int_args[_cls][k].add(v)
                            n = _eval_list_len_node(v_node, fname, cname, mname)
                            if n is not None:
                                ctor_kw_list_lens[_cls][k].add(n)
                                _kw_lens[k] = n
                        if _kw_lens and _ctor_attr:
                            instance_kw_list_lens[(cname, _ctor_attr)] = dict(_kw_lens)

                # Direct assignment: self.xxx = ClassName(...)
                if _ast_ctor_assign or _ast_init_item:
                    if _ast_ctor_assign:
                        attr_name = _ast_ctor_assign.get("attr")
                        cls_ref_full = _ast_ctor_assign.get("class_full") or ""
                    else:
                        attr_name = _ast_init_item.get("attr")
                        cls_ref_full = _ast_init_item.get("class") or ""
                    cls_ref = cls_ref_full.split('.')[-1]
                    native_cls_ref = None
                    _nn_short, _nn_canonical = _extract_nn_short(cls_ref_full, nn_import_aliases)
                    if _nn_short and _is_nn_leaf_stub(_nn_short):
                        native_cls_ref = _nn_canonical
                    if cls_ref in nn_module_classes:
                        # Iter13 Step1: track conditional attrs regardless of mode.
                        if _active_branch is not None:
                            _entry = class_conditional_attrs[cname].setdefault(
                                attr_name,
                                {"train_class": None, "infer_class": None,
                                 "train_loc": None, "infer_loc": None})
                            if _active_branch == "train":
                                _entry["train_class"] = cls_ref
                                _entry["train_loc"] = (fname, phys_lineno)
                            else:
                                _entry["infer_class"] = cls_ref
                                _entry["infer_loc"] = (fname, phys_lineno)
                        # Iter13 Step1: when conditional_mode is non-default and
                        # this assignment lives in the OTHER branch, skip the
                        # actual write so the chosen mode wins.
                        _skip = (_active_branch is not None
                                 and conditional_mode != "default"
                                 and _active_branch != conditional_mode)
                        if not _skip:
                            attrs[attr_name] = cls_ref
                            attr_def_loc.setdefault((cname, attr_name), (fname, phys_lineno))
                    elif native_cls_ref:
                        attrs[attr_name] = native_cls_ref
                        attr_def_loc.setdefault((cname, attr_name), (fname, phys_lineno))
                        torch_native_module_classes.add(native_cls_ref)
                    else:
                        # Detect LG input source modules:
                        # LG.feature_column(...), LG.dense_feature(...),
                        # LG.get_sample_rate(...), LG.get_bias(...)
                        _ast_lg = _fe.parse_local_lg_assign(_local_stmt) if _fe else None
                        if _ast_lg:
                            attr_name = _ast_lg.get("attr")
                            attrs[attr_name] = "__LG_InputSource"
                            attr_def_loc.setdefault((cname, attr_name), (fname, phys_lineno))
                            input_source_attrs[cname].add(attr_name)

                # Detect fc_dict[...].get_vector(...) assigned to self.xxx
                # Pattern: self.xxx = fc_dict[slot].get_vector(...) or similar
                _ast_fc_gv = _fe.parse_local_fc_get_vector_assign(_local_stmt) if _fe else None
                if _ast_fc_gv:
                    attr_name = _ast_fc_gv.get("attr")
                    if attr_name not in attrs:
                        attrs[attr_name] = "__LG_InputSource"
                        attr_def_loc.setdefault((cname, attr_name), (fname, phys_lineno))
                        input_source_attrs[cname].add(attr_name)

                _ast_container_ctor = _fe.parse_local_container_ctor(_local_stmt) if _fe else None
                if _ast_container_ctor and _ast_container_ctor.get("kind") == "ModuleList":
                    cont = _ast_container_ctor.get("attr")
                    _record_container_kind(cname, cont, "ModuleList")
                    _call = _ast_container_ctor.get("call")
                    _arg0 = _call.args[0] if _call and _call.args else None
                    if isinstance(_arg0, ast.ListComp) and isinstance(_arg0.elt, ast.Call):
                        _generators = _arg0.generators or []
                        _gen0 = _generators[0] if len(_generators) == 1 else None
                        _cls_ref = _node_to_text(_arg0.elt.func).split('.')[-1]
                        _n = None
                        if _gen0 and isinstance(_gen0.iter, ast.Call) and (_fe._expr_leaf_name(_gen0.iter.func) == "range") and len(_gen0.iter.args) == 1:
                            _n = _eval_int_node(_gen0.iter.args[0], fname, cname, mname, parent_cls=preferred_root, parent_attr='stack')
                        # Phase E1.1: literal list iter, e.g.
                        # ``[Cls(name) for name in ['a', 'b', 'c']]``
                        if _n is None and _gen0 is not None and isinstance(_gen0.iter, ast.List):
                            _n = len(_gen0.iter.elts)
                        # Phase E1.1: file-level / local UPPER_CASE str-list constant
                        # (resolved via ConstantTable.file_str_list_consts when available).
                        if _n is None and _gen0 is not None and isinstance(_gen0.iter, ast.Name):
                            _len = _eval_list_len_node(_gen0.iter, fname, cname, mname)
                            if _len is not None:
                                _n = _len
                        if _n is not None:
                            _short, _canonical = _extract_nn_short(_node_to_text(_arg0.elt.func), nn_import_aliases)
                            if _short and _is_nn_leaf_stub(_short):
                                attrs.setdefault(cont, _canonical)
                                attr_def_loc.setdefault((cname, cont), (fname, phys_lineno))
                                torch_native_module_classes.add(_canonical)
                                for i in range(_n):
                                    _ensure_elem(cont, cname, _elem_attr_list(cont, i), _canonical, fname, phys_lineno, attrs)
                            elif _cls_ref in nn_module_classes:
                                attrs.setdefault(cont, _cls_ref)
                                attr_def_loc.setdefault((cname, cont), (fname, phys_lineno))
                                for i in range(_n):
                                    _ensure_elem(cont, cname, _elem_attr_list(cont, i), _cls_ref, fname, phys_lineno, attrs)
                    elif isinstance(_arg0, ast.List):
                        for i, _elt in enumerate(_arg0.elts):
                            if isinstance(_elt, ast.Call):
                                cls_ref_full = _node_to_text(_elt.func)
                                cls_ref = cls_ref_full.split('.')[-1]
                                _short, _canonical = _extract_nn_short(cls_ref_full, nn_import_aliases)
                                if _short and _is_nn_leaf_stub(_short):
                                    attrs.setdefault(cont, _canonical)
                                    attr_def_loc.setdefault((cname, cont), (fname, phys_lineno))
                                    torch_native_module_classes.add(_canonical)
                                    _ensure_elem(cont, cname, _elem_attr_list(cont, i), _canonical, fname, phys_lineno, attrs)
                                elif cls_ref in nn_module_classes:
                                    attrs.setdefault(cont, cls_ref)
                                    attr_def_loc.setdefault((cname, cont), (fname, phys_lineno))
                                    _ensure_elem(cont, cname, _elem_attr_list(cont, i), cls_ref, fname, phys_lineno, attrs)

                if _ast_container_ctor and _ast_container_ctor.get("kind") == "ModuleDict":
                    cont = _ast_container_ctor.get("attr")
                    _record_container_kind(cname, cont, "ModuleDict")
                    _call = _ast_container_ctor.get("call")
                    _arg0 = _call.args[0] if _call and _call.args else None
                    if isinstance(_arg0, ast.Dict):
                        for _k_node, _v_node in zip(_arg0.keys, _arg0.values):
                            if not isinstance(_v_node, ast.Call):
                                continue
                            _key_text = _node_to_text(_k_node)
                            k = _normalize_str_key(_key_text)
                            cls_ref = _node_to_text(_v_node.func).split('.')[-1]
                            if k is not None and cls_ref in nn_module_classes:
                                attrs.setdefault(cont, cls_ref)
                                attr_def_loc.setdefault((cname, cont), (fname, phys_lineno))
                                _ensure_elem(cont, cname, _elem_attr_dict(cont, k), cls_ref, fname, phys_lineno, attrs)
                # ModuleList append: self.xxx.append(ClassName(...))
                _ast_append_ctor = _fe.parse_local_append_ctor(_local_stmt) if _fe else None
                if _ast_append_ctor:
                    attr_name = _ast_append_ctor.get("attr")
                    cls_ref_full = _ast_append_ctor.get("class_full") or ""
                    cls_ref = cls_ref_full.split('.')[-1]
                    _short, _canonical = _extract_nn_short(cls_ref_full, nn_import_aliases)
                    if _short and _is_nn_leaf_stub(_short):
                        attrs[attr_name] = _canonical
                        attr_def_loc.setdefault((cname, attr_name), (fname, phys_lineno))
                        torch_native_module_classes.add(_canonical)
                        idx = len(container_elems[cname][attr_name])
                        _ensure_elem(attr_name, cname, _elem_attr_list(attr_name, idx), _canonical, fname, phys_lineno, attrs)
                        _record_container_kind(cname, attr_name, "list")
                    elif cls_ref in nn_module_classes:
                        # 保留基线：container 名仍映射到子类
                        attrs[attr_name] = cls_ref
                        attr_def_loc.setdefault((cname, attr_name), (fname, phys_lineno))
                        # 追加展开：container[idx]
                        idx = len(container_elems[cname][attr_name])
                        _ensure_elem(attr_name, cname, _elem_attr_list(attr_name, idx), cls_ref, fname, phys_lineno, attrs)
                        # If we did not previously see a Module-typed declaration for
                        # this attr, fall back to ``list`` (could be plain list of Modules).
                        _record_container_kind(cname, attr_name, "list")
                # ModuleDict / list assignment: self.xxx[key] = ClassName(...)
                _ast_subscript_ctor = _fe.parse_local_subscript_ctor(_local_stmt) if _fe else None
                if _ast_subscript_ctor:
                    attr_name = _ast_subscript_ctor.get("attr")
                    key_expr = _ast_subscript_ctor.get("key_expr") or ""
                    cls_ref_full = _ast_subscript_ctor.get("class_full") or ""
                    cls_ref = cls_ref_full.split('.')[-1]
                    if cls_ref in nn_module_classes:
                        attrs[attr_name] = cls_ref
                        attr_def_loc.setdefault((cname, attr_name), (fname, phys_lineno))
                        # `self.xx[key] = Module(...)` is most commonly used with
                        # plain ``dict`` instances.  If we already saw a
                        # ``ModuleDict`` declaration earlier, _record_container_kind
                        # will keep the higher-priority kind.
                        _record_container_kind(cname, attr_name, "dict")
                        k = _normalize_str_key(key_expr)
                        if k is not None:
                            _ensure_elem(attr_name, cname, _elem_attr_dict(attr_name, k), cls_ref, fname, phys_lineno, attrs)
                        else:
                            # Iter13 Step2: try to resolve dynamic key to a list
                            # of string literals via class str-list attrs, local
                            # for-loop variable, or file/global UPPER_CASE
                            # constants. If successful, expand into per-key
                            # elem attrs (no [*] fallback). Otherwise warn,
                            # record an unresolved entry, and fall back to
                            # the legacy `[*]` wildcard.
                            _keys = _resolve_str_list(
                                key_expr, fname, cname, _loop_var_to_str_items)
                            if _keys:
                                for _k in _keys:
                                    _ensure_elem(attr_name, cname,
                                                 _elem_attr_dict(attr_name, _k),
                                                 cls_ref, fname, phys_lineno, attrs)
                            else:
                                _reason = "non-literal key expression"
                                print(f"[WARN] ModuleDict key not enumerable: "
                                      f"class={cname} attr={attr_name} expr={key_expr!r}",
                                      file=sys.stderr)
                                print(f"[ERROR] Cannot statically enumerate keys for "
                                      f"{cname}.{attr_name}, reason: {_reason}",
                                      file=sys.stderr)
                                unresolved_moduledict.append({
                                    "class": cname,
                                    "attr": attr_name,
                                    "expr": key_expr,
                                    "reason": _reason,
                                    "file": fname,
                                    "line": phys_lineno,
                                })
                                # 动态 key：补一个通配节点，保证 forward 的依赖能连上
                                _ensure_elem(attr_name, cname, f"{attr_name}[*]", cls_ref, fname, phys_lineno, attrs)
                # Iter16: detect local-variable constructor assignments
                #     var = ClassName(...)
                # where ClassName is a known nn.Module class and `var` is NOT
                # `self.xxx` (the `self.xxx = Cls(...)` form is handled by m
                # above). The collected entries are looked up by the two-step
                # setattr extension below, so that
                #     combine_tower = DenseTower(...)
                #     setattr(self, f"combine_tower_mha_{name}", combine_tower)
                # is treated as a direct setattr-with-ctor.
                _ast_local_ctor = _fe.parse_local_var_ctor(_local_stmt) if _fe else None
                if _ast_local_ctor:
                    _lv_name = _ast_local_ctor.get("name")
                    _lv_cls = (_ast_local_ctor.get("class_full") or "").split('.')[-1]
                    # Avoid clobbering reserved names; skip `self`/`super`.
                    if (_lv_name not in ("self", "super")
                            and _lv_cls in nn_module_classes):
                        _local_var_ctor[_lv_name] = (_lv_cls, fname, phys_lineno)

                # setattr(self, name_var_or_literal, ClassName(...))
                # - If name is a literal string, register the real attr name.
                # - Iter11: if name is a loop variable iterating over a known literal
                #   string list (e.g. `for name in self.xxx_names: setattr(self, name, Cls(...))`),
                #   expand into per-element real attrs.
                # - Otherwise, register a synthetic attr ``<ClassName>`` (i.e. the class name itself,
                #   no prefix) so the child class is at least connected in the DAG.
                #   We additionally track this attr in `dynamic_attrs_per_class` so that the
                #   downstream data-flow analyzer can wildcard-route `getattr(self, dynamic_var)(...)`
                #   calls to these dynamically-registered modules.
                # Iter16: also support the two-step pattern where the third
                # argument is a bare local variable previously assigned via
                # `<var> = ClassName(...)` (see _local_var_ctor above).
                _ast_setattr_ctor = _fe.parse_local_setattr_ctor(_local_stmt) if _fe else None
                _is_two_step = False
                if _ast_setattr_ctor:
                    name_expr = (_ast_setattr_ctor.get("name_expr") or "").strip()
                    if _ast_setattr_ctor.get("two_step"):
                        _bv = _ast_setattr_ctor.get("local_var")
                        if _bv in _local_var_ctor:
                            cls_ref_full = _local_var_ctor[_bv][0]
                            _is_two_step = True
                        else:
                            cls_ref_full = ""
                    else:
                        cls_ref_full = _ast_setattr_ctor.get("class_full") or ""
                    if cls_ref_full:
                        cls_ref = cls_ref_full.split('.')[-1]
                        _native_short, _native_canonical = _extract_nn_short(cls_ref_full, nn_import_aliases)
                        if _native_short and _is_nn_leaf_stub(_native_short):
                            # name literal: 'xxx' / "xxx" / f'xxx'(no {}) / f"xxx"(no {})
                            real_attr = None
                            try:
                                _name_node = ast.parse(name_expr, mode='eval').body
                            except SyntaxError:
                                _name_node = None
                            if isinstance(_name_node, ast.Constant) and isinstance(_name_node.value, str):
                                real_attr = _name_node.value
                            elif isinstance(_name_node, ast.JoinedStr):
                                if all(isinstance(v, ast.Constant) and isinstance(v.value, str)
                                       for v in _name_node.values):
                                    real_attr = ''.join(v.value for v in _name_node.values)
                            if real_attr:
                                attrs[real_attr] = _native_canonical
                                attr_def_loc.setdefault((cname, real_attr), (fname, phys_lineno))
                                torch_native_module_classes.add(_native_canonical)
                        elif cls_ref in nn_module_classes:
                            # name literal: 'xxx' / "xxx" / f'xxx'(no {}) / f"xxx"(no {})
                            real_attr = None
                            _fstr_handled = False
                            try:
                                _name_node = ast.parse(name_expr, mode='eval').body
                            except SyntaxError:
                                _name_node = None
                            if isinstance(_name_node, ast.Constant) and isinstance(_name_node.value, str):
                                real_attr = _name_node.value
                            elif isinstance(_name_node, ast.JoinedStr):
                                # f-string with no interpolations → all values are
                                # ast.Constant strings; concatenate them.
                                if all(isinstance(v, ast.Constant) and isinstance(v.value, str)
                                       for v in _name_node.values):
                                    real_attr = ''.join(v.value for v in _name_node.values)
                                else:
                                    # Phase B: f-string with loop variable interpolation
                                    _fstr_expanded = _expand_fstring_with_loop_vars(
                                        _name_node, _loop_var_to_str_items)
                                    if _fstr_expanded:
                                        for _real in _fstr_expanded:
                                            attrs[_real] = cls_ref
                                            attr_def_loc.setdefault((cname, _real), (fname, phys_lineno))
                                        _fstr_handled = True
                            if real_attr:
                                attrs[real_attr] = cls_ref
                                attr_def_loc.setdefault((cname, real_attr), (fname, phys_lineno))
                            elif not _fstr_handled and name_expr in _loop_var_to_str_items:
                                # Iter11: expand using the literal string list driven by the loop variable.
                                for _real in _loop_var_to_str_items[name_expr]:
                                    attrs[_real] = cls_ref
                                    attr_def_loc.setdefault((cname, _real), (fname, phys_lineno))
                            elif not _fstr_handled:
                                # Iter11: use the class name itself as the synthetic attr name
                                # (no `__setattr_` prefix). Tracked in dynamic_attrs_per_class.
                                synth_attr = cls_ref
                                # Only register if no real literal attr already maps to this class
                                if synth_attr not in attrs and not any(v == cls_ref for v in attrs.values()):
                                    attrs[synth_attr] = cls_ref
                                    attr_def_loc.setdefault((cname, synth_attr), (fname, phys_lineno))
                                    dynamic_attrs_per_class[cname].add(synth_attr)
        class_attrs[cname] = attrs

    # ------------------------------------------------------------------
    # 二次扫描（保守增强）：识别 range(N) 循环中的 `var = ClassName(...)` + `self.list.append(var)`
    # 用于展开像 Transformer.resblocks 这种"先构造到局部变量再 append"的写法。
    # 仅当 N 可解析为 int 时展开为 N 个 [i] 节点；解析失败则不影响基线（仍保留 container 映射）。
    # ------------------------------------------------------------------

    # AST helpers — replace the legacy regexes _FOR_RANGE_RE / _FOR_ENUM_RE /
    # _FOR_ITER_RE / _ASSIGN_CTOR_RE / _APPEND_VAR_RE / _APPEND_CTOR_RE /
    # _SELF_SET_RE.
    def _parse_for_header(_text: str):
        """Return (kind, base_indent, iter_expr) for a single ``for ...:`` line.

        kind: 'range' | 'enumerate' | 'iter' | None.
        Returns None when the line is not a parseable for-header.
        """
        if not _text:
            return None
        _stripped = _text.rstrip()
        _left = _stripped.lstrip()
        if not _left.startswith('for ') or not _stripped.endswith(':'):
            return None
        try:
            _mod = ast.parse(_stripped + "\n    pass")
        except SyntaxError:
            return None
        if not (_mod.body and isinstance(_mod.body[0], ast.For)):
            return None
        _f = _mod.body[0]
        _indent = len(_text) - len(_text.lstrip())
        _it = _f.iter
        # range(...)
        if (isinstance(_it, ast.Call)
                and isinstance(_it.func, ast.Name)
                and _it.func.id == 'range'
                and len(_it.args) >= 1):
            try:
                _expr = ast.unparse(_it.args[0])
            except Exception:
                _expr = ""
            return ('range', _indent, _expr)
        # enumerate(<iter>)
        if (isinstance(_it, ast.Call)
                and isinstance(_it.func, ast.Name)
                and _it.func.id == 'enumerate'
                and len(_it.args) >= 1):
            try:
                _expr = ast.unparse(_it.args[0])
            except Exception:
                _expr = ""
            return ('enumerate', _indent, _expr)
        # plain `for x in <Name|Attr>:`  — only Name/dotted Attr (no Call).
        if isinstance(_f.target, ast.Name) and isinstance(_it, (ast.Name, ast.Attribute)):
            try:
                _expr = ast.unparse(_it)
            except Exception:
                _expr = ""
            return ('iter', _indent, _expr)
        return None

    def _parse_assign_ctor_line(_text: str):
        """For ``var = ClassName(...)`` (simple Name target, Call value with
        Name/Attribute func), return (var_name, class_leaf). Else None."""
        try:
            _mod = ast.parse(_text)
        except SyntaxError:
            return None
        if not _mod.body or not isinstance(_mod.body[0], ast.Assign):
            return None
        _a = _mod.body[0]
        if len(_a.targets) != 1 or not isinstance(_a.targets[0], ast.Name):
            return None
        if not isinstance(_a.value, ast.Call):
            return None
        _f = _a.value.func
        if isinstance(_f, ast.Name):
            _leaf = _f.id
        elif isinstance(_f, ast.Attribute):
            _leaf = _f.attr
        else:
            return None
        return (_a.targets[0].id, _leaf)

    def _parse_append_var_line(_text: str):
        """For ``self.cont.append(var)`` (Name arg) return (cont, var). Else None."""
        try:
            _mod = ast.parse(_text)
        except SyntaxError:
            return None
        if not _mod.body or not isinstance(_mod.body[0], ast.Expr):
            return None
        _c = _mod.body[0].value
        if not isinstance(_c, ast.Call):
            return None
        _func = _c.func
        if not (isinstance(_func, ast.Attribute) and _func.attr == 'append'):
            return None
        _owner = _func.value
        if not (isinstance(_owner, ast.Attribute)
                and isinstance(_owner.value, ast.Name)
                and _owner.value.id == 'self'):
            return None
        if len(_c.args) != 1 or _c.keywords:
            return None
        if not isinstance(_c.args[0], ast.Name):
            return None
        return (_owner.attr, _c.args[0].id)

    def _parse_append_ctor_line(_text: str):
        """For ``self.cont.append(ClassName(...))`` return (cont, class_leaf). Else None."""
        try:
            _mod = ast.parse(_text)
        except SyntaxError:
            return None
        if not _mod.body or not isinstance(_mod.body[0], ast.Expr):
            return None
        _c = _mod.body[0].value
        if not isinstance(_c, ast.Call):
            return None
        _func = _c.func
        if not (isinstance(_func, ast.Attribute) and _func.attr == 'append'):
            return None
        _owner = _func.value
        if not (isinstance(_owner, ast.Attribute)
                and isinstance(_owner.value, ast.Name)
                and _owner.value.id == 'self'):
            return None
        if len(_c.args) < 1:
            return None
        _arg0 = _c.args[0]
        if not isinstance(_arg0, ast.Call):
            return None
        _af = _arg0.func
        if isinstance(_af, ast.Name):
            _leaf = _af.id
        elif isinstance(_af, ast.Attribute):
            _leaf = _af.attr
        else:
            return None
        return (_owner.attr, _leaf)

    # Iter17: per-class container -> driving kw param name (for per-instance prune).
    # class_container_kw[cname][container_attr] = kw_name (the ctor kw whose
    # list-length determines the iteration length).
    class_container_kw = defaultdict(dict)

    def _resolve_iter_len(expr_node, fname: str, cname: str, mname: str = "__init__",
                          parent_cls: str = None, parent_attr: str = None):
        n = _eval_list_len_node(expr_node, fname, cname, mname, parent_cls, parent_attr)
        kw_name = None
        if isinstance(expr_node, ast.Name):
            kw_name = expr_node.id
        elif (isinstance(expr_node, ast.Attribute)
              and isinstance(expr_node.value, ast.Name)
              and expr_node.value.id == "self"):
            kw_name = self_to_param.get(expr_node.attr)
        return n, kw_name

    def _resolve_range_n(expr_node, fname: str, cname: str, mname: str = "__init__",
                         parent_cls: str = None, parent_attr: str = None):
        return _eval_int_node(expr_node, fname, cname, mname, parent_cls, parent_attr)

    for (fname, cname), info in class_map.items():
        if cname not in nn_module_classes:
            continue
        lines = source_files.get(fname, [])
        attrs = class_attrs.get(cname, {})
        _fe = _get_ast_frontend(fname)
        # 建立 self.attr = param 的映射（如 self.layers = layers）
        self_to_param = _fe.get_self_param_aliases(cname, "__init__") if _fe else {}

        # Iter17 guard: collect the set of self.<attr> containers that are
        # accessed via dynamic indexing (`self.x[idx]` where idx is NOT a
        # literal integer) in any non-__init__ method of this class.  When a
        # container is used dynamically the analyzer's downstream logic
        # already handles it via a wildcard route; expanding it to per-index
        # nodes via AST loop expansion would only add nodes the dynamic call
        # can't supply var_history evidence for (Rule6_out regression).
        dyn_indexed_containers = _fe.get_dynamic_indexed_self_attrs(cname) if _fe else set()

        # 扫描各方法，找 for-range 的 append(var)
        for mname, (ms, me) in info['methods'].items():
            method_lines = lines[ms - 1: min(me, len(lines))]
            _ast_loop_records = _fe.get_loop_expansion_records(cname, mname) if _fe else []
            if _ast_loop_records:
                _ast_line_to_records = defaultdict(list)
                for _rec in _ast_loop_records:
                    _ast_line_to_records[_rec.get("line")].append(_rec)
                for _line_no in sorted(_ast_line_to_records):
                    for _rec in _ast_line_to_records[_line_no]:
                        _loop_kind = _rec.get("loop_kind")
                        _iter_expr = (_rec.get("iter_expr") or "").strip()
                        _iter_expr_node = _eval_expr_node(_iter_expr)
                        _kw_for_loop = None
                        if _loop_kind == "range":
                            n = _resolve_range_n(_iter_expr_node, fname, cname, mname)
                        elif _loop_kind == "enumerate":
                            n, _kw_for_loop = _resolve_iter_len(_iter_expr_node, fname, cname, mname)
                        else:
                            if _iter_expr.startswith('self.') or _iter_expr in ('range',):
                                continue
                            n, _kw_for_loop = _resolve_iter_len(_iter_expr_node, fname, cname, mname)
                        if n is None:
                            continue
                        cont = _rec.get("container_attr")
                        cls_ref = _rec.get("class_leaf")
                        if not cont or cls_ref not in nn_module_classes:
                            continue
                        if (_loop_kind in {"enumerate", "iter"}) and cont in dyn_indexed_containers:
                            continue
                        attrs.setdefault(cont, cls_ref)
                        attr_def_loc.setdefault((cname, cont), (fname, _line_no))
                        for k in range(n):
                            _ensure_elem(cont, cname, _elem_attr_list(cont, k), cls_ref, fname, _line_no, attrs)
                        _record_container_kind(cname, cont, "ModuleList")
                        if _kw_for_loop is not None:
                            class_container_kw[cname][cont] = _kw_for_loop
                continue

            # AST loop extraction should be the primary path. Keep a regex
            # fallback here for older sources whose loop bodies are not yet
            # covered by get_loop_expansion_records(), but route evaluation
            # through ConstantResolver only.
            _logical_method = _join_logical_lines(method_lines, ms)
            var_to_cls = {}
            i = 0
            while i < len(method_lines):
                raw = method_lines[i]
                _hdr = _parse_for_header(raw)
                if _hdr is None:
                    i += 1
                    continue
                _kind, base_indent, _iter_expr = _hdr
                _kw_for_loop = None
                if _kind == 'range':
                    n = _resolve_range_n(_eval_expr_node(_iter_expr), fname, cname, mname)
                elif _kind == 'enumerate':
                    n, _kw_for_loop = _resolve_iter_len(_eval_expr_node(_iter_expr), fname, cname, mname)
                else:  # 'iter'
                    if _iter_expr.startswith('self.') or _iter_expr in ('range',):
                        i += 1
                        continue
                    n, _kw_for_loop = _resolve_iter_len(_eval_expr_node(_iter_expr), fname, cname, mname)
                _logical_starts = {start_ln: text for (start_ln, text) in _logical_method}
                j = i + 1
                while j < len(method_lines):
                    body_raw = method_lines[j]
                    if not body_raw.strip():
                        j += 1
                        continue
                    ind = len(body_raw) - len(body_raw.lstrip())
                    if ind <= base_indent:
                        break
                    am = _parse_assign_ctor_line(body_raw)
                    if am is not None:
                        vname, cls_ref = am
                        if cls_ref in nn_module_classes:
                            var_to_cls[vname] = cls_ref
                    ap = _parse_append_var_line(body_raw)
                    ap_ctor = None
                    _phys_at_j = ms + j
                    _joined_text = _logical_starts.get(_phys_at_j)
                    if ap is None and _joined_text:
                        ap_ctor = _parse_append_ctor_line(' ' * (base_indent + 1) + _joined_text.lstrip())
                    if ap is None and ap_ctor is None:
                        ap_ctor = _parse_append_ctor_line(body_raw)
                    if ap is not None and n is not None:
                        cont, vname = ap
                        cls_ref = var_to_cls.get(vname)
                        if cls_ref and cls_ref in nn_module_classes:
                            attrs.setdefault(cont, cls_ref)
                            attr_def_loc.setdefault((cname, cont), (fname, ms + j))
                            for k in range(n):
                                _ensure_elem(cont, cname, _elem_attr_list(cont, k), cls_ref, fname, ms + j, attrs)
                            _record_container_kind(cname, cont, "ModuleList")
                            if _kw_for_loop is not None:
                                class_container_kw[cname][cont] = _kw_for_loop
                    elif ap_ctor is not None and n is not None and _kind in ('enumerate', 'iter'):
                        cont, cls_ref = ap_ctor
                        if cont in dyn_indexed_containers:
                            j += 1
                            continue
                        if cls_ref in nn_module_classes:
                            attrs.setdefault(cont, cls_ref)
                            attr_def_loc.setdefault((cname, cont), (fname, ms + j))
                            for k in range(n):
                                _ensure_elem(cont, cname, _elem_attr_list(cont, k), cls_ref, fname, ms + j, attrs)
                            _record_container_kind(cname, cont, "ModuleList")
                            if _kw_for_loop is not None:
                                class_container_kw[cname][cont] = _kw_for_loop
                    j += 1
                i = j

        class_attrs[cname] = attrs

    # ------------------------------------------------------------------
    # Iter11 unroll2 — Rule 2: range + f-string setattr expansion
    #   for i in range(N):                               # N from literal/CONST or
    #       setattr(self, f"query_tower_{i}", Cls(...))  # from ctor kw param
    # If N is resolvable, expand into N real attrs (e.g. query_tower_0..N-1).
    # Otherwise, fall back to a single wildcard synthetic attr (`<prefix>[*]`)
    # tracked in dynamic_attrs_per_class so getattr-based calls can wildcard-route.
    # This pass runs AFTER the main scan so `ctor_kw_int_args` is fully populated.
    # ------------------------------------------------------------------
    def _parse_for_range_var(_text: str):
        """For ``for <var> in range(<expr>):`` return (base_indent, var, expr_text).
        Else None."""
        if not _text:
            return None
        _stripped = _text.rstrip()
        if not _stripped.lstrip().startswith('for ') or not _stripped.endswith(':'):
            return None
        try:
            _mod = ast.parse(_stripped + "\n    pass")
        except SyntaxError:
            return None
        if not (_mod.body and isinstance(_mod.body[0], ast.For)):
            return None
        _f = _mod.body[0]
        if not isinstance(_f.target, ast.Name):
            return None
        _it = _f.iter
        if not (isinstance(_it, ast.Call)
                and isinstance(_it.func, ast.Name)
                and _it.func.id == 'range'
                and len(_it.args) >= 1):
            return None
        try:
            _expr = ast.unparse(_it.args[0])
        except Exception:
            _expr = ""
        _indent = len(_text) - len(_text.lstrip())
        return (_indent, _f.target.id, _expr)

    def _find_setattr_fstr_call(_body_line: str):
        """Find the first ``setattr(self, f"...", ClassName(...))`` call in a
        joined logical line. Returns (template_node, class_leaf, template_text)
        where template_node is the ast.JoinedStr literal and template_text is
        ast.unparse(template_node) (best effort). Returns None if not found.
        """
        if not _body_line:
            return None
        try:
            _mod = ast.parse(_body_line.strip())
        except SyntaxError:
            return None
        for _n in ast.walk(_mod):
            if not isinstance(_n, ast.Call):
                continue
            _func = _n.func
            if not (isinstance(_func, ast.Name) and _func.id == 'setattr'):
                continue
            if len(_n.args) < 3:
                continue
            _arg0, _arg1, _arg2 = _n.args[0], _n.args[1], _n.args[2]
            if not (isinstance(_arg0, ast.Name) and _arg0.id == 'self'):
                continue
            if not isinstance(_arg1, ast.JoinedStr):
                continue
            if not isinstance(_arg2, ast.Call):
                continue
            _vf = _arg2.func
            if isinstance(_vf, ast.Name):
                _leaf = _vf.id
            elif isinstance(_vf, ast.Attribute):
                _leaf = _vf.attr
            else:
                continue
            return (_arg1, _leaf)
        return None

    def _joinedstr_references_var(_jstr: ast.JoinedStr, var_name: str) -> bool:
        """True if any FormattedValue inside <_jstr> references the bare Name <var_name>."""
        for _v in _jstr.values:
            if isinstance(_v, ast.FormattedValue) and isinstance(_v.value, ast.Name) \
                    and _v.value.id == var_name:
                return True
        return False

    def _expand_fstring_template_ast(_jstr: ast.JoinedStr, loop_var: str, idx_value):
        """AST-based version of _expand_fstring_template.
        Substitutes references to <loop_var> in FormattedValue with idx_value.
        Returns None if any FormattedValue references something other than
        <loop_var>.
        Supports format spec ``[0]?<width>d`` for zero-padding.
        """
        out_parts = []
        for _v in _jstr.values:
            if isinstance(_v, ast.Constant) and isinstance(_v.value, str):
                out_parts.append(_v.value)
                continue
            if not isinstance(_v, ast.FormattedValue):
                return None
            if not (isinstance(_v.value, ast.Name) and _v.value.id == loop_var):
                return None
            # Format spec — only digit-width supported.
            _spec_text = None
            if _v.format_spec is not None:
                if not isinstance(_v.format_spec, ast.JoinedStr):
                    return None
                _parts = []
                for _sv in _v.format_spec.values:
                    if isinstance(_sv, ast.Constant) and isinstance(_sv.value, str):
                        _parts.append(_sv.value)
                    else:
                        return None
                _spec_text = ''.join(_parts)
            if _spec_text:
                # Match the original "0?<digits>d" semantics.
                _spec = _spec_text.strip()
                _digits = _spec[:-1] if _spec.endswith('d') else None
                if _digits is not None and _digits.startswith('0'):
                    _digits = _digits[1:]
                if _digits is not None and _digits.isdigit():
                    out_parts.append(str(idx_value).zfill(int(_digits)))
                else:
                    out_parts.append(str(idx_value))
            else:
                out_parts.append(str(idx_value))
        return ''.join(out_parts)

    def _wildcard_template_from_jstr(_jstr: ast.JoinedStr) -> str:
        """Replace every FormattedValue in <_jstr> with ``*`` and return the resulting string."""
        _parts = []
        for _v in _jstr.values:
            if isinstance(_v, ast.Constant) and isinstance(_v.value, str):
                _parts.append(_v.value)
            else:
                _parts.append('*')
        return ''.join(_parts)

    for (fname, cname), info in class_map.items():
        if cname not in nn_module_classes:
            continue
        lines = source_files.get(fname, [])
        attrs = class_attrs.get(cname, {})

        for mname, (ms, me) in info['methods'].items():
            method_lines = lines[ms - 1: min(me, len(lines))]
            i = 0
            while i < len(method_lines):
                raw = method_lines[i]

                _hdr = _parse_for_range_var(raw)
                if _hdr is None:
                    i += 1
                    continue
                base_indent, loop_var, range_expr = _hdr
                n = _resolve_range_n(_eval_expr_node(range_expr), fname, cname, mname)

                # Walk body and join multi-line statements (parenthesis-aware)
                # so that setattr(...) spanning multiple lines is matched as one.
                body_buf = ""
                body_buf_start = None
                body_open = 0
                joined = []  # list of (start_lineno, joined_line)
                j = i + 1
                while j < len(method_lines):
                    body_raw = method_lines[j]
                    if not body_raw.strip():
                        j += 1
                        continue
                    ind = len(body_raw) - len(body_raw.lstrip())
                    if not body_buf and ind <= base_indent:
                        break
                    stripped = _strip_inline_comment(body_raw.strip()).strip()
                    if not stripped:
                        j += 1
                        continue
                    if not body_buf:
                        body_buf_start = ms + j
                    body_buf += (" " if body_buf else "") + stripped
                    body_open += stripped.count('(') + stripped.count('[') + stripped.count('{')
                    body_open -= stripped.count(')') + stripped.count(']') + stripped.count('}')
                    if body_open <= 0:
                        joined.append((body_buf_start, body_buf))
                        body_buf = ""
                        body_buf_start = None
                        body_open = 0
                    j += 1
                if body_buf:
                    joined.append((body_buf_start, body_buf))

                for body_lineno, body_line in joined:
                    _hit = _find_setattr_fstr_call(body_line)
                    if _hit is None:
                        continue
                    template_jstr, cls_ref = _hit
                    if cls_ref not in nn_module_classes:
                        continue
                    # Skip f-strings that DO NOT reference the loop variable —
                    # those aren't this rule's domain (e.g. setattr(self, f"{name}_out", v)
                    # with `name` from a different loop). Let other passes / wildcards
                    # handle them.
                    if not _joinedstr_references_var(template_jstr, loop_var):
                        continue
                    if n is not None:
                        # Bounded expansion
                        for idx in range(n):
                            real = _expand_fstring_template_ast(template_jstr, loop_var, idx)
                            if real is None:
                                # Unsupported interpolation — fall back to wildcard
                                synth = _wildcard_template_from_jstr(template_jstr) or cls_ref
                                if synth not in attrs:
                                    attrs[synth] = cls_ref
                                    attr_def_loc.setdefault((cname, synth), (fname, body_lineno))
                                    dynamic_attrs_per_class[cname].add(synth)
                                break
                            if real not in attrs:
                                attrs[real] = cls_ref
                                attr_def_loc.setdefault((cname, real), (fname, body_lineno))
                    else:
                        # Unbounded — conservative wildcard node:
                        # replace {loop_var}/{loop_var:fmt} with '*'
                        synth = _wildcard_template_from_jstr(template_jstr) or cls_ref
                        if synth not in attrs:
                            attrs[synth] = cls_ref
                            attr_def_loc.setdefault((cname, synth), (fname, body_lineno))
                            dynamic_attrs_per_class[cname].add(synth)

                i = j

        class_attrs[cname] = attrs


    # Build data dependency edges between child modules
    dep_edges, dep_edge_locs, split_info = _build_data_dependency_edges(
        source_files,
        class_map,
        class_attrs,
        class_str_list_attrs,
        dynamic_attrs_per_class,
        ast_frontends=ast_frontends,
    )

    # Inject split nodes into class_attrs so they appear as DAG nodes.
    # e.g. if GAUBlock has split_info = {"norm": ["norm#0", "norm#1"]},
    # add "norm#0" and "norm#1" to class_attrs["GAUBlock"] with the same class ref as "norm".
    for cname, smap in split_info.items():
        if cname not in class_attrs:
            continue
        for orig_attr, split_names in smap.items():
            cls_ref = class_attrs[cname].get(orig_attr)
            if cls_ref is None:
                continue
            # Remove the original attr from class_attrs (replaced by splits)
            del class_attrs[cname][orig_attr]
            for sn in split_names:
                class_attrs[cname][sn] = cls_ref

    # Build tree using forward() call order + __init__ attrs
    tree = {}
    all_child_classes = set()
    all_tree_classes = set(nn_module_classes) | set(torch_native_module_classes)
    for cname in all_tree_classes:
        if cname not in class_attrs:
            class_attrs[cname] = {}
        attrs = class_attrs[cname]
        call_order = module_call_order.get(cname, [])

        # Order children by forward() call order
        children_ordered = []
        seen = set()
        for attr in call_order:
            if attr in attrs and attrs[attr] not in seen:
                children_ordered.append(attrs[attr])
                seen.add(attrs[attr])
                all_child_classes.add(attrs[attr])
        # Add remaining attrs not called in forward
        for attr, cls_ref in attrs.items():
            if cls_ref not in seen:
                children_ordered.append(cls_ref)
                seen.add(cls_ref)
                all_child_classes.add(cls_ref)

        tree[cname] = {"children": children_ordered, "attrs": attrs,
                       "dep_edges": dep_edges.get(cname, []),
                       "dep_edge_locs": dep_edge_locs.get(cname, {}),
                       "containers": {},
                       "is_torch_native": cname in torch_native_module_classes,
                       # Iter16: expose dynamic setattr-registered attrs so
                       # downstream scanners (e.g. _scan_root_result_edges)
                       # can fall back to wildcard producers when seeing
                       # `getattr(self, <non-literal>)`.
                       "dynamic_setattr_attrs": sorted(dynamic_attrs_per_class.get(cname, set())),
                       # Iter17: per-class container -> kw param driving its
                       # iteration length. Used by build_dag_recursive for
                       # per-instance children pruning.
                       "container_kw": dict(class_container_kw.get(cname, {}))}

    # Find root(s): classes that are NOT children of any other class
    roots = [c for c in nn_module_classes if c not in all_child_classes and c in tree]
    if not roots:
        roots = list(tree.keys())[:1]

    # Prefer roots that have children (real model roots), sort by descendant count
    def count_descendants(cname, visited=None):
        if visited is None:
            visited = set()
        if cname in visited or cname not in tree:
            return 0
        visited.add(cname)
        total = len(tree[cname]["children"])
        for child in tree[cname]["children"]:
            total += count_descendants(child, visited)
        return total

    roots_with_children = [r for r in roots if tree.get(r, {}).get("children")]
    if roots_with_children:
        roots = sorted(roots_with_children, key=lambda r: -count_descendants(r))
    # If "RootModule" is present, put it first
    if "RootModule" in roots:
        roots = ["RootModule"] + [r for r in roots if r != "RootModule"]
    # Honour caller-supplied preferred root (e.g. derived from timeline wrapper name)
    if preferred_root and preferred_root in roots:
        roots = [preferred_root] + [r for r in roots if r != preferred_root]

    # Filter tree to only include classes reachable from the primary root.
    # This eliminates orphan classes that are defined but never instantiated.
    if roots:
        reachable = set()
        def _collect_reachable(cname):
            if cname in reachable or cname not in tree:
                return
            reachable.add(cname)
            for child in tree[cname].get("children", []):
                _collect_reachable(child)
        for r in roots[:1]:  # Only from primary root
            _collect_reachable(r)
        # Keep only reachable classes in tree, filter roots
        tree = {k: v for k, v in tree.items() if k in reachable}
        roots = [r for r in roots if r in reachable]
        if not roots and reachable:
            roots = sorted(reachable, key=lambda r: -count_descendants(r))[:1]

    # Stash attr definition locations on the tree for downstream consumers
    # (HTML flowchart click-interactions). Format: tree[cname]["attr_def_loc"][attr] = (fname, lineno)
    for cname in tree:
        per_class = {}
        for attr in tree[cname].get("attrs", {}):
            loc = attr_def_loc.get((cname, attr))
            if loc:
                per_class[attr] = loc
        tree[cname]["attr_def_loc"] = per_class

    # Build first_call_loc: first forward() call line for each attr.
    # Strategy 1: derive from dep_edge_locs (edges between siblings).
    # Strategy 2: ask ASTFrontend for the earliest reachable call site,
    # including dynamic getattr(self, f"...") calls via pure AST walk.
    for cname in tree:
        edge_locs_c = tree[cname].get("dep_edge_locs", {})
        first_call = {}  # attr -> (file, lineno)
        for (from_a, to_a), ev in edge_locs_c.items():
            if not isinstance(ev, dict):
                continue
            f = ev.get("file", "")
            if from_a and ev.get("from_line"):
                if from_a not in first_call or ev["from_line"] < first_call[from_a][1]:
                    first_call[from_a] = (f, ev["from_line"])
            if to_a and ev.get("to_line"):
                if to_a not in first_call or ev["to_line"] < first_call[to_a][1]:
                    first_call[to_a] = (f, ev["to_line"])
        known_attrs_c = set(tree[cname].get("attrs", {}).keys())
        _ast_first_call_map = {}
        for (fname, cm_name), cm_info in class_map.items():
            if cm_name != cname:
                continue
            _fe = _get_ast_frontend(fname)
            if _fe:
                for _attr_name in known_attrs_c:
                    _loc = _fe.get_first_call_loc(cname, _attr_name)
                    if _loc:
                        _ast_first_call_map[_attr_name] = _loc
            break
        for _attr_name, _loc in _ast_first_call_map.items():
            if _attr_name not in first_call or _loc[1] < first_call[_attr_name][1]:
                first_call[_attr_name] = _loc
        tree[cname]["first_call_loc"] = first_call

    # Stash input_source_attrs on the tree for downstream consumers
    for cname in tree:
        tree[cname]["input_source_attrs"] = input_source_attrs.get(cname, set())
        for (fname, cm_name), _cm_info in class_map.items():
            if cm_name != cname:
                continue
            _fe = _get_ast_frontend(fname)
            if _fe:
                result_attrs = _fe.get_result_attrs(cname)
                if result_attrs:
                    tree[cname]["result_attrs"] = result_attrs
            break
        tree[cname]["static_diagnostics"] = list(_new_eval_diagnostics)

    # Iter14 container-group: stash recorded container kinds on the tree so the
    # DAG renderer can build container parent groups labelled e.g.
    # ``ModuleDict[tokens_module_dict][64]``.  Format:
    # tree[cname]["container_kinds"][container_attr] = "ModuleDict" / "ModuleList" / "dict" / "list"
    for cname in tree:
        per_class = {}
        for (kc, ka), kind in container_kinds.items():
            if kc == cname:
                per_class[ka] = kind
        tree[cname]["container_kinds"] = per_class

    # Iter13 Step1: stash conditional attrs metadata on the tree for caller's
    # tab decision. Filter to attrs where BOTH branches actually recorded a
    # class (single-branch conditionals are not "real" conditional attrs).
    for cname in tree:
        ca = class_conditional_attrs.get(cname, {})
        meaningful = {a: info for a, info in ca.items()
                      if info.get("train_class") and info.get("infer_class")
                      and info.get("train_class") != info.get("infer_class")}
        tree[cname]["conditional_attrs"] = meaningful
    # Iter13 Step2: stash unresolved moduledict warnings on the tree.
    for cname in tree:
        tree[cname]["unresolved_moduledict"] = [u for u in unresolved_moduledict
                                                 if u.get("class") == cname]

    # Iter17: stash per-instance kw-list-lens on each parent class so
    # build_dag_recursive can prune children counts per call site.
    for cname in tree:
        tree[cname]["instance_kw_list_lens"] = {
            attr: dict(kws) for (pcn, attr), kws in instance_kw_list_lens.items()
            if pcn == cname
        }

    # Iter18: stash AST ground-truth lookups on every tree[cname] entry.
    # This is purely additive metadata; the existing regex pass remains the
    # primary driver of edges/expansions.  Downstream consumers can use these
    # to:
    #   • cross-check ModuleList/ModuleDict element type
    #     (``_ast_container_elems[container_attr] -> [child_class, ...]``)
    #   • confirm dynamic setattr child class lists
    #   • spot missing or spurious attrs vs the existing class_attrs
    #
    # Sibling keys are prefixed with ``_ast_`` to avoid clashing with the
    # existing keys consumed by generate_html_flowchart.  We do NOT add
    # synthetic top-level keys to ``tree`` itself (that would break the many
    # ``for cname in tree`` loops downstream).
    if _ast_extractor is not None:
        for cname in tree:
            tree[cname]["_ast_class_attrs"] = dict(_ast_extractor.ast_class_attrs.get(cname, {}))
            tree[cname]["_ast_container_elems"] = {
                k: list(v) for k, v in _ast_extractor.ast_container_elems.get(cname, {}).items()
            }
            tree[cname]["_ast_container_kinds"] = dict(_ast_extractor.ast_container_kinds.get(cname, {}))
            tree[cname]["_ast_dynamic_attrs"] = list(_ast_extractor.ast_dynamic_attrs.get(cname, []))
            tree[cname]["_ast_is_nn_module"] = (cname in _ast_extractor.nn_module_classes)

    return tree, roots, class_map


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
    print(f"  📋 Timeline nn.Module 验证:")
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
    user_source_in_timeline_missing = []
    # We can only detect this with extra info; not done here to avoid import churn.
    # Validation output above is the authoritative pass/fail summary.

    return validation


# ---------------------------------------------------------------------------
# HTML flowchart rendering moved to frontend_html.py.
#
# Historically this file embedded a ~4800 line copy of the HTML/JS template
# generator (lines 7524..12306 in the iter17_timing baseline). That copy has
# been removed in favour of an explicit, one-direction import from the
# sibling ``frontend_html`` module:
#
#     analyze_trace  ──provides─▶  ASTFrontend, _build_class_map, ...
#     frontend_html  ──provides─▶  generate_html_flowchart, *_dual,
#                                  build_timing_data_from_trace,
#                                  _generate_flowchart_html
#
# The four names are re-exported through a module-level ``__getattr__``
# (PEP 562) instead of an eager ``from frontend_html import ...`` so the
# import remains safe even when a downstream caller imports
# ``frontend_html`` first (in which case Python is in the middle of loading
# ``frontend_html`` when this file finishes executing — an eager import
# back into ``frontend_html`` would deadlock with a "partially initialized
# module" error).
#
# The lazy form preserves the public surface area used by ``main()`` below
# as well as by ``testset/test_dag_rules.py`` (which calls
# ``mod.generate_html_flowchart(...)`` after loading this file via
# ``importlib.util.spec_from_file_location``) — both call sites resolve
# the attribute *after* both modules have finished loading, so the lazy
# lookup always sees a fully-initialized ``frontend_html``.
# ---------------------------------------------------------------------------
_FRONTEND_HTML_REEXPORT_NAMES = (
    "generate_html_flowchart",
    "generate_html_flowchart_dual",
    "build_timing_data_from_trace",
    "_generate_flowchart_html",
)


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
    # Lazy, function-scope import of the HTML rendering entry points.  See the
    # comment above ``_FRONTEND_HTML_REEXPORT_NAMES`` for why module-level
    # eager imports would risk a circular-import deadlock.  By the time
    # ``main()`` is invoked both modules are fully loaded.
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
        print(f"  如确认是训练 trace，将继续分析")

    # Detect enhanced trace (Code Location + stack_traces)
    has_code_loc, has_stack_traces = detect_enhanced_trace(events)
    is_enhanced = has_code_loc and has_stack_traces
    if is_enhanced:
        print(f"  检测到增强 trace（含 Code Location 和 stack_traces）")
        if not args.code_path:
            print(f"  ❌ 错误: 增强 trace 需要提供模型源码才能进行源码级分析")
            print(f"  请通过 --code-path 参数提供模型源码目录或 tar.gz 压缩包")
            print(f"  示例: python3 scripts/analyze_trace.py trace.json --code-path code_commit.tar.gz")
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
    thread_indices = add_func_call_parent(events)

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
