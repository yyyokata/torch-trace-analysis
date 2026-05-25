#!/usr/bin/env python3
"""
fake_runner.py — Dynamic information collector for PyTorch model analysis.

Provides FakeTensor-based dynamic module tree and call sequence collection
to supplement and calibrate the static DAG analysis in analyze_trace.py.

Three phases:
  Phase 1: Module tree extraction via named_modules() (no FakeTensor needed)
  Phase 2: FakeTensor call sequence capture via __call__ hooks
  Phase 3: Merge interface design (specification only)

Usage:
  python3 fake_runner.py --code-path <model_source_dir> [--output-dir <dir>]
  python3 fake_runner.py --code-path testset/extracted/5698781/modelcode/ --output-dir fake_runner_output
"""

import argparse
import ast
import json
import os
import sys
import traceback
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# ---------------------------------------------------------------------------
# Phase 1: Static Module Tree Extraction (AST-based, no imports needed)
# ---------------------------------------------------------------------------

class ModuleTreeExtractor:
    """Extract nn.Module tree from source code using AST analysis.
    
    This works without importing the model (no lagrange_torch dependency).
    It parses __init__ methods to find self.attr = SomeModule(...) patterns
    and builds the full module hierarchy.
    """

    def __init__(self, code_dir: str):
        self.code_dir = Path(code_dir)
        self.classes: Dict[str, dict] = {}  # class_name -> {file, node, bases, ...}
        self.module_tree: List[dict] = []
        self._parse_all_files()

    def _parse_all_files(self):
        for py_file in self.code_dir.rglob("*.py"):
            try:
                source = py_file.read_text(encoding="utf-8", errors="replace")
                tree = ast.parse(source, filename=str(py_file))
                for node in ast.walk(tree):
                    if isinstance(node, ast.ClassDef):
                        self.classes[node.name] = {
                            "file": str(py_file.relative_to(self.code_dir)),
                            "node": node,
                            "bases": [self._get_base_name(b) for b in node.bases],
                        }
            except SyntaxError:
                continue

    def _get_base_name(self, node) -> str:
        if isinstance(node, ast.Name):
            return node.id
        elif isinstance(node, ast.Attribute):
            return f"{self._get_base_name(node.value)}.{node.attr}"
        return ""

    def _is_nn_module_class(self, class_name: str) -> bool:
        if class_name not in self.classes:
            return False
        bases = self.classes[class_name]["bases"]
        for b in bases:
            if b in ("nn.Module", "Module", "torch.nn.Module"):
                return True
            if self._is_nn_module_class(b):
                return True
        return False

    def _find_init_assignments(self, class_name: str) -> List[dict]:
        """Find all self.attr = Something(...) in __init__."""
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
            # local_var = SomeClass(...)
            if isinstance(stmt, ast.Assign):
                for target in stmt.targets:
                    self._check_self_assign(target, stmt.value, assignments)
                    # Track local variable for append resolution
                    if isinstance(target, ast.Name):
                        cls, _, _ = self._infer_class_from_value(stmt.value)
                        if cls:
                            local_vars[target.id] = cls
            # self.attr.append(SomeClass(...))
            elif isinstance(stmt, ast.Expr) and isinstance(stmt.value, ast.Call):
                call = stmt.value
                # setattr(self, name, SomeClass(...))
                if isinstance(call.func, ast.Name) and call.func.id == "setattr":
                    self._check_setattr(call, assignments)
                # self.container.append(SomeClass(...) or local_var)
                elif isinstance(call.func, ast.Attribute) and call.func.attr == "append":
                    self._check_container_append(call, assignments, local_vars)
            # for ... in ...: self.attr = ...
            elif isinstance(stmt, ast.For):
                self._walk_init_body(stmt.body, assignments, parent_class, local_vars)
            # if ...: self.attr = ...
            elif isinstance(stmt, (ast.If, ast.With)):
                body = stmt.body if hasattr(stmt, "body") else []
                orelse = stmt.orelse if hasattr(stmt, "orelse") else []
                self._walk_init_body(body, assignments, parent_class, local_vars)
                self._walk_init_body(orelse, assignments, parent_class, local_vars)

    def _check_container_append(self, call_node, assignments, local_vars=None):
        """Handle self.container.append(SomeClass(...)) - record element type."""
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
        # If the argument is just a variable name, try to resolve via local_vars
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

    def _check_setattr(self, call_node, assignments):
        if not isinstance(call_node.func, ast.Name):
            return
        if call_node.func.id != "setattr":
            return
        if len(call_node.args) < 3:
            return
        obj, name_arg, value_arg = call_node.args[0], call_node.args[1], call_node.args[2]
        if not (isinstance(obj, ast.Name) and obj.id == "self"):
            return
        # Try to get the attr name (may be dynamic)
        attr_name = None
        if isinstance(name_arg, ast.Constant) and isinstance(name_arg.value, str):
            attr_name = name_arg.value
        else:
            attr_name = "<dynamic>"
        class_name, is_container, container_type = self._infer_class_from_value(value_arg)
        if class_name or is_container:
            assignments.append({
                "attr": attr_name,
                "class": class_name,
                "is_container": is_container,
                "container_type": container_type,
                "line": getattr(call_node, "lineno", 0),
                "dynamic": attr_name == "<dynamic>",
            })

    def _infer_class_from_value(self, value) -> Tuple[str, bool, str]:
        """Return (class_name, is_container, container_type)."""
        if isinstance(value, ast.Call):
            func = value.func
            func_name = ""
            if isinstance(func, ast.Name):
                func_name = func.id
            elif isinstance(func, ast.Attribute):
                func_name = func.attr
                # Check for nn.ModuleList, nn.ModuleDict etc
                if isinstance(func.value, ast.Name) and func.value.id == "nn":
                    func_name = f"nn.{func.attr}"
            if func_name in ("ModuleList", "nn.ModuleList"):
                return ("", True, "ModuleList")
            elif func_name in ("ModuleDict", "nn.ModuleDict"):
                return ("", True, "ModuleDict")
            elif func_name in ("Sequential", "nn.Sequential"):
                return ("", True, "Sequential")
            # Skip non-Module types
            non_module_types = {"OrderedDict", "dict", "list", "set", "tuple", 
                                "defaultdict", "Counter", "deque"}
            if func_name in non_module_types:
                return ("", False, "")
            # Regular class instantiation
            if func_name and func_name[0].isupper():
                return (func_name, False, "")
            if isinstance(func, ast.Attribute) and func.attr[0].isupper():
                return (func.attr, False, "")
        return ("", False, "")

    def build_tree(self, root_class: str, root_path: str = "") -> List[dict]:
        """Recursively build the module tree starting from root_class."""
        self.module_tree = []
        self._build_recursive(root_class, root_path, depth=0)
        return self.module_tree

    def _build_recursive(self, class_name: str, prefix: str, depth: int, ancestor_classes=None):
        if ancestor_classes is None:
            ancestor_classes = set()
        # Prevent infinite recursion only on direct ancestor chain (not siblings)
        if class_name in ancestor_classes:
            return
        new_ancestors = ancestor_classes | {class_name}

        assignments = self._find_init_assignments(class_name)
        for assign in assignments:
            attr = assign["attr"]
            path = f"{prefix}.{attr}" if prefix else attr
            child_class = assign["class"]
            entry = {
                "attr_path": path,
                "class": child_class if child_class else assign.get("container_type", "Unknown"),
                "depth": depth + 1,
                "is_container": assign["is_container"],
                "container_type": assign.get("container_type", ""),
                "parent_class": class_name,
                "file": self.classes.get(class_name, {}).get("file", ""),
                "line": assign.get("line", 0),
                "dynamic": assign.get("dynamic", False),
            }
            self.module_tree.append(entry)
            # Recurse into child if it's a known nn.Module class
            if child_class and child_class in self.classes and self._is_nn_module_class(child_class):
                self._build_recursive(child_class, path, depth + 1, new_ancestors)


# ---------------------------------------------------------------------------
# Phase 2: FakeTensor Call Sequence Capture
# ---------------------------------------------------------------------------

class FakeTensorRunner:
    """Run the model via MockRuntime by subprocess-executing main_model_mock.py.

    Phase 2 strategy:
      - Requires the real lagrange_torch Docker environment (MockRuntime available).
      - Executes:  TORCH_TRACE=<trace_dir> BATCH_SIZE=<n> python3 main_model_mock.py
        inside the model code directory, capturing stdout/stderr.
      - Collects the resulting TORCH_TRACE JSON/gz files from trace_dir.
      - Does NOT use any hand-written mock stubs — those are intentionally removed.

    collect_module_tree_live / collect_call_sequence remain for in-process use
    when the caller already has a live model instance (e.g. inside the Docker env).
    """

    def __init__(self, code_dir: str, batch_size: int = 1024):
        self.code_dir = Path(code_dir)
        self.batch_size = batch_size
        self.call_sequence: List[dict] = []
        self.module_tree_live: List[dict] = []
        self._tensor_id_counter = 0
        self._tensor_id_map: Dict[int, int] = {}

    def _get_tensor_id(self, t) -> int:
        """Assign a stable ID to a tensor based on id() (FakeTensor-safe)."""
        import torch
        if isinstance(t, torch.Tensor):
            key = id(t)
            if key not in self._tensor_id_map:
                self._tensor_id_map[key] = self._tensor_id_counter
                self._tensor_id_counter += 1
            return self._tensor_id_map[key]
        return -1

    def _extract_tensor_info(self, args) -> Tuple[List[int], List[List[int]]]:
        """Extract tensor IDs and shapes from args (may be nested)."""
        import torch
        ids: List[int] = []
        shapes: List[List[int]] = []
        if isinstance(args, torch.Tensor):
            ids.append(self._get_tensor_id(args))
            shapes.append(list(args.shape))
        elif isinstance(args, (tuple, list)):
            for item in args:
                sub_ids, sub_shapes = self._extract_tensor_info(item)
                ids.extend(sub_ids)
                shapes.extend(sub_shapes)
        elif isinstance(args, dict):
            for v in args.values():
                sub_ids, sub_shapes = self._extract_tensor_info(v)
                ids.extend(sub_ids)
                shapes.extend(sub_shapes)
        return ids, shapes

    def run_mock_and_collect_trace(
        self,
        trace_output_dir: str,
        model_name: str = "5698781",
        timeout: int = 300,
        extra_env: Optional[Dict] = None,
    ) -> Tuple[bool, str, List[str]]:
        """Run main_model_mock.py via subprocess and collect TORCH_TRACE output.

        Must be called inside the lagrange_torch Docker environment where
        MockRuntime and all model dependencies are available.

        Parameters
        ----------
        trace_output_dir : str
            Directory where TORCH_TRACE files will be written.
            Passed as the TORCH_TRACE environment variable.
        model_name : str
            MODEL_NAME env var value (default: "5698781").
        timeout : int
            Subprocess timeout in seconds (default: 300).
        extra_env : dict, optional
            Additional environment variables to set.

        Returns
        -------
        (success, log_output, trace_files)
            success      — True if subprocess exited 0
            log_output   — combined stdout+stderr string
            trace_files  — list of .json/.gz trace file paths found in trace_output_dir
        """
        import subprocess

        mock_script = (self.code_dir / "main_model_mock.py").resolve()
        if not mock_script.exists():
            return False, f"main_model_mock.py not found at {mock_script}", []

        trace_dir = Path(trace_output_dir)
        trace_dir.mkdir(parents=True, exist_ok=True)

        env = os.environ.copy()
        env["DO_COMPILE"] = "0"  # 关闭 torch.compile，mock 运行直接走 eager 模式，更快更稳定
        env["TORCH_COMPILE_USE_LAZY_GRAPH_MODULE"] = "0"  # 禁用 LazyGraphModule，确保图模块立即实例化，便于 trace 捕获
        env["TORCHDYNAMO_VERBOSE"] = "1"  # 开启 TorchDynamo 详细日志，输出 graph break / recompile 原因，便于调试
        env["INDUCTOR_PROVENANCE"] = "1"  # Inductor 在生成的 C++/Triton 内核中注入 provenance 注释（来源行号），便于溯源
        env["TORCH_TRACE"] = str(trace_dir)  # 输出 torch.profiler / dynamo trace JSON 文件的目录
        env["MODEL_NAME"] = model_name  # 模型名称标识（由各模型自行决定是否必须，此处设默认值 "5698781"）
        env["BATCH_SIZE"] = str(self.batch_size)  # mock 运行时的 batch 大小

        if extra_env:
            env.update(extra_env)

        cmd = [sys.executable, str(mock_script)]
        print(f"  Running: {' '.join(f'{k}={v}' for k, v in [('TORCH_TRACE', str(trace_dir)), ('BATCH_SIZE', str(self.batch_size)), ('MODEL_NAME', model_name)])} {' '.join(cmd)}")

        try:
            result = subprocess.run(
                cmd,
                cwd=str(self.code_dir),
                env=env,
                capture_output=True,
                text=True,
                timeout=timeout,
            )
            log = result.stdout + result.stderr
            success = result.returncode == 0
        except subprocess.TimeoutExpired as e:
            return False, f"Subprocess timed out after {timeout}s: {e}", []
        except Exception as e:
            return False, f"Subprocess failed to start: {e}", []

        # Collect trace files
        trace_files = sorted(
            str(p) for p in trace_dir.rglob("*")
            if p.is_file() and (p.suffix in (".json", ".gz") or "trace" in p.name.lower())
        )

        return success, log, trace_files

    def collect_module_tree_live(self, model) -> List[dict]:
        """Collect precise module tree from a live model instance."""
        import torch.nn as nn
        tree = []
        for name, mod in model.named_modules():
            if name == "":
                continue
            tree.append({
                "attr_path": name,
                "class": type(mod).__name__,
                "depth": name.count(".") + 1,
                "num_parameters": sum(p.numel() for p in mod.parameters(recurse=False)),
                "num_children": len(list(mod.children())),
                "is_leaf": len(list(mod.children())) == 0,
            })
        self.module_tree_live = tree
        return tree

    def collect_call_sequence(self, model, fake_inputs: Optional[Dict] = None) -> List[dict]:
        """Hook nn.Module.__call__ and collect call sequence with tensor shapes."""
        import torch
        import torch.nn as nn

        self.call_sequence = []
        self._tensor_id_counter = 0
        self._tensor_id_map = {}

        # Build path map
        path_map = {}
        for name, mod in model.named_modules():
            path_map[id(mod)] = name if name else "<root>"

        hooks = []
        call_order = [0]

        def make_hook(module_path):
            def hook_fn(module, input, output):
                in_ids, in_shapes = self._extract_tensor_info(input)
                out_ids, out_shapes = self._extract_tensor_info(output)
                self.call_sequence.append({
                    "order": call_order[0],
                    "module_path": module_path,
                    "class": type(module).__name__,
                    "input_tensor_ids": in_ids,
                    "output_tensor_ids": out_ids,
                    "input_shapes": in_shapes,
                    "output_shapes": out_shapes,
                })
                call_order[0] += 1
            return hook_fn

        for name, mod in model.named_modules():
            path = name if name else "<root>"
            h = mod.register_forward_hook(make_hook(path))
            hooks.append(h)

        try:
            # Try FakeTensorMode
            try:
                from torch._subclasses import FakeTensorMode
                with FakeTensorMode():
                    # Create fake input
                    fake_input = torch.randn(self.batch_size, 1)
                    model(fake_input)
            except Exception as e:
                # Fallback: try direct call with real tensors
                model.eval()
                with torch.no_grad():
                    model()
        except Exception as e:
            # Record the failure
            self.call_sequence.append({
                "error": str(e),
                "traceback": traceback.format_exc(),
            })
        finally:
            for h in hooks:
                h.remove()

        return self.call_sequence


# ---------------------------------------------------------------------------
# Phase 3: Merge Interface Design
# ---------------------------------------------------------------------------

def merge_dynamic_info(static_dag: dict, dynamic_info: dict) -> dict:
    """Merge dynamic runtime info into static DAG for calibration.
    
    Interface specification (not fully implemented):
    
    Parameters
    ----------
    static_dag : dict
        The DAG output from analyze_trace.py with keys:
        - nodes: list of node dicts
        - edges: list of edge dicts  
        - groups: dict of group definitions
        - metadata: analysis metadata
    
    dynamic_info : dict
        Dynamic info collected by fake_runner.py with keys:
        - module_tree: list from Phase 1 (precise module hierarchy)
        - call_sequence: list from Phase 2 (runtime call order + tensor flow)
    
    Returns
    -------
    dict : Calibrated DAG with:
        - Corrected ModuleList expansion counts
        - Removed spurious edges (no tensor_id dependency)
        - Added tensor shape annotations
        - Confidence scores on edges (static_only vs dynamic_confirmed)
    
    Calibration Rules
    -----------------
    1. ModuleList expansion: Use dynamic module_tree to get exact count of 
       children (e.g., DenseTower._layers has 3 elements, not 1)
    2. False edge removal: If static DAG has edge A→B but call_sequence shows
       no tensor_id overlap between A's outputs and B's inputs, flag as suspect
    3. Shape annotation: Add input_shapes/output_shapes from call_sequence to nodes
    4. Parallel vs sequential: If A and B share same input_tensor_ids but have 
       no output→input dependency, they are parallel (not sequential)
    """
    # Phase 3 is design-only; implementation deferred
    calibrated = dict(static_dag)
    
    if "module_tree" in dynamic_info:
        calibrated["_dynamic_module_tree"] = dynamic_info["module_tree"]
    if "call_sequence" in dynamic_info:
        calibrated["_dynamic_call_sequence"] = dynamic_info["call_sequence"]
    
    calibrated["_calibration_status"] = "interface_designed_not_implemented"
    return calibrated


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="FakeTensor-based dynamic module info collector")
    parser.add_argument("--code-path", required=True, help="Path to model source code directory")
    parser.add_argument("--output-dir", default="fake_runner_output", help="Output directory")
    parser.add_argument("--batch-size", type=int, default=1024, help="Batch size for FakeTensor run")
    parser.add_argument("--root-class", default=None, help="Root model class name (auto-detected if omitted)")
    parser.add_argument("--model-name", default="5698781", help="MODEL_NAME env var")
    args = parser.parse_args()

    code_path = Path(args.code_path)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    results = {
        "phase1": {"status": "not_started"},
        "phase2": {"status": "not_started"},
    }

    # ─── Phase 1: AST-based module tree ──────────────────────────────────
    print("=" * 60)
    print("Phase 1: AST-based Module Tree Extraction")
    print("=" * 60)

    extractor = ModuleTreeExtractor(str(code_path))
    
    # Auto-detect root class
    root_class = args.root_class
    if not root_class:
        # Heuristic: find the class that references the most other nn.Module classes
        # (the root model typically imports/uses many sub-modules)
        all_module_classes = set(
            cname for cname in extractor.classes
            if extractor._is_nn_module_class(cname)
        )
        candidates = []
        for cname in all_module_classes:
            assigns = extractor._find_init_assignments(cname)
            # Count how many distinct module classes are referenced
            referenced_classes = set()
            for a in assigns:
                if a["class"] in all_module_classes:
                    referenced_classes.add(a["class"])
            # Score = distinct module refs + total assigns (prefer wider trees)
            score = len(referenced_classes) * 10 + len(assigns)
            candidates.append((cname, score, len(assigns)))
        if candidates:
            candidates.sort(key=lambda x: -x[1])
            root_class = candidates[0][0]
            print(f"  Auto-detected root class: {root_class} (score={candidates[0][1]}, {candidates[0][2]} sub-module assignments)")

    if root_class:
        tree = extractor.build_tree(root_class)
        # Save module_tree.json
        tree_path = output_dir / "module_tree.json"
        with open(tree_path, "w") as f:
            json.dump(tree, f, indent=2, ensure_ascii=False)
        print(f"  Module tree: {len(tree)} entries written to {tree_path}")
        results["phase1"] = {
            "status": "success",
            "root_class": root_class,
            "total_entries": len(tree),
            "output_file": str(tree_path),
        }
        # Print first 20 entries
        print(f"\n  First 20 entries:")
        for entry in tree[:20]:
            indent = "  " * entry["depth"]
            cls = entry["class"]
            dynamic_tag = " [DYNAMIC]" if entry.get("dynamic") else ""
            container_tag = f" ({entry['container_type']})" if entry["is_container"] else ""
            print(f"    {indent}{entry['attr_path']}: {cls}{container_tag}{dynamic_tag}")
    else:
        print("  ERROR: Could not detect root class")
        results["phase1"] = {"status": "failed", "error": "No root class detected"}

    # ─── Phase 2: FakeTensor Call Sequence Capture ───────────────────────
    print("\n" + "=" * 60)
    print("Phase 2: FakeTensor Call Sequence Capture")
    print("=" * 60)

    runner = FakeTensorRunner(str(code_path), batch_size=args.batch_size)
    trace_dir = output_dir / "torch_trace"
    success, log_output, trace_files = runner.run_mock_and_collect_trace(
        trace_output_dir=str(trace_dir),
        model_name=args.model_name,
    )
    if success:
        print(f"  MockRunner subprocess exited successfully.")
        print(f"  Trace files collected: {len(trace_files)}")
        for f in trace_files:
            print(f"    {f}")
        results["phase2"] = {
            "status": "success",
            "trace_dir": str(trace_dir),
            "trace_files": trace_files,
        }
    else:
        print(f"  MockRunner subprocess FAILED. Last 2000 chars of log:")
        print(f"    {log_output[-2000:]}")
        results["phase2"] = {
            "status": "failed",
            "trace_dir": str(trace_dir),
            "log_tail": log_output[-2000:],
            "trace_files": trace_files,
        }

    # ─── Summary ─────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)
    summary_path = output_dir / "summary.json"
    with open(summary_path, "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"  Results saved to {summary_path}")
    print(f"  Phase 1: {results['phase1']['status']}")
    print(f"  Phase 2: {results['phase2']['status']}")

    return results


if __name__ == "__main__":
    main()
