import ast
import os
from typing import List, Dict, Optional, Set
from .attr_types import CallLoc, InputAttr, ResultAttr

class AttrScanner(ast.NodeVisitor):
    def __init__(self, file_path: str):
        self.file_path = file_path
        self.current_class: Optional[str] = None
        self.inputs: Dict[str, InputAttr] = {}  # attr_name -> InputAttr
        self.results: List[ResultAttr] = []
        
        # Internal state to track potential LG sources before finding usage
        # (class_name, attr_name) -> (call_loc, kind)
        self._potential_inputs: Dict[tuple[Optional[str], str], tuple[CallLoc, str]] = {}

        # LG function names that define input sources
        self.lg_funcs = {
            "dense_feature", "feature_column", "label", "get_sample_rate", 
            "get_bias", "slot", "global_step", "get_sample_bias", "group_candidate_size"
        }

    def _get_call_loc(self, node: ast.AST) -> CallLoc:
        return CallLoc(file=self.file_path, line=node.lineno, col=node.col_offset)

    def visit_ClassDef(self, node: ast.ClassDef):
        old_class = self.current_class
        self.current_class = node.name
        self.generic_visit(node)
        self.current_class = old_class

    def visit_Assign(self, node: ast.Assign):
        # Handle self.attr = LG.func(...) or VAR = LG.func(...)
        if not isinstance(node.value, ast.Call):
            self.generic_visit(node)
            return

        call = node.value
        # Check if it's LG.func(...)
        kind = self._get_lg_func_kind(call)
        if not kind:
            self.generic_visit(node)
            return

        for target in node.targets:
            attr_name = None
            if isinstance(target, ast.Attribute) and isinstance(target.value, ast.Name) and target.value.id == "self":
                attr_name = target.attr
            elif isinstance(target, ast.Name):
                attr_name = target.id
            
            if attr_name:
                loc = self._get_call_loc(node)
                self._potential_inputs[(self.current_class, attr_name)] = (loc, kind)
        
        self.generic_visit(node)

    def _get_lg_func_kind(self, call: ast.Call) -> Optional[str]:
        # Check for LG.func(...)
        func = call.func
        if isinstance(func, ast.Attribute):
            if isinstance(func.value, ast.Name) and func.value.id == "LG":
                if func.attr in self.lg_funcs:
                    return func.attr
        return None

    def visit_Call(self, node: ast.Call):
        # 1. Handle LG.result().head(...)
        if self._is_lg_result_head(node):
            self.results.append(ResultAttr(
                attr_name="result", # Default name
                class_name=self.current_class or "global",
                call_loc=self._get_call_loc(node)
            ))
        
        # 2. Handle self.attr() usage
        attr_name = None
        if isinstance(node.func, ast.Attribute) and isinstance(node.func.value, ast.Name) and node.func.value.id == "self":
            attr_name = node.func.attr
        elif isinstance(node.func, ast.Name):
            attr_name = node.func.id
        
        if attr_name:
            key = (self.current_class, attr_name)
            if key in self._potential_inputs:
                call_loc, kind = self._potential_inputs[key]
                # If not already recorded as InputAttr (to avoid duplicate with different forward_use_loc)
                # or we just want the first usage.
                if attr_name not in self.inputs:
                    self.inputs[attr_name] = InputAttr(
                        attr_name=attr_name,
                        class_name=self.current_class or "global",
                        call_loc=call_loc,
                        forward_use_loc=self._get_call_loc(node),
                        lg_source_kind=kind
                    )
        
        self.generic_visit(node)

    def _is_lg_result_head(self, node: ast.Call) -> bool:
        # result.head(...) or LG.result().head(...)
        func = node.func
        if isinstance(func, ast.Attribute) and func.attr == "head":
            # LG.result().head(...)
            if isinstance(func.value, ast.Call):
                inner_func = func.value.func
                if isinstance(inner_func, ast.Attribute):
                    if isinstance(inner_func.value, ast.Name) and inner_func.value.id == "LG":
                        if inner_func.attr == "result":
                            return True
            # result.head(...) - common alias
            elif isinstance(func.value, ast.Name) and func.value.id == "result":
                return True
        return False

def scan_project(root_dir: str) -> tuple[List[InputAttr], List[ResultAttr]]:
    all_inputs = []
    all_results = []
    for root, _, files in os.walk(root_dir):
        for file in files:
            if file.endswith(".py"):
                path = os.path.join(root, file)
                try:
                    with open(path, "r") as f:
                        tree = ast.parse(f.read(), filename=path)
                    scanner = AttrScanner(path)
                    scanner.visit(tree)
                    all_inputs.extend(scanner.inputs.values())
                    all_results.extend(scanner.results)
                except Exception as e:
                    print(f"Error scanning {path}: {e}")
    return all_inputs, all_results
