#!/usr/bin/env python3

import ast
from collections import defaultdict


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
