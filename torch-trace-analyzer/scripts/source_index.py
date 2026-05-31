#!/usr/bin/env python3

import ast
import sys

try:
    from scripts.ast_frontend import ASTFrontend
except ModuleNotFoundError:
    from ast_frontend import ASTFrontend


def _build_ast_frontends(source_files):
    ast_frontends = {}
    for fname in source_files.keys():
        try:
            ast_frontends[fname] = ASTFrontend(
                source='\n'.join(source_files.get(fname, [])),
                path=fname,
            )
        except Exception:
            ast_frontends[fname] = None
    return ast_frontends


def build_module_like_set(source_files, ast_frontends=None):
    class_defs = {}
    if ast_frontends is None:
        ast_frontends = _build_ast_frontends(source_files)
    for fname, lines in source_files.items():
        fe = ast_frontends.get(fname)
        if fe is not None:
            for node in fe.tree.body:
                if isinstance(node, ast.ClassDef):
                    bases = []
                    for base in node.bases:
                        try:
                            bases.append(fe._node_to_text(base))
                        except Exception:
                            bases.append(getattr(base, "id", getattr(base, "attr", "")))
                    class_defs[(fname, node.name)] = bases
            continue
        try:
            tree = ast.parse("\n".join(lines), filename=fname)
        except Exception:
            continue
        for node in tree.body:
            if isinstance(node, ast.ClassDef):
                bases = []
                for base in node.bases:
                    try:
                        bases.append(ast.unparse(base))
                    except Exception:
                        bases.append(getattr(base, "id", getattr(base, "attr", "")))
                class_defs[(fname, node.name)] = bases

    module_like = set()
    changed = True
    while changed:
        changed = False
        known_names = {c for _, c in module_like}
        for key, bases in class_defs.items():
            if key in module_like:
                continue
            if any(
                b in {"nn.Module", "torch.nn.Module", "Module"}
                or b.endswith(".Module")
                or b in known_names
                for b in bases
            ):
                module_like.add(key)
                changed = True
    return module_like


def _build_class_map_ast(source_files, ast_frontends=None):
    class_map = {}
    failed_files = set()
    if ast_frontends is None:
        ast_frontends = _build_ast_frontends(source_files)
    for fname, lines in source_files.items():
        fe = ast_frontends.get(fname)
        if fe is None:
            failed_files.add(fname)
            continue
        file_failed = False
        for cname, info in fe.class_registry.items():
            cls_node = info.get("node")
            start = getattr(cls_node, "lineno", None)
            end = getattr(cls_node, "end_lineno", None)
            if start is None or end is None:
                file_failed = True
                break
            methods = {}
            for method in info.get("methods", []):
                mstart = method.get("lineno")
                mend = method.get("end_lineno")
                mname = method.get("name")
                if not mname or mstart is None or mend is None:
                    file_failed = True
                    break
                methods[mname] = (mstart, mend)
            if file_failed:
                break
            class_map[(fname, cname)] = {"start": start, "end": end, "methods": methods}
        if file_failed:
            failed_files.add(fname)
            for key in [k for k in class_map if k[0] == fname]:
                class_map.pop(key, None)
    return class_map, failed_files


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
