from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class CallLoc:
    file: str
    line: int
    col: int


@dataclass
class Attr:
    # NOTE: mock 侧会构造 ParamAttr(param_name=...) / ConstantAttr(attr_name=..., op_name=..., def_loc=None)
    # 因此这里必须给基础字段提供默认值。
    attr_name: str = ""
    class_name: str = ""
    def_loc: CallLoc | None = None
    attr_id: int = 0


@dataclass
class ModuleAttr(Attr):
    is_native: bool = True


@dataclass
class InputAttr(Attr):
    kind: str = ""


@dataclass
class ResultAttr(Attr):
    head_name: str = ""
    classifier_type: object | None = None


@dataclass
class ForwardArgAttr(Attr):
    arg_index: int = 0


@dataclass
class ReturnValAttr(Attr):
    ret_index: int = 0
    ret_key: str | None = None


@dataclass
class FunctionalAttr(Attr):
    is_native: bool = True


@dataclass
class ConstantAttr(Attr):
    op_name: str = ""


@dataclass
class ParamAttr(Attr):
    param_name: str = ""


__all__ = [
    "CallLoc",
    "Attr",
    "ModuleAttr",
    "InputAttr",
    "ResultAttr",
    "ForwardArgAttr",
    "ReturnValAttr",
    "FunctionalAttr",
    "ConstantAttr",
    "ParamAttr",
]
