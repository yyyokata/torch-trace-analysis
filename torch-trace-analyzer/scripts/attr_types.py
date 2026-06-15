from __future__ import annotations

from dataclasses import dataclass, field


_NATIVE_CONTAINER_KINDS = {"ModuleList", "ModuleDict", "Sequential"}


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
    container_index: int | str | None = None
    parent: "ContainerAttr | None" = field(default=None, repr=False, compare=False)


@dataclass
class ModuleAttr(Attr):
    is_native: bool = True
    class_def_loc: "CallLoc | None" = None


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


@dataclass
class ContainerAttr(Attr):
    container_kind: str = ""
    items: dict[int | str, Attr] = field(default_factory=dict)

    def __post_init__(self) -> None:
        for key, child in self.items.items():
            child.parent = self
            child.container_index = key

    def add_child(self, key: int | str, child: Attr) -> None:
        child.parent = self
        child.container_index = key
        self.items[key] = child

    def get(self, key: int | str) -> Attr | None:
        return self.items.get(key)


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
    "ContainerAttr",
]
