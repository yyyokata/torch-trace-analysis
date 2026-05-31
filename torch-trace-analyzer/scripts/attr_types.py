from dataclasses import dataclass
from typing import Optional, Dict, Union, Any
from abc import ABC


@dataclass(frozen=True)
class CallLoc:
    file: str
    line: int
    col: int


_NATIVE_PREFIXES = ("nn.", "torch.nn.")
_NATIVE_CONTAINER_KINDS = {"ModuleList", "ModuleDict", "Sequential"}


def _infer_native_from_class_name(class_name: str) -> bool:
    return isinstance(class_name, str) and class_name.startswith(_NATIVE_PREFIXES)


class Attr(ABC):
    def __init__(
        self,
        attr_name: str,
        class_name: str,
        def_loc: CallLoc,
        attr_id: int = 0,
        parent: Optional["ContainerAttr"] = None,
        source_expr: Optional[str] = None,
        is_native: Optional[bool] = None,
        container_index: Optional[Union[int, str]] = None,
    ):
        self.attr_id = attr_id
        self.attr_name = attr_name
        self.class_name = class_name
        self.def_loc = def_loc
        self.parent = parent
        self.source_expr = source_expr
        self.is_native = _infer_native_from_class_name(class_name) if is_native is None else is_native
        self.container_index = container_index

    def __eq__(self, other: Any) -> bool:
        if not isinstance(other, Attr):
            return False
        return (
            self.attr_name == other.attr_name and
            self.class_name == other.class_name and
            self.def_loc == other.def_loc
        )


class ModuleAttr(Attr):
    def __init__(
        self,
        attr_name: str,
        class_name: str,
        def_loc: CallLoc,
        attr_id: int = 0,
        parent: Optional["ContainerAttr"] = None,
        source_expr: Optional[str] = None,
        is_native: Optional[bool] = None,
        container_index: Optional[Union[int, str]] = None,
    ):
        super().__init__(
            attr_name=attr_name,
            class_name=class_name,
            def_loc=def_loc,
            attr_id=attr_id,
            parent=parent,
            source_expr=source_expr,
            is_native=is_native,
            container_index=container_index,
        )


class ContainerAttr(Attr):
    def __init__(
        self,
        attr_name: str,
        class_name: str,
        def_loc: CallLoc,
        container_kind: str,
        attr_id: int = 0,
        parent: Optional["ContainerAttr"] = None,
        items: Optional[Dict[Union[int, str], Attr]] = None,
        source_expr: Optional[str] = None,
        is_native: Optional[bool] = None,
        container_index: Optional[Union[int, str]] = None,
    ):
        resolved_is_native = (
            container_kind in _NATIVE_CONTAINER_KINDS
            if is_native is None
            else is_native
        )
        super().__init__(
            attr_name,
            class_name,
            def_loc,
            attr_id=attr_id,
            parent=parent,
            source_expr=source_expr,
            is_native=resolved_is_native,
            container_index=container_index,
        )
        self.container_kind = container_kind
        self.items: Dict[Union[int, str], Attr] = items or {}
        for key, child in self.items.items():
            child.parent = self
            child.container_index = key

    def add_child(self, key: Union[int, str], child: Attr):
        child.parent = self
        child.container_index = key
        self.items[key] = child

    def get(self, key: Union[int, str]) -> Optional[Attr]:
        return self.items.get(key)


class InputAttr(Attr):
    def __init__(
        self,
        attr_name: str,
        class_name: str,
        def_loc: CallLoc,
        attr_id: int = 0,
        parent: Optional["ContainerAttr"] = None,
        kind: str = "",
        owner_expr: Optional[str] = None,
        slot_expr: Optional[str] = None,
        source_expr: Optional[str] = None,
        is_native: bool = False,
        container_index: Optional[Union[int, str]] = None,
    ):
        super().__init__(
            attr_name,
            class_name,
            def_loc,
            attr_id=attr_id,
            parent=parent,
            source_expr=source_expr,
            is_native=is_native,
            container_index=container_index,
        )
        resolved_kind = kind or ""
        self.kind = resolved_kind
        self.owner_expr = owner_expr
        self.slot_expr = slot_expr


class ResultAttr(Attr):
    """result.head(name, prediction, label, sample_rate, loss, classifier_type) 的解析结果。

    - head_name / classifier_type：属性类，描述 head 元信息
    - *_expr：输出 tensor 表达式原文，各自独立走 var_lineage 追踪用于建 edge
    - prediction_expr 必填（None 说明解析失败，已 warn）
    """

    def __init__(
        self,
        attr_name: str,
        class_name: str,
        def_loc: CallLoc,
        attr_id: int = 0,
        parent: Optional["ContainerAttr"] = None,
        source_expr: Optional[str] = None,
        head_name: Optional[str] = None,
        classifier_type: Optional[str] = None,
        prediction_expr: Optional[str] = None,
        label_expr: Optional[str] = None,
        sample_rate_expr: Optional[str] = None,
        loss_expr: Optional[str] = None,
        is_native: bool = False,
        container_index: Optional[Union[int, str]] = None,
    ):
        super().__init__(
            attr_name,
            class_name,
            def_loc,
            attr_id=attr_id,
            parent=parent,
            source_expr=source_expr,
            is_native=is_native,
            container_index=container_index,
        )
        self.head_name = head_name
        self.classifier_type = classifier_type
        self.prediction_expr = prediction_expr
        self.label_expr = label_expr
        self.sample_rate_expr = sample_rate_expr
        self.loss_expr = loss_expr


class ForwardArgAttr(Attr):
    """inner_dag InputNode 的 attr：对应 forward 形参"""
    def __init__(
        self,
        attr_name: str,
        class_name: str = "__arg__",
        def_loc: Optional[CallLoc] = None,
        attr_id: int = 0,
        arg_index: int = 0,
    ):
        super().__init__(attr_name=attr_name, class_name=class_name, def_loc=def_loc, attr_id=attr_id)
        self.arg_index = arg_index


class ReturnValAttr(Attr):
    """inner_dag ResultNode 的 attr：对应 forward 返回值"""
    def __init__(
        self,
        attr_name: str,
        class_name: str = "__ret__",
        def_loc: Optional[CallLoc] = None,
        attr_id: int = 0,
        ret_index: int = 0,
    ):
        super().__init__(attr_name=attr_name, class_name=class_name, def_loc=def_loc, attr_id=attr_id)
        self.ret_index = ret_index


AttrType = Union[ModuleAttr, ContainerAttr, InputAttr, ResultAttr, ForwardArgAttr, ReturnValAttr]
