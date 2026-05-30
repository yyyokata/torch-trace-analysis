from dataclasses import dataclass, field
from typing import Optional, Dict, Union, Any
from abc import ABC

@dataclass(frozen=True)
class CallLoc:
    file: str
    line: int
    col: int

class Attr(ABC):
    def __init__(
        self, 
        attr_name: str, 
        class_name: str, 
        call_loc: CallLoc, 
        parent: Optional["ContainerAttr"] = None
    ):
        self.attr_name = attr_name
        self.class_name = class_name
        self.call_loc = call_loc
        self.parent = parent

    def __eq__(self, other: Any) -> bool:
        if not isinstance(other, Attr):
            return False
        return (
            self.attr_name == other.attr_name and
            self.class_name == other.class_name and
            self.call_loc == other.call_loc
        )

class ModuleAttr(Attr):
    pass

class ContainerAttr(Attr):
    def __init__(
        self,
        attr_name: str,
        class_name: str,
        call_loc: CallLoc,
        container_kind: str,
        parent: Optional["ContainerAttr"] = None,
        items: Optional[Dict[Union[int, str], Attr]] = None
    ):
        super().__init__(attr_name, class_name, call_loc, parent)
        self.container_kind = container_kind
        self.items: Dict[Union[int, str], Attr] = items or {}
        # Ensure parent linkage for initial items
        for child in self.items.values():
            child.parent = self

    def add_child(self, key: Union[int, str], child: Attr):
        child.parent = self
        self.items[key] = child

    def get(self, key: Union[int, str]) -> Optional[Attr]:
        return self.items.get(key)

class InputAttr(Attr):
    def __init__(
        self,
        attr_name: str,
        class_name: str,
        call_loc: CallLoc,
        forward_use_loc: CallLoc,
        lg_source_kind: str,
        parent: Optional["ContainerAttr"] = None
    ):
        super().__init__(attr_name, class_name, call_loc, parent)
        self.forward_use_loc = forward_use_loc
        self.lg_source_kind = lg_source_kind

class ResultAttr(Attr):
    pass

AttrType = Union[ModuleAttr, ContainerAttr, InputAttr, ResultAttr]
