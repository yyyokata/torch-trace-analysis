from dataclasses import dataclass, field
from typing import Tuple, List, Optional

@dataclass(frozen=True)
class CallLoc:
    file: str
    line: int
    col: int

@dataclass(frozen=True)
class ModuleAttr:
    attr_name: str
    class_name: str
    call_loc: CallLoc

@dataclass(frozen=True)
class ContainerAttr:
    attr_name: str
    class_name: str
    call_loc: CallLoc
    children: Tuple['AttrType', ...] = field(default_factory=tuple)

    def flat_leaves(self) -> List[ModuleAttr]:
        leaves = []
        for child in self.children:
            if isinstance(child, ModuleAttr):
                leaves.append(child)
            elif isinstance(child, ContainerAttr):
                leaves.extend(child.flat_leaves())
        return leaves

@dataclass(frozen=True)
class InputAttr:
    attr_name: str
    class_name: str
    call_loc: CallLoc
    forward_use_loc: CallLoc
    lg_source_kind: str

@dataclass(frozen=True)
class ResultAttr:
    attr_name: str
    class_name: str
    call_loc: CallLoc

# Type alias for convenience
from typing import Union
AttrType = Union[ModuleAttr, ContainerAttr, InputAttr, ResultAttr]
