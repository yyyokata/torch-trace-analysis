from dataclasses import dataclass as _dataclass
from typing import Any, Optional, Tuple


@_dataclass(frozen=True)
class Scope:
    """Evaluation context for ``ConstantResolver`` look-ups.

    Every public ``eval_*`` call requires a fully-formed ``Scope``.  When
    ``cls`` / ``method`` / ``parent_cls`` / ``parent_attr`` are unknown
    they MUST be passed as ``None`` rather than the empty string -- the
    table look-ups treat ``None`` as "not in any class/method", whereas
    an empty string would silently match unrelated entries.
    """

    file: str
    cls: Optional[str] = None
    method: Optional[str] = None
    parent_cls: Optional[str] = None
    parent_attr: Optional[str] = None

    @property
    def class_key(self) -> Tuple[str, Optional[str]]:
        return (self.file, self.cls)

    @property
    def method_key(self) -> Tuple[str, Optional[str], Optional[str]]:
        return (self.file, self.cls, self.method)

    @property
    def instance_key(self) -> Tuple[Optional[str], Optional[str]]:
        return (self.parent_cls, self.parent_attr)


@_dataclass(frozen=True)
class IntValue:
    """A resolved integer with provenance metadata for diagnostics."""

    value: int
    origin: str = "literal"


@_dataclass(frozen=True)
class ListValue:
    """A resolved list-length (with optional original AST element nodes)."""

    length: int
    items: Optional[Tuple[Any, ...]] = None
    origin: str = "list_literal"



# Sentinel used by ConstantResolver._eval_cache to break recursive cycles
# (see case22 in the design document).  When a recursive eval re-enters the
# same (scope, node) key the cache returns this object and the outer call
# treats it as a soft-fail (None).
_RECURSING = object()

