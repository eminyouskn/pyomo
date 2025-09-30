from collections.abc import Callable, MutableSet, Set
from dataclasses import dataclass, field
import re
from typing import Dict, List, Optional, Tuple, Type, Union

from pyomo.common import unittest
from pyomo.contrib.solver.common.base import SolverBase
from .capability import Capability, CapabilityRegistry


class SolverCapabilityRegistryClass:
    _caps: Dict[Type[SolverBase], MutableSet[Capability]]

    def __init__(self):
        self._caps = {}

    def register(self, opt_cls: Type[SolverBase], *caps: Capability):
        if issubclass(opt_cls, SolverBase) is False:
            msg = f"{opt_cls} is not a subclass of SolverBase"
            raise TypeError(msg)

        if opt_cls not in self._caps:
            self._caps[opt_cls] = set()

        resolved_caps = CapabilityRegistry.resolve_implications(set(caps))
        for cap in resolved_caps:
            self._caps[opt_cls].add(cap)

    def supports(self, opt_cls: Type[SolverBase], *caps: Capability) -> bool:
        if opt_cls not in self._caps:
            return False
        return all(cap in self._caps[opt_cls] for cap in caps)

    def get_missing_caps(
        self, opt_cls: Type[SolverBase], *caps: Capability
    ) -> Set[Capability]:
        if opt_cls not in self._caps:
            return set(caps)
        return {cap for cap in caps if cap not in self._caps[opt_cls]}


SolverCapabilityRegistry = SolverCapabilityRegistryClass()


@dataclass
class SolverTestMeta:
    """Metadata for solver tests."""

    func: Callable[[unittest.TestCase, type[SolverBase]], None]
    reqs: MutableSet[Capability] = field(default_factory=set)
    skip_reason: Optional[str] = field(default=None)
    tags: MutableSet[str] = field(default_factory=set)

    def can_run_on(self, opt_cls: Type[SolverBase]) -> Tuple[bool, Optional[str]]:
        if self.skip_reason is not None:
            return False, self.skip_reason

        if SolverCapabilityRegistry.supports(opt_cls, *self.reqs):
            return True, None

        missing = SolverCapabilityRegistry.get_missing_caps(opt_cls, *self.reqs)
        msg = f"Solver {opt_cls.__name__} does not support required capabilities: {missing}"
        return False, msg


class SolverTestFilter:
    def __init__(
        self,
        include: Optional[List[Union[str, re.Pattern]]] = None,
        exclude: Optional[List[Union[str, re.Pattern]]] = None,
        include_tags: Optional[List[str]] = None,
        exclude_tags: Optional[List[str]] = None,
    ):
        self.include = include
        self.exclude = exclude or []
        self.include_tags = set(include_tags) if include_tags is not None else None
        self.exclude_tags = set(exclude_tags or [])

    def should_include(self, test_name: str, test_meta: SolverTestMeta) -> bool:
        if any(re.match(pat, test_name) for pat in self.exclude):
            return False

        if self.include is not None and not any(re.match(pat, test_name) for pat in self.include):
            return False

        if any(tag in self.exclude_tags for tag in test_meta.tags):
            return False

        if self.include_tags is not None and not any(
            tag in self.include_tags for tag in test_meta.tags
        ):
            return False

        return True


class SolverTestRegistryClass:
    _tests: Dict[str, SolverTestMeta]

    def __init__(self):
        self._tests = {}

    def _get_or_create_test_meta(self, func: Callable) -> SolverTestMeta:
        name = func.__name__
        if name not in self._tests:
            self._tests[name] = SolverTestMeta(func=func)
        return self._tests[name]

    def requires(self, *caps: Capability):
        def decorator(func: Callable):
            test_meta = self._get_or_create_test_meta(func)
            for cap in caps:
                test_meta.reqs.add(cap)
            return func

        return decorator

    def tags(self, *tags: str):
        def decorator(func: Callable):
            test_meta = self._get_or_create_test_meta(func)
            for tag in tags:
                test_meta.tags.add(tag)
            return func

        return decorator

    def skip_if(self, condition: Union[bool, Callable[[], bool]], reason: str = ""):
        def decorator(func: Callable):
            test_meta = self._get_or_create_test_meta(func)
            should_skip = condition() if callable(condition) else condition
            if should_skip:
                test_meta.skip_reason = reason or "Conditional skip"
            return func

        return decorator

    def register(self):
        def decorator(func: Callable):
            self._get_or_create_test_meta(func)
            return func

        return decorator

    def get_filtered_tests(
        self, test_filter: SolverTestFilter
    ) -> Dict[str, SolverTestMeta]:
        return {
            name: meta
            for name, meta in self._tests.items()
            if test_filter.should_include(name, meta)
        }


SolverTestRegistry = SolverTestRegistryClass()
