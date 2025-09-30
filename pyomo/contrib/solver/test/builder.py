from collections.abc import Callable
from typing import Optional, Type

from pyomo.common import unittest
from pyomo.contrib.solver.common.base import SolverBase
from .registry import SolverTestMeta


class SolverTestBuilder:
    def __init__(self, warn_unsupported: bool, warn_unavailable: bool):
        self.warn_unsupported = warn_unsupported
        self.warn_unavailable = warn_unavailable

    def build(
        self, opt_cls: Type[SolverBase], test_meta: SolverTestMeta
    ) -> Optional[Callable[[unittest.TestCase], None]]:
        try:
            avail = bool(opt_cls().available())
        except Exception as e:
            avail = False

        can_run, skip_reason = test_meta.can_run_on(opt_cls)

        if can_run and avail:
            return self._build_runnable_test(opt_cls, test_meta)
        elif can_run and not avail and self.warn_unavailable:
            return self._build_skip_test(f"Solver {opt_cls.name} is not available")
        elif not can_run and self.warn_unsupported:
            return self._build_skip_test(f"Solver {opt_cls.name} {skip_reason}")

        else:
            return None

    def _build_runnable_test(
        self, opt_cls: Type[SolverBase], test_meta: SolverTestMeta
    ):
        def test_method(self: unittest.TestCase):
            test_meta.func(self, opt_cls)

        return test_method

    def _build_skip_test(self, reason: str):
        def test_method(self: unittest.TestCase):
            self.skipTest(reason)

        return test_method
