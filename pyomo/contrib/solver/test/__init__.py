import re
from typing import List, Optional, Type, Union
from pyomo.common import unittest
from pyomo.contrib.solver.common.base import SolverBase
from .registry import SolverTestFilter, SolverTestRegistry
from .builder import SolverTestBuilder
from . import base
from . import linear
from . import dual
from . import quadratic
from . import nonlinear


def add_tests(
    test_case_cls: type[unittest.TestCase],
    opt_cls: Type[SolverBase],
    *,
    include: Optional[List[Union[str, re.Pattern]]] = None,
    exclude: Optional[List[Union[str, re.Pattern]]] = None,
    include_tags: Optional[List[str]] = None,
    exclude_tags: Optional[List[str]] = None,
    warn_unsupported: bool = False,
    warn_unavailable: bool = False,
) -> None:

    if not issubclass(test_case_cls, unittest.TestCase):
        raise TypeError(f"{test_case_cls} must be a TestCase subclass")

    if not issubclass(opt_cls, SolverBase):
        raise TypeError(f"{opt_cls} must be a SolverBase subclass")

    test_builder = SolverTestBuilder(warn_unsupported, warn_unavailable)

    test_filter = SolverTestFilter(
        include=include,
        exclude=exclude,
        include_tags=include_tags,
        exclude_tags=exclude_tags,
    )

    filtered_tests = SolverTestRegistry.get_filtered_tests(test_filter)

    for base_test_name, test_meta in filtered_tests.items():

        test_method = test_builder.build(opt_cls, test_meta)
        if test_method is not None:
            test_name = f"{base_test_name}_{opt_cls.name}"
            setattr(test_case_cls, test_name, test_method)
