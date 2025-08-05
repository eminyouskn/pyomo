#  ___________________________________________________________________________
#
#  Pyomo: Python Optimization Modeling Objects
#  Copyright (c) 2008-2025
#  National Technology and Engineering Solutions of Sandia, LLC
#  Under the terms of Contract DE-NA0003525 with National Technology and
#  Engineering Solutions of Sandia, LLC, the U.S. Government retains certain
#  rights in this software.
#  This software is distributed under the 3-clause BSD License.
#  ___________________________________________________________________________


import datetime
from typing import List, Optional

from pyomo.common.errors import ApplicationError
from pyomo.common.shutdown import python_is_shutting_down
from pyomo.common.timing import HierarchicalTimer
from pyomo.contrib.solver.common.base import Availability, PersistentSolverBase
from pyomo.contrib.solver.common.results import Results
from pyomo.contrib.solver.solvers.knitrointerface.config import KNITROConfig
from pyomo.contrib.solver.solvers.knitrointerface.util import KNITROSolverMixin
from pyomo.core.base.block import BlockData
from pyomo.core.base.constraint import ConstraintData
from pyomo.core.base.param import ParamData
from pyomo.core.base.var import VarData
from pyomo.core.staleflag import StaleFlagManager


class KNITRO(KNITROSolverMixin, PersistentSolverBase):
    """
    Interface to the KNITRO solver.
    """

    SOLVER_NAME = "KNITRO"
    CONFIG = KNITROConfig()
    _model: Optional[BlockData]

    def __init__(self, **kwds):
        PersistentSolverBase.__init__(self, **kwds)
        KNITROSolverMixin.__init__(self)
        self._model = None

    def __del__(self):
        if not python_is_shutting_down():
            self._unregister_context()

    def available(self) -> Availability:
        return self._check_availability()

    def version(self):
        return self._get_version()

    def solve(self, model: BlockData, **kwds) -> Results:
        start = datetime.datetime.now(datetime.timezone.utc)

        StaleFlagManager.mark_all_as_stale()

        timer = HierarchicalTimer()
        if model is not self._model:
            timer.start("set_instance")
            self.set_instance(model)
            timer.stop("set_instance")
        else:
            timer.start("update")
            self.update(timer=timer)
            timer.stop("update")

        results = self._solve(timer=timer)

        end = datetime.datetime.now(datetime.timezone.utc)

        results.solver_name = self.SOLVER_NAME
        results.solver_version = self.version()
        results.timing_info.start_timestamp = start
        results.timing_info.wall_time = (end - start).total_seconds()
        results.timing_info.timer = timer
        return results

    def set_instance(self, model: BlockData):
        if not self.available():
            msg = f"Solver {self.__class__.__module__}.{self.__class__.__qualname__} is not available. ({self.available()})"
            raise ApplicationError(msg)
        self._reinit()
        self._model = model
        self._register_context(model)
        self.add_block(model)

    def add_block(self, block: BlockData):
        params = self._get_block_params(block)
        if params:
            self.add_parameters(params)
        cons = self._get_block_cons(block)
        if cons:
            self.add_constraints(cons)

    def add_variables(self, variables: List[VarData]):
        self._add_vars(variables)

    def add_parameters(self, parameters: List[ParamData]):
        self._add_params(parameters)

    def add_constraints(self, constraints: List[ConstraintData]):
        self._add_cons(constraints)

    def update(self, timer: Optional[HierarchicalTimer] = None):
        if self._model is None:
            raise ApplicationError("No model is set. Cannot update.")
        if timer is None:
            timer = HierarchicalTimer()
