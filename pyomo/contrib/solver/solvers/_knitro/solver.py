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
from pyomo.contrib.solver.common.results import Results, SolutionStatus
from pyomo.contrib.solver.common.solution_loader import PersistentSolutionLoader
from pyomo.contrib.solver.solvers._knitro.config import KNConfig
from pyomo.contrib.solver.solvers._knitro.util import KNSolverUtils
from pyomo.core.base.block import BlockData
from pyomo.core.base.constraint import ConstraintData
from pyomo.core.base.objective import ObjectiveData
from pyomo.core.base.param import ParamData
from pyomo.core.base.var import VarData
from pyomo.core.staleflag import StaleFlagManager


class KNSolutionLoader(PersistentSolutionLoader):
    def load_vars(self, vars_to_load=None):
        self._solver._load_vars(vars_to_load)

    def get_primals(self, vars_to_load=None):
        return self._solver._get_primals(vars_to_load)

    def get_duals(self, constraints=None):
        return self._solver._get_duals(constraints)

    def get_reduced_costs(self, vars_to_load=None):
        return self._solver._get_reduced_costs(vars_to_load)


class KNSolver(KNSolverUtils, PersistentSolverBase):
    """
    Interface to the KNITRO solver.
    """

    SOLVER_NAME = "KNITRO"
    CONFIG = KNConfig()
    config: KNConfig

    _model: Optional[BlockData]

    def __init__(self, **kwds):
        PersistentSolverBase.__init__(self, **kwds)
        KNSolverUtils.__init__(self)
        self._model = None

    def __del__(self):
        if not python_is_shutting_down():
            self._ctx.free()

    def available(self) -> Availability:
        return self._check_availability()

    def version(self):
        return self._get_version()

    def solve(self, model: BlockData, **kwds) -> Results:
        start = datetime.datetime.now(datetime.timezone.utc)

        StaleFlagManager.mark_all_as_stale()
        config: KNConfig = self.config(value=kwds, preserve_implicit=True)

        timer = config.timer
        if timer is None:
            timer = HierarchicalTimer()

        if model is not self._model:
            timer.start("set_instance")
            self.set_instance(model)
            timer.stop("set_instance")
        else:
            timer.start("update")
            self._update(config=config, timer=timer)
            timer.stop("update")

        results = self._solve(config=config, timer=timer)

        end = datetime.datetime.now(datetime.timezone.utc)

        if (
            config.raise_exception_on_nonoptimal_result
            and results.solution_status != SolutionStatus.optimal
        ):
            msg = f"Solver {self.SOLVER_NAME} did not find an optimal solution."
            raise ValueError(msg)

        if config.load_solutions:
            timer.start("load_solution")
            self._load_vars()
            timer.stop("load_solution")

        results.solver_name = self.SOLVER_NAME
        results.solver_version = self.version()
        results.solution_loader = KNSolutionLoader(self)
        results.timing_info.start_timestamp = start
        results.timing_info.wall_time = (end - start).total_seconds()
        results.timing_info.timer = timer
        return results

    def set_instance(self, model: BlockData):
        self._ensure_solver_is_available()
        self._reinit()
        self._ctx.create()
        self._model = model
        self.add_block(model)

    def set_objective(self, obj: ObjectiveData):
        self._set_objective(obj)

    def add_block(self, block: BlockData):
        params = self._get_block_params(block)
        self.add_parameters(params)
        variables = self._get_block_vars(block)
        self.add_variables(variables)
        cons = self._get_block_cons(block)
        self.add_constraints(cons)
        obj = self._get_block_objective(block)
        self.set_objective(obj)

    def remove_block(self, block: BlockData):
        cons = self._get_block_cons(block)
        self.remove_constraints(cons)
        variables = self._get_block_vars(block)
        self.remove_variables(variables)
        params = self._get_block_params(block)
        self.remove_parameters(params)

    def add_variables(self, variables: List[VarData]):
        self._add_vars(variables)

    def add_parameters(self, parameters: List[ParamData]):
        self._add_params(parameters)

    def add_constraints(self, constraints: List[ConstraintData]):
        self._add_cons(constraints)

    def update_variables(self, variables: List[VarData]):
        self._update_vars(variables)

    def update_parameters(self):
        self._update_params()

    def update_constraints(self, constraints: List[ConstraintData]):
        self._update_cons(constraints)

    def remove_variables(self, variables):
        self._remove_vars(variables)

    def remove_parameters(self, parameters):
        self._remove_params(parameters)

    def remove_constraints(self, constraints):
        self._remove_cons(constraints)

    def _get_primals(self, vars_to_load=None):
        return self._get_primal_sol(variables=vars_to_load)

    def _get_duals(self, cons_to_load=None):
        return self._get_dual_sol(constraints=cons_to_load)

    def _get_reduced_costs(self, vars_to_load=None):
        return self._get_reduced_cost(variables=vars_to_load)

    def _ensure_solver_is_available(self):
        if not self.available():
            msg = f"Solver {self.__class__.__module__}.{self.__class__.__qualname__} is not available. ({self.available()})"
            raise ApplicationError(msg)
