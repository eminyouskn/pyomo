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


import io
import logging
from typing import Dict, List, Optional, Tuple

from pyomo.common.dependencies import attempt_import
from pyomo.common.tee import TeeStream, capture_output
from pyomo.common.timing import HierarchicalTimer
from pyomo.contrib.solver.common.results import (
    Results,
    SolutionStatus,
    TerminationCondition,
)
from pyomo.contrib.solver.common.util import get_objective
from pyomo.contrib.solver.common.base import Availability
from pyomo.core.base.block import BlockData
from pyomo.core.base.constraint import Constraint, ConstraintData
from pyomo.core.base.objective import ObjectiveData
from pyomo.core.base.param import ParamData
from pyomo.core.base.var import VarData
from pyomo.core.expr.numvalue import value

import knitro as kn

from pyomo.repn.standard_repn import generate_standard_repn

_, knitro_available = attempt_import("knitro")


logger = logging.getLogger(__name__)


class KNITROComponentData:
    _idx: Optional[int]

    def __init__(self):
        self._idx = None

    @property
    def idx(self) -> int:
        if self._idx is None:
            msg = "Component has not been added to the KNITRO context."
            raise ValueError(msg)
        return self._idx

    @idx.setter
    def idx(self, value: int):
        self._idx = value


class KNITROVarData(KNITROComponentData):
    _var: VarData

    def __init__(self, var: VarData):
        KNITROComponentData.__init__(self)
        self._var = var
        self.update()

    def update(self):
        tmp_lb, tmp_ub = self._var.lb, self._var.ub
        lb, ub, step = self._var.domain.get_interval()

        self._vtype = kn.KN_VARTYPE_CONTINUOUS
        if step == 1:
            self._vtype = kn.KN_VARTYPE_INTEGER

        self._lb = lb
        self._ub = ub

        if lb is None:
            lb = -kn.KN_INFINITY
        if ub is None:
            ub = kn.KN_INFINITY

        if self._var.fixed:
            lb = self._var.value
            ub = self._var.value
        else:
            # If lb and ub are not constant,
            # we need to save somewhere the
            # expression values to be able to
            # set the bounds later.
            if tmp_lb is not None:
                lb = max(value(tmp_lb), lb)
                self._lb = lb
            if tmp_ub is not None:
                ub = min(value(tmp_ub), ub)
                self._ub = ub

    @property
    def var(self) -> VarData:
        return self._var

    @property
    def fixed(self) -> bool:
        return self._var.fixed

    @property
    def value(self) -> float:
        return self._var.value

    @property
    def vtype(self) -> int:
        return self._vtype

    @property
    def lb(self) -> float | None:
        return self._lb

    @property
    def ub(self) -> float | None:
        return self._ub


class KNITROConstraintData(KNITROComponentData):
    con: ConstraintData

    def __init__(self, con: ConstraintData):
        KNITROComponentData.__init__(self)
        self.con = con
        self.update()

    def update(self):
        tmp_lb, tmp_ub = self.con.lower, self.con.upper
        repn = generate_standard_repn(self.con.body)
        if self.con.equality:
            self._eqb = value(tmp_lb) - repn.constant
        else:
            self._eqb = None
        if tmp_lb is not None:
            self._lb = value(tmp_lb) - repn.constant
        else:
            self._lb = None
        if tmp_ub is not None:
            self._ub = value(tmp_ub) - repn.constant
        else:
            self._ub = None

        self._repn = repn

    @property
    def equality(self) -> bool:
        return self.con.equality

    @property
    def lb(self) -> float | None:
        return self._lb

    @property
    def ub(self) -> float | None:
        return self._ub

    @property
    def eqb(self) -> float | None:
        if not self.equality:
            msg = f"Constraint {self.con.name} is not an equality constraint."
            raise ValueError(msg)
        return self._eqb

    @property
    def linear(self) -> Tuple[List[VarData], List[float]]:
        return self._repn.linear_vars, self._repn.linear_coefs

    @property
    def quadratic(self) -> Tuple[List[VarData], List[VarData], List[float]]:
        return *zip(*self._repn.quadratic_vars), self._repn.quadratic_coefs


class KNITROContextMixin:
    _ctx: Optional[kn.KN_context]

    def __init__(self, ctx: Optional[kn.KN_context] = None):
        self._ctx = ctx

    def _register_context(self):
        self._unregister_context()
        self._ctx = kn.KN_new()

    def _unregister_context(self):
        if self._ctx is not None:
            kn.KN_free(self._ctx)
            self._ctx = None

    def _get_context(self) -> kn.KN_context:
        if self._ctx is None:
            msg = "KNITRO context has not been initialized."
            raise ValueError(msg)
        return self._ctx


class KNITROPostSolveTask(KNITROContextMixin):
    def __init__(
        self,
        ctx: Optional[kn.KN_context] = None,
        status: int = -1000,
        timer: Optional[HierarchicalTimer] = None,
    ):
        KNITROContextMixin.__init__(self, ctx)
        if timer is None:
            timer = HierarchicalTimer()

        self._status = status
        self._timer = timer
        self._results = Results()
        self._primal_solution = None
        self._dual_solution = None

    def run(self):
        self._update_solution_status()
        self._update_termination_condition()
        self._update_objective_value()
        self._update_iteration_count()
        self._update_solution()

    def _update_solution_status(self):
        solution_status = SolutionStatus.noSolution
        if self._status == kn.KN_RC_OPTIMAL:
            solution_status = SolutionStatus.optimal
        elif self._status == kn.KN_RC_FEAS_FTOL:
            solution_status = SolutionStatus.feasible
        elif self._status == kn.KN_RC_INFEASIBLE:
            solution_status = SolutionStatus.infeasible
        else:
            solution_status = SolutionStatus.noSolution

        self._results.solution_status = solution_status

    def _update_termination_condition(self):
        termination_condition = TerminationCondition.unknown
        if self._status == kn.KN_RC_OPTIMAL:
            termination_condition = TerminationCondition.convergenceCriteriaSatisfied
        elif self._status == kn.KN_RC_INFEASIBLE:
            termination_condition = TerminationCondition.provenInfeasible
        elif self._status == kn.KN_RC_UNBOUNDED:
            termination_condition = TerminationCondition.unbounded
        elif self._status == kn.KN_RC_UNBOUNDED_OR_INFEAS:
            termination_condition = TerminationCondition.infeasibleOrUnbounded
        else:
            termination_condition = TerminationCondition.unknown
        self._results.termination_condition = termination_condition

    def _update_objective_value(self):
        kc = self._get_context()
        obj_value = kn.KN_get_obj_value(kc)
        if obj_value is not None:
            self._results.incumbent_objective = obj_value

    def _update_iteration_count(self):
        kc = self._get_context()
        iteration_count = kn.KN_get_number_iters(kc)
        self._results.iteration_count = iteration_count

    def _update_solution(self):
        timer = self._timer
        kc = self._get_context()
        timer.start("load_solution")
        _, _, x, y = kn.KN_get_solution(kc)
        timer.stop("load_solution")
        if x is not None:
            self._primal_solution = x
        if y is not None:
            self._dual_solution = y

    @property
    def results(self) -> Results:
        return self._results

    @property
    def primal_solution(self) -> Optional[List[float]]:
        return self._primal_solution

    @property
    def dual_solution(self) -> Optional[List[float]]:
        return self._dual_solution


class KNITROSolverMixin(KNITROContextMixin):
    _params: Dict[int, ParamData]
    _vars: Dict[int, KNITROVarData]
    _cons: Dict[int, KNITROConstraintData]

    def __init__(self):
        KNITROContextMixin.__init__(self)
        self._reinit()

    def _reinit(self):
        self._vars.clear()
        self._params.clear()
        self._cons.clear()
        self._objective = None
        self._unregister_context()

    def _check_availability(self) -> Availability:
        if not knitro_available:
            return Availability.NotFound
        try:
            kc = kn.KN_new()
            kn.KN_free(kc)
        except Exception:
            return Availability.BadLicense
        return Availability.FullLicense

    def _validate_var(self, var: VarData):
        if id(var) in self._vars:
            msg = f"Variable {var.name} already exists in KNITRO context."
            raise ValueError(msg)

    def _get_var_idxs(self, variables: List[VarData]) -> List[int]:
        return [self._vars.get(id(v)).idx for v in variables]

    def _set_var_idx(self, var: KNITROVarData):
        kc = self._get_context()
        var.idx = kn.KN_add_var(kc)

    def _set_var_type(self, var: KNITROVarData):
        kc = self._get_context()
        kn.KN_set_var_types(kc, var.idx, var.vtype)

    def _set_var_bnds(self, var: KNITROVarData):
        kc = self._get_context()
        if var.fixed:
            kn.KN_set_var_fxbnds(kc, var.idx, var.value)
        else:
            if var.lb is not None:
                kn.KN_set_var_lobnds(kc, var.idx, var.lb)
            if var.ub is not None:
                kn.KN_set_var_upbnds(kc, var.idx, var.ub)

    def _register_var(self, var: KNITROVarData):
        self._set_var_idx(var)
        self._set_var_type(var)
        self._set_var_bnds(var)

    def _add_var(self, var: VarData):
        self._validate_var(var)
        v_id = id(var)
        self._vars[v_id] = KNITROVarData(var=var)
        self._register_var(self._vars[v_id])

    def _add_vars(self, variables: List[VarData]):
        for var in variables:
            self._add_var(var)

    def _add_param(self, param: ParamData):
        p_id = id(param)
        self._params[p_id] = param

    def _add_params(self, parameters: List[ParamData]):
        for param in parameters:
            self._add_param(param)

    def _validate_con(self, con: ConstraintData):
        if id(con) in self._cons:
            msg = f"Constraint {con.name} already exists in KNITRO context."
            raise ValueError(msg)

    def _set_con_idx(self, con: KNITROConstraintData):
        kc = self._get_context()
        con.idx = kn.KN_add_con(kc)

    def _set_con_bnds(self, con: KNITROConstraintData):
        kc = self._get_context()
        if con.equality:
            kn.KN_set_con_eqbnds(kc, con.idx, con.eqb)
        else:
            if con.lb is not None:
                kn.KN_set_con_lobnds(kc, con.idx, con.lb)
            if con.ub is not None:
                kn.KN_set_con_upbnds(kc, con.idx, con.ub)

    def _set_con_linear_struct(self, con: KNITROConstraintData):
        kc = self._get_context()
        variables, coefs = con.linear
        var_idxs = self._get_var_idxs(variables)
        kn.KN_add_con_linear_struct(kc, con.idx, var_idxs, coefs)

    def _set_con_quadratic_struct(self, con: KNITROConstraintData):
        kc = self._get_context()
        variables1, variables2, coefs = con.quadratic
        var1_idxs = self._get_var_idxs(variables1)
        var2_idxs = self._get_var_idxs(variables2)
        kn.KN_add_con_quadratic_struct(kc, con.idx, var1_idxs, var2_idxs, coefs)

    def _register_con(self, con: KNITROConstraintData):
        self._set_con_idx(con)
        self._set_con_bnds(con)
        self._set_con_linear_struct(con)
        self._set_con_quadratic_struct(con)

    def _add_con(self, con: ConstraintData):
        self._validate_con(con)
        c_id = id(con)
        self._cons[c_id] = KNITROConstraintData(con=con)
        self._register_con(self._cons[c_id])

    def _add_cons(self, constraints: List[ConstraintData]):
        for con in constraints:
            self._add_con(con)

    def _solve(self, timer: Optional[HierarchicalTimer] = None) -> Results:
        if timer is None:
            timer = HierarchicalTimer()
        kc = self._get_context()
        ostreams = [io.StringIO()]
        with capture_output(TeeStream(*ostreams), capture_fd=False):
            # TODO: Add options to the context
            status = kn.KN_solve(kc)

        results = self._post_solve(status, timer=timer)
        results.solver_log = ostreams[0].getvalue()
        return results

    def _post_solve(self, status: int, timer: HierarchicalTimer) -> Results:
        kc = self._get_context()
        task = KNITROPostSolveTask(ctx=kc, status=status, timer=timer)
        task.run()
        solution = (task.primal_solution, task.dual_solution)
        print(f"Post-solve solution: {solution}")
        return task.results

    @staticmethod
    def _get_version() -> Tuple[int, int, int]:
        return map(int, kn.__version__.split("."))

    @staticmethod
    def _get_block_params(block: BlockData) -> List[ParamData]:
        params: Dict[int, ParamData] = {}
        for p in block.component_objects(ParamData, descend_into=True):
            if p.mutable:
                for pp in p.values():
                    params[id(pp)] = pp
        return list(params.values())

    @staticmethod
    def _get_block_cons(block: BlockData) -> List[ConstraintData]:
        return list(block.component_objects(Constraint, descend_into=True))

    @staticmethod
    def _get_block_objective(block: BlockData) -> Optional[ObjectiveData]:
        return get_objective(block)
