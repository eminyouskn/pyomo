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


class KNITROVarData:
    def __init__(self, var: VarData):
        self.var = var
        self.init()

    def init(self):
        tmp_lb, tmp_ub = self.var.lb, self.var.ub
        lb, ub, step = self.var.domain.get_interval()

        self._vtype = kn.KN_VARTYPE_CONTINUOUS
        if step == 1:
            self._vtype = kn.KN_VARTYPE_INTEGER

        self._lb = lb
        self._ub = ub

        if lb is None:
            lb = -kn.KN_INFINITY
        if ub is None:
            ub = kn.KN_INFINITY

        if self.var.fixed:
            lb = self.var.value
            ub = self.var.value
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
    def fixed(self) -> bool:
        return self.var.fixed

    @property
    def value(self) -> float:
        return self.var.value

    @property
    def vtype(self) -> int:
        return self._vtype

    @property
    def lb(self) -> float | None:
        return self._lb

    @property
    def ub(self) -> float | None:
        return self._ub


class KNITROConstraintData:
    con: ConstraintData

    def __init__(self, con: ConstraintData):
        self.con = con
        self.init()

    def init(self):
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


class KNITROSolverMixin:
    _ctx: Optional[kn.KN_context]
    _params: Dict[int, ParamData]
    _vars: Dict[int, KNITROVarData]
    _vars_kn: Dict[int, int]
    _cons: Dict[int, KNITROConstraintData]
    _kn_cons: Dict[int, int]

    def __init__(self):
        self._ctx = None
        self._reinit()

    def _reinit(self):
        self._vars.clear()
        self._vars_kn.clear()
        self._params.clear()
        self._cons.clear()
        self._kn_cons.clear()
        self._objective = None
        self._unregister_context()

    def _register_context(self):
        if self._ctx is not None:
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
        v_id = id(var)
        if v_id in self._vars:
            msg = f"Variable {var.name} already exists in KNITRO context."
            raise ValueError(msg)

    def _validate_vars(self, variables: List[VarData]):
        map(self._validate_var, variables)

    def _register_var(self, var: VarData):
        v_id = id(var)
        self._vars[v_id] = KNITROVarData(var=var)

    def _register_vars(self, variables: List[VarData]):
        map(self._register_var, variables)

    def _add_var(self, v: VarData):
        kc = self._get_context()
        v_id = id(v)
        var = self._vars[v_id]
        idx = kn.KN_add_var(kc)
        self._vars_kn[v_id] = idx
        kn.KN_set_var_types(kc, idx, var.vtype)
        if var.fixed:
            kn.KN_set_var_fxbnds(kc, idx, var.value)
        else:
            if var.lb is not None:
                kn.KN_set_var_lobnds(kc, idx, var.lb)
            if var.ub is not None:
                kn.KN_set_var_upbnds(kc, idx, var.ub)

    def _add_vars(self, variables: List[VarData]):
        map(self._add_var, variables)

    def _validate_params(self, parameters: List[ParamData]):
        pass

    def _register_param(self, param: ParamData):
        p_id = id(param)
        self._params[p_id] = param

    def _register_params(self, parameters: List[ParamData]):
        map(self._register_param, parameters)

    def _add_param(self, param: ParamData):
        name = param.name
        kc = self._get_context()
        id_kn = kn.KN_get_param_id(kc, name)
        ptype = kn.KN_get_param_type(kc, id_kn)
        fn = kn.KN_set_char_param
        if ptype == kn.KN_PARAMTYPE_INTEGER:
            fn = kn.KN_set_int_param
        elif ptype == kn.KN_PARAMTYPE_FLOAT:
            fn = kn.KN_set_double_param
        fn(kc, id_kn, param.value)

    def _add_params(self, parameters: List[ParamData]):
        map(self._add_param, parameters)

    def _validate_con(self, con: ConstraintData):
        c_id = id(con)
        if c_id in self._cons:
            msg = f"Constraint {con.name} already exists in KNITRO context."
            raise ValueError(msg)

    def _validate_cons(self, constraints: List[ConstraintData]):
        map(self._validate_con, constraints)

    def _register_con(self, con: ConstraintData):
        c_id = id(con)
        self._cons[c_id] = KNITROConstraintData(con=con)

    def _register_cons(self, constraints: List[ConstraintData]):
        map(self._register_con, constraints)

    def _add_con(self, c: ConstraintData):
        c_id = id(c)
        con = self._cons[c_id]
        kc = self._get_context()
        idx = kn.KN_add_con(kc)
        self._kn_cons[c_id] = idx
        if con.equality:
            kn.KN_set_con_eqbnds(kc, idx, con.eqb)
        else:
            if con.lb is not None:
                kn.KN_set_con_lobnds(kc, idx, con.lb)
            if con.ub is not None:
                kn.KN_set_con_upbnds(kc, idx, con.ub)

        variables, coefs = con.linear
        var_idxs = list(map(self._vars_kn.get, map(id, variables)))
        kn.KN_add_con_linear_struct(kc, idx, var_idxs, coefs)

        variables1, variables2, coefs = con.quadratic
        var1_idxs = list(map(self._vars_kn.get, map(id, variables1)))
        var2_idxs = list(map(self._vars_kn.get, map(id, variables2)))
        kn.KN_add_con_quadratic_struct(kc, idx, var1_idxs, var2_idxs, coefs)

    def _add_cons(self, constraints: List[ConstraintData]):
        map(self._add_con, constraints)

    def _solve(self, timer: Optional[HierarchicalTimer] = None) -> Results:
        if timer is None:
            timer = HierarchicalTimer()
        kc = self._get_context()
        ostreams = [io.StringIO()]
        with capture_output(TeeStream(*ostreams), capture_fd=False):
            options = {}
            for key, option in options.items():
                pass

            status_kn = kn.KN_solve(kc)

        results = Results()
        self._post_solve(status_kn, results, timer=timer)

        results.solver_name = "KNITRO"
        results.solver_version = self.version()
        results.solver_log = ostreams[0].getvalue()
        return results

    def _post_solve(self, status: int, results: Results, timer: HierarchicalTimer):
        # Update the solution status
        if status == kn.KN_RC_OPTIMAL:
            results.solution_status = SolutionStatus.optimal
        elif status == kn.KN_RC_FEAS_FTOL:
            results.solution_status = SolutionStatus.feasible
        elif status == kn.KN_RC_INFEASIBLE:
            results.solution_status = SolutionStatus.infeasible
        else:
            results.solution_status = SolutionStatus.noSolution

        # Update the termination condiction
        if status == kn.KN_RC_OPTIMAL:
            results.termination_condition = (
                TerminationCondition.convergenceCriteriaSatisfied
            )
        elif status == kn.KN_RC_INFEASIBLE:
            results.termination_condition = TerminationCondition.provenInfeasible
        elif status == kn.KN_RC_UNBOUNDED:
            results.termination_condition = TerminationCondition.unbounded
        elif status == kn.KN_RC_UNBOUNDED_OR_INFEAS:
            results.termination_condition = TerminationCondition.infeasibleOrUnbounded
        else:
            results.termination_condition = TerminationCondition.unknown

        # Fill objective value
        kc = self._get_context()
        obj = kn.KN_get_obj_value(kc)
        if obj is not None:
            results.incumbent_objective = obj

        # Fill the number of iterations
        n_iterations = kn.KN_get_number_iters(kc)
        results.iteration_count = n_iterations

        timer.start("load_solution")
        # Load the solution
        timer.stop("load_solution")

        return results

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
