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


from collections.abc import MutableMapping, Sequence
import io
import logging
from typing import Dict, List, Mapping, Optional, Tuple, Union

from pyomo.common.collections.component_map import ComponentMap
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
from pyomo.contrib.solver.solvers._knitro.config import KNConfig
from pyomo.core.base.block import BlockData
from pyomo.core.base.constraint import Constraint, ConstraintData
from pyomo.core.base.expression import Expression
from pyomo.core.base.objective import ObjectiveData
from pyomo.core.base.param import ParamData
from pyomo.core.base.var import Var, VarData
from pyomo.core.expr.visitor import identify_variables
from pyomo.core.kernel.objective import minimize, maximize
from pyomo.core.expr.numvalue import value

from pyomo.repn.standard_repn import generate_standard_repn


kn, KNITRO_AVAILABLE = attempt_import("knitro")

logger = logging.getLogger(__name__)


class KNC:
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


class KNVar(KNC):
    _var: VarData

    def __init__(self, var: VarData):
        KNC.__init__(self)
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
            if tmp_lb is not None:
                lb = max(value(tmp_lb), lb)
                self._lb = lb
            if tmp_ub is not None:
                ub = min(value(tmp_ub), ub)
                self._ub = ub

    @property
    def var(self) -> VarData:
        return self._var

    @var.setter
    def var(self, var: VarData):
        self._var = var
        self.update()

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


class KNCon(KNC):
    _con: ConstraintData

    def __init__(self, con: ConstraintData):
        KNC.__init__(self)
        self._con = con
        self.update()

    def update(self):
        tmp_lb, tmp_ub = self._con.lower, self._con.upper
        repn = generate_standard_repn(self._con.body)
        if self._con.equality:
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
    def con(self) -> ConstraintData:
        return self._con

    @con.setter
    def con(self, con: ConstraintData):
        self._con = con
        self.update()

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
        if not self._repn.quadratic_vars:
            return [], [], []
        return *zip(*self._repn.quadratic_vars), self._repn.quadratic_coefs


class KNObj:
    _obj: Optional[ObjectiveData]

    def __init__(self, obj: Optional[ObjectiveData] = None):
        self._obj = obj
        self.update()

    def update(self):
        if self._obj is None:
            self._repn = None
            return
        self._repn = generate_standard_repn(self._obj.expr)

    @property
    def obj(self) -> Optional[ObjectiveData]:
        return self._obj

    @obj.setter
    def obj(self, obj: Optional[ObjectiveData]):
        self._obj = obj
        self.update()

    @property
    def sense(self) -> int:
        if self._obj is None:
            return kn.KN_OBJGOAL_MINIMIZE
        elif self._obj.sense == minimize:
            return kn.KN_OBJGOAL_MINIMIZE
        elif self._obj.sense == maximize:
            return kn.KN_OBJGOAL_MAXIMIZE

    @property
    def constant(self) -> float:
        if self._repn is None:
            return 0.0
        return value(self._repn.constant)

    @property
    def linear(self) -> Tuple[List[VarData], List[float]]:
        if self._repn is None:
            return [], []
        return self._repn.linear_vars, self._repn.linear_coefs

    @property
    def quadratic(self) -> Tuple[List[VarData], List[VarData], List[float]]:
        if self._repn is None or not self._repn.quadratic_vars:
            return [], [], []
        return *zip(*self._repn.quadratic_vars), self._repn.quadratic_coefs


class KNCtx:
    def __init__(self):
        self._kc = None

    def create(self):
        self.free()
        self._kc = kn.KN_new()

    def free(self):
        if self._kc is not None:
            kn.KN_free(self._kc)
            self._kc = None

    def get(self):
        if self._kc is None:
            raise ValueError("Knitro context has not been initialized.")
        return self._kc


class KNPostSolveTask:
    def __init__(self, kc, status: int, timer: HierarchicalTimer):
        self._kc = kc
        self._status = status
        self._timer = timer
        self._results = Results()

    def run(self):
        self._update_solution_status()
        self._update_termination_condition()
        self._update_objective_value()
        self._update_iteration_count()

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
        obj_value = kn.KN_get_obj_value(self.kc)
        if obj_value is not None:
            self._results.incumbent_objective = obj_value

    def _update_iteration_count(self):
        iteration_count = kn.KN_get_number_iters(self.kc)
        self._results.iteration_count = iteration_count

    @property
    def kc(self):
        return self._kc

    @property
    def results(self) -> Results:
        return self._results


class KNSolverUtils:
    _ctx: KNCtx
    _params: Dict[int, ParamData]
    _vars: Dict[int, KNVar]
    _cons: Dict[int, KNCon]
    _objective: KNObj

    def __init__(self):
        self._ctx = KNCtx()
        self._params = {}
        self._vars = {}
        self._cons = {}

    def _reinit(self):
        self._vars.clear()
        self._params.clear()
        self._cons.clear()
        self._objective = KNObj()
        self._ctx.free()

    def _check_availability(self) -> Availability:
        if not KNITRO_AVAILABLE:
            return Availability.NotFound
        try:
            kc = kn.KN_new()
            kn.KN_free(kc)
        except Exception:
            return Availability.BadLicense
        return Availability.FullLicense

    def _set_solver_option(self, key: str, val: Union[str, int, float]):
        param_type = self._KNcall(kn.KN_get_param_type, key)
        fn = kn.KN_set_char_param
        if param_type == kn.KN_PARAMTYPE_INTEGER:
            fn = kn.KN_set_int_param
        elif param_type == kn.KN_PARAMTYPE_FLOAT:
            fn = kn.KN_set_double_param
        param_id = self._KNcall(kn.KN_get_param_id, key)
        self._KNcall(fn, param_id, val)

    def _set_default_solver_options(self, config: KNConfig):
        # TODO: remove this when the presolve # is fixed.
        self._KNcall(kn.KN_set_int_param, kn.KN_PARAM_PRESOLVE, kn.KN_PRESOLVE_NONE)
        self._KNcall(kn.KN_set_int_param, kn.KN_PARAM_OUTLEV, kn.KN_OUTLEV_ALL)
        if config.threads is not None:
            self._KNcall(kn.KN_set_int_param, kn.KN_PARAM_NUMTHREADS, config.threads)
        if config.time_limit is not None:
            self._KNcall(kn.KN_set_double_param, kn.KN_PARAM_MAXTIME, config.time_limit)

    def _set_solver_options(self, options: Dict[str, Union[str, int, float]]):
        for option in options.items():
            self._set_solver_option(*option)

    def _ensure_var_not_registered(self, var: VarData):
        if id(var) in self._vars:
            msg = f"Variable {var.name} already exists in KNITRO context."
            raise ValueError(msg)

    def _ensure_var_is_registered(self, var: VarData):
        if id(var) not in self._vars:
            msg = f"Variable {var.name} is not registered in KNITRO context."
            raise ValueError(msg)

    def _ensure_con_not_registered(self, con: ConstraintData):
        if id(con) in self._cons:
            msg = f"Constraint {con.name} already exists in KNITRO context."
            raise ValueError(msg)

    def _ensure_con_is_registered(self, con: ConstraintData):
        if id(con) not in self._cons:
            msg = f"Constraint {con.name} is not registered in KNITRO context."
            raise ValueError(msg)

    def _ensure_param_not_registered(self, param: ParamData):
        if id(param) in self._params:
            msg = f"Parameter {param.name} already exists in KNITRO context."
            raise ValueError(msg)

    def _ensure_param_is_registered(self, param: ParamData):
        if id(param) not in self._params:
            msg = f"Parameter {param.name} is not registered in KNITRO context."
            raise ValueError(msg)

    def _ensure_var_not_used(self, var: VarData, expr: Expression):
        if id(var) in identify_variables(expr):
            msg = f"Variable {var.name} is used in the expression."
            raise ValueError(msg)

    def _get_var_solver_idx(self, var: VarData) -> int:
        return self._vars.get(id(var)).idx

    def _get_var_solver_idxs(self, variables: List[VarData]) -> List[int]:
        return [self._get_var_solver_idx(var) for var in variables]

    def _get_con_solver_idx(self, con: ConstraintData) -> int:
        return self._cons.get(id(con)).idx

    def _get_con_solver_idxs(self, constraints: List[ConstraintData]) -> List[int]:
        return [self._get_con_solver_idx(con) for con in constraints]

    def _assign_var_solver_idx(self, var: KNVar):
        var.idx = self._KNcall(kn.KN_add_var)

    def _assign_con_solver_idx(self, con: KNCon):
        con.idx = self._KNcall(kn.KN_add_con)

    def _set_var_type(self, var: KNVar):
        self._KNcall(kn.KN_set_var_types, var.idx, var.vtype)

    def _set_var_bnds(self, var: KNVar):
        if var.fixed:
            self._KNcall(kn.KN_set_var_fxbnds, var.idx, var.value)
        else:
            if var.lb is not None:
                self._KNcall(kn.KN_set_var_lobnds, var.idx, var.lb)
            if var.ub is not None:
                self._KNcall(kn.KN_set_var_upbnds, var.idx, var.ub)

    def _reset_var_bnds(self, var: KNVar):
        self._KNcall(kn.KN_set_var_lobnds, var.idx, -kn.KN_INFINITY)
        self._KNcall(kn.KN_set_var_upbnds, var.idx, kn.KN_INFINITY)

    def _reset_var_type(self, var: KNVar):
        self._KNcall(kn.KN_set_var_types, var.idx, kn.KN_VARTYPE_CONTINUOUS)

    def _set_con_bnds(self, con: KNCon):
        if con.equality:
            self._KNcall(kn.KN_set_con_eqbnds, con.idx, con.eqb)
        else:
            if con.lb is not None:
                self._KNcall(kn.KN_set_con_lobnds, con.idx, con.lb)
            if con.ub is not None:
                self._KNcall(kn.KN_set_con_upbnds, con.idx, con.ub)

    def _set_con_linear(self, con: KNCon):
        variables, coefs = con.linear
        var_idxs = self._get_var_solver_idxs(variables)
        self._KNcall(kn.KN_add_con_linear_struct, con.idx, var_idxs, coefs)

    def _set_con_quadratic(self, con: KNCon):
        variables1, variables2, coefs = con.quadratic
        var1_idxs = self._get_var_solver_idxs(variables1)
        var2_idxs = self._get_var_solver_idxs(variables2)
        self._KNcall(
            kn.KN_add_con_quadratic_struct, con.idx, var1_idxs, var2_idxs, coefs
        )

    def _set_con_terms(self, con: KNCon):
        self._set_con_linear(con)
        self._set_con_quadratic(con)

    def _reset_con_bnds(self, con: KNCon):
        self._KNcall(kn.KN_set_con_lobnds, con.idx, -kn.KN_INFINITY)
        self._KNcall(kn.KN_set_con_upbnds, con.idx, kn.KN_INFINITY)

    def _reset_con_linear(self, con: KNCon):
        variables, _ = con.linear
        var_idxs = self._get_var_solver_idxs(variables)
        self._KNcall(kn.KN_del_con_linear_struct, con.idx, var_idxs)

    def _reset_con_quadratic(self, con: KNCon):
        variables1, variables2, _ = con.quadratic
        var1_idxs = self._get_var_solver_idxs(variables1)
        var2_idxs = self._get_var_solver_idxs(variables2)
        self._KNcall(kn.KN_del_con_quadratic_struct, con.idx, var1_idxs, var2_idxs)

    def _reset_con_terms(self, con: KNCon):
        self._reset_con_linear(con)
        self._reset_con_quadratic(con)

    def _set_obj_sense(self, obj: KNObj):
        self._KNcall(kn.KN_set_obj_goal, obj.sense)

    def _set_obj_constant(self, obj: KNObj):
        self._KNcall(kn.KN_add_obj_constant, obj.constant)

    def _set_obj_linear(self, obj: KNObj):
        variables, coefs = obj.linear
        if not coefs:
            return
        var_idxs = self._get_var_solver_idxs(variables)
        self._KNcall(kn.KN_add_obj_linear_struct, var_idxs, coefs)

    def _set_obj_quadratic(self, obj: KNObj):
        variables1, variables2, coefs = obj.quadratic
        if not coefs:
            return
        var1_idxs = self._get_var_solver_idxs(variables1)
        var2_idxs = self._get_var_solver_idxs(variables2)
        self._KNcall(kn.KN_add_obj_quadratic_struct, var1_idxs, var2_idxs, coefs)

    def _set_obj_terms(self, obj: KNObj):
        self._set_obj_constant(obj)
        self._set_obj_linear(obj)
        self._set_obj_quadratic(obj)

    def _reset_obj_sense(self):
        self._KNcall(kn.KN_set_obj_goal, kn.KN_OBJGOAL_MINIMIZE)

    def _reset_obj_constant(self, obj: KNObj):
        if obj.constant is None:
            return
        self._KNcall(kn.KN_del_obj_constant)

    def _reset_obj_linear(self, obj: KNObj):
        variables, _ = obj.linear
        if not variables:
            return
        var_idxs = self._get_var_solver_idxs(variables)
        self._KNcall(kn.KN_del_obj_linear_struct, var_idxs)

    def _reset_obj_quadratic(self, obj: KNObj):
        variables1, variables2, _ = obj.quadratic
        if not variables1:
            return
        var1_idxs = self._get_var_solver_idxs(variables1)
        var2_idxs = self._get_var_solver_idxs(variables2)
        self._KNcall(kn.KN_del_obj_quadratic_struct, var1_idxs, var2_idxs)

    def _reset_obj_terms(self):
        self._reset_obj_constant(self._objective)
        self._reset_obj_linear(self._objective)
        self._reset_obj_quadratic(self._objective)

    def _sync_var(self, var: KNVar, *, assign: bool = False):
        if assign:
            self._assign_var_solver_idx(var)
        self._set_var_type(var)
        self._set_var_bnds(var)

    def _clear_var(self, var: KNVar):
        self._reset_var_bnds(var)
        self._reset_var_type(var)

    def _sync_con(self, con: KNCon, *, assign: bool = False, change: bool = False):
        if assign:
            self._assign_con_solver_idx(con)
        self._set_con_bnds(con)
        if change:
            self._reset_con_terms(con)
        self._set_con_terms(con)

    def _clear_con(self, con: KNCon):
        self._reset_con_terms(con)
        self._reset_con_bnds(con)

    def _sync_obj(self, delete: bool = False):
        if delete:
            self._reset_obj_terms()
            self._reset_obj_sense()
        obj = self._objective
        if obj.obj is not None:
            self._set_obj_sense(obj)
            self._set_obj_terms(obj)

    def _clear_obj(self):
        self._reset_obj_terms()
        self._reset_obj_sense()

    def _add_var(self, var: VarData):
        self._ensure_var_not_registered(var)
        v_id = id(var)
        self._vars[v_id] = KNVar(var=var)
        self._sync_var(self._vars[v_id], assign=True)

    def _update_var(self, var: VarData):
        self._ensure_var_is_registered(var)
        v_id = id(var)
        self._vars[v_id].var = var
        self._sync_var(self._vars[v_id], assign=False)

    def _remove_var(self, var: VarData):
        self._ensure_var_is_registered(var)
        if self._objective.obj is not None:
            self._ensure_var_not_used(var, self._objective.obj.expr)
        for con in self._cons.values():
            self._ensure_var_not_used(var, con.con.expr)
        v_id = id(var)
        self._clear_var(self._vars[v_id])
        del self._vars[v_id]

    def _add_vars(self, variables: List[VarData]):
        for var in variables:
            self._add_var(var)

    def _update_vars(self, variables: List[VarData]):
        for var in variables:
            self._update_var(var)

    def _remove_vars(self, variables: List[VarData]):
        for var in variables:
            self._remove_var(var)

    def _add_con(self, con: ConstraintData):
        self._ensure_con_not_registered(con)
        c_id = id(con)
        self._cons[c_id] = KNCon(con=con)
        self._sync_con(self._cons[c_id], assign=True)

    def _update_con(self, con: ConstraintData):
        self._ensure_con_is_registered(con)
        c_id = id(con)
        self._cons[c_id].con = con
        self._sync_con(self._cons[c_id], assign=False)

    def _remove_con(self, con: ConstraintData):
        self._ensure_con_is_registered(con)
        c_id = id(con)
        self._clear_con(self._cons[c_id])
        del self._cons[c_id]

    def _add_cons(self, constraints: List[ConstraintData]):
        for con in constraints:
            self._add_con(con)

    def _update_cons(self, constraints: List[ConstraintData]):
        for con in constraints:
            self._update_con(con)

    def _remove_cons(self, constraints: List[ConstraintData]):
        for con in constraints:
            self._remove_con(con)

    def _add_param(self, param: ParamData):
        self._ensure_param_not_registered(param)
        p_id = id(param)
        self._params[p_id] = param

    def _remove_param(self, param: ParamData):
        p_id = id(param)
        self._ensure_param_is_registered(param)
        del self._params[p_id]

    def _add_params(self, parameters: List[ParamData]):
        for param in parameters:
            self._add_param(param)

    def _update_params(self):
        for var in self._vars.values():
            var.update()
            self._sync_var(var, assign=False)

        for con in self._cons.values():
            con.update()
            self._sync_con(con, assign=False, change=True)

        self._objective.update()
        self._sync_obj(delete=True)

    def _remove_params(self, parameters: List[ParamData]):
        for param in parameters:
            self._remove_param(param)

    def _set_objective(self, obj: Optional[ObjectiveData]):
        delete = self._objective.obj is not None
        self._objective.obj = obj
        self._sync_obj(delete=delete)

    def _update(self, config: KNConfig, timer: HierarchicalTimer):
        pass

    def _solve(self, config: KNConfig, timer: HierarchicalTimer) -> Results:
        stream = io.StringIO()
        ostreams = [stream, config.tee]

        with capture_output(TeeStream(*ostreams), capture_fd=False):
            self._set_default_solver_options(config)
            self._set_solver_options(config.solver_options)
            timer.start("solve")
            status = self._KNcall(kn.KN_solve)
            timer.stop("solve")

        results = self._post_solve(status, timer=timer)
        results.solver_config = config
        results.solver_log = stream.getvalue()
        return results

    def _post_solve(self, status: int, timer: HierarchicalTimer) -> Results:
        kc = self._ctx.get()
        task = KNPostSolveTask(kc=kc, status=status, timer=timer)
        task.run()
        return task.results

    def _check_values_not_none(self, values: Optional[List[float]], *, msg: str = ""):
        if values is None:
            raise ValueError(msg)

    def _get_vars(self) -> List[VarData]:
        return [v.var for v in self._vars.values()]

    def _get_cons(self) -> List[ConstraintData]:
        return [c.con for c in self._cons.values()]

    def _get_primal_sol(
        self,
        variables: Optional[Sequence[VarData]] = None,
    ) -> Mapping[VarData, float]:
        variables = list(variables or self._get_vars())
        var_idxs = self._get_var_solver_idxs(variables)
        values = self._KNcall(kn.KN_get_var_primal_values, var_idxs)
        self._check_values_not_none(values, msg="No primal solution available.")
        primal = ComponentMap()
        for val, var in zip(values, variables):
            primal[var] = val
        return primal

    def _get_dual_sol(
        self,
        constraints: Optional[Sequence[ConstraintData]] = None,
    ) -> Mapping[ConstraintData, float]:
        constraints = list(constraints or self._get_cons())
        con_idxs = self._get_con_solver_idxs(constraints)
        values = self._KNcall(kn.KN_get_con_dual_values, con_idxs)
        self._check_values_not_none(values, msg="No dual solution available.")
        dual = ComponentMap()
        for val, con in zip(values, constraints):
            dual[con] = -val
        return dual

    def _init_reduced_cost(self, variables: Sequence[VarData]):
        reduced_cost = ComponentMap()
        for var in variables:
            reduced_cost[var] = 0.0
        return reduced_cost

    def _update_reduced_cost_obj(
        self,
        reduced_cost: MutableMapping[VarData, float],
        obj: KNObj,
    ):
        variables, values = obj.linear
        for var, val in zip(variables, values):
            if var not in reduced_cost:
                continue
            reduced_cost[var] += val

    def _update_reduced_cost_con(
        self,
        reduced_cost: MutableMapping[VarData, float],
        con: KNCon,
        dual: Mapping[ConstraintData, float],
    ):
        variables, values = con.linear
        for var, val in zip(variables, values):
            if var not in reduced_cost:
                continue
            reduced_cost[var] -= val * dual[con.con]

    def _get_reduced_cost(
        self,
        variables: Optional[Sequence[VarData]] = None,
    ) -> Mapping[VarData, float]:
        variables = variables or [v.var for v in self._vars.values()]
        dual = self._get_dual_sol()

        reduced_cost = self._init_reduced_cost(variables)
        self._update_reduced_cost_obj(reduced_cost, self._objective)

        for con in self._cons.values():
            self._update_reduced_cost_con(reduced_cost, con, dual)
        return reduced_cost

    def _KNcall(self, func, *args, **kwargs):
        return func(self._ctx.get(), *args, **kwargs)

    @staticmethod
    def _get_version() -> Tuple[int, int, int]:
        return tuple(map(int, kn.__version__.split(".")))

    @staticmethod
    def _get_block_vars(block: BlockData) -> List[VarData]:
        return list(block.component_objects(Var, descend_into=True))

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
