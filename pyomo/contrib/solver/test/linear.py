from pyomo.contrib.solver.common.results import SolutionStatus
from pyomo.common import unittest
from pyomo.contrib.solver.common.base import SolverBase
import pyomo.environ as pyo


from .registry import SolverCapabilityRegistry, SolverTestRegistry
from .capability import Capability


@SolverTestRegistry.requires(Capability.OBJECTIVE_LINEAR)
@SolverTestRegistry.requires(Capability.VARIABLE_CONTINUOUS)
@SolverTestRegistry.requires(Capability.CONSTRAINT_LINEAR_EQ)
@SolverTestRegistry.requires(Capability.SOLUTION_VARIABLE_PRIMAL)
@SolverTestRegistry.tags("linear", "basic")
def test_linear_equality(test_case: unittest.TestCase, opt_cls: type[SolverBase]):
    """
    A simple linear programming test with equality constraints.
    Minimize: x + y
    Subject to: x + 2y = 1
                x, y >= 0
    Optimal solution: x = 0, y = 0.5, objective = 0.5
    """

    m = pyo.ConcreteModel()
    m.x = pyo.Var(bounds=(0, None))
    m.y = pyo.Var(bounds=(0, None))
    m.obj = pyo.Objective(expr=m.x + m.y)
    m.c = pyo.Constraint(expr=m.x + 2 * m.y == 1)

    opt = opt_cls()

    results = opt.solve(m)

    test_case.assertEqual(results.solution_status, SolutionStatus.optimal)
    test_case.assertEqual(results.incumbent_objective, 0.5)
    test_case.assertAlmostEqual(pyo.value(m.x), 0.0)
    test_case.assertAlmostEqual(pyo.value(m.y), 0.5)

    if SolverCapabilityRegistry.supports(
        opt_cls, Capability.SOLUTION_VARIABLE_REDUCED_COST
    ):
        reduced_costs = results.solution_loader.get_reduced_costs()
        test_case.assertAlmostEqual(reduced_costs[m.x], 0.5)
        test_case.assertAlmostEqual(reduced_costs[m.y], 0.0)
