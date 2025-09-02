from pyomo.common import unittest

from pyomo.contrib.solver.test import add_tests
from pyomo.contrib.solver.test.capability import Capability
from pyomo.contrib.solver.test.registry import SolverCapabilityRegistry
from pyomo.contrib.solver.solvers.gurobi_direct import GurobiDirect
from pyomo.contrib.solver.solvers.gurobi_persistent import GurobiPersistent
from pyomo.contrib.solver.solvers.ipopt import Ipopt


SolverCapabilityRegistry.register(
    GurobiDirect,
    Capability.OBJECTIVE_QUADRATIC,
    Capability.VARIABLE_CONTINUOUS,
    Capability.VARIABLE_INTEGER,
    Capability.VARIABLE_BINARY,
    Capability.CONSTRAINT_LINEAR,
    Capability.CONSTRAINT_QUADRATIC_GE,
    Capability.SOLUTION_VARIABLE_PRIMAL,
    Capability.SOLUTION_VARIABLE_REDUCED_COST,
    Capability.SOLUTION_CONSTRAINT_DUAL,
)
SolverCapabilityRegistry.register(
    GurobiPersistent,
    Capability.OBJECTIVE_QUADRATIC,
    Capability.VARIABLE_CONTINUOUS,
    Capability.VARIABLE_INTEGER,
    Capability.VARIABLE_BINARY,
    Capability.CONSTRAINT_LINEAR,
    Capability.CONSTRAINT_QUADRATIC_GE,
    Capability.SOLUTION_VARIABLE_PRIMAL,
    Capability.SOLUTION_VARIABLE_REDUCED_COST,
    Capability.SOLUTION_CONSTRAINT_DUAL,
)
SolverCapabilityRegistry.register(
    Ipopt,
    Capability.OBJECTIVE_NONLINEAR,
    Capability.VARIABLE_CONTINUOUS,
    Capability.VARIABLE_INTEGER,
    Capability.VARIABLE_BINARY,
    Capability.CONSTRAINT_NONLINEAR,
    Capability.SOLUTION_VARIABLE_PRIMAL,
    Capability.SOLUTION_VARIABLE_REDUCED_COST,
    Capability.SOLUTION_CONSTRAINT_DUAL,
)


class TestSolvers(unittest.TestCase):
    pass


add_tests(TestSolvers, GurobiDirect)
add_tests(TestSolvers, GurobiPersistent)
add_tests(TestSolvers, Ipopt, warn_unavailable=True)

