from collections.abc import Set
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, Optional


class Capability(Enum):
    """
    Solver capabilities enumeration.

    Each capability represents a specific feature that a solver may or may not support.
    The testing framework uses these to determine which tests can run on which solvers.
    """

    OBJECTIVE_SENSE = 0
    OBJECTIVE_LINEAR = 1
    OBJECTIVE_QUADRATIC = 2
    OBJECTIVE_NONLINEAR = 3
    OBJECTIVE_MULTI = 4

    VARIABLE_CONTINUOUS = 100
    VARIABLE_BINARY = 101
    VARIABLE_INTEGER = 102
    VARIABLE_SEMICONTINUOUS = 103
    VARIABLE_SEMIINTEGER = 104

    CONSTRAINT_LINEAR_EQ = 200
    CONSTRAINT_LINEAR_GE = 201
    CONSTRAINT_LINEAR_LE = 202
    CONSTRAINT_LINEAR_LG = 203
    CONSTRAINT_LINEAR = 204

    CONSTRAINT_QUADRATIC_EQ = 250
    CONSTRAINT_QUADRATIC_GE = 251
    CONSTRAINT_QUADRATIC_LE = 252
    CONSTRAINT_QUADRATIC_LG = 253
    CONSTRAINT_QUADRATIC = 254

    CONSTRAINT_NONLINEAR_EQ = 300
    CONSTRAINT_NONLINEAR_GE = 301
    CONSTRAINT_NONLINEAR_LE = 302
    CONSTRAINT_NONLINEAR_LG = 303
    CONSTRAINT_NONLINEAR = 304

    CONSTRAINT_SOS_ONE = 350
    CONSTRAINT_SOS_TWO = 351
    CONSTRAINT_SOS = 352

    CONSTRAINT_CONIC = 400
    CONSTRAINT_COMPLEMENTARITY = 401

    SOLUTION_VARIABLE_PRIMAL = 500
    SOLUTION_VARIABLE_DUAL = 501
    SOLUTION_VARIABLE_REDUCED_COST = 502
    SOLUTION_CONSTRAINT_DUAL = 503
    SOLUTION_CONSTRAINT_SLACK = 504


class CapabilityCategory(Enum):
    """High-level categories for organizing capabilities."""

    OBJECTIVE = "objective"
    VARIABLE = "variable"
    CONSTRAINT = "constraint"
    SOLUTION = "solution"


@dataclass(frozen=True)
class CapabilityMeta:
    """Metadata about a capability."""

    name: str
    description: str
    category: CapabilityCategory
    implies: Set[Capability] = field(default_factory=set)


@dataclass
class CapabilityRegistryClass:
    """
    Registry that maintains capability metadata and relationships.
    """

    _meta: Dict[Capability, CapabilityMeta] = field(default_factory=dict)

    def register(
        self,
        cap: Capability,
        name: str,
        description: str,
        category: CapabilityCategory,
        implies: Optional[Set[Capability]] = None,
    ):
        if implies is None:
            implies = set()
        self._meta[cap] = CapabilityMeta(name, description, category, implies)

    def get_meta(self, cap: Capability) -> Optional[CapabilityMeta]:
        return self._meta.get(cap)

    def resolve_implications(self, caps: Set[Capability]) -> Set[Capability]:
        """Recursively resolve all implied capabilities."""
        resolved = set(caps)
        to_process = list(caps)
        while to_process:
            current = to_process.pop()
            meta = self.get_meta(current)
            if meta:
                for implied in meta.implies:
                    if implied not in resolved:
                        resolved.add(implied)
                        to_process.append(implied)
        return resolved


CapabilityRegistry = CapabilityRegistryClass()

CapabilityRegistry.register(
    Capability.OBJECTIVE_SENSE,
    name="Objective Sense",
    description="Supports having an objective sense.",
    category=CapabilityCategory.OBJECTIVE,
)
CapabilityRegistry.register(
    Capability.OBJECTIVE_LINEAR,
    name="Objective Linear",
    description="Supports having a linear objective.",
    category=CapabilityCategory.OBJECTIVE,
    implies={Capability.OBJECTIVE_SENSE},
)
CapabilityRegistry.register(
    Capability.OBJECTIVE_QUADRATIC,
    name="Objective Quadratic",
    description="Supports having a quadratic objective.",
    category=CapabilityCategory.OBJECTIVE,
    implies={Capability.OBJECTIVE_LINEAR},
)
CapabilityRegistry.register(
    Capability.OBJECTIVE_NONLINEAR,
    name="Objective Nonlinear",
    description="Supports having a nonlinear objective.",
    category=CapabilityCategory.OBJECTIVE,
    implies={Capability.OBJECTIVE_QUADRATIC},
)
CapabilityRegistry.register(
    Capability.OBJECTIVE_MULTI,
    name="Objective Multi",
    description="Supports having a multi-objective.",
    category=CapabilityCategory.OBJECTIVE,
)
CapabilityRegistry.register(
    Capability.VARIABLE_CONTINUOUS,
    name="Variable Continuous",
    description="Supports continuous variables.",
    category=CapabilityCategory.VARIABLE,
)
CapabilityRegistry.register(
    Capability.VARIABLE_BINARY,
    name="Variable Binary",
    description="Supports binary variables.",
    category=CapabilityCategory.VARIABLE,
)
CapabilityRegistry.register(
    Capability.VARIABLE_INTEGER,
    name="Variable Integer",
    description="Supports integer variables.",
    category=CapabilityCategory.VARIABLE,
)
CapabilityRegistry.register(
    Capability.VARIABLE_SEMICONTINUOUS,
    name="Variable Semicontinuous",
    description="Supports semicontinuous variables.",
    category=CapabilityCategory.VARIABLE,
)
CapabilityRegistry.register(
    Capability.VARIABLE_SEMIINTEGER,
    name="Variable Semiinteger",
    description="Supports semi-integer variables.",
    category=CapabilityCategory.VARIABLE,
)
CapabilityRegistry.register(
    Capability.CONSTRAINT_LINEAR_EQ,
    name="Constraint Linear Equality",
    description="Supports linear equality constraints.",
    category=CapabilityCategory.CONSTRAINT,
)
CapabilityRegistry.register(
    Capability.CONSTRAINT_LINEAR_LE,
    name="Constraint Linear Less Than or Equal",
    description="Supports linear less than or equal constraints.",
    category=CapabilityCategory.CONSTRAINT,
)
CapabilityRegistry.register(
    Capability.CONSTRAINT_LINEAR_GE,
    name="Constraint Linear Greater Than or Equal",
    description="Supports linear greater than or equal constraints.",
    category=CapabilityCategory.CONSTRAINT,
)
CapabilityRegistry.register(
    Capability.CONSTRAINT_LINEAR_LG,
    name="Constraint Linear Range",
    description="Supports linear range constraints.",
    category=CapabilityCategory.CONSTRAINT,
    implies={Capability.CONSTRAINT_LINEAR_LE, Capability.CONSTRAINT_LINEAR_GE},
)
CapabilityRegistry.register(
    Capability.CONSTRAINT_LINEAR,
    name="Constraint Linear",
    description="Supports linear constraints.",
    category=CapabilityCategory.CONSTRAINT,
    implies={Capability.CONSTRAINT_LINEAR_EQ, Capability.CONSTRAINT_LINEAR_LG},
)
CapabilityRegistry.register(
    Capability.CONSTRAINT_QUADRATIC_EQ,
    name="Constraint Quadratic Equality",
    description="Supports quadratic equality constraints.",
    category=CapabilityCategory.CONSTRAINT,
    implies={Capability.CONSTRAINT_LINEAR_EQ},
)
CapabilityRegistry.register(
    Capability.CONSTRAINT_QUADRATIC_LE,
    name="Constraint Quadratic Less Than or Equal",
    description="Supports quadratic less than or equal constraints.",
    category=CapabilityCategory.CONSTRAINT,
    implies={Capability.CONSTRAINT_LINEAR_LE},
)
CapabilityRegistry.register(
    Capability.CONSTRAINT_QUADRATIC_GE,
    name="Constraint Quadratic Greater Than or Equal",
    description="Supports quadratic greater than or equal constraints.",
    category=CapabilityCategory.CONSTRAINT,
    implies={Capability.CONSTRAINT_LINEAR_GE},
)
CapabilityRegistry.register(
    Capability.CONSTRAINT_QUADRATIC_LG,
    name="Constraint Quadratic Range",
    description="Supports quadratic range constraints.",
    category=CapabilityCategory.CONSTRAINT,
    implies={
        Capability.CONSTRAINT_LINEAR_LG,
        Capability.CONSTRAINT_QUADRATIC_LE,
        Capability.CONSTRAINT_QUADRATIC_GE,
    },
)
CapabilityRegistry.register(
    Capability.CONSTRAINT_QUADRATIC,
    name="Constraint Quadratic",
    description="Supports quadratic constraints.",
    category=CapabilityCategory.CONSTRAINT,
    implies={
        Capability.CONSTRAINT_LINEAR,
        Capability.CONSTRAINT_QUADRATIC_EQ,
        Capability.CONSTRAINT_QUADRATIC_LG,
    },
)
CapabilityRegistry.register(
    Capability.CONSTRAINT_NONLINEAR_EQ,
    name="Constraint Nonlinear Equality",
    description="Supports nonlinear equality constraints.",
    category=CapabilityCategory.CONSTRAINT,
    implies={Capability.CONSTRAINT_QUADRATIC_EQ},
)
CapabilityRegistry.register(
    Capability.CONSTRAINT_NONLINEAR_LE,
    name="Constraint Nonlinear Less Than or Equal",
    description="Supports nonlinear less than or equal constraints.",
    category=CapabilityCategory.CONSTRAINT,
    implies={Capability.CONSTRAINT_QUADRATIC_LE},
)
CapabilityRegistry.register(
    Capability.CONSTRAINT_NONLINEAR_GE,
    name="Constraint Nonlinear Greater Than or Equal",
    description="Supports nonlinear greater than or equal constraints.",
    category=CapabilityCategory.CONSTRAINT,
    implies={Capability.CONSTRAINT_QUADRATIC_GE},
)
CapabilityRegistry.register(
    Capability.CONSTRAINT_NONLINEAR_LG,
    name="Constraint Nonlinear Range",
    description="Supports nonlinear range constraints.",
    category=CapabilityCategory.CONSTRAINT,
    implies={
        Capability.CONSTRAINT_QUADRATIC_LG,
        Capability.CONSTRAINT_NONLINEAR_LE,
        Capability.CONSTRAINT_NONLINEAR_GE,
    },
)
CapabilityRegistry.register(
    Capability.CONSTRAINT_NONLINEAR,
    name="Constraint Nonlinear",
    description="Supports nonlinear constraints.",
    category=CapabilityCategory.CONSTRAINT,
    implies={
        Capability.CONSTRAINT_QUADRATIC,
        Capability.CONSTRAINT_NONLINEAR_LG,
        Capability.CONSTRAINT_NONLINEAR_EQ,
    },
)
CapabilityRegistry.register(
    Capability.CONSTRAINT_SOS_ONE,
    name="Constraint SOS1",
    description="Supports SOS1 constraints.",
    category=CapabilityCategory.CONSTRAINT,
)
CapabilityRegistry.register(
    Capability.CONSTRAINT_SOS_TWO,
    name="Constraint SOS2",
    description="Supports SOS2 constraints.",
    category=CapabilityCategory.CONSTRAINT,
)
CapabilityRegistry.register(
    Capability.CONSTRAINT_SOS,
    name="Constraint SOS",
    description="Supports SOS constraints.",
    category=CapabilityCategory.CONSTRAINT,
    implies={Capability.CONSTRAINT_SOS_ONE, Capability.CONSTRAINT_SOS_TWO},
)
CapabilityRegistry.register(
    Capability.CONSTRAINT_CONIC,
    name="Constraint Conic",
    description="Supports conic constraints.",
    category=CapabilityCategory.CONSTRAINT,
)
CapabilityRegistry.register(
    Capability.CONSTRAINT_COMPLEMENTARITY,
    name="Constraint Complementarity",
    description="Supports complementarity constraints.",
    category=CapabilityCategory.CONSTRAINT,
)
CapabilityRegistry.register(
    Capability.SOLUTION_VARIABLE_PRIMAL,
    name="Solution Variable Primal",
    description="Supports primal variable solutions.",
    category=CapabilityCategory.SOLUTION,
)
CapabilityRegistry.register(
    Capability.SOLUTION_VARIABLE_DUAL,
    name="Solution Variable Dual",
    description="Supports dual variable solutions.",
    category=CapabilityCategory.SOLUTION,
)
CapabilityRegistry.register(
    Capability.SOLUTION_VARIABLE_REDUCED_COST,
    name="Solution Variable Reduced Cost",
    description="Supports reduced cost for variable solutions.",
    category=CapabilityCategory.SOLUTION,
)
CapabilityRegistry.register(
    Capability.SOLUTION_CONSTRAINT_DUAL,
    name="Solution Constraint Dual",
    description="Supports dual constraint solutions.",
    category=CapabilityCategory.SOLUTION,
)
CapabilityRegistry.register(
    Capability.SOLUTION_CONSTRAINT_SLACK,
    name="Solution Constraint Slack",
    description="Supports slack for constraint solutions.",
    category=CapabilityCategory.SOLUTION,
)
