# flake8: noqa: F401
import importlib
import warnings

# from dacbench.envs.fast_downward import FastDownwardEnv
from dacbench.envs.function_approximation import (
    FunctionApproximationEnv,
    FunctionApproximationInstance,
)
from dacbench.envs.luby import LubyEnv, LubyInstance, luby_gen
from dacbench.envs.theory import RLSTheoryEnv, OLLGATheoryEnv
from dacbench.envs.toysgd import ToySGDEnv, ToySGDInstance

__all__ = [
    "LubyEnv",
    "LubyInstance",
    "luby_gen",
    "FunctionApproximationEnv",
    "FunctionApproximationInstance",
    #   "FastDownwardEnv",
    "ToySGDEnv",
    "ToySGDInstance",
    "RLSTheoryEnv",
    "OLLGATheoryEnv",
]

modcma_spec = importlib.util.find_spec("modcma")
found = modcma_spec is not None
if found:
    from dacbench.envs.cma_es import CMAESEnv, CMAESInstance

    __all__.append("CMAESEnv")
    __all__.append("CMAESInstance")
else:
    warnings.warn(  # noqa: B028
        "CMA-ES Benchmark not installed. If you want to use this benchmark, "
        "please follow the installation guide."
    )

sgd_spec = importlib.util.find_spec("torch")
found = sgd_spec is not None
if found:
    from dacbench.envs.sgd import SGDEnv, SGDInstance

    __all__.append("SGDEnv")
    __all__.append("SGDInstance")
else:
    warnings.warn(  # noqa: B028
        "SGD Benchmark not installed. If you want to use this benchmark, "
        "please follow the installation guide."
    )
