# flake8: noqa: F401
import importlib
import warnings

# from dacbench.benchmarks.fast_downward_benchmark import FastDownwardBenchmark
from dacbench.benchmarks.function_approximation_benchmark import (
    FunctionApproximationBenchmark,
)
from dacbench.benchmarks.luby_benchmark import LubyBenchmark
from dacbench.benchmarks.toysgd_benchmark import ToySGDBenchmark

__all__ = [
    "LubyBenchmark",
    "FunctionApproximationBenchmark",
    "ToySGDBenchmark",
    #    "FastDownwardBenchmark",
]

modcma_spec = importlib.util.find_spec("modcma")
found = modcma_spec is not None
if found:
    from dacbench.benchmarks.cma_benchmark import CMAESBenchmark

    __all__.append("CMAESBenchmark")
else:
    warnings.warn(  # noqa: B028
        "CMA-ES Benchmark not installed. If you want to use this benchmark, "
        "please follow the installation guide."
    )

sgd_spec = importlib.util.find_spec("torch")
found = sgd_spec is not None
if found:
    from dacbench.benchmarks.sgd_benchmark import SGDBenchmark

    __all__.append("SGDBenchmark")
else:
    warnings.warn(  # noqa: B028
        "SGD Benchmark not installed. If you want to use this benchmark, "
        "please follow the installation guide."
    )

theory_spec = importlib.util.find_spec("uuid")
found = theory_spec is not None
if found:
    from dacbench.benchmarks.theory_benchmark import RLSTheoryBenchmark, OLLGATheoryBenchmark

    __all__.append("TheoryBenchmark")
else:
    warnings.warn(  # noqa: B028
        "Theory Benchmark not installed. If you want to use this benchmark, "
        "please follow the installation guide."
    )
