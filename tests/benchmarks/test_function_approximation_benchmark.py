from __future__ import annotations

import json
import os
import unittest

from dacbench.benchmarks import FunctionApproximationBenchmark
from dacbench.envs import FunctionApproximationEnv, FunctionApproximationInstance


class TestFunctionApproximationBenchmark(unittest.TestCase):
    def test_get_env(self):
        bench = FunctionApproximationBenchmark()
        env = bench.get_environment()
        assert issubclass(type(env), FunctionApproximationEnv)

    def test_save_conf(self):
        bench = FunctionApproximationBenchmark()
        del bench.config["config_space"]
        bench.save_config("test_conf.json")
        with open("test_conf.json") as fp:
            recovered = json.load(fp)
        for k in bench.config:
            assert k in recovered
        os.remove("test_conf.json")

    def test_from_to_json(self):
        bench = FunctionApproximationBenchmark()
        restored_bench = FunctionApproximationBenchmark.from_json(bench.to_json())
        assert bench == restored_bench

    def test_read_instances(self):
        bench = FunctionApproximationBenchmark()
        bench.read_instance_set()
        assert len(bench.config.instance_set.keys()) == 300
        assert isinstance(bench.config.instance_set[0], FunctionApproximationInstance)
        first_inst = bench.config.instance_set[0]

        bench2 = FunctionApproximationBenchmark()
        env = bench2.get_environment()
        assert isinstance(bench.config.instance_set[0], FunctionApproximationInstance)
        assert env.instance_set[0].functions[0].a == first_inst.functions[0].a    
        assert env.instance_set[0].functions[0].b == first_inst.functions[0].b    
        assert len(env.instance_set.keys()) == 300
