from __future__ import annotations

import unittest

import numpy as np

from core.sim.primitive_task_benchmark import (
    atomic_delta,
    continuous_delta,
    line_constraint_errors,
    summarize_task,
)


class PrimitiveTaskBenchmarkTest(unittest.TestCase):
    def test_atomic_is_fixed_axis_while_direct_can_finish_residual(self):
        atomic, token = atomic_delta([0.005, 0.005, 0.0], 0.01)
        direct, label = continuous_delta([0.005, 0.005, 0.0], 0.01)
        np.testing.assert_allclose(atomic, [0.01, 0.0, 0.0])
        np.testing.assert_allclose(direct, [0.005, 0.005, 0.0])
        self.assertEqual(token, "MV_FWD")
        self.assertEqual(label, "DIRECT_XYZ")

    def test_diagonal_staircase_violates_narrow_line_constraint(self):
        direct = [[0.0, 0.0, 0.0], [0.005, 0.005, 0.0], [0.01, 0.01, 0.0]]
        atomic = [[0.0, 0.0, 0.0], [0.01, 0.0, 0.0], [0.01, 0.01, 0.0]]
        target = [0.01, 0.01, 0.0]
        self.assertLess(float(line_constraint_errors(direct, direct[0], target).max()), 1e-9)
        self.assertGreater(
            float(line_constraint_errors(atomic, atomic[0], target).max()), 0.007
        )

    def test_task_success_requires_endpoint_and_constraint(self):
        trace = [[0.0, 0.0, 0.0], [0.01, 0.0, 0.0], [0.01, 0.01, 0.0]]
        errors = line_constraint_errors(trace, trace[0], trace[-1])
        result = summarize_task(
            task="narrow_diagonal_wipe",
            controller="atomic",
            trial=0,
            trace=trace,
            target=trace[-1],
            target_offset=trace[-1],
            decisions=2,
            endpoint_tolerance_m=0.001,
            constraint_errors=errors,
            constraint_tolerance_m=0.004,
            required_constraint_pass_rate=0.95,
        )
        self.assertTrue(result.endpoint_reached)
        self.assertFalse(result.success)


if __name__ == "__main__":
    unittest.main()
