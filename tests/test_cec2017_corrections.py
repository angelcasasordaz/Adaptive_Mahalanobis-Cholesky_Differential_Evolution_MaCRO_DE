from concurrent.futures import ProcessPoolExecutor
import json
import multiprocessing
import os
from pathlib import Path
import pickle
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from opfunu.cec_based import cec2017
from cec2017_corrections import (
    CORRECTED_CLASSES, OBJECTIVE_REVISIONS, LOWER_BOUND_TOLERANCE,
    assert_objective_lower_bound, corrected_schwefel,
)
from cec2017_gpu import CEC2017GpuObjective, CEC2017_GPU_CANDIDATES, verify_gpu_objective
from objective_evaluation import ObjectiveSpec, _evaluate_objective_chunk
from validate_cec2017_corrections import reference_fixture


def full_args():
    import main
    with patch("sys.argv", ["main.py", "--experiment-mode", "full"]):
        args = main.parse_args()
    main.apply_experiment_mode(args, "full")
    args.resolved_gpu_batch_size = args.estimated_gpu_batch_capacity = 1
    return args


class ObjectiveTests(unittest.TestCase):
    def test_optimum_and_reference_c(self):
        for name, cls in CORRECTED_CLASSES.items():
            b = cls(ndim=30)
            points, expected = reference_fixture(b)
            scalar = np.array([b.evaluate(x) for x in points])
            numpy = CEC2017GpuObjective(name, b, np).evaluate(points)
            np.testing.assert_allclose(scalar, expected, rtol=5e-11, atol=1e-7)
            np.testing.assert_allclose(numpy, expected, rtol=5e-11, atol=1e-7)
            self.assertAlmostEqual(b.evaluate(b.x_global), b.f_global, places=7)
            # Historical completed FULL failure is the final frozen vector.
            raw = getattr(cec2017, name)(ndim=30)
            self.assertLess(raw.evaluate(points[-1]), b.f_global - 1e-7)
            self.assertGreaterEqual(scalar[-1], b.f_global - 1e-7)

    def test_gpu_reference_c_and_batch_shapes(self):
        try:
            import cupy as cp
            cp.zeros(1).sum().item()
        except Exception as exc:
            if os.environ.get("CEC_REQUIRE_GPU") == "1":
                self.fail(f"Required GPU unavailable: {exc}")
            self.skipTest(str(exc))
        for name, cls in CORRECTED_CLASSES.items():
            b = cls(ndim=30)
            points, expected = reference_fixture(b)
            gpu = CEC2017GpuObjective(name, b, cp)
            actual = cp.asnumpy(gpu.evaluate(cp.asarray(points[None, ...])))
            np.testing.assert_allclose(actual[0], expected, rtol=5e-11, atol=1e-7)
            self.assertAlmostEqual(float(gpu.evaluate(cp.asarray(b.x_global))), b.f_global, places=7)
            with self.assertRaisesRegex(RuntimeError, "backend=cupy"):
                assert_objective_lower_bound(name, cp.asarray([b.f_global-1]),
                                             cp.asarray(points[:1]), b.f_global, xp=cp, backend="cupy")

    def test_exact_schwefel_branches(self):
        # Test modulo discontinuities directly, without inverse-rotation rounding.
        z = np.array([-1500., -1000., -500., 0., 420.9687462275036, 500., 1000., 1500.])
        inputs = np.concatenate([z-1e-7, z, z+1e-7]) - 420.9687462275036
        b = CORRECTED_CLASSES["F92017"](ndim=30)
        objective = CEC2017GpuObjective("F92017", b, np)
        for row in (inputs, inputs.reshape(3, -1)[0]):
            self.assertAlmostEqual(float(objective._schwefel(row, corrected=True)),
                                   corrected_schwefel(row), places=8)

    def test_guard_diagnostic_and_no_clamp(self):
        b = CORRECTED_CLASSES["F92017"](ndim=30)
        values = np.array([900.-LOWER_BOUND_TOLERANCE/2])
        original = values.copy()
        assert_objective_lower_bound("F92017", values, b.x_global, 900.)
        np.testing.assert_array_equal(values, original)
        for value in (899., np.nan, np.inf):
            with self.assertRaisesRegex(RuntimeError, r"function=F92017.*revision=.*backend=.*D=30.*f_global=.*x="):
                assert_objective_lower_bound("F92017", value, b.x_global, 900.)
        with patch("cec2017_corrections.corrected_schwefel", return_value=-1.):
            with self.assertRaises(RuntimeError):
                b.evaluate(b.x_global)
        vectorized = CEC2017GpuObjective("F92017", b, np)
        with patch.object(vectorized, "_schwefel", return_value=-1.):
            with self.assertRaises(RuntimeError):
                vectorized.evaluate(b.x_global)

    def test_spawned_cpu_and_legacy_spec_routing(self):
        with ProcessPoolExecutor(max_workers=1, mp_context=multiprocessing.get_context("spawn")) as pool:
            for name, cls in CORRECTED_CLASSES.items():
                b = cls(ndim=30)
                points, expected = reference_fixture(b)
                for module in (cls.__module__, "opfunu.cec_based.cec2017"):
                    task = (ObjectiveSpec(module, name, 30), points[-2:])
                    actual = pool.submit(_evaluate_objective_chunk, task).result(timeout=30)
                    np.testing.assert_allclose(actual, expected[-2:], rtol=5e-11, atol=1e-7)

    def test_upstream_verification_uses_corrected_scalar(self):
        for name in CORRECTED_CLASSES:
            raw = getattr(cec2017, name)(ndim=30)
            objective = CEC2017GpuObjective(name, raw, np)
            report = verify_gpu_objective(objective, raw, np.asarray, lambda: None, random_points=32)
            self.assertTrue(report.verified, report)

    def test_other_classes_and_objectives_unchanged(self):
        import main
        functions = main.discover_benchmark_functions("CEC2017", 30)
        for name in functions:
            if name not in CORRECTED_CLASSES:
                self.assertIs(functions[name], getattr(cec2017, name))
        for name in CEC2017_GPU_CANDIDATES - CORRECTED_CLASSES.keys():
            b = getattr(cec2017, name)(ndim=30)
            obj = CEC2017GpuObjective(name, b, np)
            self.assertTrue(verify_gpu_objective(obj, b, np.asarray, lambda: None, random_points=16).verified, name)


class CacheAndRerunTests(unittest.TestCase):
    def test_function_specific_signature_and_metadata(self):
        import main
        args = full_args()
        signatures = json.loads((Path(__file__).parent / "fixtures/cec2017_legacy_full_signatures.json").read_text())
        for opt, legacy in signatures.items():
            for n in range(1, 30):
                name = f"F{n}2017"
                signature = main.build_cache_signature(args, opt, name)
                current = main.checkpoint_metadata(args, signature, name, opt, 0, 1234)
                old = {k: v for k, v in current.items() if k != "objective_implementation_revision"}
                self.assertTrue(main.checkpoint_metadata_compatible(current, current))
                self.assertEqual(main.checkpoint_metadata_compatible(old, current), name not in CORRECTED_CLASSES)
                self.assertEqual(signature == legacy, name not in CORRECTED_CLASSES)
        args.benchmark = "CEC2014"
        self.assertEqual(main.build_cache_signature(args, "DE", "F92017"), main.build_cache_signature(args, "DE"))

    def test_legacy_fallback_rejects_invalid_revision(self):
        import main
        args = full_args()
        with tempfile.TemporaryDirectory() as directory:
            paths = main.Paths("EXP009", "full", directory, directory, directory)
            for name in CORRECTED_CLASSES:
                expected = main.checkpoint_metadata(args, "test", name, "DE", 0, 1234)
                old = {k: v for k, v in expected.items() if k != "objective_implementation_revision"}
                path = Path(main.run_checkpoint_path(paths, "old", name, "DE", 0))
                path.parent.mkdir(parents=True)
                payload = {"metadata": old, "output": dict(best_fitness=0, best_solution=[], runtime=0, curve=[])}
                path.write_bytes(pickle.dumps(payload))
                self.assertIsNone(main.load_run_checkpoint(path, expected, report_invalid=False))
                self.assertIsNone(main.load_compatible_cpu_checkpoint(paths, name, "DE", 0, expected))
                self.assertIsNone(main.compatible_checkpoint_path([paths], str(path), expected, name, "DE", 0))

    def test_rerun_configuration_and_preflight_blocks_solve(self):
        import main
        with patch("sys.argv", ["main.py"]):
            args = main.parse_args()
        main.apply_experiment_mode(args, "full")
        functions = main.discover_benchmark_functions(args.benchmark, args.dims)
        self.assertEqual(main.select_experiment_functions(args, functions), ["F92017", "F212017"])
        self.assertEqual(args.optimizers, main.DEFAULT_OPTIMIZERS)
        self.assertEqual((args.runs, args.dims, args.pop_size, args.epochs), (30, 30, 50, 2000))
        with tempfile.TemporaryDirectory() as directory:
            args.output_root = directory
            with patch("validate_cec2017_corrections.validate_corrections", side_effect=RuntimeError("blocked")), \
                    patch("main.audit_checkpoint_cache") as audit, patch("main.build_optimizer") as optimizer:
                with self.assertRaisesRegex(RuntimeError, "blocked"):
                    main.run_experiment(args)
                audit.assert_not_called()
                optimizer.assert_not_called()


if __name__ == "__main__":
    unittest.main()
