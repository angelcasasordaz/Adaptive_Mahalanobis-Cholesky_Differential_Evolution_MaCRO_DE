"""Tiny smoke validation only; all generated artifacts live in a temporary root."""
import argparse
from pathlib import Path
import sys
import tempfile
from unittest.mock import patch

import numpy as np
import pandas as pd

import main as framework
from distance_metrics import nearest_k_mask, squared_distance_from_mean
from gpu_batching import BatchedDEEngine
from macro_de_distance_variants import MaCRO_DE_Distance, BatchedMaCRODistance
from macro_de_optimizer import MaCRO_DE
from distance_ablation_reporting import plot_archive
from compute_backend import ComputeBackend


def configuration(root):
    with patch.object(sys, "argv", ["main.py", "--experiment-mode", "distance_ablation",
                                   "--output-root", root, "--epochs", "2", "--runs", "2",
                                   "--pop-size", "10", "--parallel", "no",
                                   "--objective-evaluation", "serial", "--n-workers", "1"]):
        args = framework.parse_args()
    framework.apply_experiment_mode(args, "distance_ablation")
    return args


def main():
    # Exact cardinality, deterministic boundary ties, and independent batch k.
    distances = np.array([[2., 1., 1., 3.], [4., 4., 4., 4.]])
    for k in range(5):
        selected = nearest_k_mask(distances, np.array([k, 4-k]), np)
        np.testing.assert_array_equal(selected.sum(axis=-1), [k, 4-k])
        for row, count in enumerate([k, 4-k]):
            expected = np.zeros(4, dtype=bool)
            expected[np.argsort(distances[row], kind="stable")[:count]] = True
            np.testing.assert_array_equal(selected[row], expected)
    with tempfile.TemporaryDirectory(prefix="distance-ablation-smoke-") as root:
        args = configuration(root)
        fmap = framework.discover_benchmark_functions("CEC2017", 30)
        assert framework.select_experiment_functions(args, fmap) == framework.ABLATION_FUNCTIONS
        assert len(framework.ABLATION_FUNCTIONS) == 8
        args.function_map = fmap
        args.gpu_objectives = {}
        args.vectorized_cpu_objectives = {}
        variants = list(framework.optimizer_experiment_configurations(args))
        assert len({framework.build_cache_signature(v, name) for _, name, v in variants}) == 5
        control = variants[2][2]
        normal = argparse.Namespace(**vars(control))
        normal.experiment_mode = "full"
        assert framework.build_cache_signature(control, "MaCRO-DE") != framework.build_cache_signature(normal, "MaCRO-DE")
        assert type(framework.build_optimizer("MaCRO-DE", control)) is MaCRO_DE
        assert type(framework.build_batched_engine("F12017", "MaCRO-DE", control)) is BatchedDEEngine
        # Bit-for-bit control histories, solutions, and next RNG draws.
        models = [framework.build_optimizer("MaCRO-DE", a) for a in (normal, control)]
        for model in models:
            _, problem = framework.build_problem(fmap["F12017"], 30)
            model.solve(problem, seed=args.seed_base)
        np.testing.assert_array_equal(models[0].history.list_global_best_fit, models[1].history.list_global_best_fit)
        np.testing.assert_array_equal(models[0].g_best.solution, models[1].g_best.solution)
        np.testing.assert_array_equal(models[0].generator.random(8), models[1].generator.random(8))
        assert models[0].nfe_counter == (args.epochs + 1)*args.pop_size+1
        # Analytical norms, batched shapes, and inherited donor-pool semantics.
        points = np.array([[-3., -4.], [0., 0.], [3., 4.]])
        for metric, norm in [("chebyshev", 4.), ("euclidean", 5.),
                             ("manhattan", 7.), ("minkowski", 91.**(1/3))]:
            actual = squared_distance_from_mean(points, np, metric, 3.)
            np.testing.assert_allclose(actual, [norm**2, 0., norm**2])
            _, _, variant = next(v for v in variants if v[2].distance_metric == metric)
            model = framework.build_optimizer("MaCRO-DE", variant)
            assert isinstance(model, MaCRO_DE_Distance)
            assert model.evolve.__func__ is MaCRO_DE.evolve
            assert model._mutation_pool.__func__ is MaCRO_DE._mutation_pool
            model.problem = argparse.Namespace(n_dims=2)
            model.mahalanobis_threshold = 0.1
            pool = np.array([[0., 0.], [0., 0.], [0., 0.], [-10., 0.], [10., 0.], [20., 0.], [-20., 0.]])
            np.testing.assert_array_equal(model._mutation_pool(pool, 1.), pool[:3])
            np.testing.assert_array_equal(model._mutation_pool(pool, 0.), pool[3:])
            # Every possible k, including empty/all and minimum-three fallback.
            population = np.random.default_rng(42).normal(size=(10, 2)) * [1., 20.]
            mahalanobis = ComputeBackend("cpu").mahalanobis_distances(population, 2, "cholesky")
            thresholds = [-np.inf, *np.sort(mahalanobis)]
            distances = squared_distance_from_mean(population, np, metric, 3.)
            model.generator = np.random.default_rng(123)
            rng_state = model.generator.bit_generator.state
            for threshold in thresholds:
                k = np.count_nonzero(mahalanobis <= threshold)
                close, far, returned = model.backend.close_far_indices(population, 2, threshold, "cholesky")
                expected = np.zeros(10, dtype=bool)
                expected[np.argsort(distances, kind="stable")[:k]] = True
                np.testing.assert_array_equal(close, np.flatnonzero(expected))
                np.testing.assert_array_equal(far, np.flatnonzero(~expected))
                np.testing.assert_array_equal(returned, distances)
                model.mahalanobis_threshold = threshold
                for diversity, indices in ((1., close), (0., far)):
                    wanted = population[indices] if len(indices) >= 3 else population
                    np.testing.assert_array_equal(model._mutation_pool(population, diversity), wanted)
            assert model.generator.bit_generator.state == rng_state
        # Real tiny batched CPU runs for every metric; control uses original engine.
        original = framework.build_batched_engine("F12017", "MaCRO-DE", normal)
        baseline, _, _ = original.run([0], [args.seed_base])
        for _, _, variant in variants:
            engine = framework.build_batched_engine("F12017", "MaCRO-DE", variant)
            outputs, _, _ = engine.run([0], [args.seed_base])
            assert np.isfinite(outputs[0]["curve"]).all()
            if variant.distance_metric == "mahalanobis_cholesky":
                np.testing.assert_array_equal(outputs[0]["curve"], baseline[0]["curve"])
                np.testing.assert_array_equal(outputs[0]["best_solution"], baseline[0]["best_solution"])
            else:
                assert type(engine) is BatchedMaCRODistance
                assert engine._advance_one_epoch.__func__ is BatchedDEEngine._advance_one_epoch
                # Two independently shaped runs; k comes from each current population.
                populations = np.random.default_rng(7).normal(size=(2, 10, 30))
                populations[1, 0] *= 100.
                for threshold in (0., 5., 9., np.inf):
                    original_close, _, _ = BatchedDEEngine._classification(engine, populations, "cholesky", threshold)
                    close, far, distances = engine._classification(populations, "cholesky", threshold)
                    counts = original_close.sum(axis=-1)
                    np.testing.assert_array_equal(close.sum(axis=-1), counts)
                    np.testing.assert_array_equal(far, ~close)
                    for row, count in enumerate(counts):
                        expected = np.zeros(10, dtype=bool)
                        expected[np.argsort(distances[row], kind="stable")[:count]] = True
                        np.testing.assert_array_equal(close[row], expected)
        # Exercise the real runner/report/cache path for just one function.
        # Production function selection and defaults are never edited.
        with patch.object(framework, "select_experiment_functions", return_value=["F12017"]):
            framework.run_experiment(args)
            paths = framework.make_paths(args, create=False)
            summary = pd.read_csv(Path(paths.res_dir)/"distance_ablation_summary.csv")
            assert len(summary) == 5 and summary["rank"].between(1, 5).all()
            archive = Path(paths.res_dir)/"F12017_D30_convergence.npz"
            with np.load(archive) as data:
                assert data["fitness_runs"].shape == (5, 2, 2)
                np.testing.assert_array_equal(data["function_evaluations"], [21, 31])
                np.testing.assert_allclose(data["mean_fitness"], data["fitness_runs"].mean(axis=1))
            checkpoint_bytes = {p: p.read_bytes() for p in Path(paths.cache_dir).rglob("*.pkl")}
            assert len(checkpoint_bytes) == 10
            args.distance_ablation_from_cache_only = True
            with patch.object(framework, "build_optimizer", side_effect=AssertionError("cache-only optimized")), \
                 patch.object(framework, "build_batched_engine", side_effect=AssertionError("cache-only optimized")), \
                 patch.object(framework, "initialize_gpu", side_effect=AssertionError("cache-only initialized CUDA")):
                framework.run_experiment(args)
                plot_archive(archive, Path(root)/"archive_only_figures")
            assert all(p.read_bytes() == b for p, b in checkpoint_bytes.items())
            next(iter(checkpoint_bytes)).unlink()
            try:
                framework.run_experiment(args)
            except RuntimeError as exc:
                assert "no optimization performed" in str(exc)
            else:
                raise AssertionError("Incomplete cache accepted")
        # Verify CUDA availability explicitly; unavailable CUDA must raise.
        try:
            backend = ComputeBackend("gpu")
        except (ImportError, RuntimeError) as exc:
            print(f"GPU runtime validation unavailable (no fallback): {exc}")
        else:
            for metric in ("chebyshev", "euclidean", "manhattan", "minkowski"):
                gpu = squared_distance_from_mean(backend.asarray(points[None]), backend.xp, metric, 3.)
                np.testing.assert_allclose(backend.to_cpu(gpu)[0], squared_distance_from_mean(points, np, metric, 3.))
                for k in range(4):
                    selected = nearest_k_mask(gpu, backend.xp.asarray([k]), backend.xp)
                    np.testing.assert_array_equal(backend.to_cpu(selected).sum(axis=-1), [k])
            print("CUDA distance kernels passed")
    print("PASS: scalar/batched controls, four distance variants, cache isolation, exports, and cache-only regeneration")


if __name__ == "__main__":
    main()
