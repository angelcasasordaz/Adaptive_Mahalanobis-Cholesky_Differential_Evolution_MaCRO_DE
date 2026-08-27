"""Deterministic tiny validation for true independent-run GPU batching.

This script never runs the full experiment. CPU mode validates execution and
the staged CEC formulas without CuPy; hybrid/GPU additionally performs strict
CUDA-versus-OPFUNU verification and exercises the same batched kernels.
"""
from __future__ import annotations

import argparse
import logging
import os
import tempfile
import multiprocessing
from concurrent.futures import ProcessPoolExecutor

import numpy as np
from mealpy import FloatVar
from scipy.stats import chi2

from compute_backend import ComputeBackend, initialize_gpu
from de_m_optimizer import DE_M
from de_mc_cf_optimizer import DE_MC_CF
from de_mc_optimizer import DE_MC
from gpu_batching import BATCHED_OPTIMIZERS, BatchedDEEngine
from macro_de_optimizer import MaCRO_DE
from main import audit_convergence_curves, build_optimizer_colors, plot_convergence, save_run_checkpoint
from objective_evaluation import ObjectiveEvaluator, ObjectiveSpec, initialize_objective_worker


SEED_BASE = 1234
POP_SIZE = 10
N_DIMS = 5
EPOCHS = 3
LB = np.full(N_DIMS, -5.0)
UB = np.full(N_DIMS, 5.0)


def sphere(vector):
    return float(np.sum(np.asarray(vector, dtype=np.float64) ** 2))


def problem():
    return {
        "bounds": FloatVar(lb=tuple(LB), ub=tuple(UB), name="x"),
        "minmax": "min",
        "obj_func": sphere,
        "log_to": None,
    }


class DETraceMixin:
    def before_main_loop(self):
        super().before_main_loop()
        self.validation_initial = self._positions(self.pop).copy()
        self.validation_donors = []
        self.validation_masks = []
        self.validation_trials = []

    def _sample_mutation_indices(self, pop_pos, current_idx):
        selected = super()._sample_mutation_indices(pop_pos, current_idx)
        self.validation_donors.append(np.asarray(selected).copy())
        return selected

    def _binomial_crossover(self, parent_pos, mutant_pos):
        trial = parent_pos.copy()
        j0 = self.generator.integers(0, self.problem.n_dims)
        mask = self.generator.random(self.problem.n_dims) <= self.cr
        mask[j0] = True
        trial[mask] = mutant_pos[mask]
        trial = self.correct_solution(trial)
        self.validation_masks.append(mask.copy())
        self.validation_trials.append(trial.copy())
        return trial


class RecordingGenerator:
    def __init__(self, generator):
        self.generator = generator
        self.choice_calls = []
        self.uniform_calls = []
        self.integer_calls = []
        self.random_calls = []

    def choice(self, *args, **kwargs):
        value = self.generator.choice(*args, **kwargs)
        self.choice_calls.append(np.asarray(value).copy())
        return value

    def uniform(self, *args, **kwargs):
        value = self.generator.uniform(*args, **kwargs)
        self.uniform_calls.append(np.asarray(value).copy())
        return value

    def integers(self, *args, **kwargs):
        value = self.generator.integers(*args, **kwargs)
        self.integer_calls.append(np.asarray(value).copy())
        return value

    def random(self, *args, **kwargs):
        value = self.generator.random(*args, **kwargs)
        self.random_calls.append(np.asarray(value).copy())
        return value


class TraceMaCRODE(MaCRO_DE):
    def before_main_loop(self):
        super().before_main_loop()
        self.validation_initial = self._positions(self.pop).copy()
        self.generator = RecordingGenerator(self.generator)


def make_engine(name, device, epochs=EPOCHS):
    kwargs = dict(
        optimizer_name=name,
        objective=sphere,
        lb=LB,
        ub=UB,
        epochs=epochs,
        pop_size=POP_SIZE,
        compute_device=device,
        mahalanobis_q=0.68,
    )
    return BatchedDEEngine(**kwargs)


def replay_macro_first_generation(model):
    positions = model.validation_initial
    backend = ComputeBackend("cpu")
    close, far, _ = backend.close_far_masks(
        positions, N_DIMS, float(chi2.ppf(0.68, N_DIMS)), "cholesky"
    )
    close = np.asarray(close)
    far = np.asarray(far)
    pool = np.flatnonzero(close) if np.count_nonzero(close) >= 3 else np.arange(POP_SIZE)
    donors = np.empty((POP_SIZE, 3), dtype=int)
    masks = np.empty((POP_SIZE, N_DIMS), dtype=bool)
    trials = np.empty_like(positions)
    f_vectors = np.empty_like(positions)
    recorder = model.generator
    for individual in range(POP_SIZE):
        donors[individual] = pool[recorder.choice_calls[individual]]
        f_vectors[individual] = np.clip(recorder.uniform_calls[individual] * 0.5, 0.10, 1.50)
        mask = recorder.random_calls[individual] <= 0.2
        mask[int(recorder.integer_calls[individual])] = True
        masks[individual] = mask
        d = donors[individual]
        mutant = np.clip(positions[d[0]] + f_vectors[individual] * (positions[d[1]] - positions[d[2]]), LB, UB)
        trials[individual] = np.where(mask, mutant, positions[individual])
    return donors, masks, trials, f_vectors


def validate_reference_equivalence(name, device):
    seed = SEED_BASE
    if name == "MaCRO-DE":
        reference = TraceMaCRODE(epoch=EPOCHS, pop_size=POP_SIZE, compute_device="cpu")
    else:
        base_class = {"DE-M": DE_M, "DE-MC": DE_MC, "DE-MC-CF": DE_MC_CF}[name]
        trace_class = type(f"Trace{base_class.__name__}", (DETraceMixin, base_class), {})
        reference = trace_class(epoch=EPOCHS, pop_size=POP_SIZE, compute_device="cpu")
    result = reference.solve(problem(), seed=seed)

    outputs, state, _ = make_engine(name, device).run([0], [seed], capture_trace=True)
    trace = outputs[0]["trace"]
    np.testing.assert_array_equal(reference.validation_initial, trace["initial_population"])
    if name == "MaCRO-DE":
        donors, masks, trials, f_vectors = replay_macro_first_generation(reference)
        np.testing.assert_array_equal(f_vectors, trace["f_vectors"])
    else:
        donors = np.asarray(reference.validation_donors[:POP_SIZE])
        masks = np.asarray(reference.validation_masks[:POP_SIZE])
        trials = np.asarray(reference.validation_trials[:POP_SIZE])
    np.testing.assert_array_equal(donors, trace["donor_indices"])
    np.testing.assert_array_equal(masks, trace["crossover_masks"])
    np.testing.assert_allclose(trials, trace["trial_population"], rtol=2e-11, atol=2e-12)
    expected_trial_fitness = np.asarray([sphere(vector) for vector in trials])
    np.testing.assert_allclose(expected_trial_fitness, trace["trial_fitness"], rtol=2e-11, atol=2e-12)
    np.testing.assert_allclose(reference.history.list_global_best_fit[0], trace["first_generation_best"], rtol=2e-11, atol=2e-12)
    np.testing.assert_allclose(result.target.fitness, outputs[0]["best_fitness"], rtol=2e-10, atol=2e-11)
    np.testing.assert_allclose(reference.history.list_global_best_fit, outputs[0]["curve"], rtol=2e-10, atol=2e-11)
    assert state.positions.shape == (1, POP_SIZE, N_DIMS)
    assert len(state.histories) == 1 and state.epoch_counts[0] == EPOCHS and state.stopped[0]
    print(f"{name}: single-run reference == batch_size=1")


def validate_batch_independence(name, device, batch_size):
    seeds = [SEED_BASE + idx for idx in range(batch_size)]
    run_indices = list(range(batch_size))
    outputs, state, _ = make_engine(name, device).run(run_indices, seeds, capture_trace=True)
    assert state.seeds == tuple(seeds)
    assert len({id(history) for history in state.histories}) == batch_size
    assert len({id(history) for history in state.current_histories}) == batch_size
    assert state.positions.shape == (batch_size, POP_SIZE, N_DIMS)
    assert state.fitness.shape == (batch_size, POP_SIZE)
    donors = state.trace["donor_indices"]
    assert donors.shape == (batch_size, POP_SIZE, 3)
    assert np.all((donors >= 0) & (donors < POP_SIZE))
    if name != "MaCRO-DE":
        current = np.arange(POP_SIZE)[None, :, None]
        assert not np.any(donors == current)
    assert np.all([len(history) == EPOCHS for history in state.histories])

    for offset, seed in enumerate(seeds):
        single_outputs, single_state, _ = make_engine(name, device).run([offset], [seed], capture_trace=True)
        np.testing.assert_array_equal(
            state.trace["initial_population"][offset], single_state.trace["initial_population"][0]
        )
        np.testing.assert_array_equal(donors[offset], single_state.trace["donor_indices"][0])
        np.testing.assert_array_equal(
            state.trace["crossover_masks"][offset], single_state.trace["crossover_masks"][0]
        )
        np.testing.assert_allclose(outputs[offset]["curve"], single_outputs[0]["curve"], rtol=2e-10, atol=2e-11)
        np.testing.assert_allclose(outputs[offset]["best_fitness"], single_outputs[0]["best_fitness"], rtol=2e-10, atol=2e-11)

    initial = state.trace["initial_population"]
    trial = state.trace["trial_population"]
    trial_fit = state.trace["trial_fitness"]
    expected = np.where(trial_fit[..., None] < np.asarray(
        [[sphere(vector) for vector in run] for run in initial]
    )[..., None], trial, initial)
    np.testing.assert_allclose(expected, state.trace["first_generation_population"])
    print(f"{name}: batch_size={batch_size} seed/state/selection isolation passed")


def validate_batched_kernels(device, batch_size):
    rng = np.random.default_rng(991)
    population = rng.normal(size=(batch_size, POP_SIZE, N_DIMS))
    backend = ComputeBackend(device)
    covariance, cholesky, distance = backend.mahalanobis_cpu(population, N_DIMS, "cholesky")
    close, far, _ = backend.close_far_masks(
        population, N_DIMS, float(chi2.ppf(0.68, N_DIMS)), "cholesky"
    )
    assert covariance.shape == (batch_size, N_DIMS, N_DIMS)
    assert cholesky.shape == (batch_size, N_DIMS, N_DIMS)
    assert distance.shape == (batch_size, POP_SIZE)
    assert close.shape == far.shape == (batch_size, POP_SIZE)
    np.testing.assert_array_equal(backend.to_cpu(close | far), np.ones((batch_size, POP_SIZE), dtype=bool))
    print(f"batched covariance/Cholesky/Mahalanobis/close-far: batch_size={batch_size}")


def validate_checkpoint_separation():
    with tempfile.TemporaryDirectory() as directory:
        paths = []
        for run in range(4):
            path = os.path.join(directory, f"run_{run + 1:03d}.pkl")
            save_run_checkpoint(
                path,
                {"run": run, "seed": SEED_BASE + run},
                {"best_fitness": float(run), "best_solution": np.array([run]), "runtime": 0.0, "curve": np.array([run])},
            )
            paths.append(path)
        assert len(set(paths)) == 4 and all(os.path.isfile(path) for path in paths)
    print("per-run checkpoint separation: passed")


def validate_process_objective_equivalence():
    from opfunu.cec_based.cec2017 import F292017

    benchmark = F292017(ndim=30)
    vectors = np.random.default_rng(771).uniform(-100.0, 100.0, size=(64, 30))
    serial = np.asarray([benchmark.evaluate(vector) for vector in vectors])
    with ProcessPoolExecutor(
        max_workers=4,
        mp_context=multiprocessing.get_context("spawn"),
        initializer=initialize_objective_worker,
    ) as executor:
        evaluator = ObjectiveEvaluator(
            benchmark.evaluate,
            executor=executor,
            workers=4,
            strategy="process",
            spec=ObjectiveSpec(F292017.__module__, F292017.__name__, 30),
        )
        evaluator.calibrate(benchmark.lb, benchmark.ub, len(vectors))
        parallel = evaluator.evaluate(vectors)
    np.testing.assert_array_equal(serial, parallel)
    print("persistent spawn objective evaluation: exact match")


def validate_cec_objective_candidates(device):
    from opfunu.cec_based import cec2017
    from cec2017_gpu import CEC2017_GPU_CANDIDATES, CEC2017GpuObjective, verify_gpu_objective

    backend = ComputeBackend(device)
    for ndim in (10, 30, 50):
        for function_name in sorted(CEC2017_GPU_CANDIDATES):
            benchmark = getattr(cec2017, function_name)(ndim=ndim)
            objective = CEC2017GpuObjective(function_name, benchmark, backend.xp)
            report = verify_gpu_objective(
                objective,
                benchmark,
                backend.to_cpu,
                backend.synchronize,
                random_points=200,
            )
            status = "verified" if report.verified else "CPU fallback"
            print(
                f"CEC candidate {function_name} D={ndim}: {status}; points={report.points} "
                f"max_abs={report.max_absolute_error:.3e} "
                f"max_rel={report.max_relative_error:.3e} mismatches={report.mismatches}"
            )


def validate_cec_engine_dispatch(device):
    from opfunu.cec_based.cec2017 import F12017
    from cec2017_gpu import CEC2017GpuObjective, verify_gpu_objective

    benchmark = F12017(ndim=10)
    backend = ComputeBackend(device)
    gpu_objective = CEC2017GpuObjective("F12017", benchmark, backend.xp)
    report = verify_gpu_objective(
        gpu_objective, benchmark, backend.to_cpu, backend.synchronize, random_points=200
    )
    assert report.verified
    common = dict(
        optimizer_name="DE-M",
        objective=benchmark.evaluate,
        lb=benchmark.lb,
        ub=benchmark.ub,
        epochs=3,
        pop_size=10,
        compute_device=device,
    )
    cpu_output, _, _ = BatchedDEEngine(
        **common, cec_objective_backend="opfunu"
    ).run([0, 1], [SEED_BASE, SEED_BASE + 1])
    gpu_output, _, _ = BatchedDEEngine(
        **common, gpu_objective=gpu_objective, cec_objective_backend="gpu"
    ).run([0, 1], [SEED_BASE, SEED_BASE + 1])
    for left, right in zip(cpu_output, gpu_output):
        np.testing.assert_allclose(left["curve"], right["curve"], rtol=5e-11, atol=1e-7)
        np.testing.assert_allclose(left["best_fitness"], right["best_fitness"], rtol=5e-11, atol=1e-7)
    print("verified CEC objective full-generation dispatch: passed")


def validate_plot_overlap():
    curves = {
        "DE": np.array([10.0, 5.0, 2.0, 1.0]),
        "DE-M": np.array([9.0, 4.0, 2.0, 1.0]),
        "DE-MC": np.array([9.0, 4.0, 2.0, 1.0]),
        "DE-MC-CF": np.array([8.0, 4.0, 1.8, 0.9]),
        "MaCRO-DE": np.array([7.0, 3.0, 1.5, 0.8]),
    }
    diagnostics = audit_convergence_curves(curves, "SYNTHETIC")
    assert any(pair[:2] == ("DE-M", "DE-MC") for pair in diagnostics["overlaps"])
    with tempfile.TemporaryDirectory() as directory:
        path = os.path.join(directory, "overlap.png")
        plot_convergence(curves, "Synthetic overlap", path, build_optimizer_colors(list(curves)))
        assert os.path.isfile(path) and os.path.getsize(path) > 0
    print("plot overlap diagnostics/styles: passed")


def audit_ablation_routing(device):
    ndim, pop_size, seed = 30, 50, SEED_BASE
    lb = np.full(ndim, -100.0)
    ub = np.full(ndim, 100.0)
    traces = {}
    for name in ("DE-M", "DE-MC", "DE-MC-CF"):
        engine = BatchedDEEngine(
            name,
            sphere,
            lb,
            ub,
            epochs=1,
            pop_size=pop_size,
            compute_device=device,
        )
        outputs, _, _ = engine.run([0], [seed], capture_trace=True)
        traces[name] = outputs[0]["trace"]
    direct, chol, close_far = traces["DE-M"], traces["DE-MC"], traces["DE-MC-CF"]
    np.testing.assert_array_equal(direct["close_masks"], chol["close_masks"])
    np.testing.assert_array_equal(chol["close_masks"], close_far["close_masks"])
    np.testing.assert_array_equal(direct["donor_indices"], chol["donor_indices"])
    np.testing.assert_array_equal(chol["donor_indices"], close_far["donor_indices"])
    np.testing.assert_allclose(direct["trial_population"], chol["trial_population"], rtol=2e-11, atol=2e-12)
    np.testing.assert_allclose(chol["trial_population"], close_far["trial_population"], rtol=2e-11, atol=2e-12)
    close_count = int(np.count_nonzero(direct["close_masks"]))
    print(
        "ABLATION ROUTING | direct/cholesky close masks, donors, and trials identical | "
        f"close_count={close_count}/{pop_size} | close/far fallback was not activated"
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--compute-device", choices=["cpu", "gpu", "hybrid"], default="cpu")
    parser.add_argument("--gpu-memory-fraction", type=float, default=0.85)
    args = parser.parse_args()
    logging.disable(logging.INFO)
    if args.compute_device in {"gpu", "hybrid"}:
        info = initialize_gpu(memory_fraction=args.gpu_memory_fraction)
        print(f"GPU: {info.name}; memory cap: {info.memory_fraction:.0%}")

    for name in sorted(BATCHED_OPTIMIZERS):
        validate_reference_equivalence(name, args.compute_device)
    for batch_size in (2, 4, 8):
        validate_batched_kernels(args.compute_device, batch_size)
        for name in sorted(BATCHED_OPTIMIZERS):
            validate_batch_independence(name, args.compute_device, batch_size)
    # Large grouping check for the common DE-M path without multiplying the
    # full four-optimizer validation cost.
    for batch_size in (16, 30):
        validate_batch_independence("DE-M", args.compute_device, batch_size)
    validate_checkpoint_separation()
    validate_process_objective_equivalence()
    validate_cec_objective_candidates(args.compute_device)
    validate_cec_engine_dispatch(args.compute_device)
    audit_ablation_routing(args.compute_device)
    validate_plot_overlap()
    print("All true multi-run batching validations passed.")


if __name__ == "__main__":
    main()
