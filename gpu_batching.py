"""True independent-run batching for the four GPU-backed custom DE variants.

MEALPY remains the CPU execution engine.  This module mirrors the relevant
MEALPY 3.0.2 lifecycle with dense arrays so one CUDA-owning controller can
advance several independent runs in a single ``(run, population, dimension)``
batch.  Random streams and all adaptive/history state remain per run.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from time import perf_counter
from typing import Callable, Sequence
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor

import numpy as np
from scipy.stats import chi2

from compute_backend import ComputeBackend
from objective_evaluation import ObjectiveEvaluator, ObjectiveSpec


BATCH_ENGINE_VERSION = "independent-runs-v4-parallel-plans"
BATCHED_OPTIMIZERS = frozenset({"DE-M", "DE-MC", "DE-MC-CF", "MaCRO-DE"})
_OBJECTIVE_DECISION_CACHE: dict[tuple, tuple[str, dict[str, float]]] = {}


def _de_random_plan_chunk(task):
    """Build independent run plans in an existing CPU worker process."""
    (
        optimizer_name, pop_size, n_dims, cr, generator_states,
        close, far, div_norm,
    ) = task
    all_indices = np.arange(pop_size, dtype=np.int64)
    fallback = tuple(
        np.concatenate((all_indices[:individual], all_indices[individual + 1:]))
        for individual in range(pop_size)
    )
    donors = np.empty((len(generator_states), pop_size, 3), dtype=np.int64)
    crossover = np.empty((len(generator_states), pop_size, n_dims), dtype=bool)
    updated_states = []
    for run_offset, generator_state in enumerate(generator_states):
        rng = np.random.default_rng()
        rng.bit_generator.state = generator_state
        if optimizer_name == "DE-MC-CF":
            selected_mask = (
                close[run_offset]
                if div_norm[run_offset] >= 0.5
                else far[run_offset]
            )
        else:
            selected_mask = close[run_offset]
        selected_pool = all_indices[selected_mask]
        selected_candidates = tuple(
            selected_pool[selected_pool != individual]
            if selected_mask[individual]
            else selected_pool
            for individual in range(pop_size)
        )
        for individual in range(pop_size):
            candidates = selected_candidates[individual]
            if candidates.size < 3:
                candidates = fallback[individual]
            donors[run_offset, individual] = rng.choice(candidates, 3, replace=False)
            j0 = rng.integers(0, n_dims)
            mask = crossover[run_offset, individual]
            np.less_equal(rng.random(n_dims), cr, out=mask)
            mask[j0] = True
        updated_states.append(rng.bit_generator.state)
    return donors, crossover, updated_states


def _macro_random_plan_chunk(task):
    """Build MaCRO-DE plans while retaining each NumPy generator's exact state."""
    (
        pop_size, n_dims, beta_min, beta_max, generator_states,
        close, far, div_norm, pcr_values, scale_values,
    ) = task
    all_indices = np.arange(pop_size, dtype=np.int64)
    donors = np.empty((len(generator_states), pop_size, 3), dtype=np.int64)
    crossover = np.empty((len(generator_states), pop_size, n_dims), dtype=bool)
    f_vectors = np.empty((len(generator_states), pop_size, n_dims), dtype=np.float64)
    updated_states = []
    for run_offset, generator_state in enumerate(generator_states):
        rng = np.random.default_rng()
        rng.bit_generator.state = generator_state
        if div_norm[run_offset] >= 0.5 and np.count_nonzero(close[run_offset]) >= 3:
            pool = all_indices[close[run_offset]]
        elif div_norm[run_offset] < 0.5 and np.count_nonzero(far[run_offset]) >= 3:
            pool = all_indices[far[run_offset]]
        else:
            pool = all_indices
        for individual in range(pop_size):
            selected_in_pool = rng.choice(pool.size, 3, replace=False)
            donors[run_offset, individual] = pool[selected_in_pool]
            f_vec = rng.uniform(beta_min, beta_max, n_dims) * scale_values[run_offset]
            f_vectors[run_offset, individual] = np.clip(f_vec, 0.10, 1.50)
            j0 = rng.integers(0, n_dims)
            mask = crossover[run_offset, individual]
            np.less_equal(rng.random(n_dims), pcr_values[run_offset], out=mask)
            mask[j0] = True
        updated_states.append(rng.bit_generator.state)
    return donors, crossover, f_vectors, updated_states


@dataclass
class BatchedRunState:
    """Dense numerical state plus explicitly separated state for every run."""

    run_indices: tuple[int, ...]
    seeds: tuple[int, ...]
    positions: object
    fitness: object
    best_positions: object
    best_fitness: object
    current_best_positions: object
    current_best_fitness: object
    generators: list[np.random.Generator]
    histories: list[list[float]]
    current_histories: list[list[float]]
    epoch_counts: np.ndarray
    stopped: np.ndarray
    algorithm_state: dict[str, object] = field(default_factory=dict)
    trace: dict[str, np.ndarray] = field(default_factory=dict)
    capture_trace: bool = False


@dataclass
class BatchTiming:
    totals: dict[str, float] = field(default_factory=lambda: defaultdict(float))
    epochs: int = 0

    def add(self, name: str, seconds: float) -> None:
        self.totals[name] += max(0.0, float(seconds))

    def summary(self) -> dict[str, float]:
        gpu_keys = {
            "covariance", "factorization_inverse", "mahalanobis",
            "classification", "mutation", "crossover", "selection", "history_store", "awad",
            "mutation_boundary", "trial_boundary",
        }
        transfer_keys = {"classification_transfer", "trial_download", "fitness_upload", "history_transfer", "adaptive_transfer"}
        result = dict(self.totals)
        result["gpu_kernel"] = sum(self.totals[key] for key in gpu_keys)
        result["transfer"] = sum(self.totals[key] for key in transfer_keys)
        result["fitness"] = (
            self.totals["fitness_cpu"]
            + self.totals["fitness_cpu_vectorized"]
            + self.totals["objective_gpu"]
        )
        result["donor_rng"] = self.totals["donor_construction"]
        result["python_orchestration"] = max(
            0.0,
            self.totals["epoch_total"]
            - result["gpu_kernel"]
            - result["transfer"]
            - result["fitness"]
            - result["donor_rng"],
        )
        result["other"] = result["python_orchestration"]
        return result


class BatchedDEEngine:
    """Execute independent custom-DE runs together on one array backend."""

    def __init__(
        self,
        optimizer_name: str,
        objective: Callable[[np.ndarray], float],
        lb: Sequence[float],
        ub: Sequence[float],
        epochs: int,
        pop_size: int,
        compute_device: str = "hybrid",
        wf: float = 0.5,
        cr: float = 0.9,
        mahalanobis_q: float = 0.68,
        beta_min: float = 0.2,
        beta_max: float = 0.8,
        pcr: float = 0.2,
        objective_executor=None,
        random_plan_executor=None,
        objective_workers: int = 1,
        objective_strategy: str = "auto",
        objective_spec: ObjectiveSpec | None = None,
        gpu_objective=None,
        vectorized_cpu_objective=None,
        cec_objective_backend: str = "auto",
    ):
        if optimizer_name not in BATCHED_OPTIMIZERS:
            raise ValueError(f"Batched GPU execution is not implemented for {optimizer_name!r}.")
        self.optimizer_name = optimizer_name
        self.objective = objective
        self.lb = np.asarray(lb, dtype=np.float64)
        self.ub = np.asarray(ub, dtype=np.float64)
        self.n_dims = int(self.lb.size)
        self.epochs = int(epochs)
        self.pop_size = int(pop_size)
        self.backend = ComputeBackend(compute_device)
        self.xp = self.backend.xp
        self.lb_backend = self.backend.asarray(self.lb)
        self.ub_backend = self.backend.asarray(self.ub)
        self.wf = float(wf)
        self.cr = float(cr)
        self.mahalanobis_q = float(mahalanobis_q)
        self.beta_min = float(beta_min)
        self.beta_max = float(beta_max)
        self.pcr = float(pcr)
        self.threshold = float(chi2.ppf(self.mahalanobis_q, max(self.n_dims, 1)))
        self.macro_threshold = float(chi2.ppf(self.mahalanobis_q, max(self.n_dims, 1)))
        self.objective_evaluator = ObjectiveEvaluator(
            objective,
            executor=objective_executor,
            workers=objective_workers,
            strategy=objective_strategy,
            spec=objective_spec,
        )
        # Strict-GPU execution can parallelize independent per-run RNG plans
        # without exposing the objective evaluator to that process pool.
        # CPU/hybrid callers leave this unset and retain their existing path.
        self.random_plan_executor = random_plan_executor
        if cec_objective_backend not in {"auto", "opfunu", "numpy", "gpu"}:
            raise ValueError("CEC objective backend must be auto, opfunu, numpy, or gpu")
        self.gpu_objective = gpu_objective
        self.vectorized_cpu_objective = vectorized_cpu_objective
        self.cec_objective_backend = cec_objective_backend
        self.selected_objective_backend = "cpu"
        self.objective_dispatch_timing: dict[str, float] = {}
        self.timing = BatchTiming()
        self._pending_gpu_events: list[tuple[str, object, object]] = []
        self._gpu_event_pool: list[tuple[object, object]] = []
        self._host_positions = None
        self._host_plan_buffers: dict[str, np.ndarray] = {}
        self._device_plan_buffers: dict[str, object] = {}
        self._device_rows = None
        all_indices = np.arange(self.pop_size, dtype=np.int64)
        self._de_fallback_candidates = tuple(
            np.concatenate((all_indices[:individual], all_indices[individual + 1:]))
            for individual in range(self.pop_size)
        )

    def _host_plan_buffer(self, name: str, shape, dtype) -> np.ndarray:
        """Reuse pinned random-plan storage across generations."""
        buffer = self._host_plan_buffers.get(name)
        if buffer is None or buffer.shape != tuple(shape) or buffer.dtype != np.dtype(dtype):
            buffer = self.backend.empty_pinned(shape, dtype)
            self._host_plan_buffers[name] = buffer
        return buffer

    def _device_plan_buffer(self, name: str, host: np.ndarray):
        """Upload into stable device storage without a per-generation allocation."""
        if not self.backend.uses_gpu:
            return host
        buffer = self._device_plan_buffers.get(name)
        if buffer is None or buffer.shape != host.shape or buffer.dtype != host.dtype:
            buffer = self.xp.empty(host.shape, dtype=host.dtype)
            self._device_plan_buffers[name] = buffer
        buffer.set(host, stream=self.xp.cuda.get_current_stream())
        return buffer

    def _rows(self, batch_size: int):
        if self._device_rows is None or self._device_rows.size != batch_size:
            self._device_rows = self.xp.arange(batch_size)[:, None]
        return self._device_rows

    def _stage(self, name: str, function):
        if self.backend.uses_gpu:
            event_index = len(self._pending_gpu_events)
            if event_index == len(self._gpu_event_pool):
                self._gpu_event_pool.append((self.xp.cuda.Event(), self.xp.cuda.Event()))
            start, end = self._gpu_event_pool[event_index]
            start.record()
            result = function()
            end.record()
            self._pending_gpu_events.append((name, start, end))
            return result
        started = perf_counter()
        result = function()
        self.timing.add(name, perf_counter() - started)
        return result

    def _flush_gpu_events(self) -> float:
        elapsed_total = 0.0
        for name, start, end in self._pending_gpu_events:
            seconds = float(self.xp.cuda.get_elapsed_time(start, end)) / 1000.0
            self.timing.add(name, seconds)
            elapsed_total += seconds
        self._pending_gpu_events.clear()
        return elapsed_total

    def _initial_positions(self, seeds: Sequence[int]) -> np.ndarray:
        # MEALPY owns a separate RNG in FloatVar and draws one individual at a
        # time.  Repeating that order preserves its seeded initial population.
        batches = []
        for seed in seeds:
            bounds_rng = np.random.default_rng(seed)
            batches.append(
                np.asarray(
                    [bounds_rng.uniform(self.lb, self.ub) for _ in range(self.pop_size)],
                    dtype=np.float64,
                )
            )
        return np.stack(batches, axis=0)

    def _evaluate(self, positions) -> object:
        """Make one contiguous GPU->CPU transfer, then evaluate flattened OPFUNU."""
        if self.selected_objective_backend == "gpu":
            return self._stage("objective_gpu", lambda: self.gpu_objective.evaluate(positions))
        if self._host_positions is None or self._host_positions.shape != positions.shape:
            self._host_positions = self.backend.empty_pinned(positions.shape, np.float64)
        started = perf_counter()
        cpu_positions = self.backend.to_cpu(positions, out=self._host_positions)
        download_wall = perf_counter() - started
        pending_gpu = self._flush_gpu_events() if self.backend.uses_gpu else 0.0
        self.timing.add("trial_download", max(0.0, download_wall - pending_gpu))
        cpu_positions = np.ascontiguousarray(cpu_positions, dtype=np.float64)
        flat = cpu_positions.reshape(-1, self.n_dims)
        started = perf_counter()
        if self.selected_objective_backend == "numpy":
            values = np.asarray(self.vectorized_cpu_objective.evaluate(flat), dtype=np.float64)
            self.timing.add("fitness_cpu_vectorized", perf_counter() - started)
        else:
            values = self.objective_evaluator.evaluate(flat)
            self.timing.add("fitness_cpu", perf_counter() - started)
        started = perf_counter()
        result = self.backend.asarray(values.reshape(cpu_positions.shape[:2]))
        self.timing.add("fitness_upload", perf_counter() - started)
        return result

    def _select_objective_backend(self, expected_vectors: int) -> str:
        """Benchmark verified GPU evaluation against the optimized CPU fallback."""
        self.objective_evaluator.calibrate(self.lb, self.ub, expected_vectors)
        if self.cec_objective_backend == "opfunu":
            self.selected_objective_backend = "cpu"
            return "cpu"
        if self.cec_objective_backend == "numpy":
            verification = getattr(self.vectorized_cpu_objective, "verification", None)
            if verification is None or not verification.verified:
                raise RuntimeError("Requested vectorized NumPy CEC objective is unverified")
            self.selected_objective_backend = "numpy"
            return "numpy"
        verification = getattr(self.gpu_objective, "verification", None)
        if self.gpu_objective is not None and (verification is None or not verification.verified):
            self.gpu_objective = None
        if self.gpu_objective is None and self.vectorized_cpu_objective is None:
            self.selected_objective_backend = "cpu"
            return "cpu"
        if self.cec_objective_backend == "gpu" and self.gpu_objective is not None:
            self.selected_objective_backend = "gpu"
            return "gpu"

        key = (
            getattr(self.gpu_objective or self.vectorized_cpu_objective, "function_name", "opfunu"),
            self.n_dims,
            int(expected_vectors),
            self.objective_evaluator.strategy,
            self.objective_evaluator.workers,
            self.backend.device_id,
            self.cec_objective_backend,
        )
        cached = _OBJECTIVE_DECISION_CACHE.get(key)
        if cached is not None:
            self.selected_objective_backend, self.objective_dispatch_timing = cached
            return self.selected_objective_backend

        sample_count = min(expected_vectors, max(128, self.objective_evaluator.workers * 16))
        sample = np.random.default_rng(7_700_001).uniform(
            self.lb, self.ub, size=(sample_count, self.n_dims)
        )
        # Warm persistent process workers and GPU kernels outside measurements.
        self.objective_evaluator.evaluate(sample[: min(sample_count, max(8, self.objective_evaluator.workers))])
        device_sample = self.backend.asarray(sample)
        if self.gpu_objective is not None:
            self.gpu_objective.evaluate(device_sample)
            self.backend.synchronize()
        if self.vectorized_cpu_objective is not None:
            self.vectorized_cpu_objective.evaluate(sample)

        calibration_host = self.backend.empty_pinned(device_sample.shape, np.float64)

        def time_cpu_dispatch(evaluate):
            started = perf_counter()
            host_sample = self.backend.to_cpu(device_sample, out=calibration_host)
            values = evaluate(np.ascontiguousarray(host_sample))
            self.backend.asarray(np.asarray(values, dtype=np.float64))
            self.backend.synchronize()
            return perf_counter() - started

        cpu_seconds = time_cpu_dispatch(self.objective_evaluator.evaluate)
        numpy_seconds = float("inf")
        if self.vectorized_cpu_objective is not None:
            numpy_seconds = time_cpu_dispatch(self.vectorized_cpu_objective.evaluate)
        gpu_seconds = float("inf")
        if self.gpu_objective is not None:
            started = perf_counter()
            self.gpu_objective.evaluate(device_sample)
            self.backend.synchronize()
            gpu_seconds = perf_counter() - started
        selected = min(
            (("cpu", cpu_seconds), ("numpy", numpy_seconds), ("gpu", gpu_seconds)),
            key=lambda item: item[1],
        )[0]
        timing = {
            "cpu_seconds": cpu_seconds,
            "gpu_seconds": gpu_seconds,
            "numpy_seconds": numpy_seconds,
            "vectors": float(sample_count),
            "includes_transfers": True,
        }
        _OBJECTIVE_DECISION_CACHE[key] = (selected, timing)
        self.selected_objective_backend = selected
        self.objective_dispatch_timing = timing
        return selected

    def initialize(
        self,
        run_indices: Sequence[int],
        seeds: Sequence[int],
        capture_trace: bool = False,
    ) -> BatchedRunState:
        if len(run_indices) != len(seeds) or len(seeds) < 1:
            raise ValueError("run_indices and seeds must have the same positive length.")
        initial_cpu = self._initial_positions(seeds)
        positions = self.backend.asarray(initial_cpu)
        fitness = self._evaluate(positions)
        best_indices = self.xp.argmin(fitness, axis=1)
        rows = self._rows(len(seeds))[:, 0]
        best_positions = positions[rows, best_indices].copy()
        best_fitness = fitness[rows, best_indices].copy()
        state = BatchedRunState(
            tuple(int(item) for item in run_indices),
            tuple(int(item) for item in seeds),
            positions,
            fitness,
            best_positions,
            best_fitness,
            best_positions.copy(),
            best_fitness.copy(),
            [np.random.default_rng(seed) for seed in seeds],
            [[] for _ in seeds],
            [[] for _ in seeds],
            np.zeros(len(seeds), dtype=np.int64),
            np.zeros(len(seeds), dtype=bool),
            capture_trace=bool(capture_trace),
        )
        # Histories stay device-resident for the batch lifecycle.  Downloading
        # two scalars per run every epoch forced an otherwise unnecessary CUDA
        # synchronization and was substantially more expensive than the data.
        state.algorithm_state.update(
            history_buffer=self.xp.empty((len(seeds), self.epochs), dtype=self.xp.float64),
            current_history_buffer=self.xp.empty(
                (len(seeds), self.epochs), dtype=self.xp.float64
            ),
        )
        if state.capture_trace:
            state.trace["initial_population"] = initial_cpu.copy()
        if self.optimizer_name in {"DE-MC-CF", "MaCRO-DE"}:
            div0 = self._awad(positions)
            state.algorithm_state.update(
                div_max_seen=self.xp.maximum(div0, 1.0e-9),
                div_norm_for_update=self.xp.ones(len(seeds), dtype=self.xp.float64),
                div_norm_cpu=np.ones(len(seeds), dtype=np.float64),
            )
        if self.optimizer_name == "MaCRO-DE":
            state.algorithm_state.update(
                div_awad_history=[[] for _ in seeds],
                div_norm_history=[[] for _ in seeds],
                pcr_history=[[] for _ in seeds],
                fmean_history=[[] for _ in seeds],
            )
        return state

    def _classification(self, positions, method: str, threshold: float):
        sigma = self._stage(
            "covariance",
            lambda: self.backend.covariance(positions, self.n_dims),
        )
        factor, factor_kind = self._stage(
            "factorization_inverse",
            lambda: self.backend.covariance_factor(sigma, method),
        )
        distances = self._stage(
            "mahalanobis",
            lambda: self.backend.distances_from_factor(positions, factor, factor_kind),
        )
        close = self._stage("classification", lambda: distances <= threshold)
        started = perf_counter()
        close_cpu = self.backend.to_cpu(close).astype(bool, copy=False)
        transfer_wall = perf_counter() - started
        pending_gpu = self._flush_gpu_events() if self.backend.uses_gpu else 0.0
        self.timing.add("classification_transfer", max(0.0, transfer_wall - pending_gpu))
        return close_cpu, ~close_cpu, distances

    def _de_random_plan(self, state: BatchedRunState, close: np.ndarray, far: np.ndarray):
        batch_size = len(state.seeds)
        executor = self.random_plan_executor or self.objective_evaluator.executor
        if (
            isinstance(executor, ProcessPoolExecutor)
            and self.objective_evaluator.workers > 1
            and batch_size >= 4
        ):
            chunk_indices = [
                chunk for chunk in np.array_split(
                    np.arange(batch_size), min(batch_size, self.objective_evaluator.workers)
                ) if chunk.size
            ]
            tasks = [
                (
                    self.optimizer_name,
                    self.pop_size,
                    self.n_dims,
                    self.cr,
                    [state.generators[index].bit_generator.state for index in chunk],
                    np.ascontiguousarray(close[chunk]),
                    np.ascontiguousarray(far[chunk]),
                    np.ascontiguousarray(
                        state.algorithm_state.get(
                            "div_norm_cpu",
                            np.ones(batch_size, dtype=np.float64),
                        )[chunk]
                    ),
                )
                for chunk in chunk_indices
            ]
            results = list(executor.map(_de_random_plan_chunk, tasks))
            donors = self._host_plan_buffer("donors", (batch_size, self.pop_size, 3), np.int64)
            crossover = self._host_plan_buffer(
                "crossover", (batch_size, self.pop_size, self.n_dims), bool
            )
            for chunk, (chunk_donors, chunk_crossover, generator_states) in zip(
                chunk_indices, results
            ):
                donors[chunk] = chunk_donors
                crossover[chunk] = chunk_crossover
                for index, generator_state in zip(chunk, generator_states):
                    state.generators[index].bit_generator.state = generator_state
            return donors, crossover
        donors = self._host_plan_buffer("donors", (batch_size, self.pop_size, 3), np.int64)
        crossover = self._host_plan_buffer(
            "crossover", (batch_size, self.pop_size, self.n_dims), bool
        )
        all_indices = np.arange(self.pop_size, dtype=np.int64)
        div_norm = state.algorithm_state.get(
            "div_norm_cpu",
            np.ones(batch_size, dtype=np.float64),
        )
        for run_offset, rng in enumerate(state.generators):
            if self.optimizer_name == "DE-MC-CF":
                selected_mask = (
                    close[run_offset]
                    if div_norm[run_offset] >= 0.5
                    else far[run_offset]
                )
            else:
                selected_mask = close[run_offset]
            selected_pool = all_indices[selected_mask]
            selected_candidates = tuple(
                selected_pool[selected_pool != individual]
                if selected_mask[individual]
                else selected_pool
                for individual in range(self.pop_size)
            )
            for individual in range(self.pop_size):
                candidates = selected_candidates[individual]
                if candidates.size < 3:
                    candidates = self._de_fallback_candidates[individual]
                donors[run_offset, individual] = rng.choice(candidates, 3, replace=False)
                j0 = rng.integers(0, self.n_dims)
                mask = crossover[run_offset, individual]
                np.less_equal(rng.random(self.n_dims), self.cr, out=mask)
                mask[j0] = True
        return donors, crossover

    def _macro_random_plan(self, state: BatchedRunState, close: np.ndarray, far: np.ndarray):
        batch_size = len(state.seeds)
        donors = self._host_plan_buffer("donors", (batch_size, self.pop_size, 3), np.int64)
        crossover = self._host_plan_buffer(
            "crossover", (batch_size, self.pop_size, self.n_dims), bool
        )
        f_vectors = self._host_plan_buffer(
            "f_vectors", (batch_size, self.pop_size, self.n_dims), np.float64
        )
        div_norm = state.algorithm_state["div_norm_cpu"]
        pcr_values = np.clip(self.pcr + 0.25 * (1.0 - div_norm), 0.10, 0.95)
        scale_values = np.clip(0.5 + (1.0 - div_norm), 0.5, 1.5)
        executor = self.random_plan_executor or self.objective_evaluator.executor
        if (
            isinstance(executor, ProcessPoolExecutor)
            and self.objective_evaluator.workers > 1
            and batch_size >= 4
        ):
            chunk_indices = [
                chunk for chunk in np.array_split(
                    np.arange(batch_size), min(batch_size, self.objective_evaluator.workers)
                ) if chunk.size
            ]
            tasks = [
                (
                    self.pop_size,
                    self.n_dims,
                    self.beta_min,
                    self.beta_max,
                    [state.generators[index].bit_generator.state for index in chunk],
                    np.ascontiguousarray(close[chunk]),
                    np.ascontiguousarray(far[chunk]),
                    np.ascontiguousarray(div_norm[chunk]),
                    np.ascontiguousarray(pcr_values[chunk]),
                    np.ascontiguousarray(scale_values[chunk]),
                )
                for chunk in chunk_indices
            ]
            results = list(executor.map(_macro_random_plan_chunk, tasks))
            for chunk, (
                chunk_donors, chunk_crossover, chunk_f_vectors, generator_states,
            ) in zip(chunk_indices, results):
                donors[chunk] = chunk_donors
                crossover[chunk] = chunk_crossover
                f_vectors[chunk] = chunk_f_vectors
                for index, generator_state in zip(chunk, generator_states):
                    state.generators[index].bit_generator.state = generator_state
            return donors, crossover, f_vectors, pcr_values
        all_indices = np.arange(self.pop_size, dtype=np.int64)
        for run_offset, rng in enumerate(state.generators):
            if div_norm[run_offset] >= 0.5 and np.count_nonzero(close[run_offset]) >= 3:
                pool = all_indices[close[run_offset]]
            elif div_norm[run_offset] < 0.5 and np.count_nonzero(far[run_offset]) >= 3:
                pool = all_indices[far[run_offset]]
            else:
                pool = all_indices
            for individual in range(self.pop_size):
                # MaCRO-DE samples particles from its selected pool and does not
                # exclude the current population index in the existing method.
                selected_in_pool = rng.choice(pool.size, 3, replace=False)
                donors[run_offset, individual] = pool[selected_in_pool]
                f_vec = rng.uniform(self.beta_min, self.beta_max, self.n_dims) * scale_values[run_offset]
                f_vectors[run_offset, individual] = np.clip(f_vec, 0.10, 1.50)
                j0 = rng.integers(0, self.n_dims)
                mask = crossover[run_offset, individual]
                np.less_equal(rng.random(self.n_dims), pcr_values[run_offset], out=mask)
                mask[j0] = True
        return donors, crossover, f_vectors, pcr_values

    def _gather_mutants(self, positions, donors, factors):
        donor_gpu = self._device_plan_buffer("donors", donors)
        rows = self._rows(positions.shape[0])
        x1 = positions[rows, donor_gpu[:, :, 0]]
        x2 = positions[rows, donor_gpu[:, :, 1]]
        x3 = positions[rows, donor_gpu[:, :, 2]]
        return x1 + factors * (x2 - x3)

    def _awad(self, positions):
        """Batched equivalent of MaCRO-DE's per-run AWAD definition."""
        xp = self.xp
        batch_size, pop_size, n_dims = positions.shape
        median = xp.median(positions, axis=1, keepdims=True)
        div = xp.sum(xp.mean(xp.abs(positions - median), axis=1), axis=1) / max(n_dims, 1)

        # Count unique rows independently: a row is unique when no other row is equal.
        equal_rows = xp.all(
            positions[:, :, None, :] == positions[:, None, :, :], axis=-1
        )
        # Count only each duplicate group's first row, exactly matching np.unique.
        previous = xp.tril(xp.ones((pop_size, pop_size), dtype=bool), k=-1)
        has_previous_duplicate = xp.any(equal_rows & previous[None, :, :], axis=2)
        unique_count = xp.sum(~has_previous_duplicate, axis=1)
        non_repeat_percent = unique_count * 100.0 / max(pop_size, 1)

        std = xp.std(positions, axis=1)
        std = xp.where(std == 0, 1.0e-5, std)
        scaled_diff = (positions[:, :, None, :] - positions[:, None, :, :]) / std[:, None, None, :]
        distances = xp.sqrt(xp.sum(scaled_diff * scaled_diff, axis=-1))
        diagonal = xp.eye(pop_size, dtype=bool)[None, :, :]
        min_distance = xp.min(xp.where(diagonal, xp.inf, distances), axis=(1, 2))
        min_distance = xp.where(xp.isfinite(min_distance), min_distance, 0.0)
        penalty = ((min_distance + 0.1) ** 2) / (1.0 + min_distance**2)
        return div * 0.1 * non_repeat_percent * penalty

    def _advance_one_epoch(self, state: BatchedRunState, epoch: int) -> None:
        epoch_started = perf_counter()
        if self.optimizer_name == "DE-M":
            method, threshold = "direct", self.threshold
        else:
            method = "cholesky_solve" if self.optimizer_name == "DE-MC" else "cholesky"
            threshold = self.macro_threshold if self.optimizer_name == "MaCRO-DE" else self.threshold
        close, far, _ = self._classification(state.positions, method, threshold)

        if self.optimizer_name == "MaCRO-DE":
            started = perf_counter()
            donors, crossover, f_vectors, pcr_values = self._macro_random_plan(state, close, far)
            self.timing.add("donor_construction", perf_counter() - started)
            factors = self._device_plan_buffer("f_vectors", f_vectors)
            mutants = self._stage(
                "mutation",
                lambda: self._gather_mutants(state.positions, donors, factors),
            )
        else:
            started = perf_counter()
            donors, crossover = self._de_random_plan(state, close, far)
            self.timing.add("donor_construction", perf_counter() - started)
            mutants = self._stage(
                "mutation",
                lambda: self._gather_mutants(state.positions, donors, self.wf),
            )

        mutants = self._stage(
            "mutation_boundary",
            lambda: self.xp.clip(mutants, self.lb_backend, self.ub_backend),
        )

        trial = self._stage(
            "crossover",
            lambda: self.xp.where(
                self._device_plan_buffer("crossover", crossover), mutants, state.positions
            ),
        )
        # The mutant and parent operands are already bound-corrected, making
        # the scalar implementation's second clip an exact identity here.
        trial_fitness = self._evaluate(trial)
        def select_and_track():
            improved = trial_fitness < state.fitness
            state.positions = self.xp.where(improved[..., None], trial, state.positions)
            state.fitness = self.xp.where(improved, trial_fitness, state.fitness)
            rows = self._rows(len(state.seeds))[:, 0]
            current_indices = self.xp.argmin(state.fitness, axis=1)
            current_fitness = state.fitness[rows, current_indices]
            current_positions = state.positions[rows, current_indices]
            state.current_best_fitness = current_fitness.copy()
            state.current_best_positions = current_positions.copy()
            global_improved = current_fitness < state.best_fitness
            state.best_fitness = self.xp.where(global_improved, current_fitness, state.best_fitness)
            state.best_positions = self.xp.where(
                global_improved[:, None], current_positions, state.best_positions
            )
            return current_fitness

        current_fitness = self._stage("selection", select_and_track)
        def store_history():
            state.algorithm_state["history_buffer"][:, epoch - 1] = state.best_fitness
            state.algorithm_state["current_history_buffer"][:, epoch - 1] = current_fitness

        self._stage("history_store", store_history)
        state.epoch_counts += 1

        if self.optimizer_name in {"DE-MC-CF", "MaCRO-DE"}:
            algo = state.algorithm_state
            div_awad = self._stage("awad", lambda: self._awad(state.positions))
            algo["div_max_seen"] = self.xp.maximum(algo["div_max_seen"], div_awad)
            div_norm = self.xp.clip(div_awad / (algo["div_max_seen"] + 1.0e-9), 0.0, 1.0)
            algo["div_norm_for_update"] = div_norm
            started = perf_counter()
            adaptive_values = self.backend.to_cpu(self.xp.stack((div_awad, div_norm), axis=1))
            transfer_wall = perf_counter() - started
            pending_gpu = self._flush_gpu_events() if self.backend.uses_gpu else 0.0
            self.timing.add("adaptive_transfer", max(0.0, transfer_wall - pending_gpu))
            algo["div_norm_cpu"] = adaptive_values[:, 1].copy()

        if self.optimizer_name == "MaCRO-DE":
            algo = state.algorithm_state
            fmean = np.mean(f_vectors, axis=(1, 2))
            for run_offset in range(len(state.seeds)):
                algo["div_awad_history"][run_offset].append(float(adaptive_values[run_offset, 0]))
                algo["div_norm_history"][run_offset].append(float(adaptive_values[run_offset, 1]))
                algo["pcr_history"][run_offset].append(float(pcr_values[run_offset]))
                algo["fmean_history"][run_offset].append(float(fmean[run_offset]))

        if epoch == 1 and state.capture_trace:
            state.trace.update(
                close_masks=close.copy(),
                far_masks=far.copy(),
                donor_indices=donors.copy(),
                crossover_masks=crossover.copy(),
                trial_population=self.backend.to_cpu(trial),
                trial_fitness=self.backend.to_cpu(trial_fitness),
                first_generation_population=self.backend.to_cpu(state.positions),
                first_generation_fitness=self.backend.to_cpu(state.fitness),
                first_generation_best=self.backend.to_cpu(state.best_fitness),
            )
            if self.optimizer_name == "MaCRO-DE":
                state.trace["f_vectors"] = f_vectors.copy()
                state.trace["pcr_values"] = pcr_values.copy()
        self.timing.epochs += 1
        self.timing.add("epoch_total", perf_counter() - epoch_started)

    def run(
        self,
        run_indices: Sequence[int],
        seeds: Sequence[int],
        capture_trace: bool = False,
        progress_callback=None,
        progress_interval: int = 100,
    ) -> tuple[list[dict], BatchedRunState, float]:
        self._select_objective_backend(len(seeds) * self.pop_size)
        # Calibration uses copied data and is excluded from scientific runtime.
        started = perf_counter()
        state = self.initialize(run_indices, seeds, capture_trace=capture_trace)
        objective_label = (
            f"gpu-cec:{self.gpu_objective.function_name}"
            if self.selected_objective_backend == "gpu"
            else (
                f"numpy-cec:{self.vectorized_cpu_objective.function_name}"
                if self.selected_objective_backend == "numpy"
                else f"opfunu-{self.objective_evaluator.strategy}"
            )
        )
        state.algorithm_state["objective_strategy"] = objective_label
        # Heartbeat percentages describe generations only; initialization and
        # non-scientific calibration remain part of overall wall time.
        self.timing = BatchTiming()
        for epoch in range(1, self.epochs + 1):
            epoch_started = perf_counter()
            self._advance_one_epoch(state, epoch)
            if progress_callback is not None and (
                epoch == 1 or epoch == self.epochs or epoch % max(1, progress_interval) == 0
            ):
                progress_callback(
                    epoch,
                    self.epochs,
                    perf_counter() - started,
                    perf_counter() - epoch_started,
                    self.timing.summary(),
                    objective_label,
                    self.backend.memory_stats(),
                )
        state.stopped[:] = True
        elapsed = perf_counter() - started
        started_transfer = perf_counter()
        histories = self.backend.to_cpu(
            self.xp.stack(
                (
                    state.algorithm_state["history_buffer"],
                    state.algorithm_state["current_history_buffer"],
                ),
                axis=1,
            )
        )
        transfer_wall = perf_counter() - started_transfer
        pending_gpu = self._flush_gpu_events() if self.backend.uses_gpu else 0.0
        self.timing.add("history_transfer", max(0.0, transfer_wall - pending_gpu))
        state.histories = [row[0].astype(float, copy=False).tolist() for row in histories]
        state.current_histories = [row[1].astype(float, copy=False).tolist() for row in histories]
        best_positions = self.backend.to_cpu(state.best_positions)
        best_fitness = self.backend.to_cpu(state.best_fitness)
        outputs = []
        for offset, _run_index in enumerate(state.run_indices):
            output = {
                "best_fitness": float(best_fitness[offset]),
                "best_solution": best_positions[offset].copy(),
                # Concurrent runs each experience the batch wall clock.  Batch
                # summaries separately report throughput/effective time per run.
                "runtime": float(elapsed),
                "curve": np.asarray(state.histories[offset], dtype=np.float64),
            }
            if capture_trace:
                output["trace"] = {
                    key: np.asarray(value[offset]).copy() for key, value in state.trace.items()
                }
            outputs.append(output)
        return outputs, state, elapsed
