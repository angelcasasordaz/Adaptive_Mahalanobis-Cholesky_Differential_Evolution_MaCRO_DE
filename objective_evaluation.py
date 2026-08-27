"""Persistent, ordered CPU evaluation for CPU-only benchmark objectives."""
from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter
from typing import Callable
import importlib
import warnings

import numpy as np


_WORKER_BENCHMARKS: dict[tuple[str, str, int], object] = {}
_THREADPOOL_LIMIT = None


def initialize_objective_worker() -> None:
    """Keep each spawned process single-threaded to avoid nested BLAS fan-out."""
    global _THREADPOOL_LIMIT
    warnings.filterwarnings("ignore", message="pkg_resources is deprecated as an API.*")
    try:
        from threadpoolctl import threadpool_limits

        _THREADPOOL_LIMIT = threadpool_limits(limits=1)
    except Exception:
        _THREADPOOL_LIMIT = None


def _evaluate_objective_chunk(task):
    spec, chunk = task
    key = (spec.module, spec.class_name, spec.ndim)
    benchmark = _WORKER_BENCHMARKS.get(key)
    if benchmark is None:
        module = importlib.import_module(spec.module)
        benchmark = getattr(module, spec.class_name)(ndim=spec.ndim)
        _WORKER_BENCHMARKS[key] = benchmark
    values = np.empty(len(chunk), dtype=np.float64)
    for index, vector in enumerate(chunk):
        values[index] = float(benchmark.evaluate(vector))
    return values


@dataclass(frozen=True)
class ObjectiveSpec:
    module: str
    class_name: str
    ndim: int


@dataclass
class ObjectiveEvaluationStats:
    strategy: str = "serial"
    calibration_seconds: float = 0.0
    seconds_per_vector: float = 0.0
    evaluations: int = 0


class ObjectiveEvaluator:
    """Evaluate flattened vectors serially or through one external spawn pool."""

    PROCESS_THRESHOLD_SECONDS = 0.05

    def __init__(
        self,
        objective: Callable[[np.ndarray], float],
        executor=None,
        workers: int = 1,
        strategy: str = "auto",
        spec: ObjectiveSpec | None = None,
    ):
        if strategy not in {"auto", "serial", "process"}:
            raise ValueError("objective strategy must be auto, serial, or process")
        self.objective = objective
        self.executor = executor
        self.workers = max(1, int(workers))
        self.requested_strategy = strategy
        self.spec = spec
        self.stats = ObjectiveEvaluationStats()
        self._calibrated_for_vectors = 0

    @property
    def strategy(self) -> str:
        return self.stats.strategy

    def calibrate(self, lb, ub, expected_vectors: int) -> ObjectiveEvaluationStats:
        expected_vectors = max(1, int(expected_vectors))
        if self._calibrated_for_vectors == expected_vectors:
            return self.stats
        sample_size = min(64, max(16, expected_vectors))
        rng = np.random.default_rng(0xCEC2017)
        sample = rng.uniform(lb, ub, size=(sample_size, len(lb)))
        started = perf_counter()
        self._evaluate_serial(sample)
        elapsed = perf_counter() - started
        seconds_per_vector = elapsed / sample_size
        estimated_serial = seconds_per_vector * expected_vectors
        can_process = self.executor is not None and self.spec is not None and self.workers > 1
        if self.requested_strategy == "process":
            if not can_process:
                raise RuntimeError("process objective evaluation requires a worker pool and ObjectiveSpec")
            selected = "process"
        elif self.requested_strategy == "serial":
            selected = "serial"
        else:
            selected = "process" if can_process and estimated_serial >= self.PROCESS_THRESHOLD_SECONDS else "serial"
        self.stats = ObjectiveEvaluationStats(
            strategy=selected,
            calibration_seconds=elapsed,
            seconds_per_vector=seconds_per_vector,
        )
        self._calibrated_for_vectors = expected_vectors
        return self.stats

    def _evaluate_serial(self, flat: np.ndarray) -> np.ndarray:
        values = np.empty(len(flat), dtype=np.float64)
        for index, vector in enumerate(flat):
            value = np.asarray(self.objective(vector), dtype=np.float64).reshape(-1)
            if value.size != 1:
                raise ValueError("The batched custom-DE engine requires a scalar objective.")
            values[index] = value[0]
        return values

    def _evaluate_process(self, flat: np.ndarray) -> np.ndarray:
        task_count = min(len(flat), self.workers)
        chunks = [chunk for chunk in np.array_split(flat, task_count) if len(chunk)]
        tasks = [(self.spec, np.ascontiguousarray(chunk)) for chunk in chunks]
        results = list(self.executor.map(_evaluate_objective_chunk, tasks))
        return np.concatenate(results)

    def evaluate(self, flat: np.ndarray) -> np.ndarray:
        flat = np.ascontiguousarray(flat, dtype=np.float64)
        if not self._calibrated_for_vectors:
            self.calibrate(
                np.min(flat, axis=0),
                np.max(flat, axis=0),
                len(flat),
            )
        if self.strategy == "process":
            values = self._evaluate_process(flat)
        else:
            values = self._evaluate_serial(flat)
        self.stats.evaluations += len(flat)
        return values
