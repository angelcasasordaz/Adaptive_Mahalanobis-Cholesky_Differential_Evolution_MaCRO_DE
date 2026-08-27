"""Optional NumPy/CuPy backend for custom MaCRO-DE numerical kernels."""
from __future__ import annotations

from dataclasses import dataclass
from importlib.util import find_spec
from typing import Any
import math
import os

import numpy as np


GPU_MODES = {"gpu", "hybrid"}
SUPPORTED_COMPUTE_DEVICES = {"cpu", *GPU_MODES}


class GPUBackendError(RuntimeError):
    """Raised when GPU execution was requested but cannot be initialized."""


@dataclass(frozen=True)
class GPUInfo:
    device_id: int
    device_count: int
    name: str
    cupy_version: str
    total_memory_bytes: int
    free_memory_bytes: int
    memory_fraction: float
    memory_limit_bytes: int


def normalize_compute_device(device: str) -> str:
    normalized = str(device).strip().lower()
    if normalized not in SUPPORTED_COMPUTE_DEVICES:
        choices = ", ".join(sorted(SUPPORTED_COMPUTE_DEVICES))
        raise ValueError(f"Unsupported compute device '{device}'. Choose one of: {choices}.")
    return normalized


def cupy_installed() -> bool:
    """Check package presence without importing CuPy or initializing CUDA."""
    return find_spec("cupy") is not None


def _load_cupy():
    try:
        import cupy as cp
    except Exception as exc:
        raise GPUBackendError(
            "GPU execution requested but CuPy/CUDA is not available. "
            "Use --compute-device cpu or install the Linux GPU requirements. "
            f"Details: {exc}"
        ) from exc
    return cp


def validate_memory_fraction(memory_fraction: float) -> float:
    fraction = float(memory_fraction)
    if not 0.0 < fraction <= 1.0:
        raise ValueError("GPU memory fraction must satisfy 0 < fraction <= 1.")
    return fraction


def parse_gpu_batch_size(value: Any):
    if isinstance(value, str) and value.strip().lower() == "auto":
        return "auto"
    try:
        batch_size = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError("GPU batch size must be 'auto' or a positive integer.") from exc
    if batch_size < 1:
        raise ValueError("GPU batch size must be 'auto' or a positive integer.")
    return batch_size


def resolve_gpu_batch_size(requested: Any, estimated_capacity: int, pending_runs: int) -> int:
    """Resolve run batching independently of CUDA worker/process count."""
    requested = parse_gpu_batch_size(requested)
    estimated_capacity = max(1, int(estimated_capacity))
    pending_runs = max(1, int(pending_runs))
    requested_limit = estimated_capacity if requested == "auto" else requested
    return max(1, min(requested_limit, estimated_capacity, pending_runs))


def resolve_cpu_workers(requested: int | None, logical_cpus: int | None = None) -> int:
    """Leave a useful share of the machine available unless explicitly overridden.

    The automatic policy reserves at least two logical CPUs and roughly one
    third of larger machines.  Integer rounding intentionally gives 8 -> 6
    and 12 -> 8 workers.  An explicit command-line value remains authoritative.
    """
    detected = logical_cpus if logical_cpus is not None else (os.cpu_count() or 1)
    logical = max(1, int(detected))
    if requested is not None:
        return max(1, int(requested))
    if logical == 1:
        return 1
    reserved = max(2, logical // 3)
    return max(1, logical - reserved)


def resolve_objective_workers(requested: int | None, cpu_worker_limit: int) -> int:
    """Use physical-core-like concurrency and avoid nested BLAS oversubscription."""
    logical = max(1, os.cpu_count() or 1)
    physical_like = max(1, (logical + 1) // 2)
    automatic = min(max(1, int(cpu_worker_limit)), physical_like)
    if requested is None:
        return automatic
    return max(1, min(int(requested), max(1, int(cpu_worker_limit)), logical))


def performance_batch_candidates(memory_capacity: int, pending_runs: int) -> list[int]:
    limit = max(1, min(int(memory_capacity), int(pending_runs)))
    return sorted({value for value in (1, 2, 4, 8, 16, limit) if value <= limit})


def estimate_effective_batch_size(
    memory_capacity: int,
    pending_runs: int,
    population_size: int,
    n_dims: int,
    objective_workers: int,
) -> int:
    """Initial performance heuristic, refined by optional measured calibration.

    It provides enough vectors to amortize CPU scheduling (roughly 32 per
    objective worker) and enough tensor elements to amortize small CUDA kernel
    launches, then rounds to a benchmarkable batch candidate.
    """
    population_size = max(1, int(population_size))
    n_dims = max(1, int(n_dims))
    objective_workers = max(1, int(objective_workers))
    cpu_target = math.ceil(max(128, 32 * objective_workers) / population_size)
    gpu_target = math.ceil(8192 / (population_size * n_dims))
    target = max(1, cpu_target, gpu_target)
    candidates = performance_batch_candidates(memory_capacity, pending_runs)
    return next((value for value in candidates if value >= target), candidates[-1])


def estimate_population_batch_size(
    population_size: int,
    n_dims: int,
    memory_budget_bytes: int,
    max_batch_size: int | None = None,
    safety_factor: float = 3.0,
) -> int:
    """Conservatively estimate capacity for batched DE numerical kernels.

    The estimate includes population/centered/mutation/trial buffers, three
    square matrices (covariance, factor, inverse), distances and masks. The
    safety factor covers solver workspaces and allocator fragmentation.
    """
    population_size = max(1, int(population_size))
    n_dims = max(1, int(n_dims))
    float_bytes = np.dtype(np.float64).itemsize
    vector_elements = 10 * population_size * n_dims
    # MaCRO-DE's exact batched AWAD calculation temporarily forms pairwise
    # standardized differences and distance/equality matrices.
    pairwise_elements = population_size * population_size * (n_dims + 2)
    matrix_elements = 3 * n_dims * n_dims
    scalar_elements = 4 * population_size
    bytes_per_batch = int(
        (vector_elements + pairwise_elements + matrix_elements + scalar_elements)
        * float_bytes
        * max(float(safety_factor), 1.0)
    )
    capacity = max(1, int(memory_budget_bytes) // max(bytes_per_batch, 1))
    if max_batch_size is not None:
        capacity = min(capacity, max(1, int(max_batch_size)))
    return capacity


def initialize_gpu(device_id: int = 0, memory_fraction: float = 0.85) -> GPUInfo:
    """Validate CUDA and select a device. Called only for gpu/hybrid modes."""
    cp = _load_cupy()
    memory_fraction = validate_memory_fraction(memory_fraction)
    try:
        device_count = int(cp.cuda.runtime.getDeviceCount())
        if device_count <= 0:
            raise RuntimeError("no CUDA devices were reported")
        if not 0 <= device_id < device_count:
            raise RuntimeError(
                f"CUDA device {device_id} is invalid; {device_count} device(s) available"
            )
        cp.cuda.Device(device_id).use()
        free_memory, total_memory = cp.cuda.runtime.memGetInfo()
        memory_limit = max(1, int(total_memory * memory_fraction))
        memory_pool = cp.get_default_memory_pool()
        memory_pool.set_limit(size=memory_limit)
        properties = cp.cuda.runtime.getDeviceProperties(device_id)
        raw_name = properties.get("name", "Unknown NVIDIA GPU")
        name = raw_name.decode("utf-8") if isinstance(raw_name, bytes) else str(raw_name)
        # Force a tiny allocation so driver/runtime errors occur before experiments start.
        cp.asarray([0.0], dtype=cp.float64)
        cp.cuda.get_current_stream().synchronize()
    except Exception as exc:
        raise GPUBackendError(
            "GPU execution requested but CuPy/CUDA is not available. "
            "Use --compute-device cpu or install the Linux GPU requirements. "
            f"Details: {exc}"
        ) from exc
    return GPUInfo(
        device_id,
        device_count,
        name,
        str(cp.__version__),
        int(total_memory),
        int(free_memory),
        memory_fraction,
        int(memory_pool.get_limit()),
    )


class ComputeBackend:
    """Array backend with explicit NumPy boundaries for MEALPY/OPFUNU."""

    def __init__(self, device: str = "cpu", device_id: int = 0):
        self.device = normalize_compute_device(device)
        self.device_id = int(device_id)
        self.xp = np
        if self.device in GPU_MODES:
            cp = _load_cupy()
            cp.cuda.Device(self.device_id).use()
            self.xp = cp

    @property
    def uses_gpu(self) -> bool:
        return self.device in GPU_MODES

    def asarray(self, value: Any):
        return self.xp.asarray(value, dtype=self.xp.float64)

    def to_cpu(self, value: Any, out: np.ndarray | None = None) -> np.ndarray:
        if self.uses_gpu:
            return self.xp.asnumpy(value, out=out)
        value = np.asarray(value)
        if out is not None:
            np.copyto(out, value)
            return out
        return value

    def empty_pinned(self, shape, dtype=np.float64) -> np.ndarray:
        """Allocate reusable page-locked host storage for GPU transfers."""
        if not self.uses_gpu:
            return np.empty(shape, dtype=dtype)
        dtype = np.dtype(dtype)
        memory = self.xp.cuda.alloc_pinned_memory(int(np.prod(shape)) * dtype.itemsize)
        return np.frombuffer(memory, dtype=dtype, count=int(np.prod(shape))).reshape(shape)

    def covariance(self, population, n_dims: int):
        xp = self.xp
        pop = self.asarray(population)
        if pop.ndim == 2:
            sigma = xp.cov(pop, rowvar=False)
        elif pop.ndim == 3:
            centered = pop - xp.mean(pop, axis=1, keepdims=True)
            denominator = max(pop.shape[1] - 1, 1)
            sigma = xp.matmul(centered.swapaxes(-1, -2), centered) / denominator
        else:
            raise ValueError("Population must have shape (population, dims) or (batch, population, dims).")
        if sigma.ndim == 0:
            sigma = sigma.reshape(1, 1)
        expected_tail = (n_dims, n_dims)
        if sigma.shape[-2:] != expected_tail:
            leading = pop.shape[:-2]
            sigma = xp.broadcast_to(
                xp.eye(n_dims, dtype=xp.float64) * 1e-6,
                (*leading, n_dims, n_dims),
            ).copy()
        identity = xp.eye(n_dims, dtype=xp.float64)
        return (sigma + sigma.swapaxes(-1, -2)) / 2.0 + 1e-6 * identity

    def covariance_inverse(self, sigma, method: str):
        xp = self.xp
        n_dims = sigma.shape[-1]
        try:
            if method == "direct":
                return xp.linalg.inv(sigma)
            if method in {"cholesky", "cholesky_solve"}:
                chol = xp.linalg.cholesky(sigma)
                identity = xp.broadcast_to(
                    xp.eye(n_dims, dtype=xp.float64), sigma.shape
                )
                return xp.linalg.solve(
                    chol.swapaxes(-1, -2), xp.linalg.solve(chol, identity)
                )
            raise ValueError(f"Unknown covariance inverse method: {method}")
        except xp.linalg.LinAlgError:
            return xp.linalg.pinv(sigma)

    def covariance_factor(self, sigma, method: str):
        """Return one reusable factor and how distances must consume it."""
        xp = self.xp
        try:
            if method == "direct":
                return xp.linalg.inv(sigma), "inverse"
            if method == "cholesky_solve":
                return xp.linalg.cholesky(sigma), "cholesky"
            if method == "cholesky":
                return self.covariance_inverse(sigma, "cholesky"), "inverse"
            raise ValueError(f"Unknown covariance inverse method: {method}")
        except xp.linalg.LinAlgError:
            return xp.linalg.pinv(sigma), "inverse"

    def distances_from_factor(self, population, factor, factor_kind: str):
        """Compute squared Mahalanobis distances from an existing factorization."""
        xp = self.xp
        pop = self.asarray(population)
        diff = pop - xp.mean(pop, axis=-2, keepdims=True)
        if factor_kind == "cholesky":
            whitened = xp.linalg.solve(factor, diff.swapaxes(-1, -2))
            return xp.sum(whitened * whitened, axis=-2)
        if factor_kind == "inverse":
            return xp.sum(xp.matmul(diff, factor) * diff, axis=-1)
        raise ValueError(f"Unknown covariance factor kind: {factor_kind}")

    def mahalanobis(self, population, n_dims: int, method: str = "cholesky"):
        """Return covariance, Cholesky (if valid), distances, all on this backend."""
        xp = self.xp
        pop = self.asarray(population)
        sigma = self.covariance(pop, n_dims)
        if method == "cholesky":
            try:
                chol = xp.linalg.cholesky(sigma)
            except xp.linalg.LinAlgError:
                chol = None
            sigma_inv = self.covariance_inverse(sigma, method)
            dist2 = self.distances_from_factor(pop, sigma_inv, "inverse")
            return sigma, chol, dist2
        factor, factor_kind = self.covariance_factor(sigma, method)
        chol = factor if factor_kind == "cholesky" else None
        dist2 = self.distances_from_factor(pop, factor, factor_kind)
        return sigma, chol, dist2

    def mahalanobis_distances(self, population, n_dims: int, method: str = "cholesky"):
        """Distance-only path that avoids the unused/duplicate Cholesky factor."""
        xp = self.xp
        pop = self.asarray(population)
        sigma = self.covariance(pop, n_dims)
        factor, factor_kind = self.covariance_factor(sigma, method)
        return self.distances_from_factor(pop, factor, factor_kind)

    def mahalanobis_cpu(self, population, n_dims: int, method: str = "cholesky"):
        sigma, chol, dist2 = self.mahalanobis(population, n_dims, method)
        return (
            self.to_cpu(sigma),
            None if chol is None else self.to_cpu(chol),
            self.to_cpu(dist2),
        )

    def close_far_indices(
        self,
        population,
        n_dims: int,
        threshold: float,
        method: str,
        include_distances: bool = True,
    ):
        """Transfer only final small index arrays back to the CPU boundary."""
        dist2 = self.mahalanobis_distances(population, n_dims, method)
        if include_distances:
            dist2_cpu = self.to_cpu(dist2)
            close_mask = dist2_cpu <= threshold
            return np.flatnonzero(close_mask), np.flatnonzero(~close_mask), dist2_cpu
        close_mask = self.to_cpu(dist2 <= threshold).astype(bool, copy=False)
        return np.flatnonzero(close_mask), np.flatnonzero(~close_mask)

    def close_far_masks(
        self,
        population,
        n_dims: int,
        threshold: float,
        method: str,
    ):
        """Return backend-resident masks without flattening a run batch."""
        dist2 = self.mahalanobis_distances(population, n_dims, method)
        return dist2 <= threshold, dist2 > threshold, dist2

    def memory_stats(self) -> dict[str, int]:
        if not self.uses_gpu:
            return {"used_bytes": 0, "pool_used_bytes": 0, "pool_total_bytes": 0}
        pool = self.xp.get_default_memory_pool()
        free_bytes, total_bytes = self.xp.cuda.runtime.memGetInfo()
        return {
            "used_bytes": int(total_bytes - free_bytes),
            "pool_used_bytes": int(pool.used_bytes()),
            "pool_total_bytes": int(pool.total_bytes()),
        }

    def synchronize(self) -> None:
        """Synchronize only when verification/profiling requires a wall time."""
        if self.uses_gpu:
            self.xp.cuda.get_current_stream().synchronize()

    def free_cached_blocks(self) -> None:
        """Release unused CuPy pool blocks after a failed/finished run batch."""
        if self.uses_gpu:
            self.xp.get_default_memory_pool().free_all_blocks()
            self.xp.get_default_pinned_memory_pool().free_all_blocks()


def is_gpu_out_of_memory(exc: BaseException) -> bool:
    """Recognize CuPy OOM without importing CuPy on the CPU-only path."""
    cls = type(exc)
    module = cls.__module__.lower()
    name = cls.__name__.lower()
    message = str(exc).lower()
    is_cupy_error = "cupy" in module
    return is_cupy_error and (
        "outofmemory" in name
        or "out of memory" in message
        or "memoryallocation" in message
    )
