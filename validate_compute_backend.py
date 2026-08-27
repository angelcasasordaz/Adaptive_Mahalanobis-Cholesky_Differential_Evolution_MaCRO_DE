"""Small CPU/GPU numerical equivalence smoke test (no CEC experiment)."""
from __future__ import annotations

import argparse
import sys

import numpy as np

from compute_backend import (
    ComputeBackend,
    estimate_population_batch_size,
    estimate_effective_batch_size,
    initialize_gpu,
    parse_gpu_batch_size,
    resolve_cpu_workers,
    resolve_gpu_batch_size,
    resolve_objective_workers,
)


def calculate(backend, population, threshold):
    n_dims = population.shape[1]
    covariance, cholesky, distances = backend.mahalanobis_cpu(
        population, n_dims, "cholesky"
    )
    close, far, _ = backend.close_far_indices(
        population, n_dims, threshold, "cholesky"
    )
    return covariance, cholesky, distances, close, far


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--compute-device", choices=["cpu", "gpu", "hybrid"], default="cpu")
    parser.add_argument("--gpu-memory-fraction", type=float, default=0.85)
    parser.add_argument("--gpu-batch-size", default="auto")
    parser.add_argument("--population-size", type=int, default=50)
    parser.add_argument("--dimensions", type=int, default=30)
    parser.add_argument("--max-batch-size", type=int, default=30)
    args = parser.parse_args()
    rng = np.random.default_rng(20260824)
    assert resolve_gpu_batch_size("auto", 8, 30) == 8
    assert resolve_gpu_batch_size("auto", 8, 3) == 3
    assert resolve_gpu_batch_size(4, 8, 30) == 4
    assert resolve_gpu_batch_size(12, 8, 30) == 8
    assert resolve_cpu_workers(None, 12) == 8
    assert resolve_cpu_workers(None, 8) == 6
    assert resolve_cpu_workers(3, 12) == 3
    workers = resolve_objective_workers(None, 15)
    assert workers >= 1
    assert estimate_effective_batch_size(30, 30, 50, 30, workers) in {1, 2, 4, 8, 16, 30}
    population = rng.normal(size=(64, 12))
    threshold = 12.0
    cpu_result = calculate(ComputeBackend("cpu"), population, threshold)

    if args.compute_device == "cpu":
        covariance, cholesky, distances, close, far = cpu_result
        assert covariance.shape == (12, 12)
        assert cholesky is not None
        assert distances.shape == (64,)
        assert close.size + far.size == 64
        cpu_batch = ComputeBackend("cpu").mahalanobis_cpu(
            np.stack((population, population + 0.25)), 12, "cholesky"
        )
        assert cpu_batch[0].shape == (2, 12, 12)
        assert cpu_batch[1].shape == (2, 12, 12)
        assert cpu_batch[2].shape == (2, 64)
        assert "cupy" not in sys.modules
        print("CPU backend and batched-kernel smoke test passed; CuPy was not imported.")
        return

    requested_batch = parse_gpu_batch_size(args.gpu_batch_size)
    info = initialize_gpu(memory_fraction=args.gpu_memory_fraction)
    capacity = estimate_population_batch_size(
        args.population_size,
        args.dimensions,
        min(info.free_memory_bytes, info.memory_limit_bytes),
        max_batch_size=args.max_batch_size,
    )
    resolved_batch = resolve_gpu_batch_size(
        requested_batch,
        capacity,
        args.max_batch_size,
    )
    gpu_result = calculate(ComputeBackend(args.compute_device), population, threshold)
    labels = ("covariance", "Cholesky", "Mahalanobis distances")
    for label, cpu_value, gpu_value in zip(labels, cpu_result[:3], gpu_result[:3]):
        np.testing.assert_allclose(cpu_value, gpu_value, rtol=1e-9, atol=1e-10)
        print(f"{label}: match")
    np.testing.assert_array_equal(cpu_result[3], gpu_result[3])
    np.testing.assert_array_equal(cpu_result[4], gpu_result[4])
    print("Close/far: match")

    # Exercise the larger (batch, population, dimensions) kernel independently
    # of MEALPY run state and confirm each batch equals its CPU calculation.
    synthetic_batch_size = min(resolved_batch, 4)
    batched_population = rng.normal(size=(synthetic_batch_size, 64, 12))
    gpu_batch = ComputeBackend(args.compute_device).mahalanobis_cpu(
        batched_population, 12, "cholesky"
    )
    for batch_index in range(synthetic_batch_size):
        cpu_item = ComputeBackend("cpu").mahalanobis_cpu(
            batched_population[batch_index], 12, "cholesky"
        )
        for cpu_value, gpu_value in zip(cpu_item, gpu_batch):
            np.testing.assert_allclose(
                cpu_value,
                gpu_value[batch_index],
                rtol=1e-9,
                atol=1e-10,
            )

    assert info.memory_limit_bytes == int(info.total_memory_bytes * info.memory_fraction)
    print("Batched Mahalanobis kernel: match")
    print("GPU backend validation passed")
    print(f"GPU: {info.name}")
    print(f"CuPy: {info.cupy_version}")
    print(f"Memory: {info.total_memory_bytes / 1024**3:.2f} GB")
    print(f"Memory fraction: {info.memory_fraction:.2f}")
    print(f"Memory pool limit: {info.memory_limit_bytes / 1024**3:.2f} GB")
    print(f"Resolved batch capacity: {resolved_batch}")


if __name__ == "__main__":
    main()
