"""Short throughput benchmark for the batched engine; writes no experiment output."""
from __future__ import annotations

import argparse
import importlib
import multiprocessing
import os
import numpy as np
from concurrent.futures import ProcessPoolExecutor

from compute_backend import initialize_gpu, resolve_objective_workers
from gpu_batching import BatchedDEEngine
from objective_evaluation import ObjectiveSpec, initialize_objective_worker


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--compute-device", choices=["cpu", "gpu", "hybrid"], default="cpu")
    parser.add_argument("--function", default="F12017")
    parser.add_argument("--optimizer", choices=["DE-M", "DE-MC", "DE-MC-CF", "MaCRO-DE"], default="DE-M")
    parser.add_argument("--dimensions", type=int, default=30)
    parser.add_argument("--population-size", type=int, default=50)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--runs", type=int, default=30)
    parser.add_argument("--objective-workers", type=int, default=None)
    parser.add_argument("--objective-evaluation", choices=["auto", "serial", "process"], default="auto")
    parser.add_argument("--cec-objective-backend", choices=["auto", "opfunu", "numpy", "gpu"], default="auto")
    parser.add_argument("--cec-gpu-verification-points", type=int, default=512)
    parser.add_argument("--gpu-memory-fraction", type=float, default=0.85)
    args = parser.parse_args()
    if args.compute_device in {"gpu", "hybrid"}:
        info = initialize_gpu(memory_fraction=args.gpu_memory_fraction)
        print(f"GPU: {info.name}; cap={info.memory_fraction:.0%}")

    workers = resolve_objective_workers(args.objective_workers, max(1, (os.cpu_count() or 1) - 1))
    module = importlib.import_module("opfunu.cec_based.cec2017")
    function_class = getattr(module, args.function)
    benchmark = function_class(ndim=args.dimensions)
    gpu_objective = None
    vectorized_cpu_objective = None
    if args.compute_device in {"gpu", "hybrid"} or args.cec_objective_backend == "numpy":
        from cec2017_gpu import CEC2017_GPU_CANDIDATES, CEC2017GpuObjective, verify_gpu_objective
        from compute_backend import ComputeBackend

        if args.function in CEC2017_GPU_CANDIDATES:
            cpu_candidate = CEC2017GpuObjective(args.function, benchmark, np)
            cpu_report = verify_gpu_objective(
                cpu_candidate, benchmark, np.asarray, lambda: None,
                random_points=args.cec_gpu_verification_points,
            )
            vectorized_cpu_objective = cpu_candidate if cpu_report.verified else None
        if (
            args.compute_device in {"gpu", "hybrid"}
            and args.function in CEC2017_GPU_CANDIDATES
            and args.cec_objective_backend in {"auto", "gpu"}
        ):
            verification_backend = ComputeBackend(args.compute_device)
            candidate = CEC2017GpuObjective(args.function, benchmark, verification_backend.xp)
            report = verify_gpu_objective(
                candidate,
                benchmark,
                verification_backend.to_cpu,
                verification_backend.synchronize,
                random_points=args.cec_gpu_verification_points,
            )
            print(
                f"CEC GPU verification: verified={report.verified} points={report.points} "
                f"max_abs={report.max_absolute_error:.6e} max_rel={report.max_relative_error:.6e} "
                f"mismatches={report.mismatches}"
            )
            gpu_objective = candidate if report.verified else None
    executor = None
    if workers > 1 and args.objective_evaluation != "serial":
        executor = ProcessPoolExecutor(
            max_workers=workers,
            mp_context=multiprocessing.get_context("spawn"),
            initializer=initialize_objective_worker,
        )

    candidates = [value for value in (1, 2, 4, 8, 16, 24, 30) if value <= args.runs]
    rows = []
    try:
        dispatch_warm = BatchedDEEngine(
            args.optimizer, benchmark.evaluate, benchmark.lb, benchmark.ub,
            1, args.population_size, args.compute_device,
            objective_executor=executor, objective_workers=workers,
            objective_strategy=args.objective_evaluation,
            objective_spec=ObjectiveSpec(function_class.__module__, function_class.__name__, args.dimensions),
            gpu_objective=gpu_objective, cec_objective_backend=args.cec_objective_backend,
            vectorized_cpu_objective=vectorized_cpu_objective,
        )
        dispatch_warm.objective_evaluator.calibrate(
            benchmark.lb, benchmark.ub, candidates[-1] * args.population_size
        )
        if dispatch_warm.objective_evaluator.strategy == "process":
            warm_vectors = np.random.default_rng(8_900_001).uniform(
                benchmark.lb, benchmark.ub, size=(max(workers, 2), args.dimensions)
            )
            dispatch_warm.objective_evaluator.evaluate(warm_vectors)

        # Untimed warm-up for imports, CUDA libraries, and kernels.
        warm = BatchedDEEngine(
            args.optimizer, benchmark.evaluate, benchmark.lb, benchmark.ub,
            1, args.population_size, args.compute_device,
            objective_executor=executor, objective_workers=workers,
            objective_strategy=args.objective_evaluation,
            objective_spec=ObjectiveSpec(function_class.__module__, function_class.__name__, args.dimensions),
            gpu_objective=gpu_objective, cec_objective_backend=args.cec_objective_backend,
            vectorized_cpu_objective=vectorized_cpu_objective,
        )
        warm.run([0], [9_000_001])

        for batch_size in candidates:
            local_benchmark = function_class(ndim=args.dimensions)
            engine = BatchedDEEngine(
                args.optimizer, local_benchmark.evaluate, local_benchmark.lb, local_benchmark.ub,
                args.epochs, args.population_size, args.compute_device,
                objective_executor=executor, objective_workers=workers,
                objective_strategy=args.objective_evaluation,
                objective_spec=ObjectiveSpec(function_class.__module__, function_class.__name__, args.dimensions),
                gpu_objective=gpu_objective, cec_objective_backend=args.cec_objective_backend,
                vectorized_cpu_objective=vectorized_cpu_objective,
            )
            indices = list(range(batch_size))
            seeds = [9_100_000 + index for index in indices]
            _, _, elapsed = engine.run(indices, seeds)
            timing = engine.timing.summary()
            epoch_total = max(timing.get("epoch_total", 0.0), 1.0e-12)
            memory = engine.backend.memory_stats()
            rows.append((
                batch_size,
                elapsed,
                1000.0 * elapsed / args.epochs,
                batch_size / elapsed,
                100.0 * timing["gpu_kernel"] / epoch_total,
                100.0 * timing["fitness"] / epoch_total,
                100.0 * timing["transfer"] / epoch_total,
                (
                    f"gpu-cec:{args.function}"
                    if engine.selected_objective_backend == "gpu"
                    else (
                        f"numpy-cec:{args.function}"
                        if engine.selected_objective_backend == "numpy"
                        else f"opfunu-{engine.objective_evaluator.strategy}"
                    )
                ),
                memory,
                timing,
            ))
    finally:
        if executor is not None:
            executor.shutdown(wait=True)

    kernel_label = "GPU%" if args.compute_device in {"gpu", "hybrid"} else "Numerical%"
    print(f"Batch | Total(s) | ms/epoch | runs/s | {kernel_label} | Fitness% | Transfer% | Other% | Objective")
    for row in rows:
        batch, total, milliseconds, throughput, gpu_pct, fitness_pct, transfer_pct, strategy, memory, timing = row
        other_pct = 100.0 * timing["other"] / max(timing.get("epoch_total", 0.0), 1.0e-12)
        print(
            f"{batch:>5} | {total:>8.3f} | {milliseconds:>8.2f} | {throughput:>6.3f} | "
            f"{gpu_pct:>5.1f} | {fitness_pct:>8.1f} | {transfer_pct:>9.1f} | "
            f"{other_pct:>6.1f} | {strategy}"
        )
        print(
            "      stages(s): "
            f"cov={timing.get('covariance', 0.0):.4f} | "
            f"factor={timing.get('factorization_inverse', 0.0):.4f} | "
            f"mahal={timing.get('mahalanobis', 0.0):.4f} | "
            f"donors={timing.get('donor_construction', 0.0):.4f} | "
            f"mutation={timing.get('mutation', 0.0):.4f} | "
            f"crossover={timing.get('crossover', 0.0):.4f} | "
            f"boundary={timing.get('mutation_boundary', 0.0) + timing.get('trial_boundary', 0.0):.4f} | "
            f"download={timing.get('trial_download', 0.0):.4f} | "
            f"fitness={timing.get('fitness_cpu', 0.0) + timing.get('fitness_cpu_vectorized', 0.0) + timing.get('objective_gpu', 0.0):.4f} | "
            f"upload={timing.get('fitness_upload', 0.0):.4f} | "
            f"selection={timing.get('selection', 0.0):.4f}"
        )
        print(
            f"      donor/RNG={timing.get('donor_construction', 0.0):.4f}s | "
            f"objective_gpu={timing.get('objective_gpu', 0.0):.4f}s | "
            f"history/best={timing.get('history_store', 0.0):.4f}s"
        )
        accounted = (
            timing["gpu_kernel"] + timing["fitness"] + timing["transfer"]
            + timing.get("donor_rng", 0.0) + timing.get("python_orchestration", 0.0)
        )
        print(
            f"      accounted={accounted:.4f}s / epoch_total={timing.get('epoch_total', 0.0):.4f}s | "
            f"python_orchestration={timing.get('python_orchestration', 0.0):.4f}s"
        )
        if args.compute_device in {"gpu", "hybrid"}:
            print(
                f"      VRAM used={memory['used_bytes']/1024**2:.1f}MiB | "
                f"pool used={memory['pool_used_bytes']/1024**2:.1f}MiB | "
                f"pool total={memory['pool_total_bytes']/1024**2:.1f}MiB"
            )


if __name__ == "__main__":
    main()
