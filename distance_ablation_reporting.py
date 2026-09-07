"""Independent distance-ablation exports and optimization-free regeneration."""
import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import rankdata

from distance_metrics import DISTANCE_LABELS


def plot_archive(archive_path, figure_dir):
    """Regenerate a function figure solely from its portable NPZ archive."""
    with np.load(archive_path, allow_pickle=False) as data:
        function = str(data["function"])
        dims = int(data["dims"])
        means = data["mean_fitness"]
        evaluations = data["function_evaluations"]
        labels = data["labels"]
    fig, ax = plt.subplots(figsize=(10, 5))
    for index, (label, curve) in enumerate(zip(labels, means)):
        ax.plot(evaluations, curve, label=str(label),
                color=plt.get_cmap("tab10")(index),
                marker=("o", "s", "^", "D", "X")[index],
                markevery=max(1, len(curve) // 18), markersize=4, linewidth=1.5)
    # Match the project's raw CEC fitness convention, including CEC bias.
    ax.set_yscale("log" if np.all(means > 0) else "symlog")
    ax.set(title=f"CEC2017 {function} — D{dims}",
           xlabel="Cumulative Function Evaluations", ylabel="Fitness")
    ax.grid(True, which="both", alpha=0.25)
    ax.legend()
    fig.tight_layout()
    figure_dir = Path(figure_dir)
    figure_dir.mkdir(parents=True, exist_ok=True)
    for suffix in ("png", "pdf"):
        fig.savefig(figure_dir / f"{function}_D{dims}_distance_convergence.{suffix}", dpi=300)
    plt.close(fig)


def export_distance_ablation(results, functions, args, paths):
    import main as framework

    labels = [DISTANCE_LABELS[m] for m in framework.DISTANCE_ABLATION_METRICS]
    # Initialization plus one full population evaluation per epoch. MEALPY's
    # scalar Problem probes the objective once; the batched engine does not.
    scalar_probe = int(args.compute_device == "cpu" and not framework.cpu_batching_enabled(args))
    evaluations = (np.arange(1, args.epochs + 1) + 1) * args.pop_size + scalar_probe
    seeds = args.seed_base + np.arange(args.runs)
    result_dir = Path(paths.res_dir)
    result_dir.mkdir(parents=True, exist_ok=True)
    summary = []
    for function in functions:
        entries = [results[function][label] for label in labels]
        curves = np.stack([entry["curves_runs"] for entry in entries])
        finals = np.stack([entry["fitness_runs"] for entry in entries])
        if curves.shape != (5, args.runs, args.epochs) or finals.shape != (5, args.runs):
            raise ValueError(f"Incomplete distance-ablation data for {function}")
        if not np.all(np.isfinite(curves)) or not np.all(np.isfinite(finals)):
            raise ValueError(f"Non-finite distance-ablation data for {function}")
        if not np.array_equal(curves[:, :, -1], finals):
            raise ValueError(f"Final fitness/history mismatch for {function}")
        bias = float(args.function_map[function](ndim=args.dims).f_global)
        errors = finals - bias
        means = curves.mean(axis=1)
        ranks = rankdata(errors.mean(axis=1), method="average")
        stem = f"{function}_D{args.dims}_convergence"
        archive = result_dir / f"{stem}.npz"
        np.savez_compressed(
            archive, function=function, dims=args.dims, labels=np.asarray(labels),
            metrics=np.asarray(framework.DISTANCE_ABLATION_METRICS), seeds=seeds,
            function_evaluations=evaluations, fitness_runs=curves,
            mean_fitness=means, error_runs=curves-bias, mean_error=means-bias,
            final_fitness=finals, final_error=errors, optimum=bias,
            minkowski_p=framework.DISTANCE_ABLATION_MINKOWSKI_P,
        )
        # Long-form data retain all configured independent runs and their seeds.
        per_run = []
        mean_rows = []
        for index, (metric, label) in enumerate(zip(framework.DISTANCE_ABLATION_METRICS, labels)):
            per_run.append(pd.DataFrame({
                "metric": metric, "variant": label,
                "run": np.repeat(np.arange(1, args.runs+1), args.epochs),
                "seed": np.repeat(seeds, args.epochs),
                "function_evaluations": np.tile(evaluations, args.runs),
                "fitness": curves[index].ravel(),
                "error": (curves[index]-bias).ravel(),
            }))
            mean_rows.append(pd.DataFrame({
                "metric": metric, "variant": label,
                "function_evaluations": evaluations,
                "mean_fitness": means[index], "mean_error": means[index]-bias,
            }))
            values = errors[index]
            summary.append(dict(
                function=function, dimension=args.dims, metric=metric, variant=label,
                mean=float(values.mean()), median=float(np.median(values)),
                standard_deviation=float(values.std()), best=float(values.min()),
                worst=float(values.max()), rank=float(ranks[index]),
            ))
        pd.concat(per_run, ignore_index=True).to_csv(result_dir / f"{stem}_runs.csv", index=False)
        pd.concat(mean_rows, ignore_index=True).to_csv(result_dir / f"{stem}_mean.csv", index=False)
        plot_archive(archive, paths.fig_dir)
    table = pd.DataFrame(summary)
    table.to_csv(result_dir / "distance_ablation_summary.csv", index=False)
    metadata = dict(
        benchmark=args.benchmark, functions=functions, function_list="main.ABLATION_FUNCTIONS",
        dims=args.dims, runs=args.runs, epochs=args.epochs, pop_size=args.pop_size,
        seeds=seeds.tolist(), compute_device=args.compute_device,
        fitness_convention="raw CEC fitness including bias (same as main convergence plots)",
        error_convention="fitness - benchmark.f_global; no clipping",
        standard_deviation="population (ddof=0), matching existing project summaries",
        rank="ascending mean final error within each function; average rank for ties",
        fe_convention="initial population + evaluated trial populations + scalar MEALPY probe if applicable",
        minkowski_p=framework.DISTANCE_ABLATION_MINKOWSKI_P,
        variants={metric: framework.optimizer_scientific_parameters("MaCRO-DE", variant)
                  for metric, (_, _, variant) in zip(
                      framework.DISTANCE_ABLATION_METRICS,
                      framework.optimizer_experiment_configurations(args))},
    )
    (result_dir / "distance_ablation_metadata.json").write_text(json.dumps(metadata, indent=2)+"\n")
    print(table.to_string(index=False))
    print(f"Distance-ablation results: {paths.res_dir}\nFigures: {paths.fig_dir}")


def regenerate_from_checkpoints(args):
    """Read only this mode's compatible caches; fail before any optimization."""
    import main as framework

    args.resolved_gpu_batch_size = 1
    args.estimated_gpu_batch_capacity = 1
    paths = framework.make_paths(args, create=False)
    source_paths = None
    if args.reuse_cache_from_exp_id is not None:
        source_args = argparse.Namespace(**vars(args))
        source_args.exp_id = args.reuse_cache_from_exp_id
        source_paths = framework.make_paths(source_args, create=False)
    args.function_map = framework.discover_benchmark_functions(args.benchmark, args.dims)
    functions = framework.select_experiment_functions(args, args.function_map)
    matches = framework.audit_checkpoint_cache(args, paths, source_paths, functions)
    missing = [key for key, path in matches.items() if path is None]
    if missing:
        raise RuntimeError(f"Distance cache-only: {len(missing)} missing/incompatible runs; no optimization performed. First: {missing[0]}")
    results = {}
    for function in functions:
        results[function] = {}
        for label, optimizer, variant in framework.optimizer_experiment_configurations(args):
            signature = framework.build_cache_signature(variant, optimizer)
            outputs = []
            for run in range(args.runs):
                metadata = framework.checkpoint_metadata(
                    variant, signature, function, optimizer, run, args.seed_base+run
                )
                output = framework.load_run_checkpoint(matches[function, label, run], metadata)
                if output is None:
                    raise RuntimeError("Distance cache changed during read; no optimization performed")
                outputs.append(output)
            results[function][label] = {
                "curves_runs": np.stack([o["curve"] for o in outputs]),
                "fitness_runs": np.asarray([o["best_fitness"] for o in outputs]),
            }
    export_distance_ablation(results, functions, args, paths)
