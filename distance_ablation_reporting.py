"""Independent distance-ablation exports and optimization-free regeneration."""
import argparse
import json
from pathlib import Path

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
    return plot_distance_convergence(function, dims, means, evaluations, labels, figure_dir)


def plot_distance_convergence(function, dims, means, evaluations, labels, figure_dir):
    """Use the FULL convergence renderer with the saved scientific FE coordinates."""
    import main as framework

    curves = {str(label): curve for label, curve in zip(labels, means)}
    figure_dir = Path(figure_dir)
    figure_dir.mkdir(parents=True, exist_ok=True)
    out_path = figure_dir / f"{function}_D{dims}_distance_convergence.png"
    framework.plot_convergence(
        curves, f"Convergence Curve - {function} (Log Scale)", out_path,
        framework.build_optimizer_colors(labels), yscale="log",
        show_markers=framework.CONVERGENCE_SHOW_MARKERS,
        use_line_styles=framework.CONVERGENCE_USE_LINE_STYLES,
        x_values=evaluations, x_label="Cumulative Function Evaluations",
        # Selection emphasis is fixed across functions, independent of fitness rank.
        highlighted_optimizer=DISTANCE_LABELS["mahalanobis_cholesky"],
    )
    print(f"DISTANCE FIGURE | {out_path} | PNG | {framework.FIGURE_EXPORT_DPI} DPI")
    return out_path


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
    export_saved_statistical_results(paths, functions, args.dims)
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


def export_saved_statistical_results(paths, functions, dims, *, final_errors=None):
    """Render saved error statistics without rewriting their scientific sources."""
    result_dir = Path(paths.res_dir)
    metrics = ["chebyshev", "euclidean", "mahalanobis_cholesky", "manhattan", "minkowski"]
    labels = [DISTANCE_LABELS[metric].removeprefix("MaCRO-DE/") for metric in metrics]
    fields = {"Best": "best", "Worst": "worst", "Median": "median", "SD": "standard_deviation"}
    summary_path = result_dir / "distance_ablation_summary.csv"
    saved = {}
    if summary_path.is_file():
        summary = pd.read_csv(summary_path, float_precision="round_trip")
        summary = summary.loc[summary["dimension"] == dims].set_index(["function", "metric"])
        if not summary.index.is_unique:
            raise ValueError(f"Duplicate function/metric rows in {summary_path}")
        for function in functions:
            saved[function] = {}
            for metric, label in zip(metrics, labels):
                row = summary.loc[(function, metric)]
                saved[function][label] = {stat: float(row[field]) for stat, field in fields.items()}
    else:
        # Archives retain final errors; match the summary's population SD (ddof=0).
        for function in functions:
            archive = result_dir / f"{function}_D{dims}_convergence.npz"
            if archive.is_file():
                with np.load(archive, allow_pickle=False) as data:
                    if str(data["function"]) != function or int(data["dims"]) != dims:
                        raise ValueError(f"Invalid distance convergence archive: {archive}")
                    archive_metrics = data["metrics"].tolist()
                    errors = {metric: data["final_error"][archive_metrics.index(metric)]
                              for metric in metrics}
            elif final_errors is not None:
                errors = final_errors[function]
            else:
                raise FileNotFoundError(f"Missing saved distance statistics: {archive}")
            saved[function] = {}
            for metric, label in zip(metrics, labels):
                values = errors[metric]
                saved[function][label] = {
                    "Best": float(values.min()), "Worst": float(values.max()),
                    "Median": float(np.median(values)), "SD": float(values.std()),
                }
    if not all(np.isfinite(value) for entries in saved.values()
               for stats in entries.values() for value in stats.values()):
        raise ValueError("Non-finite saved distance statistics")
    out_path = result_dir / f"Statistical_Results_{paths.exp_tag}_Distance_Ablation.xlsx"
    index = pd.MultiIndex.from_product(
        [functions, list(fields)], names=["Function", "Statistic"],
    )
    table = pd.DataFrame(
        [[saved[function][label][statistic] for label in labels]
         for function, statistic in index],
        index=index, columns=labels, dtype=float,
    )
    with pd.ExcelWriter(out_path) as writer:
        table.to_excel(writer, sheet_name="Fitness", merge_cells=True)
    print(f"Distance statistical results: {out_path}")
    return out_path


def regenerate_from_checkpoints(args):
    """Read saved data and regenerate PNGs and the Statistical workbook; never optimize."""
    import main as framework

    args.resolved_gpu_batch_size = 1
    args.estimated_gpu_batch_capacity = 1
    paths = framework.make_paths(args, create=False)
    functions = list(framework.ABLATION_FUNCTIONS)
    archives = [Path(paths.res_dir) / f"{name}_D{args.dims}_convergence.npz"
                for name in functions]
    if all(archive.is_file() for archive in archives):
        # Saved means and FE coordinates are authoritative for a style-only redraw.
        # Preflight all eight before replacing any figure; no objective discovery.
        expected_labels = [DISTANCE_LABELS[m] for m in framework.DISTANCE_ABLATION_METRICS]
        for function, archive in zip(functions, archives):
            with np.load(archive, allow_pickle=False) as data:
                if (str(data["function"]) != function or int(data["dims"]) != args.dims
                        or data["labels"].tolist() != expected_labels
                        or data["mean_fitness"].shape != (5, data["function_evaluations"].size)):
                    raise ValueError(f"Invalid distance convergence archive: {archive}")
        export_saved_statistical_results(paths, functions, args.dims)
        for archive in archives:
            plot_archive(archive, paths.fig_dir)
        print(f"DISTANCE REPORTS COMPLETE | {len(archives)} PNGs + Statistical workbook | optimization runs=0")
        return
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
    # Checkpoint fallback remains read-only: do not rewrite scientific artifacts.
    scalar_probe = int(args.compute_device == "cpu" and not framework.cpu_batching_enabled(args))
    evaluations = (np.arange(1, args.epochs + 1) + 1) * args.pop_size + scalar_probe
    labels = [DISTANCE_LABELS[m] for m in framework.DISTANCE_ABLATION_METRICS]
    final_errors = None
    if not (Path(paths.res_dir) / "distance_ablation_summary.csv").is_file():
        final_errors = {}
        for function in functions:
            bias = float(args.function_map[function](ndim=args.dims).f_global)
            final_errors[function] = {
                metric: results[function][label]["fitness_runs"] - bias
                for metric, label in zip(framework.DISTANCE_ABLATION_METRICS, labels)
            }
    export_saved_statistical_results(paths, functions, args.dims, final_errors=final_errors)
    for function in functions:
        means = np.stack([results[function][label]["curves_runs"].mean(axis=0)
                          for label in labels])
        plot_distance_convergence(function, args.dims, means, evaluations, labels, paths.fig_dir)
    print(f"DISTANCE REPORTS COMPLETE | {len(functions)} PNGs + Statistical workbook | optimization runs=0")
