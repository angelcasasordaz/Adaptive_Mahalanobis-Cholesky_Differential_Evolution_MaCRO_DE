import argparse
import hashlib
import importlib
import inspect
import json
import logging
import os
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from functools import lru_cache
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import mealpy
from mealpy import FloatVar, get_optimizer_by_name

from dbo_optimizer import DBOOptimizer
from dsade_optimizer import DSADE
from macro_de_optimizer import MaCRO_DE

DEFAULT_EPOCHS = 6000
DEFAULT_RUNS = 30

AVAILABLE_BENCHMARKS = {

    "CEC2005": "opfunu.cec_based.cec2005",
    "CEC2008": "opfunu.cec_based.cec2008",
    "CEC2010": "opfunu.cec_based.cec2010",
    "CEC2013": "opfunu.cec_based.cec2013",
    "CEC2014": "opfunu.cec_based.cec2014",
    "CEC2015": "opfunu.cec_based.cec2015",
    "CEC2017": "opfunu.cec_based.cec2017",
    "CEC2019": "opfunu.cec_based.cec2019",
    "CEC2020": "opfunu.cec_based.cec2020",
    "CEC2021": "opfunu.cec_based.cec2021",
    "CEC2022": "opfunu.cec_based.cec2022",
    "BASIC": "opfunu.name_based",
    "CEC": "opfunu.cec_based",
}

DEFAULT_BENCHMARK = "CEC2017"
DEFAULT_OPTIMIZERS = [
    #"DSADE",
    "MaCRO-DE",
    "BRO",
    "DBO",
    "DE",
    "DMO",
    "GWO",
    "HHO",
    "MFO",
    "MGO",
    "PSO",
    "SHADE",
    "WOA",
]

CHART_PALETTE = {
    "DSADE": "#3266ad",
    "DSADE_AWAD": "#d1495b",
    "MaCRO-DE": "#3266ad",
    "DBO": "#00a6a6",
    "OriginalGWO": "#e06c00",
    "OriginalWOA": "#2a9d5c",
    "OriginalCA": "#c44569",
    "OriginalPSO": "#9b59b6",
    "OriginalDE": "#6a4c93",
    "JADE": "#2d6a4f",
    "SADE": "#f4a261",
    "OriginalSHADE": "#264653",
    "OriginalFOX": "#1b9aaa",
    "OriginalRIME": "#e76f51",
    "OriginalBRO": "#577590",
    "OriginalDMOA": "#90be6d",
    "OriginalMGO": "#f9844a",
    "OriginalHHO": "#4d4d4d",
    "OriginalGOA": "#8a5a44",
    "BRO": "#577590",
    "DE": "#6a4c93",
    "DMO": "#90be6d",
    "GWO": "#e06c00",
    "HHO": "#4d4d4d",
    "MFO": "#8a5a44",
    "MGO": "#f9844a",
    "PSO": "#9b59b6",
    "SHADE": "#264653",
    "WOA": "#2a9d5c",
}

MEALPY_OPTIMIZER_ALIASES = {
    "dmo": "DMOA",
}

@dataclass
class Paths:
    exp_tag: str
    fig_dir: str
    res_dir: str
    cache_dir: str

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="OPFUNU + MEALPY Benchmark Framework"
    )
    parser.add_argument("--exp-id", type=int, default=3, help="Numeric experiment identifier")
    parser.add_argument("--output-root", default=".", help="Root directory for Figures/Results")
    parser.add_argument("--reuse-cache", action="store_true", help="Reuse cache if available")
    parser.add_argument("--benchmark", type=str, default="CEC2017", choices=list(AVAILABLE_BENCHMARKS.keys()), help="Benchmark suite")
    parser.add_argument("--functions", nargs="+", default=["ALL"], help="Functions to execute")
    parser.add_argument("--dims", type=int, default=30, help="Problem dimensions")
    parser.add_argument("--optimizers", nargs="+", default=list(DEFAULT_OPTIMIZERS), help="List of optimizers")
    parser.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS, help="Maximum optimization iterations")
    parser.add_argument("--pop-size", type=int, default=50, help="Population size")
    parser.add_argument("--runs", type=int, default=DEFAULT_RUNS, help="Independent runs per optimizer")
    parser.add_argument("--seed-base", type=int, default=1234, help="Base random seed")
    parser.add_argument("--parallel", default="yes", choices=["yes", "no"], help="Execute runs in parallel")
    parser.add_argument("--n-workers", type=int, default=None, help="Number of parallel workers")
    parser.add_argument("--convergence-extra-scale", default="none", choices=["none", "auto", "log", "symlog", "exp"], help="Save an additional convergence plot with the selected y-axis scale or transformation")
    parser.add_argument("--dsade-beta-min", type=float, default=0.2, help="Minimum adaptive beta")
    parser.add_argument("--dsade-beta-max", type=float, default=0.8, help="Maximum adaptive beta")
    parser.add_argument("--dsade-pcr", type=float, default=0.2, help="Crossover probability")
    parser.add_argument("--dsade-mahal-q", type=float, default=0.68, help="Mahalanobis threshold")
    args = parser.parse_args()

    if args.n_workers is None:
        available_workers = max(1, (os.cpu_count() or 1) - 1)
        args.n_workers = min(available_workers, max(1, args.runs))

    return args

def make_paths(args):

    exp_tag = f"EXP{args.exp_id:03d}"
    fig_dir = os.path.join(
        args.output_root,
        "Figures",
        exp_tag,
    )

    res_dir = os.path.join(
        args.output_root,
        "Results",
        exp_tag,
    )

    cache_dir = os.path.join(
        res_dir,
        "cache",
    )

    for path in [
        fig_dir,
        res_dir,
        cache_dir,
    ]:

        os.makedirs(
            path,
            exist_ok=True,
        )

    return Paths(
        exp_tag,
        fig_dir,
        res_dir,
        cache_dir,
    )

def discover_benchmark_functions(
    benchmark_name,
    ndim,
):

    if benchmark_name not in AVAILABLE_BENCHMARKS:
        raise ValueError(
            f"Unsupported benchmark: {benchmark_name}"
        )

    module_path = AVAILABLE_BENCHMARKS[
        benchmark_name
    ]

    module = importlib.import_module(
        module_path
    )

    function_map = {}

    for name, obj in inspect.getmembers(module):

        if (
            inspect.isclass(obj)
            and name.startswith("F")
        ):

            try:

                obj(ndim=ndim)

                function_map[name] = obj

            except Exception:

                print(
                    f"[SKIPPED FUNCTION] "
                    f"{name} "
                    f"does not support "
                    f"ndim={ndim}"
                )

    function_map = dict(
        sorted(function_map.items())
    )

    return function_map

def build_optimizer(
    name,
    args,
):

    optimizer_key = normalize_optimizer_name(name)

    if optimizer_key == "dsade":
        optimizer_class = DSADE
        optimizer_kwargs = custom_optimizer_kwargs(args)
    elif optimizer_key == "macrode":
        optimizer_class = MaCRO_DE
        optimizer_kwargs = custom_optimizer_kwargs(args)
    elif optimizer_key == "dbo":
        optimizer_class = DBOOptimizer
        optimizer_kwargs = {
            "epoch": args.epochs,
            "pop_size": args.pop_size,
        }
    else:
        optimizer_class = resolve_mealpy_optimizer(name)
        optimizer_kwargs = {
            "epoch": args.epochs,
            "pop_size": args.pop_size,
        }

    return optimizer_class(**optimizer_kwargs)

def normalize_optimizer_name(name):

    return "".join(
        char.lower()
        for char in str(name)
        if char.isalnum()
    )

def display_optimizer_name(name):

    label = str(name)

    for prefix in ("Original",):
        if label.startswith(prefix):
            return label[len(prefix):]

    return label

def resolve_convergence_scale(
    curves_dict,
    requested_scale,
):

    if requested_scale == "none":
        return None

    if requested_scale in ("symlog", "exp"):
        return requested_scale

    finite_values = []

    for curve in curves_dict.values():
        curve = np.asarray(
            curve,
            dtype=float,
        )
        finite_values.extend(
            curve[np.isfinite(curve)]
        )

    if len(finite_values) == 0:
        return None

    min_value = np.min(finite_values)

    if requested_scale == "auto":
        return "log" if min_value > 0 else "symlog"

    if requested_scale == "log" and min_value <= 0:
        print(
            "[CONVERGENCE SCALE] "
            "log requires positive fitness values; "
            "using symlog for this function."
        )
        return "symlog"

    return requested_scale

def custom_optimizer_kwargs(args):

    return {
        "epoch": args.epochs,
        "pop_size": args.pop_size,
        "beta_min": args.dsade_beta_min,
        "beta_max": args.dsade_beta_max,
        "pcr": args.dsade_pcr,
        "mahalanobis_q": args.dsade_mahal_q,
    }

@lru_cache(maxsize=None)
def resolve_mealpy_optimizer(name):

    optimizer_key = normalize_optimizer_name(name)
    resolved_name = MEALPY_OPTIMIZER_ALIASES.get(
        optimizer_key,
        name,
    )
    match_keys = {
        optimizer_key,
        normalize_optimizer_name(resolved_name),
    }

    for module_name in mealpy_module_candidates(resolved_name):
        optimizer_class = find_mealpy_optimizer_in_module(
            module_name,
            match_keys,
        )
        if optimizer_class is not None:
            return optimizer_class

    for module_name, obj in inspect.getmembers(mealpy):
        if not inspect.ismodule(obj):
            continue

        optimizer_class = find_mealpy_optimizer_in_module(
            module_name,
            match_keys,
        )
        if optimizer_class is not None:
            return optimizer_class

    raise ValueError(
        f"Unknown Mealpy optimizer: {name}"
    )

def mealpy_module_candidates(name):

    raw_name = str(name).replace("-", "_")
    compact_name = "".join(
        char
        for char in raw_name
        if char.isalnum() or char == "_"
    )
    parts = [
        part
        for part in compact_name.split("_")
        if part
    ]

    candidates = [
        raw_name,
        compact_name,
    ]

    for prefix in ("Original", "Dev", "Base"):
        if compact_name.lower().startswith(prefix.lower()):
            candidates.append(
                compact_name[len(prefix):]
            )

    if parts:
        candidates.append(parts[-1])

    seen = set()
    for candidate in candidates:
        if not candidate:
            continue

        normalized = candidate.upper()
        if normalized in seen:
            continue

        seen.add(normalized)
        yield normalized

def find_mealpy_optimizer_in_module(
    module_name,
    optimizer_keys,
):

    optimizers = get_optimizer_by_name(module_name)
    exact_match = None
    original_match = None
    prefixed_match = None

    for class_name, optimizer_class in optimizers.items():
        if class_name == "Optimizer":
            continue

        if normalize_optimizer_name(class_name) in optimizer_keys:
            exact_match = optimizer_class
            continue

        for prefix in ("Original", "Dev", "Base"):
            if class_name.startswith(prefix):
                stripped_key = normalize_optimizer_name(
                    class_name[len(prefix):]
                )
                if stripped_key not in optimizer_keys:
                    continue
                if prefix == "Original":
                    original_match = optimizer_class
                elif prefixed_match is None:
                    prefixed_match = optimizer_class

    if exact_match is not None:
        return exact_match

    if original_match is not None:
        return original_match

    if prefixed_match is not None:
        return prefixed_match

    return None

def build_problem(
    function_class,
    ndim,
):

    benchmark = function_class(
        ndim=ndim
    )
    lb = benchmark.lb
    ub = benchmark.ub
    bounds = FloatVar(
        lb=lb,
        ub=ub,
        name="x",
    )

    problem = {
        "bounds": bounds,
        "minmax": "min",
        "obj_func": benchmark.evaluate,
    }

    return benchmark, problem

def build_cache_signature(args):

    payload = {
        "benchmark": args.benchmark,
        "functions": args.functions,
        "optimizers": args.optimizers,
        "dims": args.dims,
        "epochs": args.epochs,
        "pop_size": args.pop_size,
        "runs": args.runs,
    }

    return hashlib.sha1(

        json.dumps(
            payload,
            sort_keys=True,
        ).encode("utf-8")

    ).hexdigest()[:10]

def print_status(message):

    print(
        f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {message}",
        flush=True,
    )

def run_single(
    function_name,
    optimizer_name,
    args,
    seed,
    run_index=None,
    total_runs=None,
):

    logging.disable(logging.INFO)
    np.random.seed(seed)
    run_label = (
        f"run {run_index + 1}/{total_runs}"
        if run_index is not None and total_runs is not None
        else "run"
    )
    print_status(
        "START | "
        f"benchmark={args.benchmark} | "
        f"function={function_name} | "
        f"optimizer={optimizer_name} | "
        f"{run_label} | "
        f"dims={args.dims} | "
        f"epochs={args.epochs} | "
        f"pop={args.pop_size} | "
        f"seed={seed}"
    )
    function_class = args.function_map[
        function_name
    ]

    benchmark, problem = build_problem(
        function_class,
        args.dims,
    )

    optimizer = build_optimizer(
        optimizer_name,
        args,
    )

    t0 = time.time()
    result = optimizer.solve(problem)
    runtime = time.time() - t0
    convergence = np.array(
        optimizer.history.list_global_best_fit,
        dtype=float,
    )
    print_status(
        "DONE  | "
        f"benchmark={args.benchmark} | "
        f"function={function_name} | "
        f"optimizer={optimizer_name} | "
        f"{run_label} | "
        f"best={float(result.target.fitness):.6e} | "
        f"time={runtime:.2f}s"
    )

    return {

        "best_fitness": float(
            result.target.fitness
        ),

        "best_solution": result.solution,
        "runtime": runtime,
        "curve": convergence,
    }

def run_parallel_task(task):

    output = run_single(
        task["function_name"],
        task["optimizer_name"],
        task["args"],
        task["seed"],
        task["run"],
        task["total_runs"],
    )

    return task["run"], output

def plot_convergence(
    curves_dict,
    title,
    out_path,
    yscale="linear",
):

    fig, ax = plt.subplots(
        figsize=(10, 5),
        facecolor="white",
    )

    for optimizer_name, curve in curves_dict.items():

        curve = np.asarray(
            curve,
            dtype=float,
        )

        if yscale == "exp":
            plot_curve = np.where(
                np.isfinite(curve),
                np.exp(
                    np.clip(
                        curve,
                        -745.0,
                        709.0,
                    )
                ),
                np.nan,
            )
        else:
            plot_curve = curve

        ax.plot(
            plot_curve,
            linewidth=2.8 if optimizer_name == "MaCRO-DE" else 2.2,
            label=display_optimizer_name(
                optimizer_name
            ),
            color=CHART_PALETTE.get(
                optimizer_name,
                None,
            ),
        )

    if yscale in ("log", "symlog"):
        ax.set_yscale(yscale)

    ax.set_xlabel("Iteration")
    ax.set_ylabel(
        "exp(Fitness)" if yscale == "exp" else "Fitness"
    )
    ax.set_title(title)
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(
        out_path,
        dpi=600,
    )

    plt.close(fig)

def export_results(
    results_struct,
    out_path,
):

    rows = []
    for function_name, optimizer_data in results_struct.items():
        for optimizer_name, data in optimizer_data.items():
            rows.append({
                "Function": function_name,
                "Optimizer": optimizer_name,
                "Best": np.min(
                    data["fitness_runs"]
                ),

                "Mean": np.mean(
                    data["fitness_runs"]
                ),

                "Std": np.std(
                    data["fitness_runs"]
                ),

                "RuntimeMean": np.mean(
                    data["runtime_runs"]
                ),
            })

    df = pd.DataFrame(rows)

    df.to_excel(
        out_path,
        index=False,
    )

    return df

def main():

    args = parse_args()
    logging.disable(logging.INFO)
    paths = make_paths(args)
    cache_signature = build_cache_signature(args)
    function_map = discover_benchmark_functions(
        args.benchmark,
        args.dims,
    )

    args.function_map = function_map

    if args.functions == ["ALL"]:

        selected_functions = list(
            function_map.keys()
        )

    else:

        selected_functions = args.functions

    print("=" * 60)
    print(
        "OPFUNU + MEALPY BENCHMARK FRAMEWORK"
    )
    print("=" * 60)
    print(
        f"Experiment     : {paths.exp_tag}"
    )
    print(
        f"Benchmark      : {args.benchmark}"
    )
    print(
        f"Functions      : {selected_functions}"
    )
    print(
        f"Optimizers     : {args.optimizers}"
    )
    print(
        f"Dimensions     : {args.dims}"
    )
    print(
        f"Epochs         : {args.epochs}"
    )
    print(
        f"Population     : {args.pop_size}"
    )
    print(
        f"Runs           : {args.runs}"
    )
    print(
        f"Parallel       : {args.parallel}"
    )
    print(
        f"Workers        : {args.n_workers}"
    )
    print(
        f"Extra scale    : {args.convergence_extra_scale}"
    )

    results_struct = {}

    for function_index, function_name in enumerate(
        selected_functions,
        start=1,
    ):
        print("\n" + "=" * 60)
        print(
            f"FUNCTION {function_index}/{len(selected_functions)}: "
            f"{function_name}",
            flush=True,
        )
        print("=" * 60)
        results_struct[function_name] = {}
        curves_plot = {}
        for optimizer_index, optimizer_name in enumerate(
            args.optimizers,
            start=1,
        ):

            try:
                print_status(
                    f"OPTIMIZER {optimizer_index}/{len(args.optimizers)} | "
                    f"function={function_name} | "
                    f"optimizer={optimizer_name}"
                )
                fitness_runs = []
                runtime_runs = []
                curves = []
                pending_runs = list(
                    range(args.runs)
                )

                if (
                    args.parallel == "yes"
                    and len(pending_runs) > 1
                ):

                    tasks = []

                    for run in pending_runs:

                        tasks.append({
                            "run": run,
                            "function_name": function_name,
                            "optimizer_name": optimizer_name,
                            "args": args,
                            "seed": args.seed_base + run,
                            "total_runs": args.runs,
                        })

                    completed = []
                    print_status(
                        f"SUBMITTED | function={function_name} | "
                        f"optimizer={optimizer_name} | "
                        f"runs={len(tasks)} | workers={args.n_workers}"
                    )

                    with ProcessPoolExecutor(
                        max_workers=args.n_workers
                    ) as executor:

                        futures = [

                            executor.submit(
                                run_parallel_task,
                                task,
                            )
                            for task in tasks
                        ]

                        for future in as_completed(futures):

                            completed.append(
                                future.result()
                            )
                            print_status(
                                f"PROGRESS | function={function_name} | "
                                f"optimizer={optimizer_name} | "
                                f"completed_runs={len(completed)}/{len(tasks)}"
                            )

                    completed = sorted(
                        completed,
                        key=lambda x: x[0],
                    )

                else:
                    completed = []
                    for run in pending_runs:
                        output = run_single(
                            function_name,
                            optimizer_name,
                            args,
                            args.seed_base + run,
                            run,
                            args.runs,
                        )

                        completed.append(
                            (run, output)
                        )

                for run, output in completed:

                    fitness_runs.append(
                        output["best_fitness"]
                    )

                    runtime_runs.append(
                        output["runtime"]
                    )

                    curves.append(
                        output["curve"]
                    )

                    print(

                        f"Run {run+1:02d} | "
                        f"Best = "
                        f"{fitness_runs[-1]:.6e} | "
                        f"Time = "
                        f"{runtime_runs[-1]:.2f}s"
                    )

                mean_curve = np.mean(

                    np.array(curves),

                    axis=0,
                )

                curves_plot[
                    optimizer_name
                ] = mean_curve

                results_struct[
                    function_name
                ][optimizer_name] = {

                    "fitness_runs": np.array(
                        fitness_runs
                    ),

                    "runtime_runs": np.array(
                        runtime_runs
                    ),

                    "curve": mean_curve,
                }

                print("-" * 50)

                print(
                    f"Mean : "
                    f"{np.mean(fitness_runs):.6e}"
                )

                print(
                    f"Std  : "
                    f"{np.std(fitness_runs):.6e}"
                )

                print(
                    f"Best : "
                    f"{np.min(fitness_runs):.6e}"
                )

                print("-" * 50)

            except Exception as e:

                print(
                    f"[SKIPPED OPTIMIZER] "
                    f"{optimizer_name} "
                    f"on {function_name}"
                )
                print(f"Reason: {e}")
                continue


        if len(curves_plot) > 0:
            plot_path = os.path.join(
                paths.fig_dir,
                f"{paths.exp_tag}_{function_name}_convergence.png",
            )

            plot_convergence(
                curves_plot,
                f"Convergence Curve - {function_name}",
                plot_path,
            )

            extra_scale = resolve_convergence_scale(
                curves_plot,
                args.convergence_extra_scale,
            )

            if extra_scale is not None:
                extra_plot_path = os.path.join(
                    paths.fig_dir,
                    (
                        f"{paths.exp_tag}_{function_name}"
                        f"_convergence_{extra_scale}.png"
                    ),
                )

                scale_title = {
                    "log": "Log Scale",
                    "symlog": "Symmetric Log Scale",
                    "exp": "Exponential Transform",
                }[extra_scale]

                plot_convergence(
                    curves_plot,
                    (
                        f"Convergence Curve - "
                        f"{function_name} ({scale_title})"
                    ),
                    extra_plot_path,
                    yscale=extra_scale,
                )

    excel_path = os.path.join(
        paths.res_dir,
        f"Global_Results_{paths.exp_tag}.xlsx",
    )

    export_results(
        results_struct,
        excel_path,
    )

    print("\n" + "=" * 60)
    print("COMPLETED")
    print("=" * 60)
    print(f"Figures: {paths.fig_dir}")
    print(f"Results: {paths.res_dir}")

if __name__ == "__main__":

    main()
