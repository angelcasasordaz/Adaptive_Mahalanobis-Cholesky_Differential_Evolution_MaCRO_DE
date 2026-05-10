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
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from mealpy import FloatVar
from mealpy.swarm_based.PSO import OriginalPSO
from mealpy.swarm_based.GWO import OriginalGWO
from mealpy.swarm_based.WOA import OriginalWOA

from mealpy.evolutionary_based.DE import (
    OriginalDE,
    JADE,
    SADE,
)

from dsade_optimizer import DSADE

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
    "DSADE",
    "OriginalPSO",
    "OriginalGWO",
    "OriginalWOA",
    "OriginalDE",
    "JADE",
    "SADE",
]

CHART_PALETTE = {

    "DSADE": "#3266ad",
    "OriginalPSO": "#9b59b6",
    "OriginalGWO": "#e06c00",
    "OriginalWOA": "#2a9d5c",
    "OriginalDE": "#6a4c93",
    "JADE": "#2d6a4f",
    "SADE": "#f4a261",
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
    parser.add_argument("--exp-id", type=int, default=1, help="Numeric experiment identifier")
    parser.add_argument("--output-root", default=".", help="Root directory for Figures/Results")
    parser.add_argument("--reuse-cache", action="store_true", help="Reuse cache if available")
    parser.add_argument("--benchmark", type=str, default="CEC2017", choices=list(AVAILABLE_BENCHMARKS.keys()), help="Benchmark suite")
    parser.add_argument("--functions", nargs="+", default=["ALL"], help="Functions to execute")
    parser.add_argument("--dims", type=int, default=30, help="Problem dimensions")
    parser.add_argument("--optimizers", nargs="+", default=list(DEFAULT_OPTIMIZERS), help="List of optimizers")
    parser.add_argument("--epochs", type=int, default=20, help="Maximum optimization iterations")
    parser.add_argument("--pop-size", type=int, default=50, help="Population size")
    parser.add_argument("--runs", type=int, default=2, help="Independent runs per optimizer")
    parser.add_argument("--seed-base", type=int, default=1234, help="Base random seed")
    parser.add_argument("--parallel", default="yes", choices=["yes", "no"], help="Execute runs in parallel")
    parser.add_argument("--n-workers", type=int, default=4, help="Number of parallel workers")
    parser.add_argument("--dsade-beta-min", type=float, default=0.2, help="Minimum adaptive beta")
    parser.add_argument("--dsade-beta-max", type=float, default=0.8, help="Maximum adaptive beta")
    parser.add_argument("--dsade-pcr", type=float, default=0.2, help="Crossover probability")
    parser.add_argument("--dsade-mahal-q", type=float, default=0.68, help="Mahalanobis threshold")
    return parser.parse_args()

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

    if name.upper() == "DSADE":
        return DSADE(
            epoch=args.epochs,
            pop_size=args.pop_size,
            beta_min=args.dsade_beta_min,
            beta_max=args.dsade_beta_max,
            pcr=args.dsade_pcr,
            mahalanobis_q=args.dsade_mahal_q,
        )

    if name.upper() == "ORIGINALPSO":

        return OriginalPSO(
            epoch=args.epochs,
            pop_size=args.pop_size,
        )

    if name.upper() == "ORIGINALGWO":

        return OriginalGWO(
            epoch=args.epochs,
            pop_size=args.pop_size,
        )

    if name.upper() == "ORIGINALWOA":

        return OriginalWOA(
            epoch=args.epochs,
            pop_size=args.pop_size,
        )

    if name.upper() == "ORIGINALDE":

        return OriginalDE(
            epoch=args.epochs,
            pop_size=args.pop_size,
        )

    if name.upper() == "JADE":

        return JADE(
            epoch=args.epochs,
            pop_size=args.pop_size,
        )

    if name.upper() == "SADE":

        return SADE(
            epoch=args.epochs,
            pop_size=args.pop_size,
        )

    raise ValueError(
        f"Unsupported optimizer: {name}"
    )

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

def run_single(
    function_name,
    optimizer_name,
    args,
    seed,
):

    np.random.seed(seed)
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
    )

    return task["run"], output

def plot_convergence(
    curves_dict,
    title,
    out_path,
):

    plt.figure(
        figsize=(10, 5),
        facecolor="white",
    )

    for optimizer_name, curve in curves_dict.items():

        plt.plot(
            curve,
            linewidth=2.2,
            label=optimizer_name,
            color=CHART_PALETTE.get(
                optimizer_name,
                None,
            ),
        )

    plt.xlabel("Iteration")
    plt.ylabel("Fitness")
    plt.title(title)
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(
        out_path,
        dpi=600,
    )

    plt.close()

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

    results_struct = {}

    for function_name in selected_functions:
        print("\n" + "=" * 60)
        print(
            f"FUNCTION: {function_name}"
        )
        print("=" * 60)
        results_struct[function_name] = {}
        curves_plot = {}
        for optimizer_name in args.optimizers:

            try:
                print(
                    f"\nOptimizer: {optimizer_name}"
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
                        })

                    completed = []

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