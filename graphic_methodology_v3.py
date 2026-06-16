# ============================================================
# graphic_methodology_v3.py
# ============================================================

import importlib
import inspect
import logging
import multiprocessing
import os
import re
import warnings
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from queue import Empty

import matplotlib.pyplot as plt
import numpy as np
from mealpy import FloatVar
from scipy.linalg import cholesky
from scipy.stats import chi2

from macro_de_optimizer_v2 import MaCRO_DE

warnings.filterwarnings("ignore", category=UserWarning)
logging.disable(logging.INFO)

# ============================================================
# CONFIGURATION
# Edit these values directly for normal runs.
# Environment variables with the GM_ prefix can still override
# them for quick smoke tests without editing the file.
# ============================================================

DEFAULT_EXPERIMENT_ID = "EXP005"
DEFAULT_OUTPUT_FOLDER = "Methodology_Graphics"
DEFAULT_BENCHMARK = "CEC2017"
DEFAULT_DIMENSIONS = 30
DEFAULT_POP_SIZE = 50
DEFAULT_EPOCHS = 5000
DEFAULT_RUNS = 2
DEFAULT_PROGRESS_INTERVAL = 25

EXPERIMENT_ID = os.getenv("GM_EXP_ID", DEFAULT_EXPERIMENT_ID)
OUTPUT_FOLDER = os.getenv("GM_OUTPUT_FOLDER", DEFAULT_OUTPUT_FOLDER)
BENCHMARK = os.getenv("GM_BENCHMARK", DEFAULT_BENCHMARK)
DIMENSIONS = int(os.getenv("GM_DIMENSIONS", str(DEFAULT_DIMENSIONS)))
POP_SIZE = int(os.getenv("GM_POP_SIZE", str(DEFAULT_POP_SIZE)))
EPOCHS = int(os.getenv("GM_EPOCHS", str(DEFAULT_EPOCHS)))
RUNS = int(os.getenv("GM_RUNS", str(DEFAULT_RUNS)))
RANDOM_SEED = 42
BETA_MIN = 0.2
BETA_MAX = 0.8
PCR = 0.2
MAHALANOBIS_Q = 0.68
PLOT_MAHALANOBIS = True
SAVE_CONVERGENCE = True
PARALLEL = os.getenv("GM_PARALLEL", "yes").lower() in ("1", "true", "yes", "y")
N_WORKERS = int(os.getenv(
    "GM_N_WORKERS",
    str(max(1, min((os.cpu_count() or 1) - 1, RUNS))),
))
PROGRESS_INTERVAL = int(os.getenv("GM_PROGRESS_INTERVAL", str(DEFAULT_PROGRESS_INTERVAL)))
SELECTED_FUNCTIONS = [
    item.strip()
    for item in os.getenv("GM_FUNCTIONS", "ALL").split(",")
    if item.strip()
]

np.random.seed(RANDOM_SEED)

BASE_OUTPUT = os.path.join(OUTPUT_FOLDER, EXPERIMENT_ID)
CONVERGENCE_DIR = os.path.join(BASE_OUTPUT, "Convergence")
MAHALANOBIS_DIR = os.path.join(BASE_OUTPUT, "Mahalanobis")

os.makedirs(CONVERGENCE_DIR, exist_ok=True)
os.makedirs(MAHALANOBIS_DIR, exist_ok=True)

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
}


def discover_functions(benchmark_name):
    module = importlib.import_module(AVAILABLE_BENCHMARKS[benchmark_name])
    function_map = {}

    for name, obj in inspect.getmembers(module):
        if inspect.isclass(obj) and re.match(r"^F\d+", name):
            function_map[name] = obj

    return dict(sorted(function_map.items()))


def create_function(function_class):
    try:
        return function_class(ndim=DIMENSIONS)
    except Exception:
        return None


def mahalanobis_distance(points):
    mean = np.mean(points, axis=0)
    covariance = np.cov(points.T)
    covariance += np.eye(covariance.shape[0]) * 1e-8
    inv_cov = np.linalg.inv(covariance)
    centered = points - mean
    distances = np.array([np.sqrt(point.T @ inv_cov @ point) for point in centered])
    return distances, mean, covariance


def generate_ellipsoid(mean, covariance, q=0.68, resolution=40):
    ndim = covariance.shape[0]
    radius = np.sqrt(chi2.ppf(q, ndim))
    matrix = cholesky(covariance, lower=True)
    u = np.linspace(0, 2 * np.pi, resolution)
    v = np.linspace(0, np.pi, resolution)
    x = np.outer(np.cos(u), np.sin(v))
    y = np.outer(np.sin(u), np.sin(v))
    z = np.outer(np.ones_like(u), np.cos(v))
    sphere = np.stack([x, y, z], axis=-1)
    ellipsoid = np.zeros_like(sphere)

    for i in range(resolution):
        for j in range(resolution):
            ellipsoid[i, j] = mean + radius * (matrix @ sphere[i, j])

    return ellipsoid


def plot_mahalanobis(function_name, population, best_solution=None):
    if not PLOT_MAHALANOBIS or population.shape[1] < 3:
        return None

    population_3d = population[:, :3]
    distances, mean, covariance = mahalanobis_distance(population_3d)
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")
    scatter = ax.scatter(
        population_3d[:, 0],
        population_3d[:, 1],
        population_3d[:, 2],
        c=distances,
        cmap="viridis",
        vmin=float(np.min(distances)),
        vmax=float(np.max(distances)),
        s=60,
        label="Population",
    )
    cbar = plt.colorbar(scatter)
    cbar.set_label("Mahalanobis distance: close to far")
    min_distance = float(np.min(distances))
    max_distance = float(np.max(distances))
    tick_values = np.linspace(min_distance, max_distance, 6)
    tick_labels = [f"{tick:.2f}" for tick in tick_values]
    if tick_labels:
        tick_labels[0] = f"Closest {tick_labels[0]}"
        tick_labels[-1] = f"Farthest {tick_labels[-1]}"
        cbar.set_ticks(tick_values)
        cbar.set_ticklabels(tick_labels)

    ellipsoid = generate_ellipsoid(mean, covariance, q=MAHALANOBIS_Q)
    ax.plot_surface(
        ellipsoid[:, :, 0],
        ellipsoid[:, :, 1],
        ellipsoid[:, :, 2],
        color="mediumpurple",
        alpha=0.18,
        linewidth=0,
    )
    ax.scatter(
        mean[0],
        mean[1],
        mean[2],
        c="red",
        s=120,
        marker="o",
        label="Population mean",
    )

    if best_solution is not None:
        best_solution_3d = np.asarray(best_solution, dtype=float)[:3]
        ax.scatter(
            best_solution_3d[0],
            best_solution_3d[1],
            best_solution_3d[2],
            c="red",
            edgecolors="red",
            linewidths=0.8,
            s=190,
            marker="*",
            label="x_best",
        )
    ax.set_title(f"{function_name} - Iteration {EPOCHS}")
    ax.set_xlabel("X1")
    ax.set_ylabel("X2")
    ax.set_zlabel("X3")
    ax.legend()
    plt.tight_layout()
    out_path = os.path.join(MAHALANOBIS_DIR, f"{function_name}_{EPOCHS}.png")
    plt.savefig(out_path, dpi=300)
    plt.close()
    return out_path


def plot_convergence(function_name, mean_curve):
    if not SAVE_CONVERGENCE:
        return None

    plt.figure(figsize=(9, 5))
    plt.plot(mean_curve, linewidth=2.5)
    plt.xlabel("Iteration")
    plt.ylabel("Fitness")
    plt.title(f"{function_name} Mean Convergence")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    out_path = os.path.join(CONVERGENCE_DIR, f"{function_name}_{EPOCHS}.png")
    plt.savefig(out_path, dpi=300)
    plt.close()
    return out_path


def run_single(function_name, function_class, run, progress_queue=None):
    logging.disable(logging.INFO)
    np.random.seed(RANDOM_SEED + run)
    benchmark = create_function(function_class)

    if benchmark is None:
        raise ValueError(f"Could not create benchmark {function_name}")

    problem = {
        "bounds": FloatVar(lb=benchmark.lb, ub=benchmark.ub, name="x"),
        "minmax": "min",
        "obj_func": benchmark.evaluate,
    }
    optimizer = MaCRO_DE(
        epoch=EPOCHS,
        pop_size=POP_SIZE,
        beta_min=BETA_MIN,
        beta_max=BETA_MAX,
        pcr=PCR,
        mahalanobis_q=MAHALANOBIS_Q,
        progress_queue=progress_queue,
        progress_interval=PROGRESS_INTERVAL,
        progress_context={
            "function_name": function_name,
            "run": run,
        },
    )
    g_best = optimizer.solve(problem)
    curve = np.array(optimizer.history.list_global_best_fit)
    last_population = None

    if len(optimizer.population_history) > 0:
        last_population = optimizer.population_history[-1]

    return {
        "run": run,
        "best_fitness": float(g_best.target.fitness),
        "best_solution": np.array(g_best.solution, dtype=float),
        "curve": curve,
        "last_population": last_population,
    }


def run_parallel_task(task):
    return run_single(
        task["function_name"],
        task["function_class"],
        task["run"],
        task.get("progress_queue"),
    )


def drain_progress(progress_queue):
    while True:
        try:
            message = progress_queue.get_nowait()
        except Empty:
            break

        print(
            f"[{message['function_name']}] "
            f"Run {message['run'] + 1:02d}/{RUNS} | "
            f"Epoch {message['epoch']:04d}/{message['total_epochs']} | "
            f"Best = {message['best_fitness']:.6e}",
            flush=True,
        )


def run_function(function_name, function_class):
    print("=" * 60)
    print(f"FUNCTION: {function_name}")
    print("=" * 60)

    if PARALLEL and RUNS > 1:
        completed = []
        manager = multiprocessing.Manager()
        progress_queue = manager.Queue()
        tasks = [
            {
                "function_name": function_name,
                "function_class": function_class,
                "run": run,
                "progress_queue": progress_queue,
            }
            for run in range(RUNS)
        ]

        with ProcessPoolExecutor(max_workers=N_WORKERS) as executor:
            futures = {
                executor.submit(run_parallel_task, task): task
                for task in tasks
            }

            while futures:
                done, _ = wait(
                    futures,
                    timeout=1,
                    return_when=FIRST_COMPLETED,
                )
                drain_progress(progress_queue)

                for future in done:
                    task = futures.pop(future)
                    try:
                        result = future.result()
                    except Exception as exc:
                        print(
                            f"[ERROR] {function_name} | "
                            f"Run {task['run'] + 1}/{RUNS}: {exc}",
                            flush=True,
                        )
                        raise

                    completed.append(result)
                    print(
                        f"Run {result['run'] + 1}/{RUNS} | "
                        f"Best Fitness = {result['best_fitness']:.6e}",
                        flush=True,
                    )

            drain_progress(progress_queue)

        manager.shutdown()
    else:
        completed = []

        for run in range(RUNS):
            result = run_single(function_name, function_class, run)
            completed.append(result)
            print(
                f"Run {result['run'] + 1}/{RUNS} | "
                f"Best Fitness = {result['best_fitness']:.6e}",
                flush=True,
            )

    completed = sorted(completed, key=lambda item: item["run"])
    curves = [item["curve"] for item in completed]
    best_result = min(completed, key=lambda item: item["best_fitness"])
    last_population = best_result["last_population"]
    best_solution = best_result["best_solution"]
    mean_curve = np.mean(np.array(curves), axis=0)
    convergence_path = plot_convergence(function_name, mean_curve)
    if convergence_path is not None:
        print(f"Saved convergence: {convergence_path}", flush=True)

    if last_population is not None:
        mean_3d = np.mean(last_population[:, :3], axis=0)
        xbest_3d = best_solution[:3]
        print(
            f"Mean vs x_best distance (first 3 dimensions): "
            f"{np.linalg.norm(mean_3d - xbest_3d):.6e}",
            flush=True,
        )
        mahalanobis_path = plot_mahalanobis(function_name, last_population, best_solution)
        if mahalanobis_path is not None:
            print(f"Saved Mahalanobis: {mahalanobis_path}", flush=True)


def main():
    print("=" * 60)
    print("GRAPHIC METHODOLOGY")
    print("=" * 60)
    print(f"Benchmark   : {BENCHMARK}")
    print(f"Dimensions  : {DIMENSIONS}")
    print(f"Population  : {POP_SIZE}")
    print(f"Epochs      : {EPOCHS}")
    print(f"Runs        : {RUNS}")
    print(f"Parallel    : {PARALLEL}")
    print(f"Workers     : {N_WORKERS}")
    print(f"Progress    : every {PROGRESS_INTERVAL} epochs")
    print("=" * 60)

    function_map = discover_functions(BENCHMARK)
    valid_functions = [
        (function_name, function_class)
        for function_name, function_class in function_map.items()
        if create_function(function_class) is not None
        and (
            SELECTED_FUNCTIONS == ["ALL"]
            or function_name in SELECTED_FUNCTIONS
        )
    ]

    for function_name, function_class in valid_functions:
        run_function(function_name, function_class)

    print("=" * 60)
    print("METHODOLOGY COMPLETED")
    print("=" * 60)
    print(f"Convergence : {CONVERGENCE_DIR}")
    print(f"Mahalanobis : {MAHALANOBIS_DIR}")


if __name__ == "__main__":
    main()
