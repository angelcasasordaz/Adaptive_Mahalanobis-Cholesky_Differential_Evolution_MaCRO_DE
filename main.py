import argparse
import hashlib
import importlib
import inspect
import json
import logging
import multiprocessing
import os
import pickle
import time
import glob
import traceback
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from dataclasses import dataclass
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from mealpy import FloatVar

from algorithm_acronym_list import (
    CUSTOM_OPTIMIZERS,
    optimizer_class as resolve_optimizer_class,
    resolve_optimizer_name,
    supports_gpu_batching,
)
from compute_backend import (
    GPU_MODES,
    estimate_population_batch_size,
    estimate_effective_batch_size,
    initialize_gpu,
    is_gpu_out_of_memory,
    normalize_compute_device,
    parse_gpu_batch_size,
    performance_batch_candidates,
    resolve_gpu_batch_size,
    resolve_cpu_workers,
    resolve_objective_workers,
    validate_memory_fraction,
)
from gpu_batching import BATCH_ENGINE_VERSION, BatchedDEEngine
from objective_evaluation import ObjectiveSpec, initialize_objective_worker

DEFAULT_EPOCHS = 2000
DEFAULT_RUNS = 30
COMPUTE_DEVICE = "hybrid"
CPU_WORKERS = None  # Auto-reserve CPU capacity for the OS/user.
GPU_WORKERS = 1
GPU_MEMORY_FRACTION = 0.85
GPU_BATCH_SIZE = "auto"
REUSE_CACHE = True
EXPERIMENT_MODE = "full"
# Options:
# "full"
# "ablation"
MACRO_BETA_MIN = 0.2
MACRO_BETA_MAX = 0.8
MACRO_PCR = 0.2
MACRO_MAHAL_Q = 0.68

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
ABLATION_OPTIMIZERS = [
    "DE",
    "DE-M",
    "DE-MC",
    "DE-MC-CF",
    "MaCRO-DE",
]
CHART_CMAP = "tab20"
LINE_STYLES = ("-", "--", "-.", ":", (0, (5, 2, 1, 2)))
MARKERS = ("o", "s", "^", "D", "X")
OVERLAP_DIAGNOSTIC_OPTIMIZERS = (
    "DE-M",
    "DE-MC",
    "DE-MC-CF",
    "MaCRO-DE",
)

# Qualitative:
# "tab10"
# "tab20"
# "Set1"
# "Set2"
# "Set3"
# "Dark2"
# "Paired"
# "Accent"

@dataclass
class Paths:
    exp_tag: str
    mode: str
    fig_dir: str
    res_dir: str
    cache_dir: str

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="OPFUNU + MEALPY Benchmark Framework"
    )
    parser.add_argument("--exp-id", type=int, default=628, help="Numeric experiment identifier")
    parser.add_argument("--output-root", default=".", help="Root directory for Figures/Results")
    cache_group = parser.add_mutually_exclusive_group()
    cache_group.add_argument(
        "--reuse-cache",
        dest="reuse_cache",
        action="store_true",
        default=REUSE_CACHE,
        help="Reuse compatible completed checkpoints (default)",
    )
    cache_group.add_argument(
        "--no-reuse-cache",
        dest="reuse_cache",
        action="store_false",
        help="Ignore completed checkpoints and rerun every requested run",
    )
    parser.add_argument("--benchmark", type=str, default="CEC2017", choices=list(AVAILABLE_BENCHMARKS.keys()), help="Benchmark suite")
    parser.add_argument("--functions", nargs="+", default=["ALL"], help="Functions to execute")
    parser.add_argument("--dims", type=int, default=30, help="Problem dimensions")
    parser.add_argument("--optimizers", nargs="+", default=None, help="List of optimizers")
    parser.add_argument("--experiment-mode", default=EXPERIMENT_MODE, choices=["full", "ablation"], help="Experiment mode")
    parser.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS, help="Maximum optimization iterations")
    parser.add_argument("--pop-size", type=int, default=50, help="Population size")
    parser.add_argument("--runs", type=int, default=DEFAULT_RUNS, help="Independent runs per optimizer")
    parser.add_argument("--seed-base", type=int, default=1234, help="Base random seed")
    parser.add_argument("--parallel", default="yes", choices=["yes", "no"], help="Execute runs in parallel")
    parser.add_argument("--compute-device", default=COMPUTE_DEVICE, choices=["cpu", "gpu", "hybrid"], help="Numerical backend for supported custom optimizers")
    parser.add_argument(
        "--n-workers",
        type=int,
        default=CPU_WORKERS,
        help="CPU process workers (default: auto; reserves about one third of logical CPUs)",
    )
    parser.add_argument("--gpu-workers", type=int, default=GPU_WORKERS, help="CUDA-owning controller processes (must be 1)")
    parser.add_argument("--gpu-memory-fraction", type=float, default=GPU_MEMORY_FRACTION, help="Maximum fraction of total VRAM available to CuPy's memory pool")
    parser.add_argument("--gpu-batch-size", default=GPU_BATCH_SIZE, help="GPU population batch capacity: auto or a positive integer")
    parser.add_argument("--objective-workers", type=int, default=None, help="Persistent CPU objective workers; default approximates physical cores")
    parser.add_argument("--objective-evaluation", choices=["auto", "serial", "process"], default="auto", help="CPU objective evaluation strategy for batched GPU runs")
    parser.add_argument("--cec-objective-backend", choices=["auto", "opfunu", "numpy", "gpu"], default="auto", help="CEC dispatch; numpy/gpu require strict runtime verification")
    parser.add_argument("--cec-gpu-verification-points", type=int, default=512, help="Deterministic random points per staged GPU objective verification")
    parser.add_argument("--gpu-progress-interval", type=int, default=50, help="Epoch interval for batched GPU heartbeat output")
    parser.add_argument("--gpu-auto-calibration", choices=["yes", "no"], default="yes", help="Measure candidate run-batch throughput before AUTO execution")
    parser.add_argument("--gpu-calibration-epochs", type=int, default=1, help="Tiny non-scientific epochs per AUTO calibration candidate")
    parser.add_argument("--convergence-extra-scale", default="none", choices=["none", "auto", "log", "symlog", "exp"], help="Save an additional convergence plot with the selected y-axis scale or transformation")
    parser.add_argument(
        "--overlap-diagnostic-function",
        default=None,
        help=(
            "Generate a cache-only, single-run overlap diagnostic for the selected "
            "CEC function, then exit without running experiments"
        ),
    )
    parser.add_argument(
        "--overlap-diagnostic-run",
        type=int,
        default=1,
        help="One-based cached run to use for the overlap diagnostic (default: 1)",
    )
    parser.add_argument(
        "--overlap-diagnostic-cache-signature",
        default=None,
        help="Cache signature to use when more than one compatible checkpoint set exists",
    )
    parser.add_argument("--macro-beta-min", dest="macro_beta_min", type=float, default=MACRO_BETA_MIN, help="Minimum MaCRO-DE beta")
    parser.add_argument("--macro-beta-max", dest="macro_beta_max", type=float, default=MACRO_BETA_MAX, help="Maximum MaCRO-DE beta")
    parser.add_argument("--macro-pcr", dest="macro_pcr", type=float, default=MACRO_PCR, help="MaCRO-DE crossover probability")
    parser.add_argument("--macro-mahal-q", dest="macro_mahal_q", type=float, default=MACRO_MAHAL_Q, help="MaCRO-DE Mahalanobis threshold")
    parser.add_argument("--dsade-beta-min", dest="macro_beta_min", type=float, default=argparse.SUPPRESS, help=argparse.SUPPRESS)
    parser.add_argument("--dsade-beta-max", dest="macro_beta_max", type=float, default=argparse.SUPPRESS, help=argparse.SUPPRESS)
    parser.add_argument("--dsade-pcr", dest="macro_pcr", type=float, default=argparse.SUPPRESS, help=argparse.SUPPRESS)
    parser.add_argument("--dsade-mahal-q", dest="macro_mahal_q", type=float, default=argparse.SUPPRESS, help=argparse.SUPPRESS)
    args = parser.parse_args()

    args.n_workers = resolve_cpu_workers(args.n_workers)

    args.compute_device = normalize_compute_device(args.compute_device)
    try:
        args.gpu_memory_fraction = validate_memory_fraction(args.gpu_memory_fraction)
        args.gpu_batch_size = parse_gpu_batch_size(args.gpu_batch_size)
    except ValueError as exc:
        parser.error(str(exc))
    if args.n_workers < 1:
        parser.error("--n-workers must be at least 1")
    args.objective_workers = resolve_objective_workers(args.objective_workers, args.n_workers)
    if args.gpu_progress_interval < 1:
        parser.error("--gpu-progress-interval must be at least 1")
    if args.gpu_calibration_epochs < 1 or args.gpu_calibration_epochs > 5:
        parser.error("--gpu-calibration-epochs must be between 1 and 5")
    if args.cec_gpu_verification_points < 100:
        parser.error("--cec-gpu-verification-points must be at least 100")
    if args.gpu_workers != 1:
        parser.error("--gpu-workers must be 1; this project currently targets one NVIDIA GPU")
    if args.overlap_diagnostic_run < 1:
        parser.error("--overlap-diagnostic-run must be at least 1")

    return args

def apply_experiment_mode(args):

    args.experiment_mode = str(args.experiment_mode).lower()

    if args.experiment_mode == "ablation":
        args.optimizers = list(ABLATION_OPTIMIZERS)
    elif args.optimizers is None:
        args.optimizers = list(DEFAULT_OPTIMIZERS)

def make_paths(args, create=True):

    exp_tag = f"EXP{args.exp_id:03d}"
    fig_dir = os.path.join(
        args.output_root,
        "Figures",
        exp_tag,
        args.experiment_mode,
    )

    res_dir = os.path.join(
        args.output_root,
        "Results",
        exp_tag,
        args.experiment_mode,
    )

    cache_dir = os.path.join(
        res_dir,
        "cache",
    )

    if create:
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
        exp_tag=exp_tag,
        mode=args.experiment_mode,
        fig_dir=fig_dir,
        res_dir=res_dir,
        cache_dir=cache_dir,
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

    optimizer_name = resolve_optimizer_name(name)
    optimizer_class = resolve_optimizer_class(name)
    optimizer_kwargs = optimizer_init_kwargs(
        optimizer_class,
        optimizer_name,
        args,
    )

    return optimizer_class(**optimizer_kwargs)

def display_optimizer_name(name):

    label = str(name)

    for prefix in ("Original",):
        if label.startswith(prefix):
            return label[len(prefix):]

    return label

def is_macro_de_optimizer(name):

    try:
        return resolve_optimizer_name(name) == "MaCRO-DE"
    except ValueError:
        return str(name) == "MaCRO-DE"

def build_optimizer_colors(optimizer_names):

    cmap = plt.get_cmap(CHART_CMAP)

    return {
        optimizer_name: cmap(index % cmap.N)
        for index, optimizer_name in enumerate(optimizer_names)
    }

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


def audit_convergence_curves(curves_dict, function_name):
    """Report completeness, non-finite data, and exact/nearly hidden curves."""
    diagnostics = {"overlaps": [], "invalid": []}
    normalized = {}
    for optimizer_name, values in curves_dict.items():
        curve = np.asarray(values, dtype=float).reshape(-1)
        normalized[optimizer_name] = curve
        finite_count = int(np.count_nonzero(np.isfinite(curve)))
        nan_count = int(np.count_nonzero(np.isnan(curve)))
        inf_count = int(np.count_nonzero(np.isinf(curve)))
        print_status(
            f"CURVE AUDIT | function={function_name} | optimizer={optimizer_name} | "
            f"length={len(curve)} | finite={finite_count} | nan={nan_count} | inf={inf_count}"
        )
        if len(curve) == 0 or finite_count != len(curve):
            diagnostics["invalid"].append(optimizer_name)

    names = list(normalized)
    for left_index, left_name in enumerate(names):
        for right_name in names[left_index + 1 :]:
            left, right = normalized[left_name], normalized[right_name]
            if left.shape != right.shape:
                print_status(
                    f"CURVE LENGTH MISMATCH | {function_name} | {left_name}={len(left)} | "
                    f"{right_name}={len(right)}"
                )
                continue
            finite = np.isfinite(left) & np.isfinite(right)
            if not np.any(finite):
                continue
            max_abs_difference = float(np.max(np.abs(left[finite] - right[finite])))
            identical = np.array_equal(left, right, equal_nan=True)
            nearly_overlapping = np.allclose(left, right, rtol=1.0e-10, atol=1.0e-12, equal_nan=True)
            if identical or nearly_overlapping:
                diagnostics["overlaps"].append((left_name, right_name, max_abs_difference))
                print_status(
                    f"OVERLAP | {left_name} and {right_name} | "
                    f"max_abs_diff={max_abs_difference:.6e}"
                )
    return diagnostics

def optimizer_init_kwargs(
    optimizer_class,
    optimizer_name,
    args,
):

    optimizer_kwargs = {
        "epoch": args.epochs,
        "pop_size": args.pop_size,
    }

    if optimizer_name not in CUSTOM_OPTIMIZERS:
        return optimizer_kwargs

    custom_kwargs = {
        "beta_min": args.macro_beta_min,
        "beta_max": args.macro_beta_max,
        "pcr": args.macro_pcr,
        "mahalanobis_q": args.macro_mahal_q,
        "compute_device": args.compute_device,
    }
    init_params = inspect.signature(
        optimizer_class.__init__
    ).parameters

    optimizer_kwargs.update({
        key: value
        for key, value in custom_kwargs.items()
        if key in init_params
    })

    return optimizer_kwargs

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
        "experiment_mode": args.experiment_mode,
        "benchmark": args.benchmark,
        "functions": args.functions,
        "optimizers": args.optimizers,
        "dims": args.dims,
        "epochs": args.epochs,
        "pop_size": args.pop_size,
        "runs": args.runs,
        "compute_device": args.compute_device,
        "cpu_workers": args.n_workers,
        "gpu_workers": args.gpu_workers,
        "gpu_memory_fraction": args.gpu_memory_fraction,
        "gpu_batch_size": args.gpu_batch_size,
        "resolved_gpu_batch_size": args.resolved_gpu_batch_size,
        "estimated_gpu_batch_capacity": args.estimated_gpu_batch_capacity,
        "gpu_batch_engine_version": BATCH_ENGINE_VERSION,
        "gpu_batch_policy": "performance-aware",
        "objective_workers": args.objective_workers,
        "objective_evaluation": args.objective_evaluation,
        "cec_objective_backend": args.cec_objective_backend,
        "cec_gpu_version": "cec2017-complete-v3",
        "cec_gpu_verification_points": args.cec_gpu_verification_points,
        "macro_beta_min": args.macro_beta_min,
        "macro_beta_max": args.macro_beta_max,
        "macro_pcr": args.macro_pcr,
        "macro_mahal_q": args.macro_mahal_q,
    }

    return hashlib.sha1(

        json.dumps(
            payload,
            sort_keys=True,
        ).encode("utf-8")

    ).hexdigest()[:10]

def safe_path_component(value):

    component = "".join(
        char
        if char.isalnum() or char in ("-", "_")
        else "_"
        for char in str(value)
    )

    return component or "item"

def run_checkpoint_path(
    paths,
    cache_signature,
    function_name,
    optimizer_name,
    run,
):

    optimizer_tag = (
        f"{safe_path_component(optimizer_name)}_"
        f"{hashlib.sha1(str(optimizer_name).encode('utf-8')).hexdigest()[:8]}"
    )
    checkpoint_dir = os.path.join(
        paths.cache_dir,
        cache_signature,
        safe_path_component(function_name),
        optimizer_tag,
    )

    return os.path.join(
        checkpoint_dir,
        f"run_{run + 1:03d}.pkl",
    )

def checkpoint_metadata(
    args,
    cache_signature,
    function_name,
    optimizer_name,
    run,
    seed,
):

    return {
        "cache_signature": cache_signature,
        "benchmark": args.benchmark,
        "function_name": function_name,
        "optimizer_name": optimizer_name,
        "dims": args.dims,
        "epochs": args.epochs,
        "pop_size": args.pop_size,
        "run": run,
        "seed": seed,
        "compute_device": args.compute_device,
        "cpu_workers": args.n_workers,
        "gpu_workers": args.gpu_workers,
        "gpu_memory_fraction": args.gpu_memory_fraction,
        "gpu_batch_size": args.gpu_batch_size,
        "resolved_gpu_batch_size": args.resolved_gpu_batch_size,
        "estimated_gpu_batch_capacity": args.estimated_gpu_batch_capacity,
        "gpu_batch_engine_version": BATCH_ENGINE_VERSION,
        "gpu_batch_policy": "performance-aware",
        "objective_workers": args.objective_workers,
        "objective_evaluation": args.objective_evaluation,
        "cec_objective_backend": args.cec_objective_backend,
        "cec_gpu_version": "cec2017-complete-v3",
        "cec_gpu_verification_points": args.cec_gpu_verification_points,
    }


def optimizer_uses_gpu(optimizer_name, compute_device):
    if compute_device not in GPU_MODES:
        return False
    return supports_gpu_batching(optimizer_name)

def save_run_checkpoint(
    checkpoint_path,
    metadata,
    output,
):

    os.makedirs(
        os.path.dirname(checkpoint_path),
        exist_ok=True,
    )
    tmp_path = (
        f"{checkpoint_path}.tmp."
        f"{os.getpid()}."
        f"{time.time_ns()}"
    )

    with open(tmp_path, "wb") as file:
        pickle.dump(
            {
                "metadata": metadata,
                "output": output,
            },
            file,
            protocol=pickle.HIGHEST_PROTOCOL,
        )

    os.replace(
        tmp_path,
        checkpoint_path,
    )

def load_run_checkpoint(
    checkpoint_path,
    expected_metadata,
):

    if not os.path.exists(checkpoint_path):
        return None

    try:
        with open(checkpoint_path, "rb") as file:
            payload = pickle.load(file)
    except Exception as exc:
        print_status(
            f"CACHE INVALID | path={checkpoint_path} | reason={exc}"
        )
        return None

    if payload.get("metadata") != expected_metadata:
        print_status(
            f"CACHE MISMATCH | path={checkpoint_path}"
        )
        return None

    output = payload.get("output")
    required_keys = {
        "best_fitness",
        "best_solution",
        "runtime",
        "curve",
    }
    if not isinstance(output, dict) or not required_keys.issubset(output):
        print_status(
            f"CACHE INVALID | path={checkpoint_path} | reason=missing keys"
        )
        return None

    return output


def load_compatible_cpu_checkpoint(
    paths,
    function_name,
    optimizer_name,
    run,
    expected_metadata,
):
    """Reuse scientifically identical CPU checkpoints across GPU policy versions."""
    optimizer_tag = (
        f"{safe_path_component(optimizer_name)}_"
        f"{hashlib.sha1(str(optimizer_name).encode('utf-8')).hexdigest()[:8]}"
    )
    pattern = os.path.join(
        paths.cache_dir,
        "*",
        safe_path_component(function_name),
        optimizer_tag,
        f"run_{run + 1:03d}.pkl",
    )
    scientific_keys = (
        "benchmark", "function_name", "optimizer_name", "dims",
        "epochs", "pop_size", "run", "seed",
    )
    for candidate_path in sorted(glob.glob(pattern), reverse=True):
        try:
            with open(candidate_path, "rb") as file:
                payload = pickle.load(file)
            metadata = payload.get("metadata", {})
            if any(metadata.get(key) != expected_metadata.get(key) for key in scientific_keys):
                continue
            output = payload.get("output")
            if not isinstance(output, dict) or not {
                "best_fitness", "best_solution", "runtime", "curve"
            }.issubset(output):
                continue
            print_status(f"CACHE COMPATIBLE | path={candidate_path}")
            return output
        except Exception:
            continue
    return None


def load_overlap_diagnostic_curves(args, paths):
    """Load one matching four-optimizer run without changing the cache."""
    function_name = args.overlap_diagnostic_function
    run = args.overlap_diagnostic_run - 1
    required_metadata = {
        "benchmark": args.benchmark,
        "function_name": function_name,
        "dims": args.dims,
        "epochs": args.epochs,
        "pop_size": args.pop_size,
        "run": run,
        "seed": args.seed_base + run,
    }
    requested_signature = args.overlap_diagnostic_cache_signature
    if requested_signature is None:
        signature_paths = sorted(glob.glob(os.path.join(paths.cache_dir, "*")))
    else:
        signature_paths = [os.path.join(paths.cache_dir, requested_signature)]

    compatible_sets = []
    for signature_path in signature_paths:
        if not os.path.isdir(signature_path):
            continue
        cache_signature = os.path.basename(signature_path)
        curves = {}
        valid_set = True
        for optimizer_name in OVERLAP_DIAGNOSTIC_OPTIMIZERS:
            checkpoint_path = run_checkpoint_path(
                paths,
                cache_signature,
                function_name,
                optimizer_name,
                run,
            )
            try:
                with open(checkpoint_path, "rb") as file:
                    payload = pickle.load(file)
            except (FileNotFoundError, OSError, pickle.PickleError, EOFError):
                valid_set = False
                break

            metadata = payload.get("metadata", {}) if isinstance(payload, dict) else {}
            output = payload.get("output") if isinstance(payload, dict) else None
            expected = {
                **required_metadata,
                "cache_signature": cache_signature,
                "optimizer_name": optimizer_name,
            }
            if any(metadata.get(key) != value for key, value in expected.items()):
                valid_set = False
                break
            if not isinstance(output, dict) or "curve" not in output:
                valid_set = False
                break
            curve = np.asarray(output["curve"], dtype=float).reshape(-1)
            if len(curve) != args.epochs:
                valid_set = False
                break
            curves[optimizer_name] = curve

        if valid_set:
            compatible_sets.append((cache_signature, curves))

    if not compatible_sets:
        signature_detail = (
            f" signature {requested_signature!r}"
            if requested_signature is not None
            else ""
        )
        raise FileNotFoundError(
            "No complete compatible cached checkpoint set was found for "
            f"{function_name}, run {run + 1},{signature_detail} under {paths.cache_dir}. "
            "The overlap diagnostic is cache-only; no experiments were run."
        )
    if len(compatible_sets) > 1:
        signatures = ", ".join(signature for signature, _ in compatible_sets)
        raise RuntimeError(
            "Multiple compatible cached checkpoint sets were found: "
            f"{signatures}. Select one with --overlap-diagnostic-cache-signature."
        )

    return compatible_sets[0]


def compare_overlap_diagnostic_curves(curves):
    """Return exact pairwise diagnostics using one-based convergence iterations."""
    diagnostics = []
    names = list(curves)
    for left_index, left_name in enumerate(names):
        for right_name in names[left_index + 1 :]:
            left = np.asarray(curves[left_name], dtype=float).reshape(-1)
            right = np.asarray(curves[right_name], dtype=float).reshape(-1)
            if left.shape != right.shape:
                raise ValueError(
                    f"Diagnostic curve length mismatch: {left_name}={len(left)}, "
                    f"{right_name}={len(right)}"
                )

            equal_values = (left == right) | (np.isnan(left) & np.isnan(right))
            differing_indices = np.flatnonzero(~equal_values)
            exactly_identical = differing_indices.size == 0
            first_different_iteration = (
                None if exactly_identical else int(differing_indices[0]) + 1
            )
            nonfinite_difference = (~equal_values) & ~(
                np.isfinite(left) & np.isfinite(right)
            )
            if np.any(nonfinite_difference):
                max_absolute_difference = float("inf")
            else:
                finite = np.isfinite(left) & np.isfinite(right)
                max_absolute_difference = (
                    float(np.max(np.abs(left[finite] - right[finite])))
                    if np.any(finite)
                    else 0.0
                )
            diagnostics.append({
                "left": left_name,
                "right": right_name,
                "max_absolute_difference": max_absolute_difference,
                "exactly_identical": exactly_identical,
                "first_different_iteration": first_different_iteration,
            })
    return diagnostics

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
    result = optimizer.solve(problem, seed=seed)
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


def build_batched_engine(
    function_name,
    optimizer_name,
    args,
    objective_executor=None,
    epochs=None,
):
    function_class = args.function_map[function_name]
    benchmark = function_class(ndim=args.dims)
    reference_optimizer = build_optimizer(optimizer_name, args)
    kwargs = {
        "optimizer_name": resolve_optimizer_name(optimizer_name),
        "objective": benchmark.evaluate,
        "lb": benchmark.lb,
        "ub": benchmark.ub,
        "epochs": args.epochs if epochs is None else epochs,
        "pop_size": args.pop_size,
        "compute_device": args.compute_device,
        "mahalanobis_q": getattr(reference_optimizer, "mahalanobis_q", args.macro_mahal_q),
        "objective_executor": objective_executor,
        "objective_workers": args.objective_workers,
        "objective_strategy": args.objective_evaluation,
        "objective_spec": ObjectiveSpec(
            function_class.__module__,
            function_class.__name__,
            args.dims,
        ),
        "gpu_objective": args.gpu_objectives.get(function_name),
        "vectorized_cpu_objective": args.vectorized_cpu_objectives.get(function_name),
        "cec_objective_backend": args.cec_objective_backend,
    }
    for name in ("wf", "cr", "beta_min", "beta_max", "pcr"):
        if hasattr(reference_optimizer, name):
            kwargs[name] = getattr(reference_optimizer, name)
    return BatchedDEEngine(**kwargs)


def initialize_verified_gpu_objectives(args, selected_functions):
    """Load support data once and enable only CUDA-validated CEC objectives."""
    args.gpu_objectives = {}
    args.gpu_objective_reports = {}
    args.vectorized_cpu_objectives = {}
    if args.compute_device not in GPU_MODES:
        return
    # Lazy imports are essential: Windows/CPU execution must not import CuPy or
    # the optional GPU objective implementation.
    from cec2017_gpu import (
        CEC2017_GPU_CANDIDATES,
        CEC2017GpuObjective,
        verify_gpu_objective,
    )
    from compute_backend import ComputeBackend

    backend = ComputeBackend(args.compute_device)
    for function_name in selected_functions:
        if function_name not in CEC2017_GPU_CANDIDATES:
            continue
        benchmark = args.function_map[function_name](ndim=args.dims)
        cpu_objective = CEC2017GpuObjective(function_name, benchmark, np)
        cpu_report = verify_gpu_objective(
            cpu_objective,
            benchmark,
            np.asarray,
            lambda: None,
            random_points=args.cec_gpu_verification_points,
        )
        if cpu_report.verified:
            args.vectorized_cpu_objectives[function_name] = cpu_objective
        if args.cec_objective_backend in {"opfunu", "numpy"}:
            continue
        objective = CEC2017GpuObjective(function_name, benchmark, backend.xp)
        try:
            report = verify_gpu_objective(
                objective,
                benchmark,
                backend.to_cpu,
                backend.synchronize,
                random_points=args.cec_gpu_verification_points,
            )
        except Exception as exc:
            print_status(
                f"GPU OBJECTIVE VERIFY ERROR | function={function_name} | {exc}"
            )
            continue
        args.gpu_objective_reports[function_name] = report
        print_status(
            f"GPU OBJECTIVE VERIFY | function={function_name} | "
            f"verified={str(report.verified).upper()} | points={report.points} | "
            f"max_abs={report.max_absolute_error:.6e} | "
            f"max_rel={report.max_relative_error:.6e} | mismatches={report.mismatches}"
        )
        if report.verified:
            args.gpu_objectives[function_name] = objective


def _batch_progress_callback(optimizer_name, run_indices):
    run_label = f"{run_indices[0] + 1}-{run_indices[-1] + 1}"

    def report(epoch, total_epochs, elapsed, epoch_time, timing, strategy, memory):
        eta = (elapsed / epoch) * (total_epochs - epoch) if epoch >= 2 else float("nan")
        eta_text = f"{eta:.1f}s" if np.isfinite(eta) else "warming-up"
        print_status(
            "GPU BATCH PROGRESS | "
            f"optimizer={optimizer_name} | epoch={epoch}/{total_epochs} | "
            f"runs={run_label} | elapsed={elapsed:.1f}s | epoch_time={epoch_time:.3f}s | "
            f"gpu_kernel_time={timing['gpu_kernel']:.2f}s | "
            f"fitness_time={timing['fitness']:.2f}s | transfer_time={timing['transfer']:.2f}s | "
            f"objective={strategy} | eta={eta_text} | "
            f"gpu_pool={memory['pool_total_bytes'] / 1024**2:.1f}MiB"
        )

    return report


def run_gpu_batch(
    function_name,
    optimizer_name,
    args,
    run_indices,
    objective_executor=None,
):
    """Advance independent runs together in the controller's one CUDA context."""
    engine = build_batched_engine(
        function_name,
        optimizer_name,
        args,
        objective_executor=objective_executor,
    )
    seeds = [args.seed_base + run for run in run_indices]
    for run, seed in zip(run_indices, seeds):
        print_status(
            "START | "
            f"benchmark={args.benchmark} | function={function_name} | "
            f"optimizer={optimizer_name} | run {run + 1}/{args.runs} | "
            f"dims={args.dims} | epochs={args.epochs} | pop={args.pop_size} | seed={seed}"
        )
    outputs, state, elapsed = engine.run(
        run_indices,
        seeds,
        progress_callback=_batch_progress_callback(optimizer_name, run_indices),
        progress_interval=args.gpu_progress_interval,
    )
    for run, output in zip(run_indices, outputs):
        print_status(
            "DONE  | "
            f"benchmark={args.benchmark} | function={function_name} | "
            f"optimizer={optimizer_name} | run {run + 1}/{args.runs} | "
            f"best={output['best_fitness']:.6e} | time={output['runtime']:.2f}s"
        )
    return list(zip(run_indices, outputs)), state, elapsed


def calibrate_gpu_batch_size(
    function_name,
    optimizer_name,
    args,
    maximum_batch_size,
    objective_executor,
):
    """Measure tiny copied-state runs; experimental seeds and state are untouched."""
    candidates = performance_batch_candidates(maximum_batch_size, args.runs)
    if len(candidates) == 1:
        return candidates[0], []

    dispatch_warm = build_batched_engine(
        function_name,
        optimizer_name,
        args,
        objective_executor=objective_executor,
        epochs=1,
    )
    dispatch_warm.objective_evaluator.calibrate(
        dispatch_warm.lb,
        dispatch_warm.ub,
        candidates[-1] * args.pop_size,
    )
    if dispatch_warm.objective_evaluator.strategy == "process":
        warm_rng = np.random.default_rng(7_900_001)
        warm_vectors = warm_rng.uniform(
            dispatch_warm.lb,
            dispatch_warm.ub,
            size=(max(args.objective_workers, 2), args.dims),
        )
        dispatch_warm.objective_evaluator.evaluate(warm_vectors)

    # Warm CUDA kernels and objective dispatch outside candidate timings.
    warm_engine = build_batched_engine(
        function_name,
        optimizer_name,
        args,
        objective_executor=objective_executor,
        epochs=1,
    )
    warm_engine.run([0], [8_000_001])

    measurements = []
    for candidate in candidates:
        engine = build_batched_engine(
            function_name,
            optimizer_name,
            args,
            objective_executor=objective_executor,
            epochs=args.gpu_calibration_epochs,
        )
        calibration_runs = list(range(candidate))
        calibration_seeds = [8_100_000 + index for index in calibration_runs]
        try:
            _, _, elapsed = engine.run(calibration_runs, calibration_seeds)
        except Exception as exc:
            if is_gpu_out_of_memory(exc):
                print_status(
                    f"AUTO CALIBRATION OOM | function={function_name} | "
                    f"optimizer={optimizer_name} | batch={candidate} | skipped"
                )
                engine.backend.free_cached_blocks()
                continue
            raise
        throughput = candidate / max(elapsed, 1.0e-12)
        timing = engine.timing.summary()
        objective_label = (
            f"gpu-cec:{function_name}"
            if engine.selected_objective_backend == "gpu"
            else (
                f"numpy-cec:{function_name}"
                if engine.selected_objective_backend == "numpy"
                else f"opfunu-{engine.objective_evaluator.strategy}"
            )
        )
        measurements.append((candidate, elapsed, throughput, timing, objective_label))
        print_status(
            f"AUTO CALIBRATION | function={function_name} | optimizer={optimizer_name} | "
            f"batch={candidate} | time={elapsed:.3f}s | runs_per_second={throughput:.4f} | "
            f"fitness={timing['fitness']:.3f}s | gpu={timing['gpu_kernel']:.3f}s | "
            f"objective={objective_label}"
        )
    if not measurements:
        raise RuntimeError("No memory-safe GPU batch candidate completed calibration.")
    best = max(measurements, key=lambda item: item[2])
    print_status(
        f"AUTO SELECTED | function={function_name} | optimizer={optimizer_name} | "
        f"effective_run_batch={best[0]} | runs_per_second={best[2]:.4f}"
    )
    return best[0], measurements


def _save_gpu_checkpoint_async(run, checkpoint_path, metadata, output):
    """Serialize one completed run outside the CUDA controller's hot path."""
    started = time.perf_counter()
    save_run_checkpoint(checkpoint_path, metadata, output)
    return run, output, checkpoint_path, time.perf_counter() - started


def execute_gpu_batches(
    function_name,
    optimizer_name,
    args,
    pending_runs,
    checkpoint_records,
    objective_executor=None,
    active_batch_size=None,
):
    """Group pending runs, checkpoint each result, and recover auto batches from OOM."""
    completed = []
    cursor = 0
    active_batch_size = min(
        args.resolved_gpu_batch_size if active_batch_size is None else active_batch_size,
        len(pending_runs),
    )
    initial_batch_count = int(np.ceil(len(pending_runs) / active_batch_size))
    batch_number = 0
    pending_saves = []

    def drain_checkpoint_buffer():
        if not pending_saves:
            return
        wait_started = time.perf_counter()
        saved = [future.result() for future in pending_saves]
        exposed_wait = time.perf_counter() - wait_started
        write_time = sum(item[3] for item in saved)
        hidden_time = max(0.0, write_time - exposed_wait)
        for run, output, checkpoint_path, _seconds in saved:
            completed.append((run, output))
            print_status(
                f"CHECKPOINT SAVED | function={function_name} | optimizer={optimizer_name} | "
                f"run={run + 1}/{args.runs} | path={checkpoint_path}"
            )
        print_status(
            f"GPU CHECKPOINT PIPELINE | writes={len(saved)} | "
            f"write_time={write_time:.4f}s | hidden_by_gpu={hidden_time:.4f}s | "
            f"exposed_wait={exposed_wait:.4f}s"
        )
        pending_saves.clear()

    # One writer is deliberate: it preserves checkpoint ordering and provides
    # a two-buffer pipeline (GPU batch N while batch N-1 is serialized).
    with ThreadPoolExecutor(max_workers=1, thread_name_prefix="gpu-checkpoint") as checkpoint_executor:
        while cursor < len(pending_runs):
            chunk_size = min(active_batch_size, len(pending_runs) - cursor)
            run_indices = pending_runs[cursor : cursor + chunk_size]
            batch_number += 1
            first_run, last_run = run_indices[0] + 1, run_indices[-1] + 1
            print_status(
                f"GPU BATCH {batch_number}/{initial_batch_count} | "
                f"function={function_name} | optimizer={optimizer_name} | "
                f"runs={first_run}-{last_run} | size={chunk_size}"
            )
            try:
                batch_outputs, state, elapsed = run_gpu_batch(
                    function_name,
                    optimizer_name,
                    args,
                    run_indices,
                    objective_executor=objective_executor,
                )
            except Exception as exc:
                if is_gpu_out_of_memory(exc) and args.gpu_batch_size == "auto" and chunk_size > 1:
                    reduced = max(1, chunk_size // 2)
                    print_status(
                        f"GPU OOM | auto batch reduced {chunk_size} -> {reduced}; retrying same runs"
                    )
                    try:
                        from compute_backend import ComputeBackend
                        ComputeBackend(args.compute_device).free_cached_blocks()
                    except Exception:
                        pass
                    active_batch_size = reduced
                    batch_number -= 1
                    initial_batch_count = batch_number + int(
                        np.ceil((len(pending_runs) - cursor) / active_batch_size)
                    )
                    continue
                if is_gpu_out_of_memory(exc):
                    raise RuntimeError(
                        f"GPU batch size {chunk_size} ran out of memory. "
                        "Use --gpu-batch-size with a smaller value."
                    ) from exc
                raise

            # Batch N-1 normally finished writing while CUDA computed batch N.
            drain_checkpoint_buffer()
            for run, output in batch_outputs:
                checkpoint_path, metadata = checkpoint_records[run]
                pending_saves.append(checkpoint_executor.submit(
                    _save_gpu_checkpoint_async,
                    run,
                    checkpoint_path,
                    metadata,
                    output,
                ))
            print_status(
                f"GPU BATCH DONE | batch={batch_number}/{initial_batch_count} | "
                f"runs={first_run}-{last_run} | time={elapsed:.2f}s | "
                f"effective_time_per_run={elapsed / len(run_indices):.2f}s | "
                f"runs_per_second={len(run_indices) / max(elapsed, 1e-12):.4f}"
            )
            cursor += chunk_size
        drain_checkpoint_buffer()
    return completed

def run_parallel_task(task):

    output = run_single(
        task["function_name"],
        task["optimizer_name"],
        task["args"],
        task["seed"],
        task["run"],
        task["total_runs"],
    )
    save_run_checkpoint(
        task["checkpoint_path"],
        task["metadata"],
        output,
    )

    return task["run"], output


def cpu_worker_args(args):
    """Return the scientific run configuration without controller-only GPU objects.

    Spawned CPU workers need the scalar configuration and benchmark classes,
    but CuPy-backed objective instances belong exclusively to the controller's
    CUDA context and are not pickleable.
    """
    worker_args = argparse.Namespace(**vars(args))
    worker_args.gpu_objectives = {}
    worker_args.gpu_objective_reports = {}
    worker_args.vectorized_cpu_objectives = {}
    return worker_args

def plot_convergence(
    curves_dict,
    title,
    out_path,
    optimizer_colors,
    yscale="linear",
):

    fig, ax = plt.subplots(
        figsize=(10, 5),
        facecolor="white",
    )

    plot_items = list(curves_dict.items())

    marker_interval = max(1, max((len(np.asarray(curve)) for _, curve in plot_items), default=1) // 18)
    for style_index, (optimizer_name, curve) in enumerate(plot_items):
        is_macro_de = is_macro_de_optimizer(
            optimizer_name
        )

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
        elif yscale == "log":
            plot_curve = np.where(
                np.isfinite(curve) & (curve > 0.0),
                curve,
                np.nan,
            )
        else:
            plot_curve = curve

        color = optimizer_colors.get(
            optimizer_name,
            None,
        )
        linestyle = LINE_STYLES[style_index % len(LINE_STYLES)]
        marker = MARKERS[style_index % len(MARKERS)]
        zorder = 2 + style_index
        linewidth = 3.0 if is_macro_de else 1.9 + 0.22 * (style_index % 4)
        marker_offset = style_index % marker_interval

        if is_macro_de:
            ax.plot(
                plot_curve,
                linewidth=4.4,
                label="_nolegend_",
                color="black",
                solid_capstyle="round",
                linestyle=linestyle,
                zorder=zorder - 0.2,
            )

        ax.plot(
            plot_curve,
            linewidth=linewidth,
            label=display_optimizer_name(
                optimizer_name
            ),
            color=color,
            solid_capstyle="round",
            linestyle=linestyle,
            marker=marker,
            markevery=(marker_offset, marker_interval),
            markersize=4.5,
            markeredgewidth=0.8,
            alpha=0.82,
            zorder=zorder,
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

def plot_log_convergence(
    curves_dict,
    function_name,
    paths,
    optimizer_colors,
):

    plot_convergence(
        curves_dict,
        f"Convergence Curve - {function_name} (Log Scale)",
        os.path.join(
            paths.fig_dir,
            f"{paths.exp_tag}_{function_name}_convergence_log.png",
        ),
        optimizer_colors,
        yscale="log",
    )


def plot_overlap_diagnostic(
    curves,
    function_name,
    run_number,
    seed,
    cache_signature,
    paths,
):
    """Save a single-run diagnostic without touching normal experiment figures."""
    diagnostic_dir = os.path.join(paths.fig_dir, "diagnostics")
    os.makedirs(diagnostic_dir, exist_ok=True)
    out_path = os.path.join(
        diagnostic_dir,
        f"{paths.exp_tag}_{function_name}_run_{run_number:03d}_overlap_diagnostic.png",
    )
    colors = build_optimizer_colors(OVERLAP_DIAGNOSTIC_OPTIMIZERS)
    fig, (curve_ax, difference_ax) = plt.subplots(
        2,
        1,
        figsize=(10, 8),
        sharex=True,
        gridspec_kw={"height_ratios": (2.3, 1.0)},
        facecolor="white",
    )

    all_positive = all(
        np.all(np.isfinite(curve) & (curve > 0.0))
        for curve in curves.values()
    )
    reference = curves[OVERLAP_DIAGNOSTIC_OPTIMIZERS[0]]
    difference_curves = {
        optimizer_name: np.abs(curves[optimizer_name] - reference)
        for optimizer_name in OVERLAP_DIAGNOSTIC_OPTIMIZERS[1:]
    }
    maximum_difference = max(
        (float(np.max(values)) for values in difference_curves.values()),
        default=0.0,
    )
    zero_display_floor = max(
        np.finfo(float).tiny,
        maximum_difference * 1.0e-16,
    )
    iterations = np.arange(1, len(reference) + 1)
    marker_interval = max(1, len(reference) // 18)
    for style_index, optimizer_name in enumerate(OVERLAP_DIAGNOSTIC_OPTIMIZERS):
        curve = curves[optimizer_name]
        curve_ax.plot(
            iterations,
            curve,
            label=display_optimizer_name(optimizer_name),
            color=colors[optimizer_name],
            linestyle=LINE_STYLES[style_index % len(LINE_STYLES)],
            marker=MARKERS[style_index % len(MARKERS)],
            markevery=(style_index % marker_interval, marker_interval),
            linewidth=2.1,
            markersize=4.5,
            alpha=0.82,
            zorder=2 + style_index,
        )
        if optimizer_name != OVERLAP_DIAGNOSTIC_OPTIMIZERS[0]:
            difference_ax.plot(
                iterations,
                np.maximum(difference_curves[optimizer_name], zero_display_floor),
                label=f"|{optimizer_name} - DE-M|",
                color=colors[optimizer_name],
                linestyle=LINE_STYLES[style_index % len(LINE_STYLES)],
                linewidth=1.9,
            )

    if all_positive:
        curve_ax.set_yscale("log")
    curve_ax.set_ylabel("Fitness")
    curve_ax.set_title(
        f"Cached Single-Run Convergence - {function_name} "
        f"(run {run_number}, seed {seed})"
    )
    curve_ax.grid(alpha=0.3)
    curve_ax.legend()

    difference_ax.set_yscale("log")
    difference_ax.set_xlim(1, len(reference))
    difference_ax.set_xlabel("Iteration")
    difference_ax.set_ylabel("Absolute difference\nfrom DE-M (zero at floor)")
    difference_ax.grid(alpha=0.3)
    difference_ax.legend()
    fig.suptitle(f"Cache signature: {cache_signature}", fontsize=9, y=0.995)
    fig.tight_layout()
    fig.savefig(out_path, dpi=600)
    plt.close(fig)
    return out_path


def run_overlap_diagnostic(args):
    """Run the explicitly cache-only overlap diagnostic and nothing else."""
    paths = make_paths(args, create=False)
    cache_signature, curves = load_overlap_diagnostic_curves(args, paths)
    diagnostics = compare_overlap_diagnostic_curves(curves)
    run_number = args.overlap_diagnostic_run
    seed = args.seed_base + run_number - 1

    print("=" * 60)
    print("CACHE-ONLY OVERLAP DIAGNOSTIC")
    print("=" * 60)
    print(f"Function       : {args.overlap_diagnostic_function}")
    print(f"Run            : {run_number}")
    print(f"Seed           : {seed}")
    print(f"Cache signature: {cache_signature}")
    for diagnostic in diagnostics:
        first_iteration = diagnostic["first_different_iteration"]
        first_iteration_text = (
            "never" if first_iteration is None else str(first_iteration)
        )
        print(
            "PAIR | "
            f"{diagnostic['left']} vs {diagnostic['right']} | "
            f"max_abs_diff={diagnostic['max_absolute_difference']:.17g} | "
            f"exactly_identical={str(diagnostic['exactly_identical']).upper()} | "
            f"first_different_iteration={first_iteration_text}"
        )

    figure_path = plot_overlap_diagnostic(
        curves,
        args.overlap_diagnostic_function,
        run_number,
        seed,
        cache_signature,
        paths,
    )
    print(f"Diagnostic figure: {figure_path}")
    print("No experiments were run and no cache files were changed.")
    return figure_path

def export_results(
    results_struct,
    out_path,
):

    rows = []
    for function_name, optimizer_data in results_struct.items():
        for optimizer_name, data in optimizer_data.items():
            final_fitness = np.asarray(
                data["fitness_runs"],
                dtype=float,
            )
            statistics = (
                ("BEST", np.min(final_fitness)),
                ("WORST", np.max(final_fitness)),
                ("MEAN", np.mean(final_fitness)),
                ("SD", np.std(final_fitness)),
            )
            for statistic, fitness in statistics:
                rows.append({
                    "Function": function_name,
                    "Optimizer": optimizer_name,
                    "Statistic": statistic,
                    "Fitness": fitness,
                })

    df = pd.DataFrame(
        rows,
        columns=["Function", "Optimizer", "Statistic", "Fitness"],
    )

    df.to_excel(
        out_path,
        index=False,
    )

    return df

def main():

    args = parse_args()
    apply_experiment_mode(args)
    if args.overlap_diagnostic_function is not None:
        run_overlap_diagnostic(args)
        return
    gpu_info = None
    args.resolved_gpu_batch_size = 1
    args.estimated_gpu_batch_capacity = 1
    if args.compute_device in GPU_MODES:
        gpu_info = initialize_gpu(memory_fraction=args.gpu_memory_fraction)
        memory_budget = min(
            gpu_info.free_memory_bytes,
            gpu_info.memory_limit_bytes,
        )
        memory_capacity = estimate_population_batch_size(
            args.pop_size,
            args.dims,
            memory_budget,
            max_batch_size=args.runs,
        )
        args.estimated_gpu_batch_capacity = memory_capacity
        if args.gpu_batch_size == "auto":
            args.resolved_gpu_batch_size = estimate_effective_batch_size(
                memory_capacity,
                args.runs,
                args.pop_size,
                args.dims,
                args.objective_workers,
            )
        else:
            args.resolved_gpu_batch_size = resolve_gpu_batch_size(
                args.gpu_batch_size,
                memory_capacity,
                args.runs,
            )
    logging.disable(logging.INFO)
    paths = make_paths(args)
    cache_signature = build_cache_signature(args)
    optimizer_colors = build_optimizer_colors(args.optimizers)
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

    initialize_verified_gpu_objectives(args, selected_functions)

    print("=" * 60)
    print(
        "OPFUNU + MEALPY BENCHMARK FRAMEWORK"
    )
    print("=" * 60)
    print(
        f"Experiment     : {paths.exp_tag}"
    )
    print(
        f"Mode           : {args.experiment_mode}"
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
    print(f"Reuse cache : {'YES' if args.reuse_cache else 'NO'}")
    print(
        f"Compute device : {args.compute_device.upper()}"
    )
    print(
        f"CPU workers    : {args.n_workers}"
    )
    if gpu_info is not None:
        print(f"GPU workers    : {args.gpu_workers}")
        print(f"GPU            : {gpu_info.name}")
        print(f"CuPy           : {gpu_info.cupy_version}")
        print(f"GPU memory     : {gpu_info.total_memory_bytes / 1024**3:.2f} GB")
        print(f"GPU memory cap : {gpu_info.memory_fraction:.0%}")
        requested_batch = str(args.gpu_batch_size).upper()
        print(f"GPU batch request       : {requested_batch}")
        print(f"Memory-safe capacity    : {args.estimated_gpu_batch_capacity}")
        effective_suffix = (
            " (initial heuristic; calibrated per function/optimizer)"
            if args.gpu_batch_size == "auto" and args.gpu_auto_calibration == "yes"
            else ""
        )
        print(f"Effective run batch     : {args.resolved_gpu_batch_size}{effective_suffix}")
        print("Batch policy            : performance-aware")
        print(f"Objective workers       : {args.objective_workers}")
        print(f"Objective evaluation    : {args.objective_evaluation.upper()}")
        verified_names = sorted(args.gpu_objectives)
        fallback_names = [name for name in selected_functions if name not in args.gpu_objectives]
        print(
            f"GPU objectives verified : {len(verified_names)}/{len(selected_functions)}"
        )
        print(
            "GPU objective functions  : "
            + (", ".join(verified_names) if verified_names else "NONE")
        )
        print(
            "CPU objective fallback   : "
            + (", ".join(fallback_names) if fallback_names else "NONE")
        )
        print(
            f"Vectorized CPU CEC      : {len(args.vectorized_cpu_objectives)}/"
            f"{len(selected_functions)}"
        )
    print(
        f"Extra scale    : {args.convergence_extra_scale}"
    )
    print(
        f"Cache signature: {cache_signature}"
    )

    objective_executor = None
    if (
        args.compute_device in GPU_MODES
        and args.objective_workers > 1
        and args.objective_evaluation != "serial"
    ):
        objective_executor = ProcessPoolExecutor(
            max_workers=args.objective_workers,
            mp_context=multiprocessing.get_context("spawn"),
            initializer=initialize_objective_worker,
        )

    results_struct = {}
    optimizer_failures = []

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
                completed = []
                pending_runs = []
                checkpoint_records = {}

                resolved_optimizer = resolve_optimizer_name(optimizer_name)
                if (
                    args.compute_device in GPU_MODES
                    and resolved_optimizer in CUSTOM_OPTIMIZERS
                    and not supports_gpu_batching(optimizer_name)
                ):
                    print_status(
                        f"GPU BATCH UNSUPPORTED | optimizer={optimizer_name} | "
                        "using the CPU MEALPY solve lifecycle"
                    )

                for run in range(args.runs):
                    seed = args.seed_base + run
                    checkpoint_path = run_checkpoint_path(
                        paths,
                        cache_signature,
                        function_name,
                        optimizer_name,
                        run,
                    )
                    metadata = checkpoint_metadata(
                        args,
                        cache_signature,
                        function_name,
                        optimizer_name,
                        run,
                        seed,
                    )
                    checkpoint_records[run] = (
                        checkpoint_path,
                        metadata,
                    )

                    cached_output = None
                    if args.reuse_cache:
                        cached_output = load_run_checkpoint(
                            checkpoint_path,
                            metadata,
                        )
                        if (
                            cached_output is None
                            and resolved_optimizer not in CUSTOM_OPTIMIZERS
                        ):
                            cached_output = load_compatible_cpu_checkpoint(
                                paths,
                                function_name,
                                optimizer_name,
                                run,
                                metadata,
                            )

                    if cached_output is None:
                        pending_runs.append(run)
                    else:
                        completed.append(
                            (run, cached_output)
                        )
                        print_status(
                            f"CACHE HIT | function={function_name} | "
                            f"optimizer={optimizer_name} | "
                            f"run={run + 1}/{args.runs}"
                        )

                if len(pending_runs) == 0:
                    print_status(
                        f"CACHE COMPLETE | function={function_name} | "
                        f"optimizer={optimizer_name} | "
                        f"runs={args.runs}/{args.runs}"
                    )

                if pending_runs and optimizer_uses_gpu(optimizer_name, args.compute_device):
                    actual_batch_size = min(args.resolved_gpu_batch_size, len(pending_runs))
                    if args.gpu_batch_size == "auto" and args.gpu_auto_calibration == "yes":
                        actual_batch_size, _ = calibrate_gpu_batch_size(
                            function_name,
                            optimizer_name,
                            args,
                            min(args.estimated_gpu_batch_capacity, len(pending_runs)),
                            objective_executor,
                        )
                    print_status(
                        f"GPU BATCHING | function={function_name} | "
                        f"optimizer={optimizer_name} | pending_runs={len(pending_runs)} | "
                        f"run_batch_size={actual_batch_size} | gpu_workers=1"
                    )
                    completed.extend(
                        execute_gpu_batches(
                            function_name,
                            optimizer_name,
                            args,
                            pending_runs,
                            checkpoint_records,
                            objective_executor=objective_executor,
                            active_batch_size=actual_batch_size,
                        )
                    )
                elif (
                    args.parallel == "yes"
                    and len(pending_runs) > 1
                    and not optimizer_uses_gpu(optimizer_name, args.compute_device)
                ):

                    tasks = []
                    worker_args = cpu_worker_args(args)

                    for run in pending_runs:
                        checkpoint_path, metadata = checkpoint_records[
                            run
                        ]

                        tasks.append({
                            "run": run,
                            "function_name": function_name,
                            "optimizer_name": optimizer_name,
                            "args": worker_args,
                            "seed": args.seed_base + run,
                            "total_runs": args.runs,
                            "checkpoint_path": checkpoint_path,
                            "metadata": metadata,
                        })

                    active_cpu_workers = min(args.n_workers, len(tasks))
                    print_status(
                        f"SUBMITTED | function={function_name} | "
                        f"optimizer={optimizer_name} | "
                        f"runs={len(tasks)} | workers={active_cpu_workers}"
                    )

                    executor_kwargs = {"max_workers": active_cpu_workers}
                    if args.compute_device in GPU_MODES:
                        # CUDA was initialized in the controller. Spawn avoids
                        # inheriting that context into CPU-only child processes.
                        executor_kwargs["mp_context"] = multiprocessing.get_context("spawn")

                    with ProcessPoolExecutor(**executor_kwargs) as executor:

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
                                f"completed_runs={len(completed)}/{args.runs}"
                            )

                else:
                    for run in pending_runs:
                        checkpoint_path, metadata = checkpoint_records[
                            run
                        ]
                        output = run_single(
                            function_name,
                            optimizer_name,
                            args,
                            args.seed_base + run,
                            run,
                            args.runs,
                        )

                        save_run_checkpoint(
                            checkpoint_path,
                            metadata,
                            output,
                        )
                        print_status(
                            f"CHECKPOINT SAVED | function={function_name} | "
                            f"optimizer={optimizer_name} | "
                            f"run={run + 1}/{args.runs} | "
                            f"path={checkpoint_path}"
                        )
                        completed.append(
                            (run, output)
                        )

                completed = sorted(
                    completed,
                    key=lambda x: x[0],
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

                if len(completed) != args.runs:
                    raise RuntimeError(
                        f"optimizer produced {len(completed)}/{args.runs} completed runs"
                    )
                curve_lengths = {len(np.asarray(curve).reshape(-1)) for curve in curves}
                if curve_lengths != {args.epochs}:
                    raise RuntimeError(
                        f"invalid convergence lengths: {sorted(curve_lengths)}; expected {args.epochs}"
                    )
                mean_curve = np.mean(np.stack(curves, axis=0), axis=0)

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

            except Exception as exc:
                print_status(
                    f"OPTIMIZER FAILED | function={function_name} | "
                    f"optimizer={optimizer_name} | "
                    f"exception={type(exc).__name__}: {exc}"
                )
                traceback.print_exc()
                optimizer_failures.append(
                    (function_name, optimizer_name, type(exc).__name__, str(exc))
                )
                continue


        missing_optimizers = [name for name in args.optimizers if name not in curves_plot]
        if missing_optimizers:
            for missing_optimizer in missing_optimizers:
                print_status(
                    f"PLOT WARNING | {function_name} | missing optimizer {missing_optimizer}"
                )
            print_status(
                f"PLOT DEFERRED | {function_name} | incomplete optimizer set"
            )
        elif len(curves_plot) > 0:
            audit_convergence_curves(curves_plot, function_name)
            plot_log_convergence(
                curves_plot,
                function_name,
                paths,
                optimizer_colors,
            )

    if objective_executor is not None:
        objective_executor.shutdown(wait=True)

    mode_label = args.experiment_mode.capitalize()
    excel_path = os.path.join(
        paths.res_dir,
        f"Global_Results_{paths.exp_tag}_{mode_label}.xlsx",
    )

    export_results(
        results_struct,
        excel_path,
    )

    if optimizer_failures:
        failure_summary = "; ".join(
            f"{function_name}/{optimizer_name}: {exception_name}: {message}"
            for function_name, optimizer_name, exception_name, message in optimizer_failures
        )
        raise RuntimeError(
            f"Experiment finished with {len(optimizer_failures)} optimizer failure(s). "
            f"No failure was silently skipped. {failure_summary}"
        )

    print("\n" + "=" * 60)
    print("COMPLETED")
    print("=" * 60)
    print(f"Figures: {paths.fig_dir}")
    print(f"Results: {paths.res_dir}")

if __name__ == "__main__":

    main()
