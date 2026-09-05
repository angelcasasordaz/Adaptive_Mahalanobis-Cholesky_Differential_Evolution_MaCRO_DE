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
from scipy.stats import friedmanchisquare, rankdata, wilcoxon

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
from mealpy_gpu_adapters import (
    configure_local_mealpy_gpu_backend,
    mealpy_gpu_adapter_class,
    strict_gpu_objective_callable,
    supports_mealpy_gpu_adapter,
)
from objective_evaluation import ObjectiveSpec, initialize_objective_worker

DEFAULT_EPOCHS = 2000
DEFAULT_RUNS = 30
EXP_ID = 7
REUSE_CACHE_FROM_EXP_ID = 5
COMPUTE_DEVICE = "gpu"
# Options:
# "cpu"
# "hybrid"
# "gpu"
CPU_WORKERS = None  # Auto-reserve CPU capacity for the OS/user.
GPU_WORKERS = "auto"
GPU_MEMORY_FRACTION = 0.85
GPU_BATCH_SIZE = "auto"
REUSE_CACHE = True
EXPERIMENT_MODES = [
    "full",
    # "ablation",
    # "sensitivity",
]

MACRO_BETA_MIN = 0.10
MACRO_BETA_MAX = 0.60
MACRO_PCR = 0.10
MACRO_MAHAL_Q = 0.50

SENSITIVITY_CONFIGS = [
    ("mahalanobis_q", [0.50, 0.68, 0.80, 0.90]),
    ("beta_min", [0.10, 0.20, 0.30, 0.40]),
    ("beta_max", [0.60, 0.70, 0.80, 0.90]),
    ("pcr", [0.10, 0.20, 0.30, 0.40]),
]

DE_MC_CF_IMPLEMENTATION_REVISION = "awad-close-far-v2"
FULL_TEST_OPTIMIZER = "MaCRO-DE-t"

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
    "MaCRO-DE-t",
    "BRO",
    "DBO",
    "DE",
    "DMOA",
    "GWO",
    "HHO",
    "MFO",
    "MGO",
    "PSO",
    "SHADE",
    "WOA",
    "JADE",
    "FLA",
]
ABLATION_OPTIMIZERS = [
    "DE",
    "DE-M",
    "DE-MC",
    "DE-MC-CF",
    "MaCRO-DE",
]

# None -> run all discovered CEC functions in ablation mode.
#
# Example:
# ABLATION_FUNCTIONS = [
#     "F12017",
#     "F82017",
#     "F152017",
#     "F242017",
# ]
ABLATION_FUNCTIONS = [
    "F12017",
    "F32017",
    "F82017",
    "F102017",
    "F152017",
    "F172017",
    "F242017",
    "F292017",
]

SENSITIVITY_FUNCTIONS = [
    "F12017",
    "F32017",
    "F82017",
    "F102017",
    "F152017",
    "F172017",
    "F242017",
    "F292017",
]
SENSITIVITY_PARAMETER_ATTRIBUTES = {
    "beta_min": "macro_beta_min",
    "beta_max": "macro_beta_max",
    "pcr": "macro_pcr",
    "mahalanobis_q": "macro_mahal_q",
}
CHART_CMAP = "tab20"
OPTIMIZER_COLOR_MAP = {
    "DE-M": "#ff7f0e",
    "DE-MC": "#2ca02c",
    "DE-MC-CF": "#9467bd",

    "BRO": "#577590",
    "DBO": "#00A6A6",
    "DE": "#6A4C93",
    "DMO": "#90BE6D",
    "GA": "#D81B60",
    "GWO": "#E06C00",
    "HHO": "#4D4D4D",
    "MFO": "#8A5A44",
    "MGO": "#F9844A",
    "PSO": "#9B59B6",
    "SHADE": "#264653",
    "WOA": "#2A9D5B",
    "RIME": "#E3B505",
    "MaCRO-DE": "#3266AD",
    FULL_TEST_OPTIMIZER: "#D62728",
}
CONVERGENCE_SCALE = "log"
CONVERGENCE_SHOW_MARKERS = True
CONVERGENCE_USE_LINE_STYLES = True
# CONVERGENCE_SCALE options:
# "linear"
# "log"
# "symlog"
# "exp"
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
    parser.add_argument("--exp-id", type=int, default=EXP_ID, help="Numeric experiment identifier")
    parser.add_argument(
        "--reuse-cache-from-exp-id",
        type=int,
        default=REUSE_CACHE_FROM_EXP_ID,
        help="Read compatible completed checkpoints from another experiment ID",
    )
    parser.add_argument("--output-root", default=".", help="Root directory for Figures/Results")
    parser.add_argument(
        "--internal-convergence-from-cache-only",
        action="store_true",
        help="Write only internal MaCRO-DE-t-as-main figures from complete compatible caches; never optimize",
    )
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
    mode_group = parser.add_mutually_exclusive_group()
    mode_group.add_argument(
        "--experiment-modes",
        nargs="+",
        choices=["full", "ablation", "sensitivity"],
        default=None,
        help="Experiment modes to execute sequentially (default: EXPERIMENT_MODES)",
    )
    mode_group.add_argument(
        "--experiment-mode",
        choices=["full", "ablation", "sensitivity"],
        default=None,
        help="Execute one experiment mode (backward-compatible alias)",
    )
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
    parser.add_argument(
        "--gpu-workers",
        type=str,
        default=GPU_WORKERS,
        help="Strict-GPU CUDA-owning run workers: auto or an integer from 1 to 4",
    )
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
    if args.exp_id < 0:
        parser.error("--exp-id must be non-negative")
    if args.reuse_cache_from_exp_id is not None and args.reuse_cache_from_exp_id < 0:
        parser.error("--reuse-cache-from-exp-id must be non-negative")
    if CONVERGENCE_SCALE not in {"linear", "log", "symlog", "exp"}:
        parser.error(
            "CONVERGENCE_SCALE must be one of: linear, log, symlog, exp"
        )
    if not isinstance(CONVERGENCE_SHOW_MARKERS, bool):
        parser.error("CONVERGENCE_SHOW_MARKERS must be True or False")
    if not isinstance(CONVERGENCE_USE_LINE_STYLES, bool):
        parser.error("CONVERGENCE_USE_LINE_STYLES must be True or False")
    args.objective_workers = resolve_objective_workers(args.objective_workers, args.n_workers)
    if args.gpu_progress_interval < 1:
        parser.error("--gpu-progress-interval must be at least 1")
    if args.gpu_calibration_epochs < 1 or args.gpu_calibration_epochs > 5:
        parser.error("--gpu-calibration-epochs must be between 1 and 5")
    if args.cec_gpu_verification_points < 100:
        parser.error("--cec-gpu-verification-points must be at least 100")
    args.gpu_worker_policy = (
        "auto" if str(args.gpu_workers).strip().lower() == "auto" else "fixed"
    )
    if args.gpu_worker_policy == "auto":
        args.gpu_workers = "auto"
    else:
        try:
            args.gpu_workers = int(args.gpu_workers)
        except (TypeError, ValueError):
            parser.error("--gpu-workers must be 'auto' or an integer from 1 to 4")
        if not 1 <= args.gpu_workers <= 4:
            parser.error("--gpu-workers must be 'auto' or an integer from 1 to 4")
    if args.overlap_diagnostic_run < 1:
        parser.error("--overlap-diagnostic-run must be at least 1")
    args.experiment_modes = (
        [args.experiment_mode]
        if args.experiment_mode is not None
        else (
            list(args.experiment_modes)
            if args.experiment_modes is not None
            else list(EXPERIMENT_MODES)
        )
    )

    return args

def apply_experiment_mode(args, experiment_mode):

    args.experiment_mode = str(experiment_mode).lower()

    if args.experiment_mode == "ablation":
        args.optimizers = list(ABLATION_OPTIMIZERS)
    elif args.experiment_mode == "sensitivity":
        if args.benchmark != "CEC2017":
            raise ValueError("Sensitivity mode supports CEC2017 only")
        if not SENSITIVITY_CONFIGS:
            raise ValueError("SENSITIVITY_CONFIGS must contain at least one group")
        for parameter, values in SENSITIVITY_CONFIGS:
            if parameter not in SENSITIVITY_PARAMETER_ATTRIBUTES:
                raise ValueError(
                    f"Unknown sensitivity parameter {parameter!r}; expected one of: "
                    f"{', '.join(SENSITIVITY_PARAMETER_ATTRIBUTES)}"
                )
            if not values:
                raise ValueError(f"Sensitivity group {parameter!r} has no values")
        args.optimizers = ["MaCRO-DE"]
        args.sensitivity_configs = [
            (parameter, list(values)) for parameter, values in SENSITIVITY_CONFIGS
        ]
        args.sensitivity_value = None
    elif args.optimizers is None:
        args.optimizers = list(DEFAULT_OPTIMIZERS)


def sensitivity_optimizer_label(parameter, value):
    return f"MaCRO-DE ({parameter}={value:g})"


def comparison_optimizer_order(args):
    if args.experiment_mode != "sensitivity":
        return list(args.optimizers)
    return [
        sensitivity_optimizer_label(parameter, value)
        for parameter, values in args.sensitivity_configs
        for value in values
    ]


def optimizer_experiment_configurations(args):
    if args.experiment_mode != "sensitivity":
        for optimizer_name in args.optimizers:
            yield optimizer_name, optimizer_name, args
        return

    for parameter, values in args.sensitivity_configs:
        parameter_attribute = SENSITIVITY_PARAMETER_ATTRIBUTES[parameter]
        for value in values:
            variant_args = argparse.Namespace(**vars(args))
            variant_args.macro_beta_min = MACRO_BETA_MIN
            variant_args.macro_beta_max = MACRO_BETA_MAX
            variant_args.macro_pcr = MACRO_PCR
            variant_args.macro_mahal_q = MACRO_MAHAL_Q
            setattr(variant_args, parameter_attribute, float(value))
            variant_args.sensitivity_parameter = parameter
            variant_args.sensitivity_value = float(value)
            yield (
                sensitivity_optimizer_label(parameter, value),
                "MaCRO-DE",
                variant_args,
            )

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
    if args.compute_device == "gpu" and not supports_gpu_batching(name):
        optimizer_class = mealpy_gpu_adapter_class(optimizer_name)
        if optimizer_class is None:
            raise RuntimeError(
                f"Strict GPU execution has no adapter for optimizer {name!r} "
                f"(resolved as {optimizer_name!r}); CPU fallback is forbidden"
            )
    else:
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

    def stable_color(optimizer_name):
        raw_name = str(optimizer_name)
        if raw_name in OPTIMIZER_COLOR_MAP:
            return OPTIMIZER_COLOR_MAP[raw_name]
        try:
            canonical_name = resolve_optimizer_name(optimizer_name)
        except ValueError:
            canonical_name = raw_name
        if canonical_name in OPTIMIZER_COLOR_MAP:
            return OPTIMIZER_COLOR_MAP[canonical_name]
        display_name = display_optimizer_name(canonical_name)
        if display_name in OPTIMIZER_COLOR_MAP:
            return OPTIMIZER_COLOR_MAP[display_name]
        stable_index = int(
            hashlib.sha1(canonical_name.encode("utf-8")).hexdigest()[:8],
            16,
        ) % cmap.N
        return cmap(stable_index)

    return {name: stable_color(name) for name in optimizer_names}

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

    if args.compute_device == "gpu" and supports_mealpy_gpu_adapter(optimizer_name):
        optimizer_kwargs.update({
            "compute_device": args.compute_device,
            "gpu_memory_fraction": args.gpu_memory_fraction,
        })
        return optimizer_kwargs

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

def optimizer_scientific_parameters(optimizer_name, args):
    """Return only parameters that can change this optimizer's results."""
    canonical_name = resolve_optimizer_name(optimizer_name)
    optimizer_class = resolve_optimizer_class(optimizer_name)
    scientific_args = argparse.Namespace(**vars(args))
    # Backend selection is execution policy, and must not suppress custom
    # scientific kwargs when a GPU adapter is available.
    scientific_args.compute_device = "cpu"
    init_kwargs = optimizer_init_kwargs(
        optimizer_class,
        canonical_name,
        scientific_args,
    )
    execution_only = {
        "compute_device",
        "gpu_memory_fraction",
    }
    parameters = {
        key: value
        for key, value in init_kwargs.items()
        if key not in execution_only and key not in {"epoch", "pop_size"}
    }
    if canonical_name == "DE-MC-CF":
        parameters["implementation_revision"] = DE_MC_CF_IMPLEMENTATION_REVISION
    return parameters


def build_cache_signature(args, optimizer_name):
    """Build an optimizer-local signature; the comparison list is reporting-only."""
    payload = {
        "benchmark": args.benchmark,
        "optimizer": resolve_optimizer_name(optimizer_name),
        "dims": args.dims,
        "epochs": args.epochs,
        "pop_size": args.pop_size,
        "seed_base": args.seed_base,
        "optimizer_parameters": optimizer_scientific_parameters(
            optimizer_name,
            args,
        ),
    }
    return hashlib.sha1(

        json.dumps(
            payload,
            sort_keys=True,
        ).encode("utf-8")

    ).hexdigest()[:10]


def select_experiment_functions(args, function_map):
    if args.experiment_mode == "sensitivity":
        requested = list(SENSITIVITY_FUNCTIONS)
        missing = [name for name in requested if name not in function_map]
        if missing:
            raise ValueError(
                "Unknown CEC2017 sensitivity function(s): "
                f"{', '.join(missing)}. Available functions: "
                f"{', '.join(function_map)}"
            )
        return requested

    if args.experiment_mode == "ablation" and args.benchmark == "CEC2017":
        requested = (
            list(function_map)
            if ABLATION_FUNCTIONS is None
            else list(ABLATION_FUNCTIONS)
        )
        missing = [name for name in requested if name not in function_map]
        if missing:
            raise ValueError(
                "Unknown CEC2017 ablation function(s): "
                f"{', '.join(missing)}. Available functions: "
                f"{', '.join(function_map)}"
            )
        return requested

    if args.functions == ["ALL"]:
        return list(function_map)
    return args.functions

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
        "optimizer_parameters": optimizer_scientific_parameters(
            optimizer_name,
            args,
        ),
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
        **(
            {
                "cpu_custom_execution": (
                    "batched-v1" if cpu_batching_enabled(args) else "mealpy-scalar"
                )
            }
            if args.compute_device == "cpu"
            else {}
        ),
        "objective_workers": args.objective_workers,
        "objective_evaluation": args.objective_evaluation,
        "cec_objective_backend": args.cec_objective_backend,
        "cec_gpu_version": "cec2017-complete-v3",
        "cec_gpu_verification_points": args.cec_gpu_verification_points,
        **(
            {"optimizer_implementation_revision": DE_MC_CF_IMPLEMENTATION_REVISION}
            if resolve_optimizer_name(optimizer_name) == "DE-MC-CF"
            else {}
        ),
    }


def optimizer_uses_gpu(optimizer_name, compute_device):
    if compute_device not in GPU_MODES:
        return False
    resolved_optimizer = resolve_optimizer_name(optimizer_name)
    return (
        supports_gpu_batching(optimizer_name)
        or (
            compute_device == "gpu"
            and supports_mealpy_gpu_adapter(resolved_optimizer)
        )
    )


def cpu_batching_enabled(args):
    return (
        args.compute_device == "cpu"
        and args.parallel == "yes"
        and any(supports_gpu_batching(name) for name in args.optimizers)
    )

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
    report_invalid=True,
):

    if not os.path.exists(checkpoint_path):
        return None

    try:
        with open(checkpoint_path, "rb") as file:
            payload = pickle.load(file)
    except Exception as exc:
        if report_invalid:
            print_status(
                f"CACHE INVALID | path={checkpoint_path} | reason={exc}"
            )
        return None

    if not checkpoint_metadata_compatible(
        payload.get("metadata"),
        expected_metadata,
    ):
        if report_invalid:
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
        if report_invalid:
            print_status(
                f"CACHE INVALID | path={checkpoint_path} | reason=missing keys"
            )
        return None

    return output


EXECUTION_METADATA_KEYS = frozenset({
    "compute_device",
    "cpu_workers",
    "gpu_workers",
    "gpu_memory_fraction",
    "gpu_batch_size",
    "resolved_gpu_batch_size",
    "estimated_gpu_batch_capacity",
    "gpu_batch_engine_version",
    "gpu_batch_policy",
    "cpu_custom_execution",
    "objective_workers",
    "objective_evaluation",
    "cec_objective_backend",
    "cec_gpu_version",
    "cec_gpu_verification_points",
})


def scientific_checkpoint_metadata(metadata):
    """Return checkpoint identity fields while retaining backend trace metadata on disk."""
    if not isinstance(metadata, dict):
        return metadata
    return {
        key: value
        for key, value in metadata.items()
        if key not in EXECUTION_METADATA_KEYS
    }


SCIENTIFIC_RUN_KEYS = (
    "benchmark",
    "function_name",
    "optimizer_name",
    "dims",
    "epochs",
    "pop_size",
    "run",
    "seed",
)


def _legacy_default_optimizer_parameters(optimizer_name, expected_metadata):
    """Infer parameters omitted by old full-mode checkpoints at their defaults."""
    canonical_name = resolve_optimizer_name(optimizer_name)
    if canonical_name not in CUSTOM_OPTIMIZERS:
        return {}
    legacy_args = argparse.Namespace(
        epochs=expected_metadata["epochs"],
        pop_size=expected_metadata["pop_size"],
        compute_device="cpu",
        gpu_memory_fraction=GPU_MEMORY_FRACTION,
        macro_beta_min=MACRO_BETA_MIN,
        macro_beta_max=MACRO_BETA_MAX,
        macro_pcr=MACRO_PCR,
        macro_mahal_q=MACRO_MAHAL_Q,
    )
    return optimizer_scientific_parameters(optimizer_name, legacy_args)


def checkpoint_metadata_compatible(cached_metadata, expected_metadata):
    """Compare scientific run identity, including legacy checkpoint metadata."""
    if not isinstance(cached_metadata, dict):
        return False
    if any(
        cached_metadata.get(key) != expected_metadata.get(key)
        for key in SCIENTIFIC_RUN_KEYS
    ):
        return False

    expected_parameters = expected_metadata.get("optimizer_parameters", {})
    if "optimizer_parameters" in cached_metadata:
        return cached_metadata["optimizer_parameters"] == expected_parameters

    # Sensitivity checkpoints already recorded the MaCRO-DE values explicitly.
    legacy_parameter_names = {
        "beta_min": "macro_beta_min",
        "beta_max": "macro_beta_max",
        "pcr": "macro_pcr",
        "mahalanobis_q": "macro_mahal_q",
    }
    recorded_parameters = {
        parameter_name: cached_metadata[metadata_name]
        for parameter_name, metadata_name in legacy_parameter_names.items()
        if metadata_name in cached_metadata
    }
    if recorded_parameters:
        return all(
            expected_parameters.get(name) == value
            for name, value in recorded_parameters.items()
        ) and all(
            name in recorded_parameters or name == "implementation_revision"
            for name in expected_parameters
        )

    # Old comparison checkpoints omitted custom parameters from full-mode
    # metadata. They are reusable only for the historical default parameter set.
    expected_revision = expected_parameters.get("implementation_revision")
    cached_revision = cached_metadata.get("optimizer_implementation_revision")
    if expected_revision is not None and cached_revision != expected_revision:
        return False
    return expected_parameters == _legacy_default_optimizer_parameters(
        expected_metadata["optimizer_name"],
        expected_metadata,
    )


def compatible_checkpoint_path(
    search_paths,
    current_checkpoint_path,
    expected_metadata,
    function_name,
    optimizer_name,
    run,
):
    """Find a compatible checkpoint in place without copying or rewriting it."""
    optimizer_tag = (
        f"{safe_path_component(optimizer_name)}_"
        f"{hashlib.sha1(str(optimizer_name).encode('utf-8')).hexdigest()[:8]}"
    )
    candidates = [current_checkpoint_path]
    for candidate_paths in search_paths:
        pattern = os.path.join(
            candidate_paths.cache_dir,
            "*",
            safe_path_component(function_name),
            optimizer_tag,
            f"run_{run + 1:03d}.pkl",
        )
        candidates.extend(sorted(glob.glob(pattern), reverse=True))

    seen = set()
    for candidate_path in candidates:
        if candidate_path in seen:
            continue
        seen.add(candidate_path)
        if load_run_checkpoint(
            candidate_path,
            expected_metadata,
            report_invalid=False,
        ) is not None:
            return candidate_path
    return None


def import_run_checkpoint(
    source_checkpoint_path,
    current_checkpoint_path,
    expected_metadata,
    source_exp_tag,
    current_exp_tag,
    function_name,
    optimizer_name,
    run,
):
    """Import one exact-compatible checkpoint without writing to its source."""
    if not os.path.exists(source_checkpoint_path):
        return None

    output = load_run_checkpoint(
        source_checkpoint_path,
        expected_metadata,
    )
    if output is None:
        print_status(
            f"CACHE SOURCE INCOMPATIBLE | from={source_exp_tag} | "
            f"function={function_name} | optimizer={optimizer_name} | run={run + 1}"
        )
        return None

    save_run_checkpoint(
        current_checkpoint_path,
        expected_metadata,
        output,
    )
    print_status(
        f"CACHE IMPORTED | from={source_exp_tag} | to={current_exp_tag} | "
        f"function={function_name} | optimizer={optimizer_name} | run={run + 1}"
    )
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


def initialize_strict_gpu_worker(memory_fraction, quiet=False):
    """Create one persistent, process-local CUDA context."""
    logging.disable(logging.INFO)
    initialize_gpu(memory_fraction=memory_fraction)
    configure_local_mealpy_gpu_backend()
    if not quiet:
        print_status(
            f"STRICT GPU WORKER READY | pid={os.getpid()} | "
            f"memory_fraction={memory_fraction:.3f} | objective_workers=0"
        )


def _available_host_memory_bytes():
    """Best-effort host-memory guard without adding a runtime dependency."""
    try:
        with open("/proc/meminfo", encoding="utf-8") as meminfo:
            for line in meminfo:
                if line.startswith("MemAvailable:"):
                    return int(line.split()[1]) * 1024
    except (OSError, ValueError, IndexError):
        pass
    try:
        return int(os.sysconf("SC_AVPHYS_PAGES")) * int(os.sysconf("SC_PAGE_SIZE"))
    except (AttributeError, OSError, ValueError):
        return 2 * 1024**3


def _gpu_worker_calibration_run(task):
    """Run one tiny, non-scientific workload in this process-local context."""
    args, function_name, optimizer_name, seed = task
    np.random.seed(seed)
    function_class = args.function_map[function_name]
    _benchmark, problem = build_problem(function_class, args.dims)
    initialize_verified_gpu_objectives(args, [function_name])
    require_verified_gpu_cec(args, function_name, optimizer_name)
    problem["obj_func"] = strict_gpu_objective_callable(
        args.gpu_objectives[function_name]
    )
    optimizer = build_optimizer(optimizer_name, args)
    result = optimizer.solve(problem, seed=seed)
    return os.getpid(), float(result.target.fitness)


def _gpu_worker_candidates(args, gpu_info, pending_runs):
    pending_limit = max(1, min(4, int(pending_runs)))
    # CUDA contexts and imported scientific stacks consume meaningful host and
    # device memory even before population arrays are allocated.
    host_limit = max(1, _available_host_memory_bytes() // (768 * 1024**2))
    vram_budget = min(
        int(gpu_info.free_memory_bytes),
        int(gpu_info.total_memory_bytes * args.gpu_memory_fraction),
    )
    vram_limit = max(1, vram_budget // (512 * 1024**2))
    limit = max(1, min(pending_limit, host_limit, vram_limit, args.n_workers))
    return list(range(1, limit + 1))


def select_strict_gpu_workers(args, gpu_info, selected_functions):
    """Measure persistent CUDA-worker throughput for scalar GPU optimizers."""
    if args.gpu_workers != "auto":
        return min(int(args.gpu_workers), max(1, args.runs))

    scalar_optimizers = [
        name for name in args.optimizers
        if optimizer_uses_gpu(name, "gpu") and not supports_gpu_batching(name)
    ]
    # One larger BatchedDEEngine tensor is preferred for MaCRO-DE. Multiple
    # contexts are considered only when scalar independent-run work exists.
    if not scalar_optimizers:
        return 1
    optimizer_name = next(
        (name for name in scalar_optimizers if "PSO" in resolve_optimizer_name(name)),
        scalar_optimizers[0],
    )
    function_name = selected_functions[0]
    candidates = _gpu_worker_candidates(args, gpu_info, args.runs)
    measurements = []
    for worker_count in candidates:
        worker_fraction = args.gpu_memory_fraction / worker_count
        calibration_args = strict_gpu_worker_args(args)
        calibration_args.epochs = args.gpu_calibration_epochs
        calibration_args.gpu_memory_fraction = worker_fraction
        calibration_runs = max(4, worker_count * 2)
        try:
            with ProcessPoolExecutor(
                max_workers=worker_count,
                mp_context=multiprocessing.get_context("spawn"),
                initializer=initialize_strict_gpu_worker,
                initargs=(worker_fraction, True),
            ) as executor:
                warm = [
                    executor.submit(
                        _gpu_worker_calibration_run,
                        (calibration_args, function_name, optimizer_name, 8_200_000 + index),
                    )
                    for index in range(worker_count * 2)
                ]
                for future in warm:
                    future.result()
                started = time.perf_counter()
                measured = [
                    executor.submit(
                        _gpu_worker_calibration_run,
                        (calibration_args, function_name, optimizer_name, 8_300_000 + index),
                    )
                    for index in range(calibration_runs)
                ]
                for future in measured:
                    future.result()
                elapsed = time.perf_counter() - started
        except Exception as exc:
            if is_gpu_out_of_memory(exc) or "CUDA" in str(exc).upper():
                print_status(
                    f"GPU WORKER TEST | workers={worker_count} | OOM/UNAVAILABLE | skipped"
                )
                continue
            raise
        throughput = calibration_runs / max(elapsed, 1.0e-12)
        measurements.append((worker_count, throughput))
        print_status(
            f"GPU WORKER TEST | workers={worker_count} | runs/s={throughput:.4f}"
        )
    if not measurements:
        raise RuntimeError("No safe persistent CUDA worker count completed calibration")
    return max(measurements, key=lambda item: item[1])[0]


def strict_gpu_worker_args(args):
    """Remove controller CUDA objects before crossing the spawn boundary."""
    worker_args = argparse.Namespace(**vars(args))
    worker_args.gpu_objectives = {}
    worker_args.gpu_objective_reports = {}
    worker_args.vectorized_cpu_objectives = {}
    worker_args.gpu_memory_fraction = (
        args.gpu_memory_fraction / args.gpu_workers
        if isinstance(args.gpu_workers, int)
        else args.gpu_memory_fraction
    )
    return worker_args


def run_strict_gpu_task(task):
    """Execute complete GPU batches locally in the CUDA-owning spawned worker."""
    worker_args = task["args"]
    function_name = task["function_name"]
    optimizer_name = task["optimizer_name"]
    run_indices = task["run_indices"]
    if not supports_gpu_batching(optimizer_name):
        initialize_verified_gpu_objectives(worker_args, [function_name])
        completed = []
        for run in run_indices:
            checkpoint_path, metadata = task["checkpoint_records"][run]
            output = run_single(
                function_name,
                optimizer_name,
                worker_args,
                worker_args.seed_base + run,
                run,
                worker_args.runs,
            )
            save_run_checkpoint(checkpoint_path, metadata, output)
            completed.append((run, output))
        return os.getpid(), completed
    initialize_verified_gpu_objectives(worker_args, [function_name])
    active_batch_size = min(task["active_batch_size"], len(run_indices))
    plan_workers = min(
        worker_args.objective_workers,
        active_batch_size,
    )
    plan_executor = None
    if plan_workers > 1 and active_batch_size >= 4:
        plan_executor = ProcessPoolExecutor(
            max_workers=plan_workers,
            mp_context=multiprocessing.get_context("spawn"),
            initializer=initialize_objective_worker,
        )
        print_status(
            f"STRICT GPU PLAN WORKERS | workers={plan_workers} | "
            f"cuda_workers=1"
        )
    try:
        objective_backends = {}
        if worker_args.gpu_batch_size == "auto" and worker_args.gpu_auto_calibration == "yes":
            active_batch_size, selected_backend, _ = calibrate_gpu_batch_size(
                function_name,
                optimizer_name,
                worker_args,
                min(worker_args.estimated_gpu_batch_capacity, len(run_indices)),
                None,
                random_plan_executor=plan_executor,
            )
            objective_backends[active_batch_size] = selected_backend
        completed = execute_gpu_batches(
            function_name,
            optimizer_name,
            worker_args,
            run_indices,
            task["checkpoint_records"],
            objective_executor=None,
            random_plan_executor=plan_executor,
            active_batch_size=active_batch_size,
            objective_backends=objective_backends,
        )
    finally:
        if plan_executor is not None:
            plan_executor.shutdown(wait=True)
    return os.getpid(), completed


def execute_strict_gpu_pool(
    executor,
    function_name,
    optimizer_name,
    args,
    pending_runs,
    checkpoint_records,
):
    """Distribute run groups across persistent local-CUDA workers."""
    active_workers = (
        1
        if supports_gpu_batching(optimizer_name)
        else min(args.gpu_workers, len(pending_runs))
    )
    run_groups = [
        [int(run) for run in group]
        for group in np.array_split(np.asarray(pending_runs, dtype=int), active_workers)
        if len(group)
    ]
    worker_args = strict_gpu_worker_args(args)
    futures = []
    for run_indices in run_groups:
        futures.append(executor.submit(
            run_strict_gpu_task,
            {
                "function_name": function_name,
                "optimizer_name": optimizer_name,
                "args": worker_args,
                "run_indices": run_indices,
                "checkpoint_records": {
                    run: checkpoint_records[run]
                    for run in run_indices
                },
                "active_batch_size": min(args.resolved_gpu_batch_size, len(run_indices)),
            },
        ))
    completed = []
    worker_pids = set()
    for future in as_completed(futures):
        worker_pid, outputs = future.result()
        worker_pids.add(worker_pid)
        completed.extend(outputs)
        print_status(
            f"STRICT GPU WORKER COMPLETE | pid={worker_pid} | "
            f"function={function_name} | optimizer={optimizer_name} | "
            f"completed_runs={len(completed)}/{len(pending_runs)}"
        )
    if active_workers > 1 and len(worker_pids) != active_workers:
        raise RuntimeError(
            f"strict GPU pool used {len(worker_pids)}/{active_workers} selected workers"
        )
    return completed

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
    strict_objective = None
    if args.compute_device == "gpu":
        require_verified_gpu_cec(args, function_name, optimizer_name)
        strict_objective = strict_gpu_objective_callable(
            args.gpu_objectives[function_name]
        )
        problem["obj_func"] = strict_objective

    optimizer = build_optimizer(
        optimizer_name,
        args,
    )

    t0 = time.time()
    result = optimizer.solve(problem, seed=seed)
    runtime = time.time() - t0
    if args.compute_device == "gpu":
        gpu_math = getattr(optimizer, "_gpu_math", None)
        if gpu_math is None or gpu_math.gpu_operation_count < 1:
            raise RuntimeError(
                f"Strict GPU adapter for {optimizer_name} executed no GPU array operations; "
                "CPU fallback is forbidden"
            )
        if strict_objective is None or strict_objective.gpu_call_count < 1:
            raise RuntimeError(
                f"Strict GPU run for {optimizer_name} executed no gpu-cec evaluations; "
                "CPU objective fallback is forbidden"
            )
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
    random_plan_executor=None,
    epochs=None,
    objective_backend=None,
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
        "random_plan_executor": random_plan_executor,
        "objective_workers": args.objective_workers,
        "objective_strategy": args.objective_evaluation,
        "objective_spec": ObjectiveSpec(
            function_class.__module__,
            function_class.__name__,
            args.dims,
        ),
        "gpu_objective": args.gpu_objectives.get(function_name),
        "vectorized_cpu_objective": args.vectorized_cpu_objectives.get(function_name),
        "cec_objective_backend": (
            args.cec_objective_backend
            if objective_backend is None
            else objective_backend
        ),
    }
    for name in ("wf", "cr", "beta_min", "beta_max", "pcr"):
        if hasattr(reference_optimizer, name):
            kwargs[name] = getattr(reference_optimizer, name)
    return BatchedDEEngine(**kwargs)


def initialize_verified_gpu_objectives(args, selected_functions):
    """Load and verify vectorized NumPy/CUDA CEC objective candidates."""
    if not hasattr(args, "gpu_objectives"):
        args.gpu_objectives = {}
    if not hasattr(args, "gpu_objective_reports"):
        args.gpu_objective_reports = {}
    if not hasattr(args, "vectorized_cpu_objectives"):
        args.vectorized_cpu_objectives = {}
    if args.benchmark != "CEC2017":
        return
    if args.compute_device not in GPU_MODES and not cpu_batching_enabled(args):
        return
    # Lazy imports are essential: Windows/CPU execution must not import CuPy or
    # the optional GPU objective implementation.
    from cec2017_gpu import (
        CEC2017_GPU_CANDIDATES,
        CEC2017GpuObjective,
        verify_gpu_objective,
    )
    from compute_backend import ComputeBackend

    backend = (
        ComputeBackend(args.compute_device)
        if args.compute_device in GPU_MODES
        else None
    )
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
        if args.compute_device not in GPU_MODES:
            continue
        strict_gpu_objective_required = args.compute_device == "gpu"
        if (
            args.cec_objective_backend in {"opfunu", "numpy"}
            and not strict_gpu_objective_required
        ):
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


def require_verified_gpu_cec(args, function_name, optimizer_name):
    """Require verified gpu-cec for GPU-batched strict-GPU execution."""
    report = args.gpu_objective_reports.get(function_name)
    objective = args.gpu_objectives.get(function_name)
    if report is not None and report.verified and objective is not None:
        return "gpu"

    if report is None:
        reason = "verification report unavailable"
    else:
        reason = (
            f"mismatches={report.mismatches} | "
            f"max_abs={report.max_absolute_error:.6e} | "
            f"max_rel={report.max_relative_error:.6e}"
        )
    print_status(
        f"STRICT GPU OBJECTIVE VERIFICATION FAILED | function={function_name} | "
        f"optimizer={optimizer_name} | {reason}"
    )
    raise RuntimeError(
        f"Strict GPU execution requires a verified gpu-cec objective for "
        f"{function_name}; {reason}"
    )


def _batch_progress_callback(optimizer_name, run_indices, compute_device):
    run_label = f"{run_indices[0] + 1}-{run_indices[-1] + 1}"
    execution_label = "GPU" if compute_device in GPU_MODES else "CPU"
    numerical_label = "gpu_kernel" if compute_device in GPU_MODES else "numerical"

    def report(epoch, total_epochs, elapsed, epoch_time, timing, strategy, memory):
        eta = (elapsed / epoch) * (total_epochs - epoch) if epoch >= 2 else float("nan")
        eta_text = f"{eta:.1f}s" if np.isfinite(eta) else "warming-up"
        memory_text = (
            f" | gpu_pool={memory['pool_total_bytes'] / 1024**2:.1f}MiB"
            if compute_device in GPU_MODES
            else ""
        )
        print_status(
            f"{execution_label} BATCH PROGRESS | "
            f"optimizer={optimizer_name} | epoch={epoch}/{total_epochs} | "
            f"runs={run_label} | elapsed={elapsed:.1f}s | epoch_time={epoch_time:.3f}s | "
            f"{numerical_label}_time={timing['gpu_kernel']:.2f}s | "
            f"fitness_time={timing['fitness']:.2f}s | transfer_time={timing['transfer']:.2f}s | "
            f"objective={strategy} | eta={eta_text}{memory_text}"
        )

    return report


def run_gpu_batch(
    function_name,
    optimizer_name,
    args,
    run_indices,
    objective_executor=None,
    random_plan_executor=None,
    objective_backend=None,
):
    """Advance independent runs together in one array-backend controller."""
    if args.compute_device == "gpu" and supports_gpu_batching(optimizer_name):
        objective_backend = require_verified_gpu_cec(
            args,
            function_name,
            optimizer_name,
        )
    engine = build_batched_engine(
        function_name,
        optimizer_name,
        args,
        objective_executor=objective_executor,
        random_plan_executor=random_plan_executor,
        objective_backend=objective_backend,
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
        progress_callback=_batch_progress_callback(
            optimizer_name, run_indices, args.compute_device
        ),
        progress_interval=args.gpu_progress_interval,
    )
    if args.compute_device == "gpu":
        if not engine.backend.uses_gpu:
            raise RuntimeError("Strict GPU batched engine did not use its CUDA backend")
        if engine.selected_objective_backend != "gpu":
            raise RuntimeError(
                f"Strict GPU batched engine selected {engine.selected_objective_backend!r}; "
                "gpu-cec is required"
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
    random_plan_executor=None,
    candidate_batch_sizes=None,
):
    """Select batch size and verified CEC backend by median end-to-end time."""
    candidates = (
        performance_batch_candidates(maximum_batch_size, args.runs)
        if candidate_batch_sizes is None
        else [
            int(candidate)
            for candidate in candidate_batch_sizes
            if 1 <= int(candidate) <= maximum_batch_size
        ]
    )
    if not candidates:
        raise ValueError("At least one valid GPU calibration batch size is required")

    strict_gpu_backend = (
        require_verified_gpu_cec(args, function_name, optimizer_name)
        if args.compute_device == "gpu" and supports_gpu_batching(optimizer_name)
        else None
    )
    requested_backend = args.cec_objective_backend
    if strict_gpu_backend is not None:
        backends = [strict_gpu_backend]
    elif requested_backend == "auto":
        backends = []
        gpu_report = args.gpu_objective_reports.get(function_name)
        if (
            function_name in args.gpu_objectives
            and gpu_report is not None
            and gpu_report.verified
        ):
            backends.append("gpu")
        if function_name in args.vectorized_cpu_objectives:
            backends.append("numpy")
        if not backends:
            backends.append("auto")
    else:
        backends = [requested_backend]

    if strict_gpu_backend is None:
        dispatch_warm = build_batched_engine(
            function_name,
            optimizer_name,
            args,
            objective_executor=objective_executor,
            random_plan_executor=random_plan_executor,
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

    measurements = []
    for candidate in candidates:
        calibration_runs = list(range(candidate))
        warmup_seeds = [8_000_000 + index for index in calibration_runs]
        calibration_seeds = [8_100_000 + index for index in calibration_runs]
        for backend_name in backends:
            engine = build_batched_engine(
                function_name,
                optimizer_name,
                args,
                objective_executor=objective_executor,
                random_plan_executor=random_plan_executor,
                epochs=args.gpu_calibration_epochs,
                objective_backend=backend_name,
            )
            try:
                # One complete unmeasured warm-up for this exact pair.
                engine.run(calibration_runs, warmup_seeds)
                elapsed_repetitions = []
                for _repetition in range(3):
                    measured_engine = build_batched_engine(
                        function_name,
                        optimizer_name,
                        args,
                        objective_executor=objective_executor,
                        random_plan_executor=random_plan_executor,
                        epochs=args.gpu_calibration_epochs,
                        objective_backend=backend_name,
                    )
                    repetition_started = time.perf_counter()
                    measured_engine.run(
                        calibration_runs,
                        calibration_seeds,
                    )
                    measured_engine.backend.synchronize()
                    elapsed_repetitions.append(
                        float(time.perf_counter() - repetition_started)
                    )
            except Exception as exc:
                if is_gpu_out_of_memory(exc):
                    print_status(
                        f"AUTO CALIBRATION OOM | function={function_name} | "
                        f"optimizer={optimizer_name} | batch={candidate} | "
                        f"backend={backend_name} | skipped"
                    )
                    engine.backend.free_cached_blocks()
                    continue
                raise

            median_elapsed = float(np.median(elapsed_repetitions))
            throughput = candidate / max(median_elapsed, 1.0e-12)
            backend_label = {
                "gpu": "gpu-cec",
                "numpy": "numpy-cec",
                "opfunu": "opfunu",
                "auto": measured_engine.selected_objective_backend,
            }[backend_name]
            measurements.append({
                "batch_size": candidate,
                "backend": backend_name,
                "backend_label": backend_label,
                "median_time": median_elapsed,
                "runs_per_second": throughput,
                "repetition_times": tuple(elapsed_repetitions),
            })
            print_status(
                f"AUTO CALIBRATION | function={function_name} | "
                f"optimizer={optimizer_name} | batch={candidate} | "
                f"backend={backend_label} | median_time={median_elapsed:.3f}s | "
                f"runs/s={throughput:.4f}"
            )
    if not measurements:
        raise RuntimeError("No memory-safe GPU batch candidate completed calibration.")
    best = max(measurements, key=lambda item: item["runs_per_second"])
    print_status(
        f"AUTO SELECTED | function={function_name} | optimizer={optimizer_name} | "
        f"batch={best['batch_size']} | backend={best['backend_label']} | "
        f"runs/s={best['runs_per_second']:.4f}"
    )
    return best["batch_size"], best["backend"], measurements


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
    random_plan_executor=None,
    active_batch_size=None,
    objective_backends=None,
):
    """Group pending runs, checkpoint results, and recover GPU batches from OOM."""
    completed = []
    cursor = 0
    active_batch_size = min(
        args.resolved_gpu_batch_size if active_batch_size is None else active_batch_size,
        len(pending_runs),
    )
    initial_batch_count = int(np.ceil(len(pending_runs) / active_batch_size))
    batch_number = 0
    pending_saves = []
    execution_label = "GPU" if args.compute_device in GPU_MODES else "CPU"
    objective_backends = dict(objective_backends or {})
    strict_gpu_backend = (
        require_verified_gpu_cec(args, function_name, optimizer_name)
        if args.compute_device == "gpu" and supports_gpu_batching(optimizer_name)
        else None
    )

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
            f"{execution_label} CHECKPOINT PIPELINE | writes={len(saved)} | "
            f"write_time={write_time:.4f}s | hidden_by_compute={hidden_time:.4f}s | "
            f"exposed_wait={exposed_wait:.4f}s"
        )
        pending_saves.clear()

    # One writer is deliberate: it preserves checkpoint ordering and provides
    # a two-buffer pipeline (compute batch N while batch N-1 is serialized).
    with ThreadPoolExecutor(max_workers=1, thread_name_prefix="gpu-checkpoint") as checkpoint_executor:
        while cursor < len(pending_runs):
            chunk_size = min(active_batch_size, len(pending_runs) - cursor)
            run_indices = pending_runs[cursor : cursor + chunk_size]
            batch_number += 1
            first_run, last_run = run_indices[0] + 1, run_indices[-1] + 1
            print_status(
                f"{execution_label} BATCH {batch_number}/{initial_batch_count} | "
                f"function={function_name} | optimizer={optimizer_name} | "
                f"runs={first_run}-{last_run} | size={chunk_size}"
            )
            objective_backend = (
                strict_gpu_backend
                if strict_gpu_backend is not None
                else objective_backends.get(chunk_size)
            )
            if (
                objective_backend is None
                and args.compute_device in GPU_MODES
                and args.gpu_batch_size == "auto"
                and args.gpu_auto_calibration == "yes"
            ):
                _, objective_backend, _ = calibrate_gpu_batch_size(
                    function_name,
                    optimizer_name,
                    args,
                    chunk_size,
                    objective_executor,
                    random_plan_executor=random_plan_executor,
                    candidate_batch_sizes=[chunk_size],
                )
                objective_backends[chunk_size] = objective_backend
            try:
                batch_outputs, state, elapsed = run_gpu_batch(
                    function_name,
                    optimizer_name,
                    args,
                    run_indices,
                    objective_executor=objective_executor,
                    random_plan_executor=random_plan_executor,
                    objective_backend=objective_backend,
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
                f"{execution_label} BATCH DONE | batch={batch_number}/{initial_batch_count} | "
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


def should_use_cpu_parallel_executor(optimizer_name, optimizer_args, pending_runs):
    """Select independent CPU run workers when no GPU execution path exists."""
    return (
        optimizer_args.compute_device != "gpu"
        and
        optimizer_args.parallel == "yes"
        and len(pending_runs) > 1
        and not optimizer_uses_gpu(optimizer_name, optimizer_args.compute_device)
    )


def plot_convergence(
    curves_dict,
    title,
    out_path,
    optimizer_colors,
    yscale="linear",
    show_markers=False,
    use_line_styles=False,
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

        color = (
            OPTIMIZER_COLOR_MAP["MaCRO-DE"]
            if is_macro_de
            else optimizer_colors.get(optimizer_name, None)
        )
        linestyle = (
            "-"
            if is_macro_de or not use_line_styles
            else LINE_STYLES[style_index % len(LINE_STYLES)]
        )
        marker = (
            MARKERS[style_index % len(MARKERS)]
            if show_markers and not is_macro_de
            else None
        )
        markevery = (
            (style_index % marker_interval, marker_interval)
            if marker is not None
            else None
        )
        zorder = 3 if is_macro_de else 2
        linewidth = 2.5 if is_macro_de else 2.0

        if is_macro_de:
            ax.plot(
                plot_curve,
                linewidth=4.0,
                label="_nolegend_",
                color="black",
                solid_capstyle="round",
                linestyle=linestyle,
                alpha=0.90,
                zorder=zorder - 0.1,
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
            markevery=markevery,
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


def plot_configured_convergence(
    curves_dict,
    function_name,
    paths,
    optimizer_colors,
    requested_scale,
    show_markers,
    use_line_styles,
):
    valid_scales = {"linear", "log", "symlog", "exp", "auto"}
    if requested_scale not in valid_scales:
        raise ValueError(
            f"Unsupported convergence scale {requested_scale!r}; "
            f"choose from {', '.join(sorted(valid_scales - {'auto'}))}"
        )

    selected_scale = resolve_convergence_scale(curves_dict, requested_scale)
    if selected_scale is None:
        return None

    scale_label = {
        "linear": "",
        "log": " (Log Scale)",
        "symlog": " (Symlog Scale)",
        "exp": " (Exp Scale)",
    }[selected_scale]
    out_path = os.path.join(
        paths.fig_dir,
        f"{paths.exp_tag}_{function_name}_convergence_{selected_scale}.png",
    )
    plot_convergence(
        curves_dict,
        f"Convergence Curve - {function_name}{scale_label}",
        out_path,
        optimizer_colors,
        yscale=selected_scale,
        show_markers=show_markers,
        use_line_styles=use_line_styles,
    )
    return selected_scale


def plot_full_test_convergence(
    curves_dict,
    function_name,
    paths,
    optimizer_colors,
    requested_scale,
    show_markers,
    use_line_styles,
):
    """Add the FULL comparison that includes the experimental optimizer."""
    selected_scale = resolve_convergence_scale(curves_dict, requested_scale)
    if selected_scale is None:
        return None

    scale_label = {
        "linear": "",
        "log": " (Log Scale)",
        "symlog": " (Symlog Scale)",
        "exp": " (Exp Scale)",
    }[selected_scale]
    out_path = os.path.join(
        paths.fig_dir,
        f"{paths.exp_tag}_{function_name}_convergence_{selected_scale}"
        "_With_MaCRO_DE_t.png",
    )
    plot_convergence(
        curves_dict,
        f"Convergence Curve - {function_name}{scale_label} - With MaCRO-DE-t",
        out_path,
        optimizer_colors,
        yscale=selected_scale,
        show_markers=show_markers,
        use_line_styles=use_line_styles,
    )
    return selected_scale

def plot_internal_macro_de_t_as_main(
    curves_dict,
    function_name,
    paths,
    optimizer_colors,
    requested_scale,
    show_markers,
    use_line_styles,
):
    """Internal diagnostic only: substitute plot data, preserving main-plot styles.

    Keep the MaCRO-DE key and its position so the shared renderer preserves its
    blue emphasis and every other optimizer's style index. Scientific results
    and the caller's curves are never modified.
    """
    if "MaCRO-DE" not in curves_dict or FULL_TEST_OPTIMIZER not in curves_dict:
        return None
    diagnostic_curves = {
        name: curves_dict[FULL_TEST_OPTIMIZER] if name == "MaCRO-DE" else curve
        for name, curve in curves_dict.items()
        if name != FULL_TEST_OPTIMIZER
    }
    selected_scale = resolve_convergence_scale(diagnostic_curves, requested_scale)
    if selected_scale is None:
        return None
    out_path = os.path.join(
        paths.fig_dir,
        f"Convergence_{function_name}_Internal_MaCRO_DE_t_As_Main.png",
    )
    plot_convergence(
        diagnostic_curves,
        f"Internal Diagnostic - {function_name} - MaCRO-DE-t as MaCRO-DE",
        out_path,
        optimizer_colors,
        yscale=selected_scale,
        show_markers=show_markers,
        use_line_styles=use_line_styles,
    )
    return out_path


def run_internal_convergence_from_cache(args):
    """Read compatible cached curves and write only the third diagnostic figure.

    Missing/incomplete functions are skipped. This path never enters the
    experiment runner, initializes a GPU, or writes tables/checkpoints.
    """
    if args.experiment_mode != "full":
        raise ValueError("The internal MaCRO-DE-t-as-main diagnostic requires full comparison mode")
    if not {"MaCRO-DE", FULL_TEST_OPTIMIZER}.issubset(args.optimizers):
        raise ValueError("The internal diagnostic requires MaCRO-DE and MaCRO-DE-t")
    paths = make_paths(args, create=False)
    search_paths = [paths]
    if args.reuse_cache_from_exp_id is not None:
        source_args = argparse.Namespace(**vars(args))
        source_args.exp_id = args.reuse_cache_from_exp_id
        search_paths.append(make_paths(source_args, create=False))
    function_names = args.functions
    if function_names == ["ALL"]:
        function_names = sorted({
            os.path.basename(path)
            for candidate in search_paths
            for path in glob.glob(os.path.join(candidate.cache_dir, "*", "F*"))
            if os.path.isdir(path)
        })
    colors = build_optimizer_colors(args.optimizers)
    written = []
    for function_name in function_names:
        curves = {}
        missing = []
        for optimizer_name in args.optimizers:
            signature = build_cache_signature(args, optimizer_name)
            run_curves = []
            for run in range(args.runs):
                expected = {
                    "benchmark": args.benchmark,
                    "function_name": function_name,
                    "optimizer_name": optimizer_name,
                    "optimizer_parameters": optimizer_scientific_parameters(optimizer_name, args),
                    "dims": args.dims,
                    "epochs": args.epochs,
                    "pop_size": args.pop_size,
                    "run": run,
                    "seed": args.seed_base + run,
                }
                current = run_checkpoint_path(paths, signature, function_name, optimizer_name, run)
                matched = compatible_checkpoint_path(
                    search_paths, current, expected, function_name, optimizer_name, run,
                )
                output = load_run_checkpoint(matched, expected) if matched else None
                if output is None or np.asarray(output["curve"]).shape != (args.epochs,):
                    break
                run_curves.append(output["curve"])
            if len(run_curves) != args.runs:
                missing.append(optimizer_name)
            else:
                curves[optimizer_name] = np.mean(np.stack(run_curves, axis=0), axis=0)
        if missing:
            print_status(f"INTERNAL PLOT SKIPPED | {function_name} | incomplete cache: {', '.join(missing)}")
            continue
        os.makedirs(paths.fig_dir, exist_ok=True)
        out_path = plot_internal_macro_de_t_as_main(
            curves, function_name, paths, colors, CONVERGENCE_SCALE,
            CONVERGENCE_SHOW_MARKERS, CONVERGENCE_USE_LINE_STYLES,
        )
        if out_path is not None:
            written.append(out_path)
            print_status(f"INTERNAL PLOT FROM CACHE | {out_path}")
    print_status(f"INTERNAL CACHE-ONLY PLOTS COMPLETE | written={len(written)} | optimization runs=0")
    return written


def plot_log_convergence(
    curves_dict,
    function_name,
    paths,
    optimizer_colors,
    show_markers=False,
    use_line_styles=False,
):
    return plot_configured_convergence(
        curves_dict,
        function_name,
        paths,
        optimizer_colors,
        "log",
        show_markers,
        use_line_styles,
    )


def _plot_sensitivity_curve(axis, curve, scale, label, color, linestyle):
    """Plot one existing mean sensitivity curve using convergence scale rules."""
    curve = np.asarray(curve, dtype=float)
    if scale == "exp":
        plot_curve = np.where(
            np.isfinite(curve),
            np.exp(np.clip(curve, -745.0, 709.0)),
            np.nan,
        )
    elif scale == "log":
        plot_curve = np.where(
            np.isfinite(curve) & (curve > 0.0),
            curve,
            np.nan,
        )
    else:
        plot_curve = curve

    axis.plot(
        plot_curve,
        label=label,
        color=color,
        linestyle=linestyle,
        linewidth=2.0,
        alpha=0.86,
    )


def plot_sensitivity_grouped_convergence(
    results_struct,
    function_names,
    parameter,
    values,
    paths,
):
    """Plot the four OFAT values for one parameter across the four CEC panels."""
    fig, axes = plt.subplots(
        2,
        2,
        figsize=(14.4, 10.4),
        squeeze=False,
        facecolor="white",
    )
    colors = plt.get_cmap("tab10")(np.arange(len(values)))

    for axis, function_name in zip(axes.reshape(-1), function_names):
        curves = {
            sensitivity_optimizer_label(parameter, value): results_struct[
                function_name
            ][sensitivity_optimizer_label(parameter, value)]["curve"]
            for value in values
        }
        scale = resolve_convergence_scale(curves, CONVERGENCE_SCALE)
        for value_index, value in enumerate(values):
            label = sensitivity_optimizer_label(parameter, value)
            _plot_sensitivity_curve(
                axis,
                curves[label],
                scale,
                f"{value:g}",
                colors[value_index],
                LINE_STYLES[value_index % len(LINE_STYLES)],
            )
        if scale in ("log", "symlog"):
            axis.set_yscale(scale)
        axis.set_xlabel("Iteration")
        axis.set_ylabel("exp(Fitness)" if scale == "exp" else "Fitness")
        axis.set_title(function_name)
        axis.grid(alpha=0.3)
        axis.legend(title=parameter)

    fig.suptitle(f"Sensitivity Convergence - {parameter}", fontsize=14)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    out_path = os.path.join(
        paths.fig_dir,
        f"Sensitivity_grouped_{parameter}_Convergence.png",
    )
    fig.savefig(out_path, dpi=600)
    plt.close(fig)
    return out_path


def plot_sensitivity_heatmap(
    results_struct,
    function_names,
    parameter,
    values,
    paths,
):
    """Show raw 30-run final-fitness means colored by within-function rank."""
    means = np.asarray([
        [
            np.mean(
                np.asarray(
                    results_struct[function_name][
                        sensitivity_optimizer_label(parameter, value)
                    ]["fitness_runs"],
                    dtype=float,
                )
            )
            for function_name in function_names
        ]
        for value in values
    ])
    ranks = np.column_stack([
        rankdata(means[:, column], method="ordinal")
        for column in range(means.shape[1])
    ])

    fig, axis = plt.subplots(figsize=(10.8, 5.8), facecolor="white")
    image = axis.imshow(
        ranks,
        cmap="RdYlGn_r",
        vmin=1,
        vmax=4,
        aspect="auto",
    )
    axis.set_xticks(np.arange(len(function_names)), function_names)
    axis.set_yticks(
        np.arange(len(values)),
        [f"{parameter}={value:g}" for value in values],
    )
    axis.set_xlabel("CEC Function")
    axis.set_ylabel("Parameter value")
    axis.set_title(f"Sensitivity Summary - {parameter}")

    for row in range(means.shape[0]):
        for column in range(means.shape[1]):
            axis.text(
                column,
                row,
                f"{means[row, column]:.6e}",
                ha="center",
                va="center",
                color="black",
                fontsize=9,
            )

    colorbar = fig.colorbar(image, ax=axis, ticks=[1, 2, 3, 4])
    colorbar.set_label("Rank within CEC function (1 = best)")
    fig.tight_layout()
    out_path = os.path.join(
        paths.fig_dir,
        f"Sensitivity_summary_{parameter}_Heatmap.png",
    )
    fig.savefig(out_path, dpi=600)
    plt.close(fig)
    return out_path


def plot_sensitivity_summary_figures(
    results_struct,
    function_names,
    sensitivity_configs,
    paths,
):
    if paths.mode != "sensitivity":
        return []

    figure_paths = []
    for parameter, values in sensitivity_configs:
        figure_paths.append(
            plot_sensitivity_grouped_convergence(
                results_struct,
                function_names,
                parameter,
                values,
                paths,
            )
        )
        figure_paths.append(
            plot_sensitivity_heatmap(
                results_struct,
                function_names,
                parameter,
                values,
                paths,
            )
        )
    return figure_paths


def _sensitivity_value_legend(parameter, value):
    prefixes = {
        "mahalanobis_q": "q",
        "beta_min": "beta_min",
        "beta_max": "beta_max",
        "pcr": "pcr",
    }
    return f"{prefixes[parameter]}={value:.2f}"


def plot_sensitivity_individual_convergence(
    results_struct,
    function_name,
    parameter,
    values,
    paths,
):
    """Plot one parameter's four existing mean curves for one CEC function."""
    curves = {
        sensitivity_optimizer_label(parameter, value): results_struct[
            function_name
        ][sensitivity_optimizer_label(parameter, value)]["curve"]
        for value in values
    }
    scale = resolve_convergence_scale(curves, CONVERGENCE_SCALE)
    colors = plt.get_cmap("tab10")(np.arange(len(values)))
    fig, axis = plt.subplots(figsize=(10, 5), facecolor="white")

    for value_index, value in enumerate(values):
        label = sensitivity_optimizer_label(parameter, value)
        _plot_sensitivity_curve(
            axis,
            curves[label],
            scale,
            _sensitivity_value_legend(parameter, value),
            colors[value_index],
            LINE_STYLES[value_index % len(LINE_STYLES)],
        )
    if scale in ("log", "symlog"):
        axis.set_yscale(scale)
    axis.set_xlabel("Iteration")
    axis.set_ylabel("exp(Fitness)" if scale == "exp" else "Fitness")
    axis.set_title(f"Sensitivity Convergence - {parameter} - {function_name}")
    axis.grid(alpha=0.3)
    axis.legend()
    fig.tight_layout()

    out_path = os.path.join(
        paths.fig_dir,
        f"Sensitivity_{parameter}_{function_name}_Convergence.png",
    )
    fig.savefig(out_path, dpi=600)
    plt.close(fig)
    return out_path


def plot_sensitivity_individual_figures(
    results_struct,
    function_names,
    sensitivity_configs,
    paths,
):
    if paths.mode != "sensitivity":
        return []

    return [
        plot_sensitivity_individual_convergence(
            results_struct,
            function_name,
            parameter,
            values,
            paths,
        )
        for parameter, values in sensitivity_configs
        for function_name in function_names
    ]


def _mean_and_sample_sd(values):
    finite = np.asarray(values, dtype=float).reshape(-1)
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        return np.nan, np.nan
    mean = float(np.mean(finite))
    sd = 0.0 if finite.size == 1 else float(np.std(finite, ddof=1))
    return mean, sd


def _ablation_panel_grid(function_names):
    panel_count = max(1, len(function_names))
    ncols = min(2, panel_count)
    nrows = int(np.ceil(panel_count / ncols))
    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(7.2 * ncols, 5.2 * nrows),
        squeeze=False,
        facecolor="white",
    )
    flat_axes = axes.reshape(-1)
    for axis in flat_axes[panel_count:]:
        axis.set_visible(False)
    return fig, flat_axes


def plot_ablation_fitness_runtime_tradeoff(
    results_struct,
    function_names,
    optimizer_order,
    optimizer_colors,
    out_path,
):
    """Plot cache/run-derived mean runtime against mean final fitness."""
    fig, axes = _ablation_panel_grid(function_names)
    subplot_annotations = []
    for axis, function_name in zip(axes, function_names):
        optimizer_data = results_struct.get(function_name, {})
        plotted_points = []
        for optimizer_name in optimizer_order:
            data = optimizer_data.get(optimizer_name, {})
            mean_runtime, _ = _mean_and_sample_sd(data.get("runtime_runs", []))
            mean_fitness, _ = _mean_and_sample_sd(data.get("fitness_runs", []))
            if not (np.isfinite(mean_runtime) and np.isfinite(mean_fitness)):
                continue
            axis.scatter(
                mean_runtime,
                mean_fitness,
                s=58,
                color=optimizer_colors[optimizer_name],
                edgecolor="white",
                linewidth=0.7,
                zorder=3,
            )
            plotted_points.append((optimizer_name, mean_runtime, mean_fitness))

        # Keep a small amount of room for labels without materially changing
        # the fitness/runtime comparison.
        axis.margins(x=0.08, y=0.08)
        x_min, x_max = axis.get_xlim()
        y_min, y_max = axis.get_ylim()
        x_span = max(abs(x_max - x_min), np.finfo(float).eps)
        y_span = max(abs(y_max - y_min), np.finfo(float).eps)
        normalized_points = []
        annotations = []
        vertical_offsets = (6, -8, 18, -20, 30, -32)
        for optimizer_name, mean_runtime, mean_fitness in plotted_points:
            x_fraction = (mean_runtime - x_min) / x_span
            y_fraction = (mean_fitness - y_min) / y_span
            close_index = sum(
                abs(x_fraction - previous_x) <= 0.10
                and abs(y_fraction - previous_y) <= 0.08
                for previous_x, previous_y in normalized_points
            )
            normalized_points.append((x_fraction, y_fraction))

            if x_fraction >= 0.72:
                horizontal_direction = -1
            elif x_fraction <= 0.28:
                horizontal_direction = 1
            else:
                horizontal_direction = 1 if close_index % 2 == 0 else -1
            horizontal_offset = horizontal_direction * (
                6 + min(close_index, 3) * 2
            )
            vertical_offset = vertical_offsets[
                close_index % len(vertical_offsets)
            ]
            annotation = axis.annotate(
                display_optimizer_name(optimizer_name),
                (mean_runtime, mean_fitness),
                xytext=(horizontal_offset, vertical_offset),
                textcoords="offset points",
                ha="left" if horizontal_direction > 0 else "right",
                va="bottom" if vertical_offset >= 0 else "top",
                fontsize=8.5,
            )
            annotations.append(annotation)
        subplot_annotations.append((axis, annotations))
        axis.set_title(function_name)
        axis.set_xlabel("Mean runtime (s)")
        axis.set_ylabel("Mean final fitness")
        axis.grid(alpha=0.25)
        axis.ticklabel_format(axis="both", style="sci", scilimits=(-3, 4))
        axis.text(
            0.98,
            0.02,
            "Preferred: lower-left",
            transform=axis.transAxes,
            ha="right",
            va="bottom",
            fontsize=8,
            color="#555555",
        )
    fig.suptitle("Ablation Fitness–Runtime Trade-off", fontsize=14)
    fig.tight_layout(rect=(0, 0, 1, 0.97))

    # Resolve any remaining close-label collisions in display space and clamp
    # every label to its own axes. Only annotation offsets are adjusted here.
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    points_per_pixel = 72.0 / fig.dpi
    axes_padding = 2.0
    label_padding = 1.5
    for axis, annotations in subplot_annotations:
        axes_box = axis.get_window_extent(renderer=renderer)
        placed_boxes = []
        for annotation_index, annotation in enumerate(annotations):
            label_box = annotation.get_window_extent(renderer=renderer)
            step = label_box.height + 3.0
            direction = 1.0 if annotation_index % 2 == 0 else -1.0
            vertical_candidates = (
                0.0,
                direction * step,
                -direction * step,
                direction * 2.0 * step,
                -direction * 2.0 * step,
                direction * 3.0 * step,
                -direction * 3.0 * step,
            )
            selected_box = None
            selected_shift = (0.0, 0.0)
            for vertical_shift in vertical_candidates:
                candidate_box = label_box.translated(0.0, vertical_shift)
                horizontal_shift = 0.0
                if candidate_box.x0 < axes_box.x0 + axes_padding:
                    horizontal_shift = axes_box.x0 + axes_padding - candidate_box.x0
                elif candidate_box.x1 > axes_box.x1 - axes_padding:
                    horizontal_shift = axes_box.x1 - axes_padding - candidate_box.x1
                candidate_box = candidate_box.translated(horizontal_shift, 0.0)

                clamp_vertical_shift = 0.0
                if candidate_box.y0 < axes_box.y0 + axes_padding:
                    clamp_vertical_shift = axes_box.y0 + axes_padding - candidate_box.y0
                elif candidate_box.y1 > axes_box.y1 - axes_padding:
                    clamp_vertical_shift = axes_box.y1 - axes_padding - candidate_box.y1
                candidate_box = candidate_box.translated(0.0, clamp_vertical_shift)

                overlaps = any(
                    candidate_box.x0 < placed_box.x1 + label_padding
                    and candidate_box.x1 > placed_box.x0 - label_padding
                    and candidate_box.y0 < placed_box.y1 + label_padding
                    and candidate_box.y1 > placed_box.y0 - label_padding
                    for placed_box in placed_boxes
                )
                selected_box = candidate_box
                selected_shift = (
                    horizontal_shift,
                    vertical_shift + clamp_vertical_shift,
                )
                if not overlaps:
                    break

            offset_x, offset_y = annotation.get_position()
            annotation.set_position((
                offset_x + selected_shift[0] * points_per_pixel,
                offset_y + selected_shift[1] * points_per_pixel,
            ))
            placed_boxes.append(selected_box)

    fig.savefig(out_path, dpi=600)
    plt.close(fig)
    return out_path


def plot_ablation_runtime_comparison(
    results_struct,
    function_names,
    optimizer_order,
    optimizer_colors,
    out_path,
    reserve_annotation_headroom=False,
    show_runtime_ratio=True,
):
    """Plot mean runtime ± sample SD and the observed DE-M/DE-MC ratio."""
    fig, axes = _ablation_panel_grid(function_names)
    positions = np.arange(len(optimizer_order))
    for axis, function_name in zip(axes, function_names):
        optimizer_data = results_struct.get(function_name, {})
        means = []
        errors = []
        for optimizer_name in optimizer_order:
            mean, sd = _mean_and_sample_sd(
                optimizer_data.get(optimizer_name, {}).get("runtime_runs", [])
            )
            means.append(mean)
            errors.append(sd)
        bars = axis.bar(
            positions,
            means,
            yerr=errors,
            capsize=4,
            color=[optimizer_colors[name] for name in optimizer_order],
            edgecolor="white",
            linewidth=0.7,
        )
        axis.bar_label(bars, fmt="%.3g", padding=3, fontsize=7.5)
        if reserve_annotation_headroom:
            y_min, y_max = axis.get_ylim()
            axis.set_ylim(y_min, y_max + 0.15 * (y_max - y_min))
        axis.set_xticks(positions, [display_optimizer_name(name) for name in optimizer_order])
        axis.tick_params(axis="x", rotation=30)
        axis.set_ylabel("Runtime (s), mean ± SD")
        axis.set_title(function_name)
        axis.grid(axis="y", alpha=0.25)

        de_m_mean = means[optimizer_order.index("DE-M")]
        de_mc_mean = means[optimizer_order.index("DE-MC")]
        ratio = (
            de_m_mean / de_mc_mean
            if np.isfinite(de_m_mean) and np.isfinite(de_mc_mean) and de_mc_mean > 0.0
            else np.nan
        )
        ratio_text = (
            f"DE-M / DE-MC runtime = {ratio:.3f}×"
            if np.isfinite(ratio)
            else "DE-M / DE-MC runtime = unavailable"
        )
        if show_runtime_ratio:
            axis.text(
                0.02,
                0.97,
                ratio_text,
                transform=axis.transAxes,
                ha="left",
                va="top",
                fontsize=9,
                bbox={"facecolor": "white", "edgecolor": "#aaaaaa", "alpha": 0.9},
            )
    fig.suptitle("Ablation Runtime Comparison", fontsize=14)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(out_path, dpi=600)
    plt.close(fig)
    return out_path


def plot_ablation_individual_tradeoff_figures(
    results_struct,
    function_names,
    optimizer_order,
    optimizer_colors,
    paths,
):
    """Save one trade-off figure per function from the combined plot's data."""
    if paths.mode != "ablation":
        return []

    figure_paths = []
    for function_name in function_names:
        out_path = os.path.join(
            paths.fig_dir,
            f"Ablation_Fitness_Runtime_Tradeoff_{function_name}.png",
        )
        plot_ablation_fitness_runtime_tradeoff(
            results_struct,
            [function_name],
            optimizer_order,
            optimizer_colors,
            out_path,
        )
        figure_paths.append(out_path)
    return figure_paths


def plot_ablation_individual_runtime_figures(
    results_struct,
    function_names,
    optimizer_order,
    optimizer_colors,
    paths,
):
    """Save one runtime-comparison figure per function from the combined data."""
    if paths.mode != "ablation":
        return []

    figure_paths = []
    for function_name in function_names:
        out_path = os.path.join(
            paths.fig_dir,
            f"Ablation_Runtime_Comparison_{function_name}.png",
        )
        plot_ablation_runtime_comparison(
            results_struct,
            [function_name],
            optimizer_order,
            optimizer_colors,
            out_path,
            reserve_annotation_headroom=True,
            show_runtime_ratio=False,
        )
        figure_paths.append(out_path)
    return figure_paths


def plot_ablation_computational_figures(
    results_struct,
    function_names,
    optimizer_order,
    optimizer_colors,
    paths,
):
    if paths.mode != "ablation":
        return []
    required = {"DE-M", "DE-MC"}
    if not required.issubset(optimizer_order):
        raise ValueError("Ablation runtime comparison requires DE-M and DE-MC")
    tradeoff_path = os.path.join(
        paths.fig_dir,
        "Ablation_Fitness_Runtime_Tradeoff.png",
    )
    runtime_path = os.path.join(
        paths.fig_dir,
        "Ablation_Runtime_Comparison.png",
    )
    plot_ablation_fitness_runtime_tradeoff(
        results_struct,
        function_names,
        optimizer_order,
        optimizer_colors,
        tradeoff_path,
    )
    plot_ablation_runtime_comparison(
        results_struct,
        function_names,
        optimizer_order,
        optimizer_colors,
        runtime_path,
    )
    individual_tradeoff_paths = plot_ablation_individual_tradeoff_figures(
        results_struct,
        function_names,
        optimizer_order,
        optimizer_colors,
        paths,
    )
    individual_runtime_paths = plot_ablation_individual_runtime_figures(
        results_struct,
        function_names,
        optimizer_order,
        optimizer_colors,
        paths,
    )
    return [
        tradeoff_path,
        runtime_path,
        *individual_tradeoff_paths,
        *individual_runtime_paths,
    ]


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
        is_macro_de = is_macro_de_optimizer(optimizer_name)
        color = (
            OPTIMIZER_COLOR_MAP["MaCRO-DE"]
            if is_macro_de
            else colors[optimizer_name]
        )
        linestyle = (
            "-"
            if is_macro_de
            else LINE_STYLES[style_index % len(LINE_STYLES)]
        )
        marker = None if is_macro_de else MARKERS[style_index % len(MARKERS)]
        if is_macro_de:
            curve_ax.plot(
                iterations,
                curve,
                label="_nolegend_",
                color="black",
                linestyle="-",
                linewidth=4.0,
                alpha=0.90,
                zorder=2 + style_index - 0.1,
            )
        curve_ax.plot(
            iterations,
            curve,
            label=display_optimizer_name(optimizer_name),
            color=color,
            linestyle=linestyle,
            marker=marker,
            markevery=(
                (style_index % marker_interval, marker_interval)
                if marker is not None
                else None
            ),
            linewidth=2.5 if is_macro_de else 2.1,
            markersize=4.5,
            alpha=0.82,
            zorder=2 + style_index,
        )
        if optimizer_name != OVERLAP_DIAGNOSTIC_OPTIMIZERS[0]:
            difference_ax.plot(
                iterations,
                np.maximum(difference_curves[optimizer_name], zero_display_floor),
                label=f"|{optimizer_name} - DE-M|",
                color=color,
                linestyle=linestyle,
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
    include_run_data=False,
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

    if not include_run_data:
        df.to_excel(
            out_path,
            index=False,
        )
        return df

    runtime_rows = []
    convergence_rows = []
    for function_name, optimizer_data in results_struct.items():
        for optimizer_name, data in optimizer_data.items():
            runtimes = np.asarray(data["runtime_runs"], dtype=float)
            runtime_statistics = (
                ("BEST", np.min(runtimes)),
                ("WORST", np.max(runtimes)),
                ("MEAN", np.mean(runtimes)),
                ("SD", np.std(runtimes)),
            )
            for statistic, runtime in runtime_statistics:
                runtime_rows.append({
                    "Function": function_name,
                    "Optimizer": optimizer_name,
                    "Statistic": statistic,
                    "Runtime": runtime,
                })
            for iteration, fitness in enumerate(data["curve"], start=1):
                convergence_rows.append({
                    "Function": function_name,
                    "Optimizer": optimizer_name,
                    "Iteration": iteration,
                    "Mean Fitness": fitness,
                })

    with pd.ExcelWriter(out_path) as writer:
        df.to_excel(writer, sheet_name="Fitness", index=False)
        pd.DataFrame(runtime_rows).to_excel(
            writer,
            sheet_name="Runtime",
            index=False,
        )
        pd.DataFrame(convergence_rows).to_excel(
            writer,
            sheet_name="Convergence",
            index=False,
        )

    return df


def _final_fitness_stats(values):
    values = np.asarray(values, dtype=float).reshape(-1)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return {
            "Best": np.nan,
            "Worst": np.nan,
            "Mean": np.nan,
            "Std": np.nan,
        }
    return {
        "Best": float(np.min(values)),
        "Worst": float(np.max(values)),
        "Mean": float(np.mean(values)),
        "Std": 0.0 if values.size == 1 else float(np.std(values, ddof=1)),
    }


def export_statistical_results(
    results_struct,
    function_names,
    optimizer_order,
    out_path,
):
    statistics = ["Best", "Worst", "Mean", "Std"]
    index = pd.MultiIndex.from_product(
        [optimizer_order, statistics],
        names=["Optimizer", "Statistic"],
    )
    fitness = pd.DataFrame(
        np.nan,
        index=index,
        columns=pd.Index(function_names, name="CEC Function"),
        dtype=float,
    )

    for function_name in function_names:
        optimizer_data = results_struct.get(function_name, {})
        for optimizer_name in optimizer_order:
            data = optimizer_data.get(optimizer_name, {})
            stats = _final_fitness_stats(data.get("fitness_runs", []))
            for statistic in statistics:
                fitness.loc[(optimizer_name, statistic), function_name] = stats[statistic]

    with pd.ExcelWriter(out_path) as writer:
        fitness.to_excel(writer, sheet_name="Fitness")
    return fitness


def _holm_adjusted_pvalues(p_values):
    raw = np.asarray(p_values, dtype=float)
    if raw.size == 0:
        return raw
    order = np.argsort(raw)
    sorted_raw = raw[order]
    multipliers = np.arange(raw.size, 0, -1, dtype=float)
    sorted_adjusted = np.minimum(
        1.0,
        np.maximum.accumulate(sorted_raw * multipliers),
    )
    adjusted = np.empty_like(sorted_adjusted)
    adjusted[order] = sorted_adjusted
    return adjusted


def build_friedman_fitness_matrix(
    results_struct,
    function_names,
    optimizer_order,
):
    matrix = pd.DataFrame(
        np.nan,
        index=pd.Index(function_names, name="CEC Function"),
        columns=pd.Index(optimizer_order, name="Optimizer"),
        dtype=float,
    )
    for function_name in function_names:
        optimizer_data = results_struct.get(function_name, {})
        for optimizer_name in optimizer_order:
            values = np.asarray(
                optimizer_data.get(optimizer_name, {}).get("fitness_runs", []),
                dtype=float,
            ).reshape(-1)
            values = values[np.isfinite(values)]
            if values.size:
                matrix.loc[function_name, optimizer_name] = float(np.mean(values))
    return matrix


def calculate_friedman_analysis(
    fitness_matrix,
    mode,
    alpha=0.05,
):
    mode_label = str(mode).upper()
    complete = fitness_matrix.dropna(axis=0, how="any")
    optimizers = list(fitness_matrix.columns)
    enough_data = len(optimizers) >= 3 and complete.shape[0] >= 2

    rank_matrix = pd.DataFrame(
        index=complete.index,
        columns=optimizers,
        dtype=float,
    )
    for block_name, row in complete.iterrows():
        rank_matrix.loc[block_name] = rankdata(
            row.to_numpy(dtype=float),
            method="average",
        )
    average_ranks = rank_matrix.mean(axis=0)

    statistic = np.nan
    p_value = np.nan
    significant = False
    conclusion = "Insufficient complete blocks or optimizers"
    if enough_data:
        samples = [complete[name].to_numpy(dtype=float) for name in optimizers]
        statistic, p_value = friedmanchisquare(*samples)
        statistic = float(statistic)
        p_value = float(p_value)
        significant = bool(np.isfinite(p_value) and p_value < alpha)
        conclusion = (
            "Significant differences detected"
            if significant
            else "No significant differences detected"
        )

    finite_ranks = average_ranks.dropna()
    if finite_ranks.empty:
        best_rank = np.nan
        best_optimizers = []
    else:
        best_rank = float(finite_ranks.min())
        best_optimizers = finite_ranks.index[
            np.isclose(finite_ranks, best_rank)
        ].tolist()

    summary = pd.DataFrame([{
        "Mode": mode_label,
        "Metric": "Final fitness (lower is better)",
        "Aggregation": "Mean final fitness across independent runs per CEC function and optimizer",
        "Configured optimizers": len(optimizers),
        "Available blocks": int(fitness_matrix.shape[0]),
        "Complete blocks used": int(complete.shape[0]),
        "Friedman statistic": statistic,
        "p-value": p_value,
        "Alpha": float(alpha),
        "Significant": "YES" if significant else "NO",
        "Conclusion": conclusion,
        "Best average rank optimizer(s)": ", ".join(best_optimizers),
        "Best average rank": best_rank,
        "Post-hoc method": (
            "Pairwise Wilcoxon signed-rank with Holm correction"
            if significant
            else "Not performed"
        ),
    }])
    ranks = pd.DataFrame({
        "Optimizer": optimizers,
        "Average rank": [float(average_ranks.get(name, np.nan)) for name in optimizers],
        "Best average rank": ["YES" if name in best_optimizers else "NO" for name in optimizers],
        "Complete blocks used": int(complete.shape[0]),
    }).sort_values(
        ["Average rank", "Optimizer"],
        na_position="last",
        ignore_index=True,
    )

    posthoc_columns = [
        "Optimizer A",
        "Optimizer B",
        "Wilcoxon statistic",
        "Raw p-value",
        "Holm adjusted p-value",
        "Significant at alpha=0.05",
        "Better average rank",
        "Note",
    ]
    posthoc_rows = []
    if significant:
        raw_p_values = []
        pair_results = []
        for left_index, left in enumerate(optimizers[:-1]):
            for right in optimizers[left_index + 1:]:
                left_values = complete[left].to_numpy(dtype=float)
                right_values = complete[right].to_numpy(dtype=float)
                if np.array_equal(left_values, right_values):
                    pair_statistic, pair_p = 0.0, 1.0
                else:
                    pair_statistic, pair_p = wilcoxon(
                        left_values,
                        right_values,
                        alternative="two-sided",
                        method="auto",
                    )
                pair_results.append(
                    (left, right, float(pair_statistic), float(pair_p))
                )
                raw_p_values.append(float(pair_p))

        adjusted_p_values = _holm_adjusted_pvalues(raw_p_values)
        for result, adjusted_p in zip(pair_results, adjusted_p_values):
            left, right, pair_statistic, pair_p = result
            left_rank = float(average_ranks[left])
            right_rank = float(average_ranks[right])
            better_rank = (
                "TIE"
                if np.isclose(left_rank, right_rank)
                else (left if left_rank < right_rank else right)
            )
            posthoc_rows.append({
                "Optimizer A": left,
                "Optimizer B": right,
                "Wilcoxon statistic": pair_statistic,
                "Raw p-value": pair_p,
                "Holm adjusted p-value": float(adjusted_p),
                "Significant at alpha=0.05": "YES" if adjusted_p < alpha else "NO",
                "Better average rank": better_rank,
                "Note": "",
            })
    else:
        posthoc_rows.append({
            "Significant at alpha=0.05": "NO",
            "Note": (
                "Post-hoc not performed because the Friedman test was not "
                "significant or lacked sufficient data."
            ),
        })

    return {
        "summary": summary,
        "ranks": ranks,
        "posthoc": pd.DataFrame(posthoc_rows, columns=posthoc_columns),
        "blocks": fitness_matrix,
        "block_ranks": rank_matrix,
    }


def export_friedman_analysis(
    results_struct,
    function_names,
    optimizer_order,
    mode,
    out_path,
    alpha=0.05,
):
    mode_label = str(mode).upper()
    matrix = build_friedman_fitness_matrix(
        results_struct,
        function_names,
        optimizer_order,
    )
    analysis = calculate_friedman_analysis(
        matrix,
        mode_label,
        alpha=alpha,
    )
    with pd.ExcelWriter(out_path) as writer:
        analysis["summary"].to_excel(
            writer,
            sheet_name=f"{mode_label}_Friedman",
            index=False,
        )
        analysis["ranks"].to_excel(
            writer,
            sheet_name=f"{mode_label}_Average_Ranks",
            index=False,
        )
        analysis["posthoc"].to_excel(
            writer,
            sheet_name=f"{mode_label}_PostHoc_Holm",
            index=False,
        )
        analysis["blocks"].to_excel(
            writer,
            sheet_name=f"{mode_label}_Block_Fitness",
        )
        analysis["block_ranks"].to_excel(
            writer,
            sheet_name=f"{mode_label}_Block_Ranks",
        )
    return analysis

def validate_optimizer_list(args):
    """Fail-fast audit for every optimizer selected in this experiment mode."""
    print("\nOPTIMIZER STARTUP AUDIT")
    invalid = []
    width = max((len(str(name)) for name in args.optimizers), default=1)
    for optimizer_name in args.optimizers:
        try:
            resolved_name = resolve_optimizer_name(optimizer_name)
            resolve_optimizer_class(optimizer_name)
            if (
                args.compute_device == "gpu"
                and not supports_gpu_batching(optimizer_name)
                and not supports_mealpy_gpu_adapter(resolved_name)
            ):
                raise ValueError(
                    f"no strict GPU adapter for resolved optimizer {resolved_name!r}"
                )
        except (ImportError, AttributeError, TypeError, ValueError) as exc:
            invalid.append((optimizer_name, str(exc)))
            print(f"{optimizer_name:<{width}} : INVALID / unknown optimizer")
        else:
            print(f"{optimizer_name:<{width}} : VALID")
    if invalid:
        invalid_names = ", ".join(str(name) for name, _ in invalid)
        details = "; ".join(
            f"{name}: {message}" for name, message in invalid
        )
        raise ValueError(
            f"Invalid optimizer(s): {invalid_names}. No experiments started. {details}"
        )


def audit_checkpoint_cache(args, paths, source_paths, selected_functions):
    """Inspect all selected runs before computation and return compatible paths."""
    compatible_paths = {}
    search_paths = [paths]
    if source_paths is not None and source_paths.cache_dir != paths.cache_dir:
        search_paths.append(source_paths)
    configurations = list(optimizer_experiment_configurations(args))
    label_width = max((len(label) for label, _, _ in configurations), default=1)

    print("\nCACHE STARTUP AUDIT")
    for function_name in selected_functions:
        print(f"Function {function_name}")
        for optimizer_label, optimizer_name, optimizer_args in configurations:
            signature = build_cache_signature(optimizer_args, optimizer_name)
            cached_count = 0
            for run in range(optimizer_args.runs):
                seed = optimizer_args.seed_base + run
                current_path = run_checkpoint_path(
                    paths,
                    signature,
                    function_name,
                    optimizer_name,
                    run,
                )
                metadata = checkpoint_metadata(
                    optimizer_args,
                    signature,
                    function_name,
                    optimizer_name,
                    run,
                    seed,
                )
                matched_path = compatible_checkpoint_path(
                    search_paths,
                    current_path,
                    metadata,
                    function_name,
                    optimizer_name,
                    run,
                )
                compatible_paths[(function_name, optimizer_label, run)] = matched_path
                cached_count += matched_path is not None

            missing_count = optimizer_args.runs - cached_count
            if missing_count == 0:
                cache_status = "COMPLETE"
            elif cached_count == 0:
                cache_status = "NEW"
            else:
                cache_status = f"RESUME {missing_count}"
            print(
                f"{optimizer_label:<{label_width}} : "
                f"{cached_count}/{optimizer_args.runs} cached | {cache_status}"
            )
    return compatible_paths


def run_experiment(args):

    if args.compute_device == "gpu" and args.objective_evaluation == "process":
        raise ValueError(
            'Strict GPU RAM-safe mode supports objective_evaluation="auto" '
            'or "serial" only; "process" would create nested CPU worker processes.'
        )
    if args.compute_device == "gpu" and args.cec_objective_backend not in {"auto", "gpu"}:
        raise ValueError(
            "Strict GPU execution requires --cec-objective-backend auto or gpu; "
            "opfunu/numpy CPU objective routing is forbidden"
        )
    # Cache inspection precedes worker pools, GPU initialization, objective
    # verification, and every optimizer solve.
    args.resolved_gpu_batch_size = 1
    args.estimated_gpu_batch_capacity = 1
    paths = make_paths(args)
    source_paths = None
    if args.reuse_cache and args.reuse_cache_from_exp_id is not None:
        source_args = argparse.Namespace(**vars(args))
        source_args.exp_id = args.reuse_cache_from_exp_id
        source_paths = make_paths(source_args, create=False)
    optimizer_order = comparison_optimizer_order(args)
    optimizer_colors = build_optimizer_colors(optimizer_order)
    function_map = discover_benchmark_functions(
        args.benchmark,
        args.dims,
    )
    args.function_map = function_map
    selected_functions = select_experiment_functions(args, function_map)
    compatible_paths = audit_checkpoint_cache(
        args,
        paths,
        source_paths,
        selected_functions,
    )

    if args.compute_device == "cpu":
        # CPU batching is the outer numerical workload in the controller, just
        # as independent runs are the outer workload in child processes.
        initialize_objective_worker()

    gpu_info = None
    strict_gpu_executor = None
    batched_gpu_executor = None
    if args.compute_device in GPU_MODES:
        gpu_info = initialize_gpu(memory_fraction=args.gpu_memory_fraction)
        if args.compute_device == "gpu":
            args.gpu_workers = select_strict_gpu_workers(
                args, gpu_info, selected_functions
            )
            per_worker_memory_fraction = args.gpu_memory_fraction / args.gpu_workers
        else:
            per_worker_memory_fraction = args.gpu_memory_fraction
        # BatchedDEEngine uses one full-budget CUDA context even when the
        # scalar independent-run pool contains multiple contexts.
        batch_memory_fraction = (
            args.gpu_memory_fraction
            if args.compute_device == "gpu"
            else per_worker_memory_fraction
        )
        memory_budget = min(
            gpu_info.free_memory_bytes,
            int(gpu_info.total_memory_bytes * batch_memory_fraction),
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
    if args.compute_device == "gpu":
        # Strict-GPU objectives are constructed only after spawn, beside the
        # worker's local CUDA context.  No CuPy object crosses this boundary.
        args.gpu_objectives = {}
        args.gpu_objective_reports = {}
        args.vectorized_cpu_objectives = {}
    elif args.compute_device == "hybrid":
        initialize_verified_gpu_objectives(args, selected_functions)
    else:
        # CPU verification is lazy so cache-only resumes do no objective work.
        args.gpu_objectives = {}
        args.gpu_objective_reports = {}
        args.vectorized_cpu_objectives = {}

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
        f"Optimizers     : {optimizer_order}"
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
        "Cache source   : "
        + (f"{source_paths.exp_tag} (read only)" if source_paths is not None else "CURRENT ONLY")
    )
    print(
        f"Compute device : {args.compute_device.upper()}"
    )
    print(
        f"CPU workers    : {args.n_workers}"
    )
    if gpu_info is not None:
        print("GPU scheduling          : throughput-optimized")
        print(f"GPU worker policy       : {args.gpu_worker_policy.upper()}")
        print(f"GPU workers selected    : {args.gpu_workers}")
        print(f"GPU batch policy        : {'AUTO' if args.gpu_batch_size == 'auto' else 'FIXED'}")
        print(f"Effective MaCRO batch   : {args.resolved_gpu_batch_size}")
        print(f"Host planner workers    : {args.objective_workers}")
        if args.experiment_mode == "sensitivity":
            print(f"Sensitivity groups      : {len(args.sensitivity_configs)}")
            print(
                "Sensitivity configs     : "
                f"{sum(len(values) for _, values in args.sensitivity_configs)}"
            )
        if args.compute_device == "gpu":
            print("Strict GPU execution path: local persistent CUDA workers")
            print("Host RAM safety: enabled")
            print(
                "Per-worker GPU memory cap: "
                f"{args.gpu_memory_fraction / args.gpu_workers:.1%}"
            )
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
        if args.compute_device == "gpu":
            print("GPU objective verification: worker-local")
            print("CPU objective fallback     : FORBIDDEN")
        else:
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
    print("Cache identity : per optimizer/run")

    objective_executor = None
    if (
        (args.compute_device == "hybrid" or cpu_batching_enabled(args))
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
    mode_had_pending_runs = False

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
        for optimizer_index, optimizer_configuration in enumerate(
            optimizer_experiment_configurations(args),
            start=1,
        ):
            optimizer_label, optimizer_name, optimizer_args = optimizer_configuration
            optimizer_cache_signature = build_cache_signature(
                optimizer_args,
                optimizer_name,
            )

            try:
                print_status(
                    f"OPTIMIZER {optimizer_index}/{len(optimizer_order)} | "
                    f"function={function_name} | "
                    f"optimizer={optimizer_label}"
                )
                fitness_runs = []
                runtime_runs = []
                curves = []
                completed = []
                pending_runs = []
                checkpoint_records = {}

                resolved_optimizer = resolve_optimizer_name(optimizer_name)
                if (
                    optimizer_args.compute_device in GPU_MODES
                    and resolved_optimizer in CUSTOM_OPTIMIZERS
                    and not supports_gpu_batching(optimizer_name)
                ):
                    print_status(
                        f"STRICT GPU SCALAR ADAPTER | optimizer={optimizer_name} | "
                        "using the GPU-backed MEALPY solve lifecycle"
                    )

                for run in range(optimizer_args.runs):
                    seed = optimizer_args.seed_base + run
                    checkpoint_path = run_checkpoint_path(
                        paths,
                        optimizer_cache_signature,
                        function_name,
                        optimizer_name,
                        run,
                    )
                    metadata = checkpoint_metadata(
                        optimizer_args,
                        optimizer_cache_signature,
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
                    if optimizer_args.reuse_cache:
                        matched_path = compatible_paths.get(
                            (function_name, optimizer_label, run)
                        )
                        if matched_path is not None:
                            cached_output = load_run_checkpoint(
                                matched_path,
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
                            f"optimizer={optimizer_label} | "
                            f"run={run + 1}/{optimizer_args.runs}"
                        )

                if pending_runs:
                    mode_had_pending_runs = True
                    print_status(
                        f"CACHE MISS | function={function_name} | "
                        f"optimizer={optimizer_label} | "
                        f"runs={len(pending_runs)}/{optimizer_args.runs}"
                    )

                if len(pending_runs) == 0:
                    print_status(
                        f"CACHE COMPLETE | function={function_name} | "
                        f"optimizer={optimizer_label} | "
                        f"runs={optimizer_args.runs}/{optimizer_args.runs}"
                    )

                if (
                    pending_runs
                    and cpu_batching_enabled(optimizer_args)
                    and supports_gpu_batching(optimizer_name)
                ):
                    if function_name not in optimizer_args.vectorized_cpu_objectives:
                        initialize_verified_gpu_objectives(optimizer_args, [function_name])
                    actual_batch_size = len(pending_runs)
                    print_status(
                        f"CPU BATCHING | function={function_name} | "
                        f"optimizer={optimizer_name} | pending_runs={len(pending_runs)} | "
                        f"run_batch_size={actual_batch_size}"
                    )
                    completed.extend(
                        execute_gpu_batches(
                            function_name,
                            optimizer_name,
                            optimizer_args,
                            pending_runs,
                            checkpoint_records,
                            objective_executor=objective_executor,
                            active_batch_size=actual_batch_size,
                        )
                    )
                elif (
                    pending_runs
                    and optimizer_args.compute_device == "gpu"
                    and optimizer_uses_gpu(optimizer_name, optimizer_args.compute_device)
                ):
                    if supports_gpu_batching(optimizer_name):
                        if batched_gpu_executor is None:
                            batched_gpu_executor = ProcessPoolExecutor(
                                max_workers=1,
                                mp_context=multiprocessing.get_context("spawn"),
                                initializer=initialize_strict_gpu_worker,
                                initargs=(optimizer_args.gpu_memory_fraction,),
                            )
                        selected_gpu_executor = batched_gpu_executor
                    else:
                        if strict_gpu_executor is None:
                            strict_gpu_executor = ProcessPoolExecutor(
                                max_workers=optimizer_args.gpu_workers,
                                mp_context=multiprocessing.get_context("spawn"),
                                initializer=initialize_strict_gpu_worker,
                                initargs=(
                                    optimizer_args.gpu_memory_fraction
                                    / optimizer_args.gpu_workers,
                                ),
                            )
                        selected_gpu_executor = strict_gpu_executor
                    active_gpu_workers = (
                        1 if supports_gpu_batching(optimizer_name)
                        else min(optimizer_args.gpu_workers, len(pending_runs))
                    )
                    print_status(
                        f"STRICT GPU POOL | function={function_name} | "
                        f"optimizer={optimizer_name} | pending_runs={len(pending_runs)} | "
                        f"gpu_workers={active_gpu_workers}"
                    )
                    while True:
                        try:
                            gpu_outputs = execute_strict_gpu_pool(
                                selected_gpu_executor,
                                function_name,
                                optimizer_name,
                                optimizer_args,
                                pending_runs,
                                checkpoint_records,
                            )
                            break
                        except Exception as exc:
                            recoverable = (
                                not supports_gpu_batching(optimizer_name)
                                and optimizer_args.gpu_worker_policy == "auto"
                                and optimizer_args.gpu_workers > 1
                                and is_gpu_out_of_memory(exc)
                            )
                            if not recoverable:
                                raise
                            old_workers = optimizer_args.gpu_workers
                            optimizer_args.gpu_workers = max(1, old_workers // 2)
                            strict_gpu_executor.shutdown(wait=True, cancel_futures=True)
                            strict_gpu_executor = ProcessPoolExecutor(
                                max_workers=optimizer_args.gpu_workers,
                                mp_context=multiprocessing.get_context("spawn"),
                                initializer=initialize_strict_gpu_worker,
                                initargs=(
                                    optimizer_args.gpu_memory_fraction
                                    / optimizer_args.gpu_workers,
                                ),
                            )
                            selected_gpu_executor = strict_gpu_executor
                            print_status(
                                f"GPU WORKER OOM | workers={old_workers} -> "
                                f"{optimizer_args.gpu_workers} | retrying unchanged runs"
                            )
                    completed.extend(gpu_outputs)
                elif pending_runs and optimizer_uses_gpu(optimizer_name, optimizer_args.compute_device):
                    actual_batch_size = min(optimizer_args.resolved_gpu_batch_size, len(pending_runs))
                    objective_backends = {}
                    if optimizer_args.gpu_batch_size == "auto" and optimizer_args.gpu_auto_calibration == "yes":
                        actual_batch_size, selected_backend, _ = calibrate_gpu_batch_size(
                            function_name,
                            optimizer_name,
                            optimizer_args,
                            min(optimizer_args.estimated_gpu_batch_capacity, len(pending_runs)),
                            objective_executor,
                        )
                        objective_backends[actual_batch_size] = selected_backend
                    print_status(
                        f"GPU BATCHING | function={function_name} | "
                        f"optimizer={optimizer_name} | pending_runs={len(pending_runs)} | "
                        f"run_batch_size={actual_batch_size} | gpu_workers=1"
                    )
                    completed.extend(
                        execute_gpu_batches(
                            function_name,
                            optimizer_name,
                            optimizer_args,
                            pending_runs,
                            checkpoint_records,
                            objective_executor=objective_executor,
                            active_batch_size=actual_batch_size,
                            objective_backends=objective_backends,
                        )
                    )
                elif should_use_cpu_parallel_executor(
                    optimizer_name,
                    optimizer_args,
                    pending_runs,
                ):

                    tasks = []
                    worker_args = cpu_worker_args(optimizer_args)

                    for run in pending_runs:
                        checkpoint_path, metadata = checkpoint_records[
                            run
                        ]

                        tasks.append({
                            "run": run,
                            "function_name": function_name,
                            "optimizer_name": optimizer_name,
                            "args": worker_args,
                            "seed": optimizer_args.seed_base + run,
                            "total_runs": optimizer_args.runs,
                            "checkpoint_path": checkpoint_path,
                            "metadata": metadata,
                        })

                    active_cpu_workers = min(optimizer_args.n_workers, len(tasks))
                    print_status(
                        f"SUBMITTED | function={function_name} | "
                        f"optimizer={optimizer_name} | "
                        f"runs={len(tasks)} | workers={active_cpu_workers}"
                    )

                    executor_kwargs = {
                        "max_workers": active_cpu_workers,
                        # Each independent run already supplies the outer
                        # parallelism.  Keep small covariance/Cholesky BLAS
                        # calls single-threaded to avoid nested fan-out across
                        # run workers.
                        "initializer": initialize_objective_worker,
                    }
                    if optimizer_args.compute_device in GPU_MODES:
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
                                f"optimizer={optimizer_label} | "
                                f"completed_runs={len(completed)}/{optimizer_args.runs}"
                            )

                else:
                    if pending_runs and optimizer_args.compute_device == "gpu":
                        raise RuntimeError(
                            f"Strict GPU routing failed for {optimizer_name}; "
                            "CPU fallback is forbidden"
                        )
                    for run in pending_runs:
                        checkpoint_path, metadata = checkpoint_records[
                            run
                        ]
                        output = run_single(
                            function_name,
                            optimizer_name,
                            optimizer_args,
                            optimizer_args.seed_base + run,
                            run,
                            optimizer_args.runs,
                        )

                        save_run_checkpoint(
                            checkpoint_path,
                            metadata,
                            output,
                        )
                        print_status(
                            f"CHECKPOINT SAVED | function={function_name} | "
                            f"optimizer={optimizer_label} | "
                            f"run={run + 1}/{optimizer_args.runs} | "
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

                if len(completed) != optimizer_args.runs:
                    raise RuntimeError(
                        f"optimizer produced {len(completed)}/{optimizer_args.runs} completed runs"
                    )
                curve_lengths = {len(np.asarray(curve).reshape(-1)) for curve in curves}
                if curve_lengths != {optimizer_args.epochs}:
                    raise RuntimeError(
                        f"invalid convergence lengths: {sorted(curve_lengths)}; expected {optimizer_args.epochs}"
                    )
                mean_curve = np.mean(np.stack(curves, axis=0), axis=0)

                curves_plot[
                    optimizer_label
                ] = mean_curve

                results_struct[
                    function_name
                ][optimizer_label] = {

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
                    f"optimizer={optimizer_label} | "
                    f"exception={type(exc).__name__}: {exc}"
                )
                traceback.print_exc()
                optimizer_failures.append(
                    (function_name, optimizer_label, type(exc).__name__, str(exc))
                )
                continue


        missing_optimizers = [name for name in optimizer_order if name not in curves_plot]
        if missing_optimizers:
            for missing_optimizer in missing_optimizers:
                print_status(
                    f"PLOT WARNING | {function_name} | missing optimizer {missing_optimizer}"
                )
            print_status(
                f"PLOT DEFERRED | {function_name} | incomplete optimizer set"
            )
        elif len(curves_plot) > 0:
            original_curves = (
                {
                    optimizer_name: curve
                    for optimizer_name, curve in curves_plot.items()
                    if optimizer_name != FULL_TEST_OPTIMIZER
                }
                if args.experiment_mode == "full"
                else curves_plot
            )
            audit_convergence_curves(original_curves, function_name)
            main_scale = plot_configured_convergence(
                original_curves,
                function_name,
                paths,
                optimizer_colors,
                CONVERGENCE_SCALE,
                CONVERGENCE_SHOW_MARKERS,
                CONVERGENCE_USE_LINE_STYLES,
            )
            if args.experiment_mode == "full":
                plot_full_test_convergence(
                    curves_plot,
                    function_name,
                    paths,
                    optimizer_colors,
                    CONVERGENCE_SCALE,
                    CONVERGENCE_SHOW_MARKERS,
                    CONVERGENCE_USE_LINE_STYLES,
                )
                plot_internal_macro_de_t_as_main(
                    curves_plot,
                    function_name,
                    paths,
                    optimizer_colors,
                    CONVERGENCE_SCALE,
                    CONVERGENCE_SHOW_MARKERS,
                    CONVERGENCE_USE_LINE_STYLES,
                )
            if args.convergence_extra_scale != "none":
                extra_scale = resolve_convergence_scale(
                    original_curves,
                    args.convergence_extra_scale,
                )
                if extra_scale is not None and extra_scale != main_scale:
                    plot_configured_convergence(
                        original_curves,
                        function_name,
                        paths,
                        optimizer_colors,
                        extra_scale,
                        CONVERGENCE_SHOW_MARKERS,
                        CONVERGENCE_USE_LINE_STYLES,
                    )
                    if args.experiment_mode == "full":
                        plot_full_test_convergence(
                            curves_plot,
                            function_name,
                            paths,
                            optimizer_colors,
                            extra_scale,
                            CONVERGENCE_SHOW_MARKERS,
                            CONVERGENCE_USE_LINE_STYLES,
                        )

    if objective_executor is not None:
        objective_executor.shutdown(wait=True)
    if strict_gpu_executor is not None:
        strict_gpu_executor.shutdown(wait=True)
    if batched_gpu_executor is not None:
        batched_gpu_executor.shutdown(wait=True)

    sensitivity_figure_paths = []
    if args.experiment_mode == "sensitivity":
        sensitivity_figure_paths = plot_sensitivity_summary_figures(
            results_struct,
            selected_functions,
            args.sensitivity_configs,
            paths,
        )
        sensitivity_figure_paths.extend(
            plot_sensitivity_individual_figures(
                results_struct,
                selected_functions,
                args.sensitivity_configs,
                paths,
            )
        )

    ablation_figure_paths = plot_ablation_computational_figures(
        results_struct,
        selected_functions,
        optimizer_order,
        optimizer_colors,
        paths,
    )

    mode_label = args.experiment_mode.capitalize()
    excel_path = os.path.join(
        paths.res_dir,
        f"Global_Results_{paths.exp_tag}_{mode_label}.xlsx",
    )

    export_results(
        results_struct,
        excel_path,
        include_run_data=(args.experiment_mode == "sensitivity"),
    )

    statistical_excel_path = os.path.join(
        paths.res_dir,
        f"Statistical_Results_{paths.exp_tag}_{mode_label}.xlsx",
    )
    export_statistical_results(
        results_struct,
        selected_functions,
        optimizer_order,
        statistical_excel_path,
    )

    friedman_excel_path = os.path.join(
        paths.res_dir,
        f"Friedman_Analysis_{paths.exp_tag}_{mode_label}.xlsx",
    )
    export_friedman_analysis(
        results_struct,
        selected_functions,
        optimizer_order,
        args.experiment_mode,
        friedman_excel_path,
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
    print(f"Statistical results: {statistical_excel_path}")
    print(f"Friedman analysis: {friedman_excel_path}")
    for figure_path in sensitivity_figure_paths:
        print(f"Sensitivity figure: {figure_path}")
    for figure_path in ablation_figure_paths:
        print(f"Ablation figure: {figure_path}")
    if args.reuse_cache and not mode_had_pending_runs:
        print_status(
            f"CACHE MODE COMPLETE | mode={args.experiment_mode} | "
            "all optimization runs reused"
        )

def experiment_configurations(args):

    for experiment_mode in args.experiment_modes:
        mode_args = argparse.Namespace(**vars(args))
        apply_experiment_mode(mode_args, experiment_mode)
        yield mode_args

def main():

    args = parse_args()
    if args.internal_convergence_from_cache_only:
        apply_experiment_mode(args, "full")
        run_internal_convergence_from_cache(args)
        return
    if args.overlap_diagnostic_function is not None:
        diagnostic_args = argparse.Namespace(**vars(args))
        apply_experiment_mode(diagnostic_args, args.experiment_modes[0])
        run_overlap_diagnostic(diagnostic_args)
        return

    configurations = list(experiment_configurations(args))
    for mode_args in configurations:
        print_status(
            f"STARTUP AUDIT | mode={mode_args.experiment_mode}"
        )
        validate_optimizer_list(mode_args)
    for config_index, mode_args in enumerate(configurations, start=1):
        print_status(
            f"EXPERIMENT CONFIGURATION {config_index}/{len(configurations)} | "
            f"mode={mode_args.experiment_mode}"
        )
        run_experiment(mode_args)

if __name__ == "__main__":

    main()
