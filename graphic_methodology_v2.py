# ============================================================
# graphic_methodology_v4.py
# ============================================================

import warnings
warnings.filterwarnings("ignore",category=UserWarning)

import os
import re
import logging
import inspect
import importlib

import numpy as np
import matplotlib.pyplot as plt

from scipy.stats import chi2
from scipy.linalg import cholesky

from mealpy import FloatVar

from macro_de_optimizer_v2 import MaCRO_DE


# ============================================================
# DISABLE LOGS
# ============================================================

logging.disable(logging.INFO)


# ============================================================
# CONFIGURATION
# ============================================================

EXPERIMENT_ID="EXP001"

OUTPUT_FOLDER="Methodology_Graphics"

BENCHMARK="CEC2017"

DIMENSIONS=10

POP_SIZE=50

EPOCHS=100

RUNS=20

RANDOM_SEED=42

BETA_MIN=0.2

BETA_MAX=0.8

PCR=0.2

MAHALANOBIS_Q=0.68

PLOT_MAHALANOBIS=True

SAVE_CONVERGENCE=True

np.random.seed(RANDOM_SEED)


# ============================================================
# OUTPUT DIRECTORIES
# ============================================================

BASE_OUTPUT=os.path.join(
    OUTPUT_FOLDER,
    EXPERIMENT_ID,
)

CONVERGENCE_DIR=os.path.join(
    BASE_OUTPUT,
    "Convergence",
)

MAHALANOBIS_DIR=os.path.join(
    BASE_OUTPUT,
    "Mahalanobis",
)

os.makedirs(
    CONVERGENCE_DIR,
    exist_ok=True,
)

os.makedirs(
    MAHALANOBIS_DIR,
    exist_ok=True,
)


# ============================================================
# AVAILABLE BENCHMARKS
# ============================================================

AVAILABLE_BENCHMARKS={

    "CEC2005":"opfunu.cec_based.cec2005",
    "CEC2008":"opfunu.cec_based.cec2008",
    "CEC2010":"opfunu.cec_based.cec2010",
    "CEC2013":"opfunu.cec_based.cec2013",
    "CEC2014":"opfunu.cec_based.cec2014",
    "CEC2015":"opfunu.cec_based.cec2015",
    "CEC2017":"opfunu.cec_based.cec2017",
    "CEC2019":"opfunu.cec_based.cec2019",
    "CEC2020":"opfunu.cec_based.cec2020",
    "CEC2021":"opfunu.cec_based.cec2021",
    "CEC2022":"opfunu.cec_based.cec2022",
}


# ============================================================
# DISCOVER FUNCTIONS
# ============================================================

def discover_functions(benchmark_name):

    module_path=AVAILABLE_BENCHMARKS[
        benchmark_name
    ]

    module=importlib.import_module(
        module_path
    )

    function_map={}

    for name,obj in inspect.getmembers(module):

        if inspect.isclass(obj):

            if re.match(r"^F\d+",name):

                function_map[name]=obj

    function_map=dict(
        sorted(
            function_map.items()
        )
    )

    return function_map


# ============================================================
# SAFE FUNCTION CREATION
# ============================================================

def create_function(function_class):

    try:

        benchmark=function_class(
            ndim=DIMENSIONS
        )

        return benchmark

    except Exception:

        return None


# ============================================================
# MAHALANOBIS DISTANCE
# ============================================================

def mahalanobis_distance(points):

    mean=np.mean(
        points,
        axis=0,
    )

    covariance=np.cov(
        points.T
    )

    covariance+=(
        np.eye(
            covariance.shape[0]
        )*1e-8
    )

    inv_cov=np.linalg.inv(
        covariance
    )

    centered=points-mean

    distances=[]

    for point in centered:

        d=np.sqrt(
            point.T@inv_cov@point
        )

        distances.append(d)

    distances=np.array(
        distances
    )

    return (

        distances,

        mean,

        covariance,
    )


# ============================================================
# ELLIPSOID
# ============================================================

def generate_ellipsoid(

    mean,

    covariance,

    q=0.68,

    resolution=40,
):

    ndim=covariance.shape[0]

    radius=np.sqrt(
        chi2.ppf(q,ndim)
    )

    L=cholesky(
        covariance,
        lower=True,
    )

    u=np.linspace(
        0,
        2*np.pi,
        resolution,
    )

    v=np.linspace(
        0,
        np.pi,
        resolution,
    )

    x=np.outer(
        np.cos(u),
        np.sin(v),
    )

    y=np.outer(
        np.sin(u),
        np.sin(v),
    )

    z=np.outer(
        np.ones_like(u),
        np.cos(v),
    )

    sphere=np.stack(
        [x,y,z],
        axis=-1,
    )

    ellipsoid=np.zeros_like(
        sphere
    )

    for i in range(resolution):

        for j in range(resolution):

            ellipsoid[i,j]=(

                mean+

                radius*

                (L@sphere[i,j])
            )

    return ellipsoid


# ============================================================
# PLOT MAHALANOBIS
# ============================================================

def plot_mahalanobis(

    function_name,

    population,

):

    if not PLOT_MAHALANOBIS:
        return

    if population.shape[1]<3:
        return

    population_3d=population[:,:3]

    distances,mean,covariance=(

        mahalanobis_distance(
            population_3d
        )
    )

    threshold=np.quantile(
        distances,
        MAHALANOBIS_Q,
    )

    selected=distances<=threshold

    fig=plt.figure(
        figsize=(10,8)
    )

    ax=fig.add_subplot(
        111,
        projection="3d",
    )

    scatter = ax.scatter(

        population_3d[:, 0],

        population_3d[:, 1],

        population_3d[:, 2],

        c=distances,

        cmap="viridis",

        s=60,
    )

    cbar = plt.colorbar(scatter)

    cbar.set_label(
        "Mahalanobis Distance"
    )

    ellipsoid=generate_ellipsoid(

        mean,

        covariance,

        q=MAHALANOBIS_Q,
    )

    ax.plot_surface(

        ellipsoid[:,:,0],

        ellipsoid[:,:,1],

        ellipsoid[:,:,2],

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
    )

    ax.set_title(
        f"{function_name} - Iteration {EPOCHS}"
    )

    ax.set_xlabel("X1")

    ax.set_ylabel("X2")

    ax.set_zlabel("X3")

    ax.legend()

    plt.tight_layout()

    out_path=os.path.join(

        MAHALANOBIS_DIR,

        f"{function_name}.png",
    )

    plt.savefig(
        out_path,
        dpi=300,
    )

    plt.close()


# ============================================================
# PLOT CONVERGENCE
# ============================================================

def plot_convergence(

    function_name,

    mean_curve,
):

    if not SAVE_CONVERGENCE:
        return

    plt.figure(
        figsize=(9,5)
    )

    plt.plot(
        mean_curve,
        linewidth=2.5,
    )

    plt.xlabel(
        "Iteration"
    )

    plt.ylabel(
        "Fitness"
    )

    plt.title(
        f"{function_name} Mean Convergence"
    )

    plt.grid(alpha=0.3)

    plt.tight_layout()

    out_path=os.path.join(

        CONVERGENCE_DIR,

        f"{function_name}.png",
    )

    plt.savefig(
        out_path,
        dpi=300,
    )

    plt.close()


# ============================================================
# MAIN
# ============================================================

def main():

    print("="*60)
    print("GRAPHIC METHODOLOGY")
    print("="*60)

    print(f"Benchmark   : {BENCHMARK}")
    print(f"Dimensions  : {DIMENSIONS}")
    print(f"Population  : {POP_SIZE}")
    print(f"Epochs      : {EPOCHS}")
    print(f"Runs        : {RUNS}")
    print("="*60)

    function_map=discover_functions(
        BENCHMARK
    )

    valid_functions=[]

    for function_name,function_class in function_map.items():

        benchmark=create_function(
            function_class
        )

        if benchmark is not None:

            valid_functions.append(

                (
                    function_name,
                    benchmark,
                )
            )

    for function_name,benchmark in valid_functions:

        print("="*60)
        print(f"FUNCTION: {function_name}")
        print("="*60)

        problem={

            "bounds":FloatVar(

                lb=benchmark.lb,

                ub=benchmark.ub,

                name="x",
            ),

            "minmax":"min",

            "obj_func":benchmark.evaluate,
        }

        curves=[]

        last_population=None

        for run in range(RUNS):

            print(
                f"Run {run+1}/{RUNS}"
            )

            optimizer=MaCRO_DE(

                epoch=EPOCHS,

                pop_size=POP_SIZE,

                beta_min=BETA_MIN,

                beta_max=BETA_MAX,

                pcr=PCR,

                mahalanobis_q=MAHALANOBIS_Q,
            )

            g_best=optimizer.solve(
                problem
            )

            curve=np.array(

                optimizer.history.list_global_best_fit
            )

            curves.append(
                curve
            )

            if len(
                optimizer.population_history
            )>0:

                last_population=(

                    optimizer.population_history[-1]
                )

            print(
                f"Best Fitness = {g_best.target.fitness:.6e}"
            )

        mean_curve=np.mean(
            np.array(curves),
            axis=0,
        )

        plot_convergence(

            function_name,

            mean_curve,
        )

        if last_population is not None:

            plot_mahalanobis(

                function_name,

                last_population,
            )

    print("="*60)
    print("METHODOLOGY COMPLETED")
    print("="*60)

    print(
        f"Convergence : {CONVERGENCE_DIR}"
    )

    print(
        f"Mahalanobis : {MAHALANOBIS_DIR}"
    )


# ============================================================
# ENTRY
# ============================================================

if __name__=="__main__":

    main()