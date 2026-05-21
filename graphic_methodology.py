# graphic_methodology.py
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
from scipy.stats import chi2
from scipy.linalg import cholesky

from macro_de_optimizer import MaCRO_DE


# ============================================================
# CONFIGURATION
# ============================================================

SAVE_ROOT = "Methodology_Graphics"

NDIM = 3
POP_SIZE = 80
EPOCHS = 30

BETA_MIN = 0.2
BETA_MAX = 0.8

MAHALANOBIS_Q = 0.68

FRAME_INTERVAL = 1

RANDOM_SEED = 42

np.random.seed(RANDOM_SEED)


# ============================================================
# OUTPUT FOLDERS
# ============================================================

FRAMES_DIR = os.path.join(SAVE_ROOT, "Frames")
CONVERGENCE_DIR = os.path.join(SAVE_ROOT, "Convergence")
ELLIPSOID_DIR = os.path.join(SAVE_ROOT, "Ellipsoids")

for folder in [
    SAVE_ROOT,
    FRAMES_DIR,
    CONVERGENCE_DIR,
    ELLIPSOID_DIR,
]:
    os.makedirs(folder, exist_ok=True)


# ============================================================
# TEST FUNCTION
# ============================================================


def sphere_function(x):
    return np.sum(x ** 2)


# ============================================================
# POPULATION INITIALIZATION
# ============================================================


def initialize_population(pop_size, ndim, lb=-5, ub=5):
    return np.random.uniform(lb, ub, (pop_size, ndim))


# ============================================================
# MAHALANOBIS DISTANCE
# ============================================================


def mahalanobis_distance(points):

    mean = np.mean(points, axis=0)

    covariance = np.cov(points.T)

    covariance += np.eye(covariance.shape[0]) * 1e-8

    inv_cov = np.linalg.inv(covariance)

    centered = points - mean

    distances = []

    for point in centered:
        d = np.sqrt(point.T @ inv_cov @ point)
        distances.append(d)

    distances = np.array(distances)

    return distances, mean, covariance


# ============================================================
# CHOLESKY ELLIPSOID
# ============================================================


def generate_ellipsoid(mean, covariance, q=0.68, resolution=40):

    ndim = covariance.shape[0]

    radius = np.sqrt(chi2.ppf(q, ndim))

    L = cholesky(covariance, lower=True)

    u = np.linspace(0, 2 * np.pi, resolution)
    v = np.linspace(0, np.pi, resolution)

    x = np.outer(np.cos(u), np.sin(v))
    y = np.outer(np.sin(u), np.sin(v))
    z = np.outer(np.ones_like(u), np.cos(v))

    sphere = np.stack([x, y, z], axis=-1)

    ellipsoid = np.zeros_like(sphere)

    for i in range(resolution):
        for j in range(resolution):
            ellipsoid[i, j] = mean + radius * (L @ sphere[i, j])

    return ellipsoid


# ============================================================
# PLOT POPULATION FRAME
# ============================================================


def plot_population_frame(
    population,
    fitness,
    distances,
    mean,
    covariance,
    epoch,
):

    fig = plt.figure(figsize=(10, 8))

    ax = fig.add_subplot(111, projection="3d")

    scatter = ax.scatter(
        population[:, 0],
        population[:, 1],
        population[:, 2],
        c=fitness,
        cmap="viridis",
        s=60,
    )

    ellipsoid = generate_ellipsoid(
        mean,
        covariance,
        q=MAHALANOBIS_Q,
    )

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
        marker="X",
        label="Population Mean",
    )

    ax.set_title(f"Population Distribution - Epoch {epoch}")

    ax.set_xlabel("X1")
    ax.set_ylabel("X2")
    ax.set_zlabel("X3")

    cbar = plt.colorbar(scatter)
    cbar.set_label("Fitness")

    ax.legend()

    plt.tight_layout()

    frame_path = os.path.join(
        FRAMES_DIR,
        f"frame_{epoch:03d}.png",
    )

    plt.savefig(frame_path, dpi=300)
    plt.close()


# ============================================================
# CONVERGENCE CURVE
# ============================================================


def plot_convergence(convergence):

    plt.figure(figsize=(9, 5))

    plt.plot(
        convergence,
        linewidth=2.5,
    )

    plt.xlabel("Iteration")
    plt.ylabel("Best Fitness")
    plt.title("MaCRO-DE Convergence")

    plt.grid(alpha=0.3)

    plt.tight_layout()

    out_path = os.path.join(
        CONVERGENCE_DIR,
        "macro_de_convergence.png",
    )

    plt.savefig(out_path, dpi=300)
    plt.close()


# ============================================================
# MUTATION REGION VISUALIZATION
# ============================================================


def plot_mutation_selection(
    population,
    distances,
    epoch,
):

    threshold = np.quantile(distances, MAHALANOBIS_Q)

    selected = distances <= threshold

    fig = plt.figure(figsize=(10, 8))

    ax = fig.add_subplot(111, projection="3d")

    ax.scatter(
        population[~selected, 0],
        population[~selected, 1],
        population[~selected, 2],
        c="lightgray",
        s=45,
        label="Discarded",
    )

    ax.scatter(
        population[selected, 0],
        population[selected, 1],
        population[selected, 2],
        c="royalblue",
        s=65,
        label="Selected for Mutation",
    )

    ax.set_title(
        f"Mahalanobis-based Mutation Selection - Epoch {epoch}"
    )

    ax.set_xlabel("X1")
    ax.set_ylabel("X2")
    ax.set_zlabel("X3")

    ax.legend()

    plt.tight_layout()

    out_path = os.path.join(
        ELLIPSOID_DIR,
        f"mutation_selection_{epoch:03d}.png",
    )

    plt.savefig(out_path, dpi=300)
    plt.close()


# ============================================================
# MAIN VISUALIZATION LOOP
# ============================================================


def main():

    population = initialize_population(
        POP_SIZE,
        NDIM,
    )

    convergence = []

    best_global = np.inf

    for epoch in range(EPOCHS):

        fitness = np.array([
            sphere_function(ind)
            for ind in population
        ])

        best_idx = np.argmin(fitness)

        if fitness[best_idx] < best_global:
            best_global = fitness[best_idx]

        convergence.append(best_global)

        distances, mean, covariance = mahalanobis_distance(
            population
        )

        if epoch % FRAME_INTERVAL == 0:

            plot_population_frame(
                population,
                fitness,
                distances,
                mean,
                covariance,
                epoch,
            )

            plot_mutation_selection(
                population,
                distances,
                epoch,
            )

        # ====================================================
        # SIMULATED MaCRO-DE MOVEMENT
        # ====================================================

        noise = np.random.normal(
            loc=0,
            scale=0.15,
            size=population.shape,
        )

        attraction = -0.08 * population

        population = population + attraction + noise

    plot_convergence(convergence)

    print("=" * 60)
    print("METHODOLOGY GRAPHICS COMPLETED")
    print("=" * 60)
    print(f"Frames      : {FRAMES_DIR}")
    print(f"Convergence : {CONVERGENCE_DIR}")
    print(f"Ellipsoids  : {ELLIPSOID_DIR}")


# ============================================================
# ENTRY POINT
# ============================================================


if __name__ == "__main__":
    main()