"""Host plans: MaCRO-DE D-scaled coordinate F and adaptive per-target pcr.

AWAD routes pools; covariance/Cholesky thresholding defines their membership.
"""
import numpy as np


def random_plan(engine, state, close, far, dM=None):
    runs, pop, dims = len(state.seeds), engine.pop_size, engine.n_dims
    donors = engine._host_plan_buffer("donors", (runs, pop, 3), np.int64)
    cross = engine._host_plan_buffer("crossover", (runs, pop, dims), bool)
    factors = engine._host_plan_buffer("f_vectors", (runs, pop, dims), np.float64)
    indices = np.arange(pop, dtype=np.int64)
    if dM is None:
        dist2 = engine.backend.mahalanobis_distances(state.positions, dims, "cholesky")
        dM = engine.backend.to_cpu(engine.backend.normalized_mahalanobis(dist2))
    pcr = 0.1 + 0.25 * (1.0 - dM)
    # Reuse MaCRO-DE's delayed normalized diversity state, independently per run.
    D = np.clip(state.algorithm_state["div_norm_cpu"], 0.0, 1.0)
    scale = np.clip(1.5 - D, 0.5, 1.5)
    for r, rng in enumerate(state.generators):
        selected = close[r] if state.algorithm_state["div_norm_cpu"][r] >= 0.5 else far[r]
        pool = indices[selected]
        for i in range(pop):
            candidates = pool[pool != i]
            if candidates.size < 3:
                candidates = engine._de_fallback_candidates[i]
            donors[r, i] = rng.choice(candidates, 3, replace=False)
            factors[r, i] = np.clip(
                rng.uniform(engine.beta_min, engine.beta_max, dims) * scale[r], 0.1, 1.5)
            j0 = rng.integers(0, dims)
            cross[r, i] = rng.random(dims) <= pcr[r, i]
            cross[r, i, j0] = True
    return donors, cross, factors, pcr
