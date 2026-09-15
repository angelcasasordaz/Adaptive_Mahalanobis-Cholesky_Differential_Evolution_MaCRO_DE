"""Host RNG plans for experimental CF V2; no AWAD parameter scaling."""
import numpy as np


def random_plan(engine, state, close, far):
    runs, pop, dims = len(state.seeds), engine.pop_size, engine.n_dims
    donors = engine._host_plan_buffer("donors", (runs, pop, 3), np.int64)
    cross = engine._host_plan_buffer("crossover", (runs, pop, dims), bool)
    factors = engine._host_plan_buffer("f_vectors", (runs, pop, dims), np.float64)
    indices = np.arange(pop, dtype=np.int64)
    for r, rng in enumerate(state.generators):
        selected = close[r] if state.algorithm_state["div_norm_cpu"][r] >= 0.5 else far[r]
        pool = indices[selected]
        for i in range(pop):
            candidates = pool[pool != i]
            if candidates.size < 3:
                candidates = engine._de_fallback_candidates[i]
            donors[r, i] = rng.choice(candidates, 3, replace=False)
            factors[r, i] = rng.uniform(engine.beta_min, engine.beta_max, dims)
            j0 = rng.integers(0, dims)
            cross[r, i] = rng.random(dims) <= engine.pcr
            cross[r, i, j0] = True
    return donors, cross, factors, np.full(runs, engine.pcr)
