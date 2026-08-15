import numpy as np
from mealpy.optimizer import Optimizer
from mealpy.utils.agent import Agent
from scipy.stats import chi2


class DE_Mahalanobis(Optimizer):
    """
    Differential Evolution with only Mahalanobis mutation-pool selection.

    Active ablation component:
    - Mahalanobis close/far mutation-pool selection using the same covariance
      stabilization and chi-square threshold as the current DSADE/MaCRO-DE
      implementation.

    Excluded components:
    - AWAD-based mutation-factor or crossover adaptation.
    - Diversity-guided survivor selection.

    The current project Mahalanobis pool switch depends on a delayed normalized
    AWAD diversity value to choose close versus far particles. This class keeps
    that minimal diversity bookkeeping only for the pool switch; F, crossover,
    and survivor selection remain fixed standard DE behavior.
    """

    def __init__(self, epoch=1000, pop_size=50, wf=0.1, cr=0.9, mahalanobis_q=0.68, **kwargs):
        super().__init__(**kwargs)
        self.epoch = self.validator.check_int("epoch", epoch, [1, 100000])
        self.pop_size = self.validator.check_int("pop_size", pop_size, [5, 10000])
        self.wf = self.validator.check_float("wf", wf, (-3.0, 3.0))
        self.cr = self.validator.check_float("cr", cr, (0.0, 1.0))
        self.mahalanobis_q = self.validator.check_float("mahalanobis_q", mahalanobis_q, (0.0, 1.0))
        self.set_parameters(["epoch", "pop_size", "wf", "cr", "mahalanobis_q"])
        self.sort_flag = False
        self.support_parallel_modes = True

        self.div_awad_hist = None
        self.div_norm_hist = None
        self.div_max_seen = None
        self.div_norm_for_update = 1.0

    def initialize_variables(self):
        self.div_awad_hist = np.full(self.epoch, np.nan, dtype=float)
        self.div_norm_hist = np.full(self.epoch, np.nan, dtype=float)
        self.div_norm_for_update = 1.0
        self.div_max_seen = None

    def before_main_loop(self):
        pop_pos = self._positions(self.pop)
        div0 = self._awad(pop_pos, self.problem.lb, self.problem.ub)
        self.div_max_seen = max(div0, self.EPSILON)

    def _positions(self, pop):
        return np.array([agent.solution for agent in pop], dtype=float)

    def _awad(self, pop_pos, lb, ub):
        _ = lb, ub
        npop, n_dims = pop_pos.shape

        # Kept only because the existing Mahalanobis pool switch needs the
        # delayed normalized diversity state to choose close versus far pools.
        med_dim = np.median(pop_pos, axis=0)
        div_dim = np.mean(np.abs(pop_pos - med_dim), axis=0)
        div = float(np.sum(div_dim) / max(n_dims, 1))

        unique_count = np.unique(pop_pos, axis=0).shape[0]
        non_repeat_percent = (unique_count * 100.0) / max(npop, 1)

        std_devs = np.std(pop_pos, axis=0)
        std_devs[std_devs == 0] = 1e-5
        if npop <= 1:
            min_distance = 0.0
        else:
            min_distance = np.inf
            for i in range(npop - 1):
                diff = (pop_pos[i + 1:] - pop_pos[i]) / std_devs
                dists = np.sqrt(np.sum(diff * diff, axis=1))
                if dists.size > 0:
                    local_min = float(np.min(dists))
                    if local_min < min_distance:
                        min_distance = local_min
            if not np.isfinite(min_distance):
                min_distance = 0.0

        epsilon = 1e-1
        penalty_factor = ((min_distance + epsilon) ** 2) / (1.0 + min_distance**2)
        div = div * 0.1 * non_repeat_percent
        div = div * penalty_factor
        return float(div)

    def _safe_cov_inv(self, pop_pos):
        n_dims = self.problem.n_dims
        sigma = np.cov(pop_pos, rowvar=False)
        if np.ndim(sigma) == 0:
            sigma = np.array([[float(sigma)]], dtype=float)
        if sigma.shape != (n_dims, n_dims):
            sigma = np.eye(n_dims) * 1e-6
        sigma = (sigma + sigma.T) / 2.0 + 1e-6 * np.eye(n_dims)
        try:
            chol = np.linalg.cholesky(sigma)
            return np.linalg.solve(chol.T, np.linalg.solve(chol, np.eye(n_dims)))
        except np.linalg.LinAlgError:
            return np.linalg.pinv(sigma)

    def _mutation_pool_indices(self, pop_pos, div_norm_used):
        n_dims = self.problem.n_dims
        mu = np.mean(pop_pos, axis=0)
        sigma_inv = self._safe_cov_inv(pop_pos)
        d = pop_pos - mu
        dist2 = np.sum((d @ sigma_inv) * d, axis=1)
        thr = chi2.ppf(self.mahalanobis_q, max(n_dims, 1))
        close_mask = dist2 <= thr

        close_indices = np.flatnonzero(close_mask)
        far_indices = np.flatnonzero(~close_mask)

        if div_norm_used >= 0.5 and close_indices.size >= 3:
            return close_indices
        if div_norm_used < 0.5 and far_indices.size >= 3:
            return far_indices
        return np.arange(self.pop_size)

    def _sample_pool_indices(self, pool_indices, current_idx):
        candidates = pool_indices[pool_indices != current_idx]
        if candidates.size < 3:
            candidates = np.array(list(set(range(self.pop_size)) - {current_idx}), dtype=int)
        return self.generator.choice(candidates, 3, replace=False)

    def _binomial_crossover(self, parent_pos, mutant_pos):
        trial = parent_pos.copy()
        j0 = self.generator.integers(0, self.problem.n_dims)
        cross_mask = self.generator.random(self.problem.n_dims) <= self.cr
        cross_mask[j0] = True
        trial[cross_mask] = mutant_pos[cross_mask]
        return self.correct_solution(trial)

    def evolve(self, epoch):
        epoch_idx = epoch - 1
        if self.div_max_seen is None:
            self.before_main_loop()

        div_norm_used = float(np.clip(self.div_norm_for_update, 0.0, 1.0))
        pop_new = []

        for idx in range(self.pop_size):
            pop_pos = self._positions(self.pop)
            pool_indices = self._mutation_pool_indices(pop_pos, div_norm_used)
            idxs = self._sample_pool_indices(pool_indices, idx)
            x1, x2, x3 = pop_pos[idxs[0]], pop_pos[idxs[1]], pop_pos[idxs[2]]

            mutant = self.correct_solution(x1 + self.wf * (x2 - x3))
            trial = self._binomial_crossover(self.pop[idx].solution, mutant)
            candidate = Agent(solution=trial)

            if self.mode not in self.AVAILABLE_MODES:
                candidate.target = self.get_target(trial)
                self.pop[idx] = self.get_better_agent(candidate, self.pop[idx], self.problem.minmax)
            else:
                pop_new.append(candidate)

        if self.mode in self.AVAILABLE_MODES:
            pop_new = self.update_target_for_population(pop_new)
            self.pop = self.greedy_selection_population(self.pop, pop_new, self.problem.minmax)

        pop_pos = self._positions(self.pop)
        div_awad = self._awad(pop_pos, self.problem.lb, self.problem.ub)
        self.div_awad_hist[epoch_idx] = div_awad
        self.div_max_seen = max(self.div_max_seen, div_awad)
        div_norm_now = float(np.clip(div_awad / (self.div_max_seen + self.EPSILON), 0.0, 1.0))
        self.div_norm_hist[epoch_idx] = div_norm_now
        self.div_norm_for_update = div_norm_now
