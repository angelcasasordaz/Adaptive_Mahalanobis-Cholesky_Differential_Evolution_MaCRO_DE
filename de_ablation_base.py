import numpy as np
from mealpy.optimizer import Optimizer
from mealpy.utils.agent import Agent
from scipy.stats import chi2


class MahalanobisDEBase(Optimizer):
    """Shared DE/rand/1/bin mechanics for MaCRO-DE ablation variants."""

    def __init__(
        self,
        epoch=1000,
        pop_size=50,
        wf=0.5,
        cr=0.9,
        mahalanobis_q=0.68,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.epoch = self.validator.check_int("epoch", epoch, [1, 100000])
        self.pop_size = self.validator.check_int("pop_size", pop_size, [5, 10000])
        self.wf = self.validator.check_float("wf", wf, (-3.0, 3.0))
        self.cr = self.validator.check_float("cr", cr, (0.0, 1.0))
        self.mahalanobis_q = self.validator.check_float(
            "mahalanobis_q",
            mahalanobis_q,
            (0.0, 1.0),
        )
        self.set_parameters(["epoch", "pop_size", "wf", "cr", "mahalanobis_q"])
        self.sort_flag = False
        self.support_parallel_modes = True

    def _positions(self, pop):
        return np.array([agent.solution for agent in pop], dtype=float)

    def _covariance_matrix(self, pop_pos):
        n_dims = self.problem.n_dims
        sigma = np.cov(pop_pos, rowvar=False)
        if np.ndim(sigma) == 0:
            sigma = np.array([[float(sigma)]], dtype=float)
        if sigma.shape != (n_dims, n_dims):
            sigma = np.eye(n_dims, dtype=float) * 1e-6
        return (sigma + sigma.T) / 2.0 + 1e-6 * np.eye(n_dims)

    def _covariance_inverse(self, sigma):
        raise NotImplementedError

    def _mahalanobis_dist2(self, pop_pos):
        n_dims = self.problem.n_dims
        mu = np.mean(pop_pos, axis=0)
        sigma = self._covariance_matrix(pop_pos)
        sigma_inv = self._covariance_inverse(sigma)
        diff = pop_pos - mu
        return np.sum((diff @ sigma_inv) * diff, axis=1)

    def _mahalanobis_threshold(self):
        return chi2.ppf(0.68, self.problem.n_dims)

    def _close_indices(self, pop_pos):
        dist2 = self._mahalanobis_dist2(pop_pos)
        threshold = self._mahalanobis_threshold()
        return np.flatnonzero(dist2 <= threshold)

    def _mutation_pool_indices(self, pop_pos, current_idx):
        raise NotImplementedError

    def _valid_candidates(self, pool_indices, current_idx):
        return pool_indices[pool_indices != current_idx]

    def _fallback_candidates(self, current_idx):
        return np.array(
            [idx for idx in range(self.pop_size) if idx != current_idx],
            dtype=int,
        )

    def _sample_mutation_indices(self, pop_pos, current_idx):
        pool_indices = self._mutation_pool_indices(pop_pos, current_idx)
        candidates = self._valid_candidates(pool_indices, current_idx)
        if candidates.size < 3:
            candidates = self._fallback_candidates(current_idx)
        return self.generator.choice(candidates, 3, replace=False)

    def _binomial_crossover(self, parent_pos, mutant_pos):
        trial = parent_pos.copy()
        j0 = self.generator.integers(0, self.problem.n_dims)
        cross_mask = self.generator.random(self.problem.n_dims) <= self.cr
        cross_mask[j0] = True
        trial[cross_mask] = mutant_pos[cross_mask]
        return self.correct_solution(trial)

    def evolve(self, epoch):
        pop_pos = self._positions(self.pop)
        pop_new = []

        for idx in range(self.pop_size):
            idxs = self._sample_mutation_indices(pop_pos, idx)
            x1, x2, x3 = pop_pos[idxs[0]], pop_pos[idxs[1]], pop_pos[idxs[2]]

            mutant = self.correct_solution(x1 + self.wf * (x2 - x3))
            trial = self._binomial_crossover(self.pop[idx].solution, mutant)
            candidate = Agent(solution=trial)

            if self.mode not in self.AVAILABLE_MODES:
                candidate.target = self.get_target(trial)
                self.pop[idx] = self.get_better_agent(
                    candidate,
                    self.pop[idx],
                    self.problem.minmax,
                )
            else:
                pop_new.append(candidate)

        if self.mode in self.AVAILABLE_MODES:
            pop_new = self.update_target_for_population(pop_new)
            self.pop = self.greedy_selection_population(
                self.pop,
                pop_new,
                self.problem.minmax,
            )
