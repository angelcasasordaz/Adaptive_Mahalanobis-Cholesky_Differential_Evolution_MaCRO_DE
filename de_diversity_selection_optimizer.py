import numpy as np
from mealpy.optimizer import Optimizer
from mealpy.utils.agent import Agent


class DE_DiversitySelection(Optimizer):
    """
    Differential Evolution with only AWAD-guided survivor selection.

    Active ablation component:
    - Fitness-first survivor selection with local AWAD contribution as the
      tie-breaker when the offspring is not fitter than the parent.

    Excluded components:
    - AWAD-based mutation-factor or crossover adaptation.
    - Mahalanobis pool selection.
    """

    def __init__(self, epoch=1000, pop_size=50, wf=0.1, cr=0.9, **kwargs):
        super().__init__(**kwargs)
        self.epoch = self.validator.check_int("epoch", epoch, [1, 100000])
        self.pop_size = self.validator.check_int("pop_size", pop_size, [5, 10000])
        self.wf = self.validator.check_float("wf", wf, (-3.0, 3.0))
        self.cr = self.validator.check_float("cr", cr, (0.0, 1.0))
        self.set_parameters(["epoch", "pop_size", "wf", "cr"])
        self.sort_flag = False
        self.support_parallel_modes = True

    def _positions(self, pop):
        return np.array([agent.solution for agent in pop], dtype=float)

    def _awad(self, pop_pos, lb, ub):
        _ = lb, ub
        npop, n_dims = pop_pos.shape

        # Median center per dimension, matching the project AWAD code.
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

    def local_awad_contribution(self, candidate_pos, base_pop_pos):
        """
        Estimate one candidate's local AWAD contribution in the same population
        context used by DSADE_AWAD. The caller removes the compared parent from
        base_pop_pos so the parent and offspring are scored under one context.
        """
        candidate_pos = np.asarray(candidate_pos, dtype=float)
        base_pop_pos = np.asarray(base_pop_pos, dtype=float)
        if base_pop_pos.size == 0:
            local_pop = candidate_pos.reshape(1, -1)
        else:
            local_pop = np.vstack((base_pop_pos, candidate_pos))
        return self._awad(local_pop, self.problem.lb, self.problem.ub)

    def diversity_selection(self, parent, offspring, base_pop_pos):
        """
        Select between parent and offspring using fitness first, then AWAD.
        Better fitness is always accepted. Otherwise, the offspring survives
        only if it improves the local AWAD contribution.
        """
        if self.compare_target(offspring.target, parent.target, self.problem.minmax):
            return offspring

        offspring_awad = self.local_awad_contribution(offspring.solution, base_pop_pos)
        parent_awad = self.local_awad_contribution(parent.solution, base_pop_pos)
        if offspring_awad > parent_awad + self.EPSILON:
            return offspring
        return parent

    def _diversity_selection_population(self, pop_old, pop_new):
        old_pos = self._positions(pop_old)
        selected = []
        for idx, (parent, offspring) in enumerate(zip(pop_old, pop_new)):
            base_pop_pos = np.delete(old_pos, idx, axis=0)
            selected.append(self.diversity_selection(parent, offspring, base_pop_pos))
        return selected

    def _random_de_indices(self, current_idx):
        candidates = list(set(range(self.pop_size)) - {current_idx})
        return self.generator.choice(candidates, 3, replace=False)

    def _binomial_crossover(self, parent_pos, mutant_pos):
        trial = parent_pos.copy()
        j0 = self.generator.integers(0, self.problem.n_dims)
        cross_mask = self.generator.random(self.problem.n_dims) <= self.cr
        cross_mask[j0] = True
        trial[cross_mask] = mutant_pos[cross_mask]
        return self.correct_solution(trial)

    def evolve(self, epoch):
        pop_new = []

        for idx in range(self.pop_size):
            pop_pos = self._positions(self.pop)
            idxs = self._random_de_indices(idx)
            x1, x2, x3 = pop_pos[idxs[0]], pop_pos[idxs[1]], pop_pos[idxs[2]]

            mutant = self.correct_solution(x1 + self.wf * (x2 - x3))
            trial = self._binomial_crossover(self.pop[idx].solution, mutant)
            candidate = Agent(solution=trial)

            if self.mode not in self.AVAILABLE_MODES:
                candidate.target = self.get_target(trial)
                base_pop_pos = np.delete(pop_pos, idx, axis=0)
                self.pop[idx] = self.diversity_selection(self.pop[idx], candidate, base_pop_pos)
            else:
                pop_new.append(candidate)

        if self.mode in self.AVAILABLE_MODES:
            pop_new = self.update_target_for_population(pop_new)
            self.pop = self._diversity_selection_population(self.pop, pop_new)
