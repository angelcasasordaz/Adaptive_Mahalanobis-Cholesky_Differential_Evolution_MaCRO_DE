import numpy as np
from mealpy.optimizer import Optimizer
from mealpy.utils.agent import Agent


class DE_AWAD(Optimizer):
    """
    Differential Evolution with only AWAD-based parameter adaptation.

    Active ablation component:
    - Delayed, normalized AWAD diversity controls the mutation factor and
      crossover probability.

    Excluded components:
    - Mahalanobis pool selection.
    - Diversity-guided survivor selection.
    """

    def __init__(
        self,
        epoch=1000,
        pop_size=50,
        beta_min=0.2,
        beta_max=0.8,
        pcr=0.2,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.epoch = self.validator.check_int("epoch", epoch, [1, 100000])
        self.pop_size = self.validator.check_int("pop_size", pop_size, [5, 10000])
        self.beta_min = self.validator.check_float("beta_min", beta_min, (0.0, 2.0))
        self.beta_max = self.validator.check_float("beta_max", beta_max, (0.0, 2.0))
        self.pcr = self.validator.check_float("pcr", pcr, (0.0, 1.0))
        if self.beta_min > self.beta_max:
            raise ValueError("beta_min must be <= beta_max.")
        self.set_parameters(["epoch", "pop_size", "beta_min", "beta_max", "pcr"])
        self.sort_flag = False
        self.support_parallel_modes = True

        self.div_awad_hist = None
        self.div_norm_hist = None
        self.pcr_hist = None
        self.fmean_hist = None
        self.div_max_seen = None
        self.div_norm_for_update = 1.0

    def initialize_variables(self):
        self.div_awad_hist = np.full(self.epoch, np.nan, dtype=float)
        self.div_norm_hist = np.full(self.epoch, np.nan, dtype=float)
        self.pcr_hist = np.full(self.epoch, np.nan, dtype=float)
        self.fmean_hist = np.full(self.epoch, np.nan, dtype=float)
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

    def _random_de_indices(self, current_idx):
        candidates = list(set(range(self.pop_size)) - {current_idx})
        return self.generator.choice(candidates, 3, replace=False)

    def _binomial_crossover(self, parent_pos, mutant_pos, crossover_rate):
        trial = parent_pos.copy()
        j0 = self.generator.integers(0, self.problem.n_dims)
        cross_mask = self.generator.random(self.problem.n_dims) <= crossover_rate
        cross_mask[j0] = True
        trial[cross_mask] = mutant_pos[cross_mask]
        return self.correct_solution(trial)

    def evolve(self, epoch):
        epoch_idx = epoch - 1
        if self.div_max_seen is None:
            self.before_main_loop()

        div_norm_used = float(np.clip(self.div_norm_for_update, 0.0, 1.0))
        scale_f = float(np.clip(0.5 + (1.0 - div_norm_used), 0.5, 1.5))
        pcr_it = float(np.clip(self.pcr + 0.25 * (1.0 - div_norm_used), 0.10, 0.95))
        self.pcr_hist[epoch_idx] = pcr_it

        f_used_sum = 0.0
        pop_new = []

        for idx in range(self.pop_size):
            pop_pos = self._positions(self.pop)
            idxs = self._random_de_indices(idx)
            x1, x2, x3 = pop_pos[idxs[0]], pop_pos[idxs[1]], pop_pos[idxs[2]]

            f_vec = self.generator.uniform(self.beta_min, self.beta_max, self.problem.n_dims) * scale_f
            f_vec = np.clip(f_vec, 0.10, 1.50)
            f_used_sum += float(np.mean(f_vec))

            mutant = self.correct_solution(x1 + f_vec * (x2 - x3))
            trial = self._binomial_crossover(self.pop[idx].solution, mutant, pcr_it)
            candidate = Agent(solution=trial)

            if self.mode not in self.AVAILABLE_MODES:
                candidate.target = self.get_target(trial)
                self.pop[idx] = self.get_better_agent(candidate, self.pop[idx], self.problem.minmax)
            else:
                pop_new.append(candidate)

        if self.mode in self.AVAILABLE_MODES:
            pop_new = self.update_target_for_population(pop_new)
            self.pop = self.greedy_selection_population(self.pop, pop_new, self.problem.minmax)

        self.fmean_hist[epoch_idx] = f_used_sum / self.pop_size

        pop_pos = self._positions(self.pop)
        div_awad = self._awad(pop_pos, self.problem.lb, self.problem.ub)
        self.div_awad_hist[epoch_idx] = div_awad
        self.div_max_seen = max(self.div_max_seen, div_awad)
        div_norm_now = float(np.clip(div_awad / (self.div_max_seen + self.EPSILON), 0.0, 1.0))
        self.div_norm_hist[epoch_idx] = div_norm_now
        self.div_norm_for_update = div_norm_now
