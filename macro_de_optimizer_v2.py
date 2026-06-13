import numpy as np
from mealpy.optimizer import Optimizer
from mealpy.utils.agent import Agent
from scipy.stats import chi2

class MaCRO_DE(Optimizer):
    """
    Diversity-based Self-Adaptive Control in Differential Evolution (DSADE)
    with:
    - Delayed AWAD diversity-adaptive control
    - Mahalanobis grouping for mutation pool sampling
    """

    def __init__(
        self,
        epoch=1000,
        pop_size=50,
        beta_min=0.2,
        beta_max=0.8,
        pcr=0.2,
        mahalanobis_q=0.68,
        progress_queue=None,
        progress_interval=50,
        progress_context=None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.epoch = self.validator.check_int("epoch", epoch, [1, 100000])
        self.pop_size = self.validator.check_int("pop_size", pop_size, [10, 10000])
        self.beta_min = self.validator.check_float("beta_min", beta_min, (0.0, 2.0))
        self.beta_max = self.validator.check_float("beta_max", beta_max, (0.0, 2.0))
        self.pcr = self.validator.check_float("pcr", pcr, (0.0, 1.0))
        self.mahalanobis_q = self.validator.check_float("mahalanobis_q", mahalanobis_q, (0.0, 1.0))
        self.progress_queue = progress_queue
        self.progress_interval = max(1, int(progress_interval))
        self.progress_context = progress_context or {}

        if self.beta_min > self.beta_max:
            raise ValueError("beta_min debe ser <= beta_max.")

        self.set_parameters(
            ["epoch", "pop_size", "beta_min", "beta_max", "pcr", "mahalanobis_q"]
        )

        self.sort_flag = False

        self.support_parallel_modes = True

        self.div_awad_hist = None
        self.div_norm_hist = None
        self.pcr_hist = None
        self.fmean_hist = None

        self.div_max_seen = None

        self.div_norm_for_update = 1.0

        # =================================================
        # POPULATION HISTORY
        # =================================================

        self.population_history = []

    def initialize_variables(self):

        self.div_awad_hist = np.full(self.epoch, np.nan, dtype=float)

        self.div_norm_hist = np.full(self.epoch, np.nan, dtype=float)

        self.pcr_hist = np.full(self.epoch, np.nan, dtype=float)

        self.fmean_hist = np.full(self.epoch, np.nan, dtype=float)

        self.div_norm_for_update = 1.0

        self.div_max_seen = None

        # =================================================
        # POPULATION HISTORY
        # =================================================

        self.population_history = []

    def before_main_loop(self):

        pop_pos = self._positions(self.pop)

        div0 = self._awad(
            pop_pos,
            self.problem.lb,
            self.problem.ub,
        )

        self.div_max_seen = max(
            div0,
            self.EPSILON,
        )

    def _positions(self, pop):

        return np.array(
            [agent.solution for agent in pop],
            dtype=float,
        )

    def _awad(self, pop_pos, lb, ub):

        _ = lb, ub

        npop, n_dims = pop_pos.shape

        med_dim = np.median(
            pop_pos,
            axis=0,
        )

        div_dim = np.mean(
            np.abs(pop_pos - med_dim),
            axis=0,
        )

        div = float(
            np.sum(div_dim) / max(n_dims, 1)
        )

        unique_count = np.unique(
            pop_pos,
            axis=0,
        ).shape[0]

        non_repeat_percent = (
            unique_count * 100.0
        ) / max(npop, 1)

        std_devs = np.std(
            pop_pos,
            axis=0,
        )

        std_devs[std_devs == 0] = 1e-5

        if npop <= 1:

            min_distance = 0.0

        else:

            min_distance = np.inf

            for i in range(npop - 1):

                diff = (
                    pop_pos[i + 1:] - pop_pos[i]
                ) / std_devs

                dists = np.sqrt(
                    np.sum(diff * diff, axis=1)
                )

                if dists.size > 0:

                    local_min = float(
                        np.min(dists)
                    )

                    if local_min < min_distance:
                        min_distance = local_min

            if not np.isfinite(min_distance):
                min_distance = 0.0

        epsilon = 1e-1

        penalty_factor = (
            (min_distance + epsilon) ** 2
        ) / (1.0 + min_distance**2)

        div = div * 0.1 * non_repeat_percent

        div = div * penalty_factor

        return float(div)

    def _safe_cov_inv(self, pop_pos):

        n_dims = self.problem.n_dims

        sigma = np.cov(
            pop_pos,
            rowvar=False,
        )

        if np.ndim(sigma) == 0:

            sigma = np.array(
                [[float(sigma)]],
                dtype=float,
            )

        if sigma.shape != (n_dims, n_dims):

            sigma = np.eye(n_dims) * 1e-6

        sigma = (
            (sigma + sigma.T) / 2.0
            + 1e-6 * np.eye(n_dims)
        )

        try:

            chol = np.linalg.cholesky(
                sigma
            )

            return np.linalg.solve(
                chol.T,
                np.linalg.solve(
                    chol,
                    np.eye(n_dims),
                ),
            )

        except np.linalg.LinAlgError:

            return np.linalg.pinv(sigma)

    def _mutation_pool(self, pop_pos, div_norm_used):

        n_dims = self.problem.n_dims

        mu = np.mean(
            pop_pos,
            axis=0,
        )

        sigma_inv = self._safe_cov_inv(
            pop_pos
        )

        d = pop_pos - mu

        dist2 = np.sum(
            (d @ sigma_inv) * d,
            axis=1,
        )

        thr = chi2.ppf(
            self.mahalanobis_q,
            max(n_dims, 1),
        )

        close_mask = dist2 <= thr

        close_particles = pop_pos[
            close_mask
        ]

        far_particles = pop_pos[
            ~close_mask
        ]

        if div_norm_used >= 0.5 and close_particles.shape[0] >= 3:
            return close_particles

        if div_norm_used < 0.5 and far_particles.shape[0] >= 3:
            return far_particles

        return pop_pos

    def evolve(self, epoch):

        epoch_idx = epoch - 1

        div_norm_used = float(
            np.clip(
                self.div_norm_for_update,
                0.0,
                1.0,
            )
        )

        scale_f = float(
            np.clip(
                0.5 + (1.0 - div_norm_used),
                0.5,
                1.5,
            )
        )

        pcr_it = float(
            np.clip(
                self.pcr + 0.25 * (1.0 - div_norm_used),
                0.10,
                0.95,
            )
        )

        self.pcr_hist[epoch_idx] = pcr_it

        f_used_sum = 0.0

        pop_new = []

        for idx in range(self.pop_size):

            pop_pos = self._positions(self.pop)

            pool = self._mutation_pool(
                pop_pos,
                div_norm_used,
            )

            idxs = self.generator.choice(
                pool.shape[0],
                3,
                replace=False,
            )

            x1 = pool[idxs[0]]

            x2 = pool[idxs[1]]

            x3 = pool[idxs[2]]

            f_vec = self.generator.uniform(
                self.beta_min,
                self.beta_max,
                self.problem.n_dims,
            ) * scale_f

            f_vec = np.clip(
                f_vec,
                0.10,
                1.50,
            )

            f_used_sum += float(
                np.mean(f_vec)
            )

            y = x1 + f_vec * (x2 - x3)

            y = self.correct_solution(y)

            z = self.pop[idx].solution.copy()

            j0 = self.generator.integers(
                0,
                self.problem.n_dims,
            )

            cross_mask = (
                self.generator.random(
                    self.problem.n_dims
                ) <= pcr_it
            )

            cross_mask[j0] = True

            z[cross_mask] = y[cross_mask]

            z = self.correct_solution(z)

            candidate = Agent(
                solution=z
            )

            if self.mode not in self.AVAILABLE_MODES:

                candidate.target = self.get_target(z)

                self.pop[idx] = self.get_better_agent(
                    candidate,
                    self.pop[idx],
                    self.problem.minmax,
                )

            else:

                pop_new.append(candidate)

        if self.mode in self.AVAILABLE_MODES:

            pop_new = self.update_target_for_population(
                pop_new
            )

            self.pop = self.greedy_selection_population(
                self.pop,
                pop_new,
                self.problem.minmax,
            )

        self.fmean_hist[epoch_idx] = (
            f_used_sum / self.pop_size
        )

        pop_pos = self._positions(self.pop)

        div_awad = self._awad(
            pop_pos,
            self.problem.lb,
            self.problem.ub,
        )

        self.div_awad_hist[epoch_idx] = div_awad

        self.div_max_seen = max(
            self.div_max_seen,
            div_awad,
        )

        div_norm_now = float(
            np.clip(
                div_awad / (
                    self.div_max_seen + self.EPSILON
                ),
                0.0,
                1.0,
            )
        )

        self.div_norm_hist[epoch_idx] = div_norm_now

        self.div_norm_for_update = div_norm_now

        # =================================================
        # SAVE POPULATION HISTORY
        # =================================================

        population_positions = np.array([
            agent.solution.copy()
            for agent in self.pop
        ])

        self.population_history.append(
            population_positions
        )

        if (
            self.progress_queue is not None
            and (
                epoch == 1
                or epoch == self.epoch
                or epoch % self.progress_interval == 0
            )
        ):
            self.progress_queue.put({
                **self.progress_context,
                "epoch": epoch,
                "total_epochs": self.epoch,
                "best_fitness": float(self.g_best.target.fitness),
            })


# Backward compatibility aliases

MACRO_DE = MaCRO_DE
