"""Adaptive Mahalanobis-Cholesky DE with frozen, target-excluding donors."""
import numpy as np
from mealpy.utils.agent import Agent
from de_mc_cf_optimizer import DE_MC_CF


class DE_MC_CF_V2(DE_MC_CF):
    """Mahalanobis thresholding defines groups; AWAD only routes donor pools."""

    # dM is each target's distance to the frozen generation mean divided by
    # that generation's largest distance (zero for a collapsed population).
    # Historical adaptive control used one U[0, 1) scalar per mutant.
    # F uses independent U(beta_min, beta_max) coordinate draws with MaCRO-DE D scaling;
    # pcr is per target. D is the inherited delayed normalized diversity state,
    # initialized to 1 and updated after greedy selection, exactly as in MaCRO-DE.

    IMPLEMENTATION_REVISION = "macro-D-scaled-coordinate-adaptive-pcr-v5"

    def __init__(self, epoch=1000, pop_size=50, beta_min=0.10, beta_max=0.60, mahalanobis_q=0.50,
                 compute_device="cpu", **kwargs):
        if any(key in kwargs for key in ("wf", "cr", "pcr")):
            raise TypeError("MaCRO-DE-t-v2 uses beta_min/beta_max and adaptive pcr from dM")
        beta_min, beta_max = map(float, (beta_min, beta_max))
        if not (np.isfinite(beta_min) and np.isfinite(beta_max)
                and 0 <= beta_min <= beta_max):
            raise ValueError("Require finite 0 <= beta_min <= beta_max")
        super().__init__(epoch=epoch, pop_size=pop_size, cr=0.9,
                         mahalanobis_q=mahalanobis_q,
                         compute_device=compute_device, **kwargs)
        # Historical fixed control: self.cr = pcr
        # Reuse the original forced-binomial-crossover method.
        self.beta_min, self.beta_max = beta_min, beta_max
        self.set_parameters(["epoch", "pop_size", "beta_min", "beta_max",
                             "mahalanobis_q", "compute_device"])

    def _sample_scale_factors(self):
        D = float(np.clip(self.div_norm_for_update, 0.0, 1.0))
        scale = float(np.clip(1.5 - D, 0.5, 1.5))
        draws = self.generator.uniform(self.beta_min, self.beta_max, self.problem.n_dims)
        return np.clip(draws * scale, 0.1, 1.5)

    # Frozen-generation mechanics copied from MahalanobisDEBase.evolve, with
    # scalar wf replaced by independent coordinate draws with MaCRO-DE D scaling.
    # Historical adaptive control instead used a Mahalanobis-adaptive scalar draw.
    def evolve(self, epoch):
        if self.div_max_seen is None:
            self.before_main_loop()
        pop_pos = self._positions(self.pop)
        # The population is fixed throughout this DE generation. Compute the
        # covariance/classification kernel once and return compact indices plus distances.
        self._epoch_pop_pos = pop_pos
        close, far, dist2 = self.backend.close_far_indices(
            pop_pos, self.problem.n_dims, self._mahalanobis_threshold(),
            self.covariance_inverse_method, include_distances=True)
        self._epoch_close, self._epoch_far = close, far
        dM = self.backend.to_cpu(self.backend.normalized_mahalanobis(dist2))
        pop_new = []

        for idx in range(self.pop_size):
            idxs = self._sample_mutation_indices(pop_pos, idx)
            x1, x2, x3 = pop_pos[idxs[0]], pop_pos[idxs[1]], pop_pos[idxs[2]]

            self.cr = 0.1 + 0.25 * (1.0 - dM[idx])
            mutant = self.correct_solution(x1 + self._sample_scale_factors() * (x2 - x3))
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
        self._epoch_pop_pos = None
        self._epoch_close = None
        self._epoch_far = None

        pop_pos = self._positions(self.pop)
        div_awad = self._awad(pop_pos, self.problem.lb, self.problem.ub)
        self.div_awad_hist[epoch - 1] = div_awad
        self.div_max_seen = max(self.div_max_seen, div_awad)
        div_norm_now = float(
            np.clip(div_awad / (self.div_max_seen + self.EPSILON), 0.0, 1.0)
        )
        self.div_norm_hist[epoch - 1] = div_norm_now
        self.div_norm_for_update = div_norm_now
