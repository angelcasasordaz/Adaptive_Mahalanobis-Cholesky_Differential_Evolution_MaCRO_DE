"""Experimental DE-MC-CF: independent coordinate scales and fixed configurable pcr."""
import numpy as np
from mealpy.utils.agent import Agent
from de_mc_cf_optimizer import DE_MC_CF


class DE_MC_CF_V2(DE_MC_CF):
    """Keep final DE-MC-CF routing/geometry; change only F and crossover parameters."""

    IMPLEMENTATION_REVISION = "awad-close-far-beta-v2"

    def __init__(self, epoch=1000, pop_size=50, beta_min=0.10, beta_max=0.60,
                 pcr=0.10, mahalanobis_q=0.50, compute_device="cpu", **kwargs):
        if "wf" in kwargs or "cr" in kwargs:
            raise TypeError("DE-MC-CF-v2 uses beta_min, beta_max and pcr, not wf/cr")
        beta_min, beta_max, pcr = map(float, (beta_min, beta_max, pcr))
        if not (np.isfinite(beta_min) and np.isfinite(beta_max)
                and 0 <= beta_min <= beta_max):
            raise ValueError("Require finite 0 <= beta_min <= beta_max")
        if not np.isfinite(pcr) or not 0 <= pcr <= 1:
            raise ValueError("Require 0 <= pcr <= 1")
        super().__init__(epoch=epoch, pop_size=pop_size, cr=0.9,
                         mahalanobis_q=mahalanobis_q,
                         compute_device=compute_device, **kwargs)
        self.beta_min, self.beta_max, self.pcr = beta_min, beta_max, pcr
        self.cr = pcr  # Reuse the original forced-binomial-crossover method.
        self.set_parameters(["epoch", "pop_size", "beta_min", "beta_max", "pcr",
                             "mahalanobis_q", "compute_device"])

    def _sample_scale_factors(self):
        return self.generator.uniform(self.beta_min, self.beta_max, self.problem.n_dims)

    # Frozen-generation mechanics copied from MahalanobisDEBase.evolve, with
    # only scalar wf replaced by an independent vector draw per mutation.
    def evolve(self, epoch):
        if self.div_max_seen is None:
            self.before_main_loop()
        pop_pos = self._positions(self.pop)
        # The population is fixed throughout this DE generation. Compute the
        # covariance/classification kernel once and return only compact indices.
        self._epoch_pop_pos = pop_pos
        self._epoch_close, self._epoch_far = self._close_far_indices(pop_pos)
        pop_new = []

        for idx in range(self.pop_size):
            idxs = self._sample_mutation_indices(pop_pos, idx)
            x1, x2, x3 = pop_pos[idxs[0]], pop_pos[idxs[1]], pop_pos[idxs[2]]

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

