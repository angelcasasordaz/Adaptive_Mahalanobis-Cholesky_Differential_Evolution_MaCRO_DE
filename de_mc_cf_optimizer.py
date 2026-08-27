import numpy as np

from de_mc_optimizer import DE_MC


class DE_MC_CF(DE_MC):
    """DE-MC with explicit close/far mutation-pool switching."""

    def _mutation_pool_indices(self, pop_pos, current_idx):
        close, far = self._close_far_indices(pop_pos)
        if self._valid_candidates(close, current_idx).size >= 3:
            return close
        if self._valid_candidates(far, current_idx).size >= 3:
            return far
        return np.arange(self.pop_size)

    @property
    def covariance_inverse_method(self):
        # Preserve this separate ablation's existing inverse-based arithmetic.
        return "cholesky"
