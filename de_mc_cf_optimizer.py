import numpy as np

from de_mc_optimizer import DE_MC


class DE_MC_CF(DE_MC):
    """DE-MC with explicit close/far mutation-pool switching."""

    def _mutation_pool_indices(self, pop_pos, current_idx):
        dist2 = self._mahalanobis_dist2(pop_pos)
        threshold = self._mahalanobis_threshold()
        close = np.flatnonzero(dist2 <= threshold)
        far = np.flatnonzero(dist2 > threshold)
        if self._valid_candidates(close, current_idx).size >= 3:
            return close
        if self._valid_candidates(far, current_idx).size >= 3:
            return far
        return np.arange(self.pop_size)
