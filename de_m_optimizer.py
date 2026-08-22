import numpy as np

from de_ablation_base import MahalanobisDEBase


class DE_M(MahalanobisDEBase):
    """DE/rand/1/bin using close Mahalanobis pool and direct covariance inverse."""

    def _covariance_inverse(self, sigma):
        try:
            return np.linalg.inv(sigma)
        except np.linalg.LinAlgError:
            return np.linalg.pinv(sigma)

    def _mutation_pool_indices(self, pop_pos, current_idx):
        close = self._close_indices(pop_pos)
        if self._valid_candidates(close, current_idx).size >= 3:
            return close
        return np.arange(self.pop_size)
