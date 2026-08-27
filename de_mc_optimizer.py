import numpy as np

from de_ablation_base import MahalanobisDEBase


class DE_MC(MahalanobisDEBase):
    """DE-M with Cholesky-solve Mahalanobis distances."""

    def _covariance_inverse(self, sigma):
        n_dims = self.problem.n_dims
        try:
            chol = np.linalg.cholesky(sigma)
            return np.linalg.solve(
                chol.T,
                np.linalg.solve(chol, np.eye(n_dims)),
            )
        except np.linalg.LinAlgError:
            return np.linalg.pinv(sigma)

    def _mutation_pool_indices(self, pop_pos, current_idx):
        close = self._close_indices(pop_pos)
        if self._valid_candidates(close, current_idx).size >= 3:
            return close
        return np.arange(self.pop_size)

    @property
    def covariance_inverse_method(self):
        return "cholesky_solve"
