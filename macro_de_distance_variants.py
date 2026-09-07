"""Distance-only extensions; evolution and random plans remain inherited."""
import numpy as np

from compute_backend import ComputeBackend
from distance_metrics import nearest_k_mask, squared_distance_from_mean, validate_metric
from gpu_batching import BatchedDEEngine
from macro_de_optimizer import MaCRO_DE


class DistanceBackend(ComputeBackend):
    def __init__(self, device, metric, minkowski_p):
        super().__init__(device)
        validate_metric(metric, minkowski_p)
        self.metric = metric
        self.minkowski_p = minkowski_p

    def close_far_masks(self, population, n_dims, threshold, method):
        # The original kernel/cutoff supplies k on this run's current population.
        mahalanobis = super().mahalanobis_distances(population, n_dims, method)
        k = self.xp.sum(mahalanobis <= threshold, axis=-1)
        distances = squared_distance_from_mean(
            self.asarray(population), self.xp, self.metric, self.minkowski_p
        )
        close = nearest_k_mask(distances, k, self.xp)
        return close, ~close, distances

    def close_far_indices(
        self, population, n_dims, threshold, method, include_distances=True
    ):
        close, _, distances = self.close_far_masks(population, n_dims, threshold, method)
        close = self.to_cpu(close).astype(bool, copy=False)
        indices = np.flatnonzero(close), np.flatnonzero(~close)
        if include_distances:
            return *indices, self.to_cpu(distances)
        return indices


class MaCRO_DE_Distance(MaCRO_DE):
    def __init__(self, *, distance_metric, minkowski_p, **kwargs):
        super().__init__(**kwargs)
        self.backend = DistanceBackend(self.compute_device, distance_metric, minkowski_p)


class BatchedMaCRODistance(BatchedDEEngine):
    def __init__(self, *, distance_metric, minkowski_p, **kwargs):
        validate_metric(distance_metric, minkowski_p)
        super().__init__(**kwargs)
        if self.optimizer_name != "MaCRO-DE":
            raise ValueError("Distance ablation requires MaCRO-DE")
        self.distance_metric = distance_metric
        self.minkowski_p = minkowski_p

    def _classification(self, positions, method, threshold):
        # Reuse the exact original batched Mahalanobis classification arithmetic.
        original_close, _, _ = super()._classification(positions, method, threshold)
        k = self.xp.sum(self.xp.asarray(original_close), axis=-1)
        distances = self._stage(
            "distance_metric",
            lambda: squared_distance_from_mean(
                positions, self.xp, self.distance_metric, self.minkowski_p
            ),
        )
        close = self._stage(
            "classification", lambda: nearest_k_mask(distances, k, self.xp)
        )
        if self._device_macro:
            return close, ~close, distances
        close = self.backend.to_cpu(close).astype(bool, copy=False)
        return close, ~close, distances
