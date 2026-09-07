"""Distance-only kernels for the optional MaCRO-DE distance ablation.

All norms use the population mean and raw coordinates. Alternative norms
only rank individuals; their close set matches the Mahalanobis close count.
The chi-square cutoff is applied exclusively to squared Mahalanobis distance.
"""
import math

DISTANCE_LABELS = {
    "chebyshev": "MaCRO-DE/Chebyshev",
    "euclidean": "MaCRO-DE/Euclidean",
    "mahalanobis_cholesky": "MaCRO-DE/Mahalanobis-Cholesky",
    "manhattan": "MaCRO-DE/Manhattan",
    "minkowski": "MaCRO-DE/Minkowski",
}
DISTANCE_IMPLEMENTATION_REVISION = "distance-matched-mahalanobis-count-v2"


def validate_metric(metric, minkowski_p):
    if metric not in DISTANCE_LABELS:
        raise ValueError(f"Unknown distance metric: {metric!r}")
    if not math.isfinite(minkowski_p) or minkowski_p <= 0 or minkowski_p == 2:
        raise ValueError("Distance-ablation Minkowski p must be finite, positive, and different from 2")


def squared_distance_from_mean(population, xp, metric, minkowski_p):
    """Return (..., population_size) squared norms on the supplied backend."""
    delta = xp.abs(population - xp.mean(population, axis=-2, keepdims=True))
    if metric == "chebyshev":
        return xp.max(delta, axis=-1) ** 2
    if metric == "euclidean":
        return xp.sum(delta * delta, axis=-1)
    if metric == "manhattan":
        return xp.sum(delta, axis=-1) ** 2
    if metric == "minkowski":
        return xp.sum(delta ** minkowski_p, axis=-1) ** (2.0 / minkowski_p)
    raise ValueError("Mahalanobis-Cholesky must use the original implementation")


def nearest_k_mask(distances, k, xp):
    """Select exactly k per run, breaking distance ties by population index.

    Pairwise ranks avoid backend-specific sort stability and consume no RNG.
    The returned mask retains population order for inherited donor sampling.
    k can be a scalar or an array with the distances' leading batch shape.
    """
    indices = xp.arange(distances.shape[-1])
    current = distances[..., :, None]
    other = distances[..., None, :]
    precedes = (other < current) | (
        (other == current) & (indices[None, :] < indices[:, None])
    )
    ranks = xp.sum(precedes, axis=-1)
    return ranks < xp.asarray(k)[..., None]
