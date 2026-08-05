import numpy as np


def awad(pop_pos):
    """
    Average Weighted Absolute Deviation (AWAD)

    Parameters
    ----------
    pop_pos : ndarray (pop_size, n_dims)
        Population matrix.

    Returns
    -------
    float
        AWAD diversity value.
    """

    pop_pos = np.asarray(pop_pos, dtype=float)

    npop, n_dims = pop_pos.shape

    # Median center per dimension
    med_dim = np.median(pop_pos, axis=0)

    # Mean absolute deviation
    div_dim = np.mean(np.abs(pop_pos - med_dim), axis=0)
    div = float(np.sum(div_dim) / max(n_dims, 1))

    # Percentage of unique individuals
    unique_count = np.unique(pop_pos, axis=0).shape[0]
    non_repeat_percent = (unique_count * 100.0) / max(npop, 1)

    # Minimum standardized Euclidean distance
    std_devs = np.std(pop_pos, axis=0)
    std_devs[std_devs == 0] = 1e-5

    if npop <= 1:
        min_distance = 0.0
    else:
        min_distance = np.inf

        for i in range(npop - 1):
            diff = (pop_pos[i + 1:] - pop_pos[i]) / std_devs
            dists = np.sqrt(np.sum(diff * diff, axis=1))

            if dists.size:
                local_min = float(np.min(dists))
                if local_min < min_distance:
                    min_distance = local_min

        if not np.isfinite(min_distance):
            min_distance = 0.0

    epsilon = 1e-1
    penalty_factor = ((min_distance + epsilon) ** 2) / (1.0 + min_distance**2)

    div *= 0.1 * non_repeat_percent
    div *= penalty_factor

    return float(div)