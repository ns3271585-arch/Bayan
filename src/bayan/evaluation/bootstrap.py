"""Lab 6: bootstrap confidence intervals."""

import numpy as np


def bootstrap_ci(values, *, n_boot=2000, seed=42, alpha=0.05):
    """Return the mean and a percentile bootstrap confidence interval."""
    values = np.asarray(values, dtype=float)

    if values.size == 0:
        raise ValueError("values must not be empty")

    if n_boot <= 0:
        raise ValueError("n_boot must be positive")

    if not 0 < alpha < 1:
        raise ValueError("alpha must be between 0 and 1")

    rng = np.random.default_rng(seed)

    point = float(values.mean())

    indices = rng.integers(
        0,
        len(values),
        size=(n_boot, len(values)),
    )

    boot_means = values[indices].mean(axis=1)

    lo = float(np.quantile(boot_means, alpha / 2))
    hi = float(np.quantile(boot_means, 1 - alpha / 2))

    return point, lo, hi


def paired_bootstrap_diff(a, b, *, n_boot=2000, seed=42, alpha=0.05):
    """Return the mean paired difference and its bootstrap interval."""
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)

    if len(a) != len(b):
        raise ValueError("a and b must have the same length")

    if len(a) == 0:
        raise ValueError("a and b must not be empty")

    if n_boot <= 0:
        raise ValueError("n_boot must be positive")

    if not 0 < alpha < 1:
        raise ValueError("alpha must be between 0 and 1")

    rng = np.random.default_rng(seed)

    differences = a - b
    delta = float(differences.mean())

    indices = rng.integers(
        0,
        len(differences),
        size=(n_boot, len(differences)),
    )

    boot_deltas = differences[indices].mean(axis=1)

    lo = float(np.quantile(boot_deltas, alpha / 2))
    hi = float(np.quantile(boot_deltas, 1 - alpha / 2))

    return delta, lo, hi