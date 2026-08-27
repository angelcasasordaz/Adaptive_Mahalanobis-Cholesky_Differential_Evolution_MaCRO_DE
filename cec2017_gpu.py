"""Verified, repository-local CuPy implementations of staged CEC2017 objectives.

Nothing in this module imports CuPy directly.  The caller supplies the active
array module, keeping the Windows/CPU path independent of CUDA.  A candidate is
usable only after its device result passes :func:`verify_gpu_objective` against
the instantiated OPFUNU benchmark and its exact support data.
"""
from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter
from typing import Any

import numpy as np


CEC_GPU_VERSION = "cec2017-complete-v3"
CEC2017_GPU_CANDIDATES = frozenset(
    {"F12017", "F32017", "F42017", "F52017", "F62017", "F72017", "F82017", "F92017"}
    | {f"F{index}2017" for index in range(10, 30)}
)


@dataclass(frozen=True)
class GpuObjectiveVerification:
    function_name: str
    verified: bool
    points: int
    max_absolute_error: float
    max_relative_error: float
    mismatches: int
    gpu_seconds: float
    reason: str = ""


class CEC2017GpuObjective:
    """Vectorized objective over ``(..., dimensions)`` using OPFUNU data."""

    def __init__(self, function_name: str, benchmark: Any, xp: Any):
        if function_name not in CEC2017_GPU_CANDIDATES:
            raise ValueError(f"{function_name} has no staged GPU implementation")
        self.function_name = function_name
        self.xp = xp
        self.ndim = int(benchmark.ndim)
        self.lb_cpu = np.asarray(benchmark.lb, dtype=np.float64)
        self.ub_cpu = np.asarray(benchmark.ub, dtype=np.float64)
        self.shift_cpu = np.asarray(benchmark.f_shift, dtype=np.float64)
        self.matrix_cpu = np.asarray(benchmark.f_matrix, dtype=np.float64)
        self.shift = xp.asarray(self.shift_cpu, dtype=xp.float64)
        self.matrix_t = xp.asarray(self.matrix_cpu.T, dtype=xp.float64)
        self.bias = float(benchmark.f_bias)
        self.composition_shifts = xp.asarray(self.shift_cpu, dtype=xp.float64)
        self.composition_matrices_t = [
            xp.asarray(
                self.matrix_cpu[index * self.ndim:(index + 1) * self.ndim].T,
                dtype=xp.float64,
            )
            for index in range(int(getattr(benchmark, "n_funcs", 0)))
        ]
        self.composition_sigmas = tuple(float(value) for value in getattr(benchmark, "xichmas", ()))
        self.composition_lambdas = tuple(float(value) for value in getattr(benchmark, "lamdas", ()))
        self.composition_biases = tuple(float(value) for value in getattr(benchmark, "bias", ()))
        self.indices = [
            xp.asarray(getattr(benchmark, f"idx{index}"), dtype=xp.int64)
            for index in range(1, 7)
            if hasattr(benchmark, f"idx{index}")
        ]
        self.composition_hybrids = []
        if function_name in {"F282017", "F292017"}:
            for child_index in range(3):
                child = getattr(benchmark, f"g{child_index}")
                child_indices = [
                    xp.asarray(getattr(child, f"idx{index}"), dtype=xp.int64)
                    for index in range(1, 7)
                    if hasattr(child, f"idx{index}")
                ]
                self.composition_hybrids.append(
                    (
                        xp.asarray(np.asarray(child.f_shift, dtype=np.float64), dtype=xp.float64),
                        xp.asarray(np.asarray(child.f_matrix, dtype=np.float64).T, dtype=xp.float64),
                        child_indices,
                    )
                )
        self.verification: GpuObjectiveVerification | None = None

    def _rotate(self, x, scale: float = 1.0):
        return ((x - self.shift) * scale) @ self.matrix_t

    def _rosenbrock(self, z, shift=0.0):
        z = z + shift
        return self.xp.sum(
            100.0 * (z[..., :-1] ** 2 - z[..., 1:]) ** 2
            + (z[..., :-1] - 1.0) ** 2,
            axis=-1,
        )

    def _rastrigin(self, z):
        return self.xp.sum(z**2 - 10.0 * self.xp.cos(2.0 * self.xp.pi * z) + 10.0, axis=-1)

    def _griewank(self, z):
        indices = self.xp.arange(1, z.shape[-1] + 1, dtype=self.xp.float64)
        return (
            self.xp.sum(z**2, axis=-1) / 4000.0
            - self.xp.prod(self.xp.cos(z / self.xp.sqrt(indices)), axis=-1)
            + 1.0
        )

    def _bent_cigar(self, z):
        return z[..., 0] ** 2 + 1.0e6 * self.xp.sum(z[..., 1:] ** 2, axis=-1)

    def _zakharov(self, z):
        temp = self.xp.sum(0.5 * z, axis=-1)
        return self.xp.sum(z**2, axis=-1) + temp**2 + temp**4

    def _elliptic(self, z):
        ndim = z.shape[-1]
        weights = 10.0 ** (6.0 * self.xp.arange(ndim) / (ndim - 1))
        return self.xp.sum(weights * z**2, axis=-1)

    def _ackley(self, z):
        ndim = z.shape[-1]
        return (
            -20.0 * self.xp.exp(-0.2 * self.xp.sqrt(self.xp.sum(z**2, axis=-1) / ndim))
            - self.xp.exp(self.xp.sum(self.xp.cos(2.0 * self.xp.pi * z), axis=-1) / ndim)
            + 20.0
            + np.e
        )

    def _schaffer_f7(self, z):
        pair = z[..., :-1] ** 2 + z[..., 1:] ** 2
        terms = self.xp.sqrt(pair) * (self.xp.sin(50.0 * pair**0.2) + 1.0)
        return (self.xp.sum(terms, axis=-1) / (z.shape[-1] - 1)) ** 2

    def _lunacek(self, z, miu0=2.5, d=1.0, shift=0.0):
        z = z + shift
        ndim = z.shape[-1]
        s = 1.0 - 1.0 / (2.0 * np.sqrt(ndim + 20.0) - 8.2)
        miu1 = -np.sqrt((miu0**2 - d) / s)
        delta = z - miu0
        return self.xp.minimum(
            self.xp.sum(delta**2, axis=-1),
            s * self.xp.sum((z - miu1) ** 2, axis=-1) + d * ndim,
        ) + 10.0 * (ndim - self.xp.sum(self.xp.cos(2.0 * self.xp.pi * delta), axis=-1))

    def _schwefel(self, z):
        xp = self.xp
        ndim = z.shape[-1]
        z = z + 420.9687462275036
        high, low = z > 500.0, z < -500.0
        remainder = xp.fmod(xp.abs(z), 500.0)
        high_term = -(
            (500.0 + remainder) * xp.sin(xp.sqrt(500.0 - remainder))
            - ((z - 500.0) / 100.0) ** 2 / ndim
        )
        low_term = -(
            (-500.0 + remainder) * xp.sin(xp.sqrt(500.0 - remainder))
            - ((z + 500.0) / 100.0) ** 2 / ndim
        )
        middle = -z * xp.sin(xp.sqrt(xp.abs(z)))
        return xp.sum(xp.where(high, high_term, xp.where(low, low_term, middle)), axis=-1) + 418.9828872724338 * ndim

    def _hgbat(self, z, shift=0.0):
        z = z + shift
        ndim = z.shape[-1]
        t1, t2 = self.xp.sum(z, axis=-1), self.xp.sum(z**2, axis=-1)
        return self.xp.abs(t2**2 - t1**2) ** 0.5 + (0.5 * t2 + t1) / ndim + 0.5

    def _happy_cat(self, z, shift=0.0):
        z = z + shift
        ndim = z.shape[-1]
        t1, t2 = self.xp.sum(z, axis=-1), self.xp.sum(z**2, axis=-1)
        return self.xp.abs(t2 - ndim) ** 0.25 + (0.5 * t2 + t1) / ndim + 0.5

    def _discus(self, z):
        return 1.0e6 * z[..., 0] ** 2 + self.xp.sum(z[..., 1:] ** 2, axis=-1)

    def _expanded_schaffer(self, z):
        shifted = self.xp.roll(z, -1, axis=-1)
        pair = z**2 + shifted**2
        return self.xp.sum(
            0.5 + (self.xp.sin(self.xp.sqrt(pair)) ** 2 - 0.5) / (1.0 + 0.001 * pair) ** 2,
            axis=-1,
        )

    def _grie_rosen(self, z):
        z = z + 1.0
        shifted = self.xp.roll(z, -1, axis=-1)
        temp = 100.0 * (z**2 - shifted) ** 2 + (z - 1.0) ** 2
        return self.xp.sum(temp**2 / 4000.0 - self.xp.cos(temp) + 1.0, axis=-1)

    def _weierstrass_norm(self, z):
        k = self.xp.arange(21, dtype=self.xp.float64)
        terms = 0.5**k * self.xp.cos(
            2.0 * self.xp.pi * 3.0**k * (z[..., :, None] + 0.5)
        )
        constant = self.xp.sum(0.5**k * self.xp.cos(self.xp.pi * 3.0**k))
        return self.xp.sum(terms, axis=(-2, -1)) - z.shape[-1] * constant

    def _katsuura(self, z):
        xp = self.xp
        ndim = z.shape[-1]
        powers = 2.0 ** xp.arange(1, 33, dtype=xp.float64)
        scaled = z[..., :, None] * powers
        temp = xp.sum(xp.abs(scaled - xp.round(scaled)) / powers, axis=-1)
        indices = xp.arange(1, ndim + 1, dtype=xp.float64)
        product = xp.prod((1.0 + indices * temp) ** (10.0 / ndim**1.2), axis=-1)
        return (product - 1.0) * 10.0 / ndim**2

    def _hybrid(self, mz, operators):
        return self._hybrid_with(mz, self.indices, operators)

    def _hybrid_with(self, mz, indices, operators):
        result = self.xp.zeros(mz.shape[:-1], dtype=self.xp.float64)
        for group, operation in zip(indices, operators):
            result = result + operation(mz[..., group])
        return result

    def _composition_weight(self, delta, sigma):
        squared_norm = self.xp.sum(delta**2, axis=-1)
        safe_norm = self.xp.where(squared_norm != 0.0, squared_norm, 1.0)
        regular = self.xp.sqrt(1.0 / safe_norm) * self.xp.exp(
            -safe_norm / (2.0 * self.ndim * sigma**2)
        )
        return self.xp.where(squared_norm != 0.0, regular, 1.0e99)

    def _composition(self, x, values):
        weights = self.xp.stack(
            [
                self._composition_weight(x - self.composition_shifts[index], sigma)
                for index, sigma in enumerate(self.composition_sigmas)
            ],
            axis=-1,
        )
        weights = weights / self.xp.sum(weights, axis=-1, keepdims=True)
        components = self.xp.stack(
            [
                self.composition_lambdas[index] * value + self.composition_biases[index]
                for index, value in enumerate(values)
            ],
            axis=-1,
        )
        return self.xp.sum(weights * components, axis=-1)

    def _composition_rotate(self, x, component, scale=1.0, shift_component=None, add=0.0):
        shift_index = component if shift_component is None else shift_component
        return (
            (x - self.composition_shifts[shift_index]) * scale
        ) @ self.composition_matrices_t[component] + add

    def evaluate(self, x):
        xp = self.xp
        x = xp.asarray(x, dtype=xp.float64)
        if x.shape[-1] != self.ndim:
            raise ValueError(f"Expected last dimension {self.ndim}, got {x.shape}")
        name = self.function_name
        if name == "F12017":
            z = self._rotate(x)
            value = z[..., 0] ** 2 + 1.0e6 * xp.sum(z[..., 1:] ** 2, axis=-1)
        elif name == "F32017":
            z = self._rotate(x, 2.048 / 100.0) + 1.0
            value = xp.sum(
                100.0 * (z[..., :-1] ** 2 - z[..., 1:]) ** 2
                + (z[..., :-1] - 1.0) ** 2,
                axis=-1,
            )
        elif name == "F42017":
            z = self._rotate(x)
            value = xp.sum(z**2 - 10.0 * xp.cos(2.0 * xp.pi * z) + 10.0, axis=-1)
        elif name == "F52017":
            z = self._rotate(x, 0.5 / 100.0)
            pair = z[..., :-1] ** 2 + z[..., 1:] ** 2
            terms = xp.sqrt(pair) * (xp.sin(50.0 * pair**0.2) + 1.0)
            value = (xp.sum(terms, axis=-1) / (self.ndim - 1)) ** 2
        elif name == "F62017":
            z = self._rotate(x, 600.0 / 100.0) + 2.5
            s = 1.0 - 1.0 / (2.0 * np.sqrt(self.ndim + 20.0) - 8.2)
            miu1 = -np.sqrt((2.5**2 - 1.0) / s)
            delta = z - 2.5
            term1 = xp.sum(delta**2, axis=-1)
            term2 = s * xp.sum((z - miu1) ** 2, axis=-1) + self.ndim
            value = xp.minimum(term1, term2) + 10.0 * (
                self.ndim - xp.sum(xp.cos(2.0 * xp.pi * delta), axis=-1)
            )
        elif name == "F72017":
            z = self._rotate(x, 5.12 / 100.0)
            twice = 2.0 * z
            fractional, integral = xp.modf(twice)
            rounded = xp.where(twice <= 0.0, integral - (fractional >= 0.5), twice)
            rounded = xp.where(fractional < 0.5, integral, rounded)
            rounded = xp.where(fractional >= 0.5, integral + 1.0, rounded) / 2.0
            y = xp.where(xp.abs(z) < 0.5, z, rounded)
            # OPFUNU column-stacks (y, roll(y)) then flattens inside rastrigin,
            # which is exactly two copies of every y coordinate.
            value = 2.0 * xp.sum(y**2 - 10.0 * xp.cos(2.0 * xp.pi * y) + 10.0, axis=-1)
        elif name == "F82017":
            z = self._rotate(x, 5.12 / 100.0) + 1.0
            w = 1.0 + (z - 1.0) / 4.0
            first_last = xp.sin(xp.pi * w[..., 0]) ** 2 + (w[..., -1] - 1.0) ** 2 * (
                1.0 + xp.sin(2.0 * xp.pi * w[..., -1]) ** 2
            )
            middle = xp.sum(
                (w[..., :-1] - 1.0) ** 2
                * (1.0 + 10.0 * xp.sin(xp.pi * w[..., :-1] + 1.0) ** 2),
                axis=-1,
            )
            value = first_last + middle
        elif name == "F92017":
            z = self._rotate(x, 1000.0 / 100.0) + 420.9687462275036
            high, low = z > 500.0, z < -500.0
            high_mod = xp.fmod(xp.abs(z), 500.0)
            low_mod = high_mod
            high_term = -(
                (500.0 + high_mod) * xp.sin(xp.sqrt(500.0 - high_mod))
                - ((z - 500.0) / 100.0) ** 2 / self.ndim
            )
            low_term = -(
                (-500.0 + low_mod) * xp.sin(xp.sqrt(500.0 - low_mod))
                - ((z + 500.0) / 100.0) ** 2 / self.ndim
            )
            middle_term = -z * xp.sin(xp.sqrt(xp.abs(z)))
            value = xp.sum(xp.where(high, high_term, xp.where(low, low_term, middle_term)), axis=-1)
            value = value + 418.9828872724338 * self.ndim
        elif name == "F102017":
            mz = self._rotate(x)
            value = self._hybrid(mz, (self._zakharov, lambda z: self._rosenbrock(z, 1.0), self._rastrigin))
        elif name == "F112017":
            mz = self._rotate(x)
            value = self._hybrid(mz, (self._elliptic, self._schwefel, self._bent_cigar))
        elif name == "F122017":
            mz = self._rotate(x)
            value = self._hybrid(mz, (self._bent_cigar, lambda z: self._rosenbrock(z, 1.0), lambda z: self._lunacek(z, 2.5, 1.0, 2.5)))
        elif name == "F132017":
            mz = self._rotate(x)
            value = self._hybrid(mz, (self._elliptic, self._ackley, self._schaffer_f7, self._rastrigin))
        elif name == "F142017":
            mz = self._rotate(x)
            value = self._hybrid(mz, (self._bent_cigar, lambda z: self._hgbat(z, -1.0), self._rastrigin, lambda z: self._rosenbrock(z, 1.0)))
        elif name == "F152017":
            mz = self._rotate(x)
            value = self._hybrid(mz, (self._expanded_schaffer, lambda z: self._hgbat(z, -1.0), lambda z: self._rosenbrock(z, 1.0), self._schwefel))
        elif name == "F162017":
            mz = self._rotate(x)
            value = self._hybrid(mz, (self._katsuura, self._ackley, self._grie_rosen, self._schwefel, self._rastrigin))
        elif name == "F172017":
            mz = self._rotate(x)
            value = self._hybrid(mz, (self._elliptic, self._ackley, self._rastrigin, lambda z: self._hgbat(z, -1.0), self._discus))
        elif name == "F182017":
            mz = self._rotate(x)
            value = self._hybrid(mz, (self._bent_cigar, self._rastrigin, self._grie_rosen, self._weierstrass_norm, self._expanded_schaffer))
        elif name == "F192017":
            mz = self._rotate(x)
            value = self._hybrid(mz, (lambda z: self._happy_cat(z, -1.0), self._katsuura, self._ackley, self._rastrigin, self._schwefel, self._schaffer_f7))
        elif name == "F202017":
            value = self._composition(x, (
                self._rosenbrock(self._composition_rotate(x, 0, 2.048 / 100.0, add=1.0)),
                self._elliptic(self._composition_rotate(x, 1)),
                self._rastrigin(self._composition_rotate(x, 2)),
            ))
        elif name == "F212017":
            value = self._composition(x, (
                self._rastrigin(self._composition_rotate(x, 0)),
                self._griewank(self._composition_rotate(x, 1)),
                self._schwefel((x - self.composition_shifts[2]) * 10.0),
            ))
        elif name == "F222017":
            value = self._composition(x, (
                self._rosenbrock(self._composition_rotate(x, 0, 2.048 / 100.0, add=1.0)),
                self._ackley(self._composition_rotate(x, 1)),
                self._schwefel(self._composition_rotate(x, 2)),
                self._rastrigin(self._composition_rotate(x, 3)),
            ))
        elif name == "F232017":
            value = self._composition(x, (
                self._ackley(self._composition_rotate(x, 0)),
                self._elliptic(self._composition_rotate(x, 1)),
                self._griewank(self._composition_rotate(x, 2)),
                self._rastrigin(self._composition_rotate(x, 3)),
            ))
        elif name == "F242017":
            value = self._composition(x, (
                self._rastrigin(self._composition_rotate(x, 0, shift_component=0)),
                self._happy_cat(self._composition_rotate(x, 1, shift_component=0)),
                self._ackley(self._composition_rotate(x, 2, shift_component=0)),
                self._discus(self._composition_rotate(x, 3, shift_component=0)),
                self._rosenbrock(self._composition_rotate(x, 4, 2.048 / 100.0, 0, 1.0)),
            ))
        elif name == "F252017":
            value = self._composition(x, (
                self._expanded_schaffer(self._composition_rotate(x, 0, shift_component=0, add=1.0)),
                self._schwefel(self._composition_rotate(x, 1, 10.0, 0)),
                self._griewank(self._composition_rotate(x, 2, 6.0, 0)),
                self._rosenbrock(self._composition_rotate(x, 3, 2.048 / 100.0, 0, 1.0)),
                self._rastrigin(self._composition_rotate(x, 4, shift_component=0)),
            ))
        elif name == "F262017":
            value = self._composition(x, (
                self._hgbat(self._composition_rotate(x, 0, 0.05, 0), -1.0),
                self._rastrigin(self._composition_rotate(x, 1, 5.12 / 100.0, 0)),
                self._schwefel(self._composition_rotate(x, 2, 10.0, 0)),
                self._bent_cigar(self._composition_rotate(x, 3, shift_component=0)),
                self._elliptic(self._composition_rotate(x, 4, shift_component=0)),
                self._expanded_schaffer(self._composition_rotate(x, 5, shift_component=0, add=1.0)),
            ))
        elif name == "F272017":
            value = self._composition(x, (
                self._ackley(self._composition_rotate(x, 0, shift_component=0)),
                self._griewank(self._composition_rotate(x, 1, 6.0, 0)),
                self._discus(self._composition_rotate(x, 2, shift_component=0)),
                self._rosenbrock(self._composition_rotate(x, 3, 2.048 / 100.0, 0, 1.0)),
                self._happy_cat(self._composition_rotate(x, 4, 0.05, 0)),
                self._expanded_schaffer(self._composition_rotate(x, 5, shift_component=0, add=1.0)),
            ))
        elif name in {"F282017", "F292017"}:
            hybrid_operations = (
                (
                    (self._bent_cigar, lambda z: self._hgbat(z, -1.0), self._rastrigin, lambda z: self._rosenbrock(z, 1.0)),
                    (self._expanded_schaffer, lambda z: self._hgbat(z, -1.0), lambda z: self._rosenbrock(z, 1.0), self._schwefel),
                    (self._katsuura, self._ackley, self._grie_rosen, self._schwefel, self._rastrigin),
                )
                if name == "F282017"
                else (
                    (self._bent_cigar, lambda z: self._hgbat(z, -1.0), self._rastrigin, lambda z: self._rosenbrock(z, 1.0)),
                    (self._elliptic, self._ackley, self._rastrigin, lambda z: self._hgbat(z, -1.0), self._discus),
                    (self._bent_cigar, self._rastrigin, self._grie_rosen, self._weierstrass_norm, self._expanded_schaffer),
                )
            )
            hybrid_values = []
            for (shift, matrix_t, indices), operators in zip(self.composition_hybrids, hybrid_operations):
                hybrid_values.append(self._hybrid_with((x - shift) @ matrix_t, indices, operators))
            value = self._composition(x, hybrid_values)
        else:  # pragma: no cover - constructor prevents this
            raise AssertionError(name)
        return value + self.bias


def verification_vectors(benchmark: Any, random_points: int = 512) -> np.ndarray:
    """Deterministic coverage without consuming an experimental RNG stream."""
    lb = np.asarray(benchmark.lb, dtype=np.float64)
    ub = np.asarray(benchmark.ub, dtype=np.float64)
    midpoint = (lb + ub) / 2.0
    shifts = np.asarray(benchmark.f_shift, dtype=np.float64).reshape(-1, len(lb))
    shifts = np.clip(shifts, lb, ub)
    fixed = np.concatenate(
        (np.stack((np.zeros_like(lb), lb, ub, midpoint), axis=0), shifts),
        axis=0,
    )
    rng = np.random.default_rng(0xCEC2017)
    random = rng.uniform(lb, ub, size=(max(1, int(random_points)), len(lb)))
    return np.ascontiguousarray(np.concatenate((fixed, random), axis=0))


def verify_gpu_objective(
    objective: CEC2017GpuObjective,
    benchmark: Any,
    to_cpu,
    synchronize,
    random_points: int = 512,
    rtol: float = 5.0e-11,
    atol: float = 1.0e-7,
) -> GpuObjectiveVerification:
    points = verification_vectors(benchmark, random_points)
    reference = np.asarray([float(benchmark.evaluate(row)) for row in points], dtype=np.float64)
    device_points = objective.xp.asarray(points, dtype=objective.xp.float64)
    synchronize()
    started = perf_counter()
    actual = np.asarray(to_cpu(objective.evaluate(device_points)), dtype=np.float64).reshape(-1)
    synchronize()
    gpu_seconds = perf_counter() - started
    absolute = np.abs(actual - reference)
    relative = absolute / np.maximum(np.abs(reference), 1.0)
    finite_values = np.isfinite(actual) & np.isfinite(reference)
    matches = finite_values & np.isclose(actual, reference, rtol=rtol, atol=atol)
    finite_error = np.isfinite(absolute)
    report = GpuObjectiveVerification(
        objective.function_name,
        bool(np.all(matches)),
        len(points),
        float(np.max(absolute[finite_error], initial=0.0)),
        float(np.max(relative[finite_error], initial=0.0)),
        int(np.count_nonzero(~matches)),
        gpu_seconds,
        "" if np.all(matches) else "strict finite OPFUNU comparison failed",
    )
    objective.verification = report
    return report
