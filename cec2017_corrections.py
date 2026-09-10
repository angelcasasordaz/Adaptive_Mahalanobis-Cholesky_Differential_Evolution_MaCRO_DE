"""Scoped F9/F21 corrections using OPFUNU's declared data, names and biases.

Formulas: P-N-Suganthan/CEC2017-BoundContrained, CEC17_fast_pow-C++.zip,
cec17_test_func.cpp: schwefel_func, cf02, rastrigin_func, griewank_func.
Do not patch opfunu.utils.operator: other historical objectives must stay intact.
"""
from __future__ import annotations

import math
import numpy as np
from opfunu.cec_based import cec2017
from opfunu.utils import operator

OBJECTIVE_REVISIONS = {
    "F92017": "schwefel-positive-wrap-v1",
    "F212017": "cf02-scales-rotation-schwefel-v1",
}
LOWER_BOUND_TOLERANCE = 1.0e-7


def objective_revision_metadata(benchmark, function_name):
    revision = OBJECTIVE_REVISIONS.get(function_name) if benchmark == "CEC2017" else None
    return {"objective_implementation_revision": revision} if revision else {}


def assert_objective_lower_bound(function_name, values, points, f_global, *,
                                 backend="cpu", xp=np, tolerance=LOWER_BOUND_TOLERANCE):
    """Reject invalid fitness without modifying it; include the offending vector."""
    values = xp.asarray(values)
    invalid = ~xp.isfinite(values) | (values < f_global - tolerance)
    if bool(xp.any(invalid)):
        index = int(xp.flatnonzero(invalid)[0])
        point = xp.asarray(points).reshape(-1, xp.asarray(points).shape[-1])[index]
        raise RuntimeError(
            f"CEC objective lower-bound validation failed: function={function_name}, "
            f"revision={OBJECTIVE_REVISIONS.get(function_name)}, backend={backend}, "
            f"D={point.size}, index={index}, f={float(values.reshape(-1)[index]):.17g}, "
            f"f_global={f_global:.17g}, tolerance={tolerance}, x={point.tolist()}"
        )


def corrected_schwefel(z):
    """Independent scalar implementation of the three reference C branches."""
    total = 0.0
    n = len(z)
    for coordinate in z:
        value = float(coordinate) + 420.9687462275036
        if value > 500.0:
            wrapped = 500.0 - math.fmod(value, 500.0)
            total -= wrapped * math.sin(math.sqrt(wrapped))
            total += ((value - 500.0) / 100.0) ** 2 / n
        elif value < -500.0:
            remainder = math.fmod(math.fabs(value), 500.0)
            total -= (-500.0 + remainder) * math.sin(math.sqrt(500.0 - remainder))
            total += ((value + 500.0) / 100.0) ** 2 / n
        else:
            total -= value * math.sin(math.sqrt(math.fabs(value)))
    return total + 418.9828872724338 * n


class F92017(cec2017.F92017):
    def evaluate(self, x, *args):
        self.n_fe += 1
        self.check_solution(x, self.dim_max, self.dim_supported)
        x = np.asarray(x, dtype=np.float64)
        z = self.f_matrix @ (10.0 * (x - self.f_shift))
        value = corrected_schwefel(z) + self.f_bias
        assert_objective_lower_bound("F92017", value, x, self.f_global)
        return value


class F212017(cec2017.F212017):
    def evaluate(self, x, *args):
        self.n_fe += 1
        self.check_solution(x, self.dim_max, self.dim_supported)
        x = np.asarray(x, dtype=np.float64)
        scales = (5.12 / 100.0, 600.0 / 100.0, 1000.0 / 100.0)
        operators = (operator.rastrigin_func, operator.griewank_func, corrected_schwefel)
        values, weights = [], []
        for i, (scale, operation) in enumerate(zip(scales, operators)):
            delta = x - self.f_shift[i]
            z = self.f_matrix[i * self.ndim:(i + 1) * self.ndim] @ (scale * delta)
            values.append(self.lamdas[i] * operation(z) + self.bias[i])
            weights.append(operator.calculate_weight(delta, self.xichmas[i]))
        weights = np.asarray(weights)
        value = float((weights / weights.sum()) @ values + self.f_bias)
        assert_objective_lower_bound("F212017", value, x, self.f_global)
        return value


CORRECTED_CLASSES = {"F92017": F92017, "F212017": F212017}


def corrected_benchmark(benchmark):
    """Select the scoped scalar reference while retaining the exact support data."""
    cls = CORRECTED_CLASSES.get(type(benchmark).__name__)
    if cls is None or isinstance(benchmark, cls):
        return benchmark
    if not type(benchmark).__module__.startswith("opfunu.cec_based.cec2017"):
        return benchmark
    return cls(ndim=benchmark.ndim, f_shift=benchmark.f_shift,
               f_matrix=benchmark.f_matrix, f_bias=benchmark.f_bias)
