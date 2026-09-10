"""Deterministic F9/F21 preflight. Never runs an optimizer or writes checkpoints."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
from opfunu.cec_based import cec2017

from cec2017_corrections import (
    CORRECTED_CLASSES, LOWER_BOUND_TOLERANCE, OBJECTIVE_REVISIONS,
    assert_objective_lower_bound,
)
from cec2017_gpu import CEC2017GpuObjective, verification_vectors

REFERENCE_FIXTURE = Path(__file__).parent / "tests/fixtures/cec2017_f9_f21_reference.json"


def reference_fixture(benchmark):
    """Frozen outputs from the author's compiled C routines, not Python outputs."""
    data = json.loads(REFERENCE_FIXTURE.read_text())["functions"][type(benchmark).__name__]
    support_hash = hashlib.sha256(
        np.ascontiguousarray(benchmark.f_shift).tobytes()
        + np.ascontiguousarray(benchmark.f_matrix).tobytes()
    ).hexdigest()
    if support_hash != data["support_data_sha256"]:
        raise RuntimeError("CEC2017 support data differs from the audited C reference fixture")
    return np.asarray(data["points"]), np.asarray(data["expected"])


def audit_vectors(benchmark, random_points=2048):
    points = list(verification_vectors(benchmark, random_points))
    rng = np.random.default_rng(9212017)
    shifts = np.asarray(benchmark.f_shift).reshape(-1, benchmark.ndim)
    for shift in shifts:
        for radius in (1e-7, 1e-3, 0.1, 1.0, 10.0):
            points.extend(shift + rng.uniform(-radius, radius, (16, benchmark.ndim)))
    # Inverse-transform branch boundaries; filter out-of-domain vectors instead
    # of clipping them, which would destroy the intended branch probes.
    component = 0 if benchmark.__class__.__name__ == "F92017" else 2
    matrix = benchmark.f_matrix[component * benchmark.ndim:(component + 1) * benchmark.ndim]
    for boundary in (-1000.0, -500.0, 500.0, 1000.0):
        for epsilon in (-1e-7, 0.0, 1e-7):
            # Schwefel's modulo extension jumps at +/-1000. An inverse rotation
            # cannot place a float exactly on that surface on every BLAS backend.
            # Probe both sides here; test exact boundaries on the kernel itself.
            if abs(boundary) == 1000.0 and epsilon == 0.0:
                continue
            for coordinate in range(benchmark.ndim):
                z = np.zeros(benchmark.ndim)
                z[coordinate] = boundary + epsilon - 420.9687462275036
                points.append(shifts[component] + np.linalg.solve(matrix, z) / 10.0)
    # Old positive-wrap false minima: all Schwefel coordinates at 579.031... .
    offset = (1000.0 - 2.0 * 420.9687462275036) / 10.0
    if component == 0:
        points.append(shifts[0] + np.linalg.solve(matrix, np.full(benchmark.ndim, offset)))
    else:
        points.append(shifts[2] + offset)  # upstream F21 omitted rotation
    points = np.asarray(points)
    valid = np.all((points >= benchmark.lb) & (points <= benchmark.ub), axis=1)
    return np.ascontiguousarray(points[valid])


def check_values(name, backend, values, points, benchmark, reference=None):
    values = np.asarray(values, dtype=np.float64).reshape(-1)
    assert_objective_lower_bound(name, values, points, benchmark.f_global, backend=backend)
    if reference is not None:
        matches = np.isclose(values, reference, rtol=5e-11, atol=1e-7)
        if not np.all(matches):
            i = int(np.flatnonzero(~matches)[0])
            raise RuntimeError(f"CEC preflight disagreement: {name}, backend={backend}, "
                               f"index={i}, actual={values[i]}, scalar={reference[i]}, "
                               f"x={points[i].tolist()}")
    return {"minimum": float(values.min()), "below_global": 0,
            "max_absolute_error": 0.0 if reference is None else float(np.max(np.abs(values-reference)))}


def validate_corrections(function_names=None, ndim=30, require_gpu=True, random_points=2048):
    names = list(CORRECTED_CLASSES if function_names is None else function_names)
    if not names or any(name not in CORRECTED_CLASSES for name in names):
        raise ValueError("Preflight supports only F92017 and F212017")
    cp = None
    gpu_error = None
    try:
        import cupy as cp
        if cp.cuda.runtime.getDeviceCount() < 1:
            raise RuntimeError("No CUDA device available")
        cp.zeros(1).sum().item()
    except Exception as exc:
        cp = None
        gpu_error = str(exc)
    reports = {}
    for name in names:
        benchmark = CORRECTED_CLASSES[name](ndim=ndim)
        points = audit_vectors(benchmark, random_points)
        oracle = None
        if ndim == 30:
            oracle_points, oracle = reference_fixture(benchmark)
            points = np.concatenate((oracle_points, points))
        scalar = np.asarray([benchmark.evaluate(x) for x in points])
        numpy_objective = CEC2017GpuObjective(name, benchmark, np)
        numpy_values = numpy_objective.evaluate(points)
        optimum = float(benchmark.evaluate(benchmark.x_global))
        if not np.isclose(optimum, benchmark.f_global, rtol=0, atol=LOWER_BOUND_TOLERANCE):
            raise RuntimeError(f"{name}: f(x_global)={optimum}, f_global={benchmark.f_global}")
        raw = getattr(cec2017, name)(ndim=ndim)
        upstream = np.asarray([raw.evaluate(x) for x in points])
        report = {
            "revision": OBJECTIVE_REVISIONS[name], "D": ndim, "points": len(points),
            "f_global": benchmark.f_global, "x_global": benchmark.x_global.tolist(),
            "f_at_x_global": optimum, "tolerance": LOWER_BOUND_TOLERANCE,
            "cpu": check_values(name, "scalar", scalar, points, benchmark),
            "numpy": check_values(name, "numpy", numpy_values, points, benchmark, scalar),
            "stock_opfunu": {
                "minimum": float(upstream.min()),
                "below_global": int(np.count_nonzero(upstream < benchmark.f_global - LOWER_BOUND_TOLERANCE)),
                "disagreements_with_corrected": int(np.count_nonzero(~np.isclose(upstream, scalar, rtol=5e-11, atol=1e-7))),
                "worst_point": points[int(upstream.argmin())].tolist(),
            },
        }
        if cp is not None:
            gpu = CEC2017GpuObjective(name, benchmark, cp)
            values = cp.asnumpy(gpu.evaluate(cp.asarray(points)))
            report["gpu"] = check_values(name, "gpu", values, points, benchmark, scalar)
            report["gpu"]["max_numpy_absolute_error"] = float(np.max(np.abs(values-numpy_values)))
            report["gpu"]["device"] = str(cp.cuda.runtime.getDeviceProperties(0)["name"])
        else:
            report["gpu"] = {"unavailable": gpu_error}
        if oracle is not None:
            report["reference_c"] = {}
            backends = {"cpu": scalar, "numpy": numpy_values}
            if cp is not None:
                backends["gpu"] = values
            for backend, actual in backends.items():
                report["reference_c"][backend] = check_values(
                    name, backend + " vs reference C", actual[:len(oracle)],
                    oracle_points, benchmark, oracle,
                )
        reports[name] = report
        print(json.dumps({name: report}, sort_keys=True), flush=True)
    if require_gpu and cp is None:
        raise RuntimeError(f"F9/F21 rerun blocked: real CPU/NumPy/GPU equivalence is required; {gpu_error}")
    return reports


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cpu-only", action="store_true", help="Allow CPU-only validation when CUDA is unavailable; does not authorize a rerun")
    parser.add_argument("--dims", type=int, default=30)
    parser.add_argument("--random-points", type=int, default=2048)
    args = parser.parse_args()
    validate_corrections(ndim=args.dims, require_gpu=not args.cpu_only, random_points=args.random_points)
