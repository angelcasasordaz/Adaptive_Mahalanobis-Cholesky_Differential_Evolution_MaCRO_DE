"""Validated MEALPY GPU adapters ported from the Diversity project.

MEALPY control flow, RNG, Agent objects, and fitness evaluation remain on the
CPU. Ordered NumPy array operations on agent state are dispatched through the
single CUDA-owning strict-GPU worker.
"""
from __future__ import annotations

import numpy as np
from mealpy.evolutionary_based.DE import OriginalDE
from mealpy.evolutionary_based.SHADE import OriginalSHADE
from mealpy.human_based.BRO import OriginalBRO
from mealpy.swarm_based.HHO import OriginalHHO
from mealpy.swarm_based.PSO import OriginalPSO
from mealpy.swarm_based.WOA import OriginalWOA

from compute_backend import ComputeBackend, GPUBackendError


_LOCAL_GPU_WORKER_BACKEND = None


def configure_local_mealpy_gpu_backend(device_id=0) -> None:
    """Retain the strict worker's already initialized CUDA backend."""
    global _LOCAL_GPU_WORKER_BACKEND
    _LOCAL_GPU_WORKER_BACKEND = ComputeBackend("gpu", device_id=device_id)


class _ArrayMathExecutor:
    """Execute ordered NumPy-compatible operations on the worker-local GPU."""

    def __init__(self, compute_device="cpu", gpu_device_id=0, gpu_memory_fraction=0.85):
        _ = gpu_memory_fraction
        self.requested_device = str(compute_device).lower()
        if self.requested_device == "gpu":
            if _LOCAL_GPU_WORKER_BACKEND is None:
                raise GPUBackendError(
                    "MEALPY GPU adapters require the initialized strict-GPU worker backend"
                )
            self.backend = _LOCAL_GPU_WORKER_BACKEND
        else:
            self.backend = ComputeBackend("cpu", device_id=gpu_device_id)

    def array_operation(self, kind, name, method, args, kwargs):
        if not self.backend.uses_gpu:
            target = getattr(np, name)
            return (
                getattr(target, method)(*args, **kwargs)
                if kind == "ufunc"
                else target(*args, **kwargs)
            )
        xp = self.backend.xp

        def device_value(value):
            if isinstance(value, np.ndarray):
                return xp.asarray(value)
            if isinstance(value, (list, tuple)):
                return type(value)(device_value(item) for item in value)
            return value

        device_args = tuple(device_value(value) for value in args)
        target = self._resolve_gpu_operation(name)
        result = (
            getattr(target, method)(*device_args, **kwargs)
            if kind == "ufunc"
            else target(*device_args, **kwargs)
        )

        def host_value(value):
            if isinstance(value, tuple):
                return tuple(host_value(item) for item in value)
            return self.backend.to_cpu(value) if isinstance(value, xp.ndarray) else value

        return host_value(result)

    def _resolve_gpu_operation(self, name):
        xp = self.backend.xp
        target = getattr(xp, name, None)
        if target is not None:
            return target
        from cupyx.scipy import special

        target = getattr(special, name, None)
        if target is not None:
            return target
        raise GPUBackendError(
            f"GPU operation '{name}' is unsupported by CuPy and cupyx.scipy.special"
        )


class _GPUArray(np.ndarray):
    """CPU-resident MEALPY state whose ordered array kernels use the GPU."""

    __array_priority__ = 1000

    def __new__(cls, value, executor):
        obj = np.asarray(value).view(cls)
        obj._gpu_executor = executor
        return obj

    def __array_finalize__(self, parent):
        self._gpu_executor = getattr(parent, "_gpu_executor", None)

    @staticmethod
    def _plain(value):
        if isinstance(value, np.ndarray):
            return np.asarray(value)
        if isinstance(value, (list, tuple)):
            return type(value)(_GPUArray._plain(item) for item in value)
        return value

    def _wrap(self, value):
        if isinstance(value, np.ndarray):
            return _GPUArray(value, self._gpu_executor)
        if isinstance(value, tuple):
            return tuple(self._wrap(item) for item in value)
        return value

    def __array_ufunc__(self, ufunc, method, *inputs, **kwargs):
        if kwargs.get("out") is not None:
            outputs = kwargs.pop("out")
            result = self._gpu_executor.array_operation(
                "ufunc", ufunc.__name__, method,
                tuple(self._plain(item) for item in inputs), kwargs,
            )
            values = result if isinstance(result, tuple) else (result,)
            for output, value in zip(outputs, values):
                np.copyto(np.asarray(output), value)
            return outputs[0] if len(outputs) == 1 else outputs
        result = self._gpu_executor.array_operation(
            "ufunc", ufunc.__name__, method,
            tuple(self._plain(item) for item in inputs), kwargs,
        )
        return self._wrap(result)

    def __array_function__(self, func, types, args, kwargs):
        result = self._gpu_executor.array_operation(
            "function", func.__name__, "__call__",
            tuple(self._plain(item) for item in args), kwargs,
        )
        return self._wrap(result)


class _MEALPYGPUAdapter:
    """Preserve MEALPY control/RNG while dispatching array equations in order."""

    def __init__(self, *args, compute_device="cpu", gpu_device_id=0,
                 gpu_memory_fraction=0.85, **kwargs):
        self.compute_device = str(compute_device).lower()
        self._gpu_math = _ArrayMathExecutor(
            self.compute_device, gpu_device_id, gpu_memory_fraction,
        )
        super().__init__(*args, **kwargs)

    def generate_empty_agent(self, solution=None):
        agent = super().generate_empty_agent(solution)
        if self.compute_device == "gpu":
            for name, value in tuple(vars(agent).items()):
                if isinstance(value, np.ndarray) and not isinstance(value, _GPUArray):
                    setattr(agent, name, _GPUArray(value, self._gpu_math))
        return agent


class GPUOriginalDE(_MEALPYGPUAdapter, OriginalDE):
    pass


class GPUOriginalSHADE(_MEALPYGPUAdapter, OriginalSHADE):
    pass


class GPUOriginalPSO(_MEALPYGPUAdapter, OriginalPSO):
    pass


class GPUOriginalWOA(_MEALPYGPUAdapter, OriginalWOA):
    pass


class GPUOriginalHHO(_MEALPYGPUAdapter, OriginalHHO):
    pass


class GPUOriginalBRO(_MEALPYGPUAdapter, OriginalBRO):
    def find_idx_min_distance__(self, target_pos=None, pop=None):
        if self.compute_device != "gpu":
            return super().find_idx_min_distance__(target_pos, pop)
        positions = _GPUArray(
            np.asarray([np.asarray(agent.solution) for agent in pop]), self._gpu_math,
        )
        target = _GPUArray(np.asarray(target_pos).reshape(1, -1), self._gpu_math)
        distances = np.sqrt(np.sum((positions - target) ** 2, axis=1))
        return self.get_idx_min__(np.asarray(distances))


_GPU_ADAPTERS = {
    "OriginalBRO": GPUOriginalBRO,
    "OriginalDE": GPUOriginalDE,
    "OriginalHHO": GPUOriginalHHO,
    "OriginalPSO": GPUOriginalPSO,
    "OriginalSHADE": GPUOriginalSHADE,
    "OriginalWOA": GPUOriginalWOA,
}


def mealpy_gpu_adapter_class(resolved_optimizer_name):
    return _GPU_ADAPTERS.get(str(resolved_optimizer_name))


def supports_mealpy_gpu_adapter(resolved_optimizer_name) -> bool:
    return mealpy_gpu_adapter_class(resolved_optimizer_name) is not None


def mealpy_gpu_optimizer_names() -> tuple[str, ...]:
    return tuple(_GPU_ADAPTERS)
