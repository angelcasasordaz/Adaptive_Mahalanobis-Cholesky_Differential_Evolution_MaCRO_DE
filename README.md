# Adaptive_Mahalanobis-Cholesky_Differential_Evolution_MaCRO_DE
The novel methodology for improving the process of the Differential Evolution algorithm consists of modifying the mutation operator based on Mahalanobis distance and further enhancing the implementation using Cholesky decomposition.

## Run benchmark

```powershell
.\.venv\Scripts\python.exe main.py --parallel yes
```

By default, `main.py` now runs 1000 epochs and uses parallel workers based on the number of runs. Convergence plots are saved under `Figures/EXP###` and Excel summaries under `Results/EXP###`.

To also save a scaled convergence plot, use `--convergence-extra-scale auto`.
Available values are `none`, `auto`, `log`, `symlog`, and `exp`.

For a smaller test run:

```powershell
.\.venv\Scripts\python.exe main.py --functions F12017 --optimizers MaCRO-DE --epochs 2 --runs 2 --parallel yes --output-root smoke_out
```

## CPU and NVIDIA GPU execution

CPU is the default and works on Windows and Linux without CuPy or CUDA:

```text
pip install -r requirements.txt
python main.py --compute-device cpu
```

For Linux with a working NVIDIA driver and supported GPU:

```text
pip install -r requirements-linux-gpu.txt
python main.py --compute-device hybrid
```

Run `nvidia-smi` first to confirm that the system NVIDIA driver recognizes the
GPU. Installing CuPy does not install or repair the system driver.

`hybrid` and `gpu` use the true independent-run batching engine for MaCRO-DE,
DE-M, DE-MC, and DE-MC-CF. Populations remain in CuPy arrays shaped
`(runs_in_batch, population, dimensions)`. Covariance, Cholesky/inverse solves,
Mahalanobis distances, close/far classification, mutation, crossover
application, survivor selection, best tracking, and MaCRO-DE's numerical
adaptive state are batched. Random plans remain per-run NumPy generators so the
seed mapping and single-run draw order are preserved.

The repository-local staged CEC2017 evaluator supports F1 and F3-F19. At GPU
startup each candidate is checked against OPFUNU on fixed edge cases and at
least 512 deterministic random points. Only candidates passing the configured
strict tolerance are eligible for CuPy dispatch; their shift, rotation, and
shuffle data remain cached on the active GPU. AUTO times three scientifically
equivalent choices: verified CuPy, verified vectorized NumPy, and OPFUNU. For a
selected CuPy objective the complete trial population and fitness stay on the
GPU throughout the generation.

F20-F29 still use the optimized CPU boundary. Each trial tensor is copied once
into a reused contiguous pinned host buffer and the fitness matrix is copied
back once. A synthetic calibration selects serial evaluation for cheap
functions or a persistent spawn-based pool for expensive functions. Workers
instantiate their own OPFUNU benchmark and limit BLAS to one thread, avoiding
unsafe CUDA forking and nested oversubscription. The pool is created once for
the GPU controller lifecycle, not once per epoch. Both GPU modes fail early if
CuPy or CUDA is unavailable.

CPU mode retains the MEALPY `solve(problem, seed=seed_base + run)` lifecycle and
run-level `ProcessPoolExecutor` parallelism. With `hybrid` or `gpu`,
`--gpu-workers 1` means one CUDA-owning controller process. Parallelism for the
four supported custom optimizers comes from `--gpu-batch-size`, not CUDA
processes. CPU-only MEALPY optimizers can still use `--n-workers` processes; in
hybrid/GPU mode those pools use `spawn` so they never inherit the controller's
CUDA context.

The GPU controller accepts `--gpu-memory-fraction` (default `0.85`) and
`--gpu-batch-size` (default `auto`). The memory fraction sets an upper limit on
CuPy's default memory pool; it does not reserve memory eagerly or promise that
the pool will consume that amount. The automatic policy first calculates a
conservative memory-safe capacity. It then benchmarks applicable candidates
from `1, 2, 4, 8, 16, pending_runs` on copied synthetic state with unrelated
seeds and chooses the measured best runs/second. Startup reports memory capacity
and effective batch separately. Calibration does not consume an experimental
RNG stream and can be disabled with `--gpu-auto-calibration no`. An integer
request bypasses performance calibration and is capped by memory safety and
pending runs.

If an auto-sized allocation still raises a CuPy OOM, the controller retries the
same run group at half size down to one. An explicit fixed request reports a
clear error instead of silently changing it. Every run has its own population,
fitness, RNG, best/current-best, history, epoch/stopping fields, and MaCRO-DE
adaptive state; checkpoints remain one file per function/optimizer/run.

Batched runs print a heartbeat at epoch 1, every 50 epochs by default, and at
completion. It reports elapsed/estimated remaining time and cumulative GPU,
fitness, transfer, and epoch timing. Convergence plotting audits curve length,
finiteness, and pairwise overlap. Missing algorithms defer the final figure;
coincident curves retain their values and use distinct dash/marker combinations.

To run the small backend validation without a CEC experiment:

```text
python validate_compute_backend.py --compute-device cpu
python validate_compute_backend.py --compute-device hybrid
python validate_gpu_batching.py --compute-device cpu
```

For the default RTX validation settings with explicit memory headroom:

```text
python validate_gpu_batching.py --compute-device hybrid --gpu-memory-fraction 0.85
```

Measure throughput without writing normal experiment outputs:

```text
python benchmark_gpu_batching.py --compute-device hybrid --function F192017 --dimensions 30 --population-size 50 --epochs 10 --runs 30 --objective-workers 8 --cec-objective-backend auto
```

Normal hybrid execution (the long experiment) is:

```text
python main.py --compute-device hybrid --gpu-workers 1 --gpu-memory-fraction 0.85 --gpu-batch-size auto --cec-objective-backend auto --objective-evaluation auto --objective-workers 8 --gpu-auto-calibration yes --gpu-progress-interval 50
```

Compatible completed checkpoints are reused by default; pass
`--no-reuse-cache` to force all requested runs to execute again.
