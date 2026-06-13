# Adaptive_Mahalanobis-Cholesky_Differential_Evolution_MaCRO_DE
The novel methodology for improving the process of the Differential Evolution algorithm consists of modifying the mutation operator based on Mahalanobis distance and further enhancing the implementation using Cholesky decomposition.

## Run benchmark

```powershell
.\.venv\Scripts\python.exe main.py --parallel yes
```

By default, `main.py` now runs 1000 epochs and uses parallel workers based on the number of runs. Convergence plots are saved under `Figures/EXP###` and Excel summaries under `Results/EXP###`.

For a smaller test run:

```powershell
.\.venv\Scripts\python.exe main.py --functions F12017 --optimizers MaCRO-DE --epochs 2 --runs 2 --parallel yes --output-root smoke_out
```
