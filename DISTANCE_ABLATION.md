# MaCRO-DE distance-metric ablation

Select `distance_ablation` immediately after `ablation` in `EXPERIMENT_MODES`,
or run:

```bash
python main.py --experiment-mode distance_ablation
```

The production defaults remain D=30, 30 independent runs, population 50,
2,000 epochs, and seeds `1234 + run_index` (zero-based). The mode shares the
existing run/budget/backend options; it does not introduce reduced defaults.
It requires CEC2017/D30 and reuses `main.ABLATION_FUNCTIONS` directly:
`F12017, F32017, F82017, F102017, F152017, F172017, F242017, F292017`.
As in the existing ablation mode, `--functions` does not override that list.

`DISTANCE_ABLATION_METRICS` fixes the five curves and their order.
`DISTANCE_ABLATION_MINKOWSKI_P = 3.0` is the only Minkowski configuration;
there is no additional CLI parameter or sensitivity sweep for p.

## Controlled change

`macro_de_distance_variants.py` inherits the current scalar MaCRO-DE and
batched engine. The Mahalanobis-Cholesky control instantiates the original
classes directly. Other variants replace only distance calculation and
classification. All retain:

- The population mean as reference point, in raw coordinates.
- The original q setting and Mahalanobis-Cholesky calculation used to determine
  neighborhood cardinality, including its original `<=` cutoff ties.
- Original close/far donor-pool routing, AWAD routing, minimum-three fallback,
  index ordering, target eligibility, and random sampling calls.
- Original mutation, crossover, adaptation, selection, bounds, objectives,
  seeds, population, and optimization schedule.

### Matched-pool-size protocol

For each generation of each independent run, evaluate the **original squared
Mahalanobis-Cholesky distance** on the current population and count
`k = count(distance_squared <= chi2.ppf(q, D))`. This is the close-set count
before AWAD chooses close versus far and before any minimum-three fallback.
The Mahalanobis-Cholesky control retains its original close mask exactly.

For each alternative metric, compute distance from the same population mean
and mark exactly its **k nearest individuals** as close. The far set is the
complement, of size `N-k`. Squared norms are retained internally because they
have the same ordering as the norms; **no chi-square cutoff is applied to
these alternative distances**. Minkowski uses p=3.

Distance ties are resolved deterministically by original population index.
Both sets are returned in population-index order, preserving donor-index
sampling semantics and consuming no extra RNG draws. The inherited AWAD
rule still chooses close when normalized diversity is at least 0.5 and far
otherwise, and falls back to the whole population when that set has fewer
than three members. This also handles k=0 and k=N without special random draws.

Here matching is conditional on the **same current population**: each variant
recomputes its own Mahalanobis reference count. Independent variants can
develop different populations, reference counts, and AWAD states over time;
they do not share a count schedule from a separately evolving control run.
For any given population and AWAD state, changing the metric preserves pool
cardinality and fallback decisions and changes only pool composition.

CPU scalar, CPU batched, hybrid, and strict GPU routing use the existing
execution architecture. NumPy/CuPy distance kernels consume no random draws.
Requested GPU execution retains the existing CUDA/objective verification and
raises on unavailable CUDA; there is no CPU fallback. Numerical equivalence
of the control is relative to the same existing execution backend.

## Separate artifacts

Outputs go only to `Results/EXPxxx/distance_ablation/` and
`Figures/EXPxxx/distance_ablation/`. Per-run checkpoints live below the new
mode's own `cache/`. Their identities include distance metric/revision,
Minkowski p where applicable, and backend/execution path. Cross-experiment
reuse searches only the same new mode; previous modes' caches are untouched.
The matched-count protocol has revision `distance-matched-mahalanobis-count-v2`,
so earlier direct-threshold distance-ablation checkpoints are incompatible.
Existing cache files are neither deleted nor rewritten by this revision change.

Each function produces:

- A 600-DPI PNG convergence figure with five mean curves and sparse markers
  (no PDF companion export).
- An NPZ archive containing per-run/mean raw fitness and error, final values,
  FE coordinates, labels, seeds, D, optimum and p.
- Long-form per-run and mean convergence CSV files.

The figures retain the existing project's **raw fitness including CEC bias**
convention, using logarithmic scale. They reuse the FULL convergence renderer,
categorical colors, typography, lines, markers, grid, and legend. Titles follow
`Convergence Curve - F12017 (Log Scale)`. Shared `main.FIGURE_EXPORT_DPI = 600`
controls export resolution; the x-axis remains `Cumulative Function Evaluations`.
The FE axis counts the initial population plus each evaluated trial
population. Scalar MEALPY additionally counts its one objective probe;
objective verification/calibration evaluations are outside the scientific
optimization budget. No initial-history point is invented.

`distance_ablation_summary.csv` contains mean, median, population standard
deviation (`ddof=0`, matching project summaries), best and worst **final
error** (`fitness - f_global`), and ascending rank of mean final error within
each function, with average ranks for ties. A JSON file records the settings
and conventions.

## Regenerate without optimization

Use the same experiment ID and scientific/backend settings as the original
run (including `--parallel no` if it used scalar CPU execution):

```bash
python main.py --exp-id 7 --distance-ablation-figures-only
# Equivalent flag: --distance-ablation-from-cache-only
```

This path reads the saved NPZ means and FE coordinates and writes only the
eight distance PNGs. CSV/NPZ data, statistics, checkpoints, and existing PDFs
remain untouched. If the complete archive set is unavailable, it reads compatible
checkpoints without rewriting scientific exports. It never initializes CUDA
or invokes optimization. Missing or incompatible source data cause an explicit
error. An existing
`--reuse-cache-from-exp-id` can supply this mode's own compatible checkpoints
read-only.

An individual NPZ can regenerate its figure independently of checkpoints:

```python
from distance_ablation_reporting import plot_archive
plot_archive(
    "Results/EXP007/distance_ablation/F12017_D30_convergence.npz",
    "Figures/EXP007/distance_ablation",
)
```

## Tiny validation

```bash
MPLBACKEND=Agg python validate_distance_ablation.py
```

This deliberately tiny test uses temporary outputs, two epochs and two runs
on one CEC function, plus direct kernel/control checks. It does not change
production defaults or write project experiment results.
