# pyomnikit

**Algorithm-first toolkit for kidney-disease multi-omics** — depth-robust gene-program scoring,
cross-species conservation, spatial-niche statistics, and cross-omics integration.

`pyomnikit` is a small, **dependency-light** (NumPy / SciPy / pandas) Python library in which every
estimator is implemented **from its definition** — nothing is a thin wrapper around an opaque
package. It was extracted from a cross-species lupus-nephritis analysis where each of its pieces
solved a concrete, reproducible problem (see *Why* below).

> **Install name vs import name:** `pip install pyomnikit` → `import omnikit`
> (same pattern as `scikit-learn` → `sklearn`).

## Install

```bash
pip install pyomnikit
# from source
pip install git+https://github.com/LingzhangMeng/pyomnikit.git
# or a Chinese mirror
pip install pyomnikit --index-url https://mirrors.aliyun.com/pypi/simple/
```

Dependencies: `numpy`, `scipy`, `pandas`. Plots need `matplotlib` (`pip install pyomnikit[plot]`).

## Modules

| module | purpose | key functions |
|---|---|---|
| `omnikit.stats`   | shared estimators | `auc`, `delong`, `mwu`, `stouffer`, `eb_shrink`, `fisher`, `cohen_d`, `spearman`, `bh`, `by`, `perm_p` |
| `omnikit.program` | **program scoring & conservation** | `ucell`, `ucell_signed`, `score_genes`, `conservation`, `carrier` |
| `omnikit.qc`      | **QC-first audit** | `qc_metrics`, `doublet_scrublet`, `mad_outlier`, `gmm2`, `mixing` |
| `omnikit.spatial` | **spatial niche** | `knn_graph`, `moran`, `geary`, `lisa`, `nhood_enrichment`, `ripley`, `spatial_lr` |
| `omnikit.integrate` | **cross-omics integration** | `joint_nmf`, `cca`, `rv`, `procrustes`, `mofa_lite`, `program_gwas` |
| `omnikit.depth`   | depth control & orthologs | `depth_control`, `depth_stratified_auc`, `casefold_map`, `collapse_orthologs` |

## Quickstart

```python
import numpy as np, omnikit as ok

# depth-robust UCell program score (correct bound: maxU = n_genes * max_rank)
s = ok.ucell(X, {"up": up_genes, "down": down_genes}, genes=gene_names)   # X: cells x genes
score = s["up"] - s["down"]

# diagnose + remove the sequencing-depth confound
print(ok.score_diagnostics(score, library_size))       # Spearman(score, log counts)
resid = ok.depth_control(score, np.log1p(library_size))["resid"]

# which cell types carry the program?
print(ok.carrier(resid, cell_type, donor=donor_id))

# cross-species conservation call (Fisher + sign consistency + thresholds)
call = ok.conservation(human_t, mouse_rho, human_p, mouse_p, genes=gene_names)
```

## Why (design principles)

- **No hidden depth confound.** UCell on sparse 10x counts correlates with sequencing depth (its
  `maxRank` cap places undetected signature genes at the rank ceiling). Every score ships a depth
  diagnostic, and `depth_control` / `depth_stratified_auc` are first-class.
- **Name the null.** Every p-value is permutation, matched-control, or analytic — and says which.
- **Deterministic.** All randomised estimators take `seed=`.
- **Auditable.** Estimators are implemented from their definitions (see `docs/MATH.md`).

## Validation

- `tests/smoke.py` — 26 checks (run: `python tests/smoke.py`).
- Reproduces a published 760-gene cross-species conserved program with **Jaccard = 1.0**.
- On a raw 10x lupus-nephritis kidney lane, `depth_control` removes the depth confound
  (Spearman +0.72 → +0.09) while preserving the myeloid-cell carriers.

## Relation to the R package

`pyomnikit` is the **Python mirror** of the R canonical package `omnikit` (same function set and
mathematics). Function names are `omnikit.<name>` here vs `omni_<name>` in R.

## Citation

See `CITATION.cff`. Please cite the toolkit and the underlying methods (UCell, Scrublet, DeLong,
Moran, NMF, MOFA, scDRS — referenced in `docs/MATH.md`).

## License

MIT — see `LICENSE`.
