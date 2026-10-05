"""omnikit — thin Python mirror of the omnikit R toolkit (Lupus-nephritis kidney programme).

Same mathematics as the R canonical package (see docs/MATH.md); modules map 1:1:
  stats, program (G4), qc (G3), spatial (G2), integrate (G1).
"""
from . import stats, depth, program, qc, spatial, integrate
from .stats import (auc, mwu, delong, stouffer, eb_shrink, fisher, cohen_d,
                    spearman, perm_p, bh, by)
from .depth import (depth_control, depth_stratified_auc, casefold_map, collapse_orthologs)
from .program import (ucell, ucell_signed, score_genes, conservation, carrier,
                      score_diagnostics)
from .qc import (qc_metrics, mad_outlier, gmm2, doublet_scrublet, mixing)
from .spatial import (knn_graph, moran, geary, lisa, nhood_enrichment, ripley, spatial_lr)
from .integrate import (joint_nmf, cca, rv, procrustes, mofa_lite, program_gwas)

__version__ = "0.1.0"
__all__ = [
    "stats", "depth", "program", "qc", "spatial", "integrate",
    "auc", "mwu", "delong", "stouffer", "eb_shrink", "fisher", "cohen_d",
    "spearman", "perm_p", "bh", "by",
    "depth_control", "depth_stratified_auc", "casefold_map", "collapse_orthologs",
    "ucell", "ucell_signed", "score_genes", "conservation", "carrier", "score_diagnostics",
    "qc_metrics", "mad_outlier", "gmm2", "doublet_scrublet", "mixing",
    "knn_graph", "moran", "geary", "lisa", "nhood_enrichment", "ripley", "spatial_lr",
    "joint_nmf", "cca", "rv", "procrustes", "mofa_lite", "program_gwas",
]
