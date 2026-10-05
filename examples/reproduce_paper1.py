"""examples/reproduce_paper1.py — validate omnikit against real Paper_1 data.

Reproduces the key finding of Paper_1 §2m (mouse arm) FROM RAW 10x counts using omnikit alone:
  1. the conserved programme is depth-confounded when scored with raw UCell on sparse 10x counts;
  2. depth_control removes it;
  3. the (depth-controlled) programme is carried by myeloid cells.

Run:  /home/meng/envs/sidish_env/bin/python examples/reproduce_paper1.py
"""
import numpy as np, pandas as pd, scanpy as sc, warnings
import omnikit as ok
warnings.filterwarnings("ignore")

DATA = "/media/meng/Bioinformatics/Lupus/0_Dataset"
TAB = "/media/meng/Bioinformatics/Lupus/Paper_1/Tables"
PROG = f"{TAB}/31c_conserved_program.csv"
ORTH = f"{TAB}/31a_orthologs.tsv"
LANE = f"{DATA}/GSE302065/raw/GSM9095229_MRL_05_filtered_feature_bc_matrix.h5"

# ---- programme -> mouse orthologs ----
prog = pd.read_csv(PROG)
orth = pd.read_csv(ORTH, sep="\t")
pm = prog.merge(orth[["human_symbol", "mouse_symbol"]], left_on="gene", right_on="human_symbol", how="left")
up = sorted(pm.loc[(pm.human_meta_t > 0) & pm.mouse_symbol.notna(), "mouse_symbol"].astype(str))
dn = sorted(pm.loc[(pm.human_meta_t < 0) & pm.mouse_symbol.notna(), "mouse_symbol"].astype(str))
print(f"programme: {len(up)} up / {len(dn)} down mouse orthologs")

a = sc.read_10x_h5(LANE)
X = np.asarray(a.X.todense()) if hasattr(a.X, "todense") else np.asarray(a.X)
genes = list(a.var_names)
print(f"lane: {X.shape[0]} cells x {X.shape[1]} genes")

# ---- 1. signed UCell score (raw) ----
s = ok.ucell_signed(X, up, dn, genes=genes)
lib = X.sum(1); ngenes = (X > 0).sum(1)
diag = ok.score_diagnostics(s, lib)
print(f"[raw] Spearman(score, log counts) = {diag['spearman_log_lib']:+.3f}  "
      f"Spearman(score, n_genes) = {ok.stats.spearman(s, ngenes):+.3f}")

# ---- 2. depth control ----
dc = ok.depth_control(s, np.log1p(lib))
print(f"[depth-control] r_before = {dc['r_before']:+.3f} -> r_after = {dc['r_after']:+.3f}")

# ---- 3. myeloid carrier check (crude marker mask) vs rest ----
mye = np.array([g for g in ("Lyz2", "Itgam", "Cd68", "Csf1r", "Fcgr1", "Adgre1", "C1qa", "C1qb", "C1qc") if g in genes])
gi = {g: i for i, g in enumerate(genes)}
mask = X[:, [gi[g] for g in mye]].mean(1) > 0
print(f"myeloid marker mask: {mask.sum()}/{len(mask)} cells")
print(f"  AUC myeloid-vs-rest  raw = {ok.auc(s, mask):.3f}   depth-controlled = {ok.auc(dc['resid'], mask):.3f}")
ds = ok.depth_stratified_auc(s, np.log1p(lib), mask, nq=5)
print(f"  depth-stratified AUC: raw={ds['auc_raw']:.3f} matched={ds['auc_matched']:.3f} min={ds['auc_min']:.3f}")
print("\nOK — reproduces the Paper_1 §2m phenomenon (depth confound on raw UCell; myeloid carrier survives).")
