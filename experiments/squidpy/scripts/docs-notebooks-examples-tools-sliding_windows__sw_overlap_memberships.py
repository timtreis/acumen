import squidpy as sq

adata = sq.datasets.mibitof()

# --- train: point16 ---
adata_point16 = adata[adata.obs["library_id"] == "point16"].copy()
sq.tl.sliding_window(adata=adata_point16, window_size=300, overlap=50, copy=False)
cols = adata_point16.obs.columns[adata_point16.obs.columns.str.startswith("sliding_window_assignment_")]
total_point16 = sum(adata_point16.obs[c].sum() for c in cols)
print("train (point16) total memberships:", total_point16)

# --- test: point23 ---
adata_point23 = adata[adata.obs["library_id"] == "point23"].copy()
sq.tl.sliding_window(adata=adata_point23, window_size=300, overlap=50, copy=False)
cols = adata_point23.obs.columns[adata_point23.obs.columns.str.startswith("sliding_window_assignment_")]
total_point23 = sum(adata_point23.obs[c].sum() for c in cols)
print("test (point23) total memberships:", total_point23)
