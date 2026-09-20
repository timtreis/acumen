import squidpy as sq

adata = sq.datasets.mibitof()

# --- train: point8 ---
adata_point8 = adata[adata.obs["library_id"] == "point8"].copy()
sq.tl.sliding_window(
    adata=adata_point8, window_size=300, overlap=50, copy=False, drop_partial_windows=True
)
cols = adata_point8.obs.columns[adata_point8.obs.columns.str.startswith("sliding_window_assignment_")]
total_point8 = sum(adata_point8.obs[c].sum() for c in cols)
print("train (point8) total memberships after dropping partial windows:", total_point8)

# --- test: point16 ---
adata_point16 = adata[adata.obs["library_id"] == "point16"].copy()
sq.tl.sliding_window(
    adata=adata_point16, window_size=300, overlap=50, copy=False, drop_partial_windows=True
)
cols = adata_point16.obs.columns[adata_point16.obs.columns.str.startswith("sliding_window_assignment_")]
total_point16 = sum(adata_point16.obs[c].sum() for c in cols)
print("test (point16) total memberships after dropping partial windows:", total_point16)
