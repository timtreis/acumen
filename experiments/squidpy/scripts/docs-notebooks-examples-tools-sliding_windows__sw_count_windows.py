import squidpy as sq

adata = sq.datasets.mibitof()

# --- train: point8 ---
adata_point8 = adata[adata.obs["library_id"] == "point8"].copy()
sq.tl.sliding_window(adata=adata_point8, window_size=200, overlap=0, copy=False)
n_windows_point8 = adata_point8.obs["sliding_window_assignment"].nunique()
print("train (point8) n_windows:", n_windows_point8)

# --- test: point23 ---
adata_point23 = adata[adata.obs["library_id"] == "point23"].copy()
sq.tl.sliding_window(adata=adata_point23, window_size=200, overlap=0, copy=False)
n_windows_point23 = adata_point23.obs["sliding_window_assignment"].nunique()
print("test (point23) n_windows:", n_windows_point23)
