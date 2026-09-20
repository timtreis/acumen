# mined from: https://github.com/Li-Lab-COH/Spatial_Dietary_Project/blob/b26a661b0d813c3c7ce12c67aec83b720d55c345/python/QC_for_raw_data.ipynb
# symbols: squidpy.datasets.visium_hne_sdata

# %%
import scanpy as sc
from pathlib import Path
import geopandas as gpd
import matplotlib.pyplot as plt
import squidpy as sq

# %%
# point this at the directory that contains:
#   ├── filtered_feature_bc_matrix.h5  (or mtx + barcodes + features)
#   └── spatial/
# C:\Users\jonan\Documents\1Work\RoseLab\Spatial\dietary_droject\data\Rose_Li_VisiumHD\BANOSSM_SSM0015_1_PR_Whole_C1_VISHD_F07833_22WJCYLT3\outs\binned_outputs\square_008um
# "/mnt/c/Users/jonan/Documents/1Work/RoseLab/Spatial/dietary_droject/data/Rose_Li_VisiumHD/BANOSSM_SSM0015_1_PR_Whole_C1_VISHD_F07833_22WJCYLT3/outs/binned_outputs/square_008um/"
data_dir = Path("/mnt/c/Users/jonan/Documents/1Work/"
                "RoseLab/Spatial/dietary_droject/"
                "data/Rose_Li_VisiumHD/"
                "BANOSSM_SSM0015_1_PR_Whole_C1_VISHD_F07833_22WJCYLT3/outs/binned_outputs/square_008um/")

# %%
# If you have a Space Ranger “outs” folder, point to that; otherwise to the folder above.
# e.g. data_dir / "outs"

adata = squidpy.read.visium(
    str(data_dir),
    load_images=["hires", "lowres"]  # pulls in the HD image as adata.uns["spatial"]…
)
adata.var_names_make_unique()


# %%
adata = sq.datasets.visium_folders(
    base_path=data_dir / "outs/binned_outputs/square_008um",
    include_hires_tiff=True
)

# %%
sq.datasets.visium_hne_sdata
