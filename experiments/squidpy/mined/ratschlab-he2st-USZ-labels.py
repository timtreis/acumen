# mined from: https://github.com/ratschlab/he2st/blob/17087753ce410a9fbd928255679c8c42c088bc58/USZ/labels.ipynb
# symbols: squidpy.pl.spatial_scatter

# %%
import scanpy as sc
import squidpy as sq
import yaml

# %%
with open("config_dataset.yaml", "r") as stream:
    samples = yaml.safe_load(stream)["SAMPLE"]
samples

# %%
for sample in samples:
    adata = sc.read_h5ad(f"out_benchmark/data/h5ad/{sample}.h5ad")
    print(adata.shape)
    sq.pl.spatial_scatter(adata, 
                          color="ground_truth",
                          dpi=300,
                          alpha=0.8,
                          size=10)
    

# %%

