# mined from: https://github.com/AnkitDash-code/Memory_RECOMB-27/blob/911dff58b986f0a86370a9f9551f6e1fc2bbddd3/notebooks/04_colab_scaleup.ipynb
# symbols: squidpy.datasets.slideseqv2, squidpy.gr.spatial_neighbors

# %%
# !pip install -q scanpy squidpy anndata torch scikit-learn
# !pip install -q GraphST pot scikit-misc

# %%
import scanpy as sc
import squidpy as sq

adata = sq.datasets.slideseqv2()
print(adata.shape, adata.X.dtype)
sparsity = 1 - adata.X.nnz / (adata.X.shape[0] * adata.X.shape[1])
print(f"sparsity: {sparsity:.4f}")

# %%
# Slide-seqV2 spots are not on a Visium hex/square grid, so build the spatial
# neighbor graph from raw coordinates (coord_type="generic") rather than
# the Visium-specific coord_type="grid" used for the crop/full datasets.
sc.pp.filter_cells(adata, min_counts=500)
sc.pp.filter_genes(adata, min_cells=10)
sc.pp.normalize_total(adata, target_sum=1e4)
sc.pp.log1p(adata)
sq.gr.spatial_neighbors(adata, coord_type="generic", n_neighs=6)

# %%
import time

start = time.time()
sc.pp.pca(adata, n_comps=50)
sc.pp.neighbors(adata)
sc.tl.leiden(adata, resolution=1.0)
baseline_time_s = time.time() - start
print(f"baseline wall time: {baseline_time_s:.2f}s, n_clusters={adata.obs['leiden'].nunique()}")

# %%
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import silhouette_score


class EmbeddedMemoryLayer(nn.Module):
    def __init__(self, feature_dim, memory_slots=512, memory_dim=128):
        super().__init__()
        self.memory_keys = nn.Parameter(torch.randn(memory_slots, feature_dim) * 0.02)
        self.memory_values = nn.Parameter(torch.randn(memory_slots, memory_dim) * 0.02)
        self.query_proj = nn.Linear(feature_dim, feature_dim)

    def forward(self, x):
        queries = self.query_proj(x)
        attn_scores = torch.matmul(queries, self.memory_keys.T)
        attn_weights = F.softmax(attn_scores, dim=-1)
        return torch.matmul(attn_weights, self.memory_values), attn_weights


class EmbeddedMemoryAutoencoder(nn.Module):
    def __init__(self, feature_dim, memory_slots=512, memory_dim=128):
        super().__init__()
        self.memory = EmbeddedMemoryLayer(feature_dim, memory_slots, memory_dim)
        self.decoder = nn.Linear(memory_dim, feature_dim)

    def forward(self, x):
        embedding, attn_weights = self.memory(x)
        reconstruction = self.decoder(embedding)
        return reconstruction, embedding, attn_weights


def attention_entropy(attn_weights):
    eps = 1e-12
    return -(attn_weights * torch.log(attn_weights + eps)).sum(dim=-1)


def spatial_smoothness_loss(embedding, edge_index, edge_weight):
    row, col = edge_index
    diff = embedding[row] - embedding[col]
    sq_dist = (diff**2).sum(dim=-1) * edge_weight
    return sq_dist.mean()


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

sc.pp.pca(adata, n_comps=50)
x = torch.tensor(adata.obsm["X_pca"].copy(), dtype=torch.float32).to(device)

coo = adata.obsp["spatial_connectivities"].tocoo()
edge_index = torch.tensor([coo.row, coo.col], dtype=torch.long).to(device)
edge_weight = torch.tensor(coo.data, dtype=torch.float32).to(device)

model = EmbeddedMemoryAutoencoder(feature_dim=x.shape[1]).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

lambda_spatial = 10.0
epochs = 300
max_entropy = math.log(model.memory.memory_keys.shape[0])

if device.type == "cuda":
    torch.cuda.reset_peak_memory_stats(device)
start = time.time()
for epoch in range(epochs):
    optimizer.zero_grad()
    reconstruction, embedding, attn_weights = model(x)
    recon_loss = F.mse_loss(reconstruction, x)
    spatial_loss = spatial_smoothness_loss(embedding, edge_index, edge_weight)
    loss = recon_loss + lambda_spatial * spatial_loss
    loss.backward()
    optimizer.step()
    if epoch % 50 == 0 or epoch == epochs - 1:
        with torch.no_grad():
            med_entropy = attention_entropy(attn_weights).median().item()
        print(f"epoch {epoch:4d}  recon={recon_loss.item():.4f}  spatial={spatial_loss.item():.4f}  "
              f"entropy={med_entropy:.4f}/{max_entropy:.4f}")
elapsed = time.time() - start

peak_mb = torch.cuda.max_memory_allocated(device) / 1024**2 if device.type == "cuda" else None
print(f"memory layer training wall time: {elapsed:.4f}s, peak VRAM: {peak_mb} MB")

model.eval()
with torch.no_grad():
    _, embedding, attn_weights = model(x)
adata.obsm["X_memory_trained"] = embedding.cpu().numpy()

final_entropy = attention_entropy(attn_weights).median().item()
if final_entropy < 0.05 * max_entropy:
    print(f"WARNING: final median entropy {final_entropy:.4f} near zero -> slot collapse")

sc.pp.neighbors(adata, use_rep="X_memory_trained", key_added="memory_neighbors")
sc.tl.leiden(adata, neighbors_key="memory_neighbors", key_added="memory_cluster_trained")

mem_labels = adata.obs["memory_cluster_trained"].to_numpy()
mem_sil = silhouette_score(adata.obsm["X_memory_trained"], mem_labels)
print(f"memory layer: n_clusters={len(set(mem_labels))}, silhouette={mem_sil:.4f}")

# %%
import GraphST
from GraphST.GraphST import GraphST as GraphSTModel

raw_adata = sq.datasets.slideseqv2()

start = time.time()
GraphST.preprocess(raw_adata)
GraphST.construct_interaction(raw_adata)
GraphST.add_contrastive_label(raw_adata)
GraphST.get_feature(raw_adata)

graphst_model = GraphSTModel(raw_adata, device=device, epochs=600)
raw_adata = graphst_model.train()

n_clusters_target = adata.obs["leiden"].nunique()
GraphST.clustering(raw_adata, n_clusters=n_clusters_target, method="leiden")
graphst_elapsed = time.time() - start

graphst_labels = raw_adata.obs["domain"].to_numpy()
graphst_sil = silhouette_score(raw_adata.obsm["emb"], graphst_labels)
print(f"GraphST slideseqv2: n_clusters={len(set(graphst_labels))}, "
      f"silhouette={graphst_sil:.4f}, wall_time={graphst_elapsed:.2f}s")

# %%
import torch
print(torch.__version__)  # confirm which torch/cuda build this runtime actually has
# then match it against https://data.pyg.org/whl/ before installing:
# !pip install -q torch_geometric torch_sparse torch_scatter -f https://data.pyg.org/whl/torch-{torch.__version__}.html
# !pip install -q git+https://github.com/QIFEIDKN/STAGATE_pyG.git

# %%
# !pip install -q garfield
