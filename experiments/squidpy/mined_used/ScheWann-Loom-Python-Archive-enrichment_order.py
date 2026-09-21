# mined from: https://github.com/ScheWann/Loom/blob/5b888fb55a9f14a0796ead97a6cf693002cfcdc8/Python/Archive/enrichment_order.ipynb
# symbols: squidpy.gr.nhood_enrichment, squidpy.gr.spatial_neighbors, squidpy.pl.nhood_enrichment

# %%
import pandas as pd
import numpy as np
import scanpy as sc
import squidpy as sq
import seaborn as sns
import matplotlib.pyplot as plt

# %%
cell_ids = [
    "53579",
    "53609",
    "53617",
    "53742",
    "53825",
    "53827",
    "53895",
    "53910",
    "53931",
    "53962",
    "53982",
    "53986",
    "53997",
    "54029",
    "54068",
    "54107",
    "54112",
    "54114",
    "54125",
    "54145",
    "54174",
    "54186",
    "54229",
    "54234",
    "54301",
    "54332",
    "54421",
    "54472",
    "54493",
    "54494",
    "54495",
    "54496",
    "54510",
    "54552",
    "54561",
    "54598",
    "54621",
    "54645",
    "54650",
    "54694",
    "54770",
    "54798",
    "54847",
    "54850",
    "54874",
    "54877",
    "54909",
    "54923",
    "54927",
    "54932",
    "54948",
    "54950",
    "54965",
    "54986",
    "54994",
    "55004",
    "55012",
    "55024",
    "55032",
    "55082",
    "55088",
    "55117",
    "55140",
    "55169",
    "55297",
    "55327",
    "55331",
    "55335",
    "55379",
    "55407",
    "55408",
    "55419",
    "55436",
    "55484",
    "55498",
    "55508",
    "55531",
    "55553",
    "55589",
    "55618",
    "55654",
    "55664",
    "55665",
    "55671",
    "55690",
    "55699",
    "55709",
    "55731",
    "55787",
    "55790",
    "55797",
    "55823",
    "55826",
    "55856",
    "55869",
    "55873",
    "55884",
    "55933",
    "55936",
    "55943",
    "55959",
    "55961",
    "55964",
    "56000",
    "56009",
    "56037",
    "56064",
    "56071",
    "56082",
    "56123",
    "56129",
    "56164",
    "56166",
    "56234",
    "56239",
    "56242",
    "56246",
    "56259",
    "56265",
    "56271",
    "56348",
    "56349",
    "56362",
    "56372",
    "56404",
    "56447",
    "56459",
    "56471",
    "56487",
    "56491",
    "56499",
    "56505",
    "56513",
    "56534",
    "56545",
    "56575",
    "56577",
    "56578",
    "56631",
    "56737",
    "56751",
    "56752",
    "56836",
    "56857",
    "56883",
    "56885",
    "56893",
    "56911",
    "56914",
    "56922",
    "56926",
    "56964",
    "57003",
    "57007",
    "57018",
    "57021",
    "57046",
    "57052",
    "57058",
    "57071",
    "57095",
    "57118",
    "57145",
    "57155",
    "57162",
    "57175",
    "57180",
    "57215",
    "57229",
    "57239",
    "57247",
    "57251",
    "57356",
    "57394",
    "57403",
    "57409",
    "57459",
    "57467",
    "57516",
    "57538",
    "57563",
    "57658",
    "57663",
    "57690",
    "57734",
    "57749",
    "57768",
    "57778",
    "57781",
    "57801",
    "57828",
    "57843",
    "57849",
    "57855",
    "57869",
    "58022",
    "58062",
    "58084",
    "58092",
    "58102",
    "58172",
    "58190",
    "58192",
    "58212",
    "58221",
    "58288",
    "58368",
    "58404",
    "58407",
    "58440",
    "58500",
    "58528",
    "58607",
    "58617",
    "58640",
    "58670",
    "58690",
    "58703",
    "58763",
    "58812",
    "58898",
    "58904",
    "58924",
    "59016",
    "59062",
    "59153",
    "59223",
    "59282",
    "59329",
    "59357",
    "59458",
    "59469",
    "59497",
    "59547",
    "59623",
    "59723",
    "59807",
    "59864",
    "59956",
    "59969",
    "60003",
    "60056",
    "60113",
    "60117",
    "60146",
    "60222",
    "60237",
    "60313",
    "60379",
    "60449",
    "60578",
    "60598",
    "60658",
    "60672",
    "60697",
    "60738",
    "60791",
    "60862",
    "60910",
    "60951",
    "61026",
    "61045",
    "61154",
    "61167",
    "61169",
    "61189",
    "61224",
    "61281",
    "61285",
    "61321",
    "61330",
    "61345",
    "61397",
    "61412",
    "61435",
    "61524",
    "61561",
    "61800",
    "61915",
    "61933",
    "62005",
    "62033",
    "62128",
    "62156",
    "62167",
    "62485",
    "62518",
    "62565",
    "62574",
    "62593",
    "62614",
    "62775",
    "62829",
    "63072",
    "63102",
    "63131",
    "63177",
    "63295",
    "63360",
    "63431",
    "63462",
    "63574",
    "63599",
    "63636",
    "63659",
    "63748",
    "63904",
    "63950",
    "64046",
    "64083",
    "64440",
    "64441",
    "64501",
    "64547",
    "64642",
    "64683",
    "64720",
    "64763",
    "64817",
    "64857",
    "64859",
    "65008",
    "65021",
    "65116",
    "65171",
    "65186",
    "65307",
    "65450",
    "65551",
    "65601",
    "65617",
    "65645",
    "65656",
    "65800",
    "65840",
    "65848",
    "66012",
    "66015",
    "66096",
    "66420",
    "66460",
    "66496",
    "66514",
    "66528",
    "66834",
    "66868",
    "66889",
    "66997",
    "67076",
    "67283",
    "67340",
    "67364",
    "67444",
    "67449",
    "67486",
    "67835",
    "67856",
    "67968",
    "67973",
    "68054",
    "138114"
]

# %%
cdata = sc.read_h5ad("../Example_Data/H1-TXK6Z4X-A1_2um/skin_TXK6Z4X_A1_2um_b2c_qc.h5ad")
cdata

# %%
df = cdata.obs[['predicted_labels', 'conf_score']].copy()

mean_conf = df.groupby('predicted_labels')['conf_score'].mean()

summary_conf = df.groupby('predicted_labels')['conf_score'].agg(['mean', 'std', 'count', 'min', 'max'])

summary_conf

# %%
def summarize_confidence_distribution(adata, threshold=0.8, figsize=(8,5)):
    df = adata.obs[['predicted_labels', 'conf_score']].copy()
    
    summary = df.groupby('predicted_labels')['conf_score'].agg(
        median='median',
        q25=lambda x: x.quantile(0.25),
        q75=lambda x: x.quantile(0.75),
        high_conf_fraction=lambda x: (x > threshold).mean(),
        n_cells='count'
    ).reset_index()

    summary = summary.rename(columns={
        'q25': 'IQR_25%',
        'q75': 'IQR_75%'
    })

    plt.figure(figsize=figsize)
    sns.violinplot(x='predicted_labels', y='conf_score', data=df, inner="box")
    plt.axhline(threshold, color='red', linestyle='--', label=f"Threshold = {threshold}")
    plt.xticks(rotation=45, ha='right')
    plt.title("Confidence score distribution per cell type")
    plt.legend()
    plt.tight_layout()
    plt.show()

    return summary

summary_table = summarize_confidence_distribution(cdata, threshold=0.8)
print(summary_table)


# %%
cdata = cdata[cell_ids].copy()

sc.pp.highly_variable_genes(cdata, n_top_genes=2000, flavor="seurat_v3")

sc.pp.normalize_total(cdata)
sc.pp.log1p(cdata)
sc.pp.scale(cdata, max_value=10)

sc.tl.pca(cdata, use_highly_variable=True)
sc.pp.neighbors(cdata, n_neighbors=15, n_pcs=30)
sc.tl.umap(cdata)

# custom umap for subset of cells
cdata.obsm['X_umap_subset'] = cdata.obsm['X_umap'].copy()

sc.tl.leiden(cdata, resolution=1, key_added='leiden_subset')
sc.pl.umap(cdata, color=['leiden_subset'], size=50)

# %%
sq.gr.spatial_neighbors(cdata)
sq.gr.nhood_enrichment(cdata, cluster_key="leiden_subset")
sq.pl.nhood_enrichment(cdata, cluster_key="leiden_subset")

# %%
# Enrichment results
nhood_results = cdata.uns['leiden_subset_nhood_enrichment']

available_keys = list(nhood_results.keys())
print("Available result keys:", available_keys)

# Extract Z-score matrix
enrichment_results = nhood_results['zscore']
print("Z-score matrix shape:", enrichment_results.shape)

# %%
# Get cluster names
cluster_names = sorted(cdata.obs['leiden_subset'].unique())
print("Cluster names:", cluster_names)
print("Number of clusters:", len(cluster_names))

# Convert numpy array to DataFrame
enrichment_df = pd.DataFrame(
    enrichment_results, 
    index=cluster_names, 
    columns=cluster_names
)

print("Enrichment matrix as DataFrame:")
print(enrichment_df.head())

# %%
def rank_by_self_enrichment(enrichment_matrix, cluster_names):
    """Sorting based on diagonal Z-score"""
    if isinstance(enrichment_matrix, pd.DataFrame):
        matrix = enrichment_matrix.values
        names = enrichment_matrix.index.tolist()
    else:
        matrix = enrichment_matrix
        names = cluster_names
    
    self_enrichment = np.diag(matrix)
    
    ranking_df = pd.DataFrame({
        'cluster': names,
        'self_enrichment_zscore': self_enrichment
    }).sort_values('self_enrichment_zscore', ascending=False)
    
    return ranking_df

def rank_by_significant_interactions(enrichment_matrix, cluster_names, zscore_threshold=2.0):
    """Sorting based on the number of significant interactions"""
    if isinstance(enrichment_matrix, pd.DataFrame):
        matrix = enrichment_matrix.values
        names = enrichment_matrix.index.tolist()
    else:
        matrix = enrichment_matrix
        names = cluster_names
    
    results = []
    n_clusters = len(names)
    
    for i, cluster in enumerate(names):
        # Get the row for this cluster (interactions with other clusters)
        interactions = matrix[i, :]
        
        # Calculate significant interactions
        significant_positive = (interactions > zscore_threshold).sum()
        significant_negative = (interactions < -zscore_threshold).sum()
        total_significant = significant_positive + significant_negative
        
        results.append({
            'cluster': cluster,
            'significant_interactions': total_significant,
            'positive_interactions': significant_positive,
            'negative_interactions': significant_negative,
            'max_zscore': interactions.max(),
            'min_zscore': interactions.min(),
            'self_enrichment': matrix[i, i]
        })
    
    return pd.DataFrame(results).sort_values('significant_interactions', ascending=False)

def rank_by_spatial_clustering(enrichment_matrix, cluster_names):
    """Sorting based on spatial clustering pattern"""
    # Ensure it's a numpy array
    if isinstance(enrichment_matrix, pd.DataFrame):
        matrix = enrichment_matrix.values
        names = enrichment_matrix.index.tolist()
    else:
        matrix = enrichment_matrix
        names = cluster_names
    
    results = []
    n_clusters = len(names)
    
    for i, cluster in enumerate(names):
        # Self-enrichment
        self_enrich = matrix[i, i]
        
        # Interactions with other clusters (excluding self)
        row_interactions = np.concatenate([matrix[i, :i], matrix[i, i+1:]])
        col_interactions = np.concatenate([matrix[:i, i], matrix[i+1:, i]])
        
        avg_outgoing = row_interactions.mean() if len(row_interactions) > 0 else 0
        avg_incoming = col_interactions.mean() if len(col_interactions) > 0 else 0
        
        results.append({
            'cluster': cluster,
            'self_enrichment': self_enrich,
            'avg_outgoing_interaction': avg_outgoing,
            'avg_incoming_interaction': avg_incoming,
            'spatial_coherence': self_enrich - max(avg_outgoing, avg_incoming)
        })
    
    return pd.DataFrame(results).sort_values('spatial_coherence', ascending=False)

# Execute sorting
cluster_names = sorted(cdata.obs['leiden_subset'].unique())

print("=== Method 1: Self-enrichment ranking ===")
rank1 = rank_by_self_enrichment(enrichment_results, cluster_names)
print(rank1)

print("\n=== Method 2: Significant interactions ranking ===")
rank2 = rank_by_significant_interactions(enrichment_results, cluster_names)
print(rank2)

print("\n=== Method 3: Spatial coherence ranking ===")
rank3 = rank_by_spatial_clustering(enrichment_results, cluster_names)
print(rank3)
