"""Task: nhood_self_enrichment
Which annotated cell type is most strongly enriched for neighboring other
cells of its own type (spatial neighborhood enrichment, self z-score).
train -> seqfish / celltype_mapped_refined
test  -> imc / cell type
"""
import squidpy as sq


def top_self_enrichment(adata, cluster_key, seed=0):
    sq.gr.spatial_neighbors(adata)
    sq.gr.nhood_enrichment(adata, cluster_key=cluster_key, seed=seed)
    z = adata.uns[f"{cluster_key}_nhood_enrichment"]["zscore"]
    cats = adata.obs[cluster_key].cat.categories.tolist()
    diag = [(z[i, i], cats[i]) for i in range(len(cats))]
    diag.sort(reverse=True)
    return diag


if __name__ == "__main__":
    train_adata = sq.datasets.seqfish()
    train_diag = top_self_enrichment(train_adata, "celltype_mapped_refined")
    print("TRAIN (seqfish) top self-enrichment cell types:")
    for score, name in train_diag[:5]:
        print(f"  {name}: {score:.3f}")
    print("TRAIN ANSWER:", train_diag[0][1])

    test_adata = sq.datasets.imc()
    test_diag = top_self_enrichment(test_adata, "cell type")
    print("\nTEST (imc) top self-enrichment cell types:")
    for score, name in test_diag[:5]:
        print(f"  {name}: {score:.3f}")
    print("TEST ANSWER:", test_diag[0][1])
