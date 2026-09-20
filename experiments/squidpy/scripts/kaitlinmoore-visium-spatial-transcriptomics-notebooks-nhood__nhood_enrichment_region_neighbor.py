"""
Confirmation script for task `nhood_enrichment_region_neighbor`.

Grounds the analysis in analysis-kaitlinmoore-visium-spatial-transcriptomics-notebooks-nhood.py
(squidpy.gr.nhood_enrichment / squidpy.pl.nhood_enrichment) on squidpy's bundled
`visium_hne_adata` dataset, which ships with expert tissue-region annotations
in `.obs["cluster"]`.

For a given region, we compute neighborhood enrichment z-scores between all
region pairs and report which OTHER region has the highest z-score with it
(i.e. is most strongly enriched as its spatial neighbor).

Train target region: "Hippocampus"  -> answer region
Test target region:  "Fiber_tract"  -> answer region
"""

import os

os.environ["TMPDIR"] = "/tmp"  # avoid AF_UNIX path-too-long from long default tmpdir


def main():
    import pandas as pd
    import squidpy as sq

    adata = sq.datasets.visium_hne_adata()

    sq.gr.spatial_neighbors(adata)
    sq.gr.nhood_enrichment(adata, cluster_key="cluster", seed=0, n_jobs=1)

    cats = adata.obs["cluster"].cat.categories.tolist()
    Z = adata.uns["cluster_nhood_enrichment"]["zscore"]
    df = pd.DataFrame(Z, index=cats, columns=cats)

    def top_neighbor(region: str) -> str:
        row = df.loc[region].drop(region)
        return row.idxmax()

    train_answer = top_neighbor("Hippocampus")
    test_answer = top_neighbor("Fiber_tract")

    print("train answer (top neighbor of Hippocampus):", train_answer)
    print("test answer (top neighbor of Fiber_tract):", test_answer)


if __name__ == "__main__":
    main()
