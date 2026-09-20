import scanpy as sc
import squidpy as sq


def top_moran_gene(adata, normalize=False, n_neighs=15):
    if normalize:
        sc.pp.normalize_total(adata)
        sc.pp.log1p(adata)
    sq.gr.spatial_neighbors(adata, coord_type='generic', n_neighs=n_neighs)
    sq.gr.spatial_autocorr(
        adata, mode='moran', genes=adata.var_names.tolist(), n_perms=10, seed=0,
        n_jobs=8, show_progress_bar=False,
    )
    moran_results = adata.uns['moranI'].sort_values('I', ascending=False)
    return moran_results


if __name__ == '__main__':
    # train: slideseqv2 (Slide-seqV2 mouse hippocampus, already log-normalized)
    train_adata = sq.datasets.slideseqv2()
    train_results = top_moran_gene(train_adata, normalize=False)
    print(train_results.head(10))
    print("TRAIN TOP GENE:", train_results.index[0])

    # test: seqfish (mouse embryo, raw counts -> normalize first)
    test_adata = sq.datasets.seqfish()
    test_results = top_moran_gene(test_adata, normalize=True)
    print(test_results.head(10))
    print("TEST TOP GENE:", test_results.index[0])
