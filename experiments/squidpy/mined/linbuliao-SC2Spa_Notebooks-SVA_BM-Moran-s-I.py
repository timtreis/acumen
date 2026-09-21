# mined from: https://github.com/linbuliao/SC2Spa_Notebooks/blob/02b681f32fa42ea6f8473774d8b406daa15d557e/SVA_BM/Moran's I.ipynb
# symbols: squidpy.gr.spatial_autocorr, squidpy.gr.spatial_neighbors

# %%
import anndata as ad
import scanpy as sc

import pandas as pd
import matplotlib.pyplot as plt

import squidpy as sq

# %matplotlib inline

# %%
def pp(adata):
    adata.var_names_make_unique()
    adata.raw = adata
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)

# %%
adata_HC1 = ad.read_h5ad('../Dataset/AdataMH1.h5ad')
adata_HC2 = ad.read_h5ad('../Dataset/AdataMH2.h5ad')
adata_Em1 = ad.read_h5ad('../Dataset/AdataEmbryo1.h5ad')

ST = pd.read_csv('../MOB_Visium/MOBRep12_Processed.csv')
loc = pd.read_csv('../MOB_Visium/MOBRep12_loc.csv')[['x', 'y']]

adata_MOB = ad.AnnData(X=ST.values, obs = loc, var = pd.DataFrame(ST.columns).set_index(0))
adata_MOB.var.index.name = None

adata_MOB.obsm['spatial'] = adata_MOB.obs[['x', 'y']].values
adata_HC1.obsm['spatial'] = adata_HC1.obs[['xcoord', 'ycoord']].values
adata_HC2.obsm['spatial'] = adata_HC2.obs[['xcoord', 'ycoord']].values
adata_Em1.obsm['spatial'] = adata_Em1.obs[['xcoord', 'ycoord']].values

pp(adata_HC1)
pp(adata_HC2)
pp(adata_Em1)
pp(adata_MOB)

#HC1
Anno = pd.read_csv('../MouseHippocampus_SSV2/ssHippo_RCTD.csv', index_col = 0)
Anno['MCT'] = 't'

index1 = Anno.index[(Anno['celltype_1'] == Anno['celltype_2'])]
Anno['MCT'][index1] = Anno['celltype_1'][index1]
index2 = Anno.index[(Anno['celltype_1'] != Anno['celltype_2'])]
Anno['MCT'][index2] = (Anno['celltype_1'][index2] + '_' + Anno['celltype_2'][index2]).apply(lambda x: '_'.join(sorted(set(x.split('_')))))

adata_HC1.obs = adata_HC1.obs.merge(Anno, left_index = True, right_index = True, how = 'left')
adata_HC1_selected = adata_HC1[adata_HC1.obs['spot_class'].apply(lambda x: x in ['doublet_certain', 'singlet'])]

#Em1
Anno = pd.read_csv('../MouseEmbryo_SSV2/slideSeq_Puck190926_03_RCTD.csv', index_col = 0)
Anno['MCT'] = 't'

index1 = Anno.index[(Anno['celltype_1'] == Anno['celltype_2'])]
Anno['MCT'][index1] = Anno['celltype_1'][index1]
index2 = Anno.index[(Anno['celltype_1'] != Anno['celltype_2'])]
Anno['MCT'][index2] = (Anno['celltype_1'][index2] + '_' + Anno['celltype_2'][index2]).apply(lambda x: '_'.join(sorted(set(x.split('_')))))

adata_Em1.obs = adata_Em1.obs.merge(Anno, left_index = True, right_index = True, how = 'left')
adata_Em1_selected = adata_Em1[adata_Em1.obs['spot_class'].apply(lambda x: x in ['doublet_certain', 'singlet'])]

# %%
sq.gr.spatial_neighbors(adata_HC1)
sq.gr.spatial_autocorr(
    adata_HC1,
    mode="moran",
    n_perms=100,
    n_jobs=16,
)
adata_HC1.uns["moranI"].to_csv('MoransI/MI_HC1.csv')

# %%
sq.gr.spatial_neighbors(adata_HC2)
sq.gr.spatial_autocorr(
    adata_HC2,
    mode="moran",
    n_perms=100,
    n_jobs=16,
)
adata_HC2.uns["moranI"].to_csv('MoransI/MI_HC2.csv')

# %%
sq.gr.spatial_neighbors(adata_Em1)
sq.gr.spatial_autocorr(
    adata_Em1,
    mode="moran",
    n_perms=100,
    n_jobs=16,
)
adata_Em1.uns["moranI"].to_csv('MoransI/MI_Em1.csv')

# %%
sq.gr.spatial_neighbors(adata_MOB)
sq.gr.spatial_autocorr(
    adata_MOB,
    mode="moran",
    n_perms=100,
    n_jobs=16,
)
adata_MOB.uns["moranI"].to_csv('MoransI/MI_MOB.csv')

# %%
CTs = adata_HC1_selected.obs['MCT'].value_counts()
is_doub = CTs.index.str.contains('_')

SCTs = CTs[~is_doub]
SCTs_selected = SCTs[SCTs > 500].index.tolist()
print(SCTs[SCTs > 500])

result_HC1_CTS = []
for CT in SCTs_selected:
    print(CT)
    sq.gr.spatial_neighbors(adata_HC1_selected)
    sq.gr.spatial_autocorr(
        adata_HC1_selected,
        mode="moran",
        n_perms=100,
        n_jobs=16,
    )
    adata_HC1_selected.uns["moranI"].to_csv('MoransI/MI_HC1_' + CT + '.csv')
    result_HC1_CTS.append(adata_HC1_selected.uns["moranI"])

# %%
CTs = adata_Em1_selected.obs['MCT'].value_counts()
is_doub = CTs.index.str.contains('_')

SCTs = CTs[~is_doub]
SCTs_selected = SCTs[SCTs > 500].index.tolist()
print(SCTs[SCTs > 500])

result_Em1_CTS = []
for CT in SCTs_selected:
    print(CT)
    sq.gr.spatial_neighbors(adata_Em1_selected)
    sq.gr.spatial_autocorr(
        adata_Em1_selected,
        mode="moran",
        n_perms=100,
        n_jobs=16,
    )
    adata_Em1_selected.uns["moranI"].to_csv('MoransI/MI_Em1_' + CT + '.csv')
    result_Em1_CTS.append(adata_Em1_selected.uns["moranI"])

# %%


# %%

