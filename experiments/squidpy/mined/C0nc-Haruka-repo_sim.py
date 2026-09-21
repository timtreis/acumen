# mined from: https://github.com/C0nc/Haruka/blob/f6943b34ef23fc671214ac4e62df9fabb7140e6b/repo_sim.ipynb
# symbols: squidpy.pl.spatial_scatter

# %%
import sys

sys.path.append('Haruka')

from Haruka import Haruka 


import scvi

scvi.utils.seed = 33

# %%
import scanpy as sc 

from sklearn.metrics import adjusted_rand_score as ari

from sklearn.metrics import normalized_mutual_info_score as nmi


def run_Haruka(adata):

    scvi.utils.seed = 33
    model = Haruka(adata, cov_gene_num=64)

    model.construct_cov()

    model.setup_model(gene_likelihood='nb', condition_label='condition', wasserstein_penalty=0.4, me_weight=155/(64*64))

    model.train(max_epochs=200)

    model.extract_rep(n_layer=2)

    model.cluster(background_cluster_number=8, salient_cluster_num=3, seed=33)

    res_haruka = model.output_adata

    return res_haruka 

# %%
from sklearn.preprocessing import *
import scanpy as sc
import numpy as np

from scipy.spatial import distance_matrix

def _compute_CHAOS(clusterlabel,location):

    clusterlabel = np.array(clusterlabel)
    location = np.array(location)
    matched_location = StandardScaler().fit_transform(location)
#     matched_location = location
    clusterlabel_unique = np.unique(clusterlabel)
    dist_val = np.zeros(len(clusterlabel_unique))
    count = 0
    for k in clusterlabel_unique:
        location_cluster = matched_location[clusterlabel==k,:]
        if len(location_cluster)<=2:
            continue
        n_location_cluster = len(location_cluster)
        results = [fx_1NN(i,location_cluster) for i in range(n_location_cluster)]
        dist_val[count] = np.sum(results)
        count = count + 1


    return np.sum(dist_val)/len(clusterlabel)

def fx_1NN(i,location_in):
    location_in = np.array(location_in)
    dist_array = distance_matrix(location_in[i,:][None,:],location_in)[0,:]
    dist_array[i] = np.inf
    return np.min(dist_array)
    
def _compute_PAS(clusterlabel,location):
    
    clusterlabel = np.array(clusterlabel)
    location = np.array(location)
    matched_location = location
    results = [fx_kNN(i,matched_location,k=10,cluster_in=clusterlabel) for i in range(matched_location.shape[0])]
    return np.sum(results)/len(clusterlabel)


def fx_kNN(i,location_in,k,cluster_in):
#     print(i)

# def
    location_in = np.array(location_in)
    cluster_in = np.array(cluster_in)


    dist_array = distance_matrix(location_in[i,:][None,:],location_in)[0,:]
    dist_array[i] = np.inf
    ind = np.argsort(dist_array)[:k]
    cluster_use = np.array(cluster_in)
    if np.sum(cluster_use[ind]!=cluster_in[i])>(k/2):
        return 1
    else:
        return 0

# %%
import squidpy as sq

ari_res_s = []

pas_res_s = []

chaos_res_s = []

nmi_res_s = []

ari_res_b = []

pas_res_b = []

nmi_res_b = []

chaos_res_b = []

for i, path in enumerate(['sim_bench.h5ad', 'sim_bench_s.h5ad', 'sim_bench_b.h5ad']):

    #adata = sc

    adata = sc.read_h5ad('/lab/solexa_sun/lab_members/yancui/' + path)

    label_encoder = {"circle":1, 'square':1, 'other':0, 'triangle':2}

    adata.obs['label'] = adata.obs['shape_region'].map(label_encoder).astype(str)

    adata = run_Haruka(adata)

    
    sq.pl.spatial_scatter(adata, color=['cluster_haruka_salient', 'cluster_haruka_background', 'label', 'Region'], shape=None)

    ari_res_s.append(ari(adata.obs['cluster_haruka_salient'].values, adata.obs['label'].values))

    nmi_res_s.append(nmi(adata.obs['cluster_haruka_salient'].values, adata.obs['label'].values))

    ari_res_b.append(ari(adata.obs['cluster_haruka_background'].values, adata.obs['Region'].values))

    nmi_res_b.append(nmi(adata.obs['cluster_haruka_background'].values, adata.obs['Region'].values))

    pas_res_s.append(_compute_PAS(adata.obs['cluster_haruka_salient'].values, adata.obsm['spatial']))

    pas_res_b.append(_compute_PAS(adata.obs['cluster_haruka_background'].values, adata.obsm['spatial']))

    chaos_res_s.append(_compute_CHAOS(adata.obs['cluster_haruka_salient'].values, adata.obsm['spatial']))
    
    chaos_res_b.append(_compute_CHAOS(adata.obs['cluster_haruka_background'].values, adata.obsm['spatial']))

            
