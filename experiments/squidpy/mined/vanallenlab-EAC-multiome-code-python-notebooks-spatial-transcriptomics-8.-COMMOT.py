# mined from: https://github.com/vanallenlab/EAC-multiome/blob/3563125b57988805fb62f51a979a8f0641d33d27/code/python/notebooks/spatial-transcriptomics/8. COMMOT.ipynb
# symbols: squidpy.pl.spatial_scatter

# %%
import commot as ct
import scanpy as sc
import squidpy as sq
import pandas as pd
import numpy as np

import os

import matplotlib.pyplot as plt
import seaborn as sns

from statsmodels.stats.multitest import multipletests
from scipy.stats import pearsonr

import pathlib as pl

from tqdm import tqdm

# %%
def get_preprocessed_sample(sample_path: pl.Path, min_counts: int, pct_mt: int, min_cells: int) -> sc.AnnData:

    adata = sc.read_visium(path=sample_path)

    adata.var_names_make_unique()
    adata.var["mt"] = adata.var_names.str.startswith("MT-")
    sc.pp.calculate_qc_metrics(adata, qc_vars=["mt"], inplace=True)

    adata.obsm["spatial"] = adata.obsm["spatial"].astype(int)

    sc.pp.filter_cells(adata, min_counts=min_counts)
    adata = adata[adata.obs["pct_counts_mt"] < pct_mt]
    print(f"#cells after MT filter: {adata.n_obs}")
    sc.pp.filter_genes(adata, min_cells=min_cells)
    
    return adata

# %%
spatial_dir = pl.Path("/add/path/here/SpaceRanger_output/")
cell2location_results_dir = pl.Path("/add/path/here/Cell2Location_results/")

# %%
df_cellchat = pd.read_csv("/add/path/here/auxiliary_data/cellchat_database.csv",index_col=0)

# %%
signature_dir = pl.Path("/add/path/here/cNMF_malignant_genes_new_cosine/")

full_sigs = {}
for s in (signature_dir).iterdir():
    sig = s.stem
    full_sigs[sig] = pd.read_csv(s,index_col=0)
    full_sigs[sig] = full_sigs[sig].head(100).index.to_numpy()

caf_dir = pl.Path("/add/path/here/marker_genes/fibroblast/")

caf_sigs = {}
for s in (caf_dir).iterdir():
    sig = s.stem
    caf_sigs[sig] = pd.read_csv(s,index_col=0)
    caf_sigs[sig] = caf_sigs[sig].head(100).names.to_numpy()

myeloid_dir = pl.Path("/add/path/here/marker_genes/myeloid/")

myeloid_sigs = {}
for s in (myeloid_dir).iterdir():
    sig = s.stem
    myeloid_sigs[sig] = pd.read_csv(s,index_col=0)
    myeloid_sigs[sig] = myeloid_sigs[sig].head(100).names.to_numpy()

normal_sigs = caf_sigs.copy()
normal_sigs.update(myeloid_sigs)
del normal_sigs["HGF-CAF"]
del normal_sigs["Kupffer cells"]

# %%
def rank_signaling_activity(
    adata,
    database_name = None
):
    df_ligrec = adata.uns['commot-%s-info' % database_name]['df_ligrec']
    pathways = list(set(list(df_ligrec.iloc[:,2])))
    lr_pair_names = []
    for i in range(df_ligrec.shape[0]):
        lr_pair_names.append('%s-%s' % (df_ligrec.iloc[i,0], df_ligrec.iloc[i,1]))
    act_pathway = np.zeros([len(pathways)], float)
    act_lrpairs = np.zeros([len(lr_pair_names)], float)
    for i in range(len(act_pathway)):
        act_pathway[i] = np.sum(adata.obsm['commot-%s-sum-receiver' % database_name]['r-%s' % pathways[i]].values)
    for i in range(len(act_lrpairs)):
        act_lrpairs[i] = np.sum(adata.obsm['commot-%s-sum-receiver' % database_name]['r-%s' % lr_pair_names[i]].values)
    idx_pathway = np.argsort(-act_pathway)
    idx_lrpairs = np.argsort(-act_lrpairs)
    pathways = np.array(pathways, str)
    lr_pair_names = np.array(lr_pair_names, str)
    
    return pathways[idx_pathway], lr_pair_names[idx_lrpairs]

# %%
def COMMOT_on_patient(patient_name: str,
                      df_cellchat: pd.DataFrame,
                      cell2location_results_dir: pl.Path, 
                      type_interaction: str= "Cell-Cell Contact", dis_thr: int=200) -> None:
    
    if type_interaction not in ["Cell-Cell Contact", "ECM-Receptor", "Secreted Signaling"]:
        print("type_interaction argument is not correct")
        return 
        
    resdir = cell2location_results_dir / patient_name 
    
    sample_path = resdir / patient_name
    
    ad_st = sc.read_h5ad(resdir / "cell2location_map" / "sp.h5ad")
    
    tissue_path = spatial_dir / patient_name / "spatial/tissue_positions_list.csv"
    tissue_position = pd.read_csv(tissue_path,index_col=0)
    tissue_position = tissue_position.loc[ad_st.obs_names]
    
    #Set coordinates
    x_array=tissue_position["array_row"].tolist()
    y_array=tissue_position["array_col"].tolist()
    x_pixel=tissue_position["pxl_row_in_fullres"].tolist()
    y_pixel=tissue_position["pxl_col_in_fullres"].tolist()
    
    x_min, x_max = np.min(x_pixel), np.max(x_pixel)
    y_min, y_max = np.min(y_pixel), np.max(y_pixel)
    
    ad_st.layers["counts"] = ad_st.X.copy()
    
    print("Normalize data")
    sc.pp.normalize_total(ad_st, target_sum=10000)
    sc.pp.log1p(ad_st)

    print("Run spatial communication")
    df_cellchat_filtered = ct.pp.filter_lr_database(df_cellchat, ad_st, min_cell_pct=0.05)
    cc_contact_df = df_cellchat_filtered[df_cellchat_filtered[3]==type_interaction]
    print("N interactions evaluated", cc_contact_df.shape[0])

    ct.tl.spatial_communication(ad_st,
        database_name='cellchat', df_ligrec=cc_contact_df, dis_thr=dis_thr, heteromeric=True, pathway_sum=True)
    
    ad_st.write(f"commot_results/commot_{patient_name}_{type_interaction}_disthr{dis_thr}_adata.h5ad")

# %%
for patient in tqdm(["EGSFR0074_A", "EGSFR1938_A", "EGSFR0148", "EGSFR1938_B", "EGSFR1938_C"]):
    COMMOT_on_patient(patient_name=patient,
                      df_cellchat=df_cellchat,
                      cell2location_results_dir=cell2location_results_dir)

# %%
def get_corrs_state_signal(ranked_pathway, ad_st, state):

    all_receiver_corrs = {}
    all_sender_corrs = {}
    for ptw in ranked_pathway:
        r,p = pearsonr(ad_st.obsm['commot-cellchat-sum-receiver'][f"r-{ptw}"],ad_st.obs[state])
        all_receiver_corrs[ptw] = [r,p]
        r,p = pearsonr(ad_st.obsm['commot-cellchat-sum-sender'][f"s-{ptw}"],ad_st.obs[state])
        all_sender_corrs[ptw] = [r,p]
    
    all_receiver_corrs = pd.DataFrame(all_receiver_corrs,index=[f"r_{state}",f"p_{state}"]).T
    all_sender_corrs = pd.DataFrame(all_sender_corrs,index=[f"r_{state}",f"p_{state}"]).T
    
    all_receiver_corrs[f"q_{state}"] = multipletests(all_receiver_corrs[f"p_{state}"], method="fdr_bh")[1]
    all_sender_corrs[f"q_{state}"] = multipletests(all_sender_corrs[f"p_{state}"], method="fdr_bh")[1]

    return all_receiver_corrs, all_sender_corrs

def get_corrs_patient(ranked_pathway, ad_st, all_states):
    full_receivers, full_senders = {}, {}
    
    for state in all_states:
        all_receiver_corrs, all_sender_corrs = get_corrs_state_signal(ranked_pathway, ad_st, state)
        full_receivers[state] = all_receiver_corrs
        full_senders[state] = all_sender_corrs

    return full_receivers, full_senders

def score_sigs_ad_st(ad_st, full_sigs, caf_sigs, myeloid_sigs):
    ad_st.layers["counts"] = ad_st.X.copy()
    ad_st.X = ad_st.layers["Carcinoma"].copy()
    
    sc.pp.normalize_total(ad_st, target_sum=10000)
    sc.pp.log1p(ad_st)
    
    gex = ad_st.to_df()
    gex = (gex - gex.mean())/gex.std()
    
    for sig, genes in full_sigs.items():
        ad_st.obs[sig] = gex[gex.columns.intersection(genes)].mean(axis=1)
    
    ad_st.X = ad_st.layers["Fibroblast"].copy()
    
    sc.pp.normalize_total(ad_st, target_sum=10000)
    sc.pp.log1p(ad_st)
    
    gex = ad_st.to_df()
    gex = (gex - gex.mean())/gex.std()
    
    for sig, genes in caf_sigs.items():
        ad_st.obs[sig] = gex[gex.columns.intersection(genes)].mean(axis=1)
    
    ad_st.X = ad_st.layers["Myeloid"].copy()
    
    sc.pp.normalize_total(ad_st, target_sum=10000)
    sc.pp.log1p(ad_st)
    
    gex = ad_st.to_df()
    gex = (gex - gex.mean())/gex.std()
    
    for sig, genes in myeloid_sigs.items():
        ad_st.obs[sig] = gex[gex.columns.intersection(genes)].mean(axis=1)

def compute_corr_perpatient(patient_name, full_sigs, caf_sigs, myeloid_sigs):

    path_file = f"commot_results/commot_{patient_name}_adata.h5ad"
    
    ad_st = sc.read_h5ad(path_file)
    
    score_sigs_ad_st(ad_st, full_sigs, caf_sigs, myeloid_sigs)
    
    celltypes = list(full_sigs.keys())+ list(normal_sigs.keys())
    
    ranked_pathway, ranked_lrpair = rank_signaling_activity(ad_st, database_name='cellchat')
    
    full_receivers, full_senders = get_corrs_patient(ranked_pathway, ad_st, celltypes)
    #full_receivers, full_senders = get_corrs_patient(ranked_lrpair, ad_st, celltypes)
    
    full_receivers = pd.concat(full_receivers,axis=1).droplevel(0,axis=1)
    full_receivers.to_csv(f"commot_results/commot_{patient_name}_receiver_corrs.csv")
    full_senders = pd.concat(full_senders,axis=1).droplevel(0,axis=1)
    full_senders.to_csv(f"commot_results/commot_{patient_name}_sender_corrs.csv")

    return full_receivers, full_senders

# %%
receiver_pp, sender_pp = {}, {}
for patient in tqdm(["EGSFR0074_A", "EGSFR1938_A", "EGSFR0148", "EGSFR1938_B", "EGSFR1938_C"]):
    receiver_pp[patient], sender_pp[patient] = compute_corr_perpatient(patient, full_sigs, caf_sigs, myeloid_sigs)

# %%
receiver_pp_secreted, sender_pp_secreted = {}, {}
for patient in tqdm(["EGSFR0074_A", "EGSFR1938_A", "EGSFR0148","EGSFR1938_B", "EGSFR1938_C"]): 
    receiver_pp_secreted[patient], sender_pp_secreted[patient] = compute_corr_perpatient(patient+"_Secreted Signaling_disthr400",
                                                                       full_sigs, caf_sigs, myeloid_sigs)

# %%
receiver_pp_ecm, sender_pp_ecm = {}, {}
for patient in tqdm(["EGSFR0074_A", "EGSFR1938_A", "EGSFR0148","EGSFR1938_B", "EGSFR1938_C"]): 
    receiver_pp_ecm[patient], sender_pp_ecm[patient] = compute_corr_perpatient(patient+"_ECM-Receptor_disthr200",
                                                                       full_sigs, caf_sigs, myeloid_sigs)

# %%
def get_all_sign(receiver_pp, state):
    
    all_sign_rec = []
    for patient in receiver_pp:
        sign_receivers = receiver_pp[patient][(receiver_pp[patient][f"q_{state}"]<0.05) & (receiver_pp[patient][f"r_{state}"]>0.2)]
        all_sign_rec.append(sign_receivers.index)
    return all_sign_rec

def get_consistent_sign(all_sign_rec, n_min=4):
    vc = pd.Series(np.hstack(all_sign_rec)).value_counts()
    return vc[vc>=n_min].index

def get_rec_send(celltypes, receiver_pp, n_min: int=3):
    all_receivers = {}
    sign_receivers = {}
    unique_receivers = []
    for state in celltypes:
        sign_receivers[state] = get_all_sign(receiver_pp, state)
        print(state)
        all_receivers[state]  = get_consistent_sign(sign_receivers[state], n_min=n_min)
        print(all_receivers[state])
        unique_receivers.append(all_receivers[state])
    unique_receivers = np.unique(np.hstack(unique_receivers))
    return all_receivers, unique_receivers

def get_heatmap_rec_send(ordered_celltypes, unique_receivers, all_receivers, receiver_pp):
    heatmap_df_rec = pd.DataFrame(index=ordered_celltypes, columns=unique_receivers)
    for rc in unique_receivers:
        for st in ordered_celltypes:
            if rc in all_receivers[st]:
                avg_r = []
                for pat in receiver_pp:
                    if rc not in receiver_pp[pat].index:
                        print(rc,"is not detected for",pat)
                        avg_r.append(0)
                    else:
                        avg_r.append(receiver_pp[pat].loc[rc,f"r_{st}"])
                avg_r = np.median(avg_r)
                if avg_r>=0.1:
                    heatmap_df_rec.loc[st, rc] = avg_r
    return heatmap_df_rec

# %%
get_all_sign(receiver_pp, "cNMF_4")+get_all_sign(receiver_pp_secreted, "cNMF_4")+get_all_sign(receiver_pp_ecm, "cNMF_4")

celltypes = list(full_sigs.keys())+ list(normal_sigs.keys())

# %%
all_receivers_cc, unique_receivers_cc = get_rec_send(celltypes, receiver_pp)
all_receivers_sec, unique_receivers_sec = get_rec_send(celltypes, receiver_pp_secreted)

# %%
ordered_celltypes = ['cNMF_1', 'cNMF_2', 'cNMF_3', 'cNMF_4', 'cNMF_5','Fibroblast',
 'Inflammatory CAF','Adipose CAF', 'TAM1','TAM2','DC','Mast']

lbls = []
for ctype in ordered_celltypes:
    if len(ctype.split("_"))>1:
        lbls.append(ctype.split("_")[0] + "$_" + ctype.split("_")[1] + "$")
    else:
        lbls.append(ctype)

heatmap_df_rec_cc = get_heatmap_rec_send(ordered_celltypes, unique_receivers_cc, all_receivers_cc, receiver_pp)
heatmap_df_rec_sec = get_heatmap_rec_send(ordered_celltypes, unique_receivers_sec, all_receivers_sec, receiver_pp_secreted)
#heatmap_df_rec_ecm = get_heatmap_rec_send(ordered_celltypes, unique_receivers_ecm, all_receivers_ecm, receiver_pp_ecm)

list_heatmaps = [heatmap_df_rec_cc,heatmap_df_rec_sec]
#list_heatmaps = [heatmap_df_rec_cc,heatmap_df_rec_sec,heatmap_df_rec_ecm]

fig, ax = plt.subplots(2,1,figsize=(12,8),)
flatax = ax.flatten()

for i,ax in enumerate(flatax):
    sns.heatmap(list_heatmaps[i].astype(float), cmap="vlag", vmin=-0.1, center=0, vmax=0.35,
                ax=ax, annot=list_heatmaps[i].astype(float).round(2).replace(np.nan, ""), fmt="")
    ax.set_yticks(ax.get_yticks(), lbls, fontsize=14)
    ax.set_xticks(ax.get_xticks(), ax.get_xticklabels(), rotation=45, ha="right", fontsize=14)
fig.tight_layout()
fig.savefig("figure_COMMOT/heatmap_receivers.svg", dpi=200, bbox_inches="tight")

# %%
all_senders_cc, unique_senders_cc = get_rec_send(celltypes, sender_pp)
all_senders_sec, unique_senders_sec = get_rec_send(celltypes, sender_pp_secreted)

# %%
ordered_celltypes = ['cNMF_1', 'cNMF_2', 'cNMF_3', 'cNMF_4', 'cNMF_5','Fibroblast',
 'Inflammatory CAF','Adipose CAF', 'TAM1','TAM2','DC','Mast']

lbls = []
for ctype in ordered_celltypes:
    if len(ctype.split("_"))>1:
        lbls.append(ctype.split("_")[0] + "$_" + ctype.split("_")[1] + "$")
    else:
        lbls.append(ctype)

heatmap_df_send_cc = get_heatmap_rec_send(ordered_celltypes, unique_senders_cc, all_senders_cc, sender_pp)
heatmap_df_send_sec = get_heatmap_rec_send(ordered_celltypes, unique_senders_sec, all_senders_sec, sender_pp_secreted)

list_heatmaps = [heatmap_df_send_cc,heatmap_df_send_sec]

fig, ax = plt.subplots(2,1,figsize=(8,8),)
flatax = ax.flatten()

for i,ax in enumerate(flatax):
    sns.heatmap(list_heatmaps[i].astype(float), cmap="vlag", vmin=-0.1, center=0, vmax=0.35,
                ax=ax, annot=list_heatmaps[i].astype(float).round(2).replace(np.nan, ""), fmt="")
    ax.set_yticks(ax.get_yticks(), lbls, fontsize=14)
    ax.set_xticks(ax.get_xticks(), ax.get_xticklabels(), rotation=45, ha="right", fontsize=14)
fig.tight_layout()
fig.savefig("figure_COMMOT/heatmap_senders.svg", dpi=200, bbox_inches="tight")

# %%


# %%
patient_name = "EGSFR1938_A"
database_name = "cellchat"

path_file = f"commot_results/commot_{patient_name}_adata.h5ad"
    
ad_st = sc.read_h5ad(path_file)

score_sigs_ad_st(ad_st, full_sigs, caf_sigs, myeloid_sigs)

cell2loc_res = pd.read_csv(cell2location_results_dir / patient_name / "cell2location_map" / "celltype_abundance.csv",index_col=0)
prob_df = (cell2loc_res.T/cell2loc_res.sum(axis=1)).T

celltypes = list(prob_df.columns)

df_ligrec = ad_st.uns['commot-%s-info' % database_name]['df_ligrec']
pathways = list(set(list(df_ligrec.iloc[:,2])))    

# %%
spatial_dir = pl.Path("/add/path/here/SpaceRanger_output/")
tissue_path = spatial_dir / patient_name / "spatial/tissue_positions_list.csv"
tissue_position = pd.read_csv(tissue_path,index_col=0)
tissue_position = tissue_position.loc[ad_st.obs_names]

#Set coordinates
x_array=tissue_position["array_row"].tolist()
y_array=tissue_position["array_col"].tolist()
x_pixel=tissue_position["pxl_row_in_fullres"].tolist()
y_pixel=tissue_position["pxl_col_in_fullres"].tolist()

x_min, x_max = np.min(x_pixel), np.max(x_pixel)
y_min, y_max = np.min(y_pixel), np.max(y_pixel)

# %%
ct.tl.communication_direction(ad_st, database_name='cellchat', pathway_name='THY1', k=5)

ax = ct.pl.plot_cell_communication(ad_st, database_name='cellchat', pathway_name='THY1', plot_method='cell', background_legend=True,
    scale=0.03, ndsize=10, grid_density=0.5, summary="sender", normalize_v = True)
ax.figure.savefig(f"figure_COMMOT/{patient_name}_THY1.png", dpi=300, bbox_inches="tight")

# %%
fig, ax = plt.subplots(1,1,figsize=(5,4))
sq.pl.spatial_scatter(ad_st, color="cNMF_4", 
                      crop_coord=(y_min, x_min, y_max, x_max), 
                      img=False, vmin=-0.4, vmax=0.4, frameon=False, colorbar=False, title="", ax=ax)
fig.savefig(f"figure_COMMOT/{patient_name}_cNMF4.png", dpi=200, bbox_inches="tight")

# %%
fig, ax = plt.subplots(1,1,figsize=(5,4))
sq.pl.spatial_scatter(ad_st, color="cNMF_1", 
                      crop_coord=(y_min, x_min, y_max, x_max), 
                      img=False, vmin=-0.4, vmax=0.4, frameon=False, colorbar=False, title="", ax=ax)
fig.savefig(f"figure_COMMOT/{patient_name}_cNMF1.png", dpi=200, bbox_inches="tight")

# %%
fig, ax = plt.subplots(1,1,figsize=(5,4))
sq.pl.spatial_scatter(ad_st, color="Inflammatory CAF", 
                      crop_coord=(y_min, x_min, y_max, x_max), 
                      img=False, vmin=-0.4, vmax=0.4, frameon=False, colorbar=False, title="", ax=ax)
fig.savefig(f"figure_COMMOT/{patient_name}_iCAF.png", dpi=200, bbox_inches="tight")

# %%
fig, ax = plt.subplots(1,1,figsize=(5,4))
sq.pl.spatial_scatter(ad_st, color="TAM1", 
                      crop_coord=(y_min, x_min, y_max, x_max), 
                      img=False, vmin=-0.4, vmax=0.4, frameon=False, colorbar=False, title="", ax=ax)
fig.savefig(f"figure_COMMOT/{patient_name}_TAM1.png", dpi=200, bbox_inches="tight")

# %%
patient_name = "EGSFR1938_A"
database_name = "cellchat"

path_file = f"commot_results/commot_{patient_name}_Secreted Signaling_disthr400_adata.h5ad"
    
ad_st = sc.read_h5ad(path_file)

score_sigs_ad_st(ad_st, full_sigs, caf_sigs, myeloid_sigs)

cell2loc_res = pd.read_csv(cell2location_results_dir / patient_name / "cell2location_map" / "celltype_abundance.csv",index_col=0)
prob_df = (cell2loc_res.T/cell2loc_res.sum(axis=1)).T

celltypes = list(prob_df.columns)

df_ligrec = ad_st.uns['commot-%s-info' % database_name]['df_ligrec']
pathways = list(set(list(df_ligrec.iloc[:,2])))    

spatial_dir = pl.Path("/add/path/here/SpaceRanger_output/")
tissue_path = spatial_dir / patient_name / "spatial/tissue_positions_list.csv"
tissue_position = pd.read_csv(tissue_path,index_col=0)
tissue_position = tissue_position.loc[ad_st.obs_names]

#Set coordinates
x_array=tissue_position["array_row"].tolist()
y_array=tissue_position["array_col"].tolist()
x_pixel=tissue_position["pxl_row_in_fullres"].tolist()
y_pixel=tissue_position["pxl_col_in_fullres"].tolist()

x_min, x_max = np.min(x_pixel), np.max(x_pixel)
y_min, y_max = np.min(y_pixel), np.max(y_pixel)

# %%
ct.tl.communication_direction(ad_st, database_name='cellchat', pathway_name='MK', k=5)

ax = ct.pl.plot_cell_communication(ad_st, database_name='cellchat', pathway_name='MK', plot_method='cell', background_legend=True,
    scale=0.03, ndsize=10, grid_density=0.5, summary="sender", normalize_v = True)
ax.figure.savefig(f"figure_COMMOT/{patient_name}_MK.png", dpi=300, bbox_inches="tight")

# %%
ct.tl.communication_direction(ad_st, database_name='cellchat', pathway_name='MK', k=5)

ax = ct.pl.plot_cell_communication(ad_st, database_name='cellchat', pathway_name='MK', plot_method='grid', background_legend=True,
    scale=0.0025, ndsize=10, grid_density=1, summary="sender", normalize_v = True)
ax.figure.savefig(f"figure_COMMOT/{patient_name}_MK.png", dpi=300, bbox_inches="tight")

# %%
ct.tl.communication_direction(ad_st, database_name='cellchat', pathway_name='MIF', k=5)

ax = ct.pl.plot_cell_communication(ad_st, database_name='cellchat', pathway_name='MIF', plot_method='cell', background_legend=True,
    scale=0.03, ndsize=10, grid_density=0.5, summary="sender", normalize_v = True)
ax.figure.savefig(f"figure_COMMOT/{patient_name}_MIF.png", dpi=300, bbox_inches="tight")

# %%
ct.tl.communication_direction(ad_st, database_name='cellchat', pathway_name='MIF', k=5)

ax = ct.pl.plot_cell_communication(ad_st, database_name='cellchat', pathway_name='MIF', plot_method='grid', background_legend=True,
    scale=0.003, ndsize=10, grid_density=1, summary="sender", normalize_v = True)
ax.figure.savefig(f"figure_COMMOT/{patient_name}_MIF.png", dpi=300, bbox_inches="tight")

# %%

