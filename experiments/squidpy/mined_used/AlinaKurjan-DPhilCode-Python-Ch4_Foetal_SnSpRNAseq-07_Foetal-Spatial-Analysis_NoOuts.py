# mined from: https://github.com/AlinaKurjan/DPhilCode/blob/c52c92003c33f41cd4477943591827650834552a/Python/Ch4_Foetal_SnSpRNAseq/07_Foetal Spatial Analysis_NoOuts.ipynb
# symbols: squidpy.gr.interaction_matrix, squidpy.gr.nhood_enrichment, squidpy.gr.spatial_autocorr, squidpy.gr.spatial_neighbors, squidpy.im.ImageContainer, squidpy.im.calculate_image_features, squidpy.im.process, squidpy.im.segment, squidpy.pl.extract, squidpy.pl.interaction_matrix, squidpy.pl.nhood_enrichment, squidpy.pl.spatial_scatter

# %%
import os
import numpy as np
import pandas as pd
import scanpy as sc
import squidpy as sq
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib as mpl

import cell2location
import scvi

from matplotlib import rcParams
rcParams['pdf.fonttype'] = 42 # enables correct plotting of text for PDFs

# Print date and time:
import datetime
e = datetime.datetime.now()
print ("Current date and time = %s" % e)

# set variables for file paths to read from and write to:

# set a working directory
wdir = "/mnt/da8aa2c4-0136-465b-87a2-d12a59afec55/akurjan/analysis/notebooks"
os.chdir( wdir )

# folder structures
RESULTS_FOLDERNAME = "foetal/results/Spatial/"
FIGURES_FOLDERNAME = "foetal/figures/Spatial/"

if not os.path.exists(RESULTS_FOLDERNAME):
    os.makedirs(RESULTS_FOLDERNAME)
if not os.path.exists(FIGURES_FOLDERNAME):
    os.makedirs(FIGURES_FOLDERNAME)

# Set folder for saving figures into
sc.settings.figdir = FIGURES_FOLDERNAME
    
sp_data_folder = "../files/Spatial/dev/"


def savesvg(fname: str, fig, folder: str=FIGURES_FOLDERNAME) -> None:
    """
    Save figure as vector-based SVG image format.
    """
    fig.tight_layout()
    fig.savefig(os.path.join(folder, fname), format='svg')

# Set other settings
sc.settings.verbosity = 3 # verbosity: errors (0), warnings (1), info (2), hints (3)
sc.logging.print_versions()
sc.set_figure_params(dpi=150, fontsize=10, dpi_save=600)

# %%
def read_and_qc(sample_name, path=sp_data_folder):
    """ 
    This function reads the data for one 10X spatial experiment into the anndata object.
    It also calculates QC metrics. Modify this function if required by your workflow.

    :param sample_name: Name of the sample
    :param path: path to data
    """

    adata = sc.read_visium(path + str(sample_name) + '/outs/',
                           count_file='filtered_feature_bc_matrix.h5', load_images=True)
    adata.obs['sample'] = sample_name
    adata.var['SYMBOL'] = adata.var_names
    adata.var.rename(columns={'gene_ids': 'ENSEMBL'}, inplace=True)
    adata.var['Gene'] = adata.var['SYMBOL'].fillna(adata.var['ENSEMBL'])
    adata.var_names = adata.var['Gene']
    adata.var.drop(columns='Gene', inplace=True)
    adata.var_names_make_unique()
    # adata.var_names = adata.var['ENSEMBL']
    # adata.var.drop(columns='ENSEMBL', inplace=True)

    # Calculate QC metrics
    from scipy.sparse import csr_matrix
    adata.X = adata.X.toarray()
    sc.pp.calculate_qc_metrics(adata, inplace=True)
    adata.X = csr_matrix(adata.X)
    adata.var['mt'] = [gene.startswith('MT-') for gene in adata.var['SYMBOL']]
    adata.var["ribo"] = adata.var['SYMBOL'].str.startswith(("RPS", "RPL"))
    adata.var["mtrnr"] = adata.var['SYMBOL'].str.startswith(("MTRNR"))
    adata.obs['mt_frac'] = adata[:, adata.var['mt'].tolist()].X.sum(1).A.squeeze()/adata.obs['total_counts']

    # add sample name to obs names
    adata.obs["sample"] = [str(i) for i in adata.obs['sample']]
    adata.obs_names = adata.obs["sample"] \
                          + '_' + adata.obs_names
    adata.obs.index.name = 'spot_id'

    return adata

# %%
# Read the list of spatial experiments
sample_name = 'Dev16126_Ach_EnthMB_H', 'Dev16126_Quad_MB_H', 'Dev16126_Quad_MB2_H'

# Read the data into anndata objects
slides = []
for i in sample_name:
    slides.append(read_and_qc(i, path=sp_data_folder))
slides

# %%
slides[0].var

# %%
slides[0].obs['sample'][0]

# %%
for adata in slides:
    adata.var_names_make_unique()
    sc.pp.calculate_qc_metrics(adata, qc_vars=["mt", "ribo"], inplace=True)

# %%
for adata in slides:
    fig, axs = plt.subplots(1, 4, figsize=(15, 4))
    sns.distplot(adata.obs["total_counts"], kde=False, ax=axs[0])
    sns.distplot(adata.obs["total_counts"][adata.obs["total_counts"] < 10000], kde=False, bins=40, ax=axs[1])
    sns.distplot(adata.obs["n_genes_by_counts"], kde=False, bins=60, ax=axs[2])
    sns.distplot(adata.obs["n_genes_by_counts"][adata.obs["n_genes_by_counts"] < 4000], kde=False, bins=60, ax=axs[3])
    print(adata.obs['sample'][0])

# %%
sc.pp.filter_cells(slides[0], min_counts=700)
sc.pp.filter_cells(slides[0], max_counts=15000)
sc.pp.filter_genes(slides[0], min_cells=10)

sc.pp.filter_cells(slides[1], min_counts=1000)
sc.pp.filter_cells(slides[1], max_counts=20000)
sc.pp.filter_genes(slides[1], min_cells=10)

sc.pp.filter_cells(slides[2], min_counts=500)
sc.pp.filter_cells(slides[2], max_counts=10000)
sc.pp.filter_genes(slides[2], min_cells=10)

# %%
for adata in slides:
    sc.pl.violin(adata, 'mt_frac')

# %%
for idx, adata in enumerate(slides):
    print(f"Sample: {adata.obs['sample'][0]}")
    print(f"#genes before MT filter: {adata.n_vars}")
    
    # remove MT genes for spatial mapping (keeping their counts in the object)
    adata.obsm['MT'] = adata[:, adata.var['mt'].values].X.toarray()
    slides[idx] = adata[:, ~adata.var['mt'].values]
    print(f"#genes after MT filter: {slides[idx].n_vars}")

# %%
for idx, adata in enumerate(slides):
    print(f"Sample: {adata.obs['sample'][0]}")
    print(f"#genes before MTrnr filter: {adata.n_vars}")
    adata.obsm['MTRNR'] = adata[:, adata.var['mtrnr'].values].X.toarray()
    slides[idx] = adata[:, ~adata.var['mtrnr'].values]
    print(f"#genes after MTrnr filter: {slides[idx].n_vars}")

# %%
for idx, adata in enumerate(slides):
    print(f"Sample: {adata.obs['sample'][0]}")
    print(f"#genes before RIBO filter: {adata.n_vars}")
    adata.obsm['ribo'] = adata[:, adata.var['ribo'].values].X.toarray()
    slides[idx] = adata[:, ~adata.var['ribo'].values]
    print(f"#genes after RIBO filter: {slides[idx].n_vars}")

# %%
slides[0].var

# %%
print(slides[0].X[1:10,1:10])

# %%
for idx, adata in enumerate(slides):
    slides[idx].layers['counts'] = adata.X.copy()
    sc.pp.normalize_total(adata, inplace=True)
    sc.pp.log1p(adata)
    slides[idx].layers['normcounts'] = adata.X.copy()
    sc.pp.highly_variable_genes(adata, flavor="cell_ranger", n_top_genes=2000)

# %%
print(slides[0].X[1:10,1:10])

# %%
for adata in slides:
    sc.pp.scale(adata)
    sc.pp.pca(adata)
    sc.pp.neighbors(adata)
    sc.tl.umap(adata)
    sc.tl.leiden(adata, key_added="clusters")

# %%
plt.rcParams["figure.figsize"] = (4, 4)
for adata in slides:
    name = adata.obs['sample'][0]
    sc.pl.umap(adata, color=["total_counts", "n_genes_by_counts", "clusters"], wspace=0.4,
               save=f'_{name}_countsAndClusters_umaps.svg')

# %%
plt.rcParams["figure.figsize"] = (8, 8)
for adata in slides:
    name = adata.obs['sample'][0]
    sc.pl.spatial(adata, img_key="hires", color=["total_counts", "n_genes_by_counts"],
                  save=f'_{name}_counts.svg')

# %%
for adata in slides:
    name = adata.obs['sample'][0]
    sq.gr.spatial_neighbors(adata, coord_type='generic', radius=3.0)
    sq.pl.spatial_scatter(adata, shape='circle', color='clusters', img_alpha=0.8,
                  frameon=False, figsize=(7, 3.5),
                  size=1.5, connectivity_key='spatial_connectivities', edges_width=2,
                  save=f'_{name}_connectivities_clusters1_spatialmap.svg'
                 )

# %%
# for adata in slides:
#     adata.var['ENSEMBL'] = adata.var.index
#     adata.var['Gene'] = adata.var['SYMBOL'].fillna(adata.var['ENSEMBL'])
#     adata.var.index = adata.var['Gene']
#     adata.var_names_make_unique()

# %%
for adata in slides:
    name = adata.obs['sample'][0]
    sc.tl.rank_genes_groups(adata, "clusters", method="wilcoxon", layer='normcounts')
    sc.pl.rank_genes_groups_heatmap(adata, n_genes=5, groupby="clusters", 
                                    cmap='seismic', vcenter=0, figsize=(10, 7),
                                    save=f'_{name}_heatmap_wilcoxonClusterDEGs.svg'
                                    # gene_symbols='Gene'
                                   )

# %%
dge_list = [] 
for adata in slides:
    result = adata.uns['rank_genes_groups']
    groups = result['names'].dtype.names
    df = pd.DataFrame(
        {group + '_' + key: result[key][group]
        for group in groups 
        for key in ['names','scores','logfoldchanges', 'pvals', 'pvals_adj']})
    df.to_csv(os.path.join(RESULTS_FOLDERNAME, f'{adata.obs["sample"][0]}_dev_DGE_wilcoxon_spatial.csv'))
    dge_list.append(df)  # Append the DataFrame to the list

# %%
dge_list[2]

# %%
# rename clusters according to their histological regions:

for adata in slides:
    name=adata.obs['sample'][0]
    if name == 'Dev16126_Ach_EnthMB_H':
        adata.obs['region_name'] = adata.obs['clusters'].astype(int)
        region_names = { 
            0: 'Tendon (Throughout)', #
            1: 'Tendon (ENTH)', #
            2: 'Tendon (MB-MTJ)', 
            3: 'Tendon (ENTH-MB)', 
            4: 'Tendon LCT (Outer)', 
            5: 'Skeletal Muscle',
            6: 'Tendon LCT (Inner)'
        }
        adata.obs['region_name'] = adata.obs['region_name'].replace(region_names)
        print('achilles done')
    elif name == 'Dev16126_Quad_MB_H':
        adata.obs['region_name'] = adata.obs['clusters'].astype(int)
        region_names = { 
            0: 'Skeletal Muscle', #
            1: 'Muscle LCT', #
            2: 'Tendon (ENTH-MB)', 
            3: 'Tendon (MB-MTJ)', 
            4: 'Tendon LCT (Outer, MTJ)', 
            5: 'Tendon LCT (Outer, ENTH-MB)',
            6: 'Tendon LCT (Outer, MB-MTJ)'
        }
        adata.obs['region_name'] = adata.obs['region_name'].replace(region_names)
        print('quads1 done')
    elif name == 'Dev16126_Quad_MB2_H':
        adata.obs['region_name'] = adata.obs['clusters'].astype(int)
        region_names = { 
            0: 'Tendon (Throughout)', #
            1: 'Muscle LCT', #
            2: 'Tendon LCT (Inner, Throughout)', 
            3: 'Skeletal Muscle', 
            4: 'Tendon LCT (Outer, MTJ)', 
            5: 'Tendon LCT (Inner, ENTH-MB)',
            6: 'Muscle LCT'
        }
        adata.obs['region_name'] = adata.obs['region_name'].replace(region_names)
        print('quads2 done')
    else:
        print('ohhhhhnooooo')

# %%
for adata in slides:
    #sq.gr.spatial_neighbors(adata, coord_type='generic', radius=3.0)
    sq.pl.spatial_scatter(adata, shape='circle', color='region_name', img_alpha=0.8,
                          size=1.5, connectivity_key='spatial_connectivities', edges_width=2,
                          frameon=False, figsize=(7, 3.5), 
                          palette='Accent',
                          save=f'_{adata.obs["sample"][0]}_annotated_region_clusters_spatialmap.svg'
                         )

# %%
import skimage.exposure

# make hne image a bit brighter
for adata in slides:
    library_id = adata.obs['sample'][0]
    img_png = adata.uns['spatial'][library_id]['images']['hires']
    p2, p98 = np.percentile(img_png, (0.5, 99.5))
    img_rescale = skimage.exposure.rescale_intensity(img_png, in_range=(p2, p98))

    fig, axes = plt.subplots(1,2)
    axes[0].imshow(img_rescale[500:1000,500:1000])
    axes[1].imshow(img_png[500:1000,500:1000])

    adata.uns['spatial'][library_id]['images']['hires'] = img_rescale

# %%
image_dict = {}
for adata in slides:
    library_id = adata.obs['sample'][0]
    img = sq.im.ImageContainer(
        adata.uns['spatial'][library_id]['images']['hires'],
        scale = adata.uns['spatial'][library_id]['scalefactors']['tissue_hires_scalef']
        )
    image_dict[library_id] = img
    
image_dict

# %%
# plot hne
for adata in slides:
    library_id = adata.obs['sample'][0]
    fig, ax = plt.subplots(
        figsize=(3, 5),
    )
    sc.pl.spatial(
        adata,
        color=None,
        img_key='hires',
        ax=ax,
        title='H&E stain',
        #legend_loc=False,
        show=False
    )
    ax.axes.xaxis.label.set_visible(False)
    ax.axes.yaxis.label.set_visible(False)
    # save figure
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_FOLDERNAME, f'{library_id}_spatial_hne.png'), dpi=300, bbox_inches='tight')

# %%
for img in image_dict.values():
    crop = img.crop_corner(900, 700, size=150)
    # smooth image
    sq.im.process(crop, 
                  layer="image", 
                  method="smooth", 
                  sigma=0)

    # plot the result
    fig, axes = plt.subplots(1, 2)
    for layer, ax in zip(["image", "image_smooth"], axes):
        crop.show(layer, ax=ax)
        ax.set_title(layer)
        
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    crop.show("image_smooth", cmap="gray", ax=axes[0])
    axes[1].imshow(crop["image_smooth"][:, :, 0, 0] < 0.4)
    _ = sns.histplot(np.array(crop["image_smooth"]).flatten(), bins=50, ax=axes[2])
    plt.tight_layout()
    
    sq.im.segment(img=crop, layer="image_smooth", method="watershed", thresh=0.4, geq=False)
    print(crop)
    print(f"Number of segments in crop: {len(np.unique(crop['segmented_watershed']))}")

    fig, axes = plt.subplots(1, 2)
    crop.show("image", channel=0, ax=axes[0])
    _ = axes[0].set_title("H&E")
    crop.show("segmented_watershed", cmap="jet", interpolation="none", ax=axes[1])
    _ = axes[1].set_title("segmentation")

# %%
for adata in slides:
    for name, img in image_dict.items():
        if name == adata.obs['sample'][0]:
            # smooth image
            sq.im.process(img, 
                          layer="image", 
                          method="smooth", 
                          sigma=0)

            # plot the result
            fig, axes = plt.subplots(1, 2)
            for layer, ax in zip(["image", "image_smooth"], axes):
                img.show(layer, ax=ax)
                ax.set_title(layer)

            fig, axes = plt.subplots(1, 3, figsize=(15, 4))
            img.show("image_smooth", cmap="gray", ax=axes[0])
            axes[1].imshow(img["image_smooth"][:, :, 0, 0] < 0.6)
            _ = sns.histplot(np.array(img["image_smooth"]).flatten(), bins=50, ax=axes[2])
            plt.tight_layout()

            sq.im.segment(img=img, layer="image_smooth", method="watershed", thresh=0.6, geq=False)
            print(img)
            print(f"Number of segments in img: {len(np.unique(img['segmented_watershed']))}")

            fig, axes = plt.subplots(1, 2)
            img.show("image", channel=0, ax=axes[0])
            _ = axes[0].set_title("H&E")
            img.show("segmented_watershed", cmap="jet", interpolation="none", ax=axes[1])
            _ = axes[1].set_title("segmentation")

            # define image layer to use for segmentation
            features_kwargs = {"segmentation": {"label_layer": "segmented_watershed"}}
            # calculate segmentation features
            sq.im.calculate_image_features(
                adata,
                img,
                features="segmentation",
                layer="image",
                key_added="features_segmentation",
                n_jobs=1,
                features_kwargs=features_kwargs,
            )
            
            # combine features in one dataframe
            adata.obsm["segments"] = pd.concat(
                [adata.obsm[f] for f in adata.obsm.keys() if "features_segmentation" in f],
                axis="columns",
            )
            # make sure that we have no duplicated feature names in the combined table
            adata.obsm["segments"].columns = ad.utils.make_index_unique(
                adata.obsm["segments"].columns
            )

            # plot results and compare with gene-space clustering
            sq.pl.spatial_scatter(
                sq.pl.extract(adata, "features_segmentation"),
                color=[
                    "segmentation_label",
                    "clusters",
                ],
                frameon=False,
                ncols=2,
            )

# %%
for adata in slides:
    for img in image_dict.keys():
        if img == adata.obs['sample'][0]:
            np.set_printoptions(threshold=10)
            print(img)
            print(adata.obsm["spatial"])
            sq.pl.spatial_scatter(adata, outline=True, size=0.3)
        else:
            pass

# %%
import anndata as ad

for adata in slides:
    for imgname, img in image_dict.items():
        if imgname == adata.obs['sample'][0]:
            # calculate features for different scales (higher value means more context)
            for scale in [1.0, 2.0, 3.0]:
                feature_name = f"features_summary_scale{scale}"
                sq.im.calculate_image_features(
                    adata,
                    img,
                    layer='image',
                    features="summary",
                    key_added=feature_name,
                    n_jobs=4,
                    scale=scale,
                    show_progress_bar=True,
                )

            # combine features in one dataframe
            adata.obsm["summary_features"] = pd.concat(
                [adata.obsm[f] for f in adata.obsm.keys() if "features_summary" in f], axis="columns"
            )
            # make sure that we have no duplicated feature names in the combined table
            adata.obsm["summary_features"].columns = ad.utils.make_index_unique(adata.obsm["summary_features"].columns)
        else:
            pass

# %%
# helper function returning a clustering
def cluster_features(features: pd.DataFrame, like=None) -> pd.Series:
    """
    Calculate leiden clustering of features.

    Specify filter of features using `like`.
    """
    # filter features
    if like is not None:
        features = features.filter(like=like)
    # create temporary adata to calculate the clustering
    adata = ad.AnnData(features)
    # important - feature values are not scaled, so need to scale them before PCA
    sc.pp.scale(adata)
    # calculate leiden clustering
    sc.pp.pca(adata, n_comps=min(10, features.shape[1] - 1))
    sc.pp.neighbors(adata)
    sc.tl.leiden(adata)

    return adata.obs["leiden"]

# %%
for adata in slides:
    name = adata.obs['sample'][0]
    # calculate feature clusters
    adata.obs["features_cluster"] = cluster_features(adata.obsm["summary_features"], like="summary")
    # compare feature and gene clusters
    sq.pl.spatial_scatter(adata, color=["features_cluster", "clusters"],
                          save=f'_{name}_featureVSleidenClusters_spatialmap.svg')

# %%
for adata in slides:
    name = adata.obs['sample'][0]
    adata_img = ad.AnnData(adata.obsm['summary_features'])
    sc.pp.neighbors(adata_img)
    joint_adj = adata_img.obsp['connectivities'] + adata.obsp['connectivities']
    sc.tl.leiden(adata, adjacency=joint_adj, key_added='joint_leiden')
    sq.pl.spatial_scatter(adata,color=['joint_leiden', 'clusters'],
                          save=f'_{name}_imageANDfeatureconnectivitiesVSleidenClusters_spatialmap.svg')

# %%
for name, adata in slides.items():
    #sq.gr.spatial_neighbors(adata)
    #joint_adj = adata.obsp['spatial_connectivities'] + adata.obsp['connectivities']
    #sc.tl.leiden(adata, adjacency=joint_adj, key_added='joint_leiden_graph')
    sq.pl.spatial_scatter(adata,color=['clusters', 'features_cluster', 'joint_leiden', 'joint_leiden_graph'],
                          save=f'_{name}_ALLclusters_spatialmap.svg')

# %%
# save maps for each sample separately
clusterings = ['clusters', 'features_cluster', 'joint_leiden', 'joint_leiden_graph']
for name, adata in slides.items():
    for clusternames in clusterings:
        s1 = adata.obs[[clusternames]]
        s1.index = s1.index.str.rsplit('_', n=1).str[-1]
        s1.index.name = 'Barcode'
        s1.to_csv(os.path.join(RESULTS_FOLDERNAME, f'{name}_{clusternames}_clusters.csv'))

# %%
for adata in slides:
    name = adata.obs['sample'][0]
    sq.gr.spatial_neighbors(adata)
    joint_adj = adata.obsp['spatial_connectivities'] + adata.obsp['connectivities']
    sc.tl.leiden(adata, adjacency=joint_adj, key_added='joint_leiden_graph')
    sq.pl.spatial_scatter(adata,color=['clusters', 'features_cluster', 'joint_leiden', 'joint_leiden_graph'],
                          save=f'_{name}_ALLclusters_spatialmap.svg')

# %%
slides[0].var

# %%
for adata in slides:
    sc.pl.spatial(adata, 
                  color=['COL1A1', 'ABI3BP', 'COL6A1', 'COL6A6', 'FMOD', 'TNMD', 'SCX',
                           'POSTN', 'SPARC', 'DCN', 'BGN', 'KERA', 'LUM', 'PI16', 'FNDC1',
                           'COL22A1', 'COL3A1', 'COL4A1', 'COL11A1', 'COL6A3', 'COL6A1',
                           'COL12A1', 'COL2A1', 'F13A1',
                           'NEGR1', 'NAV3', 'SCN7A', 'THBS4', 'FGF14', 
                           'PRG4', 'CREB5', 'joint_leiden', 'joint_leiden_graph', 'clusters'],
                  layer='normcounts',
                  size=1.25,
                  vmin=0,
                  vmax="p99",
                  frameon=False,
                  cmap="plasma",
                  save = f'{adata.obs["sample"][0]}_spatial_markersANDclusters.svg'
                  )

# %%
colormap = plt.get_cmap('PuOr')
inverted_colormap = colormap.reversed()

for adata in slides:
    sq.gr.spatial_neighbors(adata)
    sq.gr.nhood_enrichment(adata, cluster_key='region_name')
    sq.pl.nhood_enrichment(adata, cluster_key='region_name', 
                           cmap=inverted_colormap, vcenter=0, 
                           vmin=-40, vmax=50,
                           figsize=(7, 3.5),
                           save=f'_{adata.obs["sample"][0]}_annotregion_neighenrichment.svg'
                           )

# %%
for adata in slides:
    sq.gr.nhood_enrichment(adata, cluster_key='clusters')
    sq.pl.nhood_enrichment(adata, cluster_key='clusters', 
                           cmap='bwr', vcenter=0, 
                           vmin=-50, vmax=50,
                           save=f'_{adata.obs["sample"][0]}_leiden_neighenrichment.svg'
                           )

# %%
for adata in slides:
    sq.gr.nhood_enrichment(adata, cluster_key='joint_leiden')
    sq.pl.nhood_enrichment(adata, cluster_key='joint_leiden', 
                           cmap='bwr', vcenter=0, 
                           vmin=-30, vmax=40,
                           #save=f'_{adata.obs["sample"][0]}_jointleiden_neighenrichment.svg'
                           )

# %%
for adata in slides:
    genes = adata[:, adata.var.highly_variable].var_names.values[:300]
    #sq.gr.spatial_neighbors(adata)
    sq.gr.spatial_autocorr(
        adata,
        mode="moran",
        genes=genes,
        n_perms=300,
        n_jobs=4,
    )

# %%
for adata in slides:
    sc.pl.spatial(adata, 
                  color=adata.uns['moranI'].head(30).index,
                  layer='normcounts',
                  size=1.25,
                  vmin=0,
                  vmax="p99",
                  frameon=False,
                  cmap="plasma",
                  #save = f'{adata.obs["sample"][0]}_top30_SpatiallyVarGenes.svg'
                 )

# %%
for adata in slides:
    sq.gr.interaction_matrix(adata, cluster_key="region_name")
    sq.pl.interaction_matrix(adata, cluster_key="region_name", 
                             method="average", figsize=(7, 3.5),
                             save=f'_{adata.obs["sample"][0]}_annotregions_interactionmatrix.svg'
                            )

# %%
for adata in slides:
    samplename = adata.obs['sample'][0]
    adata.write(os.path.join(RESULTS_FOLDERNAME, f'filtered_{samplename}.h5ad'))

# %%
for adata in slides:
    adata.var['Gene'] = adata.var.index
    adata.var.index = adata.var['ENSEMBL']
    adata.X = adata.layers['counts'].copy()
    print(adata.X[1:10, 1:10]) 

# %%
import anndata as ad

# Combine anndata objects together
adata = ad.concat(
    slides,
    label="sample",
    uns_merge="unique",
    join='outer',
    keys=sample_name,
    index_unique=None
)
adata

# %%
adata.var['ensembl_gene_id'] = adata.var.index
annot = sc.queries.biomart_annotations(
    "hsapiens",
    ["ensembl_gene_id", "external_gene_name"],
).set_index("ensembl_gene_id")

adata.var[annot.columns] = annot

adata.var.rename(columns={"external_gene_name": "Gene"}, inplace=True)
adata.var['Gene'] = adata.var['Gene'].fillna(adata.var['ensembl_gene_id'])
adata.var = adata.var.drop(columns='ensembl_gene_id')
adata.var

# %%
adata.write(os.path.join(RESULTS_FOLDERNAME, 'concatenated_adata.h5ad'))

# %%
slides = {}
for filename in os.listdir(RESULTS_FOLDERNAME):
    if filename.startswith('filtered_') and filename.endswith(".h5ad"):
        file_path = os.path.join(RESULTS_FOLDERNAME, filename)
        try:
            # Read the h5ad file using anndata
            adata = sc.read_h5ad(file_path)
            
            # Extract the slide name from the filename (assuming filenames are like "filtered_slide_name.h5ad")
            slide_name = filename[len("filtered_") : -len(".h5ad")]
            
            # Store the data in the slides dictionary
            slides[slide_name] = adata
        except Exception as e:
            print(f"Error processing {filename}: {e}")
slides

# %%
for name, adata in slides.items():
    print(name)

# %%
slides['Dev16126_Ach_EnthMB_H'].var

# %%
for name, adata in slides.items():
    
    #sq.gr.spatial_neighbors(adata, coord_type='generic', radius=3.0)
    sq.pl.spatial_scatter(adata, shape='circle', color='region_name', img_alpha=0.8,
                          size=1.5, connectivity_key='spatial_connectivities', edges_width=2,
                          frameon=False, figsize=(7, 3.5), palette='Accent',
                          save=f'_{name}_annotated_region_clusters_spatialmap.svg'
                         )
    #adata.X = adata.layers['counts'].copy()
    #sc.pp.normalize_total(adata, inplace=True)
    #sc.pp.log1p(adata)
    #sc.tl.rank_genes_groups(adata, "region_name", method="wilcoxon", layer='normcounts', key_added='region_name_wilcoxon')
    #sc.pl.rank_genes_groups(adata, n_genes=25, sharey=False, key="region_name_wilcoxon",
    #                       save= f'_{name}_region_name_wilcoxonDEGs_rankings.svg')
    #sc.pp.scale(adata)
    #sc.tl.dendrogram(adata, 'region_name')
    #sc.pl.rank_genes_groups_heatmap(adata, n_genes=10, groupby="region_name", 
    #                                cmap='seismic', vcenter=0,
    #                                figsize=(15, 5), show_gene_labels=True,
    #                                save=f'_{name}_region_name_wilcoxonDEGs_heatmap.svg'
    #                               )

# %%
for name, adata in slides.items():
    adata.X = adata.layers['counts'].copy()
    sc.pp.normalize_total(adata, inplace=True)
    sc.pp.log1p(adata)
    sc.tl.rank_genes_groups(adata, "joint_leiden", method="wilcoxon", layer='normcounts', key_added='joint_leiden_wilcoxon')
    sc.pl.rank_genes_groups(adata, n_genes=30, sharey=False, key="joint_leiden_wilcoxon")
    sc.pp.scale(adata)
    sc.pl.rank_genes_groups_heatmap(adata, n_genes=10, groupby="joint_leiden", 
                                    cmap='seismic', vcenter=0,
                                    save=f'_{name}_joint_leiden_wilcoxonDEGs_heatmap.svg'
                                    # gene_symbols='Gene'
                                   )
