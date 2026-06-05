import os
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import scanpy as sc

# Set up clean plotting parameters
sc.settings.verbosity = 3  # show scanpy hints
sc.settings.set_figure_params(dpi=80, facecolor='white')

def main():
    parser = argparse.ArgumentParser(description="Modul 2: Single-Cell Dimensionality Reduction, Clustering & Annotation")
    parser.add_argument("--input", type=str, default="data/tnbc_tme_data_qc.h5ad", help="Path to filtered QC .h5ad dataset file")
    parser.add_argument("--output", type=str, default="data/tnbc_tme_data_annotated.h5ad", help="Path to save annotated .h5ad output")
    parser.add_argument("--plots_dir", type=str, default="plots", help="Directory to save output plots")
    parser.add_argument("--resolution", type=float, default=0.5, help="Clustering resolution for Leiden/Louvain")
    
    args = parser.parse_args()
    
    # Create output directories
    os.makedirs(args.plots_dir, exist_ok=True)
    if os.path.dirname(args.output):
        os.makedirs(os.path.dirname(args.output), exist_ok=True)
        
    print("---------------------------------------------------------")
    print("🧬 Modul 2: Dimensionality Reduction, Clustering & Annotation")
    print("---------------------------------------------------------")
    
    # Check if input exists
    if not os.path.exists(args.input):
        raise FileNotFoundError(f"QC dataset file not found at: {args.input}. Please run Modul 1 first.")
        
    print(f"Loading QC dataset from: {args.input}")
    adata = sc.read_h5ad(args.input)
    print(f"Dataset shape: {adata.n_obs} cells, {adata.n_vars} genes")
    
    # 1. Normalization & Log Transformation
    print("\n[Step 1] Normalizing and log-transforming counts...")
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)
    
    # Store log-normalized profile of all genes in raw slot for marker analysis
    adata.raw = adata
    
    # 2. Highly Variable Genes (HVG) Selection
    print("\n[Step 2] Selecting Highly Variable Genes (HVGs)...")
    sc.pp.highly_variable_genes(adata, min_mean=0.0125, max_mean=3, min_disp=0.5)
    
    # Filter dataset to HVGs for dimensional reductions
    adata_hvg = adata[:, adata.var.highly_variable].copy()
    print(f"Features reduced to {adata_hvg.n_vars} Highly Variable Genes")
    
    # 3. Scaling & Regression
    print("\n[Step 3] Scaling gene expressions...")
    sc.pp.scale(adata_hvg, max_value=10)
    
    # 4. Dimensionality Reduction (PCA)
    print("\n[Step 4] Running Principal Component Analysis (PCA)...")
    sc.tl.pca(adata_hvg, svd_solver='arpack')
    
    # 5. Batch Effect Integration (Harmony)
    batch_key = 'batch'
    if batch_key not in adata_hvg.obs.columns:
        print("No 'batch' column found in cell metadata. Creating a mock batch column (Patient_A/Patient_B) for demonstration...")
        half = len(adata_hvg) // 2
        mock_batches = ['Patient_A'] * half + ['Patient_B'] * (len(adata_hvg) - half)
        adata_hvg.obs[batch_key] = mock_batches
        adata.obs[batch_key] = mock_batches
        
    print("\n[Step 5] Running Harmony Batch Effect Correction...")
    try:
        import harmonypy
        sc.external.pp.harmony_integrate(adata_hvg, key=batch_key, basis='X_pca', adjusted_basis='X_pca_harmony')
        use_rep = 'X_pca_harmony'
        print("Harmony batch correction completed successfully (using X_pca_harmony).")
    except ImportError:
        print("Warning: harmonypy package not found. Skipping batch correction and running PCA coordinates directly.")
        use_rep = 'X_pca'
        
    # 6. Graph Construction
    print("\n[Step 6] Constructing nearest neighbor graph...")
    sc.pp.neighbors(adata_hvg, n_neighbors=10, n_pcs=20, use_rep=use_rep)
    
    # 7. Clustering (Leiden with fallback to Louvain)
    print("\n[Step 7] Clustering cells...")
    try:
        sc.tl.leiden(adata_hvg, resolution=args.resolution, key_added='leiden')
        cluster_key = 'leiden'
        print("Leiden clustering completed successfully.")
    except Exception as e:
        print(f"Leiden clustering failed or package not installed ({e}). Falling back to Louvain...")
        sc.tl.louvain(adata_hvg, resolution=args.resolution, key_added='louvain')
        cluster_key = 'louvain'
        print("Louvain clustering completed successfully.")
        
    # Copy cluster labels to main adata
    adata.obs[cluster_key] = adata_hvg.obs[cluster_key]
    
    # 8. UMAP Projection
    print("\n[Step 8] Computing UMAP coordinates...")
    sc.tl.umap(adata_hvg)
    adata.obsm['X_umap'] = adata_hvg.obsm['X_umap']
    
    # 9. Cell Type Annotation (Marker Scoring)
    print("\n[Step 9] Classifying cells by marker gene signatures...")
    
    # Detect dataset type: check if PBMC markers exist in dataset
    is_pbmc = "MS4A1" in adata.var_names and "GNLY" in adata.var_names
    
    if is_pbmc:
        print("Detected PBMC demo dataset. Loading PBMC marker profiles.")
        marker_genes = {
            'CD4 T-cells': ['IL7R', 'CD4'],
            'CD8 T-cells': ['CD8A', 'CD8B'],
            'B-cells': ['MS4A1', 'CD19'],
            'NK cells': ['GNLY', 'NKG7'],
            'CD14+ Monocytes': ['CD14', 'LYZ'],
            'FCGR3A+ Monocytes': ['FCGR3A', 'MS4A7'],
            'Dendritic cells': ['FCER1A', 'CST3'],
            'Platelets': ['PPBP']
        }
    else:
        print("Detected custom/TNBC dataset. Loading breast tumor microenvironment marker profiles.")
        marker_genes = {
            'T-cells': ['CD3D', 'CD3E', 'CD2'],
            'CD8+ T-cells': ['CD8A', 'CD8B', 'GZMB'],
            'B-cells': ['CD19', 'MS4A1', 'CD79A'],
            'Macrophages': ['CD14', 'CD68', 'CSF1R'],
            'Epithelial/Tumor': ['EPCAM', 'KRT8', 'KRT18', 'MUC1'],
            'Fibroblasts': ['COL1A1', 'DCN', 'FAP'],
            'Endothelial': ['PECAM1', 'VWF']
        }
        
    # Score clusters
    cluster_scores = {}
    for cell_type, markers in marker_genes.items():
        valid_markers = [m for m in markers if m in adata.raw.var_names]
        if len(valid_markers) > 0:
            # Average expression of valid markers per cell
            cell_scores = adata.raw[:, valid_markers].X.mean(axis=1)
            if hasattr(cell_scores, "A1"):  # handle sparse matrix format
                cell_scores = cell_scores.A1
            # Calculate mean expression score per cluster
            df_scores = pd.DataFrame({'cluster': adata.obs[cluster_key], 'score': cell_scores})
            cluster_means = df_scores.groupby('cluster')['score'].mean()
            cluster_scores[cell_type] = cluster_means
            
    df_cluster_scores = pd.DataFrame(cluster_scores)
    print("\nMarker Gene Scores per Cluster:")
    print(df_cluster_scores)
    
    # Map clusters to cell types
    cluster_annotations = df_cluster_scores.idxmax(axis=1).to_dict()
    adata.obs['cell_type'] = adata.obs[cluster_key].map(cluster_annotations)
    
    print("\nAssigned Cluster Cell Types:")
    for cluster, annot in cluster_annotations.items():
         print(f" - Cluster {cluster} -> {annot}")
         
    # 10. Save plots
    print(f"\n[Step 10] Generating visualization plots in '{args.plots_dir}/'...")
    
    # UMAP by cluster
    fig_clusters, ax = plt.subplots(figsize=(8, 7))
    sc.pl.umap(adata_hvg, color=cluster_key, show=False, ax=ax, title="Clustered Cell Populations")
    fig_clusters.savefig(os.path.join(args.plots_dir, "umap_clusters.png"), dpi=150, bbox_inches='tight')
    plt.close(fig_clusters)
    
    # UMAP by cell type
    adata_hvg.obs['cell_type'] = adata.obs['cell_type']
    fig_cell_types, ax = plt.subplots(figsize=(8, 7))
    sc.pl.umap(adata_hvg, color='cell_type', show=False, ax=ax, title="Annotated Single-Cell Phenotypes")
    fig_cell_types.savefig(os.path.join(args.plots_dir, "umap_cell_types.png"), dpi=150, bbox_inches='tight')
    plt.close(fig_cell_types)
    
    # Dotplot of marker expressions
    all_markers = []
    for markers in marker_genes.values():
        all_markers.extend([m for m in markers if m in adata.raw.var_names])
    all_markers = list(dict.fromkeys(all_markers))  # de-duplicate
    
    dp = sc.pl.dotplot(adata, all_markers, groupby='cell_type', show=False, return_fig=True)
    dp.savefig(os.path.join(args.plots_dir, "marker_genes_dotplot.png"), dpi=150)
    
    # 11. Save dataset
    print(f"\n[Step 11] Saving annotated dataset to: {args.output}")
    adata.write(args.output)
    
    print("\n========================= FINAL CELL METRICS =========================")
    cell_counts = adata.obs['cell_type'].value_counts()
    for ctype, count in cell_counts.items():
        print(f" - {ctype}: {count} cells ({count/len(adata):.1%})")
    print("=====================================================================")
    print("Modul 2 clustering and annotation pipeline ran successfully!")

if __name__ == "__main__":
    main()
