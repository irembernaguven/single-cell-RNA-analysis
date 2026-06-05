import os
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import scanpy as sc

# Set up clean plotting parameters
sc.settings.verbosity = 3
sc.settings.set_figure_params(dpi=80, facecolor='white')

def main():
    parser = argparse.ArgumentParser(description="Modul 4: Single-Cell Trajectory Inference & Pseudotime Analysis")
    parser.add_argument("--input", type=str, default="data/tnbc_tme_data_annotated.h5ad", help="Path to annotated .h5ad dataset file")
    parser.add_argument("--output", type=str, default="data/tnbc_tme_data_trajectory.h5ad", help="Path to save trajectory .h5ad output")
    parser.add_argument("--plots_dir", type=str, default="plots", help="Directory to save output plots")
    
    args = parser.parse_args()
    
    # Create output directories
    os.makedirs(args.plots_dir, exist_ok=True)
    if os.path.dirname(args.output):
        os.makedirs(os.path.dirname(args.output), exist_ok=True)
        
    print("---------------------------------------------------------")
    print("🧬 Modul 4: Trajectory Inference & Pseudotime Analysis")
    print("---------------------------------------------------------")
    
    # Check if input exists
    if not os.path.exists(args.input):
        raise FileNotFoundError(f"Annotated dataset file not found at: {args.input}. Please run Modul 2 first.")
        
    print(f"Loading annotated dataset from: {args.input}")
    adata = sc.read_h5ad(args.input)
    print(f"Full dataset shape: {adata.n_obs} cells, {adata.n_vars} genes")
    
    # 1. Subset T-cells
    print("\n[Step 1] Filtering T-cell populations...")
    t_mask = adata.obs['cell_type'].astype(str).str.contains('T-cell|CD4 T|CD8 T|T cell', case=False)
    adata_t = adata[t_mask].copy()
    
    if len(adata_t) < 20:
        print("Warning: Insufficient T-cell cells found in the dataset to calculate trajectories.")
        return
        
    print(f"Isolated T-cell subset: {adata_t.n_obs} cells")
    
    # 2. Recompute dimensional reductions specifically on the T-cell subset
    print("\n[Step 2] Recomputing PCA and neighbors on T-cell coordinates...")
    sc.tl.pca(adata_t, svd_solver='arpack')
    sc.pp.neighbors(adata_t, n_neighbors=10, n_pcs=15)
    sc.tl.umap(adata_t)
    
    # Re-cluster the subset to identify sub-states (and avoid single-node graph crashes in SciPy 1.13+)
    print("\nRe-clustering T-cell populations to identify sub-states...")
    try:
        sc.tl.leiden(adata_t, resolution=0.5, key_added='sub_leiden')
        cluster_key = 'sub_leiden'
        print("Sub-clustering completed using Leiden.")
    except Exception as e:
        print(f"Leiden sub-clustering failed ({e}). Running Louvain sub-clustering...")
        sc.tl.louvain(adata_t, resolution=0.5, key_added='sub_louvain')
        cluster_key = 'sub_louvain'
        print("Sub-clustering completed using Louvain.")
        
    # 3. PAGA Connectivity Analysis
    print(f"\n[Step 3] Running Partition-Based Graph Abstraction (PAGA) on T-cell sub-clusters ('{cluster_key}')...")
    sc.tl.paga(adata_t, groups=cluster_key)
    # Define node positions for PAGA to enable 'paga' initialization in UMAP
    sc.pl.paga(adata_t, show=False)
    
    # Reinitialize UMAP coordinates using PAGA layout mapping
    print("Reinitializing UMAP layout using PAGA graph coordinates...")
    sc.tl.umap(adata_t, init_pos='paga')
    
    # 4. Root Cell Selection
    print("\n[Step 4] Identifying naive/healthy root cell for pseudotime...")
    root_gene = 'IL7R'
    if root_gene not in adata_t.raw.var_names:
        # Fallback to the first available gene
        root_gene = adata_t.raw.var_names[0]
        print(f"Warning: Root gene 'IL7R' not found. Using '{root_gene}' as fallback.")
        
    expr_vals = adata_t.raw[:, root_gene].X
    if hasattr(expr_vals, "toarray"):
        expr_vals = expr_vals.toarray()
    expr_vals = expr_vals.flatten()
    
    root_cell_idx = np.argmax(expr_vals)
    adata_t.uns['iroot'] = root_cell_idx
    print(f"Root cell index set to: {root_cell_idx} (highest expression of naive marker '{root_gene}')")
    
    # 5. Diffusion Maps & Pseudotime Calculations
    print("\n[Step 5] Computing Diffusion Map representation...")
    sc.tl.diffmap(adata_t)
    print("Calculating Diffusion Pseudotime (DPT) scores...")
    sc.tl.dpt(adata_t)
    
    # Copy pseudotime back to adata object if user needs it
    adata.obs['dpt_pseudotime'] = np.nan
    adata.obs.update(pd.Series(adata_t.obs['dpt_pseudotime'], name='dpt_pseudotime'))
    
    # 6. Save Trajectory Figures
    print(f"\n[Step 6] Generating trajectory plots in '{args.plots_dir}/'...")
    
    # Plot 1: UMAP colored by pseudotime
    fig_pseudo, ax = plt.subplots(figsize=(8, 7))
    sc.pl.umap(adata_t, color='dpt_pseudotime', show=False, ax=ax, title="T-Cell Evolution: Diffusion Pseudotime")
    fig_pseudo.savefig(os.path.join(args.plots_dir, "umap_pseudotime.png"), dpi=150, bbox_inches='tight')
    plt.close(fig_pseudo)
    
    # Plot 2: PAGA Graph Compare
    print("Saving PAGA Compare Overlay Plot...")
    sc.pl.paga_compare(adata_t, basis='umap', show=False, title="T-Cell PAGA Trajectory Map")
    plt.savefig(os.path.join(args.plots_dir, "paga_trajectory.png"), dpi=150, bbox_inches='tight')
    plt.close()
    
    # Plot 3: Exhaustion Marker Expressions Along Pseudotime
    print("Plotting exhaustion marker trends over pseudotime...")
    genes_to_plot = ['IL7R', 'PDCD1', 'LAG3', 'GZMB']
    valid_plot_genes = [g for g in genes_to_plot if g in adata_t.raw.var_names]
    
    if len(valid_plot_genes) > 0:
        # Extract expression values for these genes
        expr_plot = adata_t.raw[:, valid_plot_genes].X
        if hasattr(expr_plot, "toarray"):
            expr_plot = expr_plot.toarray()
            
        df_plot = pd.DataFrame(expr_plot, columns=valid_plot_genes)
        df_plot['pseudotime'] = adata_t.obs['dpt_pseudotime'].values
        df_plot = df_plot.dropna(subset=['pseudotime']).sort_values('pseudotime')
        
        fig_trends, ax = plt.subplots(figsize=(10, 6))
        colors = ['#3498db', '#e74c3c', '#2ecc71', '#f1c40f']
        
        for idx, gene in enumerate(valid_plot_genes):
            window_size = max(10, len(df_plot) // 10)
            rolling_mean = df_plot[gene].rolling(window=window_size, center=True).mean()
            
            # Draw smoothed trendline
            ax.plot(df_plot['pseudotime'], rolling_mean, label=f"{gene} (Trend)", color=colors[idx % len(colors)], lw=3)
            # Draw raw scatter dots
            ax.scatter(df_plot['pseudotime'], df_plot[gene], alpha=0.1, color=colors[idx % len(colors)], s=5)
            
        ax.set_title("Expression Trends of T-Cell State Markers Along Pseudotime", fontsize=13, fontweight='bold', pad=12)
        ax.set_xlabel("Diffusion Pseudotime (0 = Naive/Healthy ➔ 1 = Exhausted)", fontsize=10, fontweight='bold')
        ax.set_ylabel("Log-Normalized Expression", fontsize=10, fontweight='bold')
        ax.legend(loc="upper right")
        plt.tight_layout()
        fig_trends.savefig(os.path.join(args.plots_dir, "exhaustion_trends.png"), dpi=150)
        plt.close(fig_trends)
    else:
        print("Warning: None of the plotting marker genes (IL7R, PDCD1, LAG3, GZMB) were found in the dataset raw slot.")
        
    # 7. Save trajectory dataset
    print(f"\n[Step 7] Saving trajectory dataset to: {args.output}")
    adata_t.write(args.output)
    
    print("\n========================= TRAJECTORY METRICS =========================")
    print("Pseudotime statistics for T-cell subset:")
    p_stats = adata_t.obs['dpt_pseudotime'].describe()
    print(f" - Min score (naive root): {p_stats['min']:.4f}")
    print(f" - Max score (most exhausted): {p_stats['max']:.4f}")
    print(f" - Average pseudotime score: {p_stats['mean']:.4f}")
    print("======================================================================")
    print("Modul 4 trajectory and pseudotime pipeline completed successfully!")

if __name__ == "__main__":
    main()
