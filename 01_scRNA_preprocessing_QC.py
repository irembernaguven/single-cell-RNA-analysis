import os
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import scanpy as sc

# Set up clean plotting parameters
sc.settings.verbosity = 3  # show scanpy hints
sc.logging.print_header()
sc.settings.set_figure_params(dpi=80, facecolor='white')

def main():
    parser = argparse.ArgumentParser(description="Modul 1: Single-Cell RNA-Seq Quality Control & Preprocessing")
    parser.add_argument("--input", type=str, default="data/tnbc_tme_data.h5ad", help="Path to raw .h5ad dataset file")
    parser.add_argument("--output", type=str, default="data/tnbc_tme_data_qc.h5ad", help="Path to save filtered .h5ad output")
    parser.add_argument("--plots_dir", type=str, default="plots", help="Directory to save QC plots")
    parser.add_argument("--min_genes", type=int, default=200, help="Minimum number of genes detected in a cell")
    parser.add_argument("--min_cells", type=int, default=3, help="Minimum number of cells expressing a gene")
    parser.add_argument("--max_mito_pct", type=float, default=15.0, help="Maximum percentage of mitochondrial counts allowed")
    parser.add_argument("--max_genes", type=int, default=4500, help="Maximum number of genes detected in a cell (doublet filter)")
    parser.add_argument("--demo", action="store_true", default=True, help="Download and run QC on a public 3k PBMC demo dataset if input is missing")
    
    args = parser.parse_args()
    
    # Create output directories
    os.makedirs(args.plots_dir, exist_ok=True)
    if os.path.dirname(args.output):
        os.makedirs(os.path.dirname(args.output), exist_ok=True)
    
    print("--------------------------------------------------")
    print("🧬 Modul 1: Single-Cell RNA-Seq QC & Preprocessing")
    print("--------------------------------------------------")
    
    # 1. Loading Dataset
    if not os.path.exists(args.input):
        if args.demo:
            print(f"Dataset '{args.input}' not found. Downloading standard 3k PBMC demo dataset for verification...")
            # Download pbmc3k dataset
            adata = sc.datasets.pbmc3k()
            # Save raw version locally
            adata.write_h5ad(args.input)
            print(f"Demo dataset downloaded and saved to: {args.input}")
        else:
            raise FileNotFoundError(f"Raw dataset file not found at: {args.input}. Pass --demo to download sample data.")
    else:
        print(f"Loading raw dataset from: {args.input}")
        adata = sc.read_h5ad(args.input)
        
    print(f"Raw dataset shape: {adata.n_obs} cells, {adata.n_vars} genes")
    
    # Ensure unique gene names
    adata.var_names_make_unique()
    
    # 2. Quality Control Metrics Calculation
    print("\nCalculating Quality Control metrics...")
    # Identify mitochondrial genes (starting with 'MT-' or 'mt-')
    adata.var['mt'] = adata.var_names.str.startswith('MT-') | adata.var_names.str.startswith('mt-')
    
    # Calculate cell and gene level QC statistics
    sc.pp.calculate_qc_metrics(adata, qc_vars=['mt'], percent_top=None, log1p=False, inplace=True)
    
    # 3. Plotting QC Metrics Before Filtering
    print(f"Saving pre-filtering QC plots to: {args.plots_dir}/")
    
    # Violin plots
    fig_violin_pre, axes = plt.subplots(1, 3, figsize=(15, 5))
    sc.pl.violin(adata, ['n_genes_by_counts'], jitter=0.4, ax=axes[0], show=False)
    sc.pl.violin(adata, ['total_counts'], jitter=0.4, ax=axes[1], show=False)
    sc.pl.violin(adata, ['pct_counts_mt'], jitter=0.4, ax=axes[2], show=False)
    plt.suptitle("QC Metrics Before Filtering", fontsize=14, y=1.02)
    plt.tight_layout()
    fig_violin_pre.savefig(os.path.join(args.plots_dir, "qc_violin_pre.png"), dpi=150)
    plt.close(fig_violin_pre)
    
    # Scatter plots (Correlation checks)
    fig_scatter_pre, axes = plt.subplots(1, 2, figsize=(12, 5))
    sc.pl.scatter(adata, x='total_counts', y='pct_counts_mt', ax=axes[0], show=False)
    sc.pl.scatter(adata, x='total_counts', y='n_genes_by_counts', ax=axes[1], show=False)
    plt.suptitle("QC Feature Correlations Before Filtering", fontsize=14, y=1.02)
    plt.tight_layout()
    fig_scatter_pre.savefig(os.path.join(args.plots_dir, "qc_scatter_pre.png"), dpi=150)
    plt.close(fig_scatter_pre)
    
    # 4. Filtering Cells and Genes
    print("\nApplying QC thresholds:")
    print(f" - Min genes per cell: {args.min_genes}")
    print(f" - Max genes per cell (doublets): {args.max_genes}")
    print(f" - Max mitochondrial percentage: {args.max_mito_pct}%")
    print(f" - Min cells expressing a gene: {args.min_cells}")
    
    # Store initial numbers
    n_cells_raw = adata.n_obs
    n_genes_raw = adata.n_vars
    
    # Apply standard filters
    sc.pp.filter_cells(adata, min_genes=args.min_genes)
    sc.pp.filter_genes(adata, min_cells=args.min_cells)
    
    # Filter by mitochondrial percentage and maximum genes
    adata = adata[adata.obs['pct_counts_mt'] < args.max_mito_pct, :]
    adata = adata[adata.obs['n_genes_by_counts'] < args.max_genes, :]
    
    # 5. Plotting QC Metrics After Filtering
    print(f"Saving post-filtering QC plots to: {args.plots_dir}/")
    fig_violin_post, axes = plt.subplots(1, 3, figsize=(15, 5))
    sc.pl.violin(adata, ['n_genes_by_counts'], jitter=0.4, ax=axes[0], show=False)
    sc.pl.violin(adata, ['total_counts'], jitter=0.4, ax=axes[1], show=False)
    sc.pl.violin(adata, ['pct_counts_mt'], jitter=0.4, ax=axes[2], show=False)
    plt.suptitle("QC Metrics After Filtering", fontsize=14, y=1.02)
    plt.tight_layout()
    fig_violin_post.savefig(os.path.join(args.plots_dir, "qc_violin_post.png"), dpi=150)
    plt.close(fig_violin_post)
    
    # 6. Save Filtered Dataset
    print(f"\nSaving preprocessed dataset to: {args.output}")
    adata.write(args.output)
    
    print("\n========================= STATS =========================")
    print(f"Cells: {n_cells_raw} -> {adata.n_obs} (Filtered out {n_cells_raw - adata.n_obs} cells, {adata.n_obs/n_cells_raw:.1%} retained)")
    print(f"Genes: {n_genes_raw} -> {adata.n_vars} (Filtered out {n_genes_raw - adata.n_vars} genes)")
    print("=========================================================")
    print("QC & Preprocessing completed successfully! Ready for Modul 2.")

if __name__ == "__main__":
    main()
