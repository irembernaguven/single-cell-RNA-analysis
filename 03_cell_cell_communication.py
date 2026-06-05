import os
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import scanpy as sc
import networkx as nx

# Set up clean plotting parameters
sc.settings.verbosity = 3
sc.settings.set_figure_params(dpi=80, facecolor='white')

# Curated database of key immune checkpoint, chemotaxis, and signaling L-R pairs
LIGAND_RECEPTOR_DATABASE = [
    # Immune Checkpoints (Immune Evasion Mechanisms)
    {"ligand": "CD274", "receptor": "PDCD1", "pathway": "PD-L1/PD-1 (Immune Evasion)"},
    {"ligand": "CD80", "receptor": "CTLA4", "pathway": "CD80/CTLA-4 (Immune Checkpoint)"},
    {"ligand": "CD86", "receptor": "CTLA4", "pathway": "CD86/CTLA-4 (Immune Checkpoint)"},
    {"ligand": "LGALS9", "receptor": "HAVCR2", "pathway": "Galectin-9/TIM-3 (Immune Checkpoint)"},
    {"ligand": "PVR", "receptor": "TIGIT", "pathway": "PVR/TIGIT (Immune Checkpoint)"},
    
    # T-cell Homing & Recruitment Chemokines
    {"ligand": "CCL5", "receptor": "CCR5", "pathway": "CCL5/CCR5 (T-cell Recruitment)"},
    {"ligand": "CXCL9", "receptor": "CXCR3", "pathway": "CXCL9/CXCR3 (T-cell Recruitment)"},
    {"ligand": "CXCL10", "receptor": "CXCR3", "pathway": "CXCL10/CXCR3 (T-cell Recruitment)"},
    {"ligand": "CXCL12", "receptor": "CXCR4", "pathway": "CXCL12/CXCR4 (Tumor Growth/Migration)"},
    
    # Cytokines & Immunosuppression
    {"ligand": "TGFB1", "receptor": "TGFBR1", "pathway": "TGFb/TGFbR (Immunosuppression)"},
    {"ligand": "TGFB1", "receptor": "TGFBR2", "pathway": "TGFb/TGFbR (Immunosuppression)"},
    {"ligand": "IL2", "receptor": "IL2RA", "pathway": "IL2/IL2R (T-cell Proliferation)"},
    {"ligand": "IL6", "receptor": "IL6R", "pathway": "IL-6/IL-6R (Pro-inflammatory)"},
    {"ligand": "TNF", "receptor": "TNFRSF1A", "pathway": "TNF/TNFR (Pro-inflammatory)"},
    
    # Growth Factors
    {"ligand": "EGF", "receptor": "EGFR", "pathway": "EGF/EGFR (Tumor growth)"}
]

def main():
    parser = argparse.ArgumentParser(description="Modul 3: Single-Cell Cell-Cell Communication Analysis")
    parser.add_argument("--input", type=str, default="data/tnbc_tme_data_annotated.h5ad", help="Path to annotated .h5ad dataset file")
    parser.add_argument("--plots_dir", type=str, default="plots", help="Directory to save communication plots")
    parser.add_argument("--results_dir", type=str, default="results", help="Directory to save CSV outputs")
    parser.add_argument("--min_pct", type=float, default=0.05, help="Minimum expression frequency (ratio 0-1) in a cell type")
    
    args = parser.parse_args()
    
    # Create directories
    os.makedirs(args.plots_dir, exist_ok=True)
    os.makedirs(args.results_dir, exist_ok=True)
    
    print("---------------------------------------------------------")
    print("🧬 Modul 3: Cell-Cell Communication Analysis")
    print("---------------------------------------------------------")
    
    # Check if input exists
    if not os.path.exists(args.input):
        raise FileNotFoundError(f"Annotated dataset file not found at: {args.input}. Please run Modul 2 first.")
        
    print(f"Loading annotated dataset from: {args.input}")
    adata = sc.read_h5ad(args.input)
    
    # Ensure cell annotations exist
    if 'cell_type' not in adata.obs.columns:
        raise KeyError("Metadata column 'cell_type' not found in dataset obs. Please annotate cells first.")
        
    cell_types = adata.obs['cell_type'].dropna().unique().tolist()
    print(f"Annotated cell types found: {cell_types}")
    
    # 1. Collect all unique genes in the L-R database
    lr_genes = list(set([p['ligand'] for p in LIGAND_RECEPTOR_DATABASE] + [p['receptor'] for p in LIGAND_RECEPTOR_DATABASE]))
    valid_genes = [g for g in lr_genes if g in adata.raw.var_names]
    print(f"Matched {len(valid_genes)} out of {len(lr_genes)} L-R database genes in dataset expressions.")
    
    if len(valid_genes) == 0:
        print("Warning: No genes from L-R database found in the dataset raw slot. Skipping analysis.")
        return
        
    # 2. Extract expression matrix for valid L-R genes
    expr_matrix = adata.raw[:, valid_genes].X
    if hasattr(expr_matrix, "toarray"):  # handle sparse matrices
        expr_matrix = expr_matrix.toarray()
        
    df_expr = pd.DataFrame(expr_matrix, index=adata.obs_names, columns=valid_genes)
    df_expr['cell_type'] = adata.obs['cell_type'].values
    
    # 3. Calculate Mean Expression and Frequency per cell type
    print("\nCalculating gene expression profiles across cell populations...")
    df_means = df_expr.groupby('cell_type')[valid_genes].mean()
    df_pcts = df_expr.groupby('cell_type')[valid_genes].apply(lambda x: (x > 0).mean())
    
    # 4. Score Ligand-Receptor Interactions
    print("\nScoring active signaling interactions...")
    interactions = []
    
    for pair in LIGAND_RECEPTOR_DATABASE:
        ligand = pair['ligand']
        receptor = pair['receptor']
        pathway = pair['pathway']
        
        if ligand in valid_genes and receptor in valid_genes:
            for sender in cell_types:
                for receiver in cell_types:
                    p_lig = df_pcts.loc[sender, ligand]
                    p_rec = df_pcts.loc[receiver, receptor]
                    
                    # Filter out low-frequency noise (minimum expression threshold)
                    if p_lig >= args.min_pct and p_rec >= args.min_pct:
                        m_lig = df_means.loc[sender, ligand]
                        m_rec = df_means.loc[receiver, receptor]
                        score = m_lig * m_rec
                        
                        interactions.append({
                            "sender": sender,
                            "receiver": receiver,
                            "ligand": ligand,
                            "receptor": receptor,
                            "pathway": pathway,
                            "score": score,
                            "p_ligand": p_lig,
                            "p_receptor": p_rec
                        })
                        
    if len(interactions) == 0:
        print(f"Warning: No ligand-receptor interactions passed the {args.min_pct:.0%} expression threshold.")
        return
        
    df_interactions = pd.DataFrame(interactions)
    # Save CSV results
    csv_path = os.path.join(args.results_dir, "cell_cell_interactions.csv")
    df_interactions.to_csv(csv_path, index=False)
    print(f"Saved {len(df_interactions)} active interactions to: {csv_path}")
    
    # 5. Plotting Cell-Cell Communication Heatmap
    print(f"\nGenerating plots in '{args.plots_dir}/'...")
    df_counts = df_interactions.pivot_table(index='sender', columns='receiver', values='score', aggfunc='size', fill_value=0)
    
    # Reindex to make sure all cell types are represented on both axes
    df_counts = df_counts.reindex(index=cell_types, columns=cell_types, fill_value=0)
    
    fig_heat, ax = plt.subplots(figsize=(8, 7))
    sns.heatmap(df_counts, annot=True, fmt="d", cmap="YlOrRd", square=True, cbar=True, ax=ax)
    ax.set_title("Active Cell-Cell Interactions (Counts)", fontsize=13, fontweight='bold', pad=12)
    ax.set_xlabel("Receiver Cell Type", fontsize=10, fontweight='bold')
    ax.set_ylabel("Sender Cell Type", fontsize=10, fontweight='bold')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    fig_heat.savefig(os.path.join(args.plots_dir, "communication_heatmap.png"), dpi=150)
    plt.close(fig_heat)
    
    # 6. Plotting Circular Communication Network
    df_strength = df_interactions.pivot_table(index='sender', columns='receiver', values='score', aggfunc='sum', fill_value=0)
    df_strength = df_strength.reindex(index=cell_types, columns=cell_types, fill_value=0)
    
    G = nx.DiGraph()
    for c in cell_types:
        G.add_node(c)
        
    for sender in df_strength.index:
        for receiver in df_strength.columns:
            val = df_strength.loc[sender, receiver]
            if val > 0:
                G.add_edge(sender, receiver, weight=val)
                
    fig_net, ax = plt.subplots(figsize=(8, 8))
    pos = nx.circular_layout(G)
    
    nx.draw_networkx_nodes(G, pos, node_size=2500, node_color='#34495e', ax=ax)
    nx.draw_networkx_labels(G, pos, font_color='white', font_size=8, font_weight='bold', ax=ax)
    
    edges = G.edges(data=True)
    if len(edges) > 0:
        weights = [d['weight'] for u, v, d in edges]
        max_w = max(weights) if len(weights) > 0 else 1.0
        edge_widths = [(w / max_w) * 6 for w in weights]
        nx.draw_networkx_edges(
            G, pos, width=edge_widths, edge_color='#e67e22', 
            arrowstyle="->", arrowsize=15, connectionstyle="arc3,rad=0.1", ax=ax
        )
        
    ax.axis('off')
    ax.set_title("Circular Cell-Cell Communication Network\n(Line thickness corresponds to total interaction score)", fontsize=12, fontweight='bold')
    plt.tight_layout()
    fig_net.savefig(os.path.join(args.plots_dir, "communication_network.png"), dpi=150)
    plt.close(fig_net)
    
    # 7. Plotting Signaling Bubble/DotPlot
    # Show top 15 interactions for readability
    df_top = df_interactions.sort_values(by='score', ascending=False).head(15)
    
    df_top['pair'] = df_top['sender'] + " ➔ " + df_top['receiver']
    df_top['lr_pair'] = df_top['ligand'] + " - " + df_top['receptor'] + "\n(" + df_top['pathway'] + ")"
    
    # Map bubble sizes
    df_top['bubble_size'] = df_top['p_ligand'] * df_top['p_receptor'] * 500 + 30
    
    fig_dot, ax = plt.subplots(figsize=(10, 8))
    scat = ax.scatter(
        x=df_top['pair'], 
        y=df_top['lr_pair'], 
        s=df_top['bubble_size'], 
        c=df_top['score'], 
        cmap='YlOrRd', 
        edgecolors='black', 
        linewidth=0.5
    )
    
    plt.xticks(rotation=45, ha='right')
    cbar = plt.colorbar(scat, ax=ax)
    cbar.set_label("Interaction Score (Mean expression product)", fontsize=10, fontweight='bold')
    
    ax.set_title("Top Active Ligand-Receptor Signaling Pathways", fontsize=13, fontweight='bold', pad=12)
    ax.set_xlabel("Sender ➔ Receiver Pairs", fontsize=10, fontweight='bold')
    ax.set_ylabel("Ligand - Receptor Pair (Pathway)", fontsize=10, fontweight='bold')
    
    # Sizing legend
    sizes = [0.1, 0.5, 1.0]
    for s in sizes:
        ax.scatter([], [], c='gray', alpha=0.6, s=s * s * 500 + 30, label=f"{s*100:.0f}% Co-expr")
    ax.legend(title="Expression Frequency", labelspacing=1, loc='upper left', bbox_to_anchor=(1.25, 1.0))
    
    plt.tight_layout()
    fig_dot.savefig(os.path.join(args.plots_dir, "communication_dotplot.png"), dpi=150, bbox_inches='tight')
    plt.close(fig_dot)
    
    print("\n========================= TOP SIGNALING PATHWAYS =========================")
    for idx, row in df_top.head(5).iterrows():
        print(f" - {row['sender']} -> {row['receiver']} via {row['ligand']}-{row['receptor']} (Score: {row['score']:.3f})")
    print("=========================================================================")
    print("Modul 3 Cell-Cell Communication analysis completed successfully!")

if __name__ == "__main__":
    main()
