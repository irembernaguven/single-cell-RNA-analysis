# Single-Cell RNA-Seq (scRNA) Analysis of Immune Evasion in TNBC

This repository focuses on analyzing single-cell RNA sequencing data to investigate how cancer cells in **Triple-Negative Breast Cancer (TNBC)** evade the immune system at a single-cell resolution. We trace the interactions between tumor cells and T-lymphocytes within the tumor microenvironment (TME) to understand the micro-level drivers of immune evasion and T-cell exhaustion.

---

## 📁 Repository Structure

```text
tnbc_immune_evasion_scRNA/
│
├── data/                                 # Raw and preprocessed single-cell .h5ad datasets (git-ignored)
│   └── tnbc_tme_data.h5ad                # Raw input dataset
│
├── plots/                                # Quality Control (QC) and analytics figures (git-ignored)
│   ├── qc_violin_pre.png                 # QC metrics violin distribution before filtering
│   └── qc_violin_post.png                # QC metrics violin distribution after filtering
│
├── 01_scRNA_preprocessing_QC.py          # Modul 1: Cell/gene quality control, mitochondrial detection & filtering
├── 02_clustering_and_annotation.py       # Modul 2: Dimension reduction (UMAP), batch correction, cell labeling
├── 03_cell_cell_communication.py         # Modul 3: Cancer-T Cell communication network mapping
├── 04_trajectory_pseudotime.py           # Modul 4: Pseudotime trajectory of T-cell exhaustion
│
├── requirements.txt                      # scanpy, anndata, pandas, matplotlib, seaborn, etc.
└── README.md                             # Project documentation and results presentation
```

---

## 🧬 Pipeline Modules

### Modul 1: Quality Control and Preprocessing (`01_scRNA_preprocessing_QC.py`)
Filtres out low-quality cell transcriptomes, including:
*   **Empty droplets / non-viable cells:** Cells expressing too few genes (`min_genes < 200`).
*   **Doublets:** Droplets containing multiple cells (`max_genes > 4500`).
*   **Dying cells:** High proportion of mitochondrial reads (`pct_counts_mt > 15.0%`).
*   **Lowly expressed genes:** Genes detected in fewer than 3 cells.

Plots the metrics using violin and scatter distributions before and after filtering to verify the cleanup process.

---

## 🚀 Getting Started

### 1. Clone the Repository
```bash
git clone https://github.com/irembernaguven/single-cell-RNA-analysis.git
cd single-cell-RNA-analysis
```

### 2. Install Dependencies
Install all required python packages:
```bash
pip install -r requirements.txt
```

### 3. Run Quality Control & Preprocessing
You can test the entire pipeline using the built-in demo flag. This will automatically download a public 3k PBMC dataset from Scanpy to run the QC calculations:
```bash
python 01_scRNA_preprocessing_QC.py --demo
```

Upon successful run:
*   Pre- and post-filtering violin plots will be saved to `plots/`.
*   The preprocessed dataset will be saved to `data/tnbc_tme_data_qc.h5ad`.
