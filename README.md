[🇬🇧 English](README.md) | [🇪🇸 Español](README.es.md)

# MNIST Unsupervised Clustering & Latent Space Analysis Pipeline

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.10%20%7C%203.11%20%7C%203.12-3776AB?style=for-the-badge&logo=python&logoColor=white" alt="Python Version"/>
  <img src="https://img.shields.io/badge/Scikit--Learn-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white" alt="Scikit-Learn"/>
  <img src="https://img.shields.io/badge/NumPy-013243?style=for-the-badge&logo=numpy&logoColor=white" alt="NumPy"/>
  <img src="https://img.shields.io/badge/SciPy-8CAAE6?style=for-the-badge&logo=scipy&logoColor=white" alt="SciPy"/>
  <img src="https://img.shields.io/badge/Domain-Unsupervised%20Learning-blueviolet?style=for-the-badge" alt="Unsupervised Learning"/>
  <img src="https://img.shields.io/badge/License-MIT-yellow?style=for-the-badge" alt="MIT License"/>
</p>

---

## 📌 Executive Summary

**MNIST Unsupervised Clustering & Latent Space Analysis** is an experimental benchmarking pipeline focused on unsupervised representation learning over high-dimensional image data (**MNIST 784 dimensions**).

The pipeline addresses the core challenge of discovering natural cluster geometry without label supervision during training. It integrates **Principal Component Analysis (PCA)** for spectral dimensionality reduction, **K-Means Clustering**, **Gaussian Mixture Models (GMM)**, and combinatorial bipartite matching via the **Hungarian Algorithm (Kuhn-Munkres)** to solve the unsupervised class-to-cluster correspondence problem and evaluate true latent partition purity.

---

## 🔬 Experimental Methodology & Visual Results

### 1. Exploratory Data Analysis & Preprocessing (EDA)
Automated ingestion of `mnist_784` from OpenML, pixel normalization to $[0, 1]$ for numeric stability, and stratified partitioning with an independent 10,000-sample test set to prevent data leakage.

<p align="center">
  <img src="assets/eda_digits_sample.png" width="650" alt="MNIST Digit Samples"/>
  <br>
  <em>Figure 1: Representative digit samples across normalized pixel space.</em>
</p>

---

### 2. Spectral Dimensionality Reduction with PCA
Evaluated spectral variance retention across latent subspace dimensions: $d \in [10, 20, 40, 60, 100, 150, 784]$.
* **Finding:** Projecting onto $d \in [40, 60]$ dimensions retains over 85% of cumulative explained variance, filtering high-frequency boundary noise and accelerating K-Means convergence by over 90% compared to the raw 784-dimensional space.

---

### 3. Hyperparameter Sweeps ($K$) & Generalization
Systematic sweeps of $K \in [8, 30]$ evaluated against intrinsic clustering metrics (Silhouette Coefficient and Calinski-Harabasz Index) alongside bipartite Class-to-Cluster Accuracy:

<p align="center">
  <img src="assets/accuracy_vs_k.png" width="700" alt="Accuracy evolution across K"/>
  <br>
  <em>Figure 2: Consistent generalization between Train and Test accuracy curves as K increases.</em>
</p>

---

### 4. Optimal Hungarian Alignment & Confusion Heatmaps
Because unsupervised clustering assigns arbitrary permutation IDs to discovered clusters, the **Hungarian Algorithm** (`scipy.optimize.linear_sum_assignment`) is applied over the contingency cross-matrix to resolve the minimum-cost one-to-one class assignment:

<p align="center">
  <img src="assets/confusion_matrix_k10.png" width="460" alt="Confusion Matrix K=10"/>
  &nbsp;&nbsp;
  <img src="assets/heatmap_best_k.png" width="460" alt="Heatmap Best K"/>
  <br>
  <em>Figure 3: Confusion matrix for K=10 (left) and dominant-class decomposition for optimal K (right).</em>
</p>

---

### 5. 2D Latent Space Projection
Comparative visualization of the first two principal components, contrasting unsupervised K-Means partitions against true ground-truth class labels:

<p align="center">
  <img src="assets/pca_2d_clusters.png" width="800" alt="PCA 2D Projection"/>
  <br>
  <em>Figure 4: Latent density separation in 2D PCA subspace (Discovered Clusters vs True Classes).</em>
</p>

---

## 💡 Key Analytical Insights & Findings

1. **The Sub-Clustering Dynamic ($K > 10$):**
   Although ground-truth labels consist of 10 digits (0 to 9), increasing $K$ to 20 or 30 significantly improves cluster purity and accuracy. K-Means operates under spherical, isotropic assumptions, whereas handwritten digits naturally exhibit **multimodal geometric distributions** based on human typographic style (e.g., European '7' with crossbar vs American continuous '7', or slanted '1' vs vertical bar '1'). Sub-clustering allows the algorithm to fit distinct Voronoi cells per style without forcing heterogeneous writing styles into a single centroid.
2. **Topological Ambiguity Boundaries:**
   Confusion matrices demonstrate that primary cluster overlap occurs between homologous stroke topologies: **(4, 9)** and **(3, 5)**. In reduced latent space, stroke closure ambiguity produces continuous density distributions that motivate probabilistic approaches like **GMM**.
3. **Out-of-Sample Inference without Data Leakage:**
   The codebase provides an inference module (`src/inference.py` / `predict_new_instance`) that projects unseen test samples strictly using pre-fitted transformations, demonstrating production readiness.

---

## 🏗️ Repository Structure

```text
MNIST_Clustering/
├── assets/                    # Experimental charts and figures
│   ├── accuracy_vs_k.png
│   ├── confusion_matrix_k10.png
│   ├── eda_digits_sample.png
│   ├── heatmap_best_k.png
│   └── pca_2d_clusters.png
├── src/                       # Production inference modules
│   ├── __init__.py
│   └── inference.py           # Hungarian alignment & out-of-sample predictor
├── notebook.ipynb             # Interactive end-to-end experimental notebook
├── requirements.txt           # Reproducible dependencies manifest
├── .gitignore                 # Clean Git exclusions
└── LICENSE                    # MIT License
```

---

## ⚙️ Installation & Usage

### 1. Clone repository and initialize environment:
```bash
git clone https://github.com/aimarlarriba/MNIST_Clustering.git
cd MNIST_Clustering

# Create virtual environment
python -m venv venv

# Activate on Windows:
venv\Scripts\activate
# Activate on Linux/macOS:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Launch the Research Notebook:
```bash
jupyter notebook notebook.ipynb
```

---

## 👥 Academic Context & Attribution

Originally developed as an experimental project for the **Data Mining** course at the **University of the Basque Country (UPV/EHU)**.

Refactored, benchmarked, and documented by **[Aimar Larriba](https://github.com/aimarlarriba)** as a portfolio piece in unsupervised learning and latent space representation analysis.

---

## ⚖️ License

Distributed under the **MIT** License. See [LICENSE](LICENSE) for more details.
