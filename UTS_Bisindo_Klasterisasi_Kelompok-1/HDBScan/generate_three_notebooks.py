"""
Script generator untuk membuat dan mengeksekusi 3 berkas Jupyter Notebook independen untuk HDBSCAN BISINDO:
1. hdbscan_euclidean.ipynb
2. hdbscan_manhattan.ipynb
3. hdbscan_cosine.ipynb
Disetarakan pada rentang 25-30 klaster dengan arsitektur 127 fitur tanpa emotikon.
"""

import os
import sys
import time
import nbformat as nbf
from nbconvert.preprocessors import ExecutePreprocessor

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

BASE_DIR = os.path.dirname(os.path.abspath(__file__)) if '__file__' in locals() else os.getcwd()
OUTPUT_DIR = os.path.join(BASE_DIR, "outputHDBScan")
os.makedirs(OUTPUT_DIR, exist_ok=True)

RANDOM_STATE = 42

CODE_LOAD_DATA = """# =======================================================================
# PEMUATAN DATASET BISINDO LANDMARKS (127 FITUR)
# =======================================================================
npz_path = os.path.join(BASE_DIR, "dataset_bisindo_landmarks.npz")
csv_path = os.path.join(BASE_DIR, "dataset_bisindo_landmarks.csv")

if os.path.exists(npz_path):
    data = np.load(npz_path, allow_pickle=True)
    X = data['X'].astype(np.float32)
    labels = data['labels']
    df_meta = pd.DataFrame({'label': labels, 'is_two_handed': data['is_two_handed']})
elif os.path.exists(csv_path):
    df_full = pd.read_csv(csv_path)
    meta_cols = ['label', 'label_idx', 'filename', 'num_hands', 'is_flip']
    meta_present = [c for c in meta_cols if c in df_full.columns]
    df_meta = df_full[meta_present]
    feat_cols = [c for c in df_full.columns if c not in meta_cols]
    X = df_full[feat_cols].values.astype(np.float32)
else:
    raise FileNotFoundError("Dataset BISINDO tidak ditemukan. Jalankan extract_features.py.")

print(f"[INFO] Dataset BISINDO dimuat: {X.shape[0]} sampel, {X.shape[1]} dimensi fitur.")"""

# =============================================================================
# BUILDER PROYEK 1: HDBSCAN EUCLIDEAN
# =============================================================================
def build_euclidean_notebook():
    nb = nbf.v4.new_notebook()
    nb.metadata = {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.13"}
    }
    cells = []

    cells.append(nbf.v4.new_markdown_cell(r"""# Proyek 1: HDBSCAN dengan Euclidean Distance (L2 Norm) pada BISINDO

**Deskripsi:**
Notebook ini mengimplementasikan algoritma HDBSCAN menggunakan Euclidean Distance pada dataset 127 fitur gestur alfabet BISINDO.

**Formulasi Euclidean:**
$$d_{\text{Euclidean}}(x, y) = \sqrt{\sum_{i=1}^{127} (x_i - y_i)^2}$$

Parameter `min_cluster_size = 46` dikalibrasi untuk menghasilkan 25 klaster stabil."""))

    cells.append(nbf.v4.new_code_cell("""# =======================================================================
# 1. SETUP LINGKUNGAN DAN IMPORT PUSTAKA
# =======================================================================
import os
import sys
import time
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.cluster import HDBSCAN
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
from scipy.cluster.hierarchy import dendrogram

warnings.filterwarnings('ignore')

plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
plt.rcParams['font.sans-serif'] = 'DejaVu Sans'
plt.rcParams['figure.dpi'] = 120

BASE_DIR = os.path.dirname(os.path.abspath(__file__)) if '__file__' in locals() else os.getcwd()
OUTPUT_DIR = os.path.join(BASE_DIR, "outputHDBScan")
os.makedirs(OUTPUT_DIR, exist_ok=True)

RANDOM_STATE = 42
MCS = 46

print(f"[INFO] Direktori: {BASE_DIR} | min_cluster_size = {MCS}")"""))

    cells.append(nbf.v4.new_code_cell(CODE_LOAD_DATA + """

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
print("[INFO] Fitur berhasil distandardisasi menggunakan StandardScaler.")"""))

    cells.append(nbf.v4.new_code_cell("""# =======================================================================
# 2. EKSEKUSI HDBSCAN EUCLIDEAN
# =======================================================================
t0 = time.perf_counter()
hdb = HDBSCAN(min_cluster_size=MCS, metric='euclidean', copy=True)
labels = hdb.fit_predict(X_scaled)
runtime = time.perf_counter() - t0

n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
noise_count = int(np.sum(labels == -1))
noise_ratio = (noise_count / len(labels)) * 100.0

print(f"[INFO] HDBSCAN Euclidean Selesai dalam {runtime:.4f} detik.")
print(f"[INFO] Jumlah Klaster Ditemukan: {n_clusters}")
print(f"[INFO] Jumlah Sampel Noise: {noise_count} ({noise_ratio:.2f}%)")

# Visualisasi Dendrogram Single Linkage Tree
tree_data = hdb._single_linkage_tree_
if hasattr(tree_data, 'dtype') and tree_data.dtype.names is not None:
    Z_scipy = np.column_stack([tree_data[name] for name in tree_data.dtype.names]).astype(float)
else:
    Z_scipy = np.array(tree_data, dtype=float)

plt.figure(figsize=(10, 5))
dendrogram(
    Z_scipy,
    truncate_mode='lastp',
    p=n_clusters,
    show_leaf_counts=True,
    above_threshold_color='#1E40AF',
    color_threshold=0
)
plt.title(f'Single Linkage Tree HDBSCAN Euclidean (mcs={MCS}, K={n_clusters})', fontsize=12, fontweight='bold', pad=10)
plt.xlabel('Cabang Klaster', fontsize=10)
plt.ylabel('Mutual Reachability Distance', fontsize=10)
plt.grid(True, linestyle=':', alpha=0.6)
plt.tight_layout()

dendro_path = os.path.join(OUTPUT_DIR, "dendrogram_euclidean.png")
plt.savefig(dendro_path, dpi=300)
print(f"[SAVE] Grafik disimpan ke: '{dendro_path}'")
plt.show()"""))

    cells.append(nbf.v4.new_code_cell("""# =======================================================================
# 3. EVALUASI METRIK KUANTITATIF EUCLIDEAN
# =======================================================================
mask = labels != -1
sil = float(silhouette_score(X_scaled[mask], labels[mask], metric='euclidean'))
db = float(davies_bouldin_score(X_scaled[mask], labels[mask]))
ch = float(calinski_harabasz_score(X_scaled[mask], labels[mask]))

df_res = pd.DataFrame([{
    'Distance Metric': 'Euclidean Distance (L2)',
    'Min Cluster Size': MCS,
    'Clusters Found': n_clusters,
    'Noise Samples (↓)': noise_count,
    'Noise Ratio (%) (↓)': round(noise_ratio, 2),
    'Clustered Samples (↑)': len(labels) - noise_count,
    'Silhouette Score (↑)': round(sil, 4),
    'Davies–Bouldin Index (↓)': round(db, 4),
    'Calinski–Harabasz Index (↑)': round(ch, 2),
    'Execution Time (s) (↓)': round(runtime, 4)
}])

print("\\n" + "="*85)
print(df_res.to_string(index=False))
print("="*85)

excel_out = os.path.join(OUTPUT_DIR, "evaluasi_hdbscan_euclidean.xlsx")
df_data_out = df_meta.copy()
df_data_out['cluster_euclidean'] = labels
df_data_out['prob_euclidean'] = hdb.probabilities_

with pd.ExcelWriter(excel_out, engine='openpyxl') as writer:
    df_res.to_excel(writer, sheet_name='Ringkasan_Metrik', index=False)
    pd.Series(labels).value_counts().to_frame(name='Jumlah_Sampel').to_excel(writer, sheet_name='Distribusi_Klaster')
    df_data_out.head(1000).to_excel(writer, sheet_name='Sampel_Prediksi_1000', index=False)

print(f"[SAVE] Berkas evaluasi Excel disimpan ke: '{excel_out}'")"""))

    cells.append(nbf.v4.new_code_cell("""# =======================================================================
# 4. VISUALISASI SEBARAN KLASTER DAN NOISE PCA 2D
# =======================================================================
pca = PCA(n_components=2, random_state=RANDOM_STATE)
X_pca = pca.fit_transform(X_scaled)

plt.figure(figsize=(8.5, 6))
plt.scatter(
    X_pca[mask, 0], X_pca[mask, 1],
    c=labels[mask], cmap='tab20', s=12, alpha=0.7, edgecolors='none'
)
plt.scatter(
    X_pca[~mask, 0], X_pca[~mask, 1],
    c='#CBD5E1', s=8, alpha=0.3, label=f'Noise ({noise_ratio:.1f}%)'
)
plt.title(f'Proyeksi PCA 2D HDBSCAN Euclidean (mcs={MCS}, K={n_clusters})\\nSil: {sil:.4f} | DBI: {db:.4f} | CH: {ch:.1f}', fontsize=11, fontweight='bold')
plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0]*100:.2f}%)')
plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1]*100:.2f}%)')
plt.grid(True, linestyle=':', alpha=0.6)
plt.legend(loc='upper right', frameon=True, facecolor='white')
plt.tight_layout()

pca_path = os.path.join(OUTPUT_DIR, "cluster_euclidean.png")
plt.savefig(pca_path, dpi=300)
print(f"[SAVE] Grafik disimpan ke: '{pca_path}'")
plt.show()"""))

    nb['cells'] = cells
    return nb

# =============================================================================
# BUILDER PROYEK 2: HDBSCAN MANHATTAN
# =============================================================================
def build_manhattan_notebook():
    nb = nbf.v4.new_notebook()
    nb.metadata = {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.13"}
    }
    cells = []

    cells.append(nbf.v4.new_markdown_cell(r"""# Proyek 2: HDBSCAN dengan Manhattan Distance (L1 Norm) pada BISINDO

**Deskripsi:**
Notebook ini mengimplementasikan algoritma HDBSCAN menggunakan Manhattan Distance pada dataset 127 fitur alfabet BISINDO.

**Formulasi Manhattan:**
$$d_{\text{Manhattan}}(x, y) = \sum_{i=1}^{127} |x_i - y_i|$$

Parameter `min_cluster_size = 48` dikalibrasi untuk menghasilkan 30 klaster stabil."""))

    cells.append(nbf.v4.new_code_cell("""# =======================================================================
# 1. SETUP LINGKUNGAN DAN IMPORT PUSTAKA
# =======================================================================
import os
import sys
import time
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.cluster import HDBSCAN
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
from scipy.cluster.hierarchy import dendrogram

warnings.filterwarnings('ignore')

plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
plt.rcParams['font.sans-serif'] = 'DejaVu Sans'
plt.rcParams['figure.dpi'] = 120

BASE_DIR = os.path.dirname(os.path.abspath(__file__)) if '__file__' in locals() else os.getcwd()
OUTPUT_DIR = os.path.join(BASE_DIR, "outputHDBScan")
os.makedirs(OUTPUT_DIR, exist_ok=True)

RANDOM_STATE = 42
MCS = 48

print(f"[INFO] Direktori: {BASE_DIR} | min_cluster_size = {MCS}")"""))

    cells.append(nbf.v4.new_code_cell(CODE_LOAD_DATA + """

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
print("[INFO] Fitur berhasil distandardisasi menggunakan StandardScaler.")"""))

    cells.append(nbf.v4.new_code_cell("""# =======================================================================
# 2. EKSEKUSI HDBSCAN MANHATTAN
# =======================================================================
t0 = time.perf_counter()
hdb = HDBSCAN(min_cluster_size=MCS, metric='manhattan', copy=True)
labels = hdb.fit_predict(X_scaled)
runtime = time.perf_counter() - t0

n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
noise_count = int(np.sum(labels == -1))
noise_ratio = (noise_count / len(labels)) * 100.0

print(f"[INFO] HDBSCAN Manhattan Selesai dalam {runtime:.4f} detik.")
print(f"[INFO] Jumlah Klaster Ditemukan: {n_clusters}")
print(f"[INFO] Jumlah Sampel Noise: {noise_count} ({noise_ratio:.2f}%)")

# Visualisasi Dendrogram Single Linkage Tree
tree_data = hdb._single_linkage_tree_
if hasattr(tree_data, 'dtype') and tree_data.dtype.names is not None:
    Z_scipy = np.column_stack([tree_data[name] for name in tree_data.dtype.names]).astype(float)
else:
    Z_scipy = np.array(tree_data, dtype=float)

plt.figure(figsize=(10, 5))
dendrogram(
    Z_scipy,
    truncate_mode='lastp',
    p=n_clusters,
    show_leaf_counts=True,
    above_threshold_color='#C2410C',
    color_threshold=0
)
plt.title(f'Single Linkage Tree HDBSCAN Manhattan (mcs={MCS}, K={n_clusters})', fontsize=12, fontweight='bold', pad=10)
plt.xlabel('Cabang Klaster', fontsize=10)
plt.ylabel('Mutual Reachability Distance (L1)', fontsize=10)
plt.grid(True, linestyle=':', alpha=0.6)
plt.tight_layout()

dendro_path = os.path.join(OUTPUT_DIR, "dendrogram_manhattan.png")
plt.savefig(dendro_path, dpi=300)
print(f"[SAVE] Grafik disimpan ke: '{dendro_path}'")
plt.show()"""))

    cells.append(nbf.v4.new_code_cell("""# =======================================================================
# 3. EVALUASI METRIK KUANTITATIF MANHATTAN
# =======================================================================
mask = labels != -1
sil = float(silhouette_score(X_scaled[mask], labels[mask], metric='euclidean'))
db = float(davies_bouldin_score(X_scaled[mask], labels[mask]))
ch = float(calinski_harabasz_score(X_scaled[mask], labels[mask]))

df_res = pd.DataFrame([{
    'Distance Metric': 'Manhattan Distance (L1)',
    'Min Cluster Size': MCS,
    'Clusters Found': n_clusters,
    'Noise Samples (↓)': noise_count,
    'Noise Ratio (%) (↓)': round(noise_ratio, 2),
    'Clustered Samples (↑)': len(labels) - noise_count,
    'Silhouette Score (↑)': round(sil, 4),
    'Davies–Bouldin Index (↓)': round(db, 4),
    'Calinski–Harabasz Index (↑)': round(ch, 2),
    'Execution Time (s) (↓)': round(runtime, 4)
}])

print("\\n" + "="*85)
print(df_res.to_string(index=False))
print("="*85)

excel_out = os.path.join(OUTPUT_DIR, "evaluasi_hdbscan_manhattan.xlsx")
df_data_out = df_meta.copy()
df_data_out['cluster_manhattan'] = labels
df_data_out['prob_manhattan'] = hdb.probabilities_

with pd.ExcelWriter(excel_out, engine='openpyxl') as writer:
    df_res.to_excel(writer, sheet_name='Ringkasan_Metrik', index=False)
    pd.Series(labels).value_counts().to_frame(name='Jumlah_Sampel').to_excel(writer, sheet_name='Distribusi_Klaster')
    df_data_out.head(1000).to_excel(writer, sheet_name='Sampel_Prediksi_1000', index=False)

print(f"[SAVE] Berkas evaluasi Excel disimpan ke: '{excel_out}'")"""))

    cells.append(nbf.v4.new_code_cell("""# =======================================================================
# 4. VISUALISASI SEBARAN KLASTER DAN NOISE PCA 2D
# =======================================================================
pca = PCA(n_components=2, random_state=RANDOM_STATE)
X_pca = pca.fit_transform(X_scaled)

plt.figure(figsize=(8.5, 6))
plt.scatter(
    X_pca[mask, 0], X_pca[mask, 1],
    c=labels[mask], cmap='tab20', s=12, alpha=0.7, edgecolors='none'
)
plt.scatter(
    X_pca[~mask, 0], X_pca[~mask, 1],
    c='#CBD5E1', s=8, alpha=0.3, label=f'Noise ({noise_ratio:.1f}%)'
)
plt.title(f'Proyeksi PCA 2D HDBSCAN Manhattan (mcs={MCS}, K={n_clusters})\\nSil: {sil:.4f} | DBI: {db:.4f} | CH: {ch:.1f}', fontsize=11, fontweight='bold')
plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0]*100:.2f}%)')
plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1]*100:.2f}%)')
plt.grid(True, linestyle=':', alpha=0.6)
plt.legend(loc='upper right', frameon=True, facecolor='white')
plt.tight_layout()

pca_path = os.path.join(OUTPUT_DIR, "cluster_manhattan.png")
plt.savefig(pca_path, dpi=300)
print(f"[SAVE] Grafik disimpan ke: '{pca_path}'")
plt.show()"""))

    nb['cells'] = cells
    return nb

# =============================================================================
# BUILDER PROYEK 3: HDBSCAN COSINE
# =============================================================================
def build_cosine_notebook():
    nb = nbf.v4.new_notebook()
    nb.metadata = {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.13"}
    }
    cells = []

    cells.append(nbf.v4.new_markdown_cell(r"""# Proyek 3: HDBSCAN dengan Cosine Similarity pada BISINDO

**Deskripsi:**
Notebook ini mengimplementasikan algoritma HDBSCAN pada permukaan hypersphere (Spherical HDBSCAN) menggunakan Cosine Similarity pada dataset 127 fitur gestur alfabet BISINDO.

**Formulasi Cosine:**
$$d_{\text{Cosine}}(x, y) = 1 - \frac{x \cdot y}{\|x\|_2 \|y\|_2}$$

Parameter `min_cluster_size = 53` dikalibrasi untuk menghasilkan 28 klaster stabil."""))

    cells.append(nbf.v4.new_code_cell("""# =======================================================================
# 1. SETUP LINGKUNGAN DAN IMPORT PUSTAKA
# =======================================================================
import os
import sys
import time
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.cluster import HDBSCAN
from sklearn.preprocessing import Normalizer, StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
from scipy.cluster.hierarchy import dendrogram

warnings.filterwarnings('ignore')

plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
plt.rcParams['font.sans-serif'] = 'DejaVu Sans'
plt.rcParams['figure.dpi'] = 120

BASE_DIR = os.path.dirname(os.path.abspath(__file__)) if '__file__' in locals() else os.getcwd()
OUTPUT_DIR = os.path.join(BASE_DIR, "outputHDBScan")
os.makedirs(OUTPUT_DIR, exist_ok=True)

RANDOM_STATE = 42
MCS = 53

print(f"[INFO] Direktori: {BASE_DIR} | min_cluster_size = {MCS}")"""))

    cells.append(nbf.v4.new_code_cell(CODE_LOAD_DATA + """

normalizer = Normalizer(norm='l2')
X_norm = normalizer.fit_transform(X)
print("[INFO] Fitur berhasil dinormalisasi ke unit hypersphere (L2 Norm = 1).")"""))

    cells.append(nbf.v4.new_code_cell("""# =======================================================================
# 2. EKSEKUSI HDBSCAN COSINE (SPHERICAL)
# =======================================================================
t0 = time.perf_counter()
hdb = HDBSCAN(min_cluster_size=MCS, metric='euclidean', copy=True)
labels = hdb.fit_predict(X_norm)
runtime = time.perf_counter() - t0

n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
noise_count = int(np.sum(labels == -1))
noise_ratio = (noise_count / len(labels)) * 100.0

print(f"[INFO] HDBSCAN Cosine Selesai dalam {runtime:.4f} detik.")
print(f"[INFO] Jumlah Klaster Ditemukan: {n_clusters}")
print(f"[INFO] Jumlah Sampel Noise: {noise_count} ({noise_ratio:.2f}%)")

# Visualisasi Dendrogram Single Linkage Tree
tree_data = hdb._single_linkage_tree_
if hasattr(tree_data, 'dtype') and tree_data.dtype.names is not None:
    Z_scipy = np.column_stack([tree_data[name] for name in tree_data.dtype.names]).astype(float)
else:
    Z_scipy = np.array(tree_data, dtype=float)

plt.figure(figsize=(10, 5))
dendrogram(
    Z_scipy,
    truncate_mode='lastp',
    p=n_clusters,
    show_leaf_counts=True,
    above_threshold_color='#15803D',
    color_threshold=0
)
plt.title(f'Single Linkage Tree HDBSCAN Cosine (mcs={MCS}, K={n_clusters})', fontsize=12, fontweight='bold', pad=10)
plt.xlabel('Cabang Klaster', fontsize=10)
plt.ylabel('Mutual Reachability Distance (Angular)', fontsize=10)
plt.grid(True, linestyle=':', alpha=0.6)
plt.tight_layout()

dendro_path = os.path.join(OUTPUT_DIR, "dendrogram_cosine.png")
plt.savefig(dendro_path, dpi=300)
print(f"[SAVE] Grafik disimpan ke: '{dendro_path}'")
plt.show()"""))

    cells.append(nbf.v4.new_code_cell("""# =======================================================================
# 3. EVALUASI METRIK KUANTITATIF COSINE
# =======================================================================
mask = labels != -1
sil = float(silhouette_score(X_norm[mask], labels[mask], metric='cosine'))
db = float(davies_bouldin_score(X_norm[mask], labels[mask]))
ch = float(calinski_harabasz_score(X_norm[mask], labels[mask]))

df_res = pd.DataFrame([{
    'Distance Metric': 'Cosine Distance (Angular)',
    'Min Cluster Size': MCS,
    'Clusters Found': n_clusters,
    'Noise Samples (↓)': noise_count,
    'Noise Ratio (%) (↓)': round(noise_ratio, 2),
    'Clustered Samples (↑)': len(labels) - noise_count,
    'Silhouette Score (↑)': round(sil, 4),
    'Davies–Bouldin Index (↓)': round(db, 4),
    'Calinski–Harabasz Index (↑)': round(ch, 2),
    'Execution Time (s) (↓)': round(runtime, 4)
}])

print("\\n" + "="*85)
print(df_res.to_string(index=False))
print("="*85)

excel_out = os.path.join(OUTPUT_DIR, "evaluasi_hdbscan_cosine.xlsx")
df_data_out = df_meta.copy()
df_data_out['cluster_cosine'] = labels
df_data_out['prob_cosine'] = hdb.probabilities_

with pd.ExcelWriter(excel_out, engine='openpyxl') as writer:
    df_res.to_excel(writer, sheet_name='Ringkasan_Metrik', index=False)
    pd.Series(labels).value_counts().to_frame(name='Jumlah_Sampel').to_excel(writer, sheet_name='Distribusi_Klaster')
    df_data_out.head(1000).to_excel(writer, sheet_name='Sampel_Prediksi_1000', index=False)

print(f"[SAVE] Berkas evaluasi Excel disimpan ke: '{excel_out}'")"""))

    cells.append(nbf.v4.new_code_cell("""# =======================================================================
# 4. VISUALISASI SEBARAN KLASTER DAN NOISE PCA 2D
# =======================================================================
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
pca = PCA(n_components=2, random_state=RANDOM_STATE)
X_pca = pca.fit_transform(X_scaled)

plt.figure(figsize=(8.5, 6))
plt.scatter(
    X_pca[mask, 0], X_pca[mask, 1],
    c=labels[mask], cmap='tab20', s=12, alpha=0.7, edgecolors='none'
)
plt.scatter(
    X_pca[~mask, 0], X_pca[~mask, 1],
    c='#CBD5E1', s=8, alpha=0.3, label=f'Noise ({noise_ratio:.1f}%)'
)
plt.title(f'Proyeksi PCA 2D HDBSCAN Cosine (mcs={MCS}, K={n_clusters})\\nSil: {sil:.4f} | DBI: {db:.4f} | CH: {ch:.1f}', fontsize=11, fontweight='bold')
plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0]*100:.2f}%)')
plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1]*100:.2f}%)')
plt.grid(True, linestyle=':', alpha=0.6)
plt.legend(loc='upper right', frameon=True, facecolor='white')
plt.tight_layout()

pca_path = os.path.join(OUTPUT_DIR, "cluster_cosine.png")
plt.savefig(pca_path, dpi=300)
print(f"[SAVE] Grafik disimpan ke: '{pca_path}'")
plt.show()"""))

    nb['cells'] = cells
    return nb

# =============================================================================
# RUNNER UTAMA
# =============================================================================
def main():
    configs = [
        ("hdbscan_euclidean.ipynb", build_euclidean_notebook),
        ("hdbscan_manhattan.ipynb", build_manhattan_notebook),
        ("hdbscan_cosine.ipynb", build_cosine_notebook)
    ]
    
    ep = ExecutePreprocessor(timeout=600, kernel_name='python3')
    
    for fname, builder in configs:
        print(f"\\n{'='*75}")
        print(f"[START] Memproses notebook: {fname}")
        print(f"{'='*75}")
        nb = builder()
        nb_path = os.path.join(BASE_DIR, fname)
        with open(nb_path, 'w', encoding='utf-8') as f:
            nbf.write(nb, f)
            
        print(f"[EXECUTE] Menjalankan kode pada {fname}...")
        t0 = time.time()
        ep.preprocess(nb, {'metadata': {'path': BASE_DIR}})
        print(f"[DONE] Eksekusi selesai dalam {time.time() - t0:.2f} detik.")
        
        with open(nb_path, 'w', encoding='utf-8') as f:
            nbf.write(nb, f)
        print(f"[SAVED] Notebook berhasil disimpan: {nb_path}")

if __name__ == "__main__":
    main()
