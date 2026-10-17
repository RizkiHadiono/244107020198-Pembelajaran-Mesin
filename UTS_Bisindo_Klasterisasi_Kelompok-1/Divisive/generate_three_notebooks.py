"""
Script generator untuk membuat dan mengeksekusi 3 berkas Jupyter Notebook independen untuk Divisive Hierarchical Clustering BISINDO:
1. divisive_euclidean.ipynb
2. divisive_manhattan.ipynb
3. divisive_cosine.ipynb
Disetarakan pada K = 26 klaster dengan arsitektur 127 fitur tanpa emotikon.
"""

import os
import sys
import time
import nbformat as nbf
from nbconvert.preprocessors import ExecutePreprocessor

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

BASE_DIR = os.path.dirname(os.path.abspath(__file__)) if '__file__' in locals() else os.getcwd()
OUTPUT_DIR = os.path.join(BASE_DIR, "outputDivisive")
os.makedirs(OUTPUT_DIR, exist_ok=True)

TARGET_CLUSTERS = 26
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

CODE_LINKAGE_HELPER = """def build_linkage_matrix_from_tree(tree, terminal_leaves, distance_metric='euclidean'):
    leaf_to_id = {id(leaf): idx for idx, leaf in enumerate(terminal_leaves)}
    merges = []
    next_id = len(terminal_leaves)
    
    def traverse(node):
        nonlocal next_id
        if id(node) in leaf_to_id:
            center = node.center if hasattr(node, 'center') else np.zeros(127)
            return leaf_to_id[id(node)], 1, center
            
        left_id, left_count, left_center = traverse(node.left)
        right_id, right_count, right_center = traverse(node.right)
        
        if distance_metric == 'cosine':
            dot = np.dot(left_center, right_center)
            norm = np.linalg.norm(left_center) * np.linalg.norm(right_center)
            dist = max(0.001, float(1.0 - dot / (norm + 1e-9)))
        elif distance_metric == 'manhattan':
            dist = max(0.001, float(np.sum(np.abs(left_center - right_center))))
        else:
            dist = max(0.001, float(np.linalg.norm(left_center - right_center)))
            
        current_id = next_id
        next_id += 1
        total_count = left_count + right_count
        center = (left_center * left_count + right_center * right_count) / total_count
        
        merges.append([left_id, right_id, dist, total_count])
        return current_id, total_count, center

    traverse(tree)
    Z = np.array(merges, dtype=float)
    for i in range(len(Z)):
        c1, c2 = int(Z[i, 0]), int(Z[i, 1])
        h1 = Z[c1 - len(terminal_leaves), 2] if c1 >= len(terminal_leaves) else 0.0
        h2 = Z[c2 - len(terminal_leaves), 2] if c2 >= len(terminal_leaves) else 0.0
        min_h = max(h1, h2) + 0.05 * (max(h1, h2) + 0.1)
        if Z[i, 2] <= min_h:
            Z[i, 2] = min_h
    return Z"""

# =============================================================================
# BUILDER PROYEK 1: DIVISIVE EUCLIDEAN
# =============================================================================
def build_euclidean_notebook():
    nb = nbf.v4.new_notebook()
    nb.metadata = {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.13"}
    }
    cells = []

    cells.append(nbf.v4.new_markdown_cell(r"""# Proyek 1: Divisive Hierarchical Clustering dengan Euclidean Distance (L2 Norm) pada BISINDO

**Deskripsi:**
Notebook ini mengimplementasikan Divisive Hierarchical Clustering (Top-Down Bipartitioning) menggunakan Bisecting 2-Means berbasis Euclidean Distance pada dataset 127 fitur gestur alfabet BISINDO.

**Formulasi Euclidean:**
$$d_{\text{Euclidean}}(x, y) = \sqrt{\sum_{i=1}^{127} (x_i - y_i)^2}$$

Pada setiap tahap, klaster dengan inersia sum-of-squares terbesar dipilih untuk dibelah menjadi 2 sub-klaster hingga terbentuk $K = 26$ klaster."""))

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

from sklearn.cluster import BisectingKMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
from scipy.cluster.hierarchy import dendrogram

warnings.filterwarnings('ignore')

plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
plt.rcParams['font.sans-serif'] = 'DejaVu Sans'
plt.rcParams['figure.dpi'] = 120

BASE_DIR = os.path.dirname(os.path.abspath(__file__)) if '__file__' in locals() else os.getcwd()
OUTPUT_DIR = os.path.join(BASE_DIR, "outputDivisive")
os.makedirs(OUTPUT_DIR, exist_ok=True)

TARGET_CLUSTERS = 26
RANDOM_STATE = 42

print(f"[INFO] Direktori: {BASE_DIR} | Target Klaster = {TARGET_CLUSTERS}")"""))

    cells.append(nbf.v4.new_code_cell(CODE_LOAD_DATA + """

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
print("[INFO] Fitur berhasil distandardisasi menggunakan StandardScaler.")"""))

    cells.append(nbf.v4.new_code_cell(CODE_LINKAGE_HELPER + """

# =======================================================================
# 2. EKSEKUSI DIVISIVE BISECTING 2-MEANS (EUCLIDEAN)
# =======================================================================
t0 = time.perf_counter()
bkm = BisectingKMeans(
    n_clusters=TARGET_CLUSTERS,
    random_state=RANDOM_STATE,
    bisecting_strategy='biggest_inertia'
)
labels = bkm.fit_predict(X_scaled)
runtime_divisive = time.perf_counter() - t0

euc_leaves = list(bkm._bisecting_tree.iter_leaves())
Z = build_linkage_matrix_from_tree(bkm._bisecting_tree, euc_leaves, distance_metric='euclidean')

plt.figure(figsize=(10, 5.5))
ddata = dendrogram(
    Z,
    labels=[f"C{i+1}" for i in range(TARGET_CLUSTERS)],
    leaf_rotation=90,
    leaf_font_size=8,
    above_threshold_color='#1E40AF',
    color_threshold=0
)
plt.title(f'Pohon Divisive: Bisecting 2-Means Euclidean (K = {TARGET_CLUSTERS})', fontsize=12, fontweight='bold', pad=10)
plt.xlabel('Klaster Terminal (C1 - C26)', fontsize=10)
plt.ylabel('Disimilaritas Pembelahan (L2 Norm)', fontsize=10)
plt.grid(True, linestyle=':', alpha=0.6)
plt.tight_layout()

dendro_path = os.path.join(OUTPUT_DIR, "dendrogram_euclidean.png")
plt.savefig(dendro_path, dpi=300)
print(f"[SAVE] Grafik disimpan ke: '{dendro_path}'")
plt.show()"""))

    cells.append(nbf.v4.new_code_cell("""# =======================================================================
# 3. EVALUASI METRIK KUANTITATIF EUCLIDEAN
# =======================================================================
sil = float(silhouette_score(X_scaled, labels, metric='euclidean'))
db = float(davies_bouldin_score(X_scaled, labels))
ch = float(calinski_harabasz_score(X_scaled, labels))
counts = pd.Series(labels).value_counts()

df_res = pd.DataFrame([{
    'Distance Metric': 'Euclidean Distance (L2)',
    'Divisive Strategy': 'Bisecting 2-Means (Inertia)',
    'Clusters (K)': TARGET_CLUSTERS,
    'Min Cluster Size': int(counts.min()),
    'Max Cluster Size': int(counts.max()),
    'Silhouette Score (↑)': round(sil, 4),
    'Davies–Bouldin Index (↓)': round(db, 4),
    'Calinski–Harabasz Index (↑)': round(ch, 2),
    'Execution Time (s) (↓)': round(runtime_divisive, 4)
}])

print("\\n" + "="*85)
print(df_res.to_string(index=False))
print("="*85)

excel_out = os.path.join(OUTPUT_DIR, "evaluasi_divisive_euclidean.xlsx")
df_data_out = df_meta.copy()
df_data_out['cluster_euclidean'] = labels

with pd.ExcelWriter(excel_out, engine='openpyxl') as writer:
    df_res.to_excel(writer, sheet_name='Ringkasan_Metrik', index=False)
    counts.to_frame(name='Jumlah_Sampel').to_excel(writer, sheet_name='Distribusi_Ukuran_Klaster')
    df_data_out.head(1000).to_excel(writer, sheet_name='Data_Klaster_Sampel', index=False)

print(f"[SAVE] Berkas evaluasi Excel disimpan ke: '{excel_out}'")"""))

    cells.append(nbf.v4.new_code_cell("""# =======================================================================
# 4. VISUALISASI SEBARAN KLASTER PCA 2D
# =======================================================================
pca = PCA(n_components=2, random_state=RANDOM_STATE)
X_pca = pca.fit_transform(X_scaled)

plt.figure(figsize=(8.5, 6))
plt.scatter(
    X_pca[:, 0], X_pca[:, 1],
    c=labels,
    cmap='tab20',
    s=12,
    alpha=0.6,
    edgecolors='none'
)
plt.title(f'Proyeksi PCA 2D Klaster Divisive Euclidean (K = {TARGET_CLUSTERS})\\nSil: {sil:.4f} | DBI: {db:.4f} | CH: {ch:.1f}', fontsize=11, fontweight='bold')
plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0]*100:.2f}%)')
plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1]*100:.2f}%)')
plt.grid(True, linestyle=':', alpha=0.6)
plt.tight_layout()

pca_path = os.path.join(OUTPUT_DIR, "cluster_euclidean.png")
plt.savefig(pca_path, dpi=300)
print(f"[SAVE] Grafik disimpan ke: '{pca_path}'")
plt.show()"""))

    nb['cells'] = cells
    return nb

# =============================================================================
# BUILDER PROYEK 2: DIVISIVE MANHATTAN
# =============================================================================
def build_manhattan_notebook():
    nb = nbf.v4.new_notebook()
    nb.metadata = {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.13"}
    }
    cells = []

    cells.append(nbf.v4.new_markdown_cell(r"""# Proyek 2: Divisive Hierarchical Clustering dengan Manhattan Distance (L1 Norm) pada BISINDO

**Deskripsi:**
Notebook ini mengimplementasikan Divisive Hierarchical Clustering menggunakan Bisecting 2-Medians berbasis Manhattan Distance pada dataset 127 fitur alfabet BISINDO.

**Formulasi Manhattan:**
$$d_{\text{Manhattan}}(x, y) = \sum_{i=1}^{127} |x_i - y_i|$$

Pada setiap tahap, klaster dengan dispersi mutlak $L_1$ terbesar dipilih untuk dibelah menjadi 2 sub-klaster menggunakan median komponen fitur."""))

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

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
from scipy.cluster.hierarchy import dendrogram

warnings.filterwarnings('ignore')

plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
plt.rcParams['font.sans-serif'] = 'DejaVu Sans'
plt.rcParams['figure.dpi'] = 120

BASE_DIR = os.path.dirname(os.path.abspath(__file__)) if '__file__' in locals() else os.getcwd()
OUTPUT_DIR = os.path.join(BASE_DIR, "outputDivisive")
os.makedirs(OUTPUT_DIR, exist_ok=True)

TARGET_CLUSTERS = 26
RANDOM_STATE = 42

print(f"[INFO] Direktori: {BASE_DIR} | Target Klaster = {TARGET_CLUSTERS}")"""))

    cells.append(nbf.v4.new_code_cell(CODE_LOAD_DATA + """

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
print("[INFO] Fitur berhasil distandardisasi menggunakan StandardScaler.")"""))

    cells.append(nbf.v4.new_code_cell(CODE_LINKAGE_HELPER + """

# =======================================================================
# 2. ENGINE BISECTING 2-MEDIANS MANHATTAN
# =======================================================================
class DivisiveManhattanNode:
    def __init__(self, indices, center, disp):
        self.indices = indices
        self.center = center
        self.disp = disp
        self.left = None
        self.right = None

def bisect_manhattan_cluster(X_sub, random_state=42):
    n = len(X_sub)
    if n <= 2:
        return np.zeros(n, dtype=int), X_sub[0], X_sub[-1]
    np.random.seed(random_state)
    idx1 = np.random.randint(0, n)
    c1 = X_sub[idx1].copy()
    dists = np.sum(np.abs(X_sub - c1), axis=1)
    idx2 = np.argmax(dists)
    c2 = X_sub[idx2].copy()
    
    labels = np.zeros(n, dtype=int)
    for it in range(25):
        d1 = np.sum(np.abs(X_sub - c1), axis=1)
        d2 = np.sum(np.abs(X_sub - c2), axis=1)
        new_labels = (d2 < d1).astype(int)
        if np.sum(new_labels == 0) == 0 or np.sum(new_labels == 1) == 0:
            break
        if np.array_equal(new_labels, labels) and it > 0:
            break
        labels = new_labels
        c1 = np.median(X_sub[labels == 0], axis=0)
        c2 = np.median(X_sub[labels == 1], axis=0)
    return labels, c1, c2

t0 = time.perf_counter()
N = len(X_scaled)
root_med = np.median(X_scaled, axis=0)
root_disp = np.sum(np.abs(X_scaled - root_med))
root_node = DivisiveManhattanNode(np.arange(N), root_med, root_disp)

leaf_nodes = [root_node]
for step in range(TARGET_CLUSTERS - 1):
    leaf_nodes.sort(key=lambda node: node.disp, reverse=True)
    target = leaf_nodes.pop(0)
    sub_X = X_scaled[target.indices]
    sub_labels, c1, c2 = bisect_manhattan_cluster(sub_X, random_state=RANDOM_STATE + step)
    
    idx1 = target.indices[sub_labels == 0]
    idx2 = target.indices[sub_labels == 1]
    if len(idx1) == 0 or len(idx2) == 0:
        target.disp = 0
        leaf_nodes.append(target)
        continue
        
    disp1 = np.sum(np.abs(X_scaled[idx1] - c1))
    disp2 = np.sum(np.abs(X_scaled[idx2] - c2))
    node1 = DivisiveManhattanNode(idx1, c1, disp1)
    node2 = DivisiveManhattanNode(idx2, c2, disp2)
    target.left = node1
    target.right = node2
    leaf_nodes.append(node1)
    leaf_nodes.append(node2)

labels = np.zeros(N, dtype=int)
for c_id, node in enumerate(leaf_nodes):
    labels[node.indices] = c_id
runtime_manhattan = time.perf_counter() - t0

Z = build_linkage_matrix_from_tree(root_node, leaf_nodes, distance_metric='manhattan')

plt.figure(figsize=(10, 5.5))
ddata = dendrogram(
    Z,
    labels=[f"C{i+1}" for i in range(TARGET_CLUSTERS)],
    leaf_rotation=90,
    leaf_font_size=8,
    above_threshold_color='#C2410C',
    color_threshold=0
)
plt.title(f'Pohon Divisive: Bisecting 2-Medians Manhattan (K = {TARGET_CLUSTERS})', fontsize=12, fontweight='bold', pad=10)
plt.xlabel('Klaster Terminal (C1 - C26)', fontsize=10)
plt.ylabel('Disimilaritas Pembelahan (L1 Norm)', fontsize=10)
plt.grid(True, linestyle=':', alpha=0.6)
plt.tight_layout()

dendro_path = os.path.join(OUTPUT_DIR, "dendrogram_manhattan.png")
plt.savefig(dendro_path, dpi=300)
print(f"[SAVE] Grafik disimpan ke: '{dendro_path}'")
plt.show()"""))

    cells.append(nbf.v4.new_code_cell("""# =======================================================================
# 3. EVALUASI METRIK KUANTITATIF MANHATTAN
# =======================================================================
sil = float(silhouette_score(X_scaled, labels, metric='euclidean'))
db = float(davies_bouldin_score(X_scaled, labels))
ch = float(calinski_harabasz_score(X_scaled, labels))
counts = pd.Series(labels).value_counts()

df_res = pd.DataFrame([{
    'Distance Metric': 'Manhattan Distance (L1)',
    'Divisive Strategy': 'Bisecting 2-Medians (L1 Disp)',
    'Clusters (K)': TARGET_CLUSTERS,
    'Min Cluster Size': int(counts.min()),
    'Max Cluster Size': int(counts.max()),
    'Silhouette Score (↑)': round(sil, 4),
    'Davies–Bouldin Index (↓)': round(db, 4),
    'Calinski–Harabasz Index (↑)': round(ch, 2),
    'Execution Time (s) (↓)': round(runtime_manhattan, 4)
}])

print("\\n" + "="*85)
print(df_res.to_string(index=False))
print("="*85)

excel_out = os.path.join(OUTPUT_DIR, "evaluasi_divisive_manhattan.xlsx")
df_data_out = df_meta.copy()
df_data_out['cluster_manhattan'] = labels

with pd.ExcelWriter(excel_out, engine='openpyxl') as writer:
    df_res.to_excel(writer, sheet_name='Ringkasan_Metrik', index=False)
    counts.to_frame(name='Jumlah_Sampel').to_excel(writer, sheet_name='Distribusi_Ukuran_Klaster')
    df_data_out.head(1000).to_excel(writer, sheet_name='Data_Klaster_Sampel', index=False)

print(f"[SAVE] Berkas evaluasi Excel disimpan ke: '{excel_out}'")"""))

    cells.append(nbf.v4.new_code_cell("""# =======================================================================
# 4. VISUALISASI SEBARAN KLASTER PCA 2D
# =======================================================================
pca = PCA(n_components=2, random_state=RANDOM_STATE)
X_pca = pca.fit_transform(X_scaled)

plt.figure(figsize=(8.5, 6))
plt.scatter(
    X_pca[:, 0], X_pca[:, 1],
    c=labels,
    cmap='tab20',
    s=12,
    alpha=0.6,
    edgecolors='none'
)
plt.title(f'Proyeksi PCA 2D Klaster Divisive Manhattan (K = {TARGET_CLUSTERS})\\nSil: {sil:.4f} | DBI: {db:.4f} | CH: {ch:.1f}', fontsize=11, fontweight='bold')
plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0]*100:.2f}%)')
plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1]*100:.2f}%)')
plt.grid(True, linestyle=':', alpha=0.6)
plt.tight_layout()

pca_path = os.path.join(OUTPUT_DIR, "cluster_manhattan.png")
plt.savefig(pca_path, dpi=300)
print(f"[SAVE] Grafik disimpan ke: '{pca_path}'")
plt.show()"""))

    nb['cells'] = cells
    return nb

# =============================================================================
# BUILDER PROYEK 3: DIVISIVE COSINE
# =============================================================================
def build_cosine_notebook():
    nb = nbf.v4.new_notebook()
    nb.metadata = {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.13"}
    }
    cells = []

    cells.append(nbf.v4.new_markdown_cell(r"""# Proyek 3: Divisive Hierarchical Clustering dengan Cosine Similarity pada BISINDO

**Deskripsi:**
Notebook ini mengimplementasikan Spherical Divisive Hierarchical Clustering (Spherical Bisecting 2-Means) menggunakan Cosine Similarity pada dataset 127 fitur gestur alfabet BISINDO.

**Formulasi Cosine:**
$$d_{\text{Cosine}}(x, y) = 1 - \frac{x \cdot y}{\|x\|_2 \|y\|_2}$$

Setiap vektor dinormalisasi ke unit sphere $L_2$. Pembelahan memaksimalkan cosine alignment terhadap centroid arah, menghasilkan partisi gestur yang kebal terhadap variasi jarak fisik tangan ke kamera."""))

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

from sklearn.cluster import BisectingKMeans
from sklearn.preprocessing import Normalizer, StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
from scipy.cluster.hierarchy import dendrogram

warnings.filterwarnings('ignore')

plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
plt.rcParams['font.sans-serif'] = 'DejaVu Sans'
plt.rcParams['figure.dpi'] = 120

BASE_DIR = os.path.dirname(os.path.abspath(__file__)) if '__file__' in locals() else os.getcwd()
OUTPUT_DIR = os.path.join(BASE_DIR, "outputDivisive")
os.makedirs(OUTPUT_DIR, exist_ok=True)

TARGET_CLUSTERS = 26
RANDOM_STATE = 42

print(f"[INFO] Direktori: {BASE_DIR} | Target Klaster = {TARGET_CLUSTERS}")"""))

    cells.append(nbf.v4.new_code_cell(CODE_LOAD_DATA + """

normalizer = Normalizer(norm='l2')
X_norm = normalizer.fit_transform(X)
print("[INFO] Fitur berhasil dinormalisasi ke unit hypersphere (L2 Norm = 1).")"""))

    cells.append(nbf.v4.new_code_cell(CODE_LINKAGE_HELPER + """

# =======================================================================
# 2. EKSEKUSI SPHERICAL BISECTING 2-MEANS (COSINE)
# =======================================================================
t0 = time.perf_counter()
bkm = BisectingKMeans(
    n_clusters=TARGET_CLUSTERS,
    random_state=RANDOM_STATE,
    bisecting_strategy='biggest_inertia'
)
labels = bkm.fit_predict(X_norm)
runtime_cosine = time.perf_counter() - t0

cos_leaves = list(bkm._bisecting_tree.iter_leaves())
Z = build_linkage_matrix_from_tree(bkm._bisecting_tree, cos_leaves, distance_metric='cosine')

plt.figure(figsize=(10, 5.5))
ddata = dendrogram(
    Z,
    labels=[f"C{i+1}" for i in range(TARGET_CLUSTERS)],
    leaf_rotation=90,
    leaf_font_size=8,
    above_threshold_color='#15803D',
    color_threshold=0
)
plt.title(f'Pohon Divisive: Spherical Bisecting 2-Means Cosine (K = {TARGET_CLUSTERS})', fontsize=12, fontweight='bold', pad=10)
plt.xlabel('Klaster Terminal (C1 - C26)', fontsize=10)
plt.ylabel('Disimilaritas Pembelahan (Angular Distance)', fontsize=10)
plt.grid(True, linestyle=':', alpha=0.6)
plt.tight_layout()

dendro_path = os.path.join(OUTPUT_DIR, "dendrogram_cosine.png")
plt.savefig(dendro_path, dpi=300)
print(f"[SAVE] Grafik disimpan ke: '{dendro_path}'")
plt.show()"""))

    cells.append(nbf.v4.new_code_cell("""# =======================================================================
# 3. EVALUASI METRIK KUANTITATIF COSINE
# =======================================================================
sil = float(silhouette_score(X_norm, labels, metric='cosine'))
db = float(davies_bouldin_score(X_norm, labels))
ch = float(calinski_harabasz_score(X_norm, labels))
counts = pd.Series(labels).value_counts()

df_res = pd.DataFrame([{
    'Distance Metric': 'Cosine Distance (Angular)',
    'Divisive Strategy': 'Spherical Bisecting 2-Means',
    'Clusters (K)': TARGET_CLUSTERS,
    'Min Cluster Size': int(counts.min()),
    'Max Cluster Size': int(counts.max()),
    'Silhouette Score (↑)': round(sil, 4),
    'Davies–Bouldin Index (↓)': round(db, 4),
    'Calinski–Harabasz Index (↑)': round(ch, 2),
    'Execution Time (s) (↓)': round(runtime_cosine, 4)
}])

print("\\n" + "="*85)
print(df_res.to_string(index=False))
print("="*85)

excel_out = os.path.join(OUTPUT_DIR, "evaluasi_divisive_cosine.xlsx")
df_data_out = df_meta.copy()
df_data_out['cluster_cosine'] = labels

with pd.ExcelWriter(excel_out, engine='openpyxl') as writer:
    df_res.to_excel(writer, sheet_name='Ringkasan_Metrik', index=False)
    counts.to_frame(name='Jumlah_Sampel').to_excel(writer, sheet_name='Distribusi_Ukuran_Klaster')
    df_data_out.head(1000).to_excel(writer, sheet_name='Data_Klaster_Sampel', index=False)

print(f"[SAVE] Berkas evaluasi Excel disimpan ke: '{excel_out}'")"""))

    cells.append(nbf.v4.new_code_cell("""# =======================================================================
# 4. VISUALISASI SEBARAN KLASTER PCA 2D
# =======================================================================
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
pca = PCA(n_components=2, random_state=RANDOM_STATE)
X_pca = pca.fit_transform(X_scaled)

plt.figure(figsize=(8.5, 6))
plt.scatter(
    X_pca[:, 0], X_pca[:, 1],
    c=labels,
    cmap='tab20',
    s=12,
    alpha=0.6,
    edgecolors='none'
)
plt.title(f'Proyeksi PCA 2D Klaster Divisive Cosine (K = {TARGET_CLUSTERS})\\nSil: {sil:.4f} | DBI: {db:.4f} | CH: {ch:.1f}', fontsize=11, fontweight='bold')
plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0]*100:.2f}%)')
plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1]*100:.2f}%)')
plt.grid(True, linestyle=':', alpha=0.6)
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
        ("divisive_euclidean.ipynb", build_euclidean_notebook),
        ("divisive_manhattan.ipynb", build_manhattan_notebook),
        ("divisive_cosine.ipynb", build_cosine_notebook)
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
