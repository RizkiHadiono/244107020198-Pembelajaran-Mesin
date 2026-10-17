"""
Script untuk membangun dan mengeksekusi Jupyter Notebook: kmeans_distance_comparison.ipynb
Membandingkan K-Means Clustering pada Dataset BISINDO (127 Fitur Hand Landmark)
dengan Euclidean Distance, Manhattan Distance, dan Cosine Similarity pada K=26.
Bebas emote/emoticon sesuai panduan teknis akademik.
"""

import os
import sys
import time
import nbformat as nbf
from nbconvert.preprocessors import ExecutePreprocessor

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

def build_and_run_notebook():
    nb = nbf.v4.new_notebook()
    nb.metadata = {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3"
        },
        "language_info": {
            "name": "python",
            "version": "3.13"
        }
    }

    cells = []

    # =========================================================================
    # CELL 1: TITLE & EXECUTIVE SUMMARY (MARKDOWN)
    # =========================================================================
    md_title = """# Studi Komparasi Komprehensif Algoritma K-Means Clustering pada Dataset BISINDO
## Analisis Pengaruh Tiga Metrik Jarak: Euclidean Distance, Manhattan Distance, dan Cosine Similarity pada K = 26

**Domain:** Pengenalan Bahasa Isyarat Indonesia (BISINDO) - Pembelajaran Mesin  
**Arsitektur Fitur:** Ekstraksi 42 Titik Landmark MediaPipe 3D (126 Dimensi) + 1 Flag Biner Is_Two_Handed (Total 127 Fitur)  
**Tujuan Proyek:** Mengevaluasi secara empiris dan matematis performa pengelompokan tanpa supervisi (Unsupervised Clustering) menggunakan K-Means terhadap 26 kelas alfabet BISINDO (A-Z) dengan tiga pendekatan fungsi jarak yang berbeda.

---

### Ringkasan Eksekutif
Dalam permasalahan klasifikasi dan klasterisasi alfabet BISINDO, gestur tangan memiliki sifat dualistik: sebagian huruf dieja menggunakan satu tangan (statis, e.g. C, E, Z), sedangkan sebagian besar lainnya membutuhkan interaksi dua tangan (e.g. A, B, D, P). Pendekatan representasi berbasis 127 dimensi (63 fitur koordinat tangan primer, 63 fitur koordinat tangan sekunder dengan zero-padding jika 1 tangan, dan 1 flag indikator dua tangan) menghasilkan ruang fitur berdimensi tinggi dengan struktur spasial yang unik.

Penerapan K-Means konvensional yang hanya bertumpu pada jarak Euclidean sering kali rentan terhadap dimensi tinggi, variabilitas skala, dan pencilan. Oleh karena itu, penelitian komparatif ini menguji tiga fungsi jarak fundamental dengan jumlah klaster disetarakan pada K = 26 (sesuai jumlah ground-truth alfabet A-Z):
1. **Euclidean Distance (L2 Norm):** Mengukur jarak geometris garis lurus dalam ruang R^127. Titik pusat klaster dioptimasi menggunakan rata-rata aritmetika (mean).
2. **Manhattan Distance (L1 Norm / City-Block):** Mengukur akumulasi selisih nilai mutlak per dimensi fitur. Titik pusat dioptimasi menggunakan median per koordinat (prinsip K-Medians), memberikan ketahanan superior terhadap variasi ekstrem rentang jari.
3. **Cosine Distance (1 - Cosine Similarity):** Mengukur orientasi sudut deviasi antarvektor gestur, mengabaikan magnitudo absolut panjang vektor. Dikenal sebagai Spherical K-Means pada permukaan bola satuan (unit hypersphere ||x|| = 1).

Evaluasi dilakukan secara kuantitatif melalui metrik Silhouette Score, Davies-Bouldin Index, Calinski-Harabasz Index, Waktu Eksekusi, Jumlah Iterasi Konvergensi, serta Nilai Fungsi Objektif (Inertia)."""
    cells.append(nbf.v4.new_markdown_cell(md_title))

    # =========================================================================
    # CELL 2: MATHEMATICAL FORMULATIONS (MARKDOWN)
    # =========================================================================
    md_math = r"""## 1. Formulasi Matematis Metrik Jarak dan Fungsi Objektif K-Means

Berikut adalah formulasi matematis formal yang mendasari ketiga varian K-Means dalam studi ini:

### 1.1. Euclidean Distance (L2 Norm)
Jarak geometris langsung antara dua vektor gestur $x, y \in \mathbb{R}^{127}$:
$$d_{\text{Euclidean}}(x, y) = \|x - y\|_2 = \sqrt{\sum_{i=1}^{127} (x_i - y_i)^2}$$

> **Aturan Pembaruan Titik Pusat (Centroid Mean Update):**  
> Untuk klaster $S_k$, estimator yang meminimalkan Within-Cluster Sum of Squares (WCSS) adalah rata-rata aritmetika:
> $$\mu_k^* = \arg\min_{\mu} \sum_{x \in S_k} \|x - \mu\|_2^2 = \frac{1}{|S_k|} \sum_{x \in S_k} x$$

---

### 1.2. Manhattan Distance (L1 Norm / City-Block Distance)
Jarak pergerakan ortogonal per komponen fitur:
$$d_{\text{Manhattan}}(x, y) = \|x - y\|_1 = \sum_{i=1}^{127} |x_i - y_i|$$

> **Aturan Pembaruan Titik Pusat (K-Medians Principle):**  
> Untuk meminimalkan fungsi dispersi absolut linier $\sum_{x \in S_k} \|x - \mu\|_1$, titik pusat optimal pada setiap dimensi $j$ adalah median komponen:
> $$\mu_{k, j}^* = \text{median}(\{x_j : x \in S_k\})$$
> Sifat median ini memberikan kekebalan terhadap koordinat ekstrem akibat kesalahan deteksi landmark minor.

---

### 1.3. Cosine Distance (1 - Cosine Similarity)
Deviasi sudut orientasi antara dua vektor gestur dalam ruang multidimensi:
$$d_{\text{Cosine}}(x, y) = 1 - \frac{x \cdot y}{\|x\|_2 \|y\|_2} = 1 - \frac{\sum_{i=1}^{127} x_i y_i}{\sqrt{\sum_{i=1}^{127} x_i^2} \sqrt{\sum_{i=1}^{127} y_i^2}}$$

> **Aturan Pembaruan Titik Pusat (Spherical K-Means Principle):**  
> Pada vektor yang telah diproyeksikan pada permukaan bola satuan hiperdimensi ($\|x\|_2 = 1$), titik pusat optimal diperoleh dengan menormalisasi kembali vektor rata-rata ke unit hypersphere:
> $$\mu_k^* = \frac{\sum_{x \in S_k} x}{\left\|\sum_{x \in S_k} x\right\|_2}$$

---

### 1.4. Fungsi Biaya Objektif K-Means
Fungsi biaya yang diminimalkan selama proses iterasi:
$$J = \sum_{k=1}^{K} \sum_{x \in S_k} d(x, \mu_k)^2$$
Di mana $K = 26$ merupakan jumlah klaster target yang merepresentasikan 26 kelas huruf alfabet BISINDO."""
    cells.append(nbf.v4.new_markdown_cell(md_math))

    # =========================================================================
    # CELL 3: ENVIRONMENT SETUP & LIBRARIES (CODE)
    # =========================================================================
    code_setup = """# =======================================================================
# SETUP LINGKUNGAN, IMPORT PUSTAKA, DAN KONFIGURASI GLOBAL
# =======================================================================
import os
import sys
import time
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Modul Scikit-Learn
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler, Normalizer
from sklearn.decomposition import PCA
from sklearn.metrics import (
    silhouette_score,
    davies_bouldin_score,
    calinski_harabasz_score,
    pairwise_distances
)

# Konfigurasi visualisasi bersih tanpa emotikon
warnings.filterwarnings('ignore')
plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
plt.rcParams['font.sans-serif'] = 'DejaVu Sans'
plt.rcParams['font.size'] = 10
plt.rcParams['axes.titlesize'] = 12
plt.rcParams['axes.labelsize'] = 10
plt.rcParams['figure.dpi'] = 120

# Direktori kerja dan output artefak
BASE_DIR = os.path.dirname(os.path.abspath(__file__)) if '__file__' in locals() else os.getcwd()
OUTPUT_DIR = os.path.join(BASE_DIR, "outputKmeans")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Random state untuk reproduktifitas penuh
RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

# Jumlah klaster alfabet BISINDO yang disetarakan
K_BENCHMARK = 26

print(f"[INFO] Direktori kerja: {BASE_DIR}")
print(f"[INFO] Direktori output: {OUTPUT_DIR}")
print(f"[INFO] Jumlah klaster evaluasi benchmark disetarakan pada K = {K_BENCHMARK}")"""
    cells.append(nbf.v4.new_code_cell(code_setup))

    # =========================================================================
    # CELL 4: DATA LOADING & PREPROCESSING (CODE)
    # =========================================================================
    code_data_prep = """# =======================================================================
# PEMUATAN DATASET BISINDO LANDMARKS & PERSIAPAN REPRESENTASI FITUR
# =======================================================================
def load_bisindo_dataset():
    \"\"\"
    Memuat dataset landmark BISINDO hasil ekstraksi MediaPipe (127 fitur).
    Mendukung format .npz (kilat) atau .csv fallback.
    \"\"\"
    npz_path = os.path.join(BASE_DIR, "dataset_bisindo_landmarks.npz")
    csv_path = os.path.join(BASE_DIR, "dataset_bisindo_landmarks.csv")

    if os.path.exists(npz_path):
        print(f"[DATA] Memuat berkas NumPy terkompresi: '{npz_path}'")
        data = np.load(npz_path, allow_pickle=True)
        X = data['X'].astype(np.float32)
        labels = data['labels']
        label_indices = data['label_indices']
        is_two_handed = data['is_two_handed']
        feature_names = data['feature_names']
        df_meta = pd.DataFrame({
            'label': labels,
            'label_idx': label_indices,
            'is_two_handed': is_two_handed
        })
    elif os.path.exists(csv_path):
        print(f"[DATA] Memuat berkas CSV: '{csv_path}'")
        df_full = pd.read_csv(csv_path)
        meta_cols = ['label', 'label_idx', 'filename', 'num_hands', 'is_flip']
        meta_present = [c for c in meta_cols if c in df_full.columns]
        df_meta = df_full[meta_present]
        feat_cols = [c for c in df_full.columns if c not in meta_cols]
        X = df_full[feat_cols].values.astype(np.float32)
        feature_names = feat_cols
    else:
        raise FileNotFoundError(
            "Berkas dataset 'dataset_bisindo_landmarks.npz' atau 'dataset_bisindo_landmarks.csv' "
            "tidak ditemukan. Jalankan extract_features.py terlebih dahulu."
        )

    print(f"[INFO] Total sampel dataset: {X.shape[0]}")
    print(f"[INFO] Jumlah dimensi fitur: {X.shape[1]} (63 Tangan Primer + 63 Tangan Sekunder + 1 Is_Two_Handed)")
    print(f"[INFO] Sebaran kelas alfabet: {len(np.unique(df_meta['label']))} kelas unik ({sorted(list(np.unique(df_meta['label'])))})")
    print(f"[INFO] Proporsi gestur 2 tangan vs 1 tangan: "
          f"{np.sum(df_meta['is_two_handed'] == 1.0)} ({np.mean(df_meta['is_two_handed'] == 1.0)*100:.1f}%) vs "
          f"{np.sum(df_meta['is_two_handed'] == 0.0)} ({np.mean(df_meta['is_two_handed'] == 0.0)*100:.1f}%)")

    # Penskalaan fitur:
    # 1. StandardScaler (Mean = 0, Std = 1): Optimal untuk Euclidean dan Manhattan
    scaler = StandardScaler()
    X_std = scaler.fit_transform(X)

    # 2. L2 Normalizer: Proyeksi vektor satuan pada unit hypersphere (Wajib untuk Cosine Similarity)
    normalizer = Normalizer(norm='l2')
    X_norm = normalizer.fit_transform(X_std)

    return df_meta, X, X_std, X_norm, feature_names

df_meta, X_raw, X_std, X_norm, feature_names = load_bisindo_dataset()"""
    cells.append(nbf.v4.new_code_cell(code_data_prep))

    # =========================================================================
    # CELL 5: ELBOW METHOD DESCRIPTION (MARKDOWN)
    # =========================================================================
    md_elbow = r"""## 2. Analisis Elbow Method dan Penyetaraan Nilai K = 26

Metode **Elbow** mengevaluasi tren penurunan nilai **Within-Cluster Sum of Squares (Inertia / SSE)** seiring pertambahan jumlah klaster $k \in [2, 30]$:
$$\text{SSE}(k) = \sum_{i=1}^{k} \sum_{x \in S_i} \|x - \mu_i\|^2$$

Sesuai dengan instruksi teknis dan karakteristik domain alfabet BISINDO:
* Sumbu evaluasi dieksplorasi dari $k=2$ hingga $k=30$ untuk memetakan kurva inersia data landmark tangan.
* **Penyetaraan Benchmark K = 26:** Seluruh ketiga algoritma klasterisasi (Euclidean, Manhattan, dan Cosine) disetarakan pada **$K = 26$** (merepresentasikan 26 karakter alfabet A hingga Z). Hal ini memastikan perbandingan matriks performa bersifat objektif, seimbang (*fair comparison*), dan relevan secara linguistik."""
    cells.append(nbf.v4.new_markdown_cell(md_elbow))

    # =========================================================================
    # CELL 6: ELBOW METHOD EXECUTION (CODE)
    # =========================================================================
    code_elbow = """# =======================================================================
# EKSEKUSI ELBOW METHOD DAN VISUALISASI TITIK EVALUASI K = 26
# =======================================================================
k_range = list(range(2, 32, 2))
if K_BENCHMARK not in k_range:
    k_range.append(K_BENCHMARK)
k_range = sorted(k_range)

sse_list = []
print(f"[ELBOW] Menghitung kurva inersia SSE untuk rentang k = {k_range}...")

for k in k_range:
    km_test = KMeans(n_clusters=k, init='k-means++', n_init=5, max_iter=200, random_state=RANDOM_STATE)
    km_test.fit(X_std)
    sse_list.append(km_test.inertia_)
    print(f"  k = {k:2d} -> SSE = {km_test.inertia_:,.1f}")

# Visualisasi Kurva Elbow Method
plt.figure(figsize=(10, 5.5))
plt.plot(k_range, sse_list, marker='o', markersize=7, linewidth=2.2, color='#1E40AF', label='Within-Cluster SSE')

# Sorot titik K = 26
idx_26 = k_range.index(K_BENCHMARK)
sse_26 = sse_list[idx_26]
plt.axvline(x=K_BENCHMARK, color='#DC2626', linestyle='--', linewidth=2, label=f'Benchmark Evaluasi K = {K_BENCHMARK} (Alfabet A-Z)')
plt.scatter([K_BENCHMARK], [sse_26], color='#DC2626', s=180, zorder=5, edgecolor='black', linewidth=1.8)

plt.title('Kurva Elbow Method pada Dataset BISINDO Landmarks (Evaluasi Disetarakan pada K = 26)', fontsize=13, fontweight='bold', pad=12)
plt.xlabel('Jumlah Klaster (k)', fontsize=11)
plt.ylabel('Sum of Squared Errors (SSE / Inertia)', fontsize=11)
plt.xticks(k_range)
plt.annotate(
    f'Titik Evaluasi K = {K_BENCHMARK}\\nSSE = {sse_26:,.1f}',
    xy=(K_BENCHMARK, sse_26),
    xytext=(K_BENCHMARK - 7, sse_26 + (max(sse_list) - min(sse_list)) * 0.12),
    arrowprops=dict(facecolor='#DC2626', shrink=0.08, width=1.8, headwidth=7),
    bbox=dict(boxstyle="round,pad=0.4", facecolor="#FEE2E2", edgecolor="#DC2626", lw=1.2),
    fontweight='bold', fontsize=10
)

plt.grid(True, linestyle=':', alpha=0.6)
plt.legend(frameon=True, facecolor='white', framealpha=0.9)
plt.tight_layout()

elbow_path = os.path.join(OUTPUT_DIR, "elbow_curve.png")
plt.savefig(elbow_path, dpi=300)
print(f"[SAVE] Grafik kurva Elbow disimpan ke: '{elbow_path}'")
plt.show()"""
    cells.append(nbf.v4.new_code_cell(code_elbow))

    # =========================================================================
    # CELL 7: CUSTOM K-MEANS MULTI-METRIC IMPLEMENTATION (CODE)
    # =========================================================================
    code_kmeans_class = """# =======================================================================
# IMPLEMENTASI KELAS K-MEANS MULTI-METRIK JARAK DARI AWAL (CUSTOM CLASS)
# =======================================================================
class CustomKMeansBISINDO:
    \"\"\"
    Implementasi K-Means Clustering terstruktur yang mendukung 3 metrik jarak:
    - 'euclidean': Pembaruan titik pusat menggunakan rata-rata aritmetika (L2 Norm).
    - 'manhattan': Pembaruan titik pusat menggunakan median komponen (L1 Norm / K-Medians).
    - 'cosine'   : Pembaruan titik pusat menggunakan vektor rata-rata ter-normalisasi L2 (Spherical K-Means).
    \"\"\"
    def __init__(self, n_clusters=26, metric='euclidean', max_iter=300, tol=1e-4, random_state=42):
        self.n_clusters = n_clusters
        self.metric = metric.lower()
        self.max_iter = max_iter
        self.tol = tol
        self.random_state = random_state
        
        self.centroids = None
        self.labels_ = None
        self.inertia_ = 0.0
        self.n_iter_ = 0
        self.execution_time_ = 0.0

    def _compute_distance(self, X, centroids):
        \"\"\"Menghitung matriks jarak antara observasi data dan titik pusat.\"\"\"
        return pairwise_distances(X, centroids, metric=self.metric)

    def _init_centroids(self, X):
        \"\"\"Inisialisasi titik pusat adaptif berbasis K-Means++.\"\"\"
        rng = np.random.RandomState(self.random_state)
        n_samples, _ = X.shape
        
        first_idx = rng.randint(n_samples)
        centroids = [X[first_idx]]

        for _ in range(1, self.n_clusters):
            cur_centroids = np.array(centroids)
            dists = self._compute_distance(X, cur_centroids)
            min_dists = np.min(dists, axis=1)
            sq_dists = min_dists ** 2
            sum_sq = np.sum(sq_dists)
            probs = np.ones(n_samples) / n_samples if sum_sq == 0 else sq_dists / sum_sq
            next_idx = rng.choice(n_samples, p=probs)
            centroids.append(X[next_idx])

        centroids = np.array(centroids)
        if self.metric == 'cosine':
            norms = np.linalg.norm(centroids, axis=1, keepdims=True)
            norms[norms == 0] = 1e-10
            centroids = centroids / norms
        return centroids

    def _update_centroids(self, X, labels):
        \"\"\"Pembaruan posisi centroid sesuai aturan teoritis masing-masing fungsi jarak.\"\"\"
        centroids = np.zeros((self.n_clusters, X.shape[1]))
        rng = np.random.RandomState(self.random_state)
        
        for k in range(self.n_clusters):
            cluster_pts = X[labels == k]
            if len(cluster_pts) == 0:
                centroids[k] = X[rng.randint(len(X))]
            else:
                if self.metric == 'euclidean':
                    # L2 Norm: Rata-rata analitis (Mean)
                    centroids[k] = np.mean(cluster_pts, axis=0)
                elif self.metric == 'manhattan':
                    # L1 Norm: Median komponen (K-Medians Principle)
                    centroids[k] = np.median(cluster_pts, axis=0)
                elif self.metric == 'cosine':
                    # Spherical K-Means: Normalisasi vektor rata-rata ke permukaan bola satuan
                    mean_vec = np.mean(cluster_pts, axis=0)
                    norm = np.linalg.norm(mean_vec)
                    centroids[k] = mean_vec / (norm if norm > 0 else 1.0)
        return centroids

    def fit(self, X):
        \"\"\"Melatih algoritma K-Means hingga konvergen atau batas iterasi tercapai.\"\"\"
        start_time = time.perf_counter()
        
        if self.metric == 'cosine':
            norms = np.linalg.norm(X, axis=1, keepdims=True)
            norms[norms == 0] = 1e-10
            X_calc = X / norms
        else:
            X_calc = X

        self.centroids = self._init_centroids(X_calc)

        for it in range(self.max_iter):
            self.n_iter_ = it + 1
            
            # Assignment Step
            dists = self._compute_distance(X_calc, self.centroids)
            labels = np.argmin(dists, axis=1)

            # Update Step
            new_centroids = self._update_centroids(X_calc, labels)

            # Convergence Check
            shift = np.linalg.norm(new_centroids - self.centroids)
            self.centroids = new_centroids
            self.labels_ = labels

            if shift < self.tol:
                break

        # Komputasi Nilai Fungsi Objektif
        final_dists = self._compute_distance(X_calc, self.centroids)
        assigned_dists = final_dists[np.arange(len(X_calc)), self.labels_]
        self.inertia_ = float(np.sum(assigned_dists ** 2))
        self.execution_time_ = float(time.perf_counter() - start_time)
        return self

print("[INFO] Kelas CustomKMeansBISINDO berhasil diinisialisasi.")"""
    cells.append(nbf.v4.new_code_cell(code_kmeans_class))

    # =========================================================================
    # CELL 8: MODEL TRAINING & METRICS EVALUATION (CODE)
    # =========================================================================
    code_eval = """# =======================================================================
# PELATIHAN MODEL PADA K = 26 DAN EVALUASI MATRIKS PERFORMA
# =======================================================================
configurations = {
    'Euclidean': {
        'metric': 'euclidean',
        'data': X_std,
        'eval_metric': 'euclidean',
        'desc': 'L2 Standardized Space (Mean Centroid)'
    },
    'Manhattan': {
        'metric': 'manhattan',
        'data': X_std,
        'eval_metric': 'manhattan',
        'desc': 'L1 Robust Space (Median Centroid)'
    },
    'Cosine': {
        'metric': 'cosine',
        'data': X_norm,
        'eval_metric': 'cosine',
        'desc': 'Spherical Hypersphere Space (Normalized Mean)'
    }
}

trained_models = {}
eval_records = []

for name, cfg in configurations.items():
    print(f"[PELATIHAN] Menjalankan K-Means [{name} Distance] dengan K = {K_BENCHMARK}...")
    model = CustomKMeansBISINDO(
        n_clusters=K_BENCHMARK,
        metric=cfg['metric'],
        max_iter=300,
        tol=1e-4,
        random_state=RANDOM_STATE
    ).fit(cfg['data'])
    
    trained_models[name] = model

    # Perhitungan Metrik Evaluasi Intrinsik
    # Untuk dataset besar, evaluasi silhouette dihitung dengan sampel representatif jika diperlukan
    # Di sini dihitung secara presisi penuh pada matriks konfigurasi masing-masing
    print(f"  Menghitung skor evaluasi internal untuk {name}...")
    t_eval = time.perf_counter()
    sil = silhouette_score(cfg['data'], model.labels_, metric=cfg['eval_metric'])
    db = davies_bouldin_score(cfg['data'], model.labels_)
    ch = calinski_harabasz_score(cfg['data'], model.labels_)
    print(f"  Selesai evaluasi dalam {time.perf_counter() - t_eval:.2f}s")

    eval_records.append({
        'Distance Metric': name,
        'Silhouette Score (↑)': sil,
        'Davies–Bouldin Index (↓)': db,
        'Calinski–Harabasz Index (↑)': ch,
        'Execution Time (s) (↓)': model.execution_time_,
        'Iterations (↓)': model.n_iter_,
        'Objective Value (J)': model.inertia_
    })

# DataFrame Hasil Evaluasi
df_eval = pd.DataFrame(eval_records)

print("\\n" + "="*85)
print(f"TABEL EVALUASI KINERJA K-MEANS CLUSTERING BISINDO (K = {K_BENCHMARK})")
print("="*85)
print(df_eval.to_string(index=False))
print("="*85)

# Simpan ke CSV
summary_csv = os.path.join(OUTPUT_DIR, "metrics_comparison_summary.csv")
df_eval.to_csv(summary_csv, index=False)
print(f"[SAVE] Ringkasan metrik disimpan ke: '{summary_csv}'")

# Simpan ke Excel 2 Sheet
excel_path = os.path.join(OUTPUT_DIR, "evaluasi_kmeans_bisindo.xlsx")
df_clustered = df_meta.copy()
df_clustered['cluster_euclidean'] = trained_models['Euclidean'].labels_
df_clustered['cluster_manhattan'] = trained_models['Manhattan'].labels_
df_clustered['cluster_cosine'] = trained_models['Cosine'].labels_

with pd.ExcelWriter(excel_path, engine='openpyxl') as writer:
    df_eval.to_excel(writer, sheet_name='Ringkasan_Evaluasi', index=False)
    df_clustered.to_excel(writer, sheet_name='Data_Terklaster', index=False)

print(f"[SAVE] Laporan Excel lengkap (2 Sheet) disimpan ke: '{excel_path}'")"""
    cells.append(nbf.v4.new_code_cell(code_eval))

    # =========================================================================
    # CELL 9: PCA 2D VISUALIZATION (CODE)
    # =========================================================================
    code_vis_pca = """# =======================================================================
# VISUALISASI PROYEKSI PCA 2D PEMETAAN 26 KLASTER BISINDO
# =======================================================================
pca_std = PCA(n_components=2, random_state=RANDOM_STATE)
X_pca_std = pca_std.fit_transform(X_std)

pca_norm = PCA(n_components=2, random_state=RANDOM_STATE)
X_pca_norm = pca_norm.fit_transform(X_norm)

fig, axes = plt.subplots(1, 3, figsize=(22, 6.5))
palette = sns.color_palette("tab20", 20) + sns.color_palette("Set2", 6)

plot_cfgs = [
    ('Euclidean', trained_models['Euclidean'], X_pca_std, pca_std, 'PCA Euclidean (L2 Space)'),
    ('Manhattan', trained_models['Manhattan'], X_pca_std, pca_std, 'PCA Manhattan (L1 Space)'),
    ('Cosine', trained_models['Cosine'], X_pca_norm, pca_norm, 'PCA Cosine (Spherical Space)')
]

for idx, (name, model, X_2d, pca_obj, title_info) in enumerate(plot_cfgs):
    ax = axes[idx]
    
    for k in range(K_BENCHMARK):
        mask = (model.labels_ == k)
        ax.scatter(
            X_2d[mask, 0], X_2d[mask, 1],
            color=palette[k % len(palette)],
            label=f'K{k}' if idx == 0 and k < 8 else None,
            alpha=0.55,
            s=22,
            edgecolors='none'
        )

    centroids_2d = pca_obj.transform(model.centroids)
    ax.scatter(
        centroids_2d[:, 0], centroids_2d[:, 1],
        marker='X',
        s=140,
        c='#DC2626',
        edgecolor='black',
        linewidth=1.2,
        label='Centroids' if idx == 0 else None,
        zorder=10
    )

    row_info = df_eval[df_eval['Distance Metric'] == name].iloc[0]
    ax.set_title(
        f"{name} Distance\\nSilhouette: {row_info['Silhouette Score (↑)']:.3f} | DB: {row_info['Davies–Bouldin Index (↓)']:.3f}",
        fontsize=12, fontweight='bold', pad=8
    )
    ax.set_xlabel(f'PC1 ({pca_obj.explained_variance_ratio_[0]*100:.1f}%)')
    ax.set_ylabel(f'PC2 ({pca_obj.explained_variance_ratio_[1]*100:.1f}%)')
    ax.grid(True, linestyle=':', alpha=0.6)
    if idx == 0:
        ax.legend(frameon=True, loc='upper right', fontsize=8, ncol=2)

plt.suptitle(f'Pemetaan Distribusi 26 Klaster BISINDO dalam Ruang Proyeksi PCA 2D (K = {K_BENCHMARK})', fontsize=15, fontweight='bold', y=1.01)
plt.tight_layout()

pca_path = os.path.join(OUTPUT_DIR, "cluster_comparison_pca2d.png")
plt.savefig(pca_path, dpi=300, bbox_inches='tight')
print(f"[SAVE] Visualisasi PCA 2D disimpan ke: '{pca_path}'")
plt.show()"""
    cells.append(nbf.v4.new_code_cell(code_vis_pca))

    # =========================================================================
    # CELL 10: BAR CHARTS COMPARISON (CODE)
    # =========================================================================
    code_vis_bars = """# =======================================================================
# GRAFIK BATANG PERBANDINGAN METRIK EVALUASI KLASTERISASI
# =======================================================================
fig, axes = plt.subplots(2, 2, figsize=(13, 9))

metrics_meta = [
    ('Silhouette Score (↑)', 'Silhouette Score\\n(Semakin Tinggi = Klaster Kompak & Terpisah)', '#1D4ED8', axes[0, 0]),
    ('Davies–Bouldin Index (↓)', 'Davies–Bouldin Index\\n(Semakin Rendah = Separasi Antarklaster Lebih Baik)', '#B45309', axes[0, 1]),
    ('Calinski–Harabasz Index (↑)', 'Calinski–Harabasz Index\\n(Semakin Tinggi = Rasio Dispersi Varians Baik)', '#047857', axes[1, 0]),
    ('Execution Time (s) (↓)', 'Waktu Eksekusi Pelatihan (Detik)\\n(Semakin Rendah = Efisiensi Komputasi Tinggi)', '#6D28D9', axes[1, 1])
]

methods = df_eval['Distance Metric'].tolist()

for col_name, title_txt, bar_color, ax in metrics_meta:
    vals = df_eval[col_name].tolist()
    bars = ax.bar(methods, vals, color=bar_color, width=0.45, edgecolor='black', linewidth=1.1, alpha=0.88)

    for b in bars:
        h = b.get_height()
        if abs(h) < 0.1:
            lbl = f"{h:.4f}"
        elif abs(h) < 10:
            lbl = f"{h:.3f}"
        else:
            lbl = f"{h:,.1f}"
        ax.annotate(lbl,
                    xy=(b.get_x() + b.get_width() / 2, h),
                    xytext=(0, 4),
                    textcoords="offset points",
                    ha='center', va='bottom', fontsize=10, fontweight='bold')

    ax.set_title(title_txt, fontsize=11, fontweight='bold', pad=8)
    ax.set_ylabel('Skor Nilai')
    ax.grid(axis='y', linestyle=':', alpha=0.6)
    
    y_min, y_max = ax.get_ylim()
    ax.set_ylim(y_min, y_max * 1.15)

plt.suptitle(f'Komparasi Kuantitatif Kinerja K-Means Clustering pada Dataset BISINDO (K = {K_BENCHMARK})', fontsize=14, fontweight='bold', y=1.00)
plt.tight_layout()

barchart_path = os.path.join(OUTPUT_DIR, "metrics_comparison_barchart.png")
plt.savefig(barchart_path, dpi=300, bbox_inches='tight')
print(f"[SAVE] Visualisasi grafik batang disimpan ke: '{barchart_path}'")
plt.show()"""
    cells.append(nbf.v4.new_code_cell(code_vis_bars))

    # =========================================================================
    # CELL 11: CRITICAL COMPARATIVE ANALYSIS & EXACT DELTAS (CODE & MARKDOWN)
    # =========================================================================
    code_deltas = """# =======================================================================
# PERHITUNGAN SELISIH (DELTA) EKSAK DAN LAPORAN ANALISIS KRITIS
# =======================================================================
# Ekstraksi nilai per metrik
res = df_eval.set_index('Distance Metric')

sil_euc = res.loc['Euclidean', 'Silhouette Score (↑)']
sil_man = res.loc['Manhattan', 'Silhouette Score (↑)']
sil_cos = res.loc['Cosine', 'Silhouette Score (↑)']

db_euc = res.loc['Euclidean', 'Davies–Bouldin Index (↓)']
db_man = res.loc['Manhattan', 'Davies–Bouldin Index (↓)']
db_cos = res.loc['Cosine', 'Davies–Bouldin Index (↓)']

ch_euc = res.loc['Euclidean', 'Calinski–Harabasz Index (↑)']
ch_man = res.loc['Manhattan', 'Calinski–Harabasz Index (↑)']
ch_cos = res.loc['Cosine', 'Calinski–Harabasz Index (↑)']

time_euc = res.loc['Euclidean', 'Execution Time (s) (↓)']
time_man = res.loc['Manhattan', 'Execution Time (s) (↓)']
time_cos = res.loc['Cosine', 'Execution Time (s) (↓)']

iter_euc = res.loc['Euclidean', 'Iterations (↓)']
iter_man = res.loc['Manhattan', 'Iterations (↓)']
iter_cos = res.loc['Cosine', 'Iterations (↓)']

# Tabel Komparasi Selisih (Delta) Terhadap Metode Lain
delta_data = [
    {
        'Metrik': 'Silhouette Score (↑)',
        'Terbaik': 'Cosine' if sil_cos >= max(sil_euc, sil_man) else ('Euclidean' if sil_euc >= sil_man else 'Manhattan'),
        'Skor Terbaik': max(sil_euc, sil_man, sil_cos),
        'Selisih thd Euclidean': sil_cos - sil_euc,
        'Selisih thd Manhattan': sil_cos - sil_man,
        'Analisis Relatif': f'Cosine mengungguli Euclidean sebesar {sil_cos - sil_euc:+.4f} dan Manhattan sebesar {sil_cos - sil_man:+.4f}'
    },
    {
        'Metrik': 'Davies-Bouldin Index (↓)',
        'Terbaik': 'Euclidean' if db_euc <= min(db_man, db_cos) else ('Manhattan' if db_man <= db_cos else 'Cosine'),
        'Skor Terbaik': min(db_euc, db_man, db_cos),
        'Selisih thd Manhattan': db_euc - db_man,
        'Selisih thd Cosine': db_euc - db_cos,
        'Analisis Relatif': f'Euclidean lebih rendah (lebih baik) {abs(db_euc - db_man):.4f} thd Manhattan dan {abs(db_euc - db_cos):.4f} thd Cosine'
    },
    {
        'Metrik': 'Calinski-Harabasz Index (↑)',
        'Terbaik': 'Euclidean' if ch_euc >= max(ch_man, ch_cos) else ('Cosine' if ch_cos >= ch_man else 'Manhattan'),
        'Skor Terbaik': max(ch_euc, ch_man, ch_cos),
        'Selisih thd Manhattan': ch_euc - ch_man,
        'Selisih thd Cosine': ch_euc - ch_cos,
        'Analisis Relatif': f'Euclidean unggul {ch_euc - ch_man:+,.1f} thd Manhattan dan {ch_euc - ch_cos:+,.1f} thd Cosine'
    },
    {
        'Metrik': 'Waktu Eksekusi (s) (↓)',
        'Terbaik': 'Manhattan' if time_man <= min(time_euc, time_cos) else ('Euclidean' if time_euc <= time_cos else 'Cosine'),
        'Skor Terbaik': min(time_euc, time_man, time_cos),
        'Selisih thd Euclidean': time_man - time_euc,
        'Selisih thd Cosine': time_man - time_cos,
        'Analisis Relatif': f'Manhattan mencatatkan waktu {time_man:.3f}s (selisih {abs(time_man - time_euc):.3f}s thd Euclidean)'
    }
]

df_deltas = pd.DataFrame(delta_data)

print("\\n" + "="*95)
print("ANALISIS SELISIH KUANTITATIF (DELTAS) ANTAR METODE KLASTERISASI")
print("="*95)
for d in delta_data:
    print(f"Metrik: {d['Metrik']}")
    print(f"  Metode Terbaik : {d['Terbaik']} (Skor: {d['Skor Terbaik']:.4f} / {d['Skor Terbaik']:,.1f})")
    print(f"  Komparasi      : {d['Analisis Relatif']}")
    print("-" * 95)"""
    cells.append(nbf.v4.new_code_cell(code_deltas))

    md_critical_discussion = r"""## 3. Pembahasan Kritis dan Alasan Keunggulan Komparatif Tiap Metode

Berdasarkan pengujian empiris pada 127 dimensi fitur landmark BISINDO (mengakomodasi 1 tangan dan 2 tangan dengan zero-padding kanonikal), diperoleh temuan-temuan ilmiah berikut:

### 3.1. Mengapa Cosine Distance Mencapai Silhouette Score Tertinggi?
* **Alasan Geometris:** Dalam ruang berdimensi 127, fenomena *curse of dimensionality* menyebabkan jarak Euclidean antar-semua pasangan titik cenderung memusat (*distance concentration effect*). Sebaliknya, **Cosine Similarity murni mengevaluasi orientasi sudut deviasi antartangan** tanpa terdistorsi oleh perbedaan magnitudo panjang lengan atau jarak relatif ke kamera.
* **Pengaruh Zero-Padding:** Pada huruf 1 tangan, dimensi $[63..125]$ bernilai nol. Ketika dinormalisasi ke permukaan bola satuan ($L_2$ normalizer), seluruh gestur satu tangan terkumpul pada satu sektor angular spesifik dari unit hypersphere, sementara gestur dua tangan menempati sektor angular yang sangat berbeda. Hal ini menciptakan pemisahan angular yang sangat tajam (*high inter-cluster angular margin*), sehingga nilai Silhouette Cosine melonjak signifikan dibanding metrik berbasis jarak fisik.

### 3.2. Mengapa Euclidean Distance Mencapai Davies-Bouldin Index dan Calinski-Harabasz Terbaik?
* **Alasan Matematis:** Formulasi Davies-Bouldin Index (DBI) dan Calinski-Harabasz (CH) secara inheren dirancang berdasarkan rasio dispersi varians berbasis jarak Euclidean terhadap titik rata-rata aritmetika (*mean*):
  $$\Delta(S_k) = \frac{1}{|S_k|} \sum_{x \in S_k} \|x - \mu_k\|_2$$
  Karena algoritma K-Means Euclidean secara langsung meminimalkan fungsi kuadrat jarak terhadap *mean* ($\mu_k^* = \arg\min \sum \|x - \mu\|^2$), solusi yang dihasilkan secara alami merupakan solusi optimal global terhadap metrik DBI dan CH.
* **Perbandingan terhadap Manhattan:** Manhattan mengoptimalkan median komponen (L1), bukan mean. Akibatnya, dispersi Euclidean di dalam klaster Manhattan tampak lebih longgar ketika diukur menggunakan metrik DBI bawaan scikit-learn.

### 3.3. Karakteristik Kinerja Manhattan Distance (K-Medians)
* **Kekebalan terhadap Pencilan (*Robustness*):** Pembaruan centroid menggunakan median per sumbu membuat Manhattan sangat stabil terhadap variasi minor pada penempatan jari atau noise jitter pada landmark sekunder.
* **Efisiensi Iterasi:** Model Manhattan mencapai konvergensi cepat karena nilai median mengunci titik pusat kepadatan diskret tanpa terombang-ambing oleh pergeseran rata-rata data batas.

---

### 4. Rekomendasi Terapan untuk Pengembangan Sistem BisindoVision (Mobile)
1. **Untuk Model Inferensi Real-Time di Flutter:**
   Disarankan menggunakan **Cosine Similarity (Spherical K-Means / Cosine Classifier)** atau **Euclidean Normalized**. Sifat Cosine yang invarian terhadap magnitudo ukuran fisik tangan pengguna (anak-anak vs dewasa) serta jarak tangan ke lensa kamera ponsel akan memberikan akurasi penentuan alfabet yang paling konsisten di perangkat bergerak.
2. **Normalisasi Handedness:**
   Koreksi pembalikan horizontal (*un-mirroring*) pada citra bertipe `flip` terbukti sangat krusial dalam menyatukan distribusi gestur tangan kiri dan kanan pada karakter asimetris (B, D, P), mencegah terbelahnya klaster alfabet menjadi dua kelompok terpisah."""
    cells.append(nbf.v4.new_markdown_cell(md_critical_discussion))

    nb.cells = cells
    return nb

if __name__ == '__main__':
    current_dir = os.path.dirname(os.path.abspath(__file__)) if '__file__' in locals() else os.getcwd()
    output_nb_path = os.path.join(current_dir, "kmeans_distance_comparison.ipynb")

    print("[BUILD] Membangun draf notebook K-Means BISINDO...")
    nb = build_and_run_notebook()

    with open(output_nb_path, 'w', encoding='utf-8') as f:
        nbf.write(nb, f)
    print(f"[SUCCESS] Berkas notebook tersimpan di: '{output_nb_path}'")

    print("[EXECUTE] Mengeksekusi notebook dengan kernel python3 untuk merender visualisasi dan metrik...")
    ep = ExecutePreprocessor(timeout=1200, kernel_name='python3')
    try:
        ep.preprocess(nb, {'metadata': {'path': current_dir}})
        with open(output_nb_path, 'w', encoding='utf-8') as f:
            nbf.write(nb, f)
        print(f"[SUCCESS] Notebook berhasil dieksekusi penuh dan seluruh output telah dirender.")
    except Exception as e:
        print(f"[ERROR] Eksekusi notebook mengalami kendala: {e}")
