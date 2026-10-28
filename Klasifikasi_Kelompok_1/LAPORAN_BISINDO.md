# LAPORAN TUGAS PEMBELAJARAN MESIN (MACHINE LEARNING)
# KLASIFIKASI BAHASA ISYARAT INDONESIA (BISINDO) BERBASIS LANDMARK TANGAN MENGGUNAKAN KNN, NAIVE BAYES, DAN SVM

**Institusi:** Politeknik Negeri Malang (POLINEMA)  
**Program Studi:** D4 Teknik Informatika  
**Mata Kuliah:** Pembelajaran Mesin (Machine Learning) — Semester 5  
**Topik:** Klasifikasi Multi-Kelas 26 Alfabet BISINDO (A–Z)  
**Metode:** K-Nearest Neighbors (KNN), Naive Bayes (NB), dan Support Vector Machine (SVM)  

---

## RINGKASAN EKSEKUTIF (ABSTRACT)

Bahasa Isyarat Indonesia (BISINDO) merupakan media komunikasi utama bagi komunitas Tuli di Indonesia yang mengandalkan orientasi dan konfigurasi spasial tangan serta gestur tubuh. Penelitian ini mengkaji performa tiga algoritma pembelajaran mesin terawasi (*supervised learning*)—yaitu *K-Nearest Neighbors* (KNN), *Naive Bayes* (NB), dan *Support Vector Machine* (SVM)—dalam mengklasifikasikan 26 kelas alfabet BISINDO (A hingga Z).

Dataset yang digunakan terdiri dari **9.173 sampel citra**, yang diekstraksi ke dalam **127 dimensi fitur numerik** menggunakan MediaPipe Hand Landmarker (mencakup 21 titik koordinat 3D per tangan serta indikator gestur satu/dua tangan). Sebanyak **9 varian model** diuji menggunakan skema validasi *Stratified 80:20 Train-Test Split* dan *Stratified 5-Fold Cross-Validation*.

Hasil eksperimen menunjukkan bahwa **Support Vector Machine dengan kernel Radial Basis Function (SVM RBF)** menghasilkan kinerja tertinggi di seluruh metrik evaluasi dengan **Akurasi 97,91%**, **F1-Score 97,91%**, dan rata-rata **Cross-Validation F1-Score 97,09%**. Model **SVM Linear** dan **KNN (k=3)** menyusul secara kompetitif dengan F1-Score masing-masing sebesar **97,02%** dan **96,12%**. Sebaliknya, kelompok algoritma **Naive Bayes** mengalami *underperformance* signifikan (F1-Score antara **30,36% – 42,64%**), yang disebabkan oleh pelanggaran mendasar terhadap asumsi independensi fitur (*conditional independence assumption*) akibat tingginya korelasi geometris antar-sendi tangan.

---

## 1. PENDAHULUAN

### 1.1 Latar Belakang
Bahasa Isyarat Indonesia (BISINDO) merupakan bahasa alami yang berkembang secara organik di kalangan komunitas Tuli di Indonesia. Berbeda dengan Sistem Isyarat Bahasa Indonesia (SIBI) yang berbasis tata bahasa lisan terstruktur dan dominan menggunakan satu tangan, BISINDO memiliki karakteristik alami yang melibatkan penggunaan satu tangan (*one-handed*) maupun dua tangan (*two-handed*) secara dinamis.

Tantangan utama dalam pengenalan alfabet BISINDO secara komputasi terletak pada:
1. Variasi orientasi sudut pandang kamera, pencahayaan, dan proporsi morfologi tangan pengguna.
2. Tingginya kemiripan visual (*high intra-class similarity* dan *low inter-class variance*) pada beberapa karakter huruf tertentu, misalnya antara huruf 'M' dan 'N', atau 'U' dan 'V'.
3. Kebutuhan inferensi yang cepat dan akurat untuk potensi implementasi pada perangkat interaktif *real-time*.

### 1.2 Tujuan Penelitian
Tujuan dari penelitian dan eksperimen ini adalah:
1. Mengimplementasikan pipeline ekstraksi fitur berbasis koordinat *hand landmarks* MediaPipe pada dataset alfabet BISINDO 26 kelas.
2. Membandingkan performa 3 algoritma klasifikasi (*KNN, Naive Bayes, SVM*) dengan total 9 variasi konfigurasi hyperparameter.
3. Menganalisis kelebihan, kelemahan, dan trade-off waktu komputasi vs akurasi dari setiap algoritma untuk rekomendasi implementasi nyata.

---

## 2. METODOLOGI PENELITIAN

### 2.1 Dataset BISINDO
Dataset yang digunakan merupakan kumpulan data citra gestur alfabet BISINDO dari huruf **A sampai Z** (26 kelas).
- **Total Sampel:** 9.173 data citra.
- **Distribusi per Kelas:** Relatif seimbang (*balanced*), berkisar antara 348 hingga 360 sampel per kelas huruf.
- **Rata-rata Sampel per Kelas:** ±353 sampel.

Grafik sebaran sampel per kelas tersimpan pada file `output_complete/00_distribusi_dataset.png`.

### 2.2 Ekstraksi Fitur (Feature Extraction)
Alih-alih menggunakan nilai piksel mentah (*raw image pixels*) yang berdimensi sangat besar dan rentan terhadap gangguan latar belakang (*background noise*), representasi citra ditransformasikan ke dalam representasi koordinat geometris menggunakan pustaka **MediaPipe Hand Landmarker**.

Struktur vektor fitur (127 dimensi numerik) terdiri dari:
1. **Tangan Pertama (Hand 1):** 21 landmark titik sendi × 3 koordinat $(x, y, z) = 63$ fitur.
2. **Tangan Kedua (Hand 2):** 21 landmark titik sendi × 3 koordinat $(x, y, z) = 63$ fitur (bernilai 0 jika gestur hanya melibatkan satu tangan).
3. **Indikator Gestur (`is_two_handed`):** 1 fitur biner bernilai $1.0$ (dua tangan) atau $0.0$ (satu tangan).
$$\text{Total Dimensi Fitur} = 63 + 63 + 1 = 127 \text{ dimensi}$$

Dataset fitur yang telah diekstraksi disimpan dalam format terkompresi `dataset_bisindo_landmarks.npz`.

### 2.3 Pra-pemrosesan Data (Data Preprocessing)
Pipeline pra-pemrosesan data dirancang secara terisolasi guna menghindari kebocoran data (*data leakage*):
1. **Pembersihan Data:** Nilai *Not-a-Number* (NaN) atau tak hingga digantikan dengan 0.0 melalui `np.nan_to_num`.
2. **Pembagian Data (Train-Test Split):**
   - Data Latih (*Training Set*): 80% (7.338 sampel)
   - Data Uji (*Testing Set*): 20% (1.835 sampel)
   - Metode partisi: *Stratified sampling* dengan `random_state=42` untuk mempertahankan proporsi kelas asli.
3. **Penyesuaian Skala Fitur (Feature Scaling):**
   - **StandardScaler ($z = \frac{x - \mu}{\sigma}$):** Diterapkan pada algoritma berbasis jarak dan margin, yaitu **KNN, Gaussian Naive Bayes, dan SVM**.
   - **MinMaxScaler ($x_{scaled} = \frac{x - x_{min}}{x_{max} - x_{min}} \in [0, 1]$):** Diterapkan khusus untuk **Complement Naive Bayes**, yang mensyaratkan nilai input non-negatif.
   - **Binarizer ($\text{threshold} = 0.0$):** Diterapkan khusus untuk **Bernoulli Naive Bayes** setelah standardisasi untuk mengonversi nilai menjadi biner $\{0, 1\}$.

### 2.4 Protokol Evaluasi
Setiap model dievaluasi menggunakan empat metrik performa utama pada data uji:
1. **Accuracy:** Proporsi total prediksi yang benar terhadap keseluruhan data uji.
2. **Precision (Weighted):** Rata-rata tertimbang presisi tiap kelas terhadap jumlah sampel sebenarnya.
3. **Recall (Weighted):** Rata-rata tertimbang kemampuan model mendeteksi sampel kelas yang benar.
4. **F1-Score (Weighted):** Rata-rata harmonik antara Precision dan Recall.
5. **Stratified 5-Fold Cross-Validation:** Mengukur stabilitas dan kemampuan generalisasi model terhadap variasi partisi data, dinyatakan dalam nilai rata-rata (*mean*) dan standar deviasi (*std*).
6. **Waktu Komputasi:** Pencatatan waktu pelatihan (*training time*) dan waktu inferensi/prediksi (*inference time*) dalam satuan detik.

---

## 3. KONFIGURASI MODEL KLASIFIKASI

Sebanyak 9 varian model dari 3 algoritma dibangun dengan konfigurasi parameter sebagai berikut:

| No | Algoritma | Varian Model | Konfigurasi Hyperparameter | Skalabilitas Fitur |
|:--:|:---|:---|:---|:---|
| 1 | K-Nearest Neighbors | KNN ($k=3$) | `n_neighbors=3`, `metric='euclidean'`, `weights='distance'`, `algorithm='ball_tree'` | StandardScaler |
| 2 | K-Nearest Neighbors | KNN ($k=5$) | `n_neighbors=5`, `metric='euclidean'`, `weights='distance'`, `algorithm='ball_tree'` | StandardScaler |
| 3 | K-Nearest Neighbors | KNN ($k=7$) | `n_neighbors=7`, `metric='euclidean'`, `weights='distance'`, `algorithm='ball_tree'` | StandardScaler |
| 4 | Naive Bayes | Gaussian NB | `var_smoothing=1e-9` | StandardScaler |
| 5 | Naive Bayes | Complement NB | `alpha=1.0`, `norm=False` | MinMaxScaler |
| 6 | Naive Bayes | Bernoulli NB | `alpha=1.0`, `binarize=None` (input data biner) | StandardScaler + Binarizer |
| 7 | Support Vector Machine | SVM Linear | `kernel='linear'`, $C=1.0$, `random_state=42` | StandardScaler |
| 8 | Support Vector Machine | SVM RBF | `kernel='rbf'`, $C=10.0$, $\gamma=\text{'scale'}$, `random_state=42` | StandardScaler |
| 9 | Support Vector Machine | SVM Polynomial | `kernel='poly'`, $\text{degree}=3$, $C=1.0$, $\gamma=\text{'scale'}$, `random_state=42` | StandardScaler |

---

## 4. HASIL EKSPERIMEN DAN TABEL PERBANDINGAN

Tabel berikut menyajikan hasil komparasi performa keseluruhan 9 model, diurutkan berdasarkan nilai **F1-Score (Weighted)** tertinggi:

### Tabel 1: Peringkat Performa Komparatif 9 Model Klasifikasi BISINDO

| Peringkat | Nama Model | Algoritma | Accuracy | Precision | Recall | F1-Score | CV F1 (Mean ± Std) | Waktu Latih (s) | Waktu Uji (s) |
|:---:|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **1** | **SVM RBF** | **SVM** | **0.9791** | **0.9797** | **0.9791** | **0.9791** | **0.9709 ± 0.0040** | 0.838 | 0.527 |
| **2** | **SVM Linear** | SVM | 0.9702 | 0.9705 | 0.9702 | 0.9702 | 0.9674 ± 0.0029 | 0.803 | 0.207 |
| **3** | **KNN (k=3)** | **KNN** | 0.9614 | 0.9619 | 0.9614 | 0.9612 | 0.9528 ± 0.0017 | **0.054** | 0.195 |
| 4 | KNN (k=5) | KNN | 0.9550 | 0.9558 | 0.9550 | 0.9547 | 0.9464 ± 0.0030 | 0.055 | 0.202 |
| 5 | KNN (k=7) | KNN | 0.9506 | 0.9513 | 0.9506 | 0.9503 | 0.9421 ± 0.0027 | **0.049** | 0.212 |
| 6 | SVM Polynomial | SVM | 0.9208 | 0.9301 | 0.9208 | 0.9223 | 0.9087 ± 0.0051 | 1.303 | 0.441 |
| 7 | Naive Bayes Bernoulli | NaiveBayes | 0.4630 | 0.4698 | 0.4630 | 0.4264 | 0.4149 ± 0.0015 | 0.014 | **0.003** |
| 8 | Naive Bayes Gaussian | NaiveBayes | 0.4573 | 0.4567 | 0.4573 | 0.4004 | 0.3545 ± 0.0242 | 0.008 | 0.042 |
| 9 | Naive Bayes Complement | NaiveBayes | 0.3711 | 0.3667 | 0.3711 | 0.3036 | 0.3054 ± 0.0136 | **0.007** | **0.001** |

### Tabel 2: Model Terbaik pada Masing-Masing Algoritma

| Algoritma | Varian Model Terbaik | Accuracy | Precision | Recall | F1-Score | CV F1 Mean |
|:---|:---|:---:|:---:|:---:|:---:|:---:|
| **Support Vector Machine (SVM)** | **SVM RBF** | **0.9791** | **0.9797** | **0.9791** | **0.9791** | **0.9709** |
| **K-Nearest Neighbors (KNN)** | **KNN (k=3)** | 0.9614 | 0.9619 | 0.9614 | 0.9612 | 0.9528 |
| **Naive Bayes** | **NB Bernoulli** | 0.4630 | 0.4698 | 0.4630 | 0.4264 | 0.4149 |

---

## 5. ANALISIS DAN DISKUSI

### 5.1 Analisis Support Vector Machine (SVM) — Performa Tertinggi
Support Vector Machine terbukti menjadi algoritma paling unggul dalam penelitian ini:
1. **Keunggulan Kernel RBF:** Varian SVM RBF ($C=10.0$) mencapai F1-Score tertinggi (**97,91%**). Fungsi kernel RBF berhasil memetakan ruang fitur 127 dimensi ke dalam ruang berdimensi tak hingga (*infinite-dimensional Hilbert space*), memungkinkan pembentukan *hyperplane* pemisah yang sangat fleksibel untuk membedakan pola lengkungan jari dan jarak spasial antar-sendi tangan yang non-linear.
2. **Kinerja Kernel Linear:** SVM Linear juga menunjukkan hasil yang sangat impresif (**97,02%** F1-Score), menegaskan bahwa pada dimensi 127 fitur, sebagian besar kelas alfabet BISINDO telah memiliki derajat separabilitas linear yang tinggi setelah diskalakan dengan StandardScaler.
3. **Keterbatasan Kernel Polynomial:** Kernel Polynomial tingkat 3 mencapai F1-Score **92,23%**, berada di bawah RBF dan Linear, dengan waktu pelatihan terlama (**1,30 detik**). Hal ini disebabkan oleh kompleksitas komputasi perkalian polinomial pada dimensi 127 serta kecenderungan *over-fitting* ringan pada data laten.

### 5.2 Analisis K-Nearest Neighbors (KNN) — Efisiensi dan Akurasi Tinggi
Algoritma KNN menunjukkan kinerja yang sangat memuaskan dan stabil:
1. **Pengaruh Nilai $k$:** Model dengan nilai $k=3$ memberikan hasil terbaik (**96,12%** F1-Score), diikuti oleh $k=5$ (**95,47%**) dan $k=7$ (**95,03%**). Tren penurunan seiring meningkatnya nilai $k$ menunjukkan bahwa batas antar-kelas alfabet BISINDO bersifat lokal (*tight local clusters*). Peningkatan nilai $k$ yang terlalu besar justru memasukkan tetangga dari kelas lain yang memiliki kemiripan gestur.
2. **Efektivitas Pembobotan Jarak (`weights='distance'`):** Penerapan bobot berbasis kebalikan jarak invers terbukti krusial dalam mereduksi ambiguitas prediksi pada sampel yang berada dekat dengan perbatasan kelas.
3. **Efisiensi Waktu Latih:** Waktu pelatihan KNN sangat cepat (**0,057 detik**) karena KNN merupakan *lazy learner* (hanya membangun indeks *Ball Tree* tanpa proses optimasi parameter berulang).

### 5.3 Analisis Kegagalan Naive Bayes (Underperformance Analysis)
Algoritma Naive Bayes menghasilkan performa yang sangat rendah dibandingkan dua algoritma lainnya (**F1-Score: 30,36% – 42,64%**). Analisis teoretis dan empiris mengungkap penyebab utamanya:
1. **Pelanggaran Asumsi Independensi Kondisional:** Naive Bayes bekerja dengan asumsi bahwa setiap fitur bersifat independen satu sama lain bersyarat pada kelas ($P(x_1, x_2, \dots, x_d | y) = \prod_{i=1}^d P(x_i | y)$). Pada data landmark tangan, posisi sendi jari (misalnya ujung telunjuk, sendi PIP, dan sendi MCP) memiliki korelasi fisik dan anatomis yang sangat kuat. Ketika jari ditekuk, seluruh koordinat sendi pada jari tersebut bergerak bersamaan. Asumsi independensi ini dilanggar secara ekstrem, mengakibatkan estimasi probabilitas posterior menjadi terdistorsi secara drastis.
2. **Distribusi Non-Gaussian pada Fitur Koordinat:** Variasi koordinat landmark tangan pada gestur tertentu tidak mengikuti kurva lonceng normal murni, sehingga varian Gaussian NB mengalami deviasi prediksi yang tinggi.
3. **Hilangnya Informasi Metrik pada Binarisasi:** Varian Bernoulli NB dan Complement NB memotong informasi granular kontinu dari koordinat posisi sendi menjadi skala biner atau frekuensi semu, yang membuang informasi sudut dan jarak presisi yang diperlukan untuk membedakan huruf.

### 5.4 Analisis Trade-off Waktu Komputasi vs Akurasi
Berdasarkan visualisasi scatter plot pada `output_complete/05_tradeoff_time_f1.png`:
- **Untuk Akurasi Maksimal:** **SVM RBF** adalah pilihan terbaik (F1: 97,91%), dengan waktu inferensi 0,504 detik untuk 1.835 data uji (~0,27 milidetik per frame citra).
- **Untuk Keseimbangan Akurasi dan Kecepatan Real-Time:** **KNN (k=3)** menawarkan efisiensi komputasi sangat tinggi (Waktu Latih: 0,057s, Waktu Prediksi: 0,184s untuk 1.835 data atau ~0,10 milidetik per frame) dengan tetap mempertahankan akurasi di atas 96,1%.

---

## 6. ANALISIS PER KELAS DAN CONFUSION MATRIX

Berdasarkan pengujian matriks konfusi (`output_complete/07_confusion_matrices_all.png` dan `08_cm_best_per_algo.png`):

1. **Huruf dengan Akurasi Sempurna (F1-Score ≈ 1.000):**
   - Karakter huruf yang memiliki karakteristik gestur unik atau melibatkan dua tangan yang khas (misalnya huruf **A, B, C, I, L, O, Y**) hampir tidak mengalami kesalahan klasifikasi pada SVM RBF maupun KNN.
2. **Pola Kesalahan / Konfusi yang Ditemukan:**
   - **Huruf M dan N:** Keduanya memiliki kemiripan gestur jempol yang terselip di antara jari-jari telunjuk/tengah, menghasilkan variasi koordinat landmark yang berjarak sangat dekat.
   - **Huruf K dan V:** Keduanya mengandalkan dua jari yang ditegakkan dengan sedikit perbedaan pada posisi ibu jari di bagian tengah telapak tangan.
   - **Huruf D dan Z:** Memiliki kemiripan posisi jari telunjuk yang tegak saat statis jika tidak disertai informasi lintasan gerak dinamis.
3. **Pada Naive Bayes:**
   - Terjadi penyebaran prediksi yang sangat lebar (*high dispersion error*), di mana model sering kali salah mengarahkan prediksi ke huruf-huruf dengan prioritas probabilitas dominan yang salah hitung.

---

## 7. KESIMPULAN DAN REKOMENDASI

### 7.1 Kesimpulan
1. **Model Terbaik Keseluruhan:** **Support Vector Machine dengan kernel RBF (SVM RBF)** terbukti sebagai model paling optimal untuk klasifikasi alfabet BISINDO pada dataset 127 landmark MediaPipe, dengan nilai **Akurasi 97,91%**, **F1-Score 97,91%**, dan stabilitas **5-Fold Cross-Validation 97,09% ± 0,004**.
2. **Performa Algoritma:** Urutan performa secara konsisten adalah:
   $$\text{SVM (RBF / Linear)} > \text{KNN (k=3, 5, 7)} \gg \text{Naive Bayes (Bernoulli / Gaussian / Complement)}$$
3. **Validitas Representasi Fitur:** Penggunaan 127 fitur MediaPipe Hand Landmarks terbukti sangat efektif, kompak, dan mampu mereduksi kompleksitas komputasi citra secara signifikan dibandingkan pendekatan piksel mentah berbasis CNN, sehingga memungkinkan proses pelatihan selesai dalam hitungan detik.

### 7.2 Rekomendasi Penerapan
1. **Implementasi Aplikasi Interaktif / Mobile:**
   - Disarankan menggunakan **SVM RBF** yang diekspor menggunakan pustaka `joblib` atau `ONNX Runtime`.
   - Alternatif lain, **KNN (k=3)** dapat digunakan pada arsitektur perangkat keras dengan keterbatasan memori karena proses pelatihan instan dan waktu inferensi yang sangat singkat.
2. **Pengembangan Lanjutan:**
   - Menambahkan fitur orientasi sudut rotasi (*wrist angle*) dan rasio jarak antar-ujung jari (*finger-tip distance ratios*) untuk membedakan gestur yang mirip secara lebih tegas.
   - Mengembangkan dataset gestur dinamis berbasis sekuens waktu (menggunakan LSTM / GRU) untuk kata atau kalimat dalam BISINDO yang melibatkan gerakan temporal.

---

## 8. DAFTAR ARTIFAK DAN FILE OUTPUT

Seluruh hasil analisis visual dan data komparasi telah tersimpan secara otomatis pada direktori:
`C:\POLINEMA\SEMESTER 5\ML\Klasifikasi_Kelompok_1\output_complete\`

Daftar file keluaran:
- `comparison_table.csv`: Tabel perbandingan numerik lengkap 9 model.
- `final_results.json`: Rekapitulasi metrik evaluasi dan skor 5-fold CV dalam format JSON.
- `00_distribusi_dataset.png`: Grafik batang distribusi 9.173 sampel per kelas alfabet.
- `01_comparison_metrics.png`: Diagram batang perbandingan 4 metrik evaluasi untuk seluruh model.
- `02_heatmap_metrics.png`: Peta panas performa model.
- `03_radar_chart.png`: Radar chart profil multi-metrik 9 model.
- `04_grouped_best_per_algo.png`: Grafik perbandingan performa model terbaik tiap algoritma.
- `05_tradeoff_time_f1.png`: Scatter plot trade-off waktu pelatihan & inferensi vs F1-score.
- `06_per_class_*.png`: Grafik metrik Precision, Recall, dan F1-Score per huruf alfabet (A–Z).
- `07_confusion_matrices_all.png`: Matriks konfusi lengkap 9 model (grid 3×3).
- `08_cm_best_per_algo.png`: Matriks konfusi model terbaik per algoritma (grid 1×3).
- `bisindo_classification_complete.ipynb`: Notebook Jupyter tunggal terstruktur dan profesional.
