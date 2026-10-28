# -*- coding: utf-8 -*-
"""
Membuat laporan PDF ringkas (maks. 2 halaman) untuk proyek
Klasifikasi Alfabet BISINDO - Kelompok 1.
"""

import os
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from reportlab.lib.pagesizes import A4
from reportlab.lib.units import cm
from reportlab.lib import colors
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.enums import TA_JUSTIFY, TA_CENTER
from reportlab.platypus import (
    BaseDocTemplate, Frame, PageTemplate, Paragraph, Spacer,
    Table, TableStyle, Image, KeepTogether, PageBreak,
)

BASE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(BASE, "output_complete")
PDF_PATH = os.path.join(BASE, "LAPORAN_BISINDO_Kelompok_1.pdf")
CHART = os.path.join(BASE, "assets_laporan", "_chart_f1_ringkas.png")

NAVY = colors.HexColor("#1F3864")
BLUE = colors.HexColor("#2E75B6")
LIGHT = colors.HexColor("#EAF1F8")
GREY = colors.HexColor("#595959")
LINE = colors.HexColor("#BFCFE0")
NBSP = "\u00a0"
MDASH = "\u2014"
PM = "\u00b1"
TIMES = "\u00d7"

# ---------------------------------------------------------------- data
df = pd.read_csv(os.path.join(OUT, "comparison_table.csv"))
df.columns = [c.strip() for c in df.columns]
df = df.rename(columns={df.columns[0]: "Rank"})
df["Rank"] = df["Rank"].astype(int)

ACCENT = {"SVM": "#1F3864", "KNN": "#2E9E6B", "NaiveBayes": "#C0504D"}


# ---------------------------------------------------------------- chart
def buat_chart():
    os.makedirs(os.path.dirname(CHART), exist_ok=True)
    d = df.sort_values("F1-Score")
    fig, ax = plt.subplots(figsize=(7.0, 2.25), dpi=300)
    warna = [ACCENT.get(a, "#888888") for a in d["Algoritma"]]
    bars = ax.barh(d["Model"], d["F1-Score"] * 100, color=warna, height=0.68)
    for b, v in zip(bars, d["F1-Score"] * 100):
        ax.text(v + 1.2, b.get_y() + b.get_height() / 2, f"{v:.2f}",
                va="center", fontsize=6.6, color="#333333")
    ax.set_xlim(0, 108)
    ax.set_xlabel("F1-Score (Weighted) pada Data Uji (%)", fontsize=6.8, color="#333333")
    ax.tick_params(axis="both", labelsize=6.6, colors="#333333")
    ax.set_axisbelow(True)
    ax.xaxis.grid(True, linestyle="--", linewidth=0.5, color="#DDDDDD")
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.spines["bottom"].set_color("#BBBBBB")
    ax.set_title("Perbandingan F1-Score Seluruh Model", fontsize=7.6,
                 fontweight="bold", color="#1F3864", loc="left")
    handles = [plt.Rectangle((0, 0), 1, 1, color=c) for c in ACCENT.values()]
    ax.legend(handles, ["SVM", "KNN", "Naive Bayes"], fontsize=6, frameon=False,
              loc="lower right", ncol=3)
    fig.tight_layout(pad=0.3)
    fig.savefig(CHART, bbox_inches="tight", facecolor="white")
    plt.close(fig)


# ---------------------------------------------------------------- styles
def S(name, **kw):
    base = dict(fontName="Helvetica", fontSize=8.3, leading=10.6,
                textColor=colors.HexColor("#222222"), spaceAfter=0)
    base.update(kw)
    return ParagraphStyle(name, **base)


st_title = S("t", fontName="Helvetica-Bold", fontSize=15.5, leading=18,
             textColor=NAVY, alignment=TA_CENTER, spaceAfter=1)
st_sub = S("s", fontName="Helvetica-Bold", fontSize=9.4, leading=11.8,
           textColor=BLUE, alignment=TA_CENTER, spaceAfter=2)
st_meta = S("m", fontSize=7.6, leading=9.4, textColor=GREY, alignment=TA_CENTER)
st_body = S("b", alignment=TA_JUSTIFY, spaceAfter=2.5)
st_abs = S("a", fontSize=7.9, leading=10.1, alignment=TA_JUSTIFY,
           textColor=colors.HexColor("#1B2A41"))
st_bul = S("bl", alignment=TA_JUSTIFY, leftIndent=9, bulletIndent=1,
           spaceAfter=1.6)
st_cap = S("c", fontSize=6.8, leading=8.2, textColor=GREY, alignment=TA_CENTER,
           spaceBefore=1.5)
st_cell = S("cl", fontSize=7.1, leading=8.6)
st_cellb = S("cb", fontName="Helvetica-Bold", fontSize=7.1, leading=8.6,
             textColor=colors.white, alignment=TA_CENTER)
st_cellc = S("cc", fontSize=7.1, leading=8.6, alignment=TA_CENTER)


def pct(x, d=2):
    return f"{x*100:.{d}f}".replace(".", ",")


def hdr(text):
    t = Table([[Paragraph(text, S("hh", fontName="Helvetica-Bold", fontSize=9.6,
                                  leading=11.5, textColor=colors.white))]],
              colWidths=[17.6 * cm])
    t.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), NAVY),
        ("LEFTPADDING", (0, 0), (-1, -1), 5),
        ("TOPPADDING", (0, 0), (-1, -1), 2.2),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 2.2),
    ]))
    return [Spacer(1, 5), t, Spacer(1, 3)]


# ---------------------------------------------------------------- tables
def tabel_hasil():
    head = ["#", "Model", "Acc.", "Prec.", "Rec.", "F1",
            "CV F1 (5-Fold)", "Latih (s)", "Uji (s)"]
    rows = [[Paragraph(h, st_cellb) for h in head]]
    for _, r in df.sort_values("F1-Score", ascending=False).iterrows():
        top = r["Rank"] <= 3
        rows.append([
            Paragraph(str(int(r["Rank"])), st_cellc),
            Paragraph(("<b>%s</b>" % r["Model"]) if top else r["Model"], st_cell),
            Paragraph(pct(r["Accuracy"]), st_cellc),
            Paragraph(pct(r["Precision"]), st_cellc),
            Paragraph(pct(r["Recall"]), st_cellc),
            Paragraph("<b>%s</b>" % pct(r["F1-Score"]), st_cellc),
            Paragraph("%s %s %s" % (pct(r["CV F1 Mean"]), PM, pct(r["CV F1 Std"])), st_cellc),
            Paragraph(f'{r["Train Time (s)"]:.3f}'.replace(".", ","), st_cellc),
            Paragraph(f'{r["Pred Time (s)"]:.3f}'.replace(".", ","), st_cellc),
        ])
    t = Table(rows, colWidths=[0.8*cm, 3.9*cm, 1.5*cm, 1.5*cm, 1.5*cm, 1.5*cm,
                               3.1*cm, 1.85*cm, 1.85*cm], repeatRows=1)
    style = [
        ("BACKGROUND", (0, 0), (-1, 0), NAVY),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("GRID", (0, 0), (-1, -1), 0.4, LINE),
        ("TOPPADDING", (0, 0), (-1, -1), 2),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 2),
        ("LEFTPADDING", (0, 0), (-1, -1), 2.5),
        ("RIGHTPADDING", (0, 0), (-1, -1), 2.5),
    ]
    for i in range(1, len(rows)):
        if i % 2 == 0:
            style.append(("BACKGROUND", (0, i), (-1, i), colors.HexColor("#F4F7FB")))
    style.append(("BACKGROUND", (0, 1), (-1, 1), colors.HexColor("#DCE8F5")))
    t.setStyle(TableStyle(style))
    return t


def tabel_konfigurasi():
    data = [
        ("KNN", "KNN (k = 3 / 5 / 7)",
         "n_neighbors = 3/5/7, metric = euclidean, weights = distance, algorithm = ball_tree",
         "StandardScaler"),
        ("Naive Bayes", "Gaussian NB", "var_smoothing = 1e-9", "StandardScaler"),
        ("Naive Bayes", "Complement NB", "alpha = 1.0, norm = False", "MinMaxScaler"),
        ("Naive Bayes", "Bernoulli NB", "alpha = 1.0, binarize = None (input biner)",
         "Std. + Binarizer"),
        ("SVM", "SVM Linear", "kernel = linear, C = 1.0", "StandardScaler"),
        ("SVM", "SVM RBF", "kernel = rbf, C = 10.0, gamma = scale", "StandardScaler"),
        ("SVM", "SVM Polynomial", "kernel = poly, degree = 3, C = 1.0, gamma = scale",
         "StandardScaler"),
    ]
    head = ["Algoritma", "Varian Model", "Hyperparameter", "Skalabilitas"]
    rows = [[Paragraph(h, st_cellb) for h in head]]
    for a, v, h, s in data:
        rows.append([Paragraph(a, st_cell), Paragraph(v, st_cell),
                     Paragraph(h, st_cell), Paragraph(s, st_cellc)])
    t = Table(rows, colWidths=[2.5*cm, 3.5*cm, 8.6*cm, 3.0*cm], repeatRows=1)
    style = [
        ("BACKGROUND", (0, 0), (-1, 0), NAVY),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("GRID", (0, 0), (-1, -1), 0.4, LINE),
        ("TOPPADDING", (0, 0), (-1, -1), 1.9),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 1.9),
        ("LEFTPADDING", (0, 0), (-1, -1), 3),
    ]
    for i in range(1, len(rows)):
        if i % 2 == 0:
            style.append(("BACKGROUND", (0, i), (-1, i), colors.HexColor("#F4F7FB")))
    t.setStyle(TableStyle(style))
    return t


def tabel_terbaik():
    head = ["Algoritma", "Varian Terbaik", "Accuracy", "Precision", "Recall", "F1-Score", "CV F1 Mean"]
    best = {"SVM": "SVM RBF", "KNN": "KNN (k=3)", "NaiveBayes": "NB Bernoulli"}
    rows = [[Paragraph(h, st_cellb) for h in head]]
    for algo in ("SVM", "KNN", "NaiveBayes"):
        sub = df[df["Algoritma"] == algo].sort_values("F1-Score", ascending=False)
        r = sub.iloc[0]
        nama = {"SVM": "Support Vector Machine", "KNN": "K-Nearest Neighbors",
                "NaiveBayes": "Naive Bayes"}[algo]
        rows.append([
            Paragraph(nama, st_cell),
            Paragraph("<b>%s</b>" % best[algo], st_cell),
            Paragraph(pct(r["Accuracy"]), st_cellc),
            Paragraph(pct(r["Precision"]), st_cellc),
            Paragraph(pct(r["Recall"]), st_cellc),
            Paragraph("<b>%s</b>" % pct(r["F1-Score"]), st_cellc),
            Paragraph(pct(r["CV F1 Mean"]), st_cellc),
        ])
    t = Table(rows, colWidths=[4.6*cm, 3.2*cm, 2.1*cm, 2.1*cm, 2.1*cm, 2.1*cm, 2.2*cm],
              repeatRows=1)
    style = [
        ("BACKGROUND", (0, 0), (-1, 0), NAVY),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("GRID", (0, 0), (-1, -1), 0.4, LINE),
        ("TOPPADDING", (0, 0), (-1, -1), 2.2),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 2.2),
        ("LEFTPADDING", (0, 0), (-1, -1), 3),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#F4F7FB")]),
    ]
    t.setStyle(TableStyle(style))
    return t


# ---------------------------------------------------------------- page frame
def footer(canvas, doc):
    canvas.saveState()
    w, h = A4
    canvas.setStrokeColor(LINE)
    canvas.setLineWidth(0.6)
    canvas.line(1.7 * cm, 1.35 * cm, w - 1.7 * cm, 1.35 * cm)
    canvas.setFont("Helvetica", 6.9)
    canvas.setFillColor(GREY)
    canvas.drawString(1.7 * cm, 0.95 * cm,
                      "Kelompok 1 %s Klasifikasi Alfabet BISINDO | Pembelajaran Mesin, "
                      "D4 Teknik Informatika POLINEMA" % MDASH)
    canvas.drawRightString(w - 1.7 * cm, 0.95 * cm, "Halaman %d dari 2" % doc.page)
    canvas.setFillColor(NAVY)
    canvas.rect(0, h - 0.42 * cm, w, 0.42 * cm, stroke=0, fill=1)
    canvas.restoreState()


def build():
    buat_chart()
    doc = BaseDocTemplate(PDF_PATH, pagesize=A4,
                          leftMargin=1.7*cm, rightMargin=1.7*cm,
                          topMargin=1.25*cm, bottomMargin=1.55*cm,
                          title="Laporan Klasifikasi BISINDO - Kelompok 1",
                          author="Kelompok 1 - D4 Teknik Informatika POLINEMA")
    frame = Frame(doc.leftMargin, doc.bottomMargin, doc.width, doc.height,
                  id="f", leftPadding=0, rightPadding=0, topPadding=0, bottomPadding=0)
    doc.addPageTemplates([PageTemplate(id="p", frames=[frame], onPage=footer)])

    f = []
    f.append(Paragraph("LAPORAN PROYEK PEMBELAJARAN MESIN", st_title))
    f.append(Paragraph("Klasifikasi Alfabet Bahasa Isyarat Indonesia (BISINDO) Berbasis "
                       "Hand Landmark MediaPipe Menggunakan KNN, Naive Bayes, dan SVM", st_sub))
    f.append(Paragraph("Kelompok 1 %s D4 Teknik Informatika, Politeknik Negeri Malang "
                       "%s Semester 5 %s Mata Kuliah Pembelajaran Mesin"
                       % (NBSP + "\u2022" + NBSP, NBSP + "\u2022" + NBSP, NBSP + "\u2022" + NBSP),
                       st_meta))
    f.append(Spacer(1, 4))

    # abstrak
    abs_tbl = Table([[Paragraph(
        "<b>ABSTRAK.</b> Penelitian ini membandingkan tiga algoritma <i>supervised learning</i>%s"
        "<i>K-Nearest Neighbors</i> (KNN), <i>Naive Bayes</i> (NB), dan <i>Support Vector Machine</i> "
        "(SVM)%suntuk mengklasifikasikan 26 kelas alfabet BISINDO (A%sZ). Sebanyak 9.173 sampel citra "
        "direpresentasikan ke dalam 127 fitur numerik hasil ekstraksi <i>MediaPipe Hand Landmarker</i> "
        "(21 titik sendi %s 3 koordinat untuk dua tangan, ditambah indikator gestur dua tangan). "
        "Sembilan varian model dievaluasi melalui <i>Stratified 80:20 Train-Test Split</i> dan "
        "<i>Stratified 5-Fold Cross-Validation</i>. Hasil menunjukkan <b>SVM RBF</b> sebagai model "
        "terbaik dengan <b>akurasi 97,91%%</b>, <b>F1-Score 97,91%%</b>, dan <b>CV F1 97,09%% %s 0,40%%</b>, "
        "diikuti SVM Linear (97,02%%) dan KNN k=3 (96,12%%). Kelompok Naive Bayes mengalami "
        "<i>underperformance</i> (F1 30,36%%%s42,64%%) akibat pelanggaran asumsi independensi kondisional "
        "antar fitur landmark yang saling berkorelasi secara anatomis."
        % (MDASH, MDASH, "\u2013", TIMES, PM, "\u2013"), st_abs)]],
        colWidths=[17.6*cm])
    abs_tbl.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), LIGHT),
        ("BOX", (0, 0), (-1, -1), 0.5, LINE),
        ("LEFTPADDING", (0, 0), (-1, -1), 7), ("RIGHTPADDING", (0, 0), (-1, -1), 7),
        ("TOPPADDING", (0, 0), (-1, -1), 5), ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
    ]))
    f.append(abs_tbl)

    # 1. dataset
    f += hdr("1. DATASET DAN METODOLOGI")
    f.append(Paragraph(
        "Dataset terdiri atas <b>9.173 citra gestur alfabet BISINDO (26 kelas)</b> dengan distribusi "
        "hampir seimbang (348%s360 sampel per kelas, rata-rata %s353). Alih-alih piksel mentah, citra "
        "ditransformasikan menjadi representasi geometris 127 dimensi: 21 landmark %s 3 koordinat (x, y, z) "
        "untuk tangan pertama (63 fitur), 63 fitur untuk tangan kedua (bernilai 0 pada gestur satu tangan), "
        "dan 1 fitur biner <i>is_two_handed</i>. Pra-pemrosesan mencakup pembersihan nilai NaN/%s "
        "(<i>np.nan_to_num</i>), pembagian data <i>stratified</i> 80:20 (7.338 latih / 1.835 uji, "
        "<i>random_state</i> = 42), serta standardisasi <i>z-score</i> yang di-<i>fit</i> hanya pada data "
        "latih untuk mencegah kebocoran data. Evaluasi menggunakan Accuracy, Precision, Recall, dan "
        "F1-Score berbobot, dilengkapi 5-<i>fold cross-validation</i> serta pengukuran waktu latih dan "
        "inferensi." % ("\u2013", PM, TIMES, "\u221e"), st_body))

    # 2. konfigurasi
    f += hdr("2. KONFIGURASI MODEL")
    f.append(tabel_konfigurasi())

    # 3. hasil
    f += hdr("3. HASIL PERBANDINGAN PERFORMA")
    f.append(tabel_hasil())
    f.append(Paragraph("Tabel 1. Performa 9 varian model pada data uji (1.835 sampel), "
                       "diurutkan berdasarkan F1-Score.", st_cap))

    # ---------------- halaman 2 ----------------
    f.append(Spacer(1, 6))
    f.append(KeepTogether([
        Image(CHART, width=17.6*cm, height=17.6*cm * 2.72/7.0),
        Paragraph("Gambar 1. Perbandingan F1-Score seluruh model; warna menunjukkan "
                  "keluarga algoritma.", st_cap),
    ]))

    f.append(PageBreak())
    f += hdr("4. ANALISIS DAN DISKUSI")
    f.append(Paragraph(
        "<b>SVM %s performa tertinggi.</b> Kernel RBF (C = 10,0) memetakan ruang fitur 127 dimensi ke "
        "ruang berdimensi lebih tinggi sehingga terbentuk <i>hyperplane</i> non-linear yang mampu "
        "memisahkan pola lengkungan jari dan jarak antar-sendi secara fleksibel. SVM Linear menyusul "
        "sangat dekat (F1 97,02%%), menandakan bahwa setelah standardisasi sebagian besar kelas sudah "
        "memiliki separabilitas linear yang tinggi. Kernel Polynomial (degree 3) tertinggal pada 92,23%% "
        "dengan waktu latih terlama (1,705 s) akibat kompleksitas komputasi perkalian polinomial di "
        "dimensi 127." % MDASH, st_body))
    f.append(Paragraph(
        "<b>KNN %s akurasi tinggi dan sangat efisien.</b> Nilai k = 3 memberikan hasil terbaik "
        "(F1 96,12%%), diikuti k = 5 (95,47%%) dan k = 7 (95,03%%). Penurunan seiring bertambahnya k "
        "mengindikasikan batas antar-kelas bersifat lokal (<i>tight local clusters</i>): k yang terlalu "
        "besar justru menyerap tetangga dari kelas lain yang gesturnya mirip. Pembobotan "
        "<i>weights = distance</i> membantu mereduksi ambiguitas pada sampel di perbatasan kelas, "
        "sementara sifat <i>lazy learner</i> membuat waktu latih hanya 0,065 s." % MDASH, st_body))
    f.append(Paragraph(
        "<b>Naive Bayes %s kegagalan asumsi independensi.</b> F1-Score hanya berkisar 30,36%%%s42,64%%. "
        "Asumsi independensi kondisional (P(x1, ..., xd | y) = produk P(xi | y)) dilanggar secara ekstrem: "
        "koordinat sendi pada satu jari bergerak bersamaan secara anatomis, sehingga estimasi probabilitas "
        "posterior terdistorsi. Distribusi koordinat yang tidak sepenuhnya Gaussian memperburuk Gaussian NB, "
        "sedangkan binarisasi pada Bernoulli/Complement NB membuang informasi sudut dan jarak presisi yang "
        "dibutuhkan untuk membedakan huruf." % (MDASH, "\u2013"), st_body))
    f.append(Paragraph(
        "<b>Trade-off akurasi vs waktu.</b> Untuk akurasi maksimal, SVM RBF adalah pilihan utama "
        "(F1 97,91%%, inferensi %s0,45 ms/frame). Untuk kebutuhan <i>real-time</i> dengan sumber daya "
        "terbatas, KNN k = 3 menawarkan latih 0,065 s dan inferensi %s0,14 ms/frame dengan akurasi tetap "
        "di atas 96%%." % (PM, PM), st_body))
    f.append(Paragraph(
        "<b>Pola kesalahan per kelas.</b> Berdasarkan confusion matrix, huruf dengan gestur khas seperti "
        "<b>A, B, C, I, L, O,</b> dan <b>Y</b> hampir tidak pernah salah diklasifikasikan oleh SVM RBF maupun "
        "KNN. Konfusi utama muncul pada pasangan <b>M%sN</b> (posisi jempol terselip di antara telunjuk dan "
        "jari tengah), <b>K%sV</b> (dua jari tegak dengan selisih posisi ibu jari yang tipis), serta "
        "<b>D%sZ</b> (telunjuk tegak yang hanya dapat dibedakan melalui lintasan gerak dinamis). "
        "Pada Naive Bayes terjadi dispersi kesalahan yang sangat lebar karena posterior tiap kelas "
        "terestimasi tidak wajar." % (MDASH, MDASH, MDASH), st_body))
    f.append(Spacer(1, 2))
    f.append(KeepTogether([
        Image(os.path.join(OUT, "08_cm_best_per_algo.png"), width=17.2*cm, height=17.2*cm * 1180/3584),
        Paragraph("Gambar 2. Confusion matrix model terbaik tiap algoritma (SVM RBF, KNN k=3, "
                  "NB Bernoulli); diagonal gelap menandakan prediksi benar.", st_cap),
    ]))

    f += hdr("5. MODEL TERBAIK PER ALGORITMA")
    f.append(tabel_terbaik())
    f.append(Paragraph("Tabel 2. Ringkasan varian terbaik pada masing-masing keluarga algoritma.", st_cap))

    f += hdr("6. KESIMPULAN DAN REKOMENDASI")
    f.append(Paragraph(
        "1. <b>Model terbaik</b> adalah <b>SVM RBF</b> (akurasi dan F1-Score 97,91%%; CV F1 97,09%% %s "
        "0,40%%), dengan urutan performa <b>SVM (RBF/Linear) &gt; KNN (k = 3, 5, 7) &gt;&gt; Naive Bayes</b>."
        % PM, st_bul))
    f.append(Paragraph(
        "2. Representasi 127 fitur <i>MediaPipe Hand Landmark</i> terbukti kompak dan efektif: pelatihan "
        "selesai dalam hitungan detik tanpa memerlukan arsitektur CNN yang berat.", st_bul))
    f.append(Paragraph(
        "3. <b>Rekomendasi penerapan:</b> gunakan SVM RBF yang diekspor via <i>joblib</i>/ONNX Runtime "
        "untuk aplikasi interaktif, atau KNN k = 3 pada perangkat bermemori terbatas. Pengembangan lanjutan "
        "dapat menambahkan fitur <i>wrist angle</i> dan rasio jarak antar-ujung jari, serta memperluas ke "
        "gestur dinamis dengan LSTM/GRU.", st_bul))
    f.append(Paragraph(
        "4. <b>Keterbatasan:</b> dataset bersifat statis sehingga huruf yang melibatkan gerakan (mis. J dan Z "
        "dalam BISINDO) berpotensi ambigu; evaluasi juga belum mencakup variasi latar belakang, pencahayaan "
        "ekstrem, maupun okolusi antar-tangan yang umum terjadi pada kondisi nyata.", st_bul))
    f.append(Spacer(1, 3))
    f.append(Paragraph(
        "<i>Artefak:</i> <font color='#2E75B6'>dataset_bisindo_landmarks.npz</font>, "
        "<font color='#2E75B6'>comparison_table.csv</font>, "
        "<font color='#2E75B6'>final_results.json</font>, serta 18 visualisasi pada folder "
        "<font color='#2E75B6'>output_complete/</font> (distribusi dataset, heatmap, radar chart, "
        "trade-off, metrik per kelas, dan confusion matrix).",
        S("f", fontSize=7.2, leading=9, textColor=GREY, alignment=TA_JUSTIFY)))

    doc.build(f)
    print("OK ->", PDF_PATH)


if __name__ == "__main__":
    build()

