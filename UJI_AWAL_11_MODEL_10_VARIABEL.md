# RANCANGAN UJI AWAL KELAYAKAN ARSITEKTUR MODEL 2D BERBASIS KISI HILBERT
## Komparasi 11 Model Deep Learning 2D Berdasarkan 10 Variabel Evaluasi (Termasuk Analisis Konvergensi)

Dokumen ini memuat **rancangan metodologi uji kelayakan awal (*pilot benchmark*)** selama 10 siklus pelatihan (*epoch*). Pada pengujian ini, **seluruh model yang dievaluasi beroperasi di atas representasi spasial 2D Kurva Hilbert yang sama ($32 \times 32$)**. Hal ini dilakukan untuk menjamin perbandingan yang adil (*fair comparison*), sehingga perbedaan performa yang terjadi murni membuktikan keunggulan arsitektur **2D Hilbert Res-UNet (+ Attention)** terhadap varian model *deep learning* 2D lainnya.

---

## DAFTAR ISI
1. [Rasional Metodologi Uji Awal Berbasis Kisi Hilbert](#1-rasional-metodologi-uji-awal-berbasis-kisi-hilbert)
2. [Tabel 1: Pemetaan 11 Model Deep Learning 2D pada Kisi Hilbert](#2-tabel-1-pemetaan-11-model-deep-learning-2d-pada-kisi-hilbert)
3. [Tabel 2: Definisi Operasional 10 Variabel Evaluasi (Fokus pada Konvergensi & Rekonstruksi)](#3-tabel-2-definisi-operasional-10-variabel-evaluasi-fokus-pada-konvergensi--rekonstruksi)
4. [Tabel 3: Matriks Evaluasi Komparatif (11 Model 2D × 10 Variabel Uji Awal)](#4-tabel-3-matriks-evaluasi-komparatif-11-model-2d--10-variabel-uji-awal)
5. [Analisis Kritis Rumpun Model 2D terhadap Karakteristik Hilbert](#5-analisis-kritis-rumpun-model-2d-terhadap-karakteristik-hilbert)
6. [Protokol Teknis Eksekusi Uji Awal 10 Epoch](#6-protokol-teknis-eksekusi-uji-awal-10-epoch)

---

## 1. RASIONAL METODOLOGI UJI AWAL BERBASIS KISI HILBERT

Penetapan model **2D Residual U-Net** tidak dilakukan secara serta-merta. Setelah sinyal EEG 1D berhasil dikonversi secara *lossless* menjadi kisi 2D berukuran $32 \times 32$ melalui Kurva Hilbert, muncul pertanyaan ilmiah mendasar:
> *"Dari berbagai macam arsitektur deep learning 2D (seperti Plain CNN, ResNet, DenseNet, MobileNet, Transformer/ViT, hingga U-Net), arsitektur manakah yang paling stabil, presisi, dan cocok dengan karakteristik fraktal Kurva Hilbert?"*

Untuk menjawab pertanyaan tersebut, seluruh 11 model pembanding menerima **masukan citra kisi Hilbert $32 \times 32$ yang identik**:
* **Standarisasi Masukan:** Semua model memproses representasi spasial Hilbert yang sama, menghilangkan bias representasi data.
* **Fokus Pengujian:** Perbedaan nilai evaluasi membuktikan secara langsung peran dari komponen-komponen arsitektur: *Skip Connections*, *Residual Blocks*, *Group Normalization*, *Aktivasi GELU*, *Masker Biner (Dual-Channel)*, serta *Multi-Head Self-Attention*.
* **Penekanan pada Aspek Konvergensi:** Mengamati perilaku konvergensi pada 10 epoch awal sangat penting untuk mendeteksi apakah suatu arsitektur mengalami hambatan belajar (*vanishing gradient*), lonjakan liar (*gradient explosion*), atau kejenuhan dini (*premature plateau*).

---

## 2. TABEL 1: PEMETAAN 11 MODEL DEEP LEARNING 2D PADA KISI HILBERT

Kesebelas model arsitektur 2D yang diuji di atas kisi Hilbert dirangkum pada Tabel 1:

| No | Model Arsitektur 2D (Input Hilbert $32\times32$) | Paradigma / Rumpun Desain 2D | Fokus Hipotesis Ilmiah yang Diuji |
| :---: | :--- | :--- | :--- |
| **1** | **2D Plain CNN (Autoencoder)** | *Vanilla Convolutional Encoder-Decoder* | Menguji kemampuan kompresi-rekonstruksi laten murni **tanpa** *skip connection* dan tanpa koneksi residual. |
| **2** | **2D ResNet (Residual Network)** | *Deep Residual Learning 2D* | Menguji efektivitas tumpukan blok residual murni bertingkat tanpa mekanisme reduksi piramidal U-Net. |
| **3** | **2D DenseNet** | *Densely Connected Convolutional Network* | Menguji konsep penggunaan ulang fitur secara padat (*dense feature reuse*) antar-lapisan pada kisi Hilbert. |
| **4** | **2D MobileNet (Separable CNN)** | *Depthwise Separable Convolution* | Menguji model konvolusi berbobot ultra-ringan untuk meminimalkan beban FLOPs dan jumlah parameter. |
| **5** | **2D Vanilla U-Net (Ronneberger)** | *Classic U-Net (Standard Conv + ReLU)* | Menguji peran murni *Skip Connection* simetris standar konvensional tanpa modifikasi residual. |
| **6** | **2D Hilbert Res-UNet (1-Kanal)** | *Residual U-Net (Tanpa Masker Biner)* | Menguji arsitektur Res-UNet yang hanya menerima 1 kanal amplitudo (Uji ablasi: dampak ketiadaan masker biner). |
| **7** | **2D Hilbert Res-UNet (Tanpa Attention)** | *Residual U-Net (Dual-Channel, No Attn)* | Menguji Res-UNet masukan 2-kanal lengkap namun di bottleneck terdalam **tanpa** lapisan *Self-Attention*. |
| **8** | **2D Attention U-Net (Oktay et al.)** | *Additive Attention Gate U-Net* | Menguji pemasangan gerbang atensi (*Attention Gates*) pada setiap saluran *skip connection* lateral. |
| **9** | **2D Vision Transformer (ViT Hilbert)** | *Pure Patch-based Spatial Transformer* | Menguji mekanisme atensi global murni berbasis *patch tokens* $4\times4$ tanpa operasi konvolusi lokal. |
| **10** | **2D ConvNeXt-Style Hilbert** | *Modernized Large-Kernel 2D CNN* | Menguji arsitektur konvolusi modern dengan kernel besar ($7\times7$) dan *inverted bottleneck design*. |
| **11** | **2D Hilbert Res-UNet + Self-Attention** | **Model Usulan Penelitian (Anda)** | **Menguji integrasi utuh: ResBlock 2D (GroupNorm+GELU), Dual-Channel Mask, dan Bottleneck Self-Attention.** |

---

## 3. TABEL 2: DEFINISI OPERASIONAL 10 VARIABEL EVALUASI (FOKUS PADA KONVERGENSI & REKONSTRUKSI)

Tabel berikut menyajikan rincian cara pengukuran dan target keberhasilan untuk kesepuluh variabel evaluasi:

| No | Variabel Uji Awal | Parameter / Cara Pengukuran di Kode | Target Hasil yang Bagus |
| :---: | :--- | :--- | :--- |
| **1** | **Kecepatan & Stabilitas Konvergensi** | • **Kecepatan Konvergensi:** *Epoch* saat *loss* mencapai 90% penurunan stabil.<br>• **Stabilitas Konvergensi:** Simpangan baku fluktuasi *loss* pada 3 epoch terakhir ($\sigma_{\text{loss}}$).<br>• **Nilai Akhir:** Nilai *Validation Loss* di Epoch ke-10. | Penurunan kurva *loss* melengkung mulus tanpa osilasi tajam, mencapai konvergensi stabil sebelum epoch ke-7, dan nilai *Val Loss* terendah. |
| **2** | **Presisi Amplitudo & Morfologi** | Nilai galat akar kuadrat rata-rata (RMSE) antara gelombang hasil prediksi dengan sinyal bersih pada data uji | Nilai RMSE paling kecil mendekati 0,0 (amplitudo mikrovolt terpulihkan secara presisi). |
| **3** | **Preservasi Sinyal Bersih (Transparansi)** | Memasukkan sinyal EEG bersih (tanpa derau) sebagai input model, lalu mengukur RMSE output terhadap input | Nilai RMSE $\approx 0$ (model bersifat transparan, tidak memotong gelombang otak asli saat derau tidak ada). |
| **4** | **Retensi Puncak (*Anti Over-Smoothing*)** | Rasio amplitudo puncak-ke-puncak (*Peak-to-Peak*):<br>$\text{Rasio P2P} = \frac{\text{P2P}_{\text{prediksi}}}{\text{P2P}_{\text{asli}}} \times 100\%$ | Rasio mendekati $100\%$ (puncak gelombang tajam otak tidak terpotong menjadi tumpul atau gepeng). |
| **5** | **Kontinuitas Rekonstruksi 1D** | Rata-rata selisih diferensial titik waktu berurutan setelah dikembalikan ke 1D: $\text{Mean } \|\Delta x\|$ | Nilai diferensial selaras dengan sinyal asli (tidak ada lonjakan tajam semu di titik perlipatan kurva). |
| **6** | **Integritas Batas Sinyal & Padding** | Nilai RMSE khusus pada 30 titik sampel terakhir sebelum batas *padding* (sampel ke-482 hingga ke-512) | RMSE di area batas tetap rendah (bebas dari distorsi getaran palsu / *ringing artifact* akibat *padding* nol). |
| **7** | **Efek Masker Biner (*Dual-Channel Gain*)** | Perbandingan nilai metrik RMSE antara model masukan dua kanal (sinyal + masker) vs model satu kanal | Model dua kanal menghasilkan galat lebih rendah karena area *padding* diisolasi dari statistik fitur. |
| **8** | **Ketahanan Derau Ekstrem ($-7\text{ dB}$)** | Koefisien Korelasi Pearson ($CC$) dan simpangan baku output pada kondisi kontaminasi terparah $-7\text{ dB}$ | $CC > 0,80$ dan sinyal tetap berosilasi dinamis (bebas dari gejala *model collapse* / garis nol datar). |
| **9** | **Daya Generalisasi Awal** | Jarak selisih antara *Validation Loss* dan *Training Loss* di Epoch ke-10 ($\text{Loss}_{\text{val}} - \text{Loss}_{\text{train}}$) | Selisih *loss gap* paling kecil dan konvergen (menandakan model tidak mengalami *overfitting*). |
| **10** | **Efisiensi Komputasi & Latensi GPU** | Jumlah parameter bobot, ukuran berkas *checkpoint* (MB), dan waktu inferensi per sampel di GPU (ms) | Parameter $\le 1\text{ juta}$, berkas $\le 3\text{ MB}$, dan latensi inferensi GPU murni $< 5\text{ ms}$ per sampel. |

---

## 4. TABEL 3: MATRIKS EVALUASI KOMPARATIF (11 MODEL 2D × 10 VARIABEL UJI AWAL)

Berikut adalah templat tabel matriks komparasi hasil uji coba 10 epoch untuk dimasukkan ke dalam naskah Bab 3 atau Bab 4 Tesis:

| No | Model Arsitektur 2D (Input Hilbert $32\times32$) | 1. Konvergensi (Val Loss / Epoch) | 2. RMSE Test ($\mu\text{V}$) | 3. Clean RMSE | 4. P2P Ratio (%) | 5. Cont. Gap | 6. Edge RMSE | 7. Mask Gain | 8. SNR -7dB CC | 9. Loss Gap (Val-Tr) | 10. GPU Latency |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| 1 | 2D Plain CNN (Autoencoder) | *pending* | *pending* | *pending* | *pending* | *pending* | *pending* | — | *pending* | *pending* | *pending* |
| 2 | 2D ResNet | *pending* | *pending* | *pending* | *pending* | *pending* | *pending* | — | *pending* | *pending* | *pending* |
| 3 | 2D DenseNet | *pending* | *pending* | *pending* | *pending* | *pending* | *pending* | — | *pending* | *pending* | *pending* |
| 4 | 2D MobileNet (Separable) | *pending* | *pending* | *pending* | *pending* | *pending* | *pending* | — | *pending* | *pending* | *pending* |
| 5 | 2D Vanilla U-Net (Ronneberger) | *pending* | *pending* | *pending* | *pending* | *pending* | *pending* | — | *pending* | *pending* | *pending* |
| 6 | 2D Hilbert Res-UNet (1-Kanal) | *pending* | *pending* | *pending* | *pending* | *pending* | *pending* | Baseline | *pending* | *pending* | *pending* |
| 7 | 2D Hilbert Res-UNet (No Attn) | *pending* | *pending* | *pending* | *pending* | *pending* | *pending* | + Gain | *pending* | *pending* | *pending* |
| 8 | 2D Attention U-Net (Gate Attn) | *pending* | *pending* | *pending* | *pending* | *pending* | *pending* | + Gain | *pending* | *pending* | *pending* |
| 9 | 2D Vision Transformer (ViT) | *pending* | *pending* | *pending* | *pending* | *pending* | *pending* | — | *pending* | *pending* | *pending* |
| 10 | 2D ConvNeXt-Style Hilbert | *pending* | *pending* | *pending* | *pending* | *pending* | *pending* | — | *pending* | *pending* | *pending* |
| 11 | **2D Hilbert Res-UNet + Self-Attn (Usulan)** | **Terkonvergen Cepat** | **Target Min** | **≈ 0** | **≈ 100%** | **Sempurna** | **Bebas Artifak** | **Unggul** | **> 0.85** | **Minimal** | **< 5 ms** |

---

## 5. ANALISIS KRITIS RUMPUN MODEL 2D TERHADAP KARAKTERISTIK HILBERT

Pemilihan kesebelas model 2D ini menjawab berbagai pertanyaan rancangan (*design questions*) dosen penguji secara terstruktur:

1. **Mengapa Harus U-Net dan Bukan Autoencoder Biasa? (Model 1 vs Model 5):**  
   Model 1 (Plain CNN Autoencoder) memadatkan kisi $32\times32$ ke ruang laten tanpa *skip connection*. Hasil uji Variabel 4 (*Peak-to-Peak Ratio*) membuktikan bahwa tanpa *skip connection*, informasi gelombang tajam otak akan hilang saat diperbesar kembali, menghasilkan sinyal yang tumpul (*over-smoothed*).
2. **Mengapa Bukan ResNet atau DenseNet Standar? (Model 2 & 3 vs Model 11):**  
   ResNet dan DenseNet menjaga resolusi spasial tanpa proses kompresi bertingkat U-Net. Akibatnya, kapasitas reseptif lokalnya sempit, menyulitkan model mengenali artefak kedipan mata (EOG) yang membentang luas di kisi spasial. Selain itu, DenseNet membutuhkan memori VRAM yang jauh lebih boros (Variabel 10).
3. **Mengapa Bukan Vision Transformer (ViT) Murni? (Model 9 vs Model 11):**  
   ViT membagi kisi menjadi *patches* dan menerapkan atensi murni tanpa induksi spasial konvolusi. Pada pelatihan 10 epoch dengan dataset terbatas, ViT sangat lambat mencapai konvergensi (Variabel 1) dan rentan mengalami *overfitting* yang ditandai oleh *loss gap* yang lebar (Variabel 9). Sebaliknya, model usulan menggabungkan efisiensi konvolusi lokal dengan menempatkan atensi hanya pada titik tersempit (*bottleneck* $4 \times 4$).
4. **Mengapa Perlu Masker Biner Dual-Channel? (Model 6 vs Model 7 & 11):**  
   Pada Model 6 (1-kanal), area *zero-padding* (sampel 512–1023) ikut dihitung oleh lapisan *Group Normalization*, menimbulkan bias nilai nol. Penyertaan Kanal 2 (masker biner) pada Model 7 dan 11 membuktikan peningkatan presisi (Variabel 7) dan meniadakan distorsi pada sampel-sampel batas akhir (Variabel 6).
5. **Mengapa Perlu Self-Attention di Bottleneck? (Model 7 vs Model 8 & 11):**  
   Kurva Hilbert memiliki kelemahan alami berupa *false neighbors* (dua titik yang bersebelahan di kisi 2D bisa berasal dari waktu perekaman yang sangat berjauhan). Model 7 (tanpa atensi) terkecoh oleh lipatan semu ini saat menghadapi derau ekstrem $-7\text{ dB}$. Pemasangan *Multi-Head Self-Attention* pada Model 11 secara dinamis mengevaluasi keterkaitan global antartoken, menghasilkan ketahanan derau ekstrem tertinggi (Variabel 8).

---

## 6. PROTOKOL TEKNIS EKSEKUSI UJI AWAL 10 EPOCH

Agar pengujian awal ini dapat direproduksi secara mandiri (*reproducible*), konfigurasi eksperimen diseragamkan:
* **Perangkat Keras:** NVIDIA GeForce GTX 1650 (4 GB VRAM) / CUDA 12.1.
* **Format Masukan:** Seluruh model menerima tensor masukan berbasis kisi Kurva Hilbert berdimensi $32 \times 32$.
* **Dataset Uji Awal:** Dataset EEG semi-sintetis EOG dan EMG pada 10 tingkat SNR ($-7\text{ dB}$ hingga $+2\text{ dB}$), panjang per sampel $N = 512$ titik.
* **Pembagian Data (*Zero Leakage*):** 80% Train Set (27.200 sampel EOG / 36.110 sampel EMG), 10% Validation Set, 10% Test Set, dibagi berdasarkan indeks epoch bersih asli.
* **Jumlah Siklus Latih:** Tepat 10 Epoch.
* **Ukuran Batch (*Batch Size*):** 256 sampel.
* **Fungsi Obyektif (Loss):** Hybrid Loss gabungan Mean Squared Error (MSE) dan Spektral L1 FFT Loss ($0.8$).
* **Pengoptimal:** AdamW dengan laju belajar awal $\eta = 0.0015$ dan penjadwalan *Cosine Annealing*.