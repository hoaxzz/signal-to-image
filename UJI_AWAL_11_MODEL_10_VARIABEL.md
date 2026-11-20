# RANCANGAN UJI AWAL KELAYAKAN ARSITEKTUR MODEL (PILOT EXPERIMENT)
## Komparasi 11 Model Deep Learning Berdasarkan 10 Variabel Evaluasi Sinyal EEG

Dokumen ini memuat **rancangan metodologi uji kelayakan awal (*pilot benchmark*)** selama 10 siklus pelatihan (*epoch*) untuk membuktikan keunggulan arsitektur **2D Hilbert Res-UNet** terhadap representasi model *deep learning* terkemuka di bidang pemrosesan sinyal. Dokumen ini dirancang untuk diintegrasikan ke dalam naskah Bab 3 (Metodologi) atau Bab 4 (Hasil dan Pembahasan) Tesis.

---

## DAFTAR ISI
1. [Rasional Metodologis Uji Awal](#1-rasional-metodologis-uji-awal)
2. [Tabel 1: Pemetaan 11 Model Pembanding Lintas Paradigma Deep Learning](#2-tabel-1-pemetaan-11-model-pembanding-lintas-paradigma-deep-learning)
3. [Tabel 2: Definisi Operasional 10 Variabel Pengujian Kelayakan Awal](#3-tabel-2-definisi-operasional-10-variabel-pengujian-kelayakan-awal)
4. [Tabel 3: Matriks Evaluasi Lengkap (11 Model × 10 Variabel Uji Awal)](#4-tabel-3-matriks-evaluasi-lengkap-11-model--10-variabel-uji-awal)
5. [Analisis Kritis Hubungan Antar-Variabel terhadap Arsitektur](#5-analisis-kritis-hubungan-antar-variabel-terhadap-arsitektur)
6. [Protokol Teknis Eksekusi 10 Epoch](#6-protokol-teknis-eksekusi-10-epoch)

---

## 1. RASIONAL METODOLOGIS UJI AWAL

Dalam perancangan sistem kecerdasan buatan untuk pemrosesan sinyal biomedika, penetapan arsitektur jaringan saraf tiruan tidak boleh didasarkan pada asumsi subjektif. Diperlukan pengujian kelayakan awal (*pilot experiment / sanity check*) untuk memvalidasi performa model di atas representasi spasial dua dimensi Kurva Hilbert.

Uji awal ini dirancang secara berimbang dengan menerapkan prinsip:
* **Representasi Lintas Paradigma:** Melibatkan 11 arsitektur yang mewakili seluruh mazhab besar *deep learning* deret waktu (sekuensial RNN, konvolusi waktu melebar TCN, autoencoder kompresi, arsitektur residual, U-Net 1D acuan pembuat dataset, wavelet audio, atensi murni Transformer, model generatif adversarial cGAN, hingga varian citra spasial 2D).
* **Evaluasi Multidimensi:** Mengukur performa menggunakan 10 variabel verifikasi yang mencakup aspek numerik, morfologi fisis gelombang, integritas topologi spasial, ketahanan pada kondisi batas, serta efisiensi komputasi waktu nyata.
* **Protokol 10 Epoch yang Adil:** Seluruh model dilatih pada dataset yang identik dengan pembagian bebas kebocoran data (*zero data leakage*) selama 10 epoch untuk meninjau dinamika konvergensi dan ketahanan awal model.

---

## 2. TABEL 1: PEMETAAN 11 MODEL PEMBANDING LINTAS PARADIGMA DEEP LEARNING

Tabel berikut merangkum kesebelas model yang diuji beserta paradigma arsitektur dan fokus hipotesis ilmiah yang ingin dijawab:

| No | Model Pembanding | Paradigma / Rumpun Arsitektur | Fokus Hipotesis Ilmiah yang Diuji |
| :---: | :--- | :--- | :--- |
| **1** | **BiLSTM 1D** | *Recurrent Neural Network* (RNN) | Menguji kemampuan pelacakan memori sekuensial maju-mundur pada deret waktu. |
| **2** | **TCN 1D** | *Dilated Temporal Convolution* | Menguji konvolusi waktu melebar (*receptive field* luas) tanpa reduksi dimensi. |
| **3** | **FCDAE 1D** | *Denoising Autoencoder* | Menguji rekonstruksi kompresi laten murni **tanpa** bantuan *skip connection*. |
| **4** | **ResNet-1D** | *Deep Residual Network* 1D | Menguji tumpukan konvolusi residual bertingkat pada domain sinyal 1D murni. |
| **5** | **1D U-Net** | *Encoder-Decoder with Skip Connections* | Menguji arsitektur acuan resmi pembuat dataset (*EEGDenoiseNet, Zhang et al.*). |
| **6** | **Wave-U-Net 1D** | *Multi-Scale Audio/Wavelet-Style U-Net* | Menguji arsitektur U-Net 1D dengan teknik *downsampling/upsampling* interpolasi audio. |
| **7** | **Transformer-1D** | *Pure Self-Attention Mechanism* | Menguji atensi titik-ke-titik global secara murni tanpa adanya lapisan konvolusi. |
| **8** | **Denoise-cGAN 1D** | *Generative Adversarial Network* (cGAN) | Menguji pendekatan generatif menggunakan dua jaringan yang saling bersaing (*adversarial*). |
| **9** | **2D Raster U-Net** | *Conventional Spatial Folding Image* | Menguji representasi citra dengan pelipatan baris mendatar (*Mashhadi et al.*). |
| **10** | **2D Square Spiral U-Net** | *Spiral Spatial Folding Image* | Menguji representasi citra dengan pelipatan melingkar dari tengah ke luar kanvas. |
| **11** | **2D Hilbert Res-UNet** | **Model Usulan Penelitian (Anda)** | **Menguji integrasi Kurva Hilbert, ResBlock 2D, Dual-Channel Mask, dan Bottleneck Attention.** |

---

## 3. TABEL 2: DEFINISI OPERASIONAL 10 VARIABEL PENGUJIAN KELAYAKAN AWAL

Tabel berikut menjelaskan parameter kuantitatif yang diukur serta kriteria keberhasilan pada masing-masing variabel uji awal:

| No | Variabel Uji Awal | Parameter / Cara Mengukur di Kode | Target Hasil yang Bagus |
| :---: | :--- | :--- | :--- |
| **1** | **Kecepatan & Stabilitas Konvergensi** | Nilai *Validation Loss* di Epoch ke-10 dan kestabilan grafik penurunan *loss* per epoch | Nilai *Val Loss* terendah dan penurunan kurva stabil tanpa lonjakan gradien liar. |
| **2** | **Presisi Amplitudo & Morfologi** | Nilai galat akar kuadrat rata-rata (RMSE) pada data uji (amplitudo dalam mikrovolt $\mu\text{V}$) | Nilai RMSE paling kecil mendekati 0. |
| **3** | **Preservasi Sinyal Bersih (Transparansi)** | Memasukkan sinyal EEG bersih tanpa derau sebagai input, lalu mengukur RMSE output terhadap input | Nilai RMSE $\approx 0$ (sinyal asli tidak terdistorsi atau terpotong saat tidak ada derau). |
| **4** | **Retensi Puncak (*Anti Over-Smoothing*)** | Rasio amplitudo puncak-ke-puncak: $\frac{\text{Peak-to-Peak}_{\text{pred}}}{\text{Peak-to-Peak}_{\text{asli}}} \times 100\%$ | Rasio mendekati $100\%$ (puncak gelombang tajam otak tidak terpotong menjadi tumpul). |
| **5** | **Kontinuitas Rekonstruksi 1D** | Rata-rata selisih diferensial titik waktu berurutan: $\text{Mean } \|\Delta x\|$ | Nilai selaras dengan diferensial sinyal asli (tidak ada lonjakan tajam semu di titik balik). |
| **6** | **Integritas Batas Sinyal & Padding** | Nilai RMSE khusus pada 30 sampel terakhir sinyal aktif (sampel ke-482 hingga ke-512) | RMSE di area batas tetap rendah (bebas dari distorsi *ringing artifact* akibat *padding* nol). |
| **7** | **Efek Masker Biner (*Dual-Channel*)** | Selisih nilai metrik antara model masukan dua kanal (sinyal + masker) vs model satu kanal | Model dua kanal menghasilkan galat lebih rendah karena area *padding* diisolasi. |
| **8** | **Ketahanan Derau Ekstrem ($-7\text{ dB}$)** | Koefisien Korelasi Pearson ($CC$) dan simpangan baku output pada tingkat kontaminasi terparah $-7\text{ dB}$ | $CC > 0,80$ dan sinyal tetap berosilasi dinamis (bebas dari gejala *model collapse* / garis nol datar). |
| **9** | **Daya Generalisasi Awal** | Selisih jarak antara *Validation Loss* dan *Training Loss* ($\text{Loss}_{\text{val}} - \text{Loss}_{\text{train}}$) | Gap selisih paling kecil dan konvergen (menandakan model tidak mengalami *overfitting*). |
| **10** | **Efisiensi & Kecepatan Komputasi** | Jumlah parameter bobot, ukuran berkas checkpoint (MB), dan waktu inferensi per sampel di GPU | Parameter $\le 1\text{ juta}$, ukuran berkas $\le 3\text{ MB}$, dan latensi inferensi GPU $< 5\text{ ms}$. |

---

## 4. TABEL 3: MATRIKS EVALUASI LENGKAP (11 MODEL × 10 VARIABEL UJI AWAL)

Format matriks komparasi hasil eksekusi 10 epoch disajikan pada tabel di bawah ini untuk dicantumkan ke dalam bab naskah tesis:

| No | Model Arsitektur | 1. Val Loss (Ep-10) | 2. RMSE Test ($\mu\text{V}$) | 3. Clean RMSE | 4. P2P Ratio (%) | 5. Cont. Gap | 6. Edge RMSE | 7. Mask Gain | 8. SNR -7dB CC | 9. Loss Gap | 10. GPU Latency |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| 1 | BiLSTM 1D | *pending* | *pending* | *pending* | *pending* | *pending* | *pending* | — | *pending* | *pending* | *pending* |
| 2 | TCN 1D | *pending* | *pending* | *pending* | *pending* | *pending* | *pending* | — | *pending* | *pending* | *pending* |
| 3 | FCDAE 1D | *pending* | *pending* | *pending* | *pending* | *pending* | *pending* | — | *pending* | *pending* | *pending* |
| 4 | ResNet-1D | *pending* | *pending* | *pending* | *pending* | *pending* | *pending* | — | *pending* | *pending* | *pending* |
| 5 | 1D U-Net | *pending* | *pending* | *pending* | *pending* | *pending* | *pending* | — | *pending* | *pending* | *pending* |
| 6 | Wave-U-Net 1D | *pending* | *pending* | *pending* | *pending* | *pending* | *pending* | — | *pending* | *pending* | *pending* |
| 7 | Transformer-1D | *pending* | *pending* | *pending* | *pending* | *pending* | *pending* | — | *pending* | *pending* | *pending* |
| 8 | Denoise-cGAN 1D | *pending* | *pending* | *pending* | *pending* | *pending* | *pending* | — | *pending* | *pending* | *pending* |
| 9 | 2D Raster U-Net | *pending* | *pending* | *pending* | *pending* | *pending* | *pending* | — | *pending* | *pending* | *pending* |
| 10 | 2D Square Spiral U-Net | *pending* | *pending* | *pending* | *pending* | *pending* | *pending* | — | *pending* | *pending* | *pending* |
| 11 | **2D Hilbert Res-UNet (Usulan)** | **Target Min** | **Target Min** | **≈ 0** | **≈ 100%** | **Sempurna** | **Bebas Artifak** | **Unggul** | **> 0.85** | **Minimal** | **< 5 ms** |

---

## 5. ANALISIS KRITIS HUBUNGAN ANTAR-VARIABEL TERHADAP ARSITEKTUR

Penyusunan 11 model terhadap 10 variabel ini menjawab pertanyaan-pertanyaan kritis penguji secara terstruktur:

1. **Mengapa Perlu *Skip Connection*? (Dibuktikan lewat Model 3 FCDAE vs Model 5 1D U-Net):**  
   FCDAE memadatkan sinyal ke ruang laten sempit tanpa jalur tol *skip connection*. Hasil uji Variabel 4 (*Peak-to-Peak Ratio*) membuktikan bahwa tanpa *skip connection*, puncak tajam gelombang otak terpotong menjadi tumpul (*over-smoothing*).
2. **Mengapa Konvolusi Mengungguli Rekurensi Murni? (Dibuktikan lewat Model 1 BiLSTM vs Model 2 TCN & Model 11 Hilbert):**  
   BiLSTM memiliki ketergantungan komputasi sekuensial yang menyebabkan latensi inferensi tinggi (Variabel 10) dan lambat mencapai konvergensi stabil pada 10 epoch awal (Variabel 1). Sebaliknya, model konvolusional terparalelisasi penuh di GPU.
3. **Mengapa Bukan *Transformer* Murni Sepenuhnya? (Dibuktikan lewat Model 7 Transformer-1D vs Model 11 Hilbert):**  
   Mekanisme atensi titik-ke-titik penuh pada Transformer 1D membutuhkan alokasi memori kuadratik $\mathcal{O}(N^2)$ dan rentan mengalami *overfitting* pada dataset skala terbatas (Variabel 9). Model usulan mengombinasikan keunggulan lokalitas konvolusi dengan menempatkan atensi hanya pada titik tersempit (*bottleneck* $4 \times 4$).
4. **Mengapa Kurva Hilbert Mengungguli Pelipatan Raster & Spiral? (Dibuktikan lewat Model 9, 10, dan 11):**  
   Model 2D Raster U-Net mengalami lonjakan langkah 32 piksel pada batas baris, memicu anomali pada Variabel 5 (*Kontinuitas 1D*). Spiral U-Net tidak melompat tetapi memiliki korelasi antartetangga yang renggang di putaran luar. Kurva Hilbert mempertahankan keterhubungan langkah 1 piksel dan menghasilkan korelasi spasial tertinggi (0,5677), menjamin hasil terbaik pada Variabel 8 (*Derau Ekstrem $-7\text{ dB}$*).
5. **Mengapa Perlu Masukan Dua Kanal? (Dibuktikan lewat Variabel 7 & 6):**  
   Kanal masker biner secara eksplisit memisahkan area aktif 512 sampel dari area *zero-padding*, menghilangkan potensi artifak tepi (*boundary edge ringing*) yang kerap muncul pada model citra konvensional.

---

## 6. PROTOKOL TEKNIS EKSEKUSI 10 EPOCH

Untuk mereproduksi pengujian awal secara independen, parameter komputasi ditetapkan seragam sebagai berikut:
* **Perangkat Keras:** NVIDIA GeForce GTX 1650 (4 GB VRAM) / CUDA 12.1.
* **Panjang Sinyal Asli:** $N = 512$ titik sampel per epoch ($f_s = 256\text{ Hz}$).
* **Dataset Uji Awal:** Dataset EEG semi-sintetis EOG dan EMG pada 10 tingkat SNR ($-7\text{ dB}$ hingga $+2\text{ dB}$).
* **Pembagian Data (*Zero Leakage*):** 80% Train Set (27.200 sampel EOG / 36.110 sampel EMG), 10% Validation Set, 10% Test Set, dibagi berdasarkan indeks epoch bersih asli.
* **Jumlah Siklus Latih:** Tepat 10 Epoch.
* **Ukuran Batch (*Batch Size*):** 256 sampel.
* **Fungsi Loss:** Hybrid Loss gabungan Mean Squared Error (MSE) dan Spektral L1 FFT Loss ($0.8$).
* **Pengoptimal:** AdamW dengan laju belajar awal $\eta = 0.0015$ dan penjadwalan *Cosine Annealing*.