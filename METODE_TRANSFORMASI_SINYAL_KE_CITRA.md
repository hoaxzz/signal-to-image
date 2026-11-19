# KOMPENDIUM LENGKAP: METODE TRANSFORMASI SINYAL 1D KE CITRA 2D PADA PEMROSESAN BIOSINYAL & EEG DENOISING

Dokumen ini menyajikan **tinjauan komprehensif seluruh metode dan teknik akademis untuk mengubah sinyal deret waktu satu dimensi (1D time-series/EEG) menjadi representasi citra dua dimensi (2D image)**, lengkap dengan mekanisme kerja, formulasi konseptual, kelebihan, kelemahan, metode rekonstruksi/inversinya, serta rujukan paper ilmiah terkait.

---

## DAFTAR ISI
1. [Latar Belakang: Mengapa Mengubah Sinyal 1D Menjadi Citra 2D?](#1-latar-belakang-mengapa-mengubah-sinyal-1d-menjadi-citra-2d)
2. [Taksonomi Rumpun Metode Transformasi](#2-taksonomi-rumpun-metode-transformasi)
3. [Rumpun 1: Representasi Waktu-Frekuensi (Time-Frequency Representations)](#3-rumpun-1-representasi-waktu-frekuensi-time-frequency-representations)
   - 3.1 Short-Time Fourier Transform (STFT) / Spektrogram
   - 3.2 Continuous Wavelet Transform (CWT) / Skalogram
   - 3.3 Stationary / Discrete Wavelet Transform (SWT/DWT/WPD) 2D Coeffs
   - 3.4 Wigner-Ville Distribution (WVD)
4. [Rumpun 2: Dynamic Fields & Ruang Geometri (Matriks Korelasi NxN)](#4-rumpun-2-dynamic-fields--ruang-geometri-matriks-korelasi-nxn)
   - 4.1 Gramian Angular Summation / Difference Fields (GASF / GADF)
   - 4.2 Markov Transition Field (MTF)
   - 4.3 Recurrence Plot (RP) & Phase Space Reconstruction
5. [Rumpun 3: Pemetaan Spasial Topografi (Topographic Spatial Mapping)](#5-rumpun-3-pemetaan-spasial-topografi-topographic-spatial-mapping)
   - 5.1 EEG Topomap (Azimuthal Equidistant Projection)
6. [Rumpun 4: Spatial Folding & Space-Filling Curves](#6-rumpun-4-spatial-folding--space-filling-curves)
   - 6.1 Raster-Scan Reshaping (Matriks 2D Naif)
   - 6.2 Morton / Z-Order Curve
   - 6.3 Kurva Ruang Hilbert (Hilbert Space-Filling Curve)
7. [Rumpun 5: Metode Hilbert RGBA PNG Lossless Bit-Packing (Metode Anda)](#7-rumpun-5-metode-hilbert-rgba-png-lossless-bit-packing-metode-anda)
8. [Tabel Perbandingan Komprehensif Antar Seluruh Metode](#8-tabel-perbandingan-komprehensif-antar-seluruh-metode)
9. [Daftar Rujukan Paper Akademik Formal](#9-daftar-rujukan-paper-akademik-formal)

---

## 1. LATAR BELAKANG: MENGAPA MENGUBAH SINYAL 1D MENJADI CITRA 2D?

Meskipun sinyal EEG aslinya direkam sebagai deret waktu 1 dimensi ($x[t]$), komunitas peneliti *deep learning* dan pemrosesan sinyal sangat gencar mengubahnya ke citra 2 dimensi karena beberapa alasan mendasar:
1. **Kekuatan Arsitektur Computer Vision 2D:** Model 2D seperti **U-Net, ResNet, DenseNet, ConvNeXt, dan Vision Transformer (ViT)** telah mengalami penyempurnaan pesat selama lebih dari satu dekade dengan *inductive bias* spasial yang kaya.
2. **Konteks Multi-Resolusi Spasial:** Melalui operasi konvolusi 2D bertingkat (*downsampling/upsampling*), model dapat mengenali hubungan antar sampel pada skala lokal (detail gelombang mikrovolt) maupun skala global (ritme gelombang otak secara keseluruhan).
3. **Akselerasi Perangkat Keras Modern:** Prosesor modern (GPU tensor cores, TPU, hingga DSP pada mikrokontroler embedded) dioptimalkan secara mendasar untuk operasi matriks 2D perkalian titik (*GEMM - General Matrix Multiply*), bukan operasi sekuensial deret 1D.

---

## 2. TAKSONOMI RUMPUN METODE TRANSFORMASI

Secara global di berbagai literatur ilmiah, metode pengubahan sinyal ke citra dikelompokkan ke dalam **5 rumpun utama**:

```
                         METODE TRANSFORMASI SINYAL 1D KE CITRA 2D
                                            │
    ┌────────────────┬──────────────────────┼──────────────────────┬────────────────┐
    ▼                ▼                      ▼                      ▼                ▼
[1. Waktu-      [2. Dynamic Fields      [3. Topografi         [4. Spatial      [5. Lossless
 Frekuensi]      & Ruang Geometri]       Elektroda Spasial]    Folding & SFC]   Bit-Packing]
 • STFT           • GASF / GADF          • EEG Topomap         • Raster 2D      • Hilbert RGBA
 • CWT            • MTF                  • Azimuthal Proj.     • Morton/Z-Curve   PNG (32x32)
 • SWT / WPD      • Recurrence Plot                            • Hilbert Curve
 • WVD
```

---

## 3. RUMPUN 1: REPRESENTASI WAKTU-FREKUENSI (TIME-FREQUENCY REPRESENTATIONS)

Rumpun ini adalah metode paling klasik dan paling umum dalam pemrosesan sinyal. Sinyal diproyeksikan ke matriks 2 dimensi di mana **sumbu horizontal = waktu ($t$)** dan **sumbu vertikal = frekuensi ($f$)**.

### 3.1 Short-Time Fourier Transform (STFT) / Spektrogram
* **Konsep Kerja:** Sinyal dibagi menjadi segmen-segmen pendek yang saling tumpang tindih (*overlapping windows*), lalu setiap segmen dihitung spektrum frekuensinya menggunakan Fast Fourier Transform (FFT).
  $$\text{STFT}\{x[t]\}(m, \omega) = \sum_{n=-\infty}^{\infty} x[n] w[n - m] e^{-j \omega n}$$
* **Format Citra:** Matriks magnitudo spektrogram $|X(m, \omega)|$ atau spektrogram daya $|X(m, \omega)|^2$ yang diplot sebagai citra intensitas atau colormap RGB.
* **Mekanisme Inversi ke Sinyal 1D:**
  * Menggunakan **Inverse STFT (ISTFT)** dengan metode *Overlap-Add (OLA)*.
  * **Kelemahan Kritis:** Jika model hanya memproses spektrogram magnitudo, informasi **fase ($\angle X$) hilang**. Untuk merekonstruksi kembali sinyal 1D, peneliti harus mengestimasi fase menggunakan algoritma iteratif seperti *Griffin-Lim Algorithm (GLA)* yang lambat dan menghasilkan distorsi fase.
* **Rujukan Paper Penting:**
  * **Yan & Wu (2022)** — *"A time-frequency denoising method for single-channel event-related EEG"*, *Frontiers in Neuroscience*, 16:991136. (Mengubah EEG ke STFT 2D, membuang artefak dengan filter citra Non-Local Means, lalu di-inversi via ISTFT).

---

### 3.2 Continuous Wavelet Transform (CWT) / Skalogram
* **Konsep Kerja:** Alih-alih menggunakan jendela waktu statis seperti STFT, CWT mengonvolusikan sinyal dengan fungsi *wavelet* induk (seperti Morlet atau Mexican Hat) yang diskalakan (*scale/frequency*) dan digeser (*translation/time*).
  $$W(a, b) = \frac{1}{\sqrt{|a|}} \int_{-\infty}^{\infty} x(t) \psi^*\left(\frac{t - b}{a}\right) dt$$
* **Karakteristik Citra:** Resolusi waktu tinggi pada frekuensi tinggi, dan resolusi frekuensi tinggi pada frekuensi rendah (mengatasi batas ketidakpastian Heisenberg-Gabor).
* **Mekanisme Inversi ke Sinyal 1D:**
  * Menggunakan **Inverse Continuous Wavelet Transform (ICWT)**.
  * **Kelemahan Kritis:** Rekonstruksi ICWT memerlukan komputasi integral ganda yang sangat berat dan rentan menimbulkan kesalahan numerik pada ujung batas sinyal (*boundary edge effects*).

---

### 3.3 Stationary / Discrete Wavelet Transform (SWT / DWT / WPD) Matriks Koefisien 2D
* **Konsep Kerja:** Sinyal didekomposisi secara berjenjang ke dalam sub-band frekuensi berbeda (Approximation & Detail coefficients) tanpa merusak keselarasan waktu (khususnya SWT yang bersifat *shift-invariant* tanpa downsampling).
* **Format Citra:** Koefisien sub-band ditumpuk secara vertikal membentuk kisi 2D berdimensi $(\text{Jumlah Band} \times \text{Panjang Waktu})$.
* **Mekanisme Inversi ke Sinyal 1D:**
  * Rekonstruksi eksak menggunakan **Inverse SWT (ISWT)** atau **Inverse Wavelet Packet (IWPD)**.
* **Rujukan Paper Penting:**
  * **SWTT Framework (2026)** — *"SWTT: A Hybrid Stationary Wavelet Transform and TCN-Transformer Network for EEG Artifact Removal"*, *MDPI Electronics*.
  * **Roy et al. (2022)** — *"Automatic Muscle Artifacts Identification and Removal from Single-Channel EEG Using Wavelet Transform with Meta-Heuristically Optimized Non-Local Means Filter"*, *Sensors / PMC9030243*.

---

### 3.4 Wigner-Ville Distribution (WVD)
* **Konsep Kerja:** Distribusi energi kuadratik waktu-frekuensi tanpa menggunakan jendela terpisah.
* **Kelebihan:** Memberikan resolusi waktu dan frekuensi paling tajam.
* **Kelemahan Kritis:** Menghasilkan komponen interferensi semu (*cross-term artifacts*) saat sinyal memuat lebih dari satu komponen frekuensi, sehingga sangat jarang dipakai untuk restorasi langsung.

---

## 4. RUMPUN 2: DYNAMIC FIELDS & RUANG GEOMETRI (MATRIKS KORELASI NxN)

Rumpun ini dipelopori oleh penelitian pemetaan deret waktu ke citra untuk mengeksploitasi korelasi non-linear dan dinamika sistem kaotik (*chaotic dynamical systems*).

### 4.1 Gramian Angular Summation / Difference Fields (GASF / GADF)
* **Pencetus:** Zhiguang Wang & Tim Oates (AAAI 2015) — *"Imaging Time-Series to Improve Classification and Imputation"*.
* **Konsep Kerja:**
  1. Sinyal deret waktu $X = \{x_1, x_2, \dots, x_N\}$ dinormalisasi ke rentang $[-1, 1]$.
  2. Dipetakan ke sistem koordinat polar: $\phi = \arccos(\tilde{x}_i)$.
  3. Dihitung hubungan antar dua titik waktu sembarang ($t_i, t_j$) menggunakan penjumlahan/pengurangan kosinus kuasi-sudut:
     $$\text{GASF}_{(i,j)} = \cos(\phi_i + \phi_j) = \tilde{x}_i \tilde{x}_j - \sqrt{1 - \tilde{x}_i^2}\sqrt{1 - \tilde{x}_j^2}$$
     $$\text{GADF}_{(i,j)} = \sin(\phi_i - \phi_j) = \sqrt{1 - \tilde{x}_i^2}\tilde{x}_j - \tilde{x}_i \sqrt{1 - \tilde{x}_j^2}$$
* **Format Citra:** Matriks kuadratik berukuran $N \times N$.
* **Kelemahan Fatal untuk Denoising:**
  * **Ukuran Meledak:** Sinyal 512 titik membengkak menjadi citra $512 \times 512 = 262.144$ piksel.
  * **Inversi Kurang Stabil:** Menemukan kembali deret asli dari matriks diagonal memerlukan eliminasi multi-solusi ambigu trigonometri.

---

### 4.2 Markov Transition Field (MTF)
* **Pencetus:** Wang & Oates (2015).
* **Konsep Kerja:**
  1. Amplitudo sinyal dibagi menjadi $Q$ kelompok kuantil (*quantile bins*), di mana setiap kuantil bertindak sebagai kondisi (*state*) Markov $q_k$.
  2. Dihitung probabilitas transisi $w_{i,j}$ dari kondisi $q_i$ ke kondisi $q_j$ sepanjang waktu.
  3. Matriks MTF memetakan probabilitas transisi antar titik waktu: $M_{i,j} = W_{q(x_i), q(x_j)}$.
* **Karakteristik:** Citra menunjukkan seberapa sering lompatan amplitudo terjadi antar detik yang berbeda.
* **Sifat Inversi:** Bersifat probabilistik non-injektif (tidak bisa di-inversi balik menjadi sinyal asal secara eksak).

---

### 4.3 Recurrence Plot (RP) & Phase Space Reconstruction
* **Pencetus:** Eckmann et al. (1987); diadaptasi ke deep learning oleh Hatami et al. (2018).
* **Konsep Kerja:** Berdasarkan Teorema Keterikatan Takens (*Takens' Embedding Theorem*), deret waktu diproyeksikan ke ruang fase $m$-dimensi dengan keterlambatan waktu $\tau$:
  $$\vec{s}_i = [x_i, x_{i+\tau}, \dots, x_{i+(m-1)\tau}]$$
  Matriks RP dibentuk dari jarak Euclidean antar vektor keadaan:
  $$R_{i,j} = \Theta(\varepsilon - \|\vec{s}_i - \vec{s}_j\|)$$
  (di mana $\Theta$ adalah fungsi Heaviside step dan $\varepsilon$ adalah ambang jarak batas).
* **Format Citra:** Citra biner/tekstur $N \times N$ dengan pola garis diagonal yang mencerminkan sifat siklis atau kekacauan sistem saraf otak.
* **Sifat Inversi:** Tidak reversibel ke amplitudo tegangan asli.

---

## 5. RUMPUN 3: PEMETAAN SPASIAL TOPOGRAFI (TOPOGRAPHIC SPATIAL MAPPING)

Khusus sinyal EEG yang direkam dari banyak elektroda di kulit kepala (multi-channel EEG).

### 5.1 EEG Topomap (Azimuthal Equidistant Projection)
* **Paper Pionir:** Pouya Bashivan et al. (ICLR 2016) — *"Learning Representations from EEG with Deep Recurrent-Convolutional Neural Networks"*.
* **Konsep Kerja:**
  1. Daya pita frekuensi (Theta: 4-7 Hz, Alpha: 8-13 Hz, Beta: 14-30 Hz) diekstrak dari setiap elektroda (misal 32 atau 64 kanal).
  2. Posisi fisik 3D elektroda diproyeksikan ke bidang 2D menggunakan proyeksi polar azimut (*Azimuthal Equidistant Projection*).
  3. Ruang kosong antar sensor diisi menggunakan interpolasi spasial permukaan (*Clough-Tocher 2D Spline* atau *Radial Basis Function*).
  4. Tiga pita daya frekuensi dipetakan ke 3 kanal warna citra (R = Theta, G = Alpha, B = Beta).
* **Format Citra:** Foto kepala berwarna menyerupai peta panas (*heatmap*).
* **Penggunaan Utama:** Sangat hebat untuk klasifikasi keadaan mental, deteksi emosi, atau BCI, **tetapi tidak dapat digunakan untuk memulihkan sinyal gelombang mikrovolt asli**.

---

## 6. RUMPUN 4: SPATIAL FOLDING & SPACE-FILLING CURVES

Rumpun ini melipat deret waktu 1D ke kisi 2D berukuran kompak $M \times M$ tanpa meledakkan dimensi menjadi $N \times N$.

### 6.1 Raster-Scan Reshaping (Matriks 2D Naif)
* **Konsep Kerja:** Sinyal panjang $N$ langsung di-reshape baris demi baris menjadi matriks $M \times M$ (seperti membaca teks buku dari kiri ke kanan lalu lompat ke baris berikutnya).
* **Paper yang Menggunakan:**
  * **Mashhadi et al. (2020)** — *"Deep learning denoising for EOG artifacts removal from EEG signals"*, arXiv:2009.08809. (Me-reshape EEG ke citra grayscale $80 \times 80$ atau $512 \times 256$, lalu diproses dengan 2D U-Net).
  * **Liu et al. (2026)** — *"E2CAR: An Efficient 2D-CNN Framework for Real-Time EEG Artifact Removal on Edge Devices"*, arXiv:2602.09035.
* **Kelemahan Fatal (Edge Discontinuity):**
  Sampel di ujung baris pertama ($x_M$) dan sampel di awal baris kedua ($x_{M+1}$) adalah tetangga langsung di domain waktu ($|t_1 - t_2| = 1$), namun setelah disusun menjadi matriks, keduanya berada di ujung yang berseberangan secara spasial. Konvolusi 2D kernel $3 \times 3$ menganggap keduanya tidak bertetangga, sehingga merusak pemahaman pola gelombang.

---

### 6.2 Kurva Morton (Z-Order Curve)
* **Konsep Kerja:** Memetakan indeks linier 1D ke koordinat 2D dengan menyisipkan bit biner koordinat secara berselang-seling (*bit-interleaving*):
  $$d = \text{interleave}(x, y)$$
* **Kelemahan:** Mengalami lompatan spasial yang kasar berbentuk huruf "Z" di setiap pergantian kuadran fraktal (*discontinuous boundary jumps*).

---

### 6.3 Kurva Ruang Hilbert (Hilbert Space-Filling Curve)
* **Konsep Kerja:** Kurva fraktal kontinu yang memetakan interval 1D ke kisi 2D orthogonal berukuran $M \times M$ ($M = 2^k$).
* **Sifat Topologis Unggul (Preservasi Lokalitas Spasial):**
  Untuk setiap titik waktu berurutan $t$ dan $t+1$, jarak Manhattan antar koordinat 2D selalu tepat satu:
  $$\|\mathbf{p}_t - \mathbf{p}_{t+1}\|_1 = |r_t - r_{t+1}| + |c_t - c_{t+1}| = 1$$
* **Keuntungan:** Tidak ada titik yang melompat; sampel yang bertetangga di waktu selalu bersebelahan di kisi 2D. Kernel konvolusi $3 \times 3$ mampu menyerap korelasi lokal temporal dengan sempurna.

---

## 7. RUMPUN 5: METODE HILBERT RGBA PNG LOSSLESS BIT-PACKING (METODE ANDA)

Ini adalah **kontribusi ilmiah utama penelitian Anda**, yang memecahkan kelemahan seluruh metode konvensional di atas.

```
+───────────────────────────────────────────────────────────────────────────────+
|               PIPELINE HILBERT RGBA PNG & DUAL-CHANNEL RES-UNET               |
|                                                                               |
| Sinyal 1D (float32) ──► IEEE 754 Bit-Packing ──► Hilbert RGBA PNG (32x32x4)    |
|                                                          │                    |
|                                                    Decode ke Grid             |
|                                                          ▼                    |
| Estimasi 1D Bersih  ◄── Inversi Hilbert ◄── Model 2D Res-UNet (32x32x2)       |
|    (Bit-Exact)                                 [Ch 0: Signal, Ch 1: Mask]     |
+───────────────────────────────────────────────────────────────────────────────+
```

### Mekanisme Kerja:
1. **IEEE 754 Byte-Exact Casting:**
   Setiap sampel amplitudo sinyal `float32` (4 byte) dikemas langsung ke 4 kanal warna piksel `RGBA` (`uint8`):
   $$\text{Byte 0} \to \text{Red}, \quad \text{Byte 1} \to \text{Green}, \quad \text{Byte 2} \to \text{Blue}, \quad \text{Byte 3} \to \text{Alpha}$$
2. **Penataan Jalur Hilbert 2D ($32 \times 32$):**
   Byte tersebut diletakkan pada grid $32 \times 32 = 1024$ piksel mengikuti jalur Kurva Hilbert orde $k=5$.
3. **Penyimpanan Metadata Terpadu (*Self-Contained*):**
   Piksel Hilbert terakhir (indeks 1023) digunakan untuk menyimpan panjang sinyal asli $N$ sebagai integer `uint32`. File PNG tidak membutuhkan file konfigurasi pendamping.
4. **Dual-Channel Grid Input untuk Neural Network:**
   Saat diumpankan ke model 2D Res-UNet, data diekstrak menjadi tensor berdimensi $(2, 32, 32)$:
   * **Kanal 0 (Bidang Amplitudo):** Nilai kontinu mikrovolt EEG pada jalur Hilbert.
   * **Kanal 1 (Masker Biner):** Nilai `1.0` untuk sampel aktif dan `0.0` untuk area padding, mencegah bias konvolusi pada area kosong.

---

## 8. TABEL PERBANDINGAN KOMPREHENSIF ANTAR SELURUH METODE

| Metode Transformasi | Dimensi Citra | Sifat Inversi | Preservasi Fase & Waktu | Beban Komputasi | Cocok untuk EEG Denoising? |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **STFT (Spektrogram)** | $F \times T$ | *Lossy* (Fase hilang/perlu GLA) | Rendah (Heisenberg-Gabor limit) | Sedang | Kurang Ideal (Fase sinyal terdistorsi) |
| **CWT (Skalogram)** | $S \times T$ | *Lossy* (Batas tepi numerik) | Sedang | Tinggi | Kurang Ideal (Inversi berat) |
| **SWT / WPD 2D Grid** | $B \times T$ | Eksak Numerik | Tinggi (Time-aligned) | Sedang | Cukup Baik (Dipakai pada SWTT 2026) |
| **GASF / GADF** | $N \times N$ | Semi-*Lossy* (Ambiguitas sudut) | Sedang | Sangat Berat ($O(N^2)$) | Tidak (Dimensi meledak) |
| **Markov Field (MTF)** | $N \times N$ | Tidak Reversibel | Rendah (Probabilistik) | Sangat Berat ($O(N^2)$) | Tidak (Hanya untuk klasifikasi) |
| **Recurrence Plot (RP)** | $N \times N$ | Tidak Reversibel | Sangat Rendah | Sangat Berat ($O(N^2)$) | Tidak (Hanya untuk analisis dinamika) |
| **EEG Topomap** | $H \times W \times 3$ | Tidak Reversibel | Tidak Ada (Hanya spasial kepala) | Sedang | Tidak (Hanya untuk klasifikasi BCI) |
| **Raster Reshaping Naif** | $M \times M$ | *Bit-Exact* | Rusak pada batas baris (*edge jump*) | Ringan ($O(N)$) | Sub-optimal (Model bingung di ujung baris) |
| **Hilbert RGBA PNG (Metode Anda)** | **$32 \times 32 \times 4$** | **100% Bit-Exact ($E=0.0$)** | **Sangat Tinggi ($\|\Delta p\|_1 = 1$)** | **Paling Ringan ($O(N)$)** | **Sangat Ideal (Preservasi lokalitas + Nol Error)** |

---

## 9. DAFTAR RUJUKAN PAPER AKADEMIK FORMAL

1. **Wang, Z., & Oates, T. (2015).** *Imaging time-series to improve classification and imputation.* Proceedings of the 24th International Joint Conference on Artificial Intelligence (IJCAI/AAAI), 3939–3945.
2. **Mashhadi, N., Khuzani, A. Z., Heidari, M., & Khaledyan, D. (2020).** *Deep learning denoising for EOG artifacts removal from EEG signals.* arXiv preprint arXiv:2009.08809.
3. **Yan, W., & Wu, Y. (2022).** *A time-frequency denoising method for single-channel event-related EEG.* Frontiers in Neuroscience, 16, 991136. https://doi.org/10.3389/fnins.2022.991136
4. **Bashivan, P., Rish, I., Yeasin, M., & Codella, N. (2016).** *Learning Representations from EEG with Deep Recurrent-Convolutional Neural Networks.* International Conference on Learning Representations (ICLR).
5. **Pei, Y., et al. (2024).** *DTP-Net: Learning to Reconstruct EEG signals in Time-Frequency Domain by Multi-scale Feature Reuse.* IEEE Journal of Biomedical and Health Informatics, 28(5), 2662–2673.
6. **Liu, H., Cai, C., Zhao, X., & Li, L. (2026).** *E2CAR: An Efficient 2D-CNN Framework for Real-Time EEG Artifact Removal on Edge Devices.* arXiv preprint arXiv:2602.09035.
7. **Yang, B., Duan, K., Fan, C., Hu, C., & Wang, J. (2018).** *Automatic ocular artifacts removal in EEG using deep learning.* Biomedical Signal Processing and Control, 43, 148–158.
8. **Huang, T. Q., et al. (2023).** *2-D GCANet Applied to Denoise 1-D EEG Signals in Online Remote Teaching Scenarios.* Journal of Network Intelligence, 8(4), 1296–1303.
9. **Chuang, C. H., Chang, K. Y., Huang, C. S., & Jung, T. P. (2022).** *IC-U-Net: A U-Net-based Denoising Autoencoder Using Mixtures of Independent Components for Automatic EEG Artifact Removal.* NeuroImage, 263, 119586.
10. **Hatami, N., Gavet, Y., & Debayle, J. (2018).** *Classification of time-series images using deep convolutional neural networks.* Tenth International Conference on Machine Vision (ICMV 2017), SPIE Vol. 10696.
