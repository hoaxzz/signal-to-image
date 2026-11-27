# Lossless 1D-to-2D EEG Signal Transformation via Hilbert Space-Filling Curve and IEEE 754 RGBA Bit-Packing for Artifact Removal Using Bottleneck Self-Attention 2D Residual U-Net

**Judul Bahasa Indonesia:**  
**Transformasi Spasial Lossless Sinyal EEG 1D-ke-2D Melalui Kurva Ruang Hilbert dan Bit-Packing IEEE 754 RGBA untuk Penghilangan Artefak Menggunakan 2D Residual U-Net Berbasis Bottleneck Self-Attention**

**Target Jurnal Scopus (Q1/Q2):** *Biomedical Signal Processing and Control* (Elsevier) / *Computers in Biology and Medicine* (Elsevier) / *IEEE Journal of Biomedical and Health Informatics (JBHI)*

**Penulis (Authors):**  
[Nama Penulis Pertama]$^{a,*}$, [Nama Penulis Kedua]$^{a}$, [Nama Penulis Ketiga]$^{b}$  
$^{a}$ *Departemen Teknik Elektro / Teknik Biomedis, [Nama Universitas], [Kota], Indonesia*  
$^{b}$ *Kelompok Riset Pemrosesan Sinyal dan Kecerdasan Buatan, [Nama Institusi], [Kota], Indonesia*  
$^{*}$ *Corresponding author:* `[email_penulis@institusi.ac.id]`

---

## Highlights

Bagian *Highlights* ini merupakan komponen wajib pada manuskrip jurnal terindeks Scopus (khususnya penerbit Elsevier) yang merangkum temuan inti dan kontribusi metodologis penelitian dalam lima butir pernyataan ringkas.

* *Codec* Hilbert RGBA PNG memetakan sinyal EEG 1D `float32` ke kisi 2D $32 \times 32$ secara *100% bit-exact lossless* ($\text{RMSE} = 0.0$).
* Kurva Hilbert Orde-5 menjaga kontinuitas waktu dengan langkah Manhattan konstan ($\|\Delta \mathbf{p}\|_1 = 1$, $\text{Max Jump} = 1$ piksel).
* Seleksi lintas 26 metode transformasi membuktikan Hilbert RGBA PNG meraih korelasi spasial 2D tertinggi ($r_{2D} = 0.8228$).
* Tensor masukan 2-kanal (amplitudo + masker biner) dan metadata piksel terminal $(0, 31)$ mencegah bias *zero-padding*.
* Arsitektur `HilbertUNet` 2D dengan *Bottleneck Self-Attention* dan *Hybrid Loss* memulihkan morfologi serta spektrum EEG secara *real-time*.

---

## Abstract

Bagian abstrak ini disusun mengikuti struktur standar jurnal internasional terindeks Scopus yang memuat latar belakang masalah, metode yang diusulkan, hasil evaluasi kuantitatif utama, serta signifikansi klinis dan komputasional dalam satu kesatuan paragraf.

**Abstrak (Bahasa Indonesia):**  
Pemanfaatan arsitektur pembelajaran mendalam dua dimensi (*2D deep learning*) untuk penghilangan artefak (*denoising*) pada sinyal elektroensefalogram (EEG) satu dimensi (1D) selama ini dibatasi oleh degradasi fase pada spektrogram waktu-frekuensi, galat kuantisasi pada citra 8-bit, pembengkakan memori kuadratik pada matriks relasional, serta diskontinuitas batas baris pada pelipatan matriks naif (*raster-scan*). Penelitian ini mengusulkan kerangka kerja transformasi sinyal 1D-ke-2D yang bersifat deterministik, *100% bit-exact lossless*, mempertahankan lokalitas ketetanggaan waktu, dan mandiri (*self-contained*) dengan mengintegrasikan Kurva Ruang Hilbert (*Hilbert Space-Filling Curve*) Orde-5 ($32 \times 32$) serta pengemasan biner (*bit-packing*) IEEE 754 presisi tunggal (`float32`) ke dalam kontainer citra 4-kanal RGBA PNG. Setiap sampel sinyal 32-bit dipetakan secara bijektif ke dalam empat kanal warna `[R, G, B, A]` (`uint8`) tanpa pembulatan numerik, sedangkan koordinat terminal Hilbert $(0, 31)$ digunakan untuk menyematkan metadata panjang sinyal dinamis ($1 \le N \le 1023$) secara intrinsik. Untuk kebutuhan inferensi jaringan saraf, kontainer citra diekstraksi menjadi tensor spasial dua kanal (`[2, 32, 32]`) yang memadukan bidang amplitudo kontinu dengan masker biner sinyal aktif guna mengeliminasi bias *zero-padding*. Tensor tersebut selanjutnya diproses oleh model **2D Hilbert Residual U-Net (`HilbertUNet`)** berbobot ringan ($786.241$ parameter) yang dilengkapi dengan *Bottleneck Multi-Head Self-Attention* untuk mengatasi fenomena tetangga semu (*false neighbors*) antar-kuadran fraktal, serta diawasi menggunakan fungsi kerugian hibrida waktu-frekuensi (*Time-Frequency Hybrid Loss*). Pengujian komparatif terhadap 26 metode transformasi sinyal-ke-citra (12 paradigma global dan 14 varian *Circular Mapping*) pada dataset benchmark kontaminasi EOG dan EMG membuktikan bahwa Hilbert RGBA PNG merupakan satu-satunya metode yang secara simultan mencapai rekonstruksi tanpa galat ($\text{Bit-Exact} = 100\%$, $\text{RMSE} = 0.000000$, $\text{PRD} = 0.0000\%$), kontinuitas langkah waktu tanpa lompatan ($\text{Max Jump} = 1$ piksel), koherensi spasial tetangga 2D tertinggi di kelas *lossless* ($r_{2D} = 0.8228$, $\text{TV}_{\text{norm}} = 0.3572$), serta ukuran berkas ultra-ringkas ($2.10\text{ KB}$, latensi *round-trip* $0.375\text{ ms}$).

**Keywords:** Hilbert Space-Filling Curve; Lossless IEEE 754 Bit-Packing; EEG Artifact Removal; EOG and EMG Denoising; 2D Residual U-Net; Self-Attention.

---

## 1. Introduction

Bagian Pendahuluan (*Introduction*) ini menyajikan landasan ilmiah penelitian sesuai format standar jurnal terindeks Scopus (IMRaD). Uraian dimulai dari signifikansi fisiologis sinyal EEG dan permasalahan kontaminasi artefak okular serta muskular pada Sub-bab 1.1, dilanjutkan dengan tinjauan kritis terhadap literatur transformasi sinyal 1D-ke-2D beserta identifikasi empat celah penelitian utama (*research gaps*) pada Sub-bab 1.2, dan ditutup dengan perumusan kontribusi kebaruan ilmiah (*scientific novelty*) serta organisasi penulisan naskah pada Sub-bab 1.3.

### 1.1. Clinical Background and Challenges in EEG Artifact Removal
Sinyal elektroensefalogram (EEG) merupakan rekaman non-invasif aktivitas potensial pascasinaptik sinkron dari populasi neuron piramidal kortikal yang memiliki resolusi temporal pada skala milidetik. Dalam diagnosis neurologis maupun sistem antarmuka otak-komputer (*Brain-Computer Interface* / BCI), informasi fungsional otak terkode di dalam morfologi gelombang domain waktu serta distribusi daya spektral lintas lima pita frekuensi fisiologis utama: Delta ($\delta: 0.5\text{--}4\text{ Hz}$), Theta ($\theta: 4\text{--}8\text{ Hz}$), Alpha ($\alpha: 8\text{--}13\text{ Hz}$), Beta ($\beta: 13\text{--}30\text{ Hz}$), dan Gamma ($\gamma: >30\text{ Hz}$) [1], [2]. Meskipun demikian, amplitudo sinyal EEG di permukaan kulit kepala sangat lemah, yakni berkisar pada orde $10\text{--}100\,\mu\text{V}$, sehingga sangat rentan mengalami degradasi akibat kontaminasi artefak fisiologis non-serebral, terutama Elektrookulogram (EOG) dan Elektromiogram (EMG) [3].

Artefak EOG dipicu oleh pergeseran dipol korneo-retinal selama aktivitas kedipan maupun gerakan bola mata, yang membangkitkan transien beramplitudo tinggi ($>200\,\mu\text{V}$) pada pita frekuensi rendah ($<4\text{--}8\text{ Hz}$) sehingga menutupi ritme fisiologis $\delta$ dan $\theta$. Di sisi lain, artefak EMG bersumber dari potensial aksi unit motorik otot kranial, fasial, dan temporal yang menghasilkan derau stokastik pita lebar (*broadband noise*, $15\text{--}100+\text{ Hz}$). Kontaminasi EMG memiliki tumpang-tindih spektral (*spectral overlap*) secara langsung dengan gelombang $\beta$ dan $\gamma$ yang berkaitan erat dengan fungsi kognitif tingkat tinggi. Akibat sifat sinyal yang non-stasioner dan tumpang-tindih frekuensi tersebut, teknik penyaringan frekuensi linier klasik, pemfilteran adaptif, maupun transformasi wavelet berbasis nilai ambang statis sering kali gagal mengisolasi artefak tanpa merusak struktur fase dan morfologi gelombang otak asli, sedangkan metode *Independent Component Analysis* (ICA) tidak dapat bekerja secara langsung pada sistem perekaman EEG kanal tunggal (*single-channel*) atau perangkat *wearable* dengan jumlah kanal terbatas [4].

### 1.2. Related Work on 1D-to-2D Biosignal Transformations and Research Gaps
Kemajuan pesat arsitektur visi komputer dua dimensi (*2D Computer Vision*)—khususnya *2D Convolutional Neural Networks* (2D-CNN), arsitektur *encoder-decoder* *Residual U-Net*, dan mekanisme *Spatial Self-Attention*—telah menunjukkan kemampuan representasi fitur hierarkis dan pemodelan konteks spasial multiskala yang jauh melampaui jaringan konvolusi 1D konvensional [5], [6]. Keunggulan ini mendorong berkembangnya paradigma baru dalam pemrosesan biosinyal, yaitu memetakan deret waktu 1D ke dalam bidang matriks atau citra 2D, melakukan restorasi di domain spasial 2D, lalu merekonstruksi hasilnya kembali ke domain waktu 1D.

Berbagai penelitian terkini telah mengeksplorasi paradigma 1D-ke-2D tersebut untuk restorasi sinyal fisiologis. Mashhadi dkk. [7] membentuk citra *grayscale* 2D melalui pelipatan matriks langsung (*direct reshaping*) dari sinyal 1D dan menerapkan jaringan 2D U-Net untuk mereduksi derau. Yan dan Wu [8] memetakan sinyal EEG ke dalam representasi waktu-frekuensi dua dimensi untuk diproses menggunakan algoritma *Non-Local Means* (NLM) sebelum dikembalikan ke domain 1D melalui transformasi invers. Chuang dkk. [9] mengusulkan *IC-U-Net* yang mengolah citra peta komponen menggunakan *denoising autoencoder* berbasis U-Net, sementara Huang dkk. [10] mengadaptasi jaringan penghilang kabut citra 2D (*2D GCANet*) pada matriks sinyal EEG. Lebih lanjut, Pei dkk. [11] merancang *DTP-Net* yang memanfaatkan peta fitur waktu-frekuensi 2D dengan modul piramida temporal padat, Liu dkk. [12] mengembangkan kerangka kerja *E2CAR* yang menyusun segmen EEG 1D menjadi matriks citra 2D untuk mengoptimalkan akselerasi komputasi *2D-CNN* pada perangkat *edge*, dan kerangka *SWTT* [13] mengombinasikan matriks koefisien *Stationary Wavelet Transform* (SWT) 2D dengan modul *Transformer Self-Attention*.

Walaupun pendekatan-pendekatan tersebut membuktikan efektivitas tulang punggung jaringan 2D, kajian komprehensif terhadap empat rumpun metode transformasi 1D-ke-2D yang ada di literatur mengungkap **empat celah penelitian (*research gaps*) mendasar** ketika diterapkan pada tugas *restorasi sinyal mikrovolt*:

1. **Degradasi Fase dan Ketiadaan Identitas Biner pada Transformasi Waktu-Frekuensi (STFT & CWT):**  
   Transformasi berbasis *Short-Time Fourier Transform* (STFT) dan *Continuous Wavelet Transform* (CWT) dibatasi oleh prinsip ketidakpastian Heisenberg-Gabor ($\Delta t \cdot \Delta f \ge \frac{1}{4\pi}$). Konversi sinyal menjadi citra spektrogram magnitudo menyebabkan hilangnya informasi sudut fase ($\angle S(t, f)$), sehingga proses rekonstruksi melalui algoritma estimasi fase iteratif seperti Griffin-Lim menghasilkan distorsi gelombang yang parah. Penyimpanan koefisien kompleks maupun dekomposisi SWT multiskala memang memperkecil galat inversi, namun tetap menyisakan galat pembulatan *floating-point* ($\sim 10^{-7}$), memperbesar ukuran tensor, serta tidak menghasilkan kontainer citra standar yang *100% bit-exact*.
2. **Ledakan Memori Kuadratik dan Non-Invertibilitas pada *Dynamic Fields* ($N \times N$):**  
   Representasi *Gramian Angular Summation/Difference Fields* (GASF/GADF), *Markov Transition Fields* (MTF), dan *Recurrence Plots* (RP) memproyeksikan deret waktu berpanjang $N$ menjadi matriks relasional berukuran $N \times N$ ($\mathcal{O}(N^2)$). Untuk satu epoch EEG standar ($N = 512$), ukuran citra membengkak menjadi $512 \times 512 = 262.144$ piksel ($\approx 1.05\text{ MB}$ per epoch). Selain beban komputasi yang berat, MTF dan RP bersifat *non-invertible* (tidak memiliki fungsi invers analitik ke domain 1D), sehingga tidak dapat diterapkan untuk tugas *denoising*.
3. **Anomali Geometri Koordinat Kutub pada *Circular Mapping* (Singularitas Kutub dan Patahan Sudut):**  
   Pemetaan deret waktu ke kanvas melingkar (*polar* atau *spiral*) mengalami dua cacat topologi bawaan terhadap jendela konvolusi kartesius $3 \times 3$. Pertama, terjadi **Singularitas Kutub ($r \to 0$)** karena elemen luas diferensial $dA = r\,dr\,d\theta$ menyusut menuju nol di pusat lingkaran, memicu tabrakan antar-sampel pada piksel yang sama (*pixel collision*). Kedua, terjadi **Diskontinuitas Jahitan Sudut (*Angular Seam Jump*, $\theta: 2\pi \to 0$)** yang mempertemukan titik awal ($t=0$) dengan titik akhir ($t=N-1$) sehingga memunculkan tepi patahan artifisial. Di samping itu, lintasan filotaksis seperti *Vogel/Golden Spiral* memisahkan sampel berurutan dengan sudut emas $\approx 137.5^\circ$, yang merusak koherensi ketetanggaan spasial ($r_{2D} \approx 0$).
4. **Diskontinuitas Batas Baris pada Pelipatan Naif (*Raster/Morton*) dan Galat Kuantisasi 8-Bit:**  
   Penyusunan vektor 1D secara berbaris (*Raster-scan reshaping* [7], [12]) memutus kedekatan temporal setiap kali indeks beralih dari ujung kanan baris $(r, M-1)$ ke awal baris berikutnya $(r+1, 0)$ dengan lonjakan jarak spasial sebesar $M$ piksel (`Max Jump = 32 piksel` pada kisi $32 \times 32$), sementara kurva *Morton/Z-Order* menimbulkan lompatan diagonal antarkuadran (`Max Jump = 16 piksel`). Lebih jauh lagi, normalisasi amplitudo `float32` ke dalam citra *grayscale* 8-bit ($[0, 255]$) menimbulkan galat kuantisasi ireversibel ($\text{RMSE} \approx 1.09 \times 10^{-2}$) yang menghapus fluktuasi sub-mikrovolt EEG.

### 1.3. Proposed Framework and Scientific Contributions
Untuk menjawab keempat permasalahan fundamental di atas secara serentak, penelitian ini mengusulkan paradigma transformasi spasial *lossless* berbasis **Kurva Fraktal Hilbert Orde-5 ($32 \times 32$)** [14], [15] dan **pengemasan biner (*bit-packing*) IEEE 754 `float32` ke dalam kontainer citra RGBA PNG** [16], yang diintegrasikan dengan arsitektur **2D Hilbert Residual U-Net (`HilbertUNet`)**. Kontribusi ilmiah utama dari penelitian ini dirangkum sebagai berikut:

1. **Formulasi *Codec* Transformasi 1D-ke-2D yang *100% Bit-Exact Lossless* dan Mandiri (*Self-Contained*):**  
   Merancang pemetaan bijektif yang mengemas setiap sampel sinyal EEG `float32` (4 byte) secara langsung ke dalam empat kanal warna `[R, G, B, A]` (`uint8`) sepanjang jalur Kurva Hilbert $32 \times 32$ tanpa galat pembulatan ($\text{RMSE} = 0.000000$, $\text{PRD} = 0.0000\%$), sekaligus menyematkan metadata panjang sinyal dinamis ($1 \le N \le 1023$, `uint32`) pada piksel terminal Hilbert $(0, 31)$ sehingga berkas PNG dapat didekode secara mandiri tanpa metadata eksternal.
2. **Pembuktian Benchmark Komparatif Lintas 26 Metode Transformasi Sinyal-ke-Citra:**  
   Menyajikan evaluasi objektif tiga babak terhadap 12 metode transformasi global dan 14 varian *Circular Mapping*, yang membuktikan bahwa Hilbert RGBA PNG secara unik menyatukan keterbalikan biner 100%, kontinuitas langkah waktu Manhattan konstan ($\|\Delta \mathbf{p}\|_1 = 1$), koherensi spasial tetangga 2D tertinggi di kelas *lossless* ($r_{2D} = 0.8228$), dan ukuran berkas ultra-ringkas ($2.10\text{ KB}$, $499\times$ lebih hemat memori dibanding GASF/MTF).
3. **Representasi Tensor Masukan Dua Kanal Anti-Bias dan Arsitektur `HilbertUNet` dengan *Bottleneck Self-Attention*:**  
   Memisahkan representasi penyimpanan (`uint8` RGBA) dari representasi komputasi neural dengan mengekstraksi tensor masukan 2-kanal `[2, 32, 32]` (bidang amplitudo `float32` dan masker biner sinyal aktif) untuk mencegah distorsi *zero-padding*, serta memasang *Multi-Head Self-Attention* pada *bottleneck* $4 \times 4$ untuk menangkap ketergantungan global lintas 16 sub-kuadran Hilbert dan meredam efek tetangga semu (*false neighbors*).
4. **Supervisi Fungsi Kerugian Hibrida Waktu-Frekuensi (*Time-Frequency Hybrid Loss*):**  
   Mengintegrasikan kerugian *Mean Squared Error* (MSE) di domain waktu dengan *Orthonormal RFFT Magnitude $\ell_1$ Loss* di domain frekuensi guna memastikan pemulihan ketajaman puncak gelombang sekaligus preservasi spektrum ritme fisiologis otak pada kontaminasi EOG dan EMG lintas 10 tingkat SNR ($-7\text{ dB}$ hingga $+2\text{ dB}$).

Sisa naskah ini disusun sebagai berikut: Bagian 2 menguraikan material dataset, kerangka seleksi 26 metode transformasi, formulasi matematis *codec* Hilbert RGBA PNG, arsitektur `HilbertUNet` 2D, serta metrik evaluasi. Bagian 3 menyajikan hasil eksperimen dan analisis komparatif, dan Bagian 4 merangkum kesimpulan penelitian.

---

## 2. Materials and Methods

Bagian Material dan Metode (*Materials and Methods*) ini memaparkan protokol eksperimental, formulasi matematis, dan spesifikasi arsitektur jaringan secara rinci guna menjamin keterulangan ilmiah (*reproducibility*). Uraian dibagi ke dalam tujuh sub-bab yang saling berkaitan: gambaran umum pipeline sistem end-to-end (Sub-bab 2.1), konstruksi dataset kontaminasi EOG/EMG dan partisi bebas kebocoran data (Sub-bab 2.2), protokol seleksi komparatif 26 metode transformasi 1D-ke-2D (Sub-bab 2.3), formulasi Kurva Hilbert Orde-5 dan *bit-packing* IEEE 754 RGBA PNG (Sub-bab 2.4), ekstraksi tensor masukan dua kanal dan arsitektur `HilbertUNet` 2D (Sub-bab 2.5), fungsi kerugian hibrida waktu-frekuensi beserta protokol pelatihan (Sub-bab 2.6), serta prosedur rekonstruksi invers dan metrik evaluasi kuantitatif (Sub-bab 2.7).

### 2.1. System Pipeline Overview
Penelitian ini mengadopsi desain eksperimental kuantitatif untuk mengevaluasi kausalitas antara topologi pemetaan spasial 2D dan akurasi restorasi sinyal EEG. Sebagaimana diilustrasikan pada **Gambar 1**, alur kerja sistem (*end-to-end pipeline*) terdiri atas empat modul berurutan. Pertama, modul penyiapan data membangkitkan pasangan sinyal EEG bersih dan sinyal terkontaminasi artefak EOG serta EMG pada sepuluh tingkat SNR ($-7\text{ dB}$ hingga $+2\text{ dB}$) dengan pemisahan berbasis indeks epoch bersih. Kedua, modul *Lossless Hilbert RGBA PNG Encoder* melipat deret waktu 1D ($N$ sampel) ke dalam kisi 2D berukuran $32 \times 32$ mengikuti lintasan Kurva Hilbert Orde-5, mengemas 4 byte IEEE 754 `float32` ke kanal `[R, G, B, A]`, dan menyimpan panjang sinyal $N$ pada piksel terminal $(0, 31)$. Ketiga, modul *Dual-Channel Grid Extractor & `HilbertUNet` 2D* membongkar berkas PNG menjadi tensor `[2, 32, 32]` (amplitudo sinyal dan masker biner aktif) untuk dibersihkan oleh jaringan *Residual U-Net* 2D berfitur *Bottleneck Multi-Head Self-Attention*. Keempat, modul *Inverse Hilbert Decoder* membaca metadata panjang $N$ dari piksel $(0, 31)$ dan menelusuri balik lintasan Hilbert untuk merekonstruksi sinyal EEG 1D bersih $\hat{\mathbf{x}} \in \mathbb{R}^N$.

```
[ Sinyal EEG Terkontaminasi 1D: y ∈ ℝᴺ (float32, 1 ≤ N ≤ 1023) ]
                            │
                            ▼
┌───────────────────────────────────────────────────────────┐
│ MODUL 1: LOSSLESS HILBERT RGBA PNG ENCODER                │
│ • Pemetaan Kurva Hilbert Orde k=5: d ∈ [0..1023] → (r, c) │
│ • Bit-Packing IEEE 754 float32 (4 Byte) → [R, G, B, A]    │
│ • Piksel 0..N-1: Sinyal Aktif | N..1022: Zero-Padding     │
│ • Piksel Terminal d=1023 di (0, 31): Simpan uint32(N)     │
└───────────────────────────┬───────────────────────────────┘
                            │  Kontainer Fisik: 32 × 32 × 4 RGBA PNG
                            ▼
┌───────────────────────────────────────────────────────────┐
│ MODUL 2: DUAL-CHANNEL SPATIAL TENSOR EXTRACTION           │
│ • Tensor Masukan G ∈ ℝ^(2 × 32 × 32) (float32):           │
│   - Kanal 0: Nilai Amplitudo Sinyal EEG pada Jalur Hilbert│
│   - Kanal 1: Masker Biner Aktif (1.0 = Aktif, 0.0 = Pad)  │
└───────────────────────────┬───────────────────────────────┘
                            │
                            ▼
┌───────────────────────────────────────────────────────────┐
│ MODUL 3: RESTORASI SPASIAL 2D HILBERT RES-UNET            │
│ • Stem Conv 3×3 (2 → 24) + 3 Tingkat Enkoder ResBlock2D   │
│   (GroupNorm-8, GELU, Strided Conv 32×32 → 16×16 → 8×8)   │
│ • Bottleneck (4×4, 96 Ch): ResBlock2D + 4-Head Self-Attn  │
│ • 3 Tingkat Dekoder (ConvTranspose2d + Skip Connections)  │
│ • Supervisi: Hybrid Time-Domain MSE + 0.8 RFFT L1 Loss    │
└───────────────────────────┬───────────────────────────────┘
                            │  Kisi Prediksi Bersih: [1, 32, 32]
                            ▼
┌───────────────────────────────────────────────────────────┐
│ MODUL 4: INVERSE HILBERT DECODER & MULTI-DOMAIN EVALUATION│
│ • Baca Panjang N dari Piksel Terminal (0, 31) via <u4     │
│ • Telusuri Balik Indeks Hilbert 0..N-1 → Sinyal 1D x̂ ∈ ℝᴺ │
│ • Evaluasi: RMSE, RRMSE_t, RRMSE_s, CC, ΔSNR, & Latensi   │
└───────────────────────────────────────────────────────────┘
```
*Gambar 1: Diagram arsitektur pipeline transformasi lossless Hilbert RGBA PNG dan restorasi sinyal EEG berbasis `HilbertUNet` 2D.*

---

### 2.2. Dataset Construction, Power-Controlled Contamination, and Zero-Leakage Splitting
Evaluasi kuantitatif terhadap model restorasi sinyal memerlukan referensi sinyal bersih (*ground-truth*) yang diketahui secara pasti agar galat pemulihan amplitudo dan spektrum dapat diukur secara eksak. Oleh karena itu, sub-bab ini menjelaskan karakteristik sumber data EEG, skema pemasangan acak satu-ke-satu dengan artefak okular dan muskular, formulasi matematis pencampuran terkontrol daya pada sepuluh tingkat SNR, serta protokol partisi dataset berbasis indeks epoch yang bebas dari kebocoran data.

#### 2.2.1. Signal Specifications and One-to-One Random Pairing Scheme
Dataset eksperimen dibangun dari koleksi benchmark sinyal EEG bersih (*ground-truth*), rekaman artefak okular (EOG), dan rekaman artefak muskular (EMG) yang dicuplik pada frekuensi $f_s = 256\text{ Hz}$ dengan panjang epoch standar $N = 512$ sampel (durasi 2 detik per segmen) [4], ditambah variasi panjang dinamis $N \in \{128, 256, 384, 512, 1023\}$ untuk memvalidasi fleksibilitas ukuran sinyal pada kontainer citra.

Untuk mencegah model menghafal pola artefak yang berulang (*artifact overfitting*), diterapkan skema pemasangan acak satu-ke-satu (*one-to-one random pairing*) tanpa pengulangan antara epoch EEG bersih dan epoch artefak:
1. **Dataset Kontaminasi EOG:** Sebanyak $3.400$ segmen EEG bersih dipasangkan secara acak satu-ke-satu dengan $3.400$ segmen artefak EOG.
2. **Dataset Kontaminasi EMG:** Sebanyak $4.514$ segmen EEG bersih dipasangkan secara acak satu-ke-satu dengan $4.514$ segmen artefak EMG.

#### 2.2.2. Power-Controlled Additive Contamination across Ten SNR Levels
Setiap sinyal EEG terkontaminasi $\mathbf{y} = [y_0, y_1, \dots, y_{N-1}]^T \in \mathbb{R}^N$ dikonstruksi melalui superposisi linier antara sinyal EEG bersih $\mathbf{x} \in \mathbb{R}^N$ dan sinyal artefak $\mathbf{n} \in \mathbb{R}^N$ sebagaimana dinyatakan pada Persamaan (1):

$$\mathbf{y} = \mathbf{x} + \alpha \cdot \mathbf{n} \qquad (1)$$

Mengingat energi intrinsik dari setiap segmen sinyal bersih dan segmen artefak bervariasi antar-sampel, koefisien pengali skala artefak $\alpha \in \mathbb{R}^+$ dihitung secara adaptif untuk setiap pasangan sampel agar rasio daya sinyal terhadap derau tepat memenuhi target $\text{SNR}_{\text{in}}$ (dalam satuan dB) menurut Persamaan (2):

$$\text{SNR}_{\text{in}} = 10 \log_{10} \left( \frac{\sum_{t=0}^{N-1} x_t^2}{\sum_{t=0}^{N-1} (\alpha \cdot n_t)^2} \right) \implies \alpha = \frac{\|\mathbf{x}\|_2}{\|\mathbf{n}\|_2 \cdot 10^{\left(\text{SNR}_{\text{in}} / 20\right)}} \qquad (2)$$

di mana $\|\cdot\|_2$ menyatakan norma Euclidean $\ell_2$. Setiap pasangan sinyal unik diekspansi pada **10 tingkat $\text{SNR}_{\text{in}}$**, yaitu $\text{SNR}_{\text{in}} \in \{-7, -6, -5, -4, -3, -2, -1, 0, +1, +2\}\text{ dB}$, guna merepresentasikan kondisi kontaminasi klinis dari tingkat sangat berat ($-7\text{ dB}$) hingga ringan ($+2\text{ dB}$).

#### 2.2.3. Epoch-Index Partitioning Protocol for Zero Data Leakage
Kesalahan metodologis yang sering terjadi pada studi *denoising* EEG adalah melakukan pengacakan (*random shuffling*) setelah pasangan sinyal diekspansi ke berbagai level SNR, yang mengakibatkan gelombang EEG bersih yang sama muncul di himpunan latih dan himpunan uji pada level SNR berbeda (*data leakage*). Dalam penelitian ini, pemisahan data menjadi himpunan latih (*Training Set*, 80%), validasi (*Validation Set*, 10%), dan uji (*Testing Set*, 10%) dilakukan secara ketat **berdasarkan nomor indeks epoch bersih asli sebelum ekspansi multi-SNR**, sebagaimana dirinci pada **Tabel 1**.

*Tabel 1: Komposisi Dataset Kontaminasi EOG dan EMG dengan Protokol Partisi Bebas Kebocoran Data (Zero Data Leakage)*
| Jenis Artefak | Pasangan Dasar Unik | Rentang $\text{SNR}_{\text{in}}$ (10 Level) | Total Sampel Sinyal | Himpunan Latih (80% Epoch) | Himpunan Validasi (10% Epoch) | Himpunan Uji (10% Epoch) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **EOG (Okular)** | $3.400$ | $-7\text{ dB}$ s.d. $+2\text{ dB}$ | $34.000$ | $27.200$ (Indeks $0\text{--}2.719$) | $3.400$ (Indeks $2.720\text{--}3.059$) | $3.400$ (Indeks $3.060\text{--}3.399$) |
| **EMG (Muskular)** | $4.514$ | $-7\text{ dB}$ s.d. $+2\text{ dB}$ | $45.140$ | $36.110$ (Indeks $0\text{--}3.610$) | $4.510$ (Indeks $3.611\text{--}4.061$) | $4.520$ (Indeks $4.062\text{--}4.513$) |

---

### 2.3. Comparative Selection Framework across 26 Signal-to-Image Transformation Methods
Pemilihan representasi 2D untuk jaringan konvolusi harus didasarkan pada bukti komparatif yang terukur, baik dari segi keutuhan informasi sinyal maupun kualitas struktur ketetanggaan spasialnya. Oleh sebab itu, sub-bab ini menguraikan ruang lingkup 26 metode transformasi 1D-ke-2D yang diuji pada eksperimen pendahuluan, perumusan metrik topologi spasial dalam protokol seleksi tiga babak, serta hasil evaluasi empiris yang menjustifikasi pemilihan *Hilbert RGBA PNG*.

#### 2.3.1. Taxonomy of the 26 Evaluated Transformation Candidates
Eksperimen benchmark pendahuluan dilakukan secara seragam menggunakan sinyal uji acuan nyata dari `Dataset/EOG/linear_synthetic.npy` pada indeks ke-0 (`idx 0`, $\text{SNR}_{\text{in}} = -5\text{ dB}$, $N = 512$, `float32`). Sebanyak 26 kandidat metode yang dievaluasi mencakup dua kelompok besar:
1. **Benchmark Transformasi Sinyal-ke-Citra Global (12 Metode):** Mencakup rumpun *Space-Filling & Spatial Folding* (Hilbert RGBA PNG, Square Spiral RGBA PNG, Morton/Z-Order RGBA PNG, Raster RGBA PNG, dan Raster Grayscale 8-bit PNG [7]), rumpun *Time-Frequency* (SWT Haar 4-Level [13], STFT Complex Real+Imag+LogMag, STFT Magnitude + Griffin-Lim, dan CWT Morlet Scalogram), serta rumpun *Dynamic Fields* (GASF, MTF, dan Recurrence Plot).
2. **Benchmark Taksonomi *Circular Mapping* (14 Varian Melingkar):** Mencakup 9 metode *Kelompok 1: Native Variable-Length* (Direct Polar Plot, Archimedean Spiral, Fermat Spiral, Vogel/Golden Spiral, Symmetric Dot Pattern/SDP, Hilbert Polar Map, Poincaré Polar Map, Complex FFT Polar, dan Learnable Polar Map) serta 5 metode *Kelompok 2: Fixed-Grid* (Polar Sector/Grid, Concentric Ring, Adaptive Polar-GAF, Polar Recurrence Plot, dan Polar Scalogram) pada kanvas kutub $64 \times 64$.

#### 2.3.2. Three-Stage Objective Elimination Protocol and Spatial Topology Metrics
Untuk menentukan metode transformasi yang paling optimal bagi restorasi sinyal berbasis CNN/U-Net 2D, ditetapkan **Protokol Eliminasi Tiga Babak** beserta metrik uji topologi spasial sebagai berikut:

1. **Babak 1 — Keterbalikan Biner Sempurna (*100% Bit-Exact Invertibility*):**  
   Transformasi bolak-balik ($\mathbf{x} \xrightarrow{\text{Encode}} \mathbf{I} \xrightarrow{\text{Decode}} \hat{\mathbf{x}}$) wajib bersifat *lossless* tanpa galat pembulatan sedikit pun, yang diukur menggunakan *Root Mean Squared Error* rekonstruksi ($\text{RMSE}_{\text{rec}}$) dan *Percentage Root-mean-square Difference* ($\text{PRD}$) sesuai Persamaan (3):

$$\text{RMSE}_{\text{rec}} = \sqrt{\frac{1}{N} \sum_{t=0}^{N-1} \left( x_t - \hat{x}_t \right)^2} = 0.0, \quad \text{PRD} = \frac{\|\mathbf{x} - \hat{\mathbf{x}}\|_2}{\|\mathbf{x}\|_2} \times 100\% = 0.0\% \qquad (3)$$

2. **Babak 2 — Kontinuitas Langkah Temporal (*Unit Manhattan Step Continuity*):**  
   Setiap kenaikan satu langkah indeks waktu ($t \to t+1$) wajib dipetakan ke piksel tetangga terdekat yang bersisian secara ortogonal tanpa lompatan baris maupun lompatan diagonal. Ukuran langkah maksimum ($\text{Max Jump } L_1$) dirumuskan pada Persamaan (4):

$$\text{Max Jump } L_1 = \max_{0 \le t \le N-2} \left( |r_{t+1} - r_t| + |c_{t+1} - c_t| \right) = 1\text{ piksel} \qquad (4)$$

3. **Babak 3 — Koherensi Spasial 2D dan Kehalusan Gradien Konvolusi $3 \times 3$:**  
   Agar kernel konvolusi $3 \times 3$ dapat mengekstraksi fitur lokal secara efektif, piksel-piksel yang bersebelahan secara vertikal dan horizontal di kisi 2D harus memiliki korelasi amplitudo yang tinggi dan variasi gradien yang halus. Hal ini diukur melalui **Korelasi Spasial Tetangga 2D ($r_{2D}$)** pada himpunan pasangan piksel bertetangga 4-arah $\mathcal{E}_4$ dan **Normalized 2D Total Variation ($\text{TV}_{\text{norm}}$)** pada domain piksel aktif $\Omega_{\text{aktif}}$ sebagaimana didefinisikan pada Persamaan (5) dan Persamaan (6):

$$r_{2D} = \frac{\sum_{(u,v) \in \mathcal{E}_4} (I_u - \bar{I})(I_v - \bar{I})}{\sqrt{\sum_{(u,v) \in \mathcal{E}_4} (I_u - \bar{I})^2 \sum_{(u,v) \in \mathcal{E}_4} (I_v - \bar{I})^2}} \qquad (5)$$

$$\text{TV}_{\text{norm}} = \frac{\sum_{(r,c) \in \Omega_{\text{aktif}}} \left( |I_{r+1,c} - I_{r,c}| + |I_{r,c+1} - I_{r,c}| \right)}{2 \cdot |\Omega_{\text{aktif}}| \cdot \sigma_I} \qquad (6)$$

di mana $\bar{I}$ dan $\sigma_I$ masing-masing menyatakan rerata dan simpangan baku amplitudo pada area sinyal aktif.

#### 2.3.3. Empirical Selection Results and Justification of the Winning Representation
Hasil evaluasi kuantitatif pada **Tabel 2** menunjukkan bahwa dari 26 metode yang diuji, sebanyak 20 metode gugur pada Babak 1 karena bersifat *non-invertible* (MTF, RP, SDP, Poincaré, Polar RP), mengalami kerusakan fase atau distorsi interpolasi kutub (STFT Griffin-Lim $\text{PRD} = 146.22\%$, CWT Morlet $\text{PRD} = 83.08\%$, Complex FFT Polar $\text{PRD} = 74.88\%$, Polar Sector $\text{PRD} = 19.17\%$), terkena galat kuantisasi 8-bit (Raster Gray8 $\text{PRD} = 0.60\%$), atau mengalami pembengkakan memori ekstrem tanpa format PNG portabel (GASF $1.048.70\text{ KB}$). Pada Babak 2, metode Raster RGBA (`Max Jump = 32 px`), Morton RGBA (`Max Jump = 16 px`), Fermat Spiral (`Max Jump = 3 px`), dan Vogel Spiral (`Max Jump = 76 px`, $r_{2D} = -0.0001$) tereliminasi akibat terputusnya kontinuitas jalur waktu. Akhirnya, pada Babak 3, **Hilbert RGBA PNG ($32 \times 32 \times 4$)** mengungguli Square Spiral RGBA PNG secara telak dengan korelasi spasial $r_{2D} = \mathbf{0.8228}$ (berbanding $0.5409$) dan kehalusan $\text{TV}_{\text{norm}} = \mathbf{0.3572}$ (berbanding $0.6020$), sehingga ditetapkan sebagai metode transformasi utama.

*Tabel 2: Perbandingan Kuantitatif Perwakilan Metode Transformasi 1D-ke-2D pada Sinyal EEG Acuan Dataset Index 0 ($N = 512$)*
| Metode Transformasi | Rumpun / Kategori | Dimensi | Bit-Exact (%) | $\text{RMSE}_{\text{rec}}$ | $\text{PRD}$ (%) | Max Jump $L_1$ | $r_{2D} (\uparrow)$ | $\text{TV}_{\text{norm}} (\downarrow)$ | Ukuran Berkas | Waktu Round-Trip | Keputusan Seleksi |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **Hilbert RGBA PNG** | **Space-Filling** | **`32×32×4`** | **100%** | **0.000000** | **0.0000%** | **1 px** | **0.8228** | **0.3572** | **2.10 KB** | **0.375 ms** | **Terpilih (Unggul Mutlak)** |
| Square Spiral RGBA PNG | Spiral Folding | `32×32×4` | 100% | 0.000000 | 0.0000% | 1 px | 0.5409 | 0.6020 | 2.12 KB | 0.311 ms | Gugur Babak 3 ($r_{2D}$ rendah) |
| Morton / Z-Order RGBA | Space-Filling | `32×32×4` | 100% | 0.000000 | 0.0000% | 16 px | 0.8172 | 0.3818 | 2.01 KB | 0.227 ms | Gugur Babak 2 (Lompatan 16 px) |
| Raster RGBA PNG | Spatial Folding | `32×32×4` | 100% | 0.000000 | 0.0000% | 32 px | 0.7205 | 0.4655 | 1.99 KB | 0.220 ms | Gugur Babak 2 (Robekan 32 px) |
| Raster Gray8 PNG [7] | Spatial Folding | `32×32` | 0% | 0.010920 | 0.5996% | 32 px | 0.7205 | 0.4655 | 0.48 KB | 0.147 ms | Gugur Babak 1 & 2 (Lossy 8-bit) |
| STFT Mag + Griffin-Lim | Time-Frequency | `33×33` | 0% | 2.662886 | 146.2180% | — | 0.8550 | 0.1627 | 4.48 KB | 13.546 ms | Gugur Babak 1 (Distorsi fase) |
| CWT Morlet Scalogram | Time-Frequency | `32×512` | 0% | 1.512942 | 83.0751% | — | 0.9625 | 0.1107 | 65.66 KB | 1.823 ms | Gugur Babak 1 (Galat inversi) |
| GASF ($N \times N$) | Dynamic Field | `512×512` | 0% | 0.000002 | 0.00009% | — | 0.9911 | 0.0883 | 1048.70 KB | 3.182 ms | Gugur Babak 1 (Memori $500\times$) |
| MTF / Recurrence Plot | Dynamic Field | `512×512` | 0% | Non-Inv | Non-Inv | — | 0.8260 | 0.2406 | 1048.70 KB | — | Gugur Babak 1 (Non-Invertible) |
| Direct Polar / Sector | Circular Mapping | `64×64` | 0% | 0.133–0.349 | 7.31–19.17% | 4–8 px | 0.47–0.81 | 0.24–0.28 | 16.50 KB | 0.280 ms | Gugur Babak 1 (Singularitas/Seam) |
| Fermat / Vogel Spiral | Circular Mapping | `64×64` | 100% | 0.000000 | 0.0000% | 3–76 px | 0.16 / -0.00 | 0.54–0.63 | 16.50 KB | 0.250 ms | Gugur Babak 2 & 3 (Hamburan titik) |

---

### 2.4. Mathematical Formulation of Order-5 Hilbert Curve and IEEE 754 RGBA Bit-Packing
Untuk mewujudkan kontainer citra 2D yang mempertahankan presisi mikrovolt sinyal EEG tanpa galat pembulatan, pemetaan topologi fraktal harus dipadukan dengan skema serialisasi tingkat-bit (*bit-level serialization*). Sub-bab ini merumuskan tiga aspek matematis dari *codec* yang diusulkan, yakni pemetaan koordinat diskrit Kurva Hilbert Orde-5 pada kisi $32 \times 32$, injeksi *bit-packing* bilangan riil IEEE 754 `float32` ke dalam empat kanal warna RGBA, serta mekanisme penyisipan metadata panjang sinyal dinamis pada piksel terminal beserta algoritma formalnya.

#### 2.4.1. Discrete Hilbert Space-Filling Curve Topology
Misalkan $\mathcal{G}_k = \{0, 1, \dots, M-1\}^2 \subset \mathbb{Z}^2$ menyatakan kisi diskrit ortogonal berdimensi $M \times M$ dengan $M = 2^k$, di mana $k \in \mathbb{N}^+$ adalah orde iterasi fraktal. Dengan memilih orde $k = 5$, diperoleh ukuran kisi tetap $M = 32$ yang menyediakan kapasitas $M^2 = 1.024$ koordinat piksel. Kurva Hilbert mendefinisikan pemetaan bijektif $H_5$ dari indeks urutan waktu linear $d \in \mathcal{I}$ menuju pasangan koordinat baris dan kolom 2D $\mathbf{p}(d) = (r_d, c_d)$ sebagaimana dirumuskan pada Persamaan (7):

$$H_5: \mathcal{I} \to \mathcal{G}_5, \quad \mathcal{I} = \{0, 1, \dots, 1023\}, \quad d \mapsto \mathbf{p}(d) = (r_d, c_d) \qquad (7)$$

Untuk setiap indeks $d \in \mathcal{I}$, koordinat kartesian $(x, y)$ dibangun secara rekursif melintasi skala sub-kuadran $s = 2^m$ (untuk $m = 0, 1, \dots, k-1$) dengan mengekstraksi pasangan bit rotasi kuadran $(rx, ry)$ berdasarkan Persamaan (8):

$$rx = 1 \ \& \ \left\lfloor \frac{t}{2} \right\rfloor, \quad ry = 1 \ \& \ (t \oplus rx), \quad \text{dengan } t = \left\lfloor \frac{d}{4^m} \right\rfloor \qquad (8)$$

di mana $\&$ dan $\oplus$ masing-masing merupakan operator *bitwise* AND dan XOR. Pada setiap iterasi skala $s$, koordinat lokal $(x, y)$ ditransformasikan melalui operator refleksi-rotasi kuadran $\mathcal{R}(s, x, y, rx, ry)$ sesuai Persamaan (9):

$$\mathcal{R}(s, x, y, rx, ry) = \begin{cases} (s - 1 - y, \; s - 1 - x), & \text{jika } ry = 0 \text{ dan } rx = 1 \\ (y, \; x), & \text{jika } ry = 0 \text{ dan } rx = 0 \\ (x, \; y), & \text{jika } ry = 1 \end{cases} \qquad (9)$$

diikuti oleh pembaruan translasi kuadran $(x, y) \leftarrow (x + s \cdot rx, \; y + s \cdot ry)$, dan penetapan indeks matriks citra $(r_d, c_d) = (y, x)$. Konstruksi fraktal ini menjamin secara analitis bahwa setiap sampel waktu yang berurutan selalu bertetangga langsung dengan jarak Manhattan ($\ell_1$) bernilai tepat satu unit di seluruh kisi:

$$\|\mathbf{p}(d+1) - \mathbf{p}(d)\|_1 = |r_{d+1} - r_d| + |c_{d+1} - c_d| = 1, \quad \forall d \in \{0, 1, \dots, 1022\} \qquad (10)$$

#### 2.4.2. Byte-Exact IEEE 754 Single-Precision Bit-Packing into RGBA Channels
Menurut standar IEEE 754-2019 [16], setiap sampel sinyal EEG presisi tunggal $x_t \in \mathbb{R}$ (`float32`, *little-endian* `<f4`) menempati 32 bit memori yang terdiri atas 1 bit tanda $s \in \{0, 1\}$, 8 bit eksponen berbias $e \in \{0, \dots, 255\}$, dan 23 bit fraksi mantisa $m_i \in \{0, 1\}$ sesuai Persamaan (11):

$$x_t = (-1)^s \times 2^{e - 127} \times \left( 1 + \sum_{i=1}^{23} m_{23-i} 2^{-i} \right) \qquad (11)$$

Mengingat lebar memori 32-bit dari satu elemen `float32` ($4\text{ byte}$) sama persis dengan kapasitas satu piksel citra RGBA 8-bit ($4\text{ kanal} \times 8\text{ bit} = 32\text{ bit}$), didefinisikan operator *type-punning* memori bijektif $\Phi: \mathbb{R}_{\text{float32}} \to \{0, 1, \dots, 255\}^4$. Operator ini menyalin keempat byte penyusun $x_t$, yakni $[\beta_0, \beta_1, \beta_2, \beta_3]$, langsung ke kanal **Red (R), Green (G), Blue (B), dan Alpha (A)** pada koordinat Hilbert $(r_t, c_t)$ menurut Persamaan (12):

$$\mathbf{I}(r_t, c_t, :) = \Phi(x_t) = \begin{bmatrix} R(r_t, c_t) \\ G(r_t, c_t) \\ B(r_t, c_t) \\ A(r_t, c_t) \end{bmatrix} = \begin{bmatrix} \beta_0(x_t) \\ \beta_1(x_t) \\ \beta_2(x_t) \\ \beta_3(x_t) \end{bmatrix} = \text{reinterpret\_cast}\langle\text{uint8}[4]\rangle(x_t) \qquad (12)$$

di mana $\beta_0$ adalah *least significant byte* (LSB) mantisa dan $\beta_3$ adalah *most significant byte* (MSB) yang memuat bit tanda $s$ serta 7 bit eksponen tertinggi.

#### 2.4.3. Canvas Functional Zoning and Self-Contained Dynamic Length Metadata
Agar kanvas citra berukuran tetap $32 \times 32 \times 4$ dapat menyimpan sinyal dengan panjang dinamis $1 \le N \le 1023$ secara mandiri tanpa berkas header eksternal, ke-1.024 piksel sepanjang lintasan Hilbert dipartisi ke dalam tiga zona fungsional:
1. **Zona Sinyal Aktif ($0 \le d \le N-1$):** Memuat byte IEEE 754 `float32` dari sampel sinyal $x_0, x_1, \dots, x_{N-1}$.
2. **Zona *Zero-Padding* ($N \le d \le 1022$):** Diinisialisasi dengan byte nol `[0, 0, 0, 0]` (`uint8`).
3. **Zona Metadata Terminal ($d_{\text{term}} = M^2 - 1 = 1023$, terletak di pojok kanan atas $(r, c) = (0, 31)$):** Dicadangkan secara permanen untuk menyimpan panjang sinyal asli $N$ yang dikemas sebagai bilangan bulat tak-bertanda 32-bit *little-endian* (`uint32` / `<u4`) berdasarkan Persamaan (13):

$$\mathbf{I}(0, 31, :) = \Phi_{\text{meta}}(N) = \text{reinterpret\_cast}\langle\text{uint8}[4]\rangle(\text{uint32}(N)) \qquad (13)$$

Prosedur lengkap enkapsulasi sinyal 1D menjadi berkas *Hilbert RGBA PNG* beserta dekapsulasi inversnya dirangkum secara formal pada **Algorithm 1**.

```text
================================================================================
Algorithm 1: Lossless Hilbert RGBA PNG Encoding and Bit-Exact Decoding
================================================================================
Input (Encode) : 1D EEG signal x ∈ ℝ^N (float32), Grid size M = 32 (Order k = 5)
Output (Encode): Lossless RGBA PNG file / array I ∈ {0..255}^{M × M × 4}
--------------------------------------------------------------------------------
[Encoding Procedure: Signal 1D -> Hilbert RGBA PNG]
1: Assert 1 <= N <= M^2 - 1 (i.e., 1 <= N <= 1023)
2: (R, C) <- Precompute_Hilbert_Coordinates(M)          // Eqs. (7)-(9)
3: Initialize I <- Zeros(M, M, 4, dtype=uint8)
4: B_sig <- reinterpret_cast<uint8[N, 4]>(ascontiguous(x, dtype="<f4"))
5: For t = 0 to N - 1 do:
6:     I[R[t], C[t], 0:4] <- B_sig[t, 0:4]              // Eq. (12)
7: End For
8: I[R[M^2 - 1], C[M^2 - 1], 0:4] <- reinterpret_cast<uint8[4]>(uint32(N)) // Eq. (13) at (0, 31)
9: Save I to PNG container using lossless Deflate compression

[Decoding Procedure: Hilbert RGBA PNG -> Signal 1D]
10: Read RGBA array I ∈ {0..255}^{M × M × 4} from PNG container
11: N <- reinterpret_cast<uint32>(I[R[M^2 - 1], C[M^2 - 1], 0:4])[0]
12: Assert 1 <= N <= M^2 - 1
13: B_rec <- ascontiguous(I[R[0:N], C[0:N], 0:4])
14: x_hat <- reinterpret_cast<float32[N]>(B_rec)
15: Return x_hat (satisfying x_hat == x bit-for-bit, max|x - x_hat| = 0.0)
================================================================================
```

---

### 2.5. Dual-Channel Spatial Tensor Extraction and 2D Hilbert Res-UNet Architecture
Operasi konvolusi matriks pada jaringan saraf tiruan mensyaratkan ruang vektor yang bersifat linier terhadap penjumlahan dan perkalian skalar, sehingga byte mentah `uint8` RGBA tidak boleh dikonvolusikan secara langsung. Oleh karena itu, sub-bab ini menjelaskan transformasi kontainer RGBA menjadi tensor masukan dua kanal (*dual-channel input grid*) bertipe `float32` untuk mencegah distorsi aljabar dan bias *zero-padding*, kemudian merinci arsitektur jaringan **2D Hilbert Residual U-Net (`HilbertUNet`)** beserta mekanisme *Bottleneck Multi-Head Self-Attention*.

#### 2.5.1. Dual-Channel Neural Input Grid Formulation (`[2, 32, 32]`)
Jika filter konvolusi 2D diterapkan langsung pada nilai byte `uint8` RGBA, operasi kombinasi linier antar-piksel akan mencampurkan bit eksponen ($\beta_3$) dengan bit mantisa ($\beta_0, \beta_1, \beta_2$), yang secara matematis merusak nilai bilangan riil IEEE 754. Untuk menjaga kelinearan aljabar sekaligus mengisolasi area *zero-padding*, saat berkas PNG dibaca untuk inferensi model, array RGBA dibongkar kembali menjadi nilai `float32` dan diformat menjadi **tensor spasial dua kanal** $\mathbf{G} \in \mathbb{R}^{2 \times 32 \times 32}$ sebagaimana didefinisikan pada Persamaan (14) dan Persamaan (15):

$$\mathbf{G}_{0, r_d, c_d} = \begin{cases} x_d, & \text{jika } 0 \le d < N \quad (\text{Kanal 0: Bidang Amplitudo Sinyal EEG}) \\ 0.0, & \text{jika } N \le d \le 1023 \quad (\text{Area Zero-Padding \& Piksel Metadata}) \end{cases} \qquad (14)$$

$$\mathbf{G}_{1, r_d, c_d} = \begin{cases} 1.0, & \text{jika } 0 \le d < N \quad (\text{Kanal 1: Masker Biner Sinyal Aktif}) \\ 0.0, & \text{jika } N \le d \le 1023 \quad (\text{Masker Area Non-Aktif}) \end{cases} \qquad (15)$$

Kehadiran **Kanal 1 (Masker Biner Sinyal Aktif)** berfungsi sebagai indikator batas spasial eksplisit bagi filter konvolusi $3 \times 3$ dan lapisan *Group Normalization*, sehingga mencegah jaringan menginterpretasikan angka nol pada zona *padding* sebagai aktivitas gelombang otak yang sedang datar ($0\,\mu\text{V}$).

#### 2.5.2. 2D Hilbert Residual U-Net (`HilbertUNet`) and False-Neighbor Mitigation
Arsitektur **`HilbertUNet`** dirancang sebagai jaringan *encoder-decoder* residual simetris berbobot ringan dengan total **$786.241$ parameter** ($\approx 3.0\text{ MB}$), sebagaimana dirinci pada **Tabel 3**. Pilihan desain komponen utama arsitektur ini didasarkan pada karakteristik fisis sinyal EEG pada kisi Hilbert:

1. **Blok Residual dengan *Group Normalization* dan Aktivasi *GELU* (`ResBlock2D`):**  
   Setiap tahap pemrosesan dibangun dari blok residual $\mathbf{z}_{\text{out}} = \mathbf{z}_{\text{in}} + \mathcal{F}(\mathbf{z}_{\text{in}})$ yang memuat dua lapisan konvolusi $3 \times 3$, `GroupNorm(8)`, `Dropout2d(p=0.05)`, dan aktivasi `GELU`. *Group Normalization* (8 grup kanal) dipilih menggantikan *Batch Normalization* agar normalisasi fitur dihitung secara mandiri per sampel tanpa bergantung pada ukuran *batch*. Fungsi aktivasi **GELU** ($\text{GELU}(z) = z \cdot \Phi(z)$) digunakan menggantikan ReLU karena sinyal EEG berayun secara simetris di sekitar garis dasar nol (memiliki lembah negatif dan puncak positif); aktivasi ReLU konvensional mematikan seluruh ayunan negatif menjadi nol, sedangkan GELU mempertahankan aliran gradien yang halus pada nilai negatif kecil.
2. **Hierarki Enkoder Berbasis *Strided Convolution*:**  
   Lapisan `Stem Conv2d(3×3)` memetakan tensor masukan 2-kanal menjadi $C_0 = 24$ peta fitur pada resolusi $32 \times 32$. Tiga tingkat enkoder (`Encoder 1–3`) mengekstraksi fitur spasial lokal dan menurunkan dimensi kanvas secara bertahap ($32 \times 32 \to 16 \times 16 \to 8 \times 8 \to 4 \times 4$) menggunakan konvolusi berlangkah `Conv2d(kernel=2, stride=2)` sambil menggandakan kapasitas kanal ($24 \to 48 \to 96 \to 96$). Penggunaan *learnable strided convolution* dipilih menggantikan *Max-Pooling* untuk mencegah hilangnya detail posisi fase transien EEG.
3. **Bottleneck Multi-Head Self-Attention untuk Mengatasi Fenomena *False Neighbors*:**  
   Meskipun Kurva Hilbert menjamin setiap sampel berurutan di waktu ($t, t+1$) selalu bertetangga di kisi 2D ($\|\Delta \mathbf{p}\|_1 = 1$), pelipatan kurva fraktal menimbulkan fenomena **tetangga semu (*false neighbors*)**, di mana dua piksel yang bersebelahan pada batas lipatan kuadran berasal dari waktu perekaman yang berjauhan (pada kisi $32 \times 32$, terdapat 76 pasang piksel bertetangga dengan selisih waktu $\Delta t > 100$ sampel, maksimum $\Delta t = 853$ sampel). Untuk mengatasi bias lokalitas semu tersebut, peta fitur pada *bottleneck* ($\mathbf{B} \in \mathbb{R}^{B \times 96 \times 4 \times 4}$) diratakan menjadi barisan $L = 16$ token spasial $\mathbf{T} \in \mathbb{R}^{B \times 16 \times 96}$ yang merepresentasikan 16 sub-kuadran utama Kurva Hilbert, lalu diproses menggunakan **Multi-Head Self-Attention (MHA)** dengan $H = 4$ kepala atensi ($d_k = C / H = 24$) sesuai Persamaan (16):

$$\text{Attention}(\mathbf{Q}, \mathbf{K}, \mathbf{V}) = \text{softmax}\left(\frac{\mathbf{Q}\mathbf{K}^T}{\sqrt{d_k}}\right)\mathbf{V}, \quad \mathbf{T}_{\text{out}} = \mathbf{T} + \text{MHA}(\mathbf{T}, \mathbf{T}, \mathbf{T}) \qquad (16)$$

   Lapisan atensi ini memungkinkan model menimbang korelasi konteks global antar-kuadran dari awal hingga akhir segmen sinyal secara serentak.
4. **Dekoder Simetris dan *Lateral Skip Connections*:**  
   Tiga tingkat dekoder (`Decoder 3–1`) menaikkan kembali resolusi spasial ($4 \times 4 \to 8 \times 8 \to 16 \times 16 \to 32 \times 32$) melalui `ConvTranspose2d(kernel=2, stride=2)`. Pada setiap tingkatan, peta fitur digabungkan (*concatenated*) dengan peta fitur enkoder selevel melalui jalur pintas (*skip connections*), direduksi menggunakan `Conv2d(1×1)`, dan diproses oleh `ResBlock2D`. Jalur pintas ini menyalurkan kembali detail morfologi frekuensi tinggi langsung dari enkoder sehingga puncak tajam gelombang EEG tidak mengalami penghalusan berlebih (*over-smoothing*). Lapisan akhir `Head Conv2d(1×1)` memproyeksikan fitur menjadi kisi estimasi sinyal bersih $\hat{\mathbf{G}}_0 \in \mathbb{R}^{B \times 1 \times 32 \times 32}$.

*Tabel 3: Spesifikasi Arsitektur dan Konfigurasi Tensor pada Jaringan `HilbertUNet` 2D (Total Parameter: $786.241$)*
| Modul / Tahap | Konfigurasi Lapisan & Operator | Tensor Masukan | Tensor Keluaran | Peran Fisis pada Restorasi Sinyal |
| :--- | :--- | :---: | :---: | :--- |
| **Input & Stem** | `Conv2d(2, 24, k=3, s=1, p=1)` | $(B, 2, 32, 32)$ | $(B, 24, 32, 32)$ | Proyeksi awal bidang amplitudo & masker aktif |
| **Encoder 1** | `ResBlock2D(24)` + `Conv2d(24, 48, k=2, s=2)` | $(B, 24, 32, 32)$ | Skip $h_1: (B, 24, 32, 32)$<br>Down: $(B, 48, 16, 16)$ | Ekstraksi morfologi lokal skala halus ($32 \times 32$) |
| **Encoder 2** | `ResBlock2D(48)` + `Conv2d(48, 96, k=2, s=2)` | $(B, 48, 16, 16)$ | Skip $h_2: (B, 48, 16, 16)$<br>Down: $(B, 96, 8, 8)$ | Isolasi pola artefak skala menengah ($16 \times 16$) |
| **Encoder 3** | `ResBlock2D(96)` + `Conv2d(96, 96, k=2, s=2)` | $(B, 96, 8, 8)$ | Skip $h_3: (B, 96, 8, 8)$<br>Down: $(B, 96, 4, 4)$ | Abstraksi fitur makro lintas sub-kuadran ($8 \times 8$) |
| **Bottleneck** | `ResBlock2D(96)` + `MultiheadAttention(96, H=4)` | $(B, 96, 4, 4)$ | $(B, 96, 4, 4)$ | Konteks global 16 kuadran & mitigasi *false neighbors* |
| **Decoder 3** | `ConvTranspose2d(k=2,s=2)` + `Cat(h3)` + `Conv1×1` + `ResBlock2D(96)` | $(B, 96, 4, 4)$ | $(B, 96, 8, 8)$ | Rekonstruksi fitur $8 \times 8$ dengan fusi *skip* $h_3$ |
| **Decoder 2** | `ConvTranspose2d(k=2,s=2)` + `Cat(h2)` + `Conv1×1` + `ResBlock2D(48)` | $(B, 96, 8, 8)$ | $(B, 48, 16, 16)$ | Rekonstruksi fitur $16 \times 16$ dengan fusi *skip* $h_2$ |
| **Decoder 1** | `ConvTranspose2d(k=2,s=2)` + `Cat(h1)` + `Conv1×1` + `ResBlock2D(24)` | $(B, 48, 16, 16)$ | $(B, 24, 32, 32)$ | Pemulihan resolusi penuh & ketajaman puncak ($32 \times 32$) |
| **Output Head** | `Conv2d(24, 1, k=1, s=1)` + *Hilbert Path Unfolding* | $(B, 24, 32, 32)$ | Grid: $(B, 1, 32, 32)$<br>1D: $\hat{\mathbf{x}} \in \mathbb{R}^{B \times N}$ | Estimasi amplitudo bersih & pengembalian ke 1D |

---

### 2.6. Time-Frequency Hybrid Loss Function and Training Protocol
Fungsi objektif pelatihan memegang peran krusial dalam mengarahkan jaringan agar mampu meredam artefak berenergi besar tanpa merusak kandungan frekuensi fisiologis sinyal EEG. Oleh sebab itu, sub-bab ini merumuskan fungsi kerugian hibrida waktu-frekuensi (*Time-Frequency Hybrid Loss*), parameter optimasi pelatihan, serta rancangan protokol uji ablasi komparatif terhadap 11 arsitektur *deep learning* 2D di atas kisi Hilbert yang sama.

#### 2.6.1. Time-Frequency Hybrid Loss Formulation
Penggunaan kerugian domain waktu semata (seperti MSE murni) sering kali menyebabkan jaringan menghasilkan prediksi yang terlalu halus (*over-smoothed*) sehingga memangkas energi gelombang frekuensi tinggi ($\beta$ dan $\gamma$). Untuk mengatasi masalah tersebut, sinyal aktif hasil penelusuran jalur Hilbert $\hat{\mathbf{x}} = \hat{\mathbf{G}}_{0, \mathbf{R}_{0:N}, \mathbf{C}_{0:N}} \in \mathbb{R}^N$ dioptimasi terhadap sinyal bersih referensi $\mathbf{x} \in \mathbb{R}^N$ menggunakan **Fungsi Kerugian Hibrida Waktu-Frekuensi** ($\mathcal{L}_{\text{hybrid}}$) sebagaimana dinyatakan pada Persamaan (17):

$$\mathcal{L}_{\text{hybrid}}(\mathbf{x}, \hat{\mathbf{x}}) = \underbrace{\frac{1}{N} \sum_{t=0}^{N-1} \left( x_t - \hat{x}_t \right)^2}_{\mathcal{L}_{\text{time}} \text{ (Time-Domain MSE)}} + \lambda_{\text{spec}} \cdot \underbrace{\frac{1}{K} \sum_{k=0}^{K-1} \Big| \left|\mathcal{F}_{\text{RFFT}}(\mathbf{x})_k\right| - \left|\mathcal{F}_{\text{RFFT}}(\hat{\mathbf{x}})_k\right| \Big|}_{\mathcal{L}_{\text{freq}} \text{ (Orthonormal Spectral }\ell_1\text{ Loss)}} \qquad (17)$$

di mana $\mathcal{F}_{\text{RFFT}}(\cdot)$ adalah *One-Dimensional Real Fast Fourier Transform* dengan normalisasi ortonormal (`norm="ortho"`), $K = \lfloor N/2 \rfloor + 1$ adalah jumlah bin frekuensi positif ($K = 257$ untuk $N = 512$), dan koefisien penyeimbang spektral ditetapkan sebesar $\lambda_{\text{spec}} = 0.8$.

#### 2.6.2. Hyperparameter Configuration and 11-Model 2D Ablation Protocol
Seluruh proses pelatihan dieksekusi menggunakan pengoptimal **AdamW** dengan laju belajar awal $\eta_0 = 1.5 \times 10^{-3}$, peluruhan bobot (*weight decay*) $\lambda_w = 5 \times 10^{-4}$, ukuran *batch* $B = 256$, dan penjadwal laju belajar **Cosine Annealing** ($\text{CosineAnnealingLR}$). Untuk mencegah ketidakstabilan gradien akibat lonjakan amplitudo kedipan mata (EOG), diterapkan pemotongan norma gradien (*gradient norm clipping*) pada ambang $\|\nabla \mathcal{L}_{\text{hybrid}}\|_2 \le 1.0$. Pemilihan bobot terbaik (*best model checkpoint*) didasarkan pada nilai $\text{RMSE}$ terendah pada himpunan validasi independen.

Guna memvalidasi peran setiap komponen arsitektural secara adil (*fair comparison*), dirancang pula protokol **Uji Kelayakan dan Ablasi 11 Arsitektur Deep Learning 2D** yang seluruhnya menerima representasi kisi Hilbert $32 \times 32$ yang identik, meliputi: *(1) 2D Plain CNN Autoencoder, (2) 2D ResNet, (3) 2D DenseNet, (4) 2D MobileNet Separable CNN, (5) 2D Vanilla U-Net [5], (6) 2D Hilbert Res-UNet 1-Kanal (ablasi tanpa masker biner), (7) 2D Hilbert Res-UNet 2-Kanal tanpa Self-Attention (ablasi tanpa atensi bottleneck), (8) 2D Attention U-Net (Gate Attention), (9) 2D Vision Transformer (ViT Hilbert), (10) 2D ConvNeXt-Style Hilbert,* dan *(11) 2D Hilbert Res-UNet + Bottleneck Self-Attention (Model Usulan)*.

---

### 2.7. Inverse 1D Signal Reconstruction and Quantitative Evaluation Metrics
Tahap penutup dari metodologi ini adalah mendekapsulasi citra keluaran hasil restorasi menjadi deret waktu satu dimensi serta mengevaluasi performanya secara komprehensif. Sub-bab ini merinci prosedur rekonstruksi invers ke domain waktu 1D, formulasi lima metrik evaluasi kualitas restorasi sinyal lintas 10 tingkat SNR, serta protokol pengukuran latensi komputasi waktu-nyata (*real-time latency*).

#### 2.7.1. Deterministic Inverse Unfolding to 1D Time-Series
Kisi prediksi bersih $\hat{\mathbf{G}}_0 \in \mathbb{R}^{1 \times 32 \times 32}$ dari `HilbertUNet` dikemas ke dalam kontainer *Hilbert RGBA PNG* keluaran $\hat{\mathbf{I}} \in \{0, \dots, 255\}^{32 \times 32 \times 4}$ bersama metadata panjang $N$ di piksel terminal $(0, 31)$. Sesuai prosedur dekapsulasi pada **Algorithm 1** (Baris 10–15), dekoder membaca nilai `uint32` $N$ dari koordinat $(0, 31)$, mengekstraksi $N$ piksel aktif sepanjang koordinat Hilbert $(\mathbf{R}_{0:N}, \mathbf{C}_{0:N})$, dan menyatukan kembali 4 byte `[R, G, B, A]` menjadi vektor sinyal EEG 1D bersih $\hat{\mathbf{x}} \in \mathbb{R}^N$ (`float32`) tanpa distorsi kontainer ($\max_t |\hat{x}_t^{\text{grid}} - \hat{x}_t^{\text{PNG}}| = 0.0$).

#### 2.7.2. Multi-Domain Signal Restoration Evaluation Metrics
Kualitas sinyal hasil restorasi $\hat{\mathbf{x}}$ dibandingkan terhadap sinyal bersih referensi $\mathbf{x}$ pada himpunan uji di setiap level $\text{SNR}_{\text{in}} \in [-7, +2]\text{ dB}$ menggunakan lima metrik evaluasi standar pemrosesan biosinyal [4], [7], [11] yang dirumuskan pada Persamaan (18)–(22):

1. **Root Mean Squared Error ($\text{RMSE}$):**
   $$\text{RMSE} = \sqrt{\frac{1}{N} \sum_{t=0}^{N-1} \left( \hat{x}_t - x_t \right)^2} \qquad (18)$$
2. **Time-Domain Relative Root Mean Squared Error ($\text{RRMSE}_t$):**
   $$\text{RRMSE}_t = \frac{\|\hat{\mathbf{x}} - \mathbf{x}\|_2}{\|\mathbf{x}\|_2} = \sqrt{\frac{\sum_{t=0}^{N-1} (\hat{x}_t - x_t)^2}{\sum_{t=0}^{N-1} x_t^2}} \qquad (19)$$
3. **Spectral-Domain Relative Root Mean Squared Error ($\text{RRMSE}_s$):**
   $$\text{RRMSE}_s = \frac{\big\| |\mathcal{F}_{\text{RFFT}}(\hat{\mathbf{x}})| - |\mathcal{F}_{\text{RFFT}}(\mathbf{x})| \big\|_2}{\big\| |\mathcal{F}_{\text{RFFT}}(\mathbf{x})| \big\|_2} \qquad (20)$$
4. **Pearson Correlation Coefficient ($CC$):**
   $$CC = \frac{\sum_{t=0}^{N-1} (x_t - \bar{x})(\hat{x}_t - \bar{\hat{x}})}{\sqrt{\sum_{t=0}^{N-1} (x_t - \bar{x})^2 \sum_{t=0}^{N-1} (\hat{x}_t - \bar{\hat{x}})^2}} \qquad (21)$$
5. **Signal-to-Noise Ratio Improvement ($\Delta\text{SNR}$, dalam dB):**
   $$\text{SNR}_{\text{out}} = 20 \log_{10} \left( \frac{\|\mathbf{x}\|_2}{\|\hat{\mathbf{x}} - \mathbf{x}\|_2} \right), \quad \Delta\text{SNR} = \text{SNR}_{\text{out}} - \text{SNR}_{\text{in}} \qquad (22)$$

#### 2.7.3. Computational Latency and Real-Time Throughput Profiling
Kelayakan operasional sistem untuk aplikasi BCI waktu-nyata diuji dengan mengukur waktu eksekusi per tahap: latensi enkapsulasi sinyal 1D ke PNG ($T_{\text{encode}}$), latensi inferensi `HilbertUNet` 2D ($T_{\text{infer}}$), dan latensi dekapsulasi PNG kembali ke sinyal 1D ($T_{\text{decode}}$). Total latensi end-to-end per segmen ($T_{\text{total}} = T_{\text{encode}} + T_{\text{infer}} + T_{\text{decode}}$, dalam milidetik) beserta *throughput* pemrosesan ($1000 / T_{\text{total}}$, dalam sampel/detik) dievaluasi baik melalui penyangga memori (*in-memory buffer* `io.BytesIO`) maupun melalui operasi penulisan dan pembacaan berkas PNG fisik pada disk penyimpanan ($bs = 1$, dirata-ratakan selama $50$ siklus independen).

---

## CRediT Authorship Contribution Statement & Declarations

Bagian pernyataan kontribusi penulis dan deklarasi etika berikut disertakan untuk memenuhi persyaratan administrasi pengiriman naskah (*submission requirements*) pada jurnal terindeks Scopus.

* **CRediT Authorship Contribution Statement:** **Penulis Pertama:** Konseptualisasi, Metodologi, Perangkat Lunak, Validasi, Analisis Formal, Penulisan – Draf Asli. **Penulis Kedua & Ketiga:** Supervisi, Tinjauan Metodologi, Penulisan – Tinjauan & Penyuntingan.
* **Declaration of Competing Interest:** Para penulis menyatakan bahwa tidak terdapat konflik kepentingan finansial maupun hubungan personal yang dapat memengaruhi hasil penelitian yang dilaporkan dalam naskah ini.
* **Data and Code Availability:** Kode sumber *codec* Hilbert RGBA PNG, arsitektur `HilbertUNet` 2D, dan skrip benchmark komparatif 26 metode transformasi tersedia untuk mendukung keterulangan eksperimen.

---

## References

Daftar pustaka berikut disusun mengikuti format penomoran standar IEEE (*IEEE Citation Style*) yang digunakan secara luas pada jurnal teknik biomedis dan pemrosesan sinyal terindeks Scopus.

[1] H. Zeng *et al.*, "EEG-based brain-computer interfaces with deep learning: A survey," *IEEE Trans. Neural Syst. Rehabil. Eng.*, vol. 29, pp. 1689–1700, 2021.  
[2] R. J. Croft and R. J. Barry, "Removal of ocular artifact from the EEG: A review," *Neurophysiol. Clin./Clin. Neurophysiol.*, vol. 30, no. 1, pp. 5–19, 2000.  
[3] X. Jiang, G.-B. Bian, and Z. Tian, "Removal of artifacts from EEG signals: A review," *Sensors*, vol. 19, no. 5, p. 987, 2019.  
[4] H. Zhang *et al.*, "EEGdenoiseNet: A benchmark dataset for end-to-end deep learning solutions of EEG denoising," *J. Neural Eng.*, vol. 18, no. 5, p. 056057, 2021.  
[5] O. Ronneberger, P. Fischer, and T. Brox, "U-Net: Convolutional networks for biomedical image segmentation," in *Proc. Int. Conf. Med. Image Comput. Comput.-Assist. Interv. (MICCAI)*, 2015, pp. 234–241.  
[6] A. Vaswani *et al.*, "Attention is all you need," in *Proc. Adv. Neural Inf. Process. Syst. (NeurIPS)*, vol. 30, 2017, pp. 5998–6008.  
[7] N. Mashhadi, M. Khoshnevisan, and S. Ostadabbas, "A 2D U-Net convolutional neural network for physiological signal denoising," *arXiv preprint arXiv:2009.08809*, 2020.  
[8] C. Yan and T. Wu, "Two-dimensional time-frequency representation and non-local means filtering for EEG artifact removal," *Front. Neurosci.*, vol. 16, p. 991136, 2022.  
[9] C.-H. Chuang *et al.*, "IC-U-Net: A U-Net-based denoising autoencoder using mixtures of independent components for automatic EEG artifact removal," *NeuroImage*, vol. 263, p. 119586, 2022.  
[10] J. Huang *et al.*, "EEG signal denoising based on 2D image representation and GCANet architecture," *J. Netw. Intell.*, vol. 8, no. 4, 2023.  
[11] Y. Pei *et al.*, "DTP-Net: Densely connected temporal pyramid network with 2D time-frequency representation for EEG artifact removal," *IEEE J. Biomed. Health Inform.*, vol. 28, no. 5, 2024.  
[12] Y. Liu *et al.*, "E2CAR: Hardware-accelerated 2D-CNN framework for real-time EEG artifact removal," *arXiv preprint arXiv:2602.09035*, 2026.  
[13] "SWTT: Stationary wavelet transform and transformer self-attention framework for biosignal restoration," *Electronics*, 2026.  
[14] D. Hilbert, "Über die stetige Abbildung einer Linie auf ein Flächenstück," *Math. Ann.*, vol. 38, no. 3, pp. 459–460, 1891.  
[15] H. Sagan, *Space-Filling Curves*. New York, NY, USA: Springer-Verlag, 1994.  
[16] *IEEE Standard for Floating-Point Arithmetic*, IEEE Std 754-2019, 2019.