# DRAFT NASKAH PUBLIKASI ILMIAH (JOURNAL / CONFERENCE PAPER)

Dokumen ini merupakan draf naskah publikasi ilmiah (*academic manuscript draft*) yang menyusun bagian **Judul**, **Abstrak (Bilingual)**, **Bab I (Pendahuluan / Introduction)**, dan **Bab II (Material dan Metode / Materials and Methods)** secara lengkap, sistematis, dan siap diadaptasi ke format jurnal bereputasi (IEEE / Elsevier / Springer). Seluruh struktur penulisan telah dirancang mengikuti kaidah tata tulis akademik yang baku, di mana setiap bab dan sub-bab selalu diawali dengan paragraf pengantar sebelum memasuki sub-bagian di bawahnya.

**Judul Bahasa Indonesia:**  
**Restorasi Sinyal EEG Satu Dimensi Melalui Transformasi Spasial Lossless Kurva Fraktal Hilbert dan Bit-Packing IEEE 754 RGBA Menggunakan Arsitektur 2D Residual U-Net Berbasis Self-Attention**

**English Title (IEEE / Elsevier Standard):**  
**Lossless 1D-to-2D EEG Signal Transformation via Hilbert Space-Filling Curve and IEEE 754 RGBA Bit-Packing for Artifact Removal Using Bottleneck Self-Attention 2D Residual U-Net**

---

## ABSTRAK / ABSTRACT

Bagian abstrak berikut merangkum latar belakang permasalahan, kebaruan metodologi transformasi spasial *lossless* berbasis Kurva Hilbert dan *bit-packing* IEEE 754 RGBA PNG, arsitektur jaringan restorasi *2D Hilbert Residual U-Net* (`HilbertUNet`), serta bukti empiris keunggulan representasi yang diusulkan dalam bahasa Indonesia dan bahasa Inggris.

**Abstrak (Bahasa Indonesia)**  
Pemanfaatan arsitektur pembelajaran mendalam dua dimensi (*2D deep learning*) untuk restorasi dan penghilangan artefak (*denoising*) pada sinyal elektroensefalogram (EEG) satu dimensi (1D) umumnya terkendala oleh hilangnya informasi fase pada transformasi waktu-frekuensi, galat kuantisasi pada citra 8-bit, pembengkakan dimensi pada matriks relasional, atau terputusnya kontinuitas waktu pada pelipatan baris (*raster-scan*). Penelitian ini mengusulkan kerangka kerja transformasi sinyal 1D ke citra 2D yang bersifat **100% *bit-exact lossless*, mempertahankan lokalitas ketetanggaan waktu, dan mandiri (*self-contained*)** dengan mengintegrasikan **Kurva Ruang Hilbert (*Hilbert Space-Filling Curve*) Orde-5 ($32 \times 32$)** dan **pengemasan biner (*bit-packing*) IEEE 754 `float32` ke dalam kontainer citra 4-kanal RGBA PNG**. Setiap sampel sinyal 32-bit dipetakan tanpa pembulatan ke dalam empat kanal warna `[R, G, B, A]` (`uint8`), sementara piksel terminal Hilbert $(0, 31)$ dicadangkan untuk menyimpan metadata panjang sinyal dinamis ($N \le 1023$). Untuk mencegah distorsi non-linier saat operasi konvolusi, citra RGBA diekstraksi menjadi tensor masukan dua kanal (`[2, 32, 32]`) yang memisahkan bidang amplitudo kontinu dan masker biner sinyal aktif, kemudian diproses menggunakan arsitektur **2D Hilbert Residual U-Net (`HilbertUNet`)** yang dilengkapi **Bottleneck Multi-Head Self-Attention** serta fungsi kerugian hibrida waktu-frekuensi (*Hybrid MSE + Orthonormal RFFT $\ell_1$ Loss*). Hasil evaluasi komparatif terhadap 26 metode transformasi (12 metode global dan 14 varian *Circular Mapping*) membuktikan bahwa Hilbert RGBA PNG merupakan satu-satunya representasi yang secara simultan mencapai keterbalikan sempurna ($\text{Bit-Exact} = 100\%$, $\text{RMSE} = 0,000000$, $\text{PRD} = 0,0000\%$), langkah waktu kontinu tanpa lompatan ($\text{Max Jump} = 1$ piksel), korelasi spasial tetangga 2D tertinggi di kelas *lossless* ($r_{2D} = 0,8228$), serta ukuran berkas ultra-ringkas ($2,10\text{ KB}$, latensi *round-trip* $0,375\text{ ms}$), menjadikannya jembatan representasi yang eksak dan efisien untuk restorasi biosinyal berbasis visi komputer.

**Kata Kunci:** Kurva Ruang Hilbert, *Lossless IEEE 754 Bit-Packing*, Restorasi Sinyal EEG, Artefak EOG dan EMG, 2D Residual U-Net, *Self-Attention*.

**Abstract (English)**  
*Leveraging two-dimensional (2D) deep learning backbones for one-dimensional (1D) electroencephalogram (EEG) artifact removal typically suffers from phase degradation in time-frequency spectrograms, quantization error in 8-bit images, quadratic memory explosion in relational fields, or boundary discontinuities in naive raster folding. This paper proposes a deterministic, 100% bit-exact lossless, locality-preserving, and self-contained 1D-to-2D transformation framework by integrating an Order-5 discrete Hilbert space-filling curve ($32 \times 32$) with IEEE 754 single-precision (`float32`) byte-level bit-packing into standard 4-channel RGBA PNG containers. Each 32-bit signal sample is bijectively cast into four 8-bit `[R, G, B, A]` channels without numerical rounding, while the terminal Hilbert coordinate $(0, 31)$ intrinsically encodes dynamic sequence length metadata ($N \le 1023$). For neural inference, the container is unpacked into a dual-channel spatial tensor (`[2, 32, 32]`) pairing continuous signal amplitudes with a binary active mask to eliminate zero-padding bias, and processed by a lightweight 2D Residual U-Net (`HilbertUNet`) equipped with Bottleneck Multi-Head Self-Attention and a Time-Frequency Hybrid Loss. Comparative benchmarking across 26 signal-to-image transformation methods (12 global paradigms and 14 circular mappings) demonstrates that Hilbert RGBA PNG uniquely achieves exact zero-loss inversion ($\text{Bit-Exact} = 100\%$, $\text{RMSE} = 0.000000$, $\text{PRD} = 0.0000\%$), strict unit Manhattan temporal step continuity ($\text{Max Jump} = 1$ pixel), the highest 2D spatial neighbor coherence among lossless codecs ($r_{2D} = 0.8228$), and a compact $2.10\text{ KB}$ footprint ($0.375\text{ ms}$ round-trip latency).*

**Keywords:** *Hilbert Space-Filling Curve, Lossless IEEE 754 Bit-Packing, EEG Artifact Removal, EOG/EMG Denoising, 2D Residual U-Net, Self-Attention.*

---

## I. PENDAHULUAN (INTRODUCTION)

Bab Pendahuluan ini menguraikan konteks ilmiah dan urgensi permasalahan yang melandasi penelitian restorasi sinyal elektroensefalogram (EEG) melalui pendekatan transformasi spasial dua dimensi. Pembahasan disusun secara berkesinambungan mulai dari karakteristik fisis sinyal EEG dan kerentanannya terhadap kontaminasi artefak biologis, tinjauan kritis terhadap kesenjangan riset (*research gap*) pada empat rumpun metode transformasi 1D-ke-2D di literatur mutakhir, hingga perumusan solusi *lossless Hilbert RGBA PNG* beserta poin-poin kontribusi ilmiah yang ditawarkan.

### 1.1 Latar Belakang dan Urgensi Restorasi Sinyal EEG
Sinyal elektroensefalogram (EEG) merekam fluktuasi potensial bioelektrik pascasinaptik dari populasi neuron piramidal di korteks serebral dengan resolusi temporal tinggi. Dalam praktik klinis neurologis maupun sistem antarmuka otak-komputer (*Brain-Computer Interface* / BCI), informasi diagnostik utama EEG terkandung di dalam struktur morfologi gelombang waktu serta distribusi energi pada lima pita frekuensi fisiologis: Delta ($\delta: 0,5\text{--}4\text{ Hz}$), Theta ($\theta: 4\text{--}8\text{ Hz}$), Alpha ($\alpha: 8\text{--}13\text{ Hz}$), Beta ($\beta: 13\text{--}30\text{ Hz}$), dan Gamma ($\gamma: >30\text{ Hz}$) [1], [2]. Namun, karena amplitudo permukaan EEG berada pada skala mikrovolt ($10\text{--}100\,\mu\text{V}$), rekaman sinyal sangat rentan terkontaminasi oleh artefak fisiologis non-serebral, terutama **Elektrookulogram (EOG)** dan **Elektromiogram (EMG)** [3].

Artefak EOG timbul dari pergeseran dipol korneo-retinal saat mata berkedip atau bergerak, menghasilkan defleksi transien beramplitudo besar ($>200\,\mu\text{V}$) pada frekuensi rendah ($<4\text{--}8\text{ Hz}$) yang mendistorsi pita $\delta$ dan $\theta$. Sebaliknya, artefak EMG berasal dari potensial aksi kontraksi otot kranial dan fasial yang memiliki karakteristik derau pita lebar (*broadband noise*, $15\text{--}100+\text{ Hz}$) dengan tumpang-tindih spektral (*spectral overlap*) langsung terhadap pita $\beta$ dan $\gamma$. Karakteristik non-stasioner dan tumpang-tindih spektral ini menyebabkan metode pemfilteran linier klasik, dekomposisi wavelet ambang statis, maupun *Independent Component Analysis* (ICA) pada rekaman kanal tunggal gagal memisahkan artefak tanpa menggerus komponen gelombang saraf asli [4].

### 1.2 Perkembangan Paradigma Transformasi 1D-ke-2D dan Kesenjangan Riset (*Research Gap*)
Dalam beberapa tahun terakhir, arsitektur pembelajaran mendalam visi komputer dua dimensi (*2D Computer Vision*)—seperti *2D Convolutional Neural Networks* (2D-CNN), *Residual U-Net*, dan *Spatial Self-Attention*—telah terbukti memiliki kapasitas ekstraksi fitur multiskala dan bidang reseptif spasial yang melampaui jaringan konvolusi 1D standar [5], [6]. Hal ini mendorong munculnya paradigma **"Transformasi 1D-ke-2D $\to$ Pemrosesan Restorasi Citra 2D $\to$ Rekonstruksi Invers ke 1D"** pada pemrosesan sinyal biomedis.

Sejumlah studi mutakhir telah mengadopsi pendekatan ini. Mashhadi dkk. [7] melipat deret waktu fisiologis menjadi citra *grayscale* 2D untuk dibersihkan menggunakan arsitektur 2D U-Net. Yan dan Wu [8] mentransformasikan sinyal EEG ke dalam matriks waktu-frekuensi dua dimensi untuk pemfilteran berbasis *Non-Local Means* (NLM), lalu mengembalikannya melalui transformasi invers. Chuang dkk. [9] mengembangkan *IC-U-Net* berbasis citra komponen untuk penekanan artefak, sementara Huang dkk. [10] mengadaptasi jaringan *image dehazing* 2D (GCANet) pada kisi citra EEG. Baru-baru ini, Pei dkk. [11] mengusulkan *DTP-Net* menggunakan peta fitur waktu-frekuensi 2D dengan dekoder terpelajar, Liu dkk. [12] memperkenalkan kerangka *E2CAR* yang memetakan sinyal EEG 1D menjadi matriks 2D untuk akselerasi perangkat keras *2D-CNN*, dan kerangka *SWTT* [13] memanfaatkan matriks koefisien *Stationary Wavelet Transform* (SWT) 2D yang dikombinasikan dengan *Self-Attention*.

Meskipun menjanjikan, telaah kritis terhadap keempat rumpun transformasi 1D-ke-2D yang digunakan dalam literatur mengungkap **empat kelemahan fundamental** ketika diterapkan pada tugas *restorasi/denoising* sinyal EEG:

1. **Rumpun Transformasi Waktu-Frekuensi (STFT & CWT):**  
   *Short-Time Fourier Transform* (STFT) dan *Continuous Wavelet Transform* (CWT) terikat oleh prinsip ketidakpastian Heisenberg-Gabor ($\Delta t \cdot \Delta f \ge \frac{1}{4\pi}$). Ketika sinyal diubah menjadi citra spektrogram magnitudo, informasi sudut fase ($\angle S(t, f)$) hilang sehingga rekonstruksi sinyal 1D melalui estimasi fase iteratif (seperti algoritma Griffin-Lim) mengalami distorsi bentuk gelombang yang berat. Penyimpanan komponen kompleks secara terpisah maupun koefisien SWT multiskala memang menekan galat, tetapi tidak menghasilkan berkas citra standar yang *100% bit-exact* serta memperbesar dimensi tensor.
2. **Rumpun *Dynamic Fields* dan Geometri Ruang Fase ($N \times N$):**  
   Metode *Gramian Angular Summation/Difference Fields* (GASF/GADF), *Markov Transition Fields* (MTF), dan *Recurrence Plots* (RP) memetakan sinyal berpanjang $N$ menjadi matriks korelasi berdimensi $N \times N$. Untuk satu segmen EEG berpanjang $N=512$, ukuran matriks meledak secara kuadratik menjadi $512 \times 512 = 262.144$ piksel ($\mathcal{O}(N^2)$). Lebih fatal lagi, MTF dan RP bersifat *non-invertible* (tidak dapat dikembalikan menjadi sinyal mikrovolt 1D), sehingga tidak dapat digunakan untuk *denoising*.
3. **Rumpun Pemetaan Melingkar (*Circular Mapping* — Polar & Spiral):**  
   Pemetaan sinyal ke koordinat kutub/melingkar memiliki dua cacat geometri bawaan terhadap operasi konvolusi kartesius $3 \times 3$: **(a) Singularitas Kutub ($r \to 0$)**, di mana elemen luas diferensial $dA = r\,dr\,d\theta$ menyempit di pusat lingkaran sehingga sampel-sampel sinyal saling bertabrakan (*pixel collision*); serta **(b) Diskontinuitas Jahitan Sudut (*Angular Seam Jump*, $\theta: 2\pi \to 0$)**, yang memaksa sampel awal ($t=0$) bersentuhan langsung dengan sampel akhir ($t=N-1$) sehingga menciptakan garis patahan semu. Sementara itu, spiral filotaksis seperti *Vogel/Golden Spiral* menyebarkan titik berurutan dengan lompatan sudut emas $\approx 137,5^\circ$ yang menghancurkan korelasi spasial lokal ($r_{2D} \approx 0$).
4. **Rumpun Pelipatan Spasial Naif (*Raster-Scan*, *Morton*) & Kuantisasi Visual 8-Bit:**  
   Pendekatan pelipatan matriks baris demi baris (*Raster-scan reshaping* [7], [12]) memutus kontinuitas waktu setiap kali indeks berpindah dari ujung kanan baris $(r, M-1)$ ke awal baris berikutnya $(r+1, 0)$, menghasilkan lonjakan jarak spasial sebesar $M$ piksel (`Max Jump = 32 piksel` pada kisi $32 \times 32$), sedangkan kurva *Morton/Z-Order* menimbulkan lompatan diagonal antarkuadran (`Max Jump = 16 piksel`). Selain itu, pengonversian amplitudo `float32` ke citra *grayscale* 8-bit ($[0, 255]$) memicu galat pembulatan kuantisasi permanen yang menghapus fluktuasi halus sub-mikrovolt EEG.

### 1.3 Solusi yang Diusulkan dan Kontribusi Ilmiah
Untuk mengatasi keempat keterbatasan tersebut secara simultan, penelitian ini merancang kerangka kerja transformasi spasial *lossless* berbasis **Kurva Fraktal Hilbert Orde-5** dan **IEEE 754 RGBA Bit-Packing** yang dipadukan dengan arsitektur **2D Hilbert Residual U-Net (`HilbertUNet`)**. Kontribusi ilmiah utama dari penelitian ini meliputi empat poin berikut:

1. **Codec Transformasi 1D-ke-2D yang *100% Bit-Exact Lossless* dan *Self-Contained*:**  
   Merumuskan pemetaan bijektif sinyal EEG `float32` berpanjang dinamis ($1 \le N \le 1023$) ke dalam kontainer citra **Hilbert RGBA PNG** berdimensi konstan $32 \times 32 \times 4$ melalui *type-punning* 4 byte IEEE 754 ke kanal `[R, G, B, A]` tanpa kehilangan presisi sedikit pun ($\text{RMSE} = 0,0$, $\text{PRD} = 0,0\%$), dilengkapi penyisipan otomatis metadata panjang sinyal $N$ (`uint32`) pada piksel terminal Hilbert $(0, 31)$.
2. **Pembuktian Benchmark Komparatif Lintas 26 Metode Transformasi:**  
   Melakukan evaluasi objektif yang membandingkan metode usulan terhadap 12 metode transformasi sinyal-ke-citra global dan 14 varian *Circular Mapping*, membuktikan keunggulan simultan Hilbert RGBA PNG dalam hal keterbalikan biner, kontinuitas langkah Manhattan ($\|\Delta \mathbf{p}\|_1 = 1$), dan korelasi ketetanggaan spasial 2D tertinggi di kelas *lossless* ($r_{2D} = 0,8228$).
3. **Ekstraksi Tensor Masukan Dua Kanal Anti-Bias dan Arsitektur `HilbertUNet` dengan *Bottleneck Self-Attention*:**  
   Memisahkan fungsi kontainer penyimpanan (`uint8` RGBA) dari representasi inferensi jaringan dengan membentuk tensor masukan 2-kanal `[2, 32, 32]` (bidang amplitudo `float32` + masker biner sinyal aktif) untuk mengeliminasi *zero-padding bias*, serta mengintegrasikan *Multi-Head Self-Attention* pada *bottleneck* $4 \times 4$ untuk mengatasi fenomena tetangga semu (*false neighbors*) antar-lipatan kuadran kurva Hilbert.
4. **Supervisi Fungsi Kerugian Hibrida Waktu-Frekuensi (*Time-Frequency Hybrid Loss*):**  
   Menggabungkan fungsi kerugian *Mean Squared Error* (MSE) di domain waktu dan *Orthonormal RFFT Magnitude $\ell_1$ Loss* di domain frekuensi untuk memulihkan ketajaman morfologi gelombang sekaligus menjaga keutuhan spektrum ritme fisiologis otak pada rentang kontaminasi artefak EOG dan EMG ekstrem ($-7\text{ dB}$ hingga $+2\text{ dB}$).

---

## II. MATERIAL DAN METODE (MATERIALS AND METHODS)

Bab Material dan Metode ini memaparkan rancangan teknis dan matematis dari seluruh eksperimen kuantitatif yang dilaksanakan dalam penelitian. Pembahasan disusun secara berurutan mencakup tujuh bagian utama, yaitu gambaran umum kerangka kerja sistem (*system pipeline overview*), spesifikasi dataset dan protokol kontaminasi bebas kebocoran data, kerangka seleksi komparatif 26 metode transformasi 1D-ke-2D, formulasi matematis Kurva Hilbert Orde-5 dan *bit-packing* IEEE 754 RGBA PNG, ekstraksi tensor masukan dua kanal beserta arsitektur jaringan `HilbertUNet` 2D, fungsi kerugian hibrida waktu-frekuensi dan protokol pelatihan, serta prosedur rekonstruksi invers dan instrumen metrik evaluasi.

### 2.1 Gambaran Umum Kerangka Kerja Sistem (*System Pipeline Overview*)
Penelitian ini menerapkan pendekatan **eksperimental kuantitatif** untuk menguji hubungan sebab-akibat antara representasi spasial 2D Kurva Hilbert dan kinerja restorasi sinyal EEG secara objektif. Secara keseluruhan, arsitektur sistem terbagi ke dalam empat tahapan terintegrasi sebagaimana ditunjukkan pada **Gambar 1**:
1. **Konstruksi Data Kontaminasi & Partisi Bebas Kebocoran (*Zero Data Leakage*):** Pembangkitan pasangan sinyal EEG bersih dan sinyal terkontaminasi artefak EOG/EMG pada 10 tingkat SNR ($-7\text{ dB}$ s.d. $+2\text{ dB}$) dengan pembagian data berdasarkan indeks epoch bersih asli.
2. **Enkapsulasi Sinyal 1D ke Citra Hilbert RGBA PNG (*Lossless Encoding*):** Pemetaan deret waktu 1D ($N$ sampel) mengikuti lintasan Kurva Hilbert Orde-5 pada kisi $32 \times 32$, pengepakan 4 byte IEEE 754 `float32` ke kanal warna `[R, G, B, A]`, pengisian *zero-padding*, dan pencatatan panjang $N$ di piksel terminal $(0, 31)$.
3. **Ekstraksi Tensor 2-Kanal & Inferensi Restorasi `HilbertUNet` 2D:** Pembongkaran citra PNG menjadi tensor `[2, 32, 32]` (*Kanal 0: Amplitudo Sinyal*, *Kanal 1: Masker Biner Aktif*) yang diproses melalui jaringan *Residual U-Net* 2D berfitur *GroupNorm*, aktivasi *GELU*, dan *Bottleneck Multi-Head Self-Attention*.
4. **Rekonstruksi Invers ke Sinyal 1D (*Lossless Decoding*) & Evaluasi Multi-Domain:** Pembacaan metadata panjang $N$ dari piksel $(0, 31)$, penelusuran balik lintasan Hilbert untuk membentuk sinyal EEG 1D bersih $\hat{\mathbf{x}} \in \mathbb{R}^N$, serta evaluasi objektif di domain waktu, frekuensi, dan latensi komputasi.

```
[ Sinyal EEG Terkontaminasi 1D: y ∈ ℝᴺ (float32, N ≤ 1023) ]
                            │
                            ▼
┌───────────────────────────────────────────────────────────┐
│ TAHAP 1: ENKODER HILBERT RGBA PNG (LOSSLESS CONTAINER)    │
│ • Pemetaan Kurva Hilbert Orde k=5: d ∈ [0..1023] → (r, c) │
│ • Bit-Packing IEEE 754 float32 (4 Byte) → [R, G, B, A]    │
│ • Piksel 0..N-1: Sinyal Aktif | N..1022: Zero-Padding     │
│ • Piksel Terminal d=1023 di (0, 31): Simpan uint32(N)     │
└───────────────────────────┬───────────────────────────────┘
                            │  Berkas Citra: 32 × 32 × 4 RGBA PNG
                            ▼
┌───────────────────────────────────────────────────────────┐
│ TAHAP 2: EKSTRAKSI TENSOR MASUKAN DUA KANAL               │
│ • Tensor G ∈ ℝ^(2 × 32 × 32) (float32):                   │
│   - Kanal 0: Nilai Amplitudo Sinyal EEG pada Jalur Hilbert│
│   - Kanal 1: Masker Biner Aktif (1.0 = Aktif, 0.0 = Pad)  │
└───────────────────────────┬───────────────────────────────┘
                            │
                            ▼
┌───────────────────────────────────────────────────────────┐
│ TAHAP 3: RESTORASI SPASIAL 2D HILBERT RES-UNET            │
│ • Stem Conv 3×3 (2 → 24) + 3 Tingkat Enkoder ResBlock2D   │
│   (GroupNorm-8, GELU, Strided Conv 32×32 → 16×16 → 8×8)   │
│ • Bottleneck (4×4, 96 Ch): ResBlock2D + 4-Head Self-Attn  │
│ • 3 Tingkat Dekoder (ConvTranspose2d + Skip Connections)  │
│ • Supervision: Hybrid Time-Domain MSE + 0.8 RFFT L1 Loss  │
└───────────────────────────┬───────────────────────────────┘
                            │  Prediksi Kisi Bersih: [1, 32, 32]
                            ▼
┌───────────────────────────────────────────────────────────┐
│ TAHAP 4: DEKODER INVERS HILBERT & EVALUASI                │
│ • Baca Panjang N dari Piksel Terminal (0, 31) via <u4     │
│ • Telusuri Balik Indeks Hilbert 0..N-1 → Sinyal 1D x̂ ∈ ℝᴺ │
│ • Evaluasi: RMSE, RRMSE_t, RRMSE_s, CC, ΔSNR, & Latensi   │
└───────────────────────────────────────────────────────────┘
```
*Gambar 1: Diagram alur arsitektur sistem transformasi lossless Hilbert RGBA PNG dan restorasi sinyal `HilbertUNet` 2D.*

---

### 2.2 Dataset, Protokol Kontaminasi Semi-Sintetis, dan Partisi Bebas Kebocoran Data
Pelatihan dan evaluasi model restorasi sinyal berbasis pembelajaran terawasi (*supervised deep learning*) memerlukan pasangan sinyal EEG terkontaminasi dan sinyal referensi bersih (*ground-truth*) yang terukur secara presisi. Oleh sebab itu, sub-bab ini menguraikan spesifikasi akuisisi data, mekanisme pemasangan satu-ke-satu antara sinyal otak bersih dan sinyal artefak, formulasi pencampuran semi-sintetis terkontrol daya pada sepuluh tingkatan *Signal-to-Noise Ratio* (SNR), serta protokol partisi dataset yang dirancang ketat untuk mencegah kebocoran informasi (*zero data leakage*).

#### 2.2.1 Spesifikasi Sinyal dan Skema Pemasangan Satu-ke-Satu (*One-to-One Pairing*)
Eksperimen dilakukan menggunakan dataset benchmark EEG semi-sintetis yang terdiri atas segmen sinyal EEG bersih (*ground-truth*), sinyal artefak okular (EOG), dan sinyal artefak muskular (EMG) dengan frekuensi pencuplikan $f_s = 256\text{ Hz}$ dan panjang segmen standar $N = 512$ sampel (durasi jendela 2 detik), serta variasi panjang sinyal dinamis $N \in [128, 1023]$ untuk pengujian fleksibilitas kontainer.

Untuk menghindari bias pengulangan pola artefak tertentu selama pelatihan, pencampuran dilakukan menggunakan skema pemasangan acak satu-ke-satu (*one-to-one random pairing*) tanpa duplikasi:
* **Dataset Kontaminasi EOG:** Sebanyak $3.400$ epoch sinyal EEG bersih dipasangkan secara acak satu-ke-satu dengan $3.400$ epoch artefak EOG.
* **Dataset Kontaminasi EMG:** Sebanyak $4.514$ epoch sinyal EEG bersih dipasangkan secara acak satu-ke-satu dengan $4.514$ epoch artefak EMG.

#### 2.2.2 Formulasi Pencampuran Terkontrol Daya (*Power-Controlled SNR Scaling*)
Setiap sinyal EEG terkontaminasi $\mathbf{y} \in \mathbb{R}^N$ dibangkitkan melalui superposisi aditif antara sinyal EEG bersih $\mathbf{x} \in \mathbb{R}^N$ dan sinyal artefak $\mathbf{n} \in \mathbb{R}^N$:

$$\mathbf{y} = \mathbf{x} + \alpha \cdot \mathbf{n}$$

Karena daya intrinsik antar-epoch bervariasi secara alami, faktor skala pengali artefak $\alpha$ dihitung secara adaptif per pasangan sampel agar memenuhi target *Signal-to-Noise Ratio* masukan ($\text{SNR}_{\text{in}}$ dalam satuan dB) secara eksak:

$$\text{SNR}_{\text{in}} = 10 \log_{10} \left( \frac{\sum_{t=0}^{N-1} x_t^2}{\sum_{t=0}^{N-1} (\alpha \cdot n_t)^2} \right) \implies \alpha = \frac{\|\mathbf{x}\|_2}{\|\mathbf{n}\|_2 \cdot 10^{\left(\text{SNR}_{\text{in}} / 20\right)}}$$

Setiap pasangan unik diekspansi pada **10 tingkat $\text{SNR}_{\text{in}}$** yang mencakup kontaminasi sangat berat hingga ringan, yaitu $\text{SNR}_{\text{in}} \in \{-7, -6, -5, -4, -3, -2, -1, 0, +1, +2\}\text{ dB}$.

#### 2.2.3 Protokol Partisi Bebas Kebocoran Data (*Zero Data Leakage*)
Untuk menjamin validitas generalisasi model, pembagian data menjadi himpunan latih (*Train*, 80%), validasi (*Validation*, 10%), dan uji (*Test*, 10%) dilakukan **pada tingkat indeks epoch sinyal bersih asli sebelum diekspansi ke 10 level SNR**, sebagaimana dirangkum pada **Tabel 1**. Protokol ini memastikan tidak ada satu pun gelombang dasar EEG pada himpunan uji yang pernah dilihat oleh model selama pelatihan pada level SNR mana pun.

*Tabel 1: Komposisi dan Partisi Dataset Kontaminasi EOG dan EMG Bebas Kebocoran Data*
| Jenis Kontaminasi | Jumlah Pasangan Dasar Unik | Variasi Level $\text{SNR}_{\text{in}}$ | Total Sampel Sinyal | Himpunan Latih (80% Epoch) | Himpunan Validasi (10% Epoch) | Himpunan Uji (10% Epoch) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Artefak EOG** | $3.400$ pasang | $10$ level ($-7\text{ s.d. }+2\text{ dB}$) | $34.000$ | $27.200$ (Epoch $0\text{--}2.719$) | $3.400$ (Epoch $2.720\text{--}3.059$) | $3.400$ (Epoch $3.060\text{--}3.399$) |
| **Artefak EMG** | $4.514$ pasang | $10$ level ($-7\text{ s.d. }+2\text{ dB}$) | $45.140$ | $36.110$ (Epoch $0\text{--}3.610$) | $4.510$ (Epoch $3.611\text{--}4.061$) | $4.520$ (Epoch $4.062\text{--}4.513$) |

---

### 2.3 Kerangka Seleksi dan Benchmark 26 Metode Transformasi 1D-ke-2D
Penetapan **Hilbert RGBA PNG** sebagai metode representasi spasial utama dalam penelitian ini tidak dilakukan secara subjektif, melainkan didasarkan pada eksperimen komparatif pendahuluan yang menguji 26 metode transformasi sinyal-ke-citra lintas rumpun keilmuan. Sub-bab ini memaparkan pengelompokan ke-26 kandidat metode yang diuji, kriteria seleksi berjenjang tiga babak beserta perumusan metrik topologi spasialnya, serta rangkuman bukti empiris yang mendasari pemilihan metode terbaik.

#### 2.3.1 Ruang Lingkup 26 Kandidat Metode Transformasi (Global dan *Circular Mapping*)
Seluruh kandidat metode diuji secara konsisten pada sinyal EEG nyata acuan Dataset Index 0 (`Dataset/EOG/linear_synthetic.npy`, indeks 0, $\text{SNR} = -5\text{ dB}$, $N = 512$, `float32`). Ke-26 metode tersebut terbagi ke dalam dua kelompok eksperimen:
1. **Kelompok Benchmark Global (12 Metode Lintas Rumpun):** Meliputi *Space-Filling & Spatial Folding* (Hilbert RGBA PNG, Square Spiral RGBA PNG, Morton/Z-Order RGBA PNG, Raster RGBA PNG, Raster Grayscale 8-bit PNG), *Time-Frequency* (SWT Haar 4-Level, STFT Complex R+I+L, STFT Magnitude + Griffin-Lim, CWT Morlet Scalogram), dan *Dynamic Fields* (GASF, MTF, Recurrence Plot).
2. **Kelompok Taksonomi *Circular Mapping* (14 Varian Melingkar):** Meliputi 9 teknik *Native Variable-Length* (Direct Polar Plot, Archimedean Spiral, Fermat Spiral, Vogel/Golden Spiral, Symmetric Dot Pattern/SDP, Hilbert Polar Map, Poincaré Polar Map, Complex FFT Polar, Learnable Polar Map) dan 5 teknik *Fixed-Grid* (Polar Sector/Grid, Concentric Ring, Adaptive Polar-GAF, Polar Recurrence Plot, Polar Scalogram).

#### 2.3.2 Protokol Tiga Babak Eliminasi Objektif dan Metrik Topologi Spasial 2D
Setiap metode transformasi dievaluasi melalui **Tiga Babak Eliminasi Objektif** yang mencerminkan syarat mutlak arsitektur *2D deep learning denoising*:
* **Babak 1 — Syarat Keterbalikan Mutlak (*100% Bit-Exact Invertibility*):** Metode wajib mampu mengembalikan citra 2D menjadi sinyal 1D dengan $\text{RMSE}_{\text{rec}} = 0,000000$ dan $\text{PRD} = 0,0000\%$ (`np.array_equal(x, x_hat) == True`) dalam format berkas citra standar yang ringkas.
* **Babak 2 — Syarat Kontinuitas Langkah Temporal (*Unit Step Continuity*):** Jarak langkah Manhattan maksimum antar-sampel waktu berurutan ($t$ dan $t+1$) pada bidang 2D wajib memenuhi $\text{Max Jump } L_1 = \max_t (\|\mathbf{p}(t+1) - \mathbf{p}(t)\|_1) = 1\text{ piksel}$.
* **Babak 3 — Koherensi Ketetanggaan Spasial 2D untuk Konvolusi $3 \times 3$:** Diukur menggunakan **Korelasi Spasial Tetangga 2D ($r_{2D}$)** pada himpunan pasangan piksel bertetangga 4-arah ortogonal $\mathcal{E}_4$ dan **Normalized 2D Total Variation ($\text{TV}_{\text{norm}}$)** pada wilayah sinyal aktif $\Omega_{\text{aktif}}$:

$$r_{2D} = \frac{\sum_{(u,v) \in \mathcal{E}_4} (I_u - \bar{I})(I_v - \bar{I})}{\sqrt{\sum (I_u - \bar{I})^2 \sum (I_v - \bar{I})^2}}, \quad \text{TV}_{\text{norm}} = \frac{\sum_{r,c} \left( |I_{r+1,c} - I_{r,c}| + |I_{r,c+1} - I_{r,c}| \right)}{2 \cdot |\Omega_{\text{aktif}}| \cdot \sigma_I}$$

#### 2.3.3 Hasil Seleksi Empiris dan Penetapan Representasi Pemenang
Ringkasan hasil seleksi pada **Tabel 2** dan **Tabel 3** membuktikan bahwa **Hilbert RGBA PNG ($32 \times 32 \times 4$)** merupakan satu-satunya metode yang lolos ketiga babak secara mutlak: mencapai $\text{Bit-Exact} = 100\%$, $\text{Max Jump} = 1\text{ piksel}$ (mengeliminasi Raster `32 px`, Morton `16 px`, Fermat Spiral `3 px`, dan Vogel Spiral `76 px`), serta meraih korelasi spasial tertinggi di kelas *lossless* ($r_{2D} = \mathbf{0,8228}$, $\text{TV}_{\text{norm}} = \mathbf{0,3572}$) dengan ukuran berkas hanya $2,10\text{ KB}$ ($499\times$ lebih hemat memori dibandingkan GASF/MTF $1.048,7\text{ KB}$).

*Tabel 2: Hasil Eksperimen Perbandingan 12 Metode Transformasi Sinyal 1D-ke-2D Global pada Sinyal EEG Dataset Index 0 ($N=512$)*
| No | Kandidat Metode Transformasi | Rumpun Metode | Dimensi Citra | Sifat Rekonstruksi (Bit-Exact) | Galat Rekonstruksi (RMSE) | Langkah Maksimum | Korelasi Spasial 2D | Ukuran Berkas per Sinyal |
|:---:|:---|:---|:---:|:---:|:---:|:---:|:---:|:---:|
| **1** | **Hilbert RGBA PNG (Dipilih)** | **Space-Filling** | **`32x32x4`** | **100% (Lossless)** | **0,000000** | **1 piksel** | **0,8228** | **2,10 KB** |
| 2 | Square Spiral RGBA PNG | Spiral Folding | `32x32x4` | 100% (Lossless) | 0,000000 | 1 piksel | 0,5409 | 2,12 KB |
| 3 | Morton / Z-Order RGBA PNG | Space-Filling | `32x32x4` | 100% (Lossless) | 0,000000 | 16 piksel | 0,8172 | 2,01 KB |
| 4 | Raster RGBA PNG | Spatial Folding | `32x32x4` | 100% (Lossless) | 0,000000 | 32 piksel | 0,7205 | 1,99 KB |
| 5 | Raster Grayscale 8-bit PNG [7] | Spatial Folding | `32x32` | 0% (Lossy) | 0,010920 | 32 piksel | 0,7205 | 0,48 KB |
| 6 | SWT 2D Coeffs (Haar 4L) [13] | Time-Frequency | `5x512` (`.npy`) | 0% (Lossy)* | 0,000000* | — | 0,6782 | 10,37 KB |
| 7 | STFT Complex (R+I+L) | Time-Frequency | `3x33x33` (`.npy`) | 0% (Lossy)* | 0,000000* | — | 0,8550 | 13,20 KB |
| 8 | STFT Mag + Griffin-Lim | Time-Frequency | `33x33` (`.npy`) | 0% (Lossy) | 2,662886 | — | 0,8550 | 4,48 KB |
| 9 | CWT Scalogram (Morlet) | Time-Frequency | `32x512` (`.npy`) | 0% (Lossy) | 1,512942 | — | 0,9625 | 65,66 KB |
| 10 | GASF ([0,1] Best-Case) | Dynamic Field | `512x512` (`.npy`) | 0% (Lossy) | 0,000002 | — | 0,9911 | 1.048,70 KB |
| 11 | MTF (Markov Field) | Dynamic Field | `512x512` (`.npy`) | 0% (Non-Invertible) | Tidak Dapat Dibalik | — | 0,8260 | 1.048,70 KB |
| 12 | Recurrence Plot (RP) | Dynamic Field | `512x512` (PNG) | 0% (Non-Invertible) | Tidak Dapat Dibalik | — | 0,6431 | 18,48 KB |

*Tabel 4: Hasil Eksperimen Perbandingan 14 Varian Circular Mapping pada Sinyal EEG Dataset Index 0 ($N=512$, Kanvas $64 \times 64$)*
| No | Kandidat Metode Transformasi | Kelompok Mekanisme | Dimensi Citra | Sifat Rekonstruksi (Bit-Exact) | Galat Rekonstruksi (RMSE) | Langkah Maksimum | Korelasi Spasial 2D | Ukuran Berkas per Sinyal |
|:---:|:---|:---|:---:|:---:|:---:|:---:|:---:|:---:|
| | **KELOMPOK 1: LANGSUNG VARIABEL** | | | | | | | |
| 1 | Direct Polar Plot | Langsung Variabel | `64x64` | 0% (Lossy) | 0,133059 | 4 piksel | 0,4741 | 1,42 KB |
| 2 | Archimedean Spiral | Langsung Variabel | `64x64` | 0% (Lossy) | 0,481926 | 5 piksel | 0,1637 | 2,40 KB |
| 3 | Fermat Spiral | Langsung Variabel | `64x64` | 100% (Lossless) | 0,000000 | 3 piksel | 0,1625 | 2,78 KB |
| 4 | Vogel / Golden Spiral | Langsung Variabel | `64x64` | 100% (Lossless) | 0,000000 | 76 piksel | -0,0001 | 2,88 KB |
| 5 | Symmetric Dot (SDP) | Langsung Variabel | `64x64` | 0% (Non-Invertible) | Tidak Dapat Dibalik | — | 0,7678 | 2,18 KB |
| 6 | Hilbert Polar Map | Langsung Variabel | `64x64` | 0% (Lossy) | 0,063951 | 10 piksel | 0,2607 | 1,64 KB |
| 7 | Poincaré Polar Map | Langsung Variabel | `64x64` | 0% (Non-Invertible) | Tidak Dapat Dibalik | — | 0,7295 | 0,67 KB |
| 8 | Complex FFT Polar | Langsung Variabel | `64x64` | 0% (Lossy) | 1,363625 | 50 piksel | 0,3285 | 1,06 KB |
| 9 | Learnable Polar Map | Langsung Variabel | `64x64` | 0% (Lossy) | 0,545156 | 9 piksel | 0,0776 | 2,61 KB |
| | **KELOMPOK 2: GRID TETAP** | | | | | | | |
| 10 | Polar Sector / Grid | Grid Tetap | `64x64` | 0% (Lossy) | 0,349102 | 8 piksel | 0,8105 | 4,96 KB |
| 11 | Concentric Ring | Grid Tetap | `64x64` | 0% (Lossy) | 0,295122 | 7 piksel | 0,1754 | 2,43 KB |
| 12 | Adaptive Polar-GAF | Grid Tetap | `64x64` | 0% (Lossy) | 2,691449 | — | 0,2601 | 2,07 KB |
| 13 | Polar Recurrence Plot | Grid Tetap | `64x64` | 0% (Non-Invertible) | Tidak Dapat Dibalik | — | 0,2000 | 0,25 KB |
| 14 | Polar Scalogram | Grid Tetap | `64x64` | 0% (Lossy) | 1,631531 | — | 0,4221 | 8,00 KB |

---

### 2.4 Formulasi Matematis Kurva Hilbert Orde-5 dan IEEE 754 RGBA Bit-Packing
Setelah Kurva Hilbert ditetapkan sebagai representasi pemenang, diperlukan perumusan matematis yang mengikat pemetaan koordinat topologi 2D dengan mekanisme penyimpanan presisi angka *floating-point* 32-bit secara utuh. Sub-bab ini merinci tiga komponen formulasi *codec* yang diusulkan, yaitu algoritma pembangkitan koordinat Kurva Hilbert Orde-5 pada kisi $32 \times 32$, operator *type-punning* byte IEEE 754 `float32` ke dalam empat kanal warna RGBA PNG, serta struktur alokasi piksel terminal untuk menyematkan metadata panjang sinyal secara mandiri (*self-contained*).

#### 2.4.1 Pemetaan Topologi Spasial Kurva Hilbert Diskrit
Misalkan $\mathcal{G}_k = \{0, 1, \dots, M-1\}^2 \subset \mathbb{Z}^2$ menyatakan kisi ortogonal 2D berukuran $M \times M$ dengan $M = 2^k$. Untuk orde fraktal $k = 5$, diperoleh dimensi kanvas tetap $M = 32$ dengan kapasitas total $M^2 = 1.024$ piksel. Kurva Hilbert mendefinisikan pemetaan bijektif:

$$H_5: \mathcal{I} \to \mathcal{G}_5, \quad \mathcal{I} = \{0, 1, \dots, 1023\}, \quad d \mapsto \mathbf{p}(d) = (r_d, c_d)$$

Untuk setiap indeks linear $d \in \mathcal{I}$, koordinat kartesian $(x, y)$ dihitung secara iteratif melintasi skala sub-kuadran $s = 2^m$ ($m = 0, 1, \dots, k-1$) dengan mengekstraksi bit status rotasi $rx$ dan $ry$:

$$rx = 1 \ \& \ \left\lfloor \frac{t}{2} \right\rfloor, \quad ry = 1 \ \& \ (t \oplus rx), \quad \text{dengan } t = \left\lfloor \frac{d}{4^m} \right\rfloor$$

Pada setiap skala $s$, koordinat lokal mengalami transformasi refleksi dan rotasi kuadran $\mathcal{R}(s, x, y, rx, ry)$:

$$\mathcal{R}(s, x, y, rx, ry) = \begin{cases} (s - 1 - y, \; s - 1 - x), & \text{jika } ry = 0 \text{ dan } rx = 1 \\ (y, \; x), & \text{jika } ry = 0 \text{ dan } rx = 0 \\ (x, \; y), & \text{jika } ry = 1 \end{cases}$$

diikuti oleh translasi kuadran $(x, y) \leftarrow (x + s \cdot rx, \; y + s \cdot ry)$, lalu ditetapkan indeks baris $r_d = y$ dan kolom $c_d = x$. Pemetaan ini menjamin sifat ketetanggaan Manhattan konstan:

$$\|\mathbf{p}(d+1) - \mathbf{p}(d)\|_1 = |r_{d+1} - r_d| + |c_{d+1} - c_d| = 1, \quad \forall d \in \{0, 1, \dots, 1022\}$$

#### 2.4.2 Pengemasan Bit IEEE 754 `float32` ke Kanal Warna RGBA PNG
Setiap sampel sinyal EEG $x_t \in \mathbb{R}$ direpresentasikan dalam format standar IEEE 754 *single-precision floating-point* *little-endian* (`<f4`) berukuran 32-bit (1 bit tanda $s$, 8 bit eksponen berbias $e$, dan 23 bit mantisa $m$):

$$x_t = (-1)^s \times 2^{e - 127} \times \left( 1 + \sum_{i=1}^{23} m_{23-i} 2^{-i} \right)$$

Karena ukuran memori satu sampel `float32` ($32\text{ bit} = 4\text{ byte}$) identik secara eksak dengan ukuran satu piksel citra RGBA 8-bit ($4 \times 8\text{ bit} = 32\text{ bit}$), didefinisikan operator *type-punning* bijektif $\Phi: \mathbb{R}_{\text{float32}} \to \{0, 1, \dots, 255\}^4$ yang memetakan keempat byte penyusun $x_t$ langsung ke kanal **Red (R), Green (G), Blue (B), dan Alpha (A)** pada koordinat Hilbert $(r_t, c_t)$:

$$\mathbf{I}(r_t, c_t, :) = \Phi(x_t) = \begin{bmatrix} R(r_t, c_t) \\ G(r_t, c_t) \\ B(r_t, c_t) \\ A(r_t, c_t) \end{bmatrix} = \begin{bmatrix} \beta_0(x_t) \\ \beta_1(x_t) \\ \beta_2(x_t) \\ \beta_3(x_t) \end{bmatrix} = \text{reinterpret\_cast}\langle\text{uint8}[4]\rangle(x_t)$$

di mana $\beta_0$ adalah *least significant byte* (LSB) mantisa dan $\beta_3$ adalah *most significant byte* (MSB) yang memuat bit tanda dan eksponen.

#### 2.4.3 Pembagian Zona Kanvas dan Penyisipan Metadata Panjang Sinyal Dinamis
Untuk mengakomodasi sinyal dengan panjang variabel $1 \le N \le 1023$ di dalam dimensi citra fisik yang terkunci tetap pada $32 \times 32 \times 4$, ke-1.024 piksel sepanjang lintasan Hilbert dibagi menjadi tiga zona fungsional:
1. **Zona Sinyal Aktif ($0 \le d \le N-1$):** Menyimpan byte IEEE 754 `float32` dari sampel sinyal $x_0, x_1, \dots, x_{N-1}$.
2. **Zona *Zero-Padding* ($N \le d \le 1022$):** Diisi dengan byte nol `[0, 0, 0, 0]` (`uint8`) untuk melengkapi geometri matriks persegi.
3. **Zona Metadata Terminal ($d_{\text{term}} = M^2 - 1 = 1023$, pada koordinat $(r, c) = (0, 31)$):** Menyimpan nilai panjang sinyal asli $N$ yang dikemas sebagai bilangan bulat tak-bertanda 32-bit *little-endian* (`uint32` / `<u4`):

$$\mathbf{I}(0, 31, :) = \text{reinterpret\_cast}\langle\text{uint8}[4]\rangle(\text{uint32}(N))$$

Matriks $\mathbf{I} \in \{0, \dots, 255\}^{32 \times 32 \times 4}$ kemudian disimpan sebagai berkas PNG standar menggunakan kompresi *Deflate lossless*.

---

### 2.5 Ekstraksi Tensor Masukan Dua Kanal dan Arsitektur 2D Hilbert Res-UNet (`HilbertUNet`)
Meskipun format RGBA PNG sangat ideal sebagai kontainer penyimpanan *lossless*, byte warna `uint8` tidak dapat dikonvolusikan secara langsung karena struktur eksponen dan mantisa IEEE 754 bersifat non-linier terhadap operasi aritmetika matriks. Oleh karena itu, sub-bab ini menjelaskan prosedur ekstraksi citra RGBA menjadi tensor masukan dua kanal (*dual-channel spatial tensor*) bertipe `float32` serta menguraikan rancangan arsitektur **2D Hilbert Residual U-Net (`HilbertUNet`)** yang memadukan konvolusi residual lokal dengan mekanisme *Bottleneck Multi-Head Self-Attention*.

#### 2.5.1 Formulasi Tensor Masukan Dua Kanal (`[2, 32, 32]`)
Saat berkas Hilbert RGBA PNG dibaca ke dalam pipeline jaringan saraf, keempat byte RGBA pada jalur sinyal aktif disatukan kembali menjadi nilai `float32` dan disusun menjadi **tensor spasial dua kanal** $\mathbf{G} \in \mathbb{R}^{2 \times 32 \times 32}$:

$$\mathbf{G}_{0, r_d, c_d} = \begin{cases} x_d, & \text{jika } 0 \le d < N \quad (\text{Bidang Amplitudo Sinyal EEG}) \\ 0,0, & \text{jika } N \le d \le 1023 \quad (\text{Padding \& Piksel Metadata}) \end{cases}$$

$$\mathbf{G}_{1, r_d, c_d} = \begin{cases} 1,0, & \text{jika } 0 \le d < N \quad (\text{Masker Biner Sinyal Aktif}) \\ 0,0, & \text{jika } N \le d \le 1023 \quad (\text{Masker Area Non-Aktif}) \end{cases}$$

Penyertaan **Kanal 1 (Masker Biner Aktif)** memberikan panduan spasial eksplisit kepada filter konvolusi $3 \times 3$ dan lapisan *Group Normalization*, sehingga nilai $0,0$ pada zona *padding* tidak disalahartikan sebagai aktivitas gelombang otak datar bernilai $0\,\mu\text{V}$ (*anti-padding bias*).

#### 2.5.2 Rancangan Arsitektur `HilbertUNet` 2D dan Penanganan *False Neighbors*
Model restorasi **`HilbertUNet`** dirancang sebagai arsitektur *encoder-decoder* residual simetris berbobot ringan (total **$786.241$ parameter**, ukuran bobot $\approx 3,0\text{ MB}$) yang rincian lapisannya disajikan pada **Tabel 4**:

1. **Blok Residual Kustom (`ResBlock2D`):**  
   Setiap blok terdiri atas dua rangkaian `[GroupNorm(8) → GELU → Conv2d(3×3, pad=1)]` dengan sisipan `Dropout2d(p=0.05)` dan koneksi jalan pintas identitas $\mathbf{z}_{\text{out}} = \mathbf{z}_{\text{in}} + \mathcal{F}(\mathbf{z}_{\text{in}})$. *Group Normalization* (8 grup) dipilih menggantikan *Batch Normalization* agar statistik normalisasi dihitung secara independen per sampel (stabil pada inferensi *real-time* dengan $bs=1$). Fungsi aktivasi **GELU** ($\text{GELU}(z) = z \cdot \Phi(z)$) dipilih menggantikan ReLU agar ayunan lembah tegangan negatif EEG tidak terpotong mati (*dead neuron*) menjadi nol.
2. **Enkoder Berlangkah (*Strided Convolution Downsampling*):**  
   Lapisan pembuka (`Stem`) memproyeksikan masukan 2-kanal menjadi $C_0 = 24$ kanal fitur pada resolusi $32 \times 32$. Tiga tahap enkoder (`e1`, `e2`, `e3`) mengekstraksi fitur lokal, diikuti oleh penurunan resolusi spasial menggunakan konvolusi berlangkah `Conv2d(kernel=2, stride=2)` ($32 \times 32 \to 16 \times 16 \to 8 \times 8 \to 4 \times 4$) dengan penggandaan kanal ($24 \to 48 \to 96 \to 96$). Penggunaan *strided convolution* yang dapat dipelajari menghindari hilangnya informasi fase mikrovolt yang lazim terjadi pada operasi *Max-Pooling* kasar.
3. **Bottleneck Multi-Head Self-Attention (Solusi Fenomena *False Neighbors*):**  
   Meskipun Kurva Hilbert menjamin setiap sampel berurutan $t$ dan $t+1$ selalu bertetangga di kisi 2D ($\|\Delta \mathbf{p}\|_1 = 1$), sifat perlipatan kurva menyebabkan fenomena **tetangga semu (*false neighbors*)**: pada kisi $32 \times 32$, terdapat 76 pasang piksel yang bersebelahan secara spasial tetapi berasal dari waktu perekaman yang berjauhan ($\Delta t > 100$ sampel, selisih maksimum mencapai $853$ sampel). Untuk mencegah konvolusi lokal terkecoh oleh lipatan batas kuadran ini, peta fitur pada titik terdalam (*bottleneck* $\mathbf{B} \in \mathbb{R}^{B \times 96 \times 4 \times 4}$) diratakan menjadi $16$ token spasial $\mathbf{T} \in \mathbb{R}^{B \times 16 \times 96}$ (mewakili 16 sub-kuadran utama Kurva Hilbert) dan diproses oleh **Multi-Head Self-Attention** ($H = 4$ kepala atensi, dimensi per kepala $d_k = 24$):

$$\text{Attention}(\mathbf{Q}, \mathbf{K}, \mathbf{V}) = \text{softmax}\left(\frac{\mathbf{Q}\mathbf{K}^T}{\sqrt{d_k}}\right)\mathbf{V}, \quad \mathbf{T}_{\text{out}} = \mathbf{T} + \text{MHA}(\mathbf{T}, \mathbf{T}, \mathbf{T})$$

   Mekanisme ini mengevaluasi korelasi global antar-kuadran secara dinamis dari awal hingga akhir sinyal.
4. **Dekoder Simetris dan *Lateral Skip Connections*:**  
   Tiga tahap dekoder (`u3`, `u2`, `u1`) memulihkan resolusi spasial menggunakan `ConvTranspose2d(kernel=2, stride=2)` ($4 \times 4 \to 8 \times 8 \to 16 \times 16 \to 32 \times 32$). Pada setiap skala, fitur hasil *upsampling* digabungkan (*concatenation*) dengan fitur resolusi tinggi dari enkoder selevel melalui jalur pintas (*skip connections*), diringkas menggunakan konvolusi titik `Conv2d(1×1)`, dan diperhalus oleh `ResBlock2D`. Jalur pintas ini menjaga ketajaman puncak gelombang otak asli agar tidak tumpul (*over-smoothed*). Lapisan penutup `Head Conv2d(1×1)` memetakan 24 kanal fitur menjadi 1 kanal kisi bersih $\hat{\mathbf{G}}_0 \in \mathbb{R}^{1 \times 32 \times 32}$.

*Tabel 4: Spesifikasi Rinci Arsitektur Jaringan `HilbertUNet` 2D ($786.241$ Parameter)*
| Tahap / Modul | Operasi Lapisan Utama | Dimensi Tensor Masukan | Dimensi Tensor Keluaran | Fungsi Fisis & Arsitektural |
| :--- | :--- | :---: | :---: | :--- |
| **Input & Stem** | `Conv2d(2, 24, k=3, s=1, p=1)` | $(B, 2, 32, 32)$ | $(B, 24, 32, 32)$ | Ekstraksi fitur awal amplitudo & masker aktif |
| **Encoder 1** | `ResBlock2D(24)` + `DownConv(24→48, k=2, s=2)` | $(B, 24, 32, 32)$ | Skip $h_1: (B, 24, 32, 32)$<br>Down: $(B, 48, 16, 16)$ | Pemfilteran morfologi skala halus ($32 \times 32$) |
| **Encoder 2** | `ResBlock2D(48)` + `DownConv(48→96, k=2, s=2)` | $(B, 48, 16, 16)$ | Skip $h_2: (B, 48, 16, 16)$<br>Down: $(B, 96, 8, 8)$ | Abstraksi pola artefak skala menengah ($16 \times 16$) |
| **Encoder 3** | `ResBlock2D(96)` + `DownConv(96→96, k=2, s=2)` | $(B, 96, 8, 8)$ | Skip $h_3: (B, 96, 8, 8)$<br>Down: $(B, 96, 4, 4)$ | Abstraksi struktur makro gelombang ($8 \times 8$) |
| **Bottleneck** | `ResBlock2D(96)` + `MultiheadAttention(96, heads=4)` | $(B, 96, 4, 4)$ | $(B, 96, 4, 4)$ | Korelasi global 16 kuadran & koreksi *false neighbors* |
| **Decoder 3** | `ConvTranspose2d(k=2,s=2)` + `Cat(h3)` + `Conv1×1` + `ResBlock2D(96)` | $(B, 96, 4, 4)$ | $(B, 96, 8, 8)$ | Restorasi spasi $8 \times 8$ dengan fusi jalur pintas $h_3$ |
| **Decoder 2** | `ConvTranspose2d(k=2,s=2)` + `Cat(h2)` + `Conv1×1` + `ResBlock2D(48)` | $(B, 96, 8, 8)$ | $(B, 48, 16, 16)$ | Restorasi spasi $16 \times 16$ dengan fusi jalur pintas $h_2$ |
| **Decoder 1** | `ConvTranspose2d(k=2,s=2)` + `Cat(h1)` + `Conv1×1` + `ResBlock2D(24)` | $(B, 48, 16, 16)$ | $(B, 24, 32, 32)$ | Pemulihan ketajaman puncak gelombang ($32 \times 32$) |
| **Output Head** | `Conv2d(24, 1, k=1, s=1)` + *Hilbert 1D Unfolding* | $(B, 24, 32, 32)$ | Grid: $(B, 1, 32, 32)$<br>Sinyal: $\hat{\mathbf{x}} \in \mathbb{R}^{B \times N}$ | Proyeksi amplitudo bersih & rekonstruksi deret 1D |

---

### 2.6 Fungsi Kerugian Hibrida Waktu-Frekuensi (*Time-Frequency Hybrid Loss*) dan Protokol Pelatihan
Keberhasilan restorasi sinyal EEG sangat bergantung pada cara jaringan saraf diarahkan selama optimasi bobot agar tidak sekadar menekan rata-rata galat amplitudo, melainkan juga menjaga komposisi spektral ritme otak. Oleh sebab itu, sub-bab ini menjelaskan perumusan fungsi kerugian hibrida waktu-frekuensi (*Time-Frequency Hybrid Loss*) yang menggabungkan pengawasan morfologi waktu dan magnitudo Fourier, diikuti oleh rincian konfigurasi hiperparameter pelatihan serta protokol uji kelayakan dan ablasi terhadap 11 arsitektur *deep learning* 2D pembanding.

#### 2.6.1 Formulasi Fungsi Kerugian Hibrida (*Time-Frequency Hybrid Loss*)
Pelatihan yang hanya mengandalkan galat domain waktu (seperti MSE murni) cenderung mendorong jaringan mengambil jalan pintas berupa penghalusan berlebih (*over-smoothing*) yang memangkas komponen frekuensi tinggi fisiologis otak ($\beta$ dan $\gamma$). Untuk mencegah hal tersebut, sinyal aktif hasil ekstraksi jalur Hilbert $\hat{\mathbf{x}} = \hat{\mathbf{G}}_{0, \mathbf{R}_{0:N}, \mathbf{C}_{0:N}} \in \mathbb{R}^N$ diawasi terhadap sinyal bersih target $\mathbf{x} \in \mathbb{R}^N$ menggunakan **Fungsi Kerugian Hibrida Waktu-Frekuensi**:

$$\mathcal{L}_{\text{hybrid}}(\mathbf{x}, \hat{\mathbf{x}}) = \mathcal{L}_{\text{time}}(\mathbf{x}, \hat{\mathbf{x}}) + \lambda_{\text{spec}} \cdot \mathcal{L}_{\text{freq}}(\mathbf{x}, \hat{\mathbf{x}})$$

dengan rincian kedua komponen kerugian sebagai berikut:
1. **Kerugian Morfologi Domain Waktu (*Mean Squared Error* / MSE):**
   $$\mathcal{L}_{\text{time}}(\mathbf{x}, \hat{\mathbf{x}}) = \frac{1}{N} \sum_{t=0}^{N-1} \left( x_t - \hat{x}_t \right)^2$$
2. **Kerugian Spektral Domain Frekuensi (*Orthonormal RFFT Magnitude $\ell_1$ Loss*):**
   $$\mathcal{L}_{\text{freq}}(\mathbf{x}, \hat{\mathbf{x}}) = \frac{1}{K} \sum_{k=0}^{K-1} \Big| \left|\mathcal{F}_{\text{RFFT}}(\mathbf{x})_k\right| - \left|\mathcal{F}_{\text{RFFT}}(\hat{\mathbf{x}})_k\right| \Big|, \quad K = \left\lfloor \frac{N}{2} \right\rfloor + 1$$
   di mana $\mathcal{F}_{\text{RFFT}}(\cdot)$ adalah *One-Dimensional Discrete Fourier Transform* untuk masukan bernilai riil dengan normalisasi ortonormal (`norm="ortho"`), dan faktor bobot penyeimbang ditetapkan sebesar $\lambda_{\text{spec}} = 0,8$.

#### 2.6.2 Konfigurasi Hiperparameter dan Rancangan Uji Ablasi 11 Model 2D
Model dilatih menggunakan pengoptimal **AdamW** dengan laju belajar awal $\eta_0 = 1,5 \times 10^{-3}$, peluruhan bobot (*weight decay*) $\lambda_w = 5 \times 10^{-4}$, ukuran *batch* $256$, serta penjadwal laju belajar **Cosine Annealing** ($\text{CosineAnnealingLR}$). Untuk menjaga stabilitas pembaruan bobot terhadap lonjakan amplitudo ekstrem artefak EOG, diterapkan pemotongan norma gradien (*gradient norm clipping*) pada batas maksimum $\|\nabla \mathcal{L}\|_2 \le 1,0$. Pada akhir setiap epoch, model dievaluasi pada himpunan validasi independen dan hanya bobot dengan nilai $\text{RMSE}_{\text{val}}$ terendah yang disimpan sebagai *best checkpoint*.

Selain pelatihan penuh untuk restorasi EOG dan EMG, metodologi penelitian ini juga mencakup protokol **Uji Kelayakan dan Ablasi 11 Arsitektur Deep Learning 2D** di atas kisi masukan Hilbert $32 \times 32$ yang identik, yaitu: *(1) 2D Plain CNN Autoencoder, (2) 2D ResNet, (3) 2D DenseNet, (4) 2D MobileNet Separable CNN, (5) 2D Vanilla U-Net, (6) 2D Hilbert Res-UNet 1-Kanal (tanpa masker biner), (7) 2D Hilbert Res-UNet 2-Kanal tanpa Self-Attention, (8) 2D Attention U-Net (Gate Attention), (9) 2D Vision Transformer (ViT Hilbert), (10) 2D ConvNeXt-Style Hilbert,* dan *(11) 2D Hilbert Res-UNet + Bottleneck Self-Attention (Usulan)* untuk memvalidasi kontribusi spesifik setiap komponen arsitektural.

---

### 2.7 Rekonstruksi Invers ke Sinyal 1D dan Metrik Evaluasi Kinerja
Tahap akhir dari keseluruhan metodologi penelitian adalah mengembalikan kisi dua dimensi hasil prediksi model menjadi deret waktu satu dimensi serta mengukur keberhasilan restorasi sinyal menggunakan instrumen kuantitatif yang objektif. Sub-bab ini merinci prosedur dekapsulasi invers dari kontainer *Hilbert RGBA PNG* kembali menjadi sinyal EEG 1D, perumusan lima metrik evaluasi restorasi sinyal di domain waktu dan frekuensi, serta protokol pengujian latensi komputasi waktu-nyata (*real-time latency*).

#### 2.7.1 Prosedur Dekapsulasi Invers Citra ke Deret Waktu 1D
Setelah jaringan `HilbertUNet` menghasilkan kisi keluaran bersih $\hat{\mathbf{G}}_0 \in \mathbb{R}^{1 \times 32 \times 32}$, nilai amplitudo pada jalur aktif dikemas kembali ke dalam kontainer citra *Hilbert RGBA PNG* $\hat{\mathbf{I}} \in \{0, \dots, 255\}^{32 \times 32 \times 4}$ bersama nilai panjang $N$ pada piksel terminal $(0, 31)$. Proses dekapsulasi (*decoding*) dari berkas citra kembali menjadi sinyal 1D bersih $\hat{\mathbf{x}} \in \mathbb{R}^N$ dijalankan melalui tiga langkah deterministik:
1. Membaca 4 byte pada piksel terminal $(0, 31)$ dan menginterpretasikannya sebagai bilangan bulat `uint32` (`<u4`) untuk memperoleh panjang sinyal aktif $N$:
   $$N = \text{reinterpret\_cast}\langle\text{uint32}\rangle\left(\hat{\mathbf{I}}(0, 31, :)\right)$$
2. Mengambil $N$ piksel pertama secara berurutan mengikuti koordinat lintasan Kurva Hilbert $(\mathbf{R}_{0:N}, \mathbf{C}_{0:N})$ dan mengabaikan seluruh piksel pada zona *zero-padding*.
3. Menyatukan kembali keempat byte `[R, G, B, A]` pada setiap piksel aktif menjadi bilangan riil IEEE 754 `float32` (`<f4`), sehingga diperoleh deret waktu 1D $\hat{\mathbf{x}} = [\hat{x}_0, \hat{x}_1, \dots, \hat{x}_{N-1}]^T \in \mathbb{R}^N$ dengan galat kontainer bernilai tepat nol ($\max_t |\hat{x}_t^{\text{grid}} - \hat{x}_t^{\text{PNG}}| = 0,0$).

#### 2.7.2 Formulasi Metrik Evaluasi Kualitas Restorasi Sinyal
Kinerja restorasi sinyal EEG pada himpunan data uji dievaluasi secara kuantitatif pada setiap tingkat $\text{SNR}_{\text{in}} \in [-7, +2]\text{ dB}$ menggunakan lima metrik baku internasional pemrosesan biosinyal [4], [7], [11]:

1. **Root Mean Squared Error ($\text{RMSE}$):**
   $$\text{RMSE} = \sqrt{\frac{1}{N} \sum_{t=0}^{N-1} \left( \hat{x}_t - x_t \right)^2}$$
2. **Relative Root Mean Squared Error Domain Waktu ($\text{RRMSE}_t$):**
   $$\text{RRMSE}_t = \frac{\|\hat{\mathbf{x}} - \mathbf{x}\|_2}{\|\mathbf{x}\|_2} = \sqrt{\frac{\sum_{t=0}^{N-1} (\hat{x}_t - x_t)^2}{\sum_{t=0}^{N-1} x_t^2}}$$
3. **Relative Root Mean Squared Error Domain Spektral ($\text{RRMSE}_s$):**
   $$\text{RRMSE}_s = \frac{\big\| |\mathcal{F}_{\text{RFFT}}(\hat{\mathbf{x}})| - |\mathcal{F}_{\text{RFFT}}(\mathbf{x})| \big\|_2}{\big\| |\mathcal{F}_{\text{RFFT}}(\mathbf{x})| \big\|_2}$$
4. **Koefisien Korelasi Pearson ($CC$):**
   $$CC = \frac{\sum_{t=0}^{N-1} (x_t - \bar{x})(\hat{x}_t - \bar{\hat{x}})}{\sqrt{\sum_{t=0}^{N-1} (x_t - \bar{x})^2 \sum_{t=0}^{N-1} (\hat{x}_t - \bar{\hat{x}})^2}}$$
5. **Peningkatan Rasio Sinyal terhadap Derau ($\Delta\text{SNR}$ dalam dB):**
   $$\text{SNR}_{\text{out}} = 20 \log_{10} \left( \frac{\|\mathbf{x}\|_2}{\|\hat{\mathbf{x}} - \mathbf{x}\|_2} \right), \quad \Delta\text{SNR} = \text{SNR}_{\text{out}} - \text{SNR}_{\text{in}}$$

#### 2.7.3 Protokol Pengukuran Latensi Komputasi dan Throughput Sistem
Untuk memverifikasi kelayakan implementasi pada aplikasi antarmuka otak-komputer waktu-nyata (*real-time BCI deployment*), dilakukan pengukuran waktu eksekusi pada setiap tahapan pipeline: waktu enkapsulasi sinyal ke PNG ($T_{\text{encode}}$), waktu inferensi model `HilbertUNet` ($T_{\text{infer}}$), dan waktu dekapsulasi PNG ke sinyal 1D ($T_{\text{decode}}$). Total latensi per sampel ($T_{\text{total}} = T_{\text{encode}} + T_{\text{infer}} + T_{\text{decode}}$ dalam satuan milidetik) dan kapasitas pemrosesan (*throughput* $= 1000 / T_{\text{total}}$ dalam satuan sampel/detik) diuji baik pada mode *in-memory buffer* (`io.BytesIO`) maupun mode pembacaan/penulisan berkas PNG fisik di disk ($bs = 1$ selama $50$ siklus berulang).

---

## REFERENSI / REFERENCES

Daftar pustaka berikut memuat sumber-sumber literatur ilmiah yang menjadi acuan teoretis, metodologis, dan komparatif pada bagian Pendahuluan serta Material dan Metode dalam penelitian ini.

1. H. Zeng *et al.*, "EEG-based brain-computer interfaces with deep learning: A survey," *IEEE Trans. Neural Syst. Rehabil. Eng.*, vol. 29, pp. 1689–1700, 2021.
2. R. J. Croft and R. J. Barry, "Removal of ocular artifact from the EEG: A review," *Neurophysiol. Clin./Clin. Neurophysiol.*, vol. 30, no. 1, pp. 5–19, 2000.
3. X. Jiang, G.-B. Bian, and Z. Tian, "Removal of artifacts from EEG signals: A review," *Sensors*, vol. 19, no. 5, p. 987, 2019.
4. H. Zhang *et al.*, "EEGdenoiseNet: A benchmark dataset for end-to-end deep learning solutions of EEG denoising," *J. Neural Eng.*, vol. 18, no. 5, p. 056057, 2021.
5. O. Ronneberger, P. Fischer, and T. Brox, "U-Net: Convolutional networks for biomedical image segmentation," in *Proc. MICCAI*, 2015, pp. 234–241.
6. A. Vaswani *et al.*, "Attention is all you need," in *Proc. Adv. Neural Inf. Process. Syst. (NeurIPS)*, vol. 30, 2017, pp. 5998–6008.
7. N. Mashhadi, M. Khoshnevisan, and S. Ostadabbas, "A 2D U-Net convolutional neural network for physiological signal denoising," *arXiv preprint arXiv:2009.08809*, 2020.
8. C. Yan and T. Wu, "Two-dimensional time-frequency representation and non-local means filtering for EEG artifact removal," *Front. Neurosci.*, vol. 16, p. 991136, 2022.
9. C.-H. Chuang *et al.*, "IC-U-Net: A U-Net-based denoising autoencoder using mixtures of independent components for automatic EEG artifact removal," *NeuroImage*, vol. 263, p. 119586, 2022.
10. J. Huang *et al.*, "EEG signal denoising based on 2D image representation and GCANet architecture," *J. Netw. Intell.*, vol. 8, no. 4, 2023.
11. Y. Pei *et al.*, "DTP-Net: Densely connected temporal pyramid network with 2D time-frequency representation for EEG artifact removal," *IEEE J. Biomed. Health Inform. (JBHI)*, vol. 28, no. 5, 2024.
12. Y. Liu *et al.*, "E2CAR: Hardware-accelerated 2D-CNN framework for real-time EEG artifact removal," *arXiv preprint arXiv:2602.09035*, 2026.
13. "SWTT: Stationary wavelet transform and transformer self-attention framework for biosignal restoration," *Electronics (MDPI)*, 2026.
14. D. Hilbert, "Über die stetige Abbildung einer Linie auf ein Flächenstück," *Math. Ann.*, vol. 38, no. 3, pp. 459–460, 1891.
15. H. Sagan, *Space-Filling Curves*. New York, NY, USA: Springer-Verlag, 1994.
16. *IEEE Standard for Floating-Point Arithmetic*, IEEE Std 754-2019, 2019.
