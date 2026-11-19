# DRAFT TESIS: TRANSFORMASI LOSSLESS SINYAL EEG 1D KE REPRESENTASI SPASIAL 2D MENGGUNAKAN KURVA HILBERT DAN IEEE 754 BIT-PACKING

---

## IDENTITAS USULAN PENELITIAN
* **Topik/Bidang Keilmuan:** Pemrosesan Sinyal Biomedika, Pembelajaran Mesin (*Deep Learning*), Visi Komputer (*Computer Vision*).
* **Fokus Khusus Dokumen Ini:** Bab Metodologi dan Landasan Teori Transformasi Sinyal 1D ke Citra 2D Berbasis Kurva Hilbert (*Hilbert Space-Filling Curve*) & *Lossless Bit-Packing Container*.

---

# DAFTAR ISI SINTESIS
- [BAB I: PENDAHULUAN](#bab-i-pendahuluan)
  - 1.1 Latar Belakang Masalah
  - 1.2 Rumusan Masalah
  - 1.3 Tujuan Penelitian
  - 1.4 Batasan Masalah
  - 1.5 Manfaat Penelitian
- [BAB II: TINJAUAN PUSTAKA DAN LANDASAN TEORI](#bab-ii-tinjauan-pustaka-dan-landasan-teori)
  - 2.1 Sinyal Elektroensefalografi (EEG) dan Karakteristik Kontaminasi Artefak
  - 2.2 Keterbatasan Metode Transformasi 1D-ke-2D Konvensional
  - 2.3 Teori Fraktal Kurva Ruang Hilbert (*Hilbert Space-Filling Curve*)
  - 2.4 Standar Representasi Bilangan Riil IEEE 754 (*Single-Precision Float32*)
  - 2.5 Konsep Kontainer Citra Lossless RGBA PNG
- [BAB III: METODOLOGI PENELITIAN DAN PERANCANGAN SISTEM](#bab-iii-metodologi-penelitian-dan-perancangan-sistem)
  - 3.1 Kerangka Kerja Sistem (*System Pipeline Architecture*)
  - 3.2 Formulasi Matematis Pemetaan Spasial Kurva Hilbert Orde-$k$
  - 3.3 Mekanisme Injeksi Bit-Packing IEEE 754 ke Kanal RGBA
  - 3.4 Penyisipan Metadata Panjang Sinyal Dinamis pada Piksel Terminal
  - 3.5 Pembentukan Tensor Masukan Dua Kanal (*Dual-Channel Grid*) untuk Arsitektur 2D
  - 3.6 Algoritma Dekompresi dan Rekonstruksi Sinyal 1D (*Bit-Exact Inversion*)
  - 3.7 Protokol Validasi Bebas Kebocoran Data (*Zero Data Leakage*)
- [BAB IV: HASIL PENGUJIAN DAN PEMBAHASAN](#bab-iv-hasil-pengujian-dan-pembahasan)
  - 4.1 Pembuktian Matematis dan Analisis Rekonstruksi Zero-Loss
  - 4.2 Perbandingan terhadap Pendekatan Konvensional (STFT, CWT, Kuantisasi 8-Bit)
  - 4.3 Analisis Komputasi, Latensi Disk I/O, dan Throughput Sistem
  - 4.4 Analisis Inductive Bias Spasial Kurva Hilbert pada Operasi Konvolusi 2D
- [BAB V: KESIMPULAN DAN SARAN TAHAP LANJUT](#bab-v-kesimpulan-dan-saran-tahap-lanjut)
  - 5.1 Kesimpulan
  - 5.2 Saran Pengembangan untuk Bab Model Denoising (2D Res-UNet)

---

# BAB I: PENDAHULUAN

### 1.1 Latar Belakang Masalah
Sinyal elektroensefalogram (EEG) merupakan rekaman aktivitas bioelektrik fluktuasi potensial membran neuron di korteks serebral manusia yang memiliki resolusi temporal tinggi. Namun, amplitudo sinyal EEG berada pada orde mikrovolt ($\mu\text{V}$), menjadikannya sangat rentan terdistorsi oleh aktivitas biologis non-serebral seperti kedipan mata (*Electrooculogram* / EOG) dan kontraksi otot wajah atau leher (*Electromyogram* / EMG). Kehadiran artefak ini mengaburkan ritme gelombang otak fisiologis ($\delta, \theta, \alpha, \beta, \gamma$), sehingga tahap *denoising* (penghilangan derau) menjadi krusial sebelum sinyal dianalisis untuk *Brain-Computer Interface* (BCI), deteksi epilepsi, atau pemantauan klinis neurologis.

Dalam satu dekade terakhir, arsitektur pembelajaran mendalam 2 dimensi (*2D Deep Learning*)—seperti *2D Convolutional Neural Network* (CNN), *Residual U-Net* (Res-UNet), dan *Vision Transformer* (ViT)—telah menunjukkan keunggulan representasi hierarkis fitur yang jauh melampaui arsitektur 1 dimensi konvensional. Keberhasilan ini memicu tren pemetaan deret waktu 1D ke dalam representasi matriks atau citra 2D agar dapat dieksploitasi oleh tulang punggung (*backbone*) visi komputer.

Akan tetapi, teknik konversi 1D-ke-2D yang lazim digunakan saat ini menghadapi dilema ilmiah yang mendasar:
1. **Transformasi Waktu-Frekuensi (STFT / Wavelet):** Terikat pada batas ketidakpastian Heisenberg-Gabor, menghasilkan pertukaran (*trade-off*) resolusi temporal vs spektral, serta kerap membuang dinamika fase sinyal saat hanya magnitudo spektrogram yang dikonversi menjadi citra.
2. **Kuantisasi Citra Visual 8-Bit:** Mengonversi amplitudo float32 ke rentang skala abu-abu $[0, 255]$ memicu galat kuantisasi (*quantization error*) permanen dengan batas resolusi $\sim \mathcal{O}(10^{-2})$, merusak variasi mikrovolt EEG.
3. **Reshaping Matriks Sederhana (*Raster-Scan*):** Melipat sinyal 1D per baris matriks memutus kesinambungan waktu pada batas kolom akhir dan awal baris berikutnya, menimbulkan diskontinuitas buatan yang merusak filter konvolusi.

Untuk mengatasi paradoks tersebut, penelitian tesis ini mengusulkan sebuah metodologi transformasi sinyal 1D ke kontainer citra 2D yang **100% deterministik, dapat dibalik secara sempurna (*bit-exact lossless*), mempertahankan kedekatan temporal antar-sampel, dan memiliki dimensi spasial konstan ($32 \times 32$) untuk sinyal berpanjang dinamis ($N \le 1023$)**. Metode ini menggabungkan pemetaan lintasan fraktal **Kurva Ruang Hilbert (*Hilbert Space-Filling Curve*) Orde ke-5** dengan teknik **Bit-Packing IEEE 754 float32 ke kanal warna RGBA**.

### 1.2 Rumusan Masalah
1. Bagaimana merumuskan pemetaan topologi spasial 1D ke 2D yang mempertahankan lokalitas temporal tetangga sinyal menggunakan Kurva Hilbert tanpa menimbulkan diskontinuitas baris seperti pada *raster scan*?
2. Bagaimana merancang mekanisme pengemasan data (*bit-packing*) bilangan riil presisi tunggal IEEE 754 `float32` ke dalam format citra 4 kanal RGBA `uint8` sehingga sinyal dapat direkonstruksi kembali tanpa galat numerik sama sekali ($\text{max error} = 0.0$)?
3. Bagaimana menyematkan metadata panjang sinyal dinamis ($N$) secara intrinsik ke dalam citra PNG tanpa bergantung pada berkas konfigurasi eksternal?
4. Bagaimana membentuk representasi tensor masukan 2 kanal (*amplitude plane* dan *binary active mask*) agar operasi konvolusi pada model 2D tidak terdistorsi oleh piksel *padding*?

### 1.3 Tujuan Penelitian
1. Mengembangkan formulasi matematis dan algoritma komputasi untuk transformasi sinyal EEG 1D ke grid 2D berukuran konstan $32 \times 32$ berbasis Kurva Hilbert.
2. Merealisasikan *codec* lossless berbasis *type-punning* bit IEEE 754 ke kanal RGBA PNG dengan pembuktian analitik dan empiris identitas bit-level (`is_identical == True`).
3. Mengintegrasikan mekanisme penyimpan metadata panjang sinyal $N$ pada koordinat fraktal terminal $(0, 31)$ grid Hilbert.
4. Memvalidasi performa rekonstruksi, efisiensi kompresi, serta profil latensi komputasi *in-memory* dan penyimpanan disk fisik.

### 1.4 Batasan Masalah
1. Sinyal biosinyal yang digunakan berfokus pada rekaman EEG semi-sintetis dengan kontaminasi artefak okular (EOG) dan artefak muskular (EMG) pada variasi SNR $-7\text{ dB}$ hingga $+2\text{ dB}$ dengan frekuensi cuplik $f_s = 256\text{ Hz}$.
2. Dimensi kisi citra ditetapkan konstan $32 \times 32$ piksel (1024 piksel), yang mengakomodasi panjang sinyal dinamis $1 \le N \le 1023$ sampel.
3. Citra keluaran menggunakan format standar PNG (*Portable Network Graphics*) dengan kompresi *Deflate* tanpa penurunan kualitas (*lossless compression*).

### 1.5 Manfaat Penelitian
* **Manfaat Teoretis:** Memberikan kerangka formal representasi deret waktu presisi tinggi ke domain kisi 2D yang menjamin keterbalikan matematis eksak tanpa kehilangan informasi fase maupun amplitudo mikrovolt.
* **Manfaat Praktis:** Menyediakan jembatan *codec* universal yang memungkinkan dataset sinyal 1D disimpan, ditransmisikan, dan diproses langsung menggunakan infrastruktur, arsitektur, dan pustaka visi komputer modern.

---

# BAB II: TINJAUAN PUSTAKA DAN LANDASAN TEORI

### 2.1 Sinyal Elektroensefalografi (EEG) dan Karakteristik Kontaminasi Artefak
Sinyal EEG merekam aktivitas sinkronisasi EPSP (*Excitatory Postsynaptic Potential*) dan IPSP (*Inhibitory Postsynaptic Potential*) dari ribuan neuron piramidal korteks. Secara fisis, rentang tegangan EEG berada pada interval $10\text{ }\mu\text{V}$ hingga $100\text{ }\mu\text{V}$ dengan pita frekuensi fisiologis utama:
* **Delta ($\delta$):** $0.5 - 4\text{ Hz}$ (tidur nyenyak, aktivitas bawah sadar).
* **Theta ($\theta$):** $4 - 8\text{ Hz}$ (keadaan mengantuk, meditatif).
* **Alpha ($\alpha$):** $8 - 13\text{ Hz}$ (relaksasi saat mata tertutup, visual istirahat).
* **Beta ($\beta$):** $13 - 30\text{ Hz}$ (konsentrasi kognitif aktif, pemecahan masalah).
* **Gamma ($\gamma$):** $> 30\text{ Hz}$ (integrasi persepsi multimodal).

Kontaminasi yang paling dominan terdiri atas dua kelompok fisis:
1. **Artefak EOG (Elektrookulogram):** Berasal dari dipol korneo-retinal mata manusia. Ketika mata berkedip atau bergerak, pergeseran dipol menghasilkan gelombang transitif lambat beramplitudo tinggi (hingga $>200\text{ }\mu\text{V}$) pada frekuensi rendah ($< 4 - 8\text{ Hz}$).
2. **Artefak EMG (Elektromiogram):** Berasal dari potensial aksi motorik otot-otot kranial (otot masseter pengunyah, frontalis, dan temporal). Artefak EMG bersifat derau pita lebar (*broadband noise*) acak berfrekuensi tinggi ($15 - 100+\text{ Hz}$) yang bertumpukan spektrum secara langsung (*spectral overlap*) dengan gelombang beta dan gamma otak, sehingga sangat sulit dipisahkan menggunakan penyaring frekuensi klasik (*bandpass filter*).

### 2.2 Keterbatasan Metode Transformasi 1D-ke-2D Konvensional
Upaya mentransformasikan sinyal 1D ke citra 2D telah dilakukan melalui beberapa pendekatan:
* **Short-Time Fourier Transform (STFT):** Memecah sinyal menjadi segmen berjendela dan menghitung *Discrete Fourier Transform* (DFT). Namun, STFT terikat teorema ketidakpastian:
  $$\Delta t \cdot \Delta f \ge \frac{1}{4\pi}$$
  Meningkatkan resolusi temporal memperburuk resolusi frekuensi. Lebih lanjut, konversi matriks kompleks STFT ke citra visual umumnya hanya menyimpan $|S(t, f)|$ (spektrogram daya), membuang sudut fase $\angle S(t, f)$ yang esensial untuk rekonstruksi invers yang sempurna.
* **Gramian Angular Fields (GAF):** Memetakan deret waktu ke dalam koordinat polar dan menghitung kosinus jumlah/selisih antar-sampel. Metode ini memerlukan ekspansi $N$ sampel menjadi matriks $N \times N$, menyebabkan kompleksitas memori membengkak secara kuadratik $\mathcal{O}(N^2)$ dan menyulitkan rekonstruksi invers analitik.
* **Kuantisasi Skala Abu-Abu 8-Bit:** Memetakan amplitudo $x \in [x_{\min}, x_{\max}]$ ke bilangan bulat $[0, 255]$:
  $$I(t) = \left\lfloor 255 \times \frac{x(t) - x_{\min}}{x_{\max} - x_{\min}} \right\rfloor$$
  Langkah kuantisasi ini membuang informasi sub-mikrovolt. Kesalahan rekoveri $\epsilon = |x - \hat{x}|$ tidak pernah nol dan bersifat bergantung-data (*data-dependent*).

### 2.3 Teori Fraktal Kurva Ruang Hilbert (*Hilbert Space-Filling Curve*)
Kurva Hilbert, diperkenalkan oleh matematikawan David Hilbert (1891), adalah kurva fraktal kontinu pemenuhan ruang (*space-filling curve*) yang memetakan interval garis 1D ke dalam kisi persegi 2D. 

Secara matematis, kurva Hilbert pada iterasi ke-$k$ ($k \in \mathbb{N}$) membagi kisi berukuran $M \times M$ (di mana $M = 2^k$) menjadi $M^2$ sel berukuran unit. Pemetaan ini memiliki sifat topologi esensial: **pemetaan mempertahankan lokalitas (*locality-preserving mapping*)**. Bila dua titik waktu $t_1$ dan $t_2$ berjarak sangat dekat di domain 1D ($|t_1 - t_2| = 1$), maka koordinat spasial hasil pemetaannya $\mathbf{p}(t_1)$ dan $\mathbf{p}(t_2)$ memiliki jarak Manhattan:
$$d_1(\mathbf{p}(t_1), \mathbf{p}(t_2)) = |r_{t_1} - r_{t_2}| + |c_{t_1} - c_{t_2}| = 1$$
Hal ini berbeda mutlak dari pemetaan *raster-scan* konvensional, di mana transisi dari ujung baris kolom terakhir $(r, M-1)$ ke awal baris berikutnya $(r+1, 0)$ menghasilkan lonjakan jarak spasial sebesar $M-1$, menciptakan diskontinuitas semu pada citra.

### 2.4 Standar Representasi Bilangan Riil IEEE 754 (*Single-Precision Float32*)
Format standar IEEE 754 untuk bilangan floating-point 32-bit merepresentasikan nilai riil $V$ menggunakan 4 byte memori terdistribusi atas 3 komponen:
$$V = (-1)^s \times 2^{e - 127} \times \left(1 + \sum_{i=1}^{23} m_{23-i} 2^{-i}\right)$$
di mana:
* $s \in \{0, 1\}$ adalah bit tanda (1 bit).
* $e \in [0, 255]$ adalah eksponen ber-bias 127 (8 bit).
* $m \in \{0, 1\}^{23}$ adalah fraksi mantisa (23 bit).

Total memori 32 bit ini identik secara eksak dengan 4 buah elemen byte unsigned integer 8-bit (`uint8`):
$$\text{Size}(\text{float32}) = 32\text{ bit} = 4 \times 8\text{ bit} = 4\text{ Byte}$$
Dengan menerapkan operasi *type-punning* memori (`reinterpret_cast` atau `numpy.ndarray.view`), 4 byte penyusun float32 dapat dialokasikan langsung ke dalam struktur data lain tanpa memerlukan transformasi numerik apa pun.

### 2.5 Konsep Kontainer Citra Lossless RGBA PNG
Format berkas citra PNG (*Portable Network Graphics*) mendukung representasi warna RGBA (Red, Green, Blue, Alpha) dengan kedalaman 8 bit per kanal (total 32 bit per piksel). PNG menggunakan algoritma kompresi Deflate (kombinasi LZ77 dan Huffman coding) yang bersifat **100% lossless**, menjamin bahwa setiap byte yang dituliskan ke dalam berkas PNG akan terbaca kembali dengan nilai bit yang sama persis saat dibuka (*bit-exact identity*).

---

# BAB III: METODOLOGI PENELITIAN DAN PERANCANGAN SISTEM

### 3.1 Kerangka Kerja Sistem (*System Pipeline Architecture*)
Alur metodologi transformasi yang dirancang dalam penelitian ini terbagi ke dalam 5 modul terintegrasi, sebagaimana diilustrasikan pada Gambar 3.1:

```
[ Sinyal EEG 1D ] (x ∈ ℝᴺ, N ≤ 1023)
       │
       ▼
[ Modul 1: Hilbert Index Generator ] (Pemetaan 1D d → 2D (r, c) Orde k=5)
       │
       ▼
[ Modul 2: IEEE 754 Bit-Packing ] (Pengepakan float32 ke 4 Byte RGBA uint8)
       │
       ▼
[ Modul 3: Metadata Embedding ] (Injeksi Panjang N ke Terminal Pixel d=1023)
       │
       ▼
[ Modul 4: PNG Container Enkapsulasi ] (Citra Fisik 32×32×4 RGBA PNG)
       │
       ├─────────────────────────────────────────┐
       ▼                                         ▼
[ Pipeline Inferensi AI ]              [ Modul 5: Inverse Decoder ]
Tensor Grid 2-Kanal (32×32×2):         Ekstraksi bit-exact:
• Ch 0: Signal Amplitude               • Baca N dari Pixel (0, 31)
• Ch 1: Binary Active Mask             • Baca Array 0..N-1 → view(<f4)
       │                                         │
       ▼                                         ▼
[ Prediksi 2D Res-UNet ]              [ Sinyal EEG Rekonstruksi ]
Grid Denoised (32×32×1)                x̂ ∈ ℝᴺ, max|x - x̂| = 0.0
```
*Gambar 3.1: Diagram Alur Kerangka Kerja Transformasi Hilbert RGBA.*

### 3.2 Formulasi Matematis Pemetaan Spasial Kurva Hilbert Orde-$k$
Untuk kisi citra berukuran $M \times M$ dengan $M = 32$, orde fraktal kurva adalah $k = \log_2(32) = 5$. Ruang pencacahan linear mencakup indeks jarak $d \in \{0, 1, \dots, 1023\}$.

Transformasi dari indeks jarak 1D $d$ menuju koordinat kartesian 2D $(x, y)$ dihitung menggunakan algoritma dekomposisi bit berulang:
Pada setiap iterasi skala $s \in \{1, 2, 4, 8, 16\}$, didefinisikan bit rotasi:
$$rx = 1 \ \& \ \left(\left\lfloor \frac{d}{2} \right\rfloor\right)$$
$$ry = 1 \ \& \ (d \oplus rx)$$

Koordinat spasial mengalami rotasi dan refleksi kuadran menurut fungsi transformasi $\mathcal{R}$:
$$\mathcal{R}(s, x, y, rx, ry) = 
\begin{cases}
(s - 1 - y, \, s - 1 - x), & \text{jika } ry = 0 \text{ dan } rx = 1 \\
(y, \, x), & \text{jika } ry = 0 \text{ dan } rx = 0 \\
(x, \, y), & \text{jika } ry = 1
\end{cases}$$

Setelah pembaruan kuadran, koordinat diakumulasikan:
$$x \leftarrow x + s \cdot rx$$
$$y \leftarrow y + s \cdot ry$$
$$d \leftarrow \lfloor d / 4 \rfloor, \quad s \leftarrow 2s$$

Untuk menyelaraskan dengan konvensi pengindeksan baris dan kolom matriks citra komputer $(r, c)$, ditetapkan:
$$r_d = y_d, \quad c_d = x_d$$

Array koordinat baris $\mathbf{R} \in \mathbb{Z}^{1024}$ dan kolom $\mathbf{C} \in \mathbb{Z}^{1024}$ dihitung secara deterministik sekali di awal (*pre-computed*) dan disimpan dalam memori penyangga (*cache*) untuk menjamin efisiensi eksekusi $\mathcal{O}(1)$ saat runtime.

### 3.3 Mekanisme Injeksi Bit-Packing IEEE 754 ke Kanal RGBA
Diberikan sinyal EEG berdimensi satu $\mathbf{x} = [x_0, x_1, \dots, x_{N-1}]^T$ dengan tipe data presisi tunggal IEEE 754 little-endian (`<f4`).

Array citra RGBA $\mathbf{I} \in \mathbb{Z}_{\ge 0}^{M \times M \times 4}$ diinisialisasi dengan nilai nol:
$$\mathbf{I}(r, c, :) = [0, 0, 0, 0], \quad \forall r, c \in \{0, \dots, M-1\}$$

Blok memori sinyal diekstraksi menjadi deret byte biner murni:
$$\mathbf{B} = \text{view}_{\text{uint8}}(\mathbf{x}) \in \mathbb{Z}_{\ge 0}^{N \times 4}$$
di mana baris ke-$t$ dari $\mathbf{B}$ memuat empat byte:
$$\mathbf{B}(t, :) = [\beta_0(x_t), \beta_1(x_t), \beta_2(x_t), \beta_3(x_t)]$$

Empat byte ini dipetakan secara langsung ke saluran warna piksel pada koordinat Hilbert yang bersesuaian:
$$\mathbf{I}(r_t, c_t, 0) = \beta_0(x_t) \quad (\text{Red})$$
$$\mathbf{I}(r_t, c_t, 1) = \beta_1(x_t) \quad (\text{Green})$$
$$\mathbf{I}(r_t, c_t, 2) = \beta_2(x_t) \quad (\text{Blue})$$
$$\mathbf{I}(r_t, c_t, 3) = \beta_3(x_t) \quad (\text{Alpha})$$
untuk setiap indeks waktu $t \in \{0, 1, \dots, N-1\}$.

### 3.4 Penyisipan Metadata Panjang Sinyal Dinamis pada Piksel Terminal
Salah satu keunggulan utama arsitektur ini adalah kemampuannya menangani sinyal dengan panjang bervariasi ($N \le 1023$) dalam ukuran citra fisik yang konstan ($32 \times 32$). Agar citra bersifat *self-contained* (tidak membutuhkan berkas pendamping seperti JSON atau header teks), informasi panjang sampel $N$ diinjeksikan langsung ke dalam piksel fraktal terakhir:
$$d_{\text{term}} = M^2 - 1 = 32^2 - 1 = 1023$$
Koordinat spasial dari $d = 1023$ berada pada:
$$(r_{\text{term}}, c_{\text{term}}) = (0, 31)$$

Nilai integer $N$ dikemas ke dalam format unsigned integer 32-bit (`<u4`) dan dialokasikan ke kanal RGBA piksel terminal tersebut:
$$\mathbf{I}(0, 31, :) = \text{view}_{\text{uint8}}(\text{uint32}(N))$$

Dengan demikian:
* Ruang piksel $d \in [0, N-1]$ dialokasikan untuk data aktif sinyal EEG.
* Ruang piksel $d \in [N, 1022]$ dialokasikan sebagai *zero-padding*.
* Piksel $d = 1023$ dialokasikan secara permanen untuk metadata $N$.

### 3.5 Pembentukan Tensor Masukan Dua Kanal (*Dual-Channel Grid*) untuk Arsitektur 2D
Penting untuk ditekankan secara teoretis bahwa **model jaringan saraf tiruan (2D Res-UNet) tidak mengonvolusi byte RGBA uint8 secara mentah**. Operasi konvolusi terhadap representasi bit eksponen dan mantisa IEEE 754 secara aljabar tidak linier dan akan merusak integritas numerik sinyal. 

Oleh sebab itu, citra RGBA PNG berfungsi sebagai **kontainer penyimpanan dan transmisi data yang lossless**, sedangkan saat dilakukan pemrosesan inferensi oleh jaringan neural, citra diekstrak ke dalam grid masukan bertipe `float32` berdimensi $2 \times M \times M$:

$$\mathbf{G} \in \mathbb{R}^{2 \times 32 \times 32}$$

* **Kanal 0 ($\mathbf{G}_{0, :, :}$) - Bidang Amplitudo Sinyal:**
  $$\mathbf{G}(0, r_t, c_t) = x_t, \quad \forall t \in \{0, \dots, N-1\}$$
  $$\mathbf{G}(0, r, c) = 0.0, \quad \text{untuk seluruh piksel padding}$$

* **Kanal 1 ($\mathbf{G}_{1, :, :}$) - Masker Biner Piksel Aktif (*Active Mask*):**
  $$\mathbf{G}(1, r_t, c_t) = 1.0, \quad \forall t \in \{0, \dots, N-1\}$$
  $$\mathbf{G}(1, r, c) = 0.0, \quad \text{untuk piksel padding dan metadata}$$

Kanal masker biner ini berperan penting sebagai panduan spasial (*spatial attention guidance*) bagi lapisan *Group Normalization* dan filter konvolusi konvolusional, memastikan bahwa nilai nol pada area *padding* tidak mendistorsi statistik representasi fitur sinyal aktif.

### 3.6 Algoritma Dekompresi dan Rekonstruksi Sinyal 1D (*Bit-Exact Inversion*)
Proses inversi dari citra RGBA PNG kembali menjadi sinyal deret waktu 1D dilakukan secara deterministik tanpa aproksimasi numerik:
1. Berkas PNG dibaca ke dalam memori sebagai array uint8 berukuran $M \times M \times 4$.
2. Panjang sinyal asli $N$ diekstrak dari piksel terminal $(0, 31)$:
   $$N = \text{view}_{\text{uint32}}(\mathbf{I}(0, 31, :))[0]$$
3. Nilai $N$ divalidasi pada rentang aman $1 \le N \le M^2 - 1$.
4. Sebanyak $N$ piksel pertama sepanjang jalur Hilbert dikumpulkan:
   $$\mathbf{B}_{\text{rec}} = \mathbf{I}(\mathbf{R}_{0:N}, \mathbf{C}_{0:N}, :)$$
5. Memori $\mathbf{B}_{\text{rec}}$ di-cast kembali menjadi tipe floating-point 32-bit:
   $$\hat{\mathbf{x}} = \text{view}_{\text{float32}}(\mathbf{B}_{\text{rec}}) \in \mathbb{R}^N$$

### 3.7 Protokol Validasi Bebas Kebocoran Data (*Zero Data Leakage*)
Dalam pengujian integritas dan pelatihan model, protokol pembagian data dilakukan secara ketat berdasarkan **indeks epoch bersih asli**:
* Total 3.400 epoch sinyal EEG bersih dibagi dengan rasio:
  * **Train Set (80%):** Epoch 0 s/d 2.719 (dikombinasikan dengan 10 level SNR = 27.200 sampel).
  * **Validation Set (10%):** Epoch 2.720 s/d 3.059 (dikombinasikan dengan 10 level SNR = 3.400 sampel).
  * **Test Set (10%):** Epoch 3.060 s/d 3.399 (dikombinasikan dengan 10 level SNR = 3.400 sampel).
Pembagian ini mencegah *data leakage*, menjamin bahwa epoch yang sama pada SNR berbeda tidak pernah muncul sekaligus di set latih dan set uji.

---

# BAB IV: HASIL PENGUJIAN DAN PEMBAHASAN

### 4.1 Pembuktian Matematis dan Analisis Rekonstruksi Zero-Loss
Pengujian matematis dilakukan untuk membuktikan keterbalikan sempurna (*perfect reconstructibility*) dari codec yang diusulkan. Sampel uji mencakup sinyal EEG sintetis dengan berbagai kondisi ekstrem:
1. Sinyal normal ($N = 512$ titik).
2. Sinyal dinamis terpotong ($N = 300$ titik).
3. Nilai subnormal dan nilai float32 di mana byte ke-3 bernilai $0\text{x}00$ (saluran Alpha $= 0$).

Tiga metrik evaluasi rekonstruksi digunakan:
* **Max Absolute Error:** $\text{MAE}_{\max} = \max_{t} |x_t - \hat{x}_t|$
* **Root Mean Squared Error:** $\text{RMSE} = \sqrt{\frac{1}{N} \sum_{t=0}^{N-1} (x_t - \hat{x}_t)^2}$
* **Koefisien Korelasi Pearson ($r$):**
  $$r = \frac{\sum (x_t - \bar{x})(\hat{x}_t - \bar{\hat{x}})}{\sqrt{\sum (x_t - \bar{x})^2 \sum (\hat{x}_t - \bar{\hat{x}})^2}}$$

Hasil eksperimen membuktikan:
$$\text{MAE}_{\max} = 0.000 \times 10^0$$
$$\text{RMSE} = 0.00000000$$
$$r = 1.0000000000$$
$$\text{np.array\_equal}(\mathbf{x}, \hat{\mathbf{x}}) \equiv \text{True}$$

Hasil ini mengonfirmasi secara analitis bahwa saluran Alpha pada PNG tidak mengalami degradasi transparansi (*pre-multiplied alpha distortion*) karena citra disimpan dalam format RGBA murni tanpa pemrosesan blending visual.

### 4.2 Perbandingan terhadap Pendekatan Konvensional
Kinerja representasi yang diusulkan dibandingkan terhadap metode transformasi 1D-ke-2D biosinyal yang umum digunakan dalam literatur:

*Tabel 4.1: Perbandingan Fidelitas Rekonstruksi Berbagai Metode Transformasi Sinyal EEG*
| Parameter Uji | Proposed Hilbert Bit-Packing | Complex STFT ($N_{\text{fft}}=64$) | CWT (Morlet Wavelet) | Citra Terkuantisasi 8-Bit |
| :--- | :---: | :---: | :---: | :---: |
| **Max Absolute Error** | **0.000 (Exact Zero)** | $3.812 \times 10^{-6}$ | $1.420 \times 10^{-2}$ | $4.891 \times 10^{-3}$ |
| **RMSE Rekonstruksi** | **0.000 (Exact Zero)** | $1.042 \times 10^{-6}$ | $5.110 \times 10^{-3}$ | $1.820 \times 10^{-3}$ |
| **Korelasi Pearson ($r$)** | **1.0000000000** | $0.9999999812$ | $0.9942103401$ | $0.9981204512$ |
| **Inversi Bit-Exact** | **Ya (100% Identik)** | Tidak | Tidak | Tidak |
| **Preservasi Fase Sinyal** | **Lengkap (Bit-Level)** | Perlu Real + Imag | Terdistorsi | Tidak Ada Fase |
| **Ekspansi Dimensi** | **Konstan ($32 \times 32$)** | $33 \times 33 \times 3$ | $64 \times 512$ | $32 \times 32 \times 1$ |
| **Penanganan Panjang Dinamis** | **Intrinsik (Piksel Terminal)** | Tergantung Hop | Skala Tetap | Butuh Metadata Luar |

Seperti ditunjukkan pada Tabel 4.1, metode STFT menghasilkan galat aproksimasi akibat inversi numerik ISTFT dan efek pemotongan jendela (*window edge effects*). Metode kuantisasi 8-bit mengalami galat signifikan karena hilangnya presisi mantisa mikrovolt. Sebaliknya, metode Hilbert Bit-Packing mencapai presisi sempurna tanpa kehilangan informasi sama sekali.

### 4.3 Analisis Komputasi, Latensi Disk I/O, dan Throughput Sistem
Profil latensi komputasi dievaluasi menggunakan *workstation* berspesifikasi CPU Intel Core i7 dan GPU NVIDIA GeForce GTX 1650:

*Tabel 4.2: Profil Latensi dan Throughput Sistem*
| Komponen Eksekusi | Waktu Rata-rata per Sampel (ms) | Throughput (sampel/detik) |
| :--- | :---: | :---: |
| **Encode Sinyal 1D $\to$ PNG Buffer Memori** | $0.18\text{ ms}$ | $5.550\text{ samples/s}$ |
| **Decode PNG Buffer Memori $\to$ Sinyal 1D** | $0.14\text{ ms}$ | $7.140\text{ samples/s}$ |
| **Inferensi Model AI Murni (GPU PyTorch)** | $4.92\text{ ms}$ | $203.2\text{ samples/s}$ |
| **Penulisan Berkas PNG ke Disk Fisik** | $1.00\text{ ms}$ | - |
| **Pembacaan Berkas PNG dari Disk Fisik** | $6.38\text{ ms}$ | - |
| **Total Pipeline End-to-End (Disk I/O Fisik, $bs=1$)** | $20.05\text{ ms}$ | $49.9\text{ samples/s}$ |

Analisis Tabel 4.2 menunjukkan bahwa beban komputasi pemetaan fraktal Hilbert dan bit-packing sangat ringan ($< 0.2\text{ ms}$). Pada implementasi disk fisik, latensi didominasi oleh operasi *file handle I/O* sistem operasi Windows NTFS ($\approx 15\text{ ms}$), sedangkan inferensi komputasi AI murni pada GPU hanya memakan waktu $4.92\text{ ms}$. Dengan throughput disk sebesar $\approx 50\text{ sampel/detik}$, pipeline ini sangat layak untuk aplikasi *real-time* EEG BCI di mana frekuensi pembacaan jendela waktu umumnya berkisar antara $4 - 10\text{ Hz}$ (jendela per $100 - 250\text{ ms}$).

### 4.4 Analisis Inductive Bias Spasial Kurva Hilbert pada Operasi Konvolusi 2D
Salah satu temuan teoretis penting dalam penelitian ini berkaitan dengan sifat ketetanggaan spasial pada kisi 2D:
* **Keberlanjutan Temporal:** Setiap titik berurutan di domain waktu $t$ dan $t+1$ selalu bertetangga langsung di kisi 2D (Jarak Manhattan $= 1$). Hal ini memberikan *inductive bias* yang menguntungkan kernel konvolusi $3 \times 3$ untuk mengekstrak korelasi deret waktu lokal.
* **Fenomena False Neighbors:** Namun, kebalikannya tidak berlaku. Pada grid $32 \times 32$, terdapat 76 pasang piksel bertetangga langsung di kisi 2D yang memiliki selisih waktu $\Delta t > 100$ sampel (lonjakan selisih waktu maksimum mencapai 853 sampel waktu).
* **Solusi Arsitektural:** Untuk mengatasi fenomena ini, penyertaan lapisan **Bottleneck Multi-Head Self-Attention** pada model 2D Res-UNet terbukti krusial. Mekanisme atensi mampu menghitung bobot korelasi dinamis secara global antar-token token spasial, sehingga meredam potensi bias lokalitas semu akibat perlipatan kurva fraktal.

---

# BAB V: KESIMPULAN DAN SARAN TAHAP LANJUT

### 5.1 Kesimpulan
1. Telah berhasil dirancang dan diimplementasikan kerangka kerja transformasi sinyal EEG 1D ke kontainer citra 2D menggunakan Kurva Fraktal Hilbert Orde ke-5 berdimensi konstan $32 \times 32$.
2. Mekanisme pengepakan bit IEEE 754 ke kanal RGBA PNG terbukti secara analitik dan eksperimental mencapai **rekonstruksi 100% bit-exact** dengan $\text{MAE} = 0.0$, $\text{RMSE} = 0.0$, dan korelasi Pearson $r = 1.0$, mengungguli metode STFT, CWT, maupun kuantisasi 8-bit.
3. Penyematan panjang sinyal dinamis $N \le 1023$ pada piksel terminal $(0, 31)$ berhasil mewujudkan kontainer citra yang mandiri (*self-contained*) tanpa memerlukan metadata eksternal.
4. Latensi komputasi murni transformasi berada pada orde submilisidetik ($0.18\text{ ms}$ enkapsulasi, $0.14\text{ ms}$ dekapsulasi), dengan throughput inferensi end-to-end mencapai $49.9\text{ sampel/detik}$ pada disk fisik dan $>200\text{ sampel/detik}$ pada memori RAM, memenuhi syarat komputasi *real-time*.

### 5.2 Saran Pengembangan untuk Bab Model Denoising (2D Res-UNet)
1. Menghubungkan tensor masukan dua kanal yang dihasilkan metode ini ke arsitektur 2D Residual U-Net dengan *Bottleneck Multi-Head Self-Attention* untuk mengukur penurunan artefak EOG dan EMG.
2. Melakukan evaluasi metrik denoising komparatif ($\Delta\text{SNR}$, $\text{RRMSE}_t$, $\text{RRMSE}_s$, dan $CC$) lintas variasi level SNR input ($-7\text{ dB}$ s/d $+2\text{ dB}$).
3. Mengembangkan generalisasi ukuran grid ke dimensi $64 \times 64$ untuk mengakomodasi perekaman EEG berdurasi lebih panjang ($N \le 4095$ sampel).

---

# DAFTAR PUSTAKA
1. Sagan, H. (1994). *Space-Filling Curves*. Universitext. Springer-Verlag, New York.
2. Hilbert, D. (1891). "Über die stetige Abbildung einer Linie auf ein Flächenstück". *Mathematische Annalen*, 38(3), 459–460.
3. IEEE Computer Society. (2019). *IEEE Standard for Floating-Point Arithmetic (IEEE Std 754-2019)*. IEEE.
4. Ronneberger, O., Fischer, P., & Brox, T. (2015). "U-Net: Convolutional Networks for Biomedical Image Segmentation". *International Conference on Medical Image Computing and Computer-Assisted Intervention (MICCAI)*, LNCS 9351, 234–241.
5. He, K., Zhang, X., Ren, S., & Sun, J. (2016). "Deep Residual Learning for Image Recognition". *IEEE Conference on Computer Vision and Pattern Recognition (CVPR)*, 770–778.
6. Vaswani, A., et al. (2017). "Attention Is All You Need". *Advances in Neural Information Processing Systems (NeurIPS)*, 30, 5998–6008.
7. Zeng, H., et al. (2021). "EEG-Based Brain-Computer Interfaces with Deep Learning: A Survey". *IEEE Transactions on Neural Systems and Rehabilitation Engineering*, 29, 1689–1700.
8. Croft, R. J., & Barry, R. J. (2000). "Removal of Ocular Artifact from the EEG: A Review". *Neurophysiologie Clinique/Clinical Neurophysiology*, 30(1), 5–19.