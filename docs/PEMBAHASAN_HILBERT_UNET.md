# PEMBAHASAN MENDALAM: CARA KERJA MODEL 2D HILBERT RES-UNET + ATTENTION
*(Versi Naratif & Konseptual Bebas Rumus Matematis)*

Dokumen ini memuat **pembahasan mendalam mengenai arsitektur, mekanisme kerja, dan alasan pemilihan komponen pada model Hilbert UNet 2D**. Seluruh isi disajikan secara naratif dan konseptual tanpa menggunakan rumus matematis, sehingga siap disadur langsung ke dalam naskah pembahasan tesis maupun bahan presentasi sidang.

---

## DAFTAR ISI
1. [Filosofi Desain: Mengapa Menggunakan Arsitektur U-Net 2D?](#1-filosofi-desain-mengapa-menggunakan-arsitektur-u-net-2d)
2. [Desain Masukan Dua Kanal dan Tujuannya](#2-desain-masukan-dua-kanal-dan-tujuannya)
3. [Alur Perjalanan Data di Dalam Model](#3-alur-perjalanan-data-di-dalam-model)
4. [Bedah Komponen dan Alasan Pemilihannya](#4-bedah-komponen-dan-alasan-pemilihannya)
   - 4.1 Blok Residual Kustom (ResBlock2D)
   - 4.2 Jalur Penurunan Ukuran (Enkoder)
   - 4.3 Lapisan Atensi Mandiri di Titik Terbawah (Bottleneck Self-Attention)
   - 4.4 Jalur Pembesaran Kembali dan Jalan Pintas (Dekoder & Skip Connections)
   - 4.5 Kepala Proyeksi Akhir dan Rekonstruksi Sinyal
5. [Strategi Evaluasi Ganda: Menjaga Bentuk Gelombang dan Irama Otak](#5-strategi-evaluasi-ganda-menjaga-bentuk-gelombang-dan-irama-otak)
6. [Analisis Fisis: Mengapa Model Sangat Efektif Membuang Artefak?](#6-analisis-fisis-mengapa-model-sangat-efektif-membuang-artefak)

---

## 1. FILOSOFI DESAIN: MENGAPA MENGGUNAKAN ARSITEKTUR U-NET 2D?

Setelah sinyal EEG satu dimensi dipetakan ke dalam kisi dua dimensi berukuran 32×32 melalui Kurva Hilbert, tantangan pembersihan derau tidak lagi diperlakukan sebagai pemfilteran deret waktu klasik, melainkan ditransformasikan menjadi **permasalahan restorasi citra**.

### Analogi Restorasi Foto Bersejarah yang Rusak
Proses kerja model ini sangat mirip dengan pekerjaan seorang kurator yang merestorasi foto tua yang tertutup noda kotoran:
1. **Melihat Gambaran Garis Besar (Sisi Kiri U-Net / Enkoder):**
   Model tidak langsung terburu-buru membersihkan setiap titik kecil, melainkan memperkecil ukuran tampilan kanvas secara bertahap. Tujuannya adalah meredam gangguan lokal yang acak dan menangkap garis besar bentuk gelombang otak serta pola noda artefaknya.
2. **Mata Pengawas Global (Titik Terbawah / Bottleneck):**
   Pada tampilan paling ringkas, model mengaktifkan mekanisme atensi mandiri untuk melihat keseluruhan konteks sinyal dari awal hingga akhir secara serentak. Model memastikan bahwa bentuk gelombang di satu bagian cocok dan berkesinambungan dengan bagian lainnya.
3. **Menambal dan Mengembalikan Ketajaman (Sisi Kanan U-Net / Dekoder):**
   Kanvas diperbesar kembali ke resolusi aslinya. Agar detail puncak gelombang otak yang tajam tidak ikut memudar atau menjadi kabur, model mengambil salinan gelombang asli dari tahap awal melalui jalan pintas (*skip connection*), lalu membuang komponen deraunya saja.
4. **Hasil Akhir:**
   Kanvas yang sudah bersih ditelusuri kembali mengikuti jalur Kurva Hilbert semula, menghasilkan sinyal deret waktu satu dimensi yang utuh dan jernih.

---

## 2. DESAIN MASUKAN DUA KANAL DAN TUJUANNYA

Model dirancang untuk menerima data masukan berupa kisi dua dimensi berukuran 32×32 dengan **dua kanal informasi**:

### Kanal Pertama: Bidang Nilai Amplitudo Sinyal
Kanal ini memuat nilai amplitudo tegangan mikrovolt sinyal EEG yang tersusun rapi mengikuti jalur Kurva Hilbert. Karena panjang sinyal yang diolah adalah 512 titik, maka 512 titik pertama pada kurva terisi oleh nilai sinyal asli, sedangkan sisa ruang pada kisi (sampel 512 hingga 1023) dibiarkan bernilai nol sebagai ruang pengisi (*padding*).

### Kanal Kedua: Bidang Masker Penanda Sinyal Aktif
Kanal ini berisi nilai biner sederhana, yaitu bernilai satu pada wilayah yang memuat data sinyal EEG aktif, dan bernilai nol pada seluruh wilayah pengisi (*padding*) serta piksel penyimpan metadata panjang sinyal di pojok kisi.

### Mengapa Kanal Masker Penanda Wajib Disediakan?
Kanal penanda ini berfungsi sebagai kacamata pemandu bagi lapisan konvolusi dan normalisasi di dalam model. Tanpa adanya masker ini, model akan salah mengira bahwa angka nol pada area *padding* adalah sinyal gelombang otak yang sedang datar bernilai nol mikrovolt. Dengan adanya masker, model secara tegas hanya memusatkan perhatian pada area yang bertanda satu dan mengabaikan area bertanda nol, sehingga statistik fitur tidak terdistorsi oleh ruang kosong.

---

## 3. ALUR PERJALANAN DATA DI DALAM MODEL

Perjalanan data di dalam model mengikuti struktur huruf "U" yang simetris dan seimbang:

```
[ Masukan 2 Kanal (32×32) ]
             │
             ▼
[ Lapisan Pembuka (Stem) ] ──► Memperluas 2 kanal masukan menjadi 24 kanal fitur
             │
┌────────────┴───────────────────────────────────────────────────────┐
│ ENKODER TINGKAT 1 (Ukuran 32×32, 24 Kanal Fitur)                   │───┐ (Jalan Pintas 1)
└────────────┬───────────────────────────────────────────────────────┘   │
             ▼ Penurunan Ukuran Spasial (Pengecilan menjadi 16×16)       │
┌────────────┴───────────────────────────────────────────────────────┐   │
│ ENKODER TINGKAT 2 (Ukuran 16×16, 48 Kanal Fitur)                   │─┐ │ (Jalan Pintas 2)
└────────────┬───────────────────────────────────────────────────────┘ │ │
             ▼ Penurunan Ukuran Spasial (Pengecilan menjadi 8×8)       │ │
┌────────────┴───────────────────────────────────────────────────────┐ │ │
│ ENKODER TINGKAT 3 (Ukuran 8×8, 96 Kanal Fitur)                     │┐│ │ (Jalan Pintas 3)
└────────────┬───────────────────────────────────────────────────────┘││ │
             ▼ Penurunan Ukuran Spasial (Pengecilan menjadi 4×4)      ││ │
                                                                      ││ │
┌────────────────────────────────────────────────────────────────────┐││ │
│ TITIK TERBAWAH (BOTTLENECK):                                       │││ │
│ • Pemadatan Fitur (Ukuran 4×4, 96 Kanal Fitur)                     │││ │
│ • Lapisan Atensi Mandiri Global (4 Kepala Atensi pada 16 Wilayah)  │││ │
│ • Penggabungan Kembali dengan Fitur Awal                           │││ │
└────────────────────────────┬───────────────────────────────────────┘││ │
                             │                                        ││ │
             ▲ Pembesaran Ukuran Spasial (Penaikan menjadi 8×8)       ││ │
┌────────────┴────────────────────────────────────────────────────────┴┼─┘
│ DEKODER TINGKAT 3 (Ukuran 8×8, Penggabungan dengan Jalan Pintas 3)   │
└────────────┬─────────────────────────────────────────────────────────┘
             ▲ Pembesaran Ukuran Spasial (Penaikan menjadi 16×16)      │
┌────────────┴─────────────────────────────────────────────────────────┴─┐
│ DEKODER TINGKAT 2 (Ukuran 16×16, Penggabungan dengan Jalan Pintas 2)   │
└────────────┬───────────────────────────────────────────────────────────┘
             ▲ Pembesaran Ukuran Spasial (Penaikan menjadi 32×32)        │
┌────────────┴───────────────────────────────────────────────────────────┴─┐
│ DEKODER TINGKAT 1 (Ukuran 32×32, Penggabungan dengan Jalan Pintas 1)     │
└────────────┬─────────────────────────────────────────────────────────────┘
             │
             ▼
[ Kepala Proyeksi Akhir ] ──► Meringkas 24 kanal menjadi 1 kanal estimasi bersih (32×32)
             │
             ▼
[ Penelusuran Balik Jalur Hilbert ] ──► Menghasilkan kembali sinyal EEG 1D yang utuh
```

---

## 4. BEDAH KOMPONEN DAN ALASAN PEMILIHANNYA

### 4.1 Blok Residual Kustom (ResBlock2D)
Seluruh blok pemrosesan pada jalur enkoder maupun dekoder dibangun menggunakan blok residual khusus. Di dalam blok ini terdapat beberapa komponen penting:

1. **Normalisasi Kelompok (Group Normalization):**
   * *Alasan tidak memakai Batch Normalization:* Normalisasi kelompok membagi kanal fitur menjadi 8 kelompok kecil dan menstabilkan persebaran nilai secara mandiri pada setiap sampel sinyal. Teknik ini tidak bergantung pada banyaknya data dalam satu kelompok latih, sehingga tetap bekerja sangat stabil saat model mengolah sinyal satu per satu dalam pengujian langsung.
2. **Fungsi Aktivasi GELU:**
   * *Alasan tidak memakai ReLU konvensional:* Sinyal EEG selalu berayun naik turun melintasi garis tengah, memiliki lembah negatif dan puncak positif yang sama-sama penting. Fungsi ReLU biasa langsung mematikan semua nilai negatif menjadi nol, sehingga merusak informasi lembah gelombang. Aktivasi GELU membentuk lengkungan aktivasi yang halus dan membiarkan nilai negatif kecil tetap mengalirkan informasi.
3. **Lapisan Konvolusi Dua Dimensi:**
   * Bertugas menangkap keterkaitan pola antar piksel yang berdampingan dalam jendela lokal berukuran tiga kali tiga piksel.
4. **Pencegah Kejenuhan (Dropout Ringan 5%):**
   * Memberikan efek pengacakan ringan selama proses belajar agar model tidak sekadar menghafal bentuk derau dari data latih, melainkan mampu membersihkan derau dari rekaman sinyal yang baru.
5. **Jalur Pintas Residual:**
   * Fitur masukan awal dijumlahkan secara langsung dengan hasil olahan blok. Jalur pintas ini menciptakan jalan tol bagi aliran informasi, mencegah melemahnya daya belajar model (*vanishing gradient*), dan membuat model berfokus hanya mempelajari pola derau yang perlu disingkirkan.

---

### 4.2 Jalur Penurunan Ukuran (Enkoder)
Jalur enkoder bertugas menyaring informasi secara bertingkat:
* **Lapisan Pembuka:** Mengubah 2 kanal data awal menjadi 24 kanal fitur representasi dasar.
* **Penurunan Bertahap:** Ukuran kanvas diperkecil sebanyak tiga kali secara berurutan, yaitu dari 32×32 menjadi 16×16, lalu menjadi 8×8, dan berakhir pada 4×4 piksel.
* **Peningkatan Kapasitas Fitur:** Bersamaan dengan mengecilnya luas kanvas, jumlah kanal representasi dilipatgandakan dari 24 menjadi 48, lalu mencapai 96 kanal.
* **Alasan Menggunakan Konvolusi Berlangkah Dua (bukan Pemotongan Rata-Rata / MaxPooling):**
  Teknik pemotongan kasar membuang sebagian besar informasi secara kaku. Dengan konvolusi berlangkah dua, model memiliki kebebasan untuk mempelajari sendiri cara terbaik merangkum informasi spasial tanpa menghilangkan ciri khas gelombang.

---

### 4.3 Lapisan Atensi Mandiri di Titik Terbawah (Bottleneck Self-Attention)
Pada titik kanvas terkecil (ukuran 4×4 dengan 96 kanal), model memasang lapisan atensi mandiri dengan empat kepala pengawas (*4 attention heads*).

#### Mengapa Lapisan Ini Menjadi Kunci Utama Keberhasilan Model?
Kurva Hilbert memiliki satu sifat unik yang memerlukan penanganan khusus:
* Dua titik waktu yang berdampingan pada sinyal asli dipastikan selalu berdampingan di kanvas dua dimensi.
* **Namun kebalikannya belum tentu benar (Masalah Tetangga Semu):** Dua piksel yang tampak bersebelahan di kanvas dua dimensi bisa jadi berasal dari waktu perekaman yang sangat berjauhan (misalnya titik awal sinyal berdekatan secara fisik dengan titik akhir sinyal akibat liukan kurva).
* Lapisan konvolusi lokal biasa hanya melihat jendela kecil di sekitarnya, sehingga berisiko mengira kedua titik tersebut berkaitan erat.

#### Solusi yang Diberikan oleh Lapisan Atensi Mandiri:
Kanvas berukuran 4×4 dipandang sebagai 16 wilayah spasial. Lapisan atensi mandiri memungkinkan ke-16 wilayah ini saling menimbang dan menilai keterkaitan satu sama lain secara serentak di seluruh kanvas. Model secara cerdas dapat mengenali: *"Kedua titik ini memang berdampingan di kanvas, tetapi karakteristik ritmenya tidak berhubungan, sehingga pengaruhnya harus dikesampingkan."* Kemampuan inilah yang menyatukan konteks awal dan akhir gelombang menjadi satu pemahaman utuh.

---

### 4.4 Jalur Pembesaran Kembali dan Jalan Pintas (Dekoder & Skip Connections)
Jalur dekoder bertugas mengembalikan kanvas ke ukuran resolusi semula:
* **Pembesaran Bertahap:** Ukuran kanvas dibesarkan kembali dari 4×4 menjadi 8×8, kemudian 16×16, hingga kembali utuh berukuran 32×32 piksel menggunakan lapisan konvolusi pembalik (*transposed convolution*).
* **Peran Penyelamat dari Jalan Pintas (Skip Connections):**
  Pada setiap tahap pembesaran, hasil kanvas yang baru diperbesar digabungkan dengan salinan fitur asli yang dikirim langsung dari jalur enkoder pada tingkat ukuran yang sama.

#### Mengapa Jalan Pintas Ini Mutlak Diperlukan?
Ketika sinyal dikompresi hingga ukuran terkecil di titik terdalam, informasi detail mengenai puncak-puncak tajam gelombang otak dan batas waktu yang presisi mengalami penurunan ketajaman. Jika kanvas dibesarkan kembali tanpa panduan, gelombang yang dihasilkan akan terlihat tumpul, datar, dan buram. Jalan pintas berfungsi mengalirkan kembali detail tajam dari rekaman awal, sehingga dekoder hanya bertugas membersihkan komponen kotorannya tanpa merusak ketajaman puncak gelombang otak asli.

---

### 4.5 Kepala Proyeksi Akhir dan Rekonstruksi Sinyal
Pada lapisan paling ujung:
1. Seluruh 24 kanal fitur yang telah dibersihkan dirampingkan menggunakan konvolusi titik menjadi **satu kanal tunggal berukuran 32×32**. Kanal tunggal ini merupakan wujud kisi dua dimensi dari sinyal EEG yang sudah terbebas dari artefak.
2. **Cara Mengambil Kembali Sinyal Satu Dimensi:**
   Sistem tidak membaca citra secara mendatar baris demi baris, melainkan **menelusuri kembali urutan jalur Kurva Hilbert semula** dari titik indeks ke-0 hingga titik ke-511. Nilai-nilai amplitudo yang ditemui sepanjang jalur tersebut diambil secara berurutan dan disusun kembali menjadi deret waktu satu dimensi berpanjang 512 sampel.

---

## 5. STRATEGI EVALUASI GANDA: MENJAGA BENTUK GELOMBANG DAN IRAMA OTAK

Selama proses pelatihan, kemampuan model dalam membersihkan sinyal diawasi menggunakan strategi evaluasi gabungan antara **domain waktu** dan **domain frekuensi**:

### Kriteria Pertama: Kesesuaian Bentuk Gelombang (Domain Waktu)
Kriteria ini membandingkan bentuk lekukan sinyal hasil keluaran model dengan sinyal bersih aslinya pada setiap titik waktu. Tujuannya adalah memastikan bahwa tinggi puncak dan kedalaman lembah gelombang hasil restorasi berada pada posisi amplitudo yang tepat.

### Kriteria Kedua: Keutuhan Ritme Frekuensi (Domain Frekuensi)
Jika pelatihan hanya berpatokan pada kriteria bentuk gelombang, model cenderung mengambil jalan pintas yang merugikan: model akan menghaluskan seluruh sinyal secara berlebihan (*over-smoothed*) agar nilai galat tampak kecil, padahal tindakan tersebut memotong frekuensi gelombang otak yang sebenarnya.
* Melalui kriteria kedua, kandungan spektrum frekuensi sinyal dihitung menggunakan transformasi Fourier.
* Kriteria ini memaksa model untuk membuang getaran derau sambil **tetap mempertahankan keutuhan irama alami gelombang otak manusia** (seperti gelombang alfa saat relaksasi, gelombang beta saat berpikir aktif, dan gelombang gama).

---

## 6. ANALISIS FISIS: MENGAPA MODEL SANGAT EFEKTIF MEMBUANG ARTEFAK?

Performa model dalam penelitian ini menunjukkan efektivitas pembersihan yang berbeda secara alami antara dua jenis gangguan utama:

### 1. Keberhasilan Sangat Tinggi pada Artefak Kedipan Mata (EOG)
* **Karakter Fisis:** Kedipan mata menghasilkan lonjakan gelombang beramplitudo raksasa namun bergerak lambat pada frekuensi sangat rendah (di bawah 4 hingga 8 Hertz).
* **Respon Model:** Pada kisi Kurva Hilbert, gelombang besar yang lambat ini membentuk gugusan pola spasial yang luas dan teratur. Pola ini sangat mudah dikenali dan diisolasi oleh filter konvolusi tingkat tinggi pada enkoder, sehingga model mampu mengangkat kualitas sinyal secara drastis dengan peningkatan rasio sinyal terhadap derau mencapai lebih dari 10 desibel dan tingkat kemiripan bentuk gelombang mendekati sempurna (korelasi di atas 0,89).

### 2. Tantangan dan Ketepatan pada Artefak Aktivitas Otot (EMG)
* **Karakter Fisis:** Kontraksi otot wajah atau rahang memancarkan getaran derau acak berfrekuensi tinggi (di atas 15 Hertz) yang bertumpukan secara langsung dengan frekuensi gelombang beta dan gama otak.
* **Respon Model:** Memisahkan derau otot jauh lebih menantang karena batas antara getaran otot dan aktivitas berpikir otak sangat tipis. Kehadiran pengawasan domain frekuensi memastikan model mampu meredam lonjakan tajam artefak otot tanpa memangkas irama beta dan gama asli otak, menghasilkan peningkatan kualitas sinyal sekitar 6,7 desibel.

### 3. Kecepatan dan Efisiensi Komputasi
Arsitektur model dirancang sangat ramping dengan total parameter 786.241 parameter (hanya berukuran sekitar 3,0 Megabyte). Kerampingan ini membuat komputasi inferensi murni pada unit pengolah grafis (GPU) berlangsung sangat cepat, yaitu hanya memerlukan waktu sekitar **4,9 milidetik per sampel sinyal**, menjadikannya sangat ideal untuk diterapkan pada sistem antarmuka otak-komputer (*Brain-Computer Interface*) secara langsung (*real-time*).