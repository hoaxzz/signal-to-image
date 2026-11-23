# Riset Restorasi Sinyal EEG Berbasis Transformasi Spasial Kurva Hilbert & IEEE 754 Bit-Packing

Repositori ini memuat metodologi, eksperimen komparasi representasi 1D-ke-2D, serta implementasi **Transformasi Sinyal EEG 1D ke Citra 2D Lossless berbasis Kurva Fraktal Hilbert Orde-5 ($32 \times 32$) dan Bit-Packing IEEE 754 (`float32` $\leftrightarrow$ `RGBA PNG`)** untuk penghilangan artefak (*denoising*) menggunakan arsitektur **2D Hilbert Res-UNet + Bottleneck Multi-Head Self-Attention**.

---

## Struktur Direktori Workspace

```text
D:\Tsaqib\
├── BAB_3_METODOLOGI_PENELITIAN_HILBERT.md         # [DOKUMEN UTAMA] Bab III Metodologi Penelitian Hilbert
├── README.md                                      # Dokumentasi struktur folder & panduan eksekusi
├── src/                                           # Modul kode sumber Python
│   ├── __init__.py
│   ├── hilbert_png_codec.py                       # Codec lossless 1D float32 <-> 2D Hilbert RGBA PNG (32×32×4)
│   ├── hilbert_png_denoiser.py                    # Arsitektur 2D Hilbert Res-UNet + Self-Attention & pipeline evaluasi
│   ├── compare_signal_to_image_methods.py         # Modul benchmark perbandingan metode transformasi sinyal ke citra
│   ├── test_lossless_eeg_image.py                 # Skrip uji rekonstruksi lossless Hilbert PNG
│   └── test_downstream.py                         # Skrip uji downstream tata letak spasial
├── notebooks/                                     # Notebook eksperimen & benchmark
│   ├── hilbert_png_denoiser.ipynb                 # Notebook utama pelatihan & evaluasi 2D Hilbert Res-UNet
│   ├── compare_signal_to_image_methods.ipynb      # Notebook benchmark 12 metode global transformasi 1D ke 2D
│   ├── compare_circular_mapping_methods.ipynb     # Notebook benchmark 15 varian Circular Mapping
│   └── syntetic-data-construction.ipynb           # Notebook konstruksi data EEG semi-sintetis
├── docs/                                          # Dokumentasi tesis & naskah ilmiah
│   ├── BAB_3_METODOLOGI_PENELITIAN_HILBERT.md     # Salinan Bab III Metodologi Penelitian Hilbert
│   ├── DRAFT_TESIS_TRANSFORMASI_HILBERT_EEG.md    # Draft Bab I - V Transformasi Hilbert EEG
│   ├── PEMBAHASAN_HILBERT_UNET.md                 # Pembahasan konseptual arsitektur 2D Hilbert Res-UNet
│   ├── HILBERT_TRANSFORMATION_PAPER_DRAFT.md      # Draft paper ilmiah Hilbert RGBA Bit-Packing
│   ├── SINTESIS_BENCHMARK_SIGNAL_TO_IMAGE_DAN_CIRCULAR_MAPPING.md
│   ├── METODE_TRANSFORMASI_SINYAL_KE_CITRA.md
│   ├── REKAP_KOMPREHENSIF_PEMBAHASAN_RISET.md
│   ├── UJI_AWAL_11_MODEL_10_VARIABEL.md
│   └── hilbert_step_diagram.drawio
├── outputs/                                       # Hasil keluaran eksperimen & visualisasi
│   ├── comparison_outputs/                        # Tabel CSV/JSON & grafik benchmark 12 metode 1D-ke-2D
│   ├── circular_outputs/                          # Tabel CSV/JSON & grafik benchmark 15 varian Circular Mapping
│   ├── pipeline_outputs/                          # Citra Hilbert PNG (input/denoised) & benchmark_results.json
│   └── demo_outputs/                              # Sampel uji konversi lossless
├── checkpoints/                                   # Bobot model terlatih 2D Hilbert Res-UNet (EOG & EMG)
└── Dataset/                                       # Dataset EEG semi-sintetis (EOG & EMG, bersih & multi-SNR)
```