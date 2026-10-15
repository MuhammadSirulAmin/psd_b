# Klasifikasi Tutupan Lahan Jawa Timur

Proyek ini mengolah mosaik Sentinel-2 dan poligon digitasi menjadi peta lima kelas: build-up, farm land, forest, mangrove, dan water body. Hasil ditampilkan melalui aplikasi Streamlit serta peta dan evaluasi pada notebook laporan.

## Menjalankan aplikasi Streamlit

Dari direktori proyek:

```powershell
pip install -r requirements.txt
streamlit run app.py
```

Kontrol layer pada peta memisahkan lima kelas prediksi raster dan lima layer sampel digitasi GeoJSON. Sampel berasal dari `labels_clean/` dan dapat dinyalakan atau dimatikan satu kelas pada satu waktu; sampel dimatikan secara default agar tidak tertukar dengan prediksi. Basemap awal adalah peta jalan OpenStreetMap; komposit Sentinel-2 lokal B04/B03/B02 dan Esri World Imagery tersedia sebagai pilihan citra. Raster citra dan klasifikasi diproyeksikan ke grid Web Mercator yang sama agar sejajar dengan basemap. Tampilan kelas memakai agregasi mayoritas dan filter kelompok piksel kecil pada resolusi peta untuk meredam bintik. Kelas `Water body` dibatasi ke piksel SCL 6 (air); prediksi air model di kelas SCL lain disembunyikan agar tidak menutupi daratan. Basemap beralih otomatis ke sumber lain jika tile gagal dimuat dan tetap terlihat meskipun semua overlay dimatikan. Data klasifikasi hanya tersedia dalam cakupan raster proyek.

## Menjalankan pipeline dan laporan

- `pipeline_klasifikasi.ipynb` adalah pengatur tahap alur kerja. Tahapan dijalankan satu per satu secara eksplisit; pelatihan ulang dan prediksi raster tidak berjalan otomatis.
- Skrip operasional yang dipanggil notebook disimpan di `archive/pipeline_scripts/`. Notebook memeriksa prasyarat dan mencegah penimpaan keluaran yang terdaftar kecuali `allow_overwrite=True` diberikan.
- Skrip diagnosis/eksperimen lama berada di `archive/legacy_scripts/`; model dan ringkasan kandidat opsional berada di `archive/rf_optimization/`. Keduanya tidak diperlukan untuk membuka aplikasi atau laporan yang sudah ada.
- `laporan_klasifikasi.ipynb` adalah laporan analisis yang berdiri sendiri, terpisah dari pipeline. Sel peta menampilkan peta Folium interaktif dengan pilihan basemap dan layer klasifikasi/sampel; percayai notebook melalui perintah **Trust Notebook** di VS Code jika diminta.

## Data dan metode

- Mosaik band Sentinel-2: `data/mosaic/`; resolusi analisis 20 m, CRS EPSG:32749.
- GeoJSON digitasi: `labels_clean/`.
- Dataset sampel: `data/processed/training_samples.csv` (49.062 sampel, 395 poligon).
- Fitur: 10 band B02/B03/B04/B05/B06/B07/B08/B8A/B11/B12 ditambah NDVI, NDMI, MNDWI, NDBI.
- Split validasi: GroupShuffleSplit 75:25 berdasarkan polygon_id, 37.078 train dan 11.984 test, tanpa poligon overlap.
- Raster peta yang ditampilkan dihasilkan oleh Random Forest baseline. Eksperimen kandidat SVM RBF memiliki skor accuracy lebih tinggi pada tabel eksperimen; skor itu belum menjadi klaim akurasi raster Random Forest.
- Referensi nomenklatur penutup lahan Indonesia yang relevan: SNI 7645-1:2014. Lima label lokal adalah agregasi operasional dan `water body` belum memisahkan laut, danau/ranu, dan sungai.

Raster indeks spektral hasil turunan tidak disertakan dalam folder kerja untuk mengurangi ruang penyimpanan. Jika ingin mengulang ekstraksi fitur atau prediksi raster, buat kembali indeks dan mask dengan `python archive\pipeline_scripts\26_buat_indeks_dan_masker.py`.

Tanggal akuisisi sumber Sentinel-2 tidak tersedia pada artefak lokal yang diperiksa. Lengkapi metadata tanggal/sumber citra sebelum publikasi ilmiah.
