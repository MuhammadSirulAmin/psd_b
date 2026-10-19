
from pathlib import Path

import streamlit as st
import numpy as np
import pandas as pd
import rasterio
import matplotlib.pyplot as plt

from rasterio.warp import transform
from rasterio.transform import rowcol
from streamlit_folium import st_folium

from peta_interaktif import (
    build_landcover_map,
    prepare_landcover_layers,
)


# =========================================================
# 1. KONFIGURASI
# =========================================================

ROOT = Path(__file__).resolve().parent
RASTER = ROOT / "results" / "classification" / "land_cover_baseline.tif"
SENTINEL_BANDS_DIR = ROOT / "data" / "mosaic"

CLASSES = {
    1: {"name": "Build-up", "color": "#d73027"},
    2: {"name": "Farm land", "color": "#fdae61"},
    3: {"name": "Forest", "color": "#1a9850"},
    4: {"name": "Mangrove", "color": "#006837"},
    5: {"name": "Water body", "color": "#4575b4"},
}

MAP_VIEWS = {
    "Jawa Timur": [[-9.2, 110.3], [-6.3, 116.3]],
    "Pulau Jawa": [[-9.2, 105.0], [-5.7, 115.0]],
    "Indonesia": [[-11.5, 94.5], [6.5, 141.1]],
}

st.set_page_config(
    page_title="Peta Tutupan Lahan Jawa Timur",
    page_icon="🌍",
    layout="wide",
)

st.title("🌍 Peta Klasifikasi Tutupan Lahan")
st.caption("Sentinel-2 | Random Forest | Resolusi 20 meter")

if not RASTER.exists():
    st.error(f"Raster klasifikasi tidak ditemukan:\n{RASTER}")
    st.stop()

with st.expander("Tujuan dan pemahaman data", expanded=True):
    st.markdown(
        "Analisis ini menyediakan informasi awal sebaran tutupan lahan untuk "
        "pemantauan lingkungan dan diskusi perencanaan wilayah Jawa Timur. "
        "Peta bersifat tematik/eksploratif, bukan peta legal atau pengganti "
        "verifikasi lapangan."
    )
    st.markdown(
        "**Sumber:** mosaik Sentinel-2 lokal di `data/mosaic/`, poligon "
        "referensi digitasi di `labels_clean/`, dan 49.062 sampel dari "
        "395 poligon. Split berbasis poligon menghasilkan 37.078 data train "
        "(296 poligon) dan 11.984 data test (99 poligon), tanpa overlap."
    )
    st.markdown(
        "**Kelas:** build-up, farm land (termasuk sawah menurut label yang "
        "tersedia), forest, mangrove, dan water body. Kelas water body saat "
        "ini menggabungkan laut, danau/ranu, sungai, dan perairan terbuka."
    )
    st.markdown(
        "**14 fitur:** B02/B03/B04/B05/B06/B07/B08/B8A/B11/B12 pada grid "
        "20 m, ditambah NDVI=(B08−B04)/(B08+B04), "
        "NDMI=(B08−B11)/(B08+B11), MNDWI=(B03−B11)/(B03+B11), "
        "NDBI=(B11−B08)/(B11+B08). B8A/B11/B12 menambah informasi untuk "
        "pemisahan mangrove dan hutan biasa; citra tampilan memakai RGB "
        "B04/B03/B02."
    )
    st.caption(
        "Rujukan nomenklatur: SNI 7645-1:2014; lima kelas proyek merupakan "
        "agregasi operasional, bukan pemetaan satu-ke-satu seluruh kode SNI. "
        "Perbandingan tersimpan: SVM RBF accuracy 93,94%, Random Forest "
        "92,72%, Gradient Boosting 91,37%. Raster pada peta ini masih "
        "menggunakan Random Forest baseline (validasi final accuracy 93,09%, "
        "macro-F1 88,52%). Tanggal akuisisi Sentinel-2 belum tercatat pada "
        "metadata proyek."
    )


# =========================================================
# 2. STATISTIK TUTUPAN LAHAN
# =========================================================

@st.cache_data
def calculate_statistics(
    raster_path,
    scl_path,
    raster_modified_time,
    scl_modified_time,
):
    raster_path = Path(raster_path)
    scl_path = Path(scl_path) if scl_path else None
    if (
        raster_path.stat().st_mtime != raster_modified_time
        or (scl_path and scl_path.is_file() and scl_path.stat().st_mtime != scl_modified_time)
    ):
        raise RuntimeError(
            "Raster berubah saat statistik sedang disiapkan; muat ulang."
        )

    pixel_counts = {code: 0 for code in CLASSES}
    nodata_count = 0

    has_scl = scl_path is not None and scl_path.is_file()
    with rasterio.open(raster_path) as src:
        pixel_area_m2 = abs(src.transform.a * src.transform.e)
        pixel_area_ha = pixel_area_m2 / 10000

        scl_src = rasterio.open(scl_path) if has_scl else None
        try:
            for _, window in src.block_windows(1):
                data = src.read(1, window=window)
                if scl_src is not None:
                    scl = scl_src.read(1, window=window)
                    data[(data == 5) & (scl != 6)] = 0
                    data[scl == 6] = 5
                nodata_count += int(np.count_nonzero(data == 0))

                values, counts = np.unique(data, return_counts=True)

                for value, count in zip(values, counts):
                    value = int(value)

                    if value in pixel_counts:
                        pixel_counts[value] += int(count)
        finally:
            if scl_src is not None:
                scl_src.close()

    total_valid = sum(pixel_counts.values())
    rows = []

    for code, info in CLASSES.items():
        count = pixel_counts[code]

        rows.append({
            "Kode": code,
            "Kelas": info["name"],
            "Jumlah piksel": count,
            "Luas (ha)": count * pixel_area_ha,
            "Persentase (%)": (
                count / total_valid * 100 if total_valid else 0
            ),
            "Warna": info["color"],
        })

    return (
        pd.DataFrame(rows),
        nodata_count,
        total_valid,
        pixel_area_ha,
    )


scl_file_candidate = SENTINEL_BANDS_DIR / "SCL_20m_mosaic.tif"
scl_path_str = str(scl_file_candidate) if scl_file_candidate.is_file() else None
scl_mtime = scl_file_candidate.stat().st_mtime if scl_file_candidate.is_file() else 0

stats, nodata_count, total_valid, pixel_area_ha = (
    calculate_statistics(
        str(RASTER),
        scl_path_str,
        RASTER.stat().st_mtime,
        scl_mtime,
    )
)

total_area_ha = stats["Luas (ha)"].sum()

st.subheader("📊 Statistik Tutupan Lahan")

col1, col2, col3, col4 = st.columns(4)

col1.metric("Luas terklasifikasi", f"{total_area_ha:,.2f} ha")
col2.metric("Total piksel kelas", f"{total_valid:,}")
col3.metric("Luas per piksel", f"{pixel_area_ha:.4f} ha")
col4.metric("Piksel NoData", f"{nodata_count:,}")

st.caption(
    "Luas dihitung berdasarkan jumlah piksel setiap kelas. "
    "Piksel berkode 0 dihitung sebagai NoData."
)

display_stats = stats[
    ["Kelas", "Jumlah piksel", "Luas (ha)", "Persentase (%)"]
].copy()

display_stats["Luas (ha)"] = display_stats["Luas (ha)"].map(
    lambda value: f"{value:,.2f}"
)
display_stats["Persentase (%)"] = display_stats["Persentase (%)"].map(
    lambda value: f"{value:.2f}%"
)

st.dataframe(
    display_stats,
    width="stretch",
    hide_index=True,
)

st.subheader("Grafik Luas Setiap Kelas")

fig, ax = plt.subplots(figsize=(10, 5))

bars = ax.bar(
    stats["Kelas"],
    stats["Luas (ha)"],
    color=stats["Warna"],
)

ax.set_xlabel("Kelas tutupan lahan")
ax.set_ylabel("Luas (hektare)")
ax.set_title("Luas Tutupan Lahan Hasil Random Forest")
ax.tick_params(axis="x", rotation=20)
ax.grid(axis="y", linestyle="--", alpha=0.3)

ax.bar_label(
    bars,
    labels=[f"{value:,.0f}" for value in stats["Luas (ha)"]],
    padding=3,
    fontsize=8,
)

fig.tight_layout()
st.pyplot(fig)
plt.close(fig)

csv_data = stats[
    ["Kode", "Kelas", "Jumlah piksel", "Luas (ha)", "Persentase (%)"]
].to_csv(index=False).encode("utf-8-sig")

st.download_button(
    "⬇️ Unduh statistik CSV",
    data=csv_data,
    file_name="statistik_tutupan_lahan.csv",
    mime="text/csv",
)


# =========================================================
# 3. PENGATURAN PETA
# =========================================================

st.sidebar.header("Pengaturan Peta")

map_region = st.sidebar.selectbox(
    "Cakupan tampilan peta",
    options=list(MAP_VIEWS),
)

opacity = st.sidebar.slider(
    "Transparansi klasifikasi",
    min_value=0.1,
    max_value=1.0,
    value=0.65,
    step=0.05,
)

# =========================================================
# 4. BACA RASTER DAN BUAT PREVIEW
# =========================================================

with rasterio.open(RASTER) as src:
    scl_path = SENTINEL_BANDS_DIR / "SCL_20m_mosaic.tif"
    if scl_path.is_file():
        with rasterio.open(scl_path) as scl_src:
            if (
                src.width != scl_src.width
                or src.height != scl_src.height
                or src.crs != scl_src.crs
                or src.transform != scl_src.transform
            ):
                raise ValueError(
                    "Grid SCL tidak sama dengan raster klasifikasi."
                )
    else:
        scl_path = None

    raster_crs = src.crs
    raster_transform = src.transform
    raster_width = src.width
    raster_height = src.height

try:
    landcover, sentinel_rgba, sentinel_bounds = prepare_landcover_layers(
        RASTER,
        scl_path,
    )
except (FileNotFoundError, ValueError) as exc:
    st.error(f"Layer peta tidak dapat disiapkan: {exc}")
    st.stop()

# =========================================================
# 6. BANGUN PETA FOLIUM
# =========================================================

m = build_landcover_map(
    landcover,
    sentinel_bounds,
    MAP_VIEWS[map_region],
    sentinel_rgba,
    class_opacity=opacity,
    sample_geojson_dir=ROOT / "labels_clean",
)


# =========================================================
# 9. TAMPILKAN PETA INTERAKTIF
# =========================================================

st.subheader("🗺️ Peta Interaktif")
st.caption(
    "Tampilan awal memakai peta jalan OpenStreetMap. Pilih citra satelit pada kontrol "
    "basemap di kanan atas; aktifkan layer Klasifikasi untuk prediksi raster atau "
    "layer Sampel untuk melihat poligon GeoJSON digitasi. Keduanya dapat diatur "
    "terpisah."
)

result = st_folium(
    m,
    height=650,
    use_container_width=True,
    returned_objects=["last_clicked"],
)

st.caption(
    "Peta dasar menampilkan cakupan wilayah yang dipilih. "
    "Overlay klasifikasi hanya tersedia pada area yang tercakup raster."
)

# =========================================================
# 10. INFORMASI LOKASI SAAT PETA DIKLIK
# =========================================================

if result and result.get("last_clicked"):
    point = result["last_clicked"]

    lon = point["lng"]
    lat = point["lat"]

    st.subheader("📍 Informasi Lokasi")
    st.write(f"Lintang: {lat:.6f}")
    st.write(f"Bujur: {lon:.6f}")

    xs, ys = transform(
        "EPSG:4326",
        raster_crs,
        [lon],
        [lat],
    )

    row, col = rowcol(
        raster_transform,
        xs[0],
        ys[0],
    )

    if 0 <= row < raster_height and 0 <= col < raster_width:
        with rasterio.open(RASTER) as src:
            value = int(
                src.read(
                    1,
                    window=((row, row + 1), (col, col + 1)),
                )[0, 0]
            )
        with rasterio.open(scl_path) as scl_src:
            scl_value = int(
                scl_src.read(
                    1,
                    window=((row, row + 1), (col, col + 1)),
                )[0, 0]
            )

        if scl_value == 6:
            value = 5
        elif value == 5:
            value = 0

        if value in CLASSES:
            info = CLASSES[value]

            st.markdown(
                f"**Kelas prediksi:** {info['name']}"
            )
            st.markdown(f"**Kode kelas:** {value}")

            clicked_stats = stats.loc[
                stats["Kode"] == value
            ].iloc[0]

            st.write(
                f"**Luas seluruh kelas ini:** "
                f"{clicked_stats['Luas (ha)']:,.2f} ha"
            )

        else:
            st.warning(
                "Piksel ini termasuk NoData atau area tanpa prediksi."
            )

    else:
        st.warning(
            "Titik yang dipilih berada di luar cakupan raster."
        )


# =========================================================
# 11. KETERANGAN
# =========================================================

st.divider()

st.caption(
    "Pilih layer klasifikasi pada kontrol peta untuk melihat prediksi Random "
    "Forest per kelas. Statistik luas dihitung dari piksel raster hasil klasifikasi."
)