
from pathlib import Path
import time

import joblib
import numpy as np
import pandas as pd
import rasterio
from rasterio.windows import Window

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data" / "mosaic"
INDEX = DATA / "features"
OUT_DIR = ROOT / "results" / "classification"
OUT_DIR.mkdir(parents=True, exist_ok=True)

MODEL_PATH = ROOT / "results" / "validasi_final" / "baseline" / "model.joblib"
OUTPUT_PATH = OUT_DIR / "land_cover_baseline.tif"
MASK_PATH = INDEX / "mask_valid.tif"
SCL_PATH = DATA / "SCL_20m_mosaic.tif"

BANDS = [
    "B02", "B03", "B04", "B05", "B06",
    "B07", "B08", "B8A", "B11", "B12"
]

INDICES = ["NDVI", "NDMI", "MNDWI", "NDBI"]
FEATURES = BANDS + INDICES

PATHS = {
    name: DATA / f"{name}_20m_mosaic.tif"
    for name in BANDS
}
PATHS.update({
    name: INDEX / f"{name}.tif"
    for name in INDICES
})

BLOCK = 64
NODATA = 0


def main():
    required_rasters = [*PATHS.values(), MASK_PATH, SCL_PATH]
    missing_rasters = [path for path in required_rasters if not path.is_file()]
    if missing_rasters:
        missing = "\n".join(str(path) for path in missing_rasters)
        raise FileNotFoundError(
            "Raster fitur belum tersedia. Buat kembali dengan "
            "`python 26_buat_indeks_dan_masker.py`.\n"
            f"File yang belum ada:\n{missing}"
        )

    print("Memuat model...", flush=True)
    model = joblib.load(MODEL_PATH)
    model.set_params(n_jobs=1)

    if list(model.feature_names_in_) != FEATURES:
        raise ValueError(
            "Urutan fitur model tidak sesuai dengan FEATURES."
        )

    with rasterio.open(PATHS["B02"]) as ref:
        profile = ref.profile.copy()
        height, width = ref.height, ref.width
        reference_crs = ref.crs
        reference_transform = ref.transform

    # Pastikan semua raster memiliki grid yang sama
    for name, path in PATHS.items():
        with rasterio.open(path) as src:
            if (
                src.width != width
                or src.height != height
                or src.crs != reference_crs
                or src.transform != reference_transform
            ):
                raise ValueError(
                    f"Grid raster {name} tidak sama dengan B02."
                )

    with rasterio.open(MASK_PATH) as src:
        if (
            src.width != width
            or src.height != height
            or src.crs != reference_crs
            or src.transform != reference_transform
        ):
            raise ValueError("Grid mask_valid.tif tidak sesuai.")

    with rasterio.open(SCL_PATH) as src:
        if (
            src.width != width
            or src.height != height
            or src.crs != reference_crs
            or src.transform != reference_transform
        ):
            raise ValueError("Grid SCL_20m_mosaic.tif tidak sesuai.")

    profile.update(
        driver="GTiff",
        count=1,
        dtype="uint8",
        nodata=NODATA,
        compress="lzw",
        tiled=True,
        blockxsize=256,
        blockysize=256,
        BIGTIFF="IF_SAFER"
    )

    total = (
        ((height + BLOCK - 1) // BLOCK)
        * ((width + BLOCK - 1) // BLOCK)
    )
    done = 0
    started = time.time()

    class_codes = {
        label: i + 1
        for i, label in enumerate(model.classes_)
    }
    water_body_code = class_codes.get("water body")
    if water_body_code is None:
        raise ValueError(
            "Model tidak memiliki kelas 'water body'."
        )

    print("Kode kelas:", class_codes, flush=True)
    print(
        f"Ukuran raster: {width} x {height}",
        flush=True
    )

    with rasterio.open(MASK_PATH) as mask_src, \
         rasterio.open(SCL_PATH) as scl_src, \
         rasterio.open(OUTPUT_PATH, "w", **profile) as dst:

        sources = {
            name: rasterio.open(path)
            for name, path in PATHS.items()
        }

        try:
            for row in range(0, height, BLOCK):
                for col in range(0, width, BLOCK):
                    h = min(BLOCK, height - row)
                    w = min(BLOCK, width - col)
                    win = Window(col, row, w, h)

                    mask = mask_src.read(1, window=win) > 0
                    scl = scl_src.read(1, window=win)
                    arrays = []

                    for name in FEATURES:
                        arr = sources[name].read(
                            1, window=win
                        ).astype("float32")

                        # Hanya band reflektansi yang diskalakan.
                        # Indeks spektral tidak dibagi 10000.
                        if name in BANDS:
                            arr /= 10000.0

                        arrays.append(arr)

                    stack = np.stack(arrays, axis=-1)

                    valid = (
                        mask
                        & np.all(np.isfinite(stack), axis=-1)
                    )

                    result = np.zeros((h, w), dtype="uint8")

                    if valid.any():
                        X = pd.DataFrame(
                            stack[valid],
                            columns=FEATURES
                        )

                        pred = model.predict(X)

                        result[valid] = np.array(
                            [class_codes[p] for p in pred],
                            dtype="uint8"
                        )

                    result[(result == water_body_code) & (scl != 6)] = NODATA
                    result[scl == 6] = water_body_code

                    dst.write(result, 1, window=win)

                    done += 1

                print(
                    f"Progres: {done}/{total} blok | "
                    f"{done / total:.1%} | "
                    f"{(time.time() - started) / 60:.1f} menit",
                    flush=True
                )

        finally:
            for src in sources.values():
                src.close()

    # Periksa distribusi kelas hasil prediksi
    with rasterio.open(OUTPUT_PATH) as src:
        data = src.read(1)
        values, counts = np.unique(data, return_counts=True)

    print("\nDistribusi hasil klasifikasi:", flush=True)
    for value, count in zip(values, counts):
        if value == NODATA:
            label = "NoData"
        else:
            label = str(model.classes_[value - 1])

        print(
            f"{label}: {count:,} piksel "
            f"({count / data.size:.2%})",
            flush=True
        )

    print(f"\nSelesai: {OUTPUT_PATH}", flush=True)


if __name__ == "__main__":
    main()