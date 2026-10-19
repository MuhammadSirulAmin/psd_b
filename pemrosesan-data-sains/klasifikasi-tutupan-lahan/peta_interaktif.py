
import json
from pathlib import Path

import folium
import numpy as np
import rasterio
from affine import Affine
from branca.element import MacroElement, Template
from rasterio.enums import Resampling
from rasterio.features import sieve
from rasterio.transform import from_bounds
from rasterio.warp import reproject, transform_bounds


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

MAP_MAX_DIMENSION = 1800
MIN_DISPLAY_PATCH_PIXELS = 5


def add_responsive_resize_handler(map_object: folium.Map) -> None:
    resize_handler = MacroElement()
    resize_handler._template = Template(
        """
        {% macro script(this, kwargs) %}
        (() => {
          const map = {{ this._parent.get_name() }};
          const refreshSize = () => map.invalidateSize({animate: false, pan: false});
          window.addEventListener("resize", refreshSize);
          if (window.ResizeObserver) {
            new ResizeObserver(refreshSize).observe(map.getContainer());
          }
          refreshSize();
          requestAnimationFrame(refreshSize);
          setTimeout(refreshSize, 300);
        })();
        {% endmacro %}
        """
    )
    map_object.add_child(resize_handler)


def create_sentinel_rgb(
    reference_raster: Path,
    output_shape: tuple[int, int],
) -> tuple[np.ndarray, list[list[float]]]:
    from PIL import Image

    preview_path = (
        reference_raster.parent / "sentinel_rgb_preview.png"
    )

    if not preview_path.is_file():
        raise FileNotFoundError(
            f"Preview RGB tidak ditemukan: {preview_path}"
        )

    with Image.open(preview_path) as image:
        rgba = np.array(image.convert("RGBA"))

    with rasterio.open(reference_raster) as reference:
        target_bounds = transform_bounds(
            reference.crs,
            "EPSG:3857",
            *reference.bounds,
            densify_pts=21,
        )
        source_transform = reference.transform * Affine.scale(
            reference.width / rgba.shape[1],
            reference.height / rgba.shape[0],
        )
        target_transform = from_bounds(
            *target_bounds,
            width=output_shape[1],
            height=output_shape[0],
        )

        projected_rgba = np.zeros(
            (*output_shape, 4),
            dtype=np.uint8,
        )
        for channel in range(4):
            reproject(
                source=rgba[:, :, channel],
                destination=projected_rgba[:, :, channel],
                src_transform=source_transform,
                src_crs=reference.crs,
                dst_transform=target_transform,
                dst_crs="EPSG:3857",
                resampling=Resampling.nearest,
            )

    west, south, east, north = transform_bounds(
        "EPSG:3857",
        "EPSG:4326",
        *target_bounds,
        densify_pts=21,
    )
    geographic_bounds = [[south, west], [north, east]]

    return projected_rgba, geographic_bounds


def reproject_landcover(
    landcover: np.ndarray,
    reference_raster: Path,
) -> np.ndarray:
    with rasterio.open(reference_raster) as reference:
        target_bounds = transform_bounds(
            reference.crs,
            "EPSG:3857",
            *reference.bounds,
            densify_pts=21,
        )
        source_transform = reference.transform * Affine.scale(
            reference.width / landcover.shape[1],
            reference.height / landcover.shape[0],
        )
        target_transform = from_bounds(
            *target_bounds,
            width=landcover.shape[1],
            height=landcover.shape[0],
        )

        projected_landcover = np.zeros_like(landcover)
        reproject(
            source=landcover,
            destination=projected_landcover,
            src_transform=source_transform,
            src_crs=reference.crs,
            dst_transform=target_transform,
            dst_crs="EPSG:3857",
            resampling=Resampling.nearest,
        )

    return projected_landcover


def prepare_landcover_layers(
    classification_raster: Path,
    scl_raster: Path | None = None,
    max_dimension: int = MAP_MAX_DIMENSION,
) -> tuple[np.ndarray, np.ndarray, list[list[float]]]:
    if max_dimension < 1:
        raise ValueError("Ukuran maksimum peta harus lebih besar dari nol.")

    with rasterio.open(classification_raster) as classification:
        scale = max(
            1,
            int(
                np.ceil(
                    max(classification.width, classification.height)
                    / max_dimension
                )
            ),
        )
        output_shape = (
            max(1, int(np.ceil(classification.height / scale))),
            max(1, int(np.ceil(classification.width / scale))),
        )
        landcover = classification.read(
            1,
            out_shape=output_shape,
            resampling=Resampling.mode,
        )

    water_mask = None
    if scl_raster is not None:
        scl_file = Path(scl_raster)
        if scl_file.is_file():
            with rasterio.open(scl_file) as scl:
                if (
                    classification.width == scl.width
                    and classification.height == scl.height
                    and classification.crs == scl.crs
                    and classification.transform == scl.transform
                ):
                    scl_preview = scl.read(
                        1,
                        out_shape=output_shape,
                        resampling=Resampling.mode,
                    )
                    water_mask = (scl_preview == 6).astype(np.uint8)
                    water_mask = reproject_landcover(water_mask, classification_raster) > 0

    landcover = reproject_landcover(landcover, classification_raster)

    if water_mask is not None:
        landcover[(landcover == 5) & ~water_mask] = 0
        landcover[water_mask] = 5

    # Filter ekologis: Mangrove hanya berada di wilayah pesisir pantai & muara sungai
    coastline_path = (
        classification_raster.parents[1]
        / "labels_clean"
        / "east_java_coastline.geojson"
    )
    if coastline_path.is_file():
        try:
            import geopandas as gpd
            from rasterio.features import rasterize
            from scipy.ndimage import distance_transform_edt

            coast_gdf = gpd.read_file(coastline_path)
            with rasterio.open(classification_raster) as ref:
                coast_utm = gpd.GeoSeries(
                    [coast_gdf.union_all()], crs="EPSG:4326"
                ).to_crs(ref.crs).iloc[0]
                coast_buffer = coast_utm.buffer(3500) # Buffer pesisir/muara 3,5 km
                coastal_mask = rasterize(
                    [(coast_buffer, 1)],
                    out_shape=output_shape,
                    transform=ref.transform * Affine.scale(scale, scale),
                    fill=0,
                    dtype="uint8",
                ) == 1
                coastal_mask = (
                    reproject_landcover(
                        coastal_mask.astype(np.uint8), classification_raster
                    )
                    > 0
                )

            # Jika ada piksel mangrove di luar zona pesisir/muara, reklasifikasi ke kelas terdekat
            inland_mangrove = (landcover == 4) & (~coastal_mask)
            if np.any(inland_mangrove):
                valid_land = np.isin(landcover, (1, 2, 3))
                if np.any(valid_land):
                    _, (r_idx, c_idx) = distance_transform_edt(
                        ~valid_land, return_indices=True
                    )
                    landcover[inland_mangrove] = landcover[
                        r_idx[inland_mangrove], c_idx[inland_mangrove]
                    ]
                else:
                    landcover[inland_mangrove] = 3
        except Exception:
            pass

    landcover = sieve(
        landcover,
        size=MIN_DISPLAY_PATCH_PIXELS,
        mask=np.isin(landcover, (1, 2, 3, 4)),
        connectivity=8,
    )

    if water_mask is not None:
        landcover[(landcover == 5) & ~water_mask] = 0
        landcover[water_mask] = 5

    sentinel_rgb, bounds = create_sentinel_rgb(
        classification_raster,
        output_shape,
    )

    return landcover, sentinel_rgb, bounds


def build_landcover_map(
    landcover: np.ndarray,
    bounds: list[list[float]],
    map_bounds: list[list[float]],
    sentinel_rgb: np.ndarray,
    *,
    class_opacity: float = 0.65,
    sample_geojson_dir: Path | None = None,
) -> folium.Map:
    if not 0 <= class_opacity <= 1:
        raise ValueError(
            "Transparansi kelas harus berada di antara 0 dan 1."
        )

    center = [
        (map_bounds[0][0] + map_bounds[1][0]) / 2,
        (map_bounds[0][1] + map_bounds[1][1]) / 2,
    ]

    map_object = folium.Map(
        location=center,
        zoom_start=6,
        width="100%",
        height="650px",
        tiles=None,
        control_scale=True,
    )

    satellite_layer = folium.TileLayer(
        tiles=(
            "https://server.arcgisonline.com/ArcGIS/rest/services/"
            "World_Imagery/MapServer/tile/{z}/{y}/{x}"
        ),
        attr="Tiles © Esri — Source: Esri, Maxar, Earthstar Geographics",
        name="Citra satelit global (Esri)",
        overlay=False,
        show=False,
        max_zoom=19,
    ).add_to(map_object)

    road_layer = folium.TileLayer(
        "OpenStreetMap",
        name="Peta jalan (OpenStreetMap)",
        overlay=False,
        show=True,
        max_zoom=19,
    ).add_to(map_object)

    # Preview RGB memiliki resolusi lebih kecil daripada raster klasifikasi.
    # Keduanya ditampilkan pada cakupan geografis yang sama.
    folium.raster_layers.ImageOverlay(
        image=sentinel_rgb,
        bounds=bounds,
        name="Citra Sentinel-2 (B04/B03/B02)",
        overlay=False,
        control=True,
        show=False,
        opacity=1,
        interactive=False,
        cross_origin=False,
        zindex=1,
    ).add_to(map_object)

    add_classification_layers(
        map_object,
        landcover,
        bounds,
        class_opacity,
    )

    if sample_geojson_dir is not None:
        add_sample_layers(map_object, sample_geojson_dir)

    add_basemap_error_fallback(
        map_object,
        satellite_layer,
        road_layer,
    )

    legend_items = "".join(
        f"""
        <div style="display:flex;align-items:center;margin:5px 0;">
          <span style="display:inline-block;width:15px;height:15px;
            background:{info['color']};margin-right:8px;
            border:1px solid #555;flex-shrink:0;"></span>
          <span>{info['name']}</span>
        </div>
        """
        for info in CLASSES.values()
    )

    legend_html = f"""
    <div style="position:fixed;bottom:35px;left:20px;z-index:9999;
      background:white;padding:12px 15px;border:1px solid #777;
      border-radius:6px;box-shadow:0 2px 6px rgba(0,0,0,.25);
      font-size:13px;color:#222;">
      <div style="font-weight:bold;margin-bottom:8px;">
        Legenda Tutupan Lahan</div>
      {legend_items}
    </div>
    """

    map_object.get_root().html.add_child(
        folium.Element(legend_html)
    )

    folium.LayerControl(collapsed=False).add_to(map_object)
    map_object.fit_bounds(map_bounds)
    add_responsive_resize_handler(map_object)

    return map_object


def add_basemap_error_fallback(
    map_object: folium.Map,
    satellite_layer: folium.TileLayer,
    road_layer: folium.TileLayer,
) -> None:
    fallback = MacroElement()
    fallback.satellite_layer = satellite_layer
    fallback.road_layer = road_layer
    fallback._template = Template(
        """
        {% macro script(this, kwargs) %}
        (() => {
          const map = {{ this._parent.get_name() }};
          const satellite = {{ this.satellite_layer.get_name() }};
          const roads = {{ this.road_layer.get_name() }};
          let failures = 0;
          let firstFailure = 0;
          let switched = false;

          const showNotice = () => {
            let notice = document.getElementById("basemap-load-notice");
            if (!notice) {
              notice = document.createElement("div");
              notice.id = "basemap-load-notice";
              notice.style.cssText =
                "position:absolute;top:10px;left:50px;z-index:1000;" +
                "max-width:360px;padding:8px 12px;background:#fff3cd;" +
                "border:1px solid #e0b44c;border-radius:4px;" +
                "color:#493b12;font:13px sans-serif;box-shadow:0 1px 4px #777;";
              map.getContainer().appendChild(notice);
            }
            notice.textContent =
              "Tile peta dasar gagal dimuat. Periksa koneksi internet " +
              "atau pilih basemap lain.";
          };

          const registerErrors = (layer, alternative) => {
            layer.on("tileerror", () => {
              if (!map.hasLayer(layer)) return;
              const now = Date.now();
              if (!firstFailure || now - firstFailure > 4000) {
                firstFailure = now;
                failures = 0;
              }
              failures += 1;
              if (failures < 4 || switched) return;
              switched = true;
              map.removeLayer(layer);
              map.addLayer(alternative);
              showNotice();
            });
          };

          registerErrors(satellite, roads);
          registerErrors(roads, satellite);
        })();
        {% endmacro %}
        """
    )
    map_object.add_child(fallback)


def add_sample_layers(
    map_object: folium.Map,
    geojson_dir: Path,
) -> None:
    for info in CLASSES.values():
        filename = (
            info["name"].lower().replace("-", "_") + ".geojson"
        )

        if info["name"] == "Farm land":
            filename = "farm land.geojson"
        elif info["name"] == "Water body":
            filename = "water body.geojson"

        geojson_path = geojson_dir / filename

        if not geojson_path.is_file():
            raise FileNotFoundError(
                f"GeoJSON sampel tidak ditemukan: {geojson_path}"
            )

        with geojson_path.open(encoding="utf-8") as source:
            geojson_data = json.load(source)

        folium.GeoJson(
            data=geojson_data,
            name=f"Sampel: {info['name']}",
            style_function=lambda feature, color=info["color"]: {
                "color": color,
                "weight": 2,
                "fillColor": color,
                "fillOpacity": 0.15,
            },
            show=False,
            overlay=True,
            control=True,
        ).add_to(map_object)


def add_classification_layers(
    map_object: folium.Map,
    landcover: np.ndarray,
    bounds: list[list[float]],
    opacity: float,
) -> None:
    if not 0 <= opacity <= 1:
        raise ValueError(
            "Transparansi kelas harus berada di antara 0 dan 1."
        )

    for code, info in CLASSES.items():
        class_rgba = np.zeros(
            (*landcover.shape, 4),
            dtype=np.uint8,
        )

        color = info["color"]
        rgb = tuple(
            int(color[index:index + 2], 16)
            for index in (1, 3, 5)
        )

        mask = landcover == code
        class_rgba[mask, :3] = rgb
        class_rgba[mask, 3] = round(opacity * 255)

        folium.raster_layers.ImageOverlay(
            image=class_rgba,
            bounds=bounds,
            name=f"Klasifikasi: {info['name']}",
            overlay=True,
            control=True,
            show=False,
            opacity=1,
            interactive=False,
            cross_origin=False,
            zindex=2,
        ).add_to(map_object)
