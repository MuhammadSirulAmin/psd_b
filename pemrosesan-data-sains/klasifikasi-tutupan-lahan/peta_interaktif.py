
import json
from pathlib import Path

import folium
import numpy as np
import rasterio
from branca.element import MacroElement, Template
from rasterio.warp import transform_bounds


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
    bands_dir: Path,
    output_shape: tuple[int, int],
) -> tuple[np.ndarray, list[list[float]]]:
    """
    Membaca preview RGB PNG yang ringan.
    Tidak lagi membutuhkan raster B04, B03, dan B02 asli.
    """
    from PIL import Image

    preview_path = (
        reference_raster.parent / "sentinel_rgb_preview.png"
    )

    if not preview_path.is_file():
        raise FileNotFoundError(
            f"Preview RGB tidak ditemukan: {preview_path}"
        )

    with rasterio.open(reference_raster) as reference:
        bounds = transform_bounds(
            reference.crs,
            "EPSG:4326",
            *reference.bounds,
            densify_pts=21,
        )

    with Image.open(preview_path) as image:
        rgba = np.array(image.convert("RGBA"))

    west, south, east, north = bounds
    geographic_bounds = [[south, west], [north, east]]

    return rgba, geographic_bounds


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

    folium.TileLayer(
        tiles=(
            "https://server.arcgisonline.com/ArcGIS/rest/services/"
            "World_Imagery/MapServer/tile/{z}/{y}/{x}"
        ),
        attr="Tiles © Esri — Source: Esri, Maxar, Earthstar Geographics",
        name="Citra satelit global (Esri)",
        overlay=False,
        show=False,
    ).add_to(map_object)

    folium.TileLayer(
        "OpenStreetMap",
        name="Peta jalan (OpenStreetMap)",
        overlay=False,
        show=True,
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
            show=True,
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
