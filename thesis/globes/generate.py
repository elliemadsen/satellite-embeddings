"""
Globe visualizations of 64 AlphaEarth embedding bands.

Part 1 → bands_grid.png
    8×8 grid of orthographic globes, one per embedding band (A00–A63).
    Land is scaled white → black by band value.
    Oceans are absent — you can see through to back-hemisphere land.

Part 2 → clusters_grid.png
    3×3 grid of orthographic globes after k-means (k=9) on all land pixels.
    Cluster land = black; other land = light grey; ocean = white.

GEE project: gsapp-map
"""

import ee
import numpy as np
import requests
import zipfile
import io
import os
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
from sklearn.cluster import MiniBatchKMeans
from scipy.ndimage import map_coordinates
import rasterio
from rasterio.io import MemoryFile

# ── Earth Engine ──────────────────────────────────────────────────────────────
try:
    ee.Initialize(project="gsapp-map")
except Exception:
    ee.Authenticate()
    ee.Initialize(project="gsapp-map")

YEAR  = 2024
BANDS = [f"A{i:02d}" for i in range(64)]
N_CLUSTERS = [4, 9, 16, 25]

OUTPUT_DIR = Path(__file__).parent
CACHE_DIR  = OUTPUT_DIR / "cache"
CACHE_DIR.mkdir(exist_ok=True)

# ── Globe view direction (slightly tilted — shows land on both hemispheres) ───
CENTER_LON = 20.0
CENTER_LAT = 30.0

# ── Layout — Part 1 (8×8 band grid) ─────────────────────────────────────────
GLOBE_D1       = 380    # globe diameter in the grid (px)
GLOBE_D1_INDIV = 1024   # diameter for individual saved globes
GAP1       = 18    # gap between globes
LABEL_H1   = 28    # height of label row below each globe
PAD1       = 64    # outer canvas padding

# ── Layout — Part 2 (3×3 cluster grid) ───────────────────────────────────────
GLOBE_D2   = 580
GAP2       = 30
LABEL_H2   = 38
PAD2       = 44

# ── Style ─────────────────────────────────────────────────────────────────────
BG_COLOR    = (255, 255, 255)
LABEL_COLOR = (25,  25,  25)
RING_COLOR  = (195, 195, 195)    # thin circle border around each globe
BACK_BLEND        = 0.38    # opacity of back-hemisphere land (0–1)
CLUSTER_OTHER_VAL = 0.13    # brightness of non-cluster land in Part 2
POLAR_PAD         = 5       # extra lat rows beyond ±85° for polar-cap fill
NO_DATA_GREY      = 210     # grey value for land pixels without GEE embedding data


# ─────────────────────────────────────────────────────────────────────────────
# FONTS
# ─────────────────────────────────────────────────────────────────────────────

def _load_font(size: int) -> ImageFont.FreeTypeFont:
    search = [
        str(Path.home() / "Library/Fonts/Roboto-Regular.ttf"),
        "/System/Library/Fonts/SFNS.ttf",
        "/System/Library/Fonts/HelveticaNeue.ttc",
        "/System/Library/Fonts/Helvetica.ttc",
        "/System/Library/Fonts/Avenir Next.ttc",
        "/usr/share/fonts/truetype/roboto/Roboto-Regular.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
    ]
    for p in search:
        if os.path.exists(p):
            return ImageFont.truetype(p, size)
    return ImageFont.load_default()


FONT1 = _load_font(18)
FONT2 = _load_font(24)


# ─────────────────────────────────────────────────────────────────────────────
# DATA DOWNLOAD
# ─────────────────────────────────────────────────────────────────────────────

GRID_W, GRID_H = 360, 170   # ~1° per pixel globally
DL_BATCH  = 8               # bands per GEE download request (8 × smaller = faster server-side)


def _gee_mosaic() -> ee.Image:
    return (
        ee.ImageCollection("GOOGLE/SATELLITE_EMBEDDING/V1/ANNUAL")
        .filterDate(f"{YEAR}-01-01", f"{YEAR + 1}-01-01")
        .mosaic()
        .select(BANDS)
    )


def _parse_tiff_response(content: bytes) -> np.ndarray:
    """
    Return (bands, H, W) float32 from a raw HTTP response body.
    Handles both a bare multi-band GeoTIFF and a ZIP containing one or
    more single-band GeoTIFFs (which GEE sometimes returns even when
    filePerBand=False is requested).
    """
    if content[:4] == b"PK\x03\x04":   # ZIP magic bytes
        with zipfile.ZipFile(io.BytesIO(content)) as zf:
            tif_names = sorted(n for n in zf.namelist() if n.lower().endswith(".tif"))
            if not tif_names:
                raise ValueError("ZIP response contained no .tif files")
            bands_list = []
            for name in tif_names:
                with MemoryFile(zf.read(name)) as mf:
                    with mf.open() as ds:
                        bands_list.append(ds.read().astype(np.float32))
        return np.concatenate(bands_list, axis=0)   # (n_bands, H, W)
    else:                               # bare GeoTIFF
        with MemoryFile(content) as mf:
            with mf.open() as ds:
                return ds.read().astype(np.float32)


def _download_url(url: str) -> bytes:
    resp = requests.get(url, timeout=600)
    resp.raise_for_status()
    return resp.content


def _download_batch(image_obj, band_list, region, dims, max_retries: int = 5) -> np.ndarray:
    """
    Download a band batch from GEE with URL regeneration on every retry.
    Catches timeouts, connection errors, and mid-stream broken-pipe errors.
    """
    import time
    from requests.exceptions import (
        ReadTimeout, ConnectionError as ReqConnError,
        ChunkedEncodingError,
    )
    _RETRYABLE = (ReadTimeout, ReqConnError, ChunkedEncodingError)

    for attempt in range(max_retries):
        try:
            url = (
                image_obj.select(band_list).unmask(0)
                .getDownloadURL({
                    "region":      region,
                    "dimensions":  dims,
                    "format":      "GEO_TIFF",
                    "filePerBand": False,
                    "crs":         "EPSG:4326",
                })
            )
            resp = requests.get(url, timeout=1800)   # 30-minute ceiling
            if resp.status_code == 200:
                return _parse_tiff_response(resp.content)
            if resp.status_code in (400, 500, 502, 503) and attempt < max_retries - 1:
                wait = 60 * (attempt + 1)
                print(
                    f"    HTTP {resp.status_code} on attempt {attempt + 1}"
                    f" — waiting {wait}s then regenerating URL …",
                    flush=True,
                )
                time.sleep(wait)
            else:
                resp.raise_for_status()
        except _RETRYABLE as exc:
            if attempt < max_retries - 1:
                wait = 60 * (attempt + 1)
                print(
                    f"    {type(exc).__name__} on attempt {attempt + 1}"
                    f" — waiting {wait}s then regenerating URL …",
                    flush=True,
                )
                time.sleep(wait)
            else:
                raise
    raise RuntimeError("All retries exhausted")


def _build_ne_mask(lat_grid: np.ndarray, lon_grid: np.ndarray) -> np.ndarray:
    """
    Rasterise Natural Earth country polygons onto the given lat/lon grid.
    Includes Antarctica, so there is no polar-cap hole.
    """
    import geopandas as gpd
    from rasterio.features import rasterize
    from rasterio.transform import from_bounds

    shp = (Path(__file__).parent.parent.parent
           / "distance_experiment" / "ne_110m_admin_0_countries"
           / "ne_110m_admin_0_countries.shp")
    world = gpd.read_file(str(shp))
    H, W  = len(lat_grid), len(lon_grid)
    dlat  = abs(lat_grid[0] - lat_grid[1]) if H > 1 else 1.0
    dlon  = abs(lon_grid[1] - lon_grid[0]) if W > 1 else 1.0
    top    = min(lat_grid[0]  + dlat / 2,  90.0)
    bottom = max(lat_grid[-1] - dlat / 2, -90.0)
    left   = lon_grid[0]  - dlon / 2
    right  = lon_grid[-1] + dlon / 2
    transform = from_bounds(left, bottom, right, top, W, H)
    shapes = [(geom, 1) for geom in world.geometry if geom is not None]
    mask = rasterize(shapes, out_shape=(H, W), transform=transform,
                     fill=0, dtype=np.uint8, all_touched=False)
    return mask.astype(bool)


def load_global_data(force: bool = False):
    """
    Returns
    -------
    data_ext  : (64, H_ext, W) float32 – GEE embedding values; 0 outside coverage
    ne_mask   : (H_ext, W)    bool     – True = land (Natural Earth, ≈±90° coverage)
    has_data  : (H_ext, W)    bool     – True = pixel has a real GEE embedding
    lat_grid  : (H_ext,)      float    – latitudes, decreasing (≈+90 … ≈−90)
    lon_grid  : (W,)          float    – longitudes, increasing (−180 … +180)
    """
    f_data = CACHE_DIR / "embeddings.npy"
    f_land = CACHE_DIR / "land_mask.npy"

    if not force and f_data.exists() and f_land.exists():
        print("Loading cached global embeddings …", flush=True)
        data     = np.load(f_data)
        gee_mask = np.load(f_land).astype(bool)
    else:
        region = ee.Geometry.Rectangle([-180, -85, 180, 85], None, False)
        image  = _gee_mosaic()
        dims   = f"{GRID_W}x{GRID_H}"

        # GEE land mask — cached separately so download is only done once
        if not force and f_land.exists():
            print("Loading cached land mask …", flush=True)
            gee_mask = np.load(f_land).astype(bool)
        else:
            print("Downloading land mask …", flush=True)
            mask_url = (
                image.select("A00").mask().unmask(0)
                .getDownloadURL({
                    "region":     region,
                    "dimensions": dims,
                    "format":     "GEO_TIFF",
                    "crs":        "EPSG:4326",
                })
            )
            land_raw = _parse_tiff_response(_download_url(mask_url))
            gee_mask = land_raw[0] > 0.5
            np.save(f_land, gee_mask.astype(np.uint8))
        print(f"  GEE land pixels: {gee_mask.sum():,}", flush=True)

        # Embeddings — one .npy per batch so crashes can be resumed
        data = np.zeros((64, GRID_H, GRID_W), dtype=np.float32)
        n_batches = (64 + DL_BATCH - 1) // DL_BATCH
        for bi, b0 in enumerate(range(0, 64, DL_BATCH)):
            batch_bands = BANDS[b0 : b0 + DL_BATCH]
            f_batch = CACHE_DIR / f"batch_{bi:02d}.npy"

            if not force and f_batch.exists():
                print(
                    f"  Batch {bi + 1}/{n_batches} ({batch_bands[0]}–{batch_bands[-1]})"
                    f" already cached — skipping",
                    flush=True,
                )
                batch_arr = np.load(f_batch)
            else:
                print(
                    f"  Downloading bands {batch_bands[0]}–{batch_bands[-1]}"
                    f"  (batch {bi + 1}/{n_batches}) …",
                    flush=True,
                )
                batch_arr = _download_batch(image, batch_bands, region, dims)
                np.save(f_batch, batch_arr)
                print(f"    ✓ batch {bi + 1} saved", flush=True)

            n = min(batch_arr.shape[0], DL_BATCH)
            data[b0 : b0 + n] = batch_arr[:n]

        data[:, ~gee_mask] = 0.0
        np.save(f_data, data)
        print("  All batches done. Saved combined embeddings.npy", flush=True)

    # ── Extend the lat grid by POLAR_PAD rows at each pole ───────────────────
    gee_lat  = np.linspace(85.0, -85.0, GRID_H)
    lon_grid = np.linspace(-180.0, 180.0, GRID_W)
    step     = (gee_lat[0] - gee_lat[-1]) / (GRID_H - 1)   # ≈ 1.006 °/row
    lat_grid = np.concatenate([
        gee_lat[0] + np.arange(POLAR_PAD, 0, -1) * step,   # rows above +85 °N
        gee_lat,
        gee_lat[-1] - np.arange(1, POLAR_PAD + 1) * step,  # rows below −85 °S
    ])
    H_ext = len(lat_grid)   # GRID_H + 2 * POLAR_PAD

    # Embed GEE data in the centre of the extended arrays
    data_ext = np.zeros((64, H_ext, GRID_W), dtype=np.float32)
    data_ext[:, POLAR_PAD : POLAR_PAD + GRID_H, :] = data

    has_data = np.zeros((H_ext, GRID_W), dtype=bool)
    has_data[POLAR_PAD : POLAR_PAD + GRID_H, :] = gee_mask

    # ── Natural Earth land mask (covers the full extended grid) ──────────────
    f_ne = CACHE_DIR / "ne_land_mask.npy"
    if not force and f_ne.exists():
        ne_mask = np.load(f_ne).astype(bool)
    else:
        print("Building Natural Earth land mask …", flush=True)
        ne_mask = _build_ne_mask(lat_grid, lon_grid)
        np.save(f_ne, ne_mask.astype(np.uint8))
        print(f"  NE land pixels: {ne_mask.sum():,}", flush=True)

    # GEE Landsat coverage extends over sea ice / ocean at high latitudes,
    # so has_data alone cannot be used as a land mask.  Restrict it to pixels
    # that are also classified as land by Natural Earth, which correctly
    # delineates coastlines including Antarctic bays.
    has_data = has_data & ne_mask

    return data_ext, ne_mask, has_data, lat_grid, lon_grid


# ─────────────────────────────────────────────────────────────────────────────
# ORTHOGRAPHIC GLOBE RENDERER
# ─────────────────────────────────────────────────────────────────────────────

def _camera_basis(clon_deg: float, clat_deg: float):
    """
    Return orthonormal basis (f, r, u) for orthographic projection.
      f = unit vector pointing at the camera centre (into the screen)
      r = east direction at the centre (screen-right)
      u = north direction at the centre (screen-up)
    """
    la = np.radians(clat_deg)
    lo = np.radians(clon_deg)
    f = np.array([np.cos(la) * np.cos(lo),  np.cos(la) * np.sin(lo),  np.sin(la)])
    r = np.array([-np.sin(lo),               np.cos(lo),               0.0       ])
    u = np.array([-np.sin(la) * np.cos(lo), -np.sin(la) * np.sin(lo), np.cos(la) ])
    return f, r, u


def render_globe(
    values: np.ndarray,
    ne_mask: np.ndarray,
    has_data: np.ndarray,
    lat_grid: np.ndarray,
    lon_grid: np.ndarray,
    globe_d: int,
    clon: float = CENTER_LON,
    clat: float = CENTER_LAT,
    min_val: float = None,
    max_val: float = None,
    back_blend: float = BACK_BLEND,
    active_mask: np.ndarray = None,
    passive_grey: int = None,
) -> Image.Image:
    """
    Render a single orthographic globe as an RGBA PIL image.

    ne_mask     : (H, W) bool – Natural Earth land (includes polar caps).
    has_data    : (H, W) bool – pixels with actual GEE embedding values.
    active_mask : (H, W) bool or None – cluster land in cluster mode.
    passive_grey: grey for non-active data land (cluster mode only).
    """
    f, r, u = _camera_basis(clon, clat)
    H, W = ne_mask.shape

    # Normalise values using only pixels that have real GEE embedding data
    data_land = values[has_data]
    if min_val is None:
        min_val = float(np.percentile(data_land, 2))  if len(data_land) else 0.0
    if max_val is None:
        max_val = float(np.percentile(data_land, 98)) if len(data_land) else 1.0
    rng = max(float(max_val - min_val), 1e-8)
    vals_norm = np.clip((values - min_val) / rng, 0.0, 1.0)  # 0 → white, 1 → black

    # Screen-space coordinates in normalised [−1…+1]
    half = globe_d / 2.0
    gy, gx = np.mgrid[0:globe_d, 0:globe_d]
    sx = (gx + 0.5 - half) / half
    sy = -(gy + 0.5 - half) / half
    d2     = sx ** 2 + sy ** 2
    inside = d2 <= 1.0
    nz     = np.where(inside, np.sqrt(np.clip(1.0 - d2, 0.0, 1.0)), 0.0)

    def _idx(nz_sign: float):
        """Project screen pixels to equirectangular coords + validity flag."""
        Wx = sx * r[0] + sy * u[0] + nz_sign * nz * f[0]
        Wy = sx * r[1] + sy * u[1] + nz_sign * nz * f[1]
        Wz = sx * r[2] + sy * u[2] + nz_sign * nz * f[2]
        lat_p = np.degrees(np.arcsin(np.clip(Wz, -1.0, 1.0)))
        lon_p = np.degrees(np.arctan2(Wy, Wx))
        # valid: inside the lat/lon extent of the extended grid (≈ ±90°)
        valid = (lat_p >= lat_grid[-1]) & (lat_p <= lat_grid[0])
        rowf  = np.clip(
            (lat_grid[0] - lat_p) / (lat_grid[0] - lat_grid[-1]) * (H - 1),
            0.0, H - 1.0,
        )
        colf  = np.clip(
            (lon_p - lon_grid[0]) / (lon_grid[-1] - lon_grid[0]) * (W - 1),
            0.0, W - 1.0,
        )
        return rowf, colf, valid

    row_ff, col_ff, valid_f = _idx(+1)
    row_bf, col_bf, valid_b = _idx(-1)

    G = globe_d

    def _smp(arr, rowf, colf):
        return map_coordinates(arr.astype(np.float32),
                               [rowf.ravel(), colf.ravel()],
                               order=1, mode='nearest').reshape(G, G)

    vf   = _smp(vals_norm, row_ff, col_ff)
    vb   = _smp(vals_norm, row_bf, col_bf)
    ne_f = (_smp(ne_mask.astype(np.float32),  row_ff, col_ff) > 0.5) & inside & valid_f
    ne_b = (_smp(ne_mask.astype(np.float32),  row_bf, col_bf) > 0.5) & inside & valid_b
    hd_f = (_smp(has_data.astype(np.float32), row_ff, col_ff) > 0.5) & inside & valid_f
    hd_b = (_smp(has_data.astype(np.float32), row_bf, col_bf) > 0.5) & inside & valid_b

    if active_mask is not None:
        act_f = (_smp(active_mask.astype(np.float32), row_ff, col_ff) > 0.5) & inside & valid_f
        act_b = (_smp(active_mask.astype(np.float32), row_bf, col_bf) > 0.5) & inside & valid_b
    else:
        act_f = hd_f   # band mode: all GEE-covered land is "active"
        act_b = hd_b

    # Build RGBA canvas: transparent outside, white inside
    img = np.zeros((G, G, 4), dtype=np.uint8)
    img[inside] = (255, 255, 255, 255)

    def _paint(mask, grey):
        img[mask, 0] = grey
        img[mask, 1] = grey
        img[mask, 2] = grey

    # ── Back hemisphere (paint first; front will overwrite where needed) ──────
    # Back land is only visible through front-hemisphere ocean gaps (~ne_f).

    # 1. Back no-data land (NE land but no GEE embedding)
    nd_grey_b = int(255 - (255 - NO_DATA_GREY) * back_blend)
    _paint(ne_b & ~hd_b & ~ne_f, nd_grey_b)

    # 2. Back passive data land (non-cluster, cluster mode only)
    if passive_grey is not None:
        pg_b = int(255 - (255 - passive_grey) * back_blend)
        _paint(hd_b & ~act_b & ~ne_f, pg_b)

    # 3. Back active land
    g_b = np.clip(255 - vb * 255.0 * back_blend, 0, 255).astype(np.uint8)
    m_b = act_b & ~ne_f
    img[m_b, 0] = g_b[m_b]
    img[m_b, 1] = g_b[m_b]
    img[m_b, 2] = g_b[m_b]

    # ── Front hemisphere ──────────────────────────────────────────────────────

    # 4. Front no-data land (NE land, no GEE embedding — polar ice caps etc.)
    _paint(ne_f & ~hd_f, NO_DATA_GREY)

    # 5. Front passive data land (cluster mode only)
    if passive_grey is not None:
        _paint(hd_f & ~act_f, passive_grey)

    # 6. Front active land
    g_f = np.clip(255 - vf * 255.0, 0, 255).astype(np.uint8)
    img[act_f, 0] = g_f[act_f]
    img[act_f, 1] = g_f[act_f]
    img[act_f, 2] = g_f[act_f]

    return Image.fromarray(img, "RGBA")


# ─────────────────────────────────────────────────────────────────────────────
# GRID ASSEMBLY
# ─────────────────────────────────────────────────────────────────────────────

def assemble_grid(
    globe_imgs: list,
    labels: list,
    grid_rows: int,
    grid_cols: int,
    globe_d: int,
    gap: int,
    label_h: int,
    pad: int,
    font: ImageFont.FreeTypeFont,
) -> Image.Image:
    """Composite RGBA globe images and labels onto a white RGB canvas."""
    cell_w   = globe_d + gap
    cell_h   = globe_d + gap + label_h
    canvas_w = grid_cols * cell_w - gap + 2 * pad
    canvas_h = grid_rows * cell_h - gap + 2 * pad

    canvas = Image.new("RGB", (canvas_w, canvas_h), BG_COLOR)
    draw   = ImageDraw.Draw(canvas)

    for idx, (globe_img, label) in enumerate(zip(globe_imgs, labels)):
        row = idx // grid_cols
        col = idx  % grid_cols
        x0  = pad + col * cell_w
        y0  = pad + row * cell_h

        # Alpha-composite globe onto local white background, then paste
        bg = Image.new("RGBA", (globe_d, globe_d), (255, 255, 255, 255))
        bg.alpha_composite(globe_img)
        canvas.paste(bg.convert("RGB"), (x0, y0))

        # Thin circle border
        draw.ellipse(
            [x0, y0, x0 + globe_d - 1, y0 + globe_d - 1],
            outline=RING_COLOR, width=1,
        )

        # Centred label
        bb  = draw.textbbox((0, 0), label, font=font)
        tw, th = bb[2] - bb[0], bb[3] - bb[1]
        tx  = x0 + (globe_d - tw) // 2
        ty  = y0 + globe_d + (label_h - th) // 2
        draw.text((tx, ty), label, fill=LABEL_COLOR, font=font)

    return canvas


# ─────────────────────────────────────────────────────────────────────────────
# PART 1 — 8×8 band grid
# ─────────────────────────────────────────────────────────────────────────────

def _band_rotations(n: int, clat: float = CENTER_LAT):
    """Evenly spaced longitudes so consecutive globes rotate around Earth."""
    return [(i * 360.0 / n, clat) for i in range(n)]


def make_bands_grid(data_ext, ne_mask, has_data, lat_grid, lon_grid):
    print("\n── Part 1: rendering 64 band globes ──────────────────────────────", flush=True)
    rotations = _band_rotations(64)

    indiv_dir = OUTPUT_DIR / "globes_bands"
    indiv_dir.mkdir(exist_ok=True)

    globe_imgs = []   # downscaled versions for the grid
    for b in range(64):
        clon, clat = rotations[b]
        print(f"  band {b + 1:2d}/64  {BANDS[b]}  lon={clon:.1f}° …", flush=True)

        # Render at full individual resolution
        globe_hi = render_globe(
            values=data_ext[b],
            ne_mask=ne_mask,
            has_data=has_data,
            lat_grid=lat_grid,
            lon_grid=lon_grid,
            globe_d=GLOBE_D1_INDIV,
            clon=clon,
            clat=clat,
        )

        # Save individual high-res PNG (white background, no alpha)
        bg = Image.new("RGBA", (GLOBE_D1_INDIV, GLOBE_D1_INDIV), (255, 255, 255, 255))
        bg.alpha_composite(globe_hi)
        bg.convert("RGB").save(str(indiv_dir / f"{BANDS[b]}.png"), "PNG", compress_level=3)

        # Downscale for the grid composite
        globe_imgs.append(globe_hi.resize((GLOBE_D1, GLOBE_D1), Image.LANCZOS))

    canvas = assemble_grid(
        globe_imgs, labels=BANDS,
        grid_rows=8, grid_cols=8,
        globe_d=GLOBE_D1, gap=GAP1,
        label_h=LABEL_H1, pad=PAD1,
        font=FONT1,
    )
    out = OUTPUT_DIR / "globes_bands.png"
    canvas.save(str(out), "PNG", compress_level=3)
    print(f"  → saved {out.name}  ({canvas.width}×{canvas.height} px)", flush=True)
    print(f"  → saved {len(globe_imgs)} individual globes ({GLOBE_D1_INDIV}px) to {indiv_dir.name}/", flush=True)


# ─────────────────────────────────────────────────────────────────────────────
# PART 2 — k-means cluster grid
# ─────────────────────────────────────────────────────────────────────────────

def make_clusters_grid(data_ext, ne_mask, has_data, lat_grid, lon_grid, n_clusters: int):
    print(f"\n── Part 2: k-means (k={n_clusters}) on land pixels ──────────────", flush=True)

    # Feature matrix — only pixels with real GEE embedding data
    land_feats = data_ext[:, has_data].T.astype(np.float32)  # (N, 64)
    print(f"  {land_feats.shape[0]} land pixels × {land_feats.shape[1]} bands", flush=True)

    km = MiniBatchKMeans(
        n_clusters=n_clusters,
        n_init=10,
        batch_size=4096,
        random_state=42,
        max_iter=500,
        verbose=0,
    )
    km.fit(land_feats)
    labels_flat = km.labels_   # (N_land,)

    # Map back to 2-D grid (-1 = no embedding data)
    cluster_map = np.full(has_data.shape, -1, dtype=np.int32)
    cluster_map[has_data] = labels_flat

    # Value map: 1.0 on all GEE land (→ cluster pixels render black)
    vals_uniform = np.zeros(has_data.shape, dtype=np.float32)
    vals_uniform[has_data] = 1.0
    # passive grey for non-cluster data land
    passive_grey = int(255 * (1.0 - CLUSTER_OTHER_VAL))  # ≈ 222

    print("  Computing cluster centroids and rendering globes …", flush=True)
    globe_imgs = []
    for k in range(n_clusters):
        cnt = int((labels_flat == k).sum())

        # Centroid lat/lon — rotate globe to face this cluster
        k_rows, k_cols = np.where(cluster_map == k)
        k_lats = lat_grid[k_rows]
        k_lons = lon_grid[k_cols]
        clon = float(np.degrees(
            np.arctan2(np.sin(np.radians(k_lons)).mean(),
                       np.cos(np.radians(k_lons)).mean())
        ))
        clat = float(np.clip(k_lats.mean(), -50.0, 50.0))

        print(f"  cluster {k + 1}/{n_clusters}  ({cnt:,} px)  centroid=({clat:.1f}°N, {clon:.1f}°E) …",
              flush=True)

        active = (cluster_map == k)

        globe = render_globe(
            values=vals_uniform,
            ne_mask=ne_mask,
            has_data=has_data,
            lat_grid=lat_grid,
            lon_grid=lon_grid,
            globe_d=GLOBE_D2,
            clon=clon,
            clat=clat,
            min_val=0.0,
            max_val=1.0,
            active_mask=active,
            passive_grey=passive_grey,
        )
        globe_imgs.append(globe)

    import math
    grid_cols = math.ceil(math.sqrt(n_clusters))
    grid_rows = math.ceil(n_clusters / grid_cols)
    labels = [f"Cluster {k + 1}" for k in range(n_clusters)]
    canvas = assemble_grid(
        globe_imgs, labels=labels,
        grid_rows=grid_rows, grid_cols=grid_cols,
        globe_d=GLOBE_D2, gap=GAP2,
        label_h=LABEL_H2, pad=PAD2,
        font=FONT2,
    )
    out = OUTPUT_DIR / f"clusters_grid_k{n_clusters:02d}.png"
    canvas.save(str(out), "PNG", compress_level=3)
    print(f"  → saved {out.name}  ({canvas.width}×{canvas.height} px)", flush=True)


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    data_ext, ne_mask, has_data, lat_grid, lon_grid = load_global_data()
    print(f"Data: {data_ext.shape},  GEE land pixels: {has_data.sum():,},  NE land pixels: {ne_mask.sum():,}", flush=True)

    make_bands_grid(data_ext, ne_mask, has_data, lat_grid, lon_grid)
    for k in N_CLUSTERS:
        make_clusters_grid(data_ext, ne_mask, has_data, lat_grid, lon_grid, k)

    print("\nDone.", flush=True)
