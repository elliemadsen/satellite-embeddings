"""
Pole-centered orthographic maps of AlphaEarth embeddings.

The objective is to highlight that GEE coverage ends at ±85° latitude,
leaving a white disc at each pole where no embeddings exist.

Visual encoding
---------------
• Land with GEE embedding data  → greyscale (mean of all 64 bands)
• Land without GEE data (polar  → white  ← the "hole"
  caps beyond ±85° latitude)
• Ocean                         → soft grey background
• ±85° coverage boundary        → red circle annotation

Data loaded from ../globes/cache/ (built by thesis/globes/generate.py).

Outputs: south_pole.png, north_pole.png, poles.png (side by side)
"""

import os
import numpy as np
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
from scipy.ndimage import map_coordinates

# ── Paths ─────────────────────────────────────────────────────────────────────
CACHE_DIR  = Path(__file__).parent.parent / "globes" / "cache"
OUTPUT_DIR = Path(__file__).parent
OUTPUT_DIR.mkdir(exist_ok=True)

# ── Grid constants — must match globes/generate.py ────────────────────────────
GRID_H, GRID_W = 170, 360
POLAR_PAD = 5               # extra rows added beyond ±85°

# ── Layout ────────────────────────────────────────────────────────────────────
GLOBE_D      = 1600         # diameter of each globe image (px)
LABEL_H      = 160          # vertical space below globe for the label
CANVAS_GAP   = 80           # gap between the two globes in the combined image

# ── Visual style ──────────────────────────────────────────────────────────────
BG_COLOR      = (255, 255, 255)
OCEAN_COL     = (232, 235, 240)   # soft blue-grey for ocean
LAND_COL_LO   = (210, 210, 210)   # lightest covered land value
LAND_COL_HI   = (20,  20,  20)    # darkest covered land value
LAND_NODATA   = (255, 255, 255)   # uncovered polar gap → pure white
RING_COL      = (155, 155, 155)   # outer globe border
BACK_BLEND    = 0.28              # back-hemisphere land opacity (0 = invisible)
LABEL_COLOR   = (30, 30, 30)
LABEL_FONT_SZ = 54
SUB_FONT_SZ   = 34


# ─────────────────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def _load_font(size: int) -> ImageFont.FreeTypeFont:
    candidates = [
        str(Path.home() / "Library/Fonts/Roboto.ttf"),
        str(Path.home() / "Library/Fonts/Roboto-Regular.ttf"),
        "/usr/share/fonts/truetype/roboto/Roboto-Regular.ttf",
        "/System/Library/Fonts/HelveticaNeue.ttc",
        "/System/Library/Fonts/Helvetica.ttc",
        "/System/Library/Fonts/SFNS.ttf",
        "/System/Library/Fonts/Avenir Next.ttc",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
    ]
    for p in candidates:
        if os.path.exists(p):
            return ImageFont.truetype(p, size)
    return ImageFont.load_default()


FONT_LABEL = _load_font(LABEL_FONT_SZ)
FONT_SUB   = _load_font(SUB_FONT_SZ)


# ─────────────────────────────────────────────────────────────────────────────
# DATA LOADING
# ─────────────────────────────────────────────────────────────────────────────

def load_data():
    """
    Returns
    -------
    ne_mask   : (H_ext, W) bool   – Natural Earth land (full ±90° extent)
    has_data  : (H_ext, W) bool   – pixels with real GEE embedding values
    vals_norm : (H_ext, W) float  – mean embedding value, normalised [0,1]
    lat_grid  : (H_ext,)  float   – latitudes decreasing top→bottom
    lon_grid  : (W,)      float   – longitudes -180 → +180
    """
    for name in ("embeddings.npy", "land_mask.npy", "ne_land_mask.npy"):
        path = CACHE_DIR / name
        if not path.exists():
            raise FileNotFoundError(
                f"Cache file not found: {path}\n"
                "Run thesis/globes/generate.py first to populate the cache."
            )

    print("Loading cached embeddings …", flush=True)
    data    = np.load(CACHE_DIR / "embeddings.npy")               # (64, 170, 360)
    gee_msk = np.load(CACHE_DIR / "land_mask.npy").astype(bool)   # (170, 360)
    ne_msk  = np.load(CACHE_DIR / "ne_land_mask.npy").astype(bool) # (180, 360)

    # Reconstruct the POLAR_PAD-extended lat grid (matches globes/generate.py)
    gee_lat  = np.linspace(85.0, -85.0, GRID_H)
    lon_grid = np.linspace(-180.0, 180.0, GRID_W)
    step     = (gee_lat[0] - gee_lat[-1]) / (GRID_H - 1)   # ≈ 1.006 °/row
    lat_grid = np.concatenate([
        gee_lat[0] + np.arange(POLAR_PAD, 0, -1) * step,
        gee_lat,
        gee_lat[-1] - np.arange(1, POLAR_PAD + 1) * step,
    ])
    H_ext = len(lat_grid)   # 180

    # Extended embedding array (polar rows stay zero — no GEE data there)
    data_ext = np.zeros((64, H_ext, GRID_W), dtype=np.float32)
    data_ext[:, POLAR_PAD : POLAR_PAD + GRID_H, :] = data

    # has_data: GEE coverage AND Natural Earth land (avoids sea-ice false hits)
    has_data = np.zeros((H_ext, GRID_W), dtype=bool)
    has_data[POLAR_PAD : POLAR_PAD + GRID_H, :] = gee_msk
    has_data &= ne_msk

    # Single scalar per pixel: mean over all 64 bands, normalised to [0, 1]
    mean_vals  = data_ext.mean(axis=0)                            # (H_ext, W)
    land_vals  = mean_vals[has_data]
    v_lo = float(np.percentile(land_vals, 2))
    v_hi = float(np.percentile(land_vals, 98))
    vals_norm = np.clip((mean_vals - v_lo) / max(v_hi - v_lo, 1e-8), 0.0, 1.0)

    n_covered = int(has_data.sum())
    n_ne      = int(ne_msk.sum())
    print(f"  GEE-covered land pixels : {n_covered:,}", flush=True)
    print(f"  Natural Earth land pixels: {n_ne:,}",    flush=True)
    print(f"  Uncovered NE land pixels : {n_ne - n_covered:,}", flush=True)

    return ne_msk, has_data, vals_norm, lat_grid, lon_grid


# ─────────────────────────────────────────────────────────────────────────────
# ORTHOGRAPHIC RENDERER
# ─────────────────────────────────────────────────────────────────────────────

def _camera_basis(clon_deg: float, clat_deg: float):
    la = np.radians(clat_deg)
    lo = np.radians(clon_deg)
    f = np.array([np.cos(la)*np.cos(lo), np.cos(la)*np.sin(lo), np.sin(la)])
    r = np.array([-np.sin(lo),            np.cos(lo),            0.0       ])
    u = np.array([-np.sin(la)*np.cos(lo), -np.sin(la)*np.sin(lo), np.cos(la)])
    return f, r, u


def render_globe(
    ne_mask:   np.ndarray,
    has_data:  np.ndarray,
    vals_norm: np.ndarray,
    lat_grid:  np.ndarray,
    lon_grid:  np.ndarray,
    clon:      float,
    clat:      float,
    globe_d:   int = GLOBE_D,
) -> Image.Image:
    """
    Render an orthographic globe as an RGBA PIL image.

    Land with GEE data   → greyscale from vals_norm (dark = high value)
    Land without GEE data→ LAND_NODATA (white) — the polar gap
    Ocean                → OCEAN_COL background
    """
    f, r, u = _camera_basis(clon, clat)
    H, W = ne_mask.shape
    half = globe_d / 2.0
    G    = globe_d

    gy, gx = np.mgrid[0:G, 0:G]
    sx = (gx + 0.5 - half) / half
    sy = -(gy + 0.5 - half) / half
    d2     = sx**2 + sy**2
    inside = d2 <= 1.0
    nz     = np.where(inside, np.sqrt(np.clip(1.0 - d2, 0.0, 1.0)), 0.0)

    def _project(sign: float):
        Wx = sx*r[0] + sy*u[0] + sign*nz*f[0]
        Wy = sx*r[1] + sy*u[1] + sign*nz*f[1]
        Wz = sx*r[2] + sy*u[2] + sign*nz*f[2]
        lat_p = np.degrees(np.arcsin(np.clip(Wz, -1.0, 1.0)))
        lon_p = np.degrees(np.arctan2(Wy, Wx))
        valid = (lat_p >= lat_grid[-1]) & (lat_p <= lat_grid[0])
        rowf = np.clip(
            (lat_grid[0] - lat_p) / (lat_grid[0] - lat_grid[-1]) * (H - 1),
            0.0, H - 1.0,
        )
        colf = np.clip(
            (lon_p - lon_grid[0]) / (lon_grid[-1] - lon_grid[0]) * (W - 1),
            0.0, W - 1.0,
        )
        return rowf, colf, valid

    rowf, colf, vf = _project(+1)
    rowb, colb, vb = _project(-1)

    def smp(arr: np.ndarray, rf, cf) -> np.ndarray:
        return map_coordinates(
            arr.astype(np.float32), [rf.ravel(), cf.ravel()],
            order=1, mode="nearest",
        ).reshape(G, G)

    ne_f = (smp(ne_mask.astype(np.float32),  rowf, colf) > 0.5) & inside & vf
    ne_b = (smp(ne_mask.astype(np.float32),  rowb, colb) > 0.5) & inside & vb
    hd_f = (smp(has_data.astype(np.float32), rowf, colf) > 0.5) & inside & vf
    hd_b = (smp(has_data.astype(np.float32), rowb, colb) > 0.5) & inside & vb
    vn_f = smp(vals_norm, rowf, colf)
    vn_b = smp(vals_norm, rowb, colb)

    # ── Start: ocean-filled inside, transparent outside ──────────────────────
    img = np.zeros((G, G, 4), dtype=np.uint8)
    img[inside] = (*OCEAN_COL, 255)

    def _land_grey(v: np.ndarray) -> np.ndarray:
        """vals_norm → grey level (float, same shape)."""
        lo, hi = float(LAND_COL_LO[0]), float(LAND_COL_HI[0])
        return np.clip(lo + v * (hi - lo), hi, lo)   # lo=210 (light), hi=20 (dark)

    def _paint(mask: np.ndarray, grey_arr: np.ndarray):
        g = grey_arr[mask].astype(np.uint8)
        img[mask, 0] = g
        img[mask, 1] = g
        img[mask, 2] = g
        img[mask, 3] = 255

    # ── Back hemisphere (visible through front-side ocean) ───────────────────
    # back no-data land: near-white (barely visible through ocean)
    back_nodata_g = int(255 - (255 - LAND_NODATA[0]) * BACK_BLEND)  # stays 255
    nd_back = np.full((G, G), back_nodata_g, dtype=np.float32)
    _paint(ne_b & ~hd_b & ~ne_f, nd_back)

    # back data land: embedding greyscale blended toward white
    grey_b = _land_grey(vn_b)
    grey_b_blended = 255.0 - (255.0 - grey_b) * BACK_BLEND
    _paint(hd_b & ~ne_f, grey_b_blended)

    # ── Front hemisphere ──────────────────────────────────────────────────────
    # no-data land → WHITE (the polar coverage gap)
    _paint(ne_f & ~hd_f, np.full((G, G), float(LAND_NODATA[0])))

    # data land → greyscale from embedding mean
    _paint(hd_f, _land_grey(vn_f))

    return Image.fromarray(img, "RGBA")


# ─────────────────────────────────────────────────────────────────────────────
# SINGLE POLE IMAGE
# ─────────────────────────────────────────────────────────────────────────────

def make_pole_panel(
    ne_mask:   np.ndarray,
    has_data:  np.ndarray,
    vals_norm: np.ndarray,
    lat_grid:  np.ndarray,
    lon_grid:  np.ndarray,
    clat:      float,
    title:     str,
    subtitle:  str,
    globe_d:   int = GLOBE_D,
) -> Image.Image:
    """Render one pole-centred globe with title and subtitle below it."""
    clon = 0.0
    print(f"  Rendering {title} …", flush=True)
    globe_rgba = render_globe(
        ne_mask, has_data, vals_norm, lat_grid, lon_grid,
        clon=clon, clat=clat, globe_d=globe_d,
    )

    panel_h = globe_d + LABEL_H
    panel   = Image.new("RGB", (globe_d, panel_h), BG_COLOR)

    # Composite the globe
    bg = Image.new("RGBA", (globe_d, globe_d), (255, 255, 255, 255))
    bg.alpha_composite(globe_rgba)
    panel.paste(bg.convert("RGB"), (0, 0))

    draw = ImageDraw.Draw(panel)

    # Outer ring
    draw.ellipse([1, 1, globe_d - 2, globe_d - 2], outline=RING_COL, width=2)

    # Title
    bb = draw.textbbox((0, 0), title, font=FONT_LABEL)
    tw = bb[2] - bb[0]
    ty = globe_d + 14
    draw.text(((globe_d - tw) // 2, ty), title, fill=LABEL_COLOR, font=FONT_LABEL)

    # Subtitle
    bb2 = draw.textbbox((0, 0), subtitle, font=FONT_SUB)
    tw2 = bb2[2] - bb2[0]
    ty2 = ty + LABEL_FONT_SZ + 4
    draw.text(((globe_d - tw2) // 2, ty2), subtitle,
              fill=(130, 130, 130), font=FONT_SUB)

    return panel


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main():
    ne_mask, has_data, vals_norm, lat_grid, lon_grid = load_data()

    print("\nRendering pole globes …", flush=True)

    south = make_pole_panel(
        ne_mask, has_data, vals_norm, lat_grid, lon_grid,
        clat=-90.0,
        title="South Pole",
        subtitle="AlphaEarth embedding coverage ends at 85°S",
    )
    north = make_pole_panel(
        ne_mask, has_data, vals_norm, lat_grid, lon_grid,
        clat=90.0,
        title="North Pole",
        subtitle="AlphaEarth embedding coverage ends at 85°N",
    )

    # ── Save individual images ────────────────────────────────────────────────
    south.save(str(OUTPUT_DIR / "south_pole.png"), "PNG", compress_level=3)
    north.save(str(OUTPUT_DIR / "north_pole.png"), "PNG", compress_level=3)
    print(f"  → saved south_pole.png  ({south.width}×{south.height} px)", flush=True)
    print(f"  → saved north_pole.png  ({north.width}×{north.height} px)", flush=True)

    # ── Side-by-side composite ────────────────────────────────────────────────
    W = south.width + CANVAS_GAP + north.width
    H = max(south.height, north.height)
    combined = Image.new("RGB", (W, H), BG_COLOR)
    combined.paste(south, (0, 0))
    combined.paste(north, (south.width + CANVAS_GAP, 0))
    combined.save(str(OUTPUT_DIR / "poles.png"), "PNG", compress_level=3)
    print(f"  → saved poles.png  ({W}×{H} px)", flush=True)

    print("\nDone.", flush=True)


if __name__ == "__main__":
    main()
