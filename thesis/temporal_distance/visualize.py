"""
visualize.py — Script 2
~~~~~~~~~~~~~~~~~~~~~~~
Reads sites.json (produced by find_sites.py) and writes six figures to outputs/:

  01_distance_over_time.png      – cumulative cosine distance from 2017 baseline
  02_falsecolor_{name}.png       – 3-row strip: true color / embedding A00–A02 / B&W distance from 2017
  03_trajectory_{name}.png       – 2-D PCA trajectory through embedding space per site
  04_trajectory_combined.png     – all sites in a shared PCA space
  05_delta_over_time.png         – year-to-year (consecutive) cosine distance
  06_embedding_profile_{name}.png– 64-dim embedding value lines, one per year

Sentinel-2 tiles are cached in tile_cache/ so re-runs are fast.

Usage:
    conda run -n geo python visualize.py
"""

import ee
import io
import json
import math
import os
import requests

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.lines as mlines
from matplotlib import font_manager
from pathlib import Path
from scipy.spatial.distance import cosine
from sklearn.decomposition import PCA
from PIL import Image

# ── Earth Engine ──────────────────────────────────────────────────────────────
try:
    ee.Initialize(project="gsapp-map")
except Exception:
    ee.Authenticate()
    ee.Initialize(project="gsapp-map")

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE_DIR   = Path(__file__).parent
SITES_JSON = BASE_DIR / "sites.json"
OUT_DIR    = BASE_DIR / "outputs"
TILE_DIR   = BASE_DIR / "tile_cache"
OUT_DIR.mkdir(exist_ok=True)
TILE_DIR.mkdir(exist_ok=True)

# ── Font ──────────────────────────────────────────────────────────────────────
for _fp in [
    str(Path.home() / "Library/Fonts/Roboto-Regular.ttf"),
    "/System/Library/Fonts/SFNS.ttf",
    "/System/Library/Fonts/HelveticaNeue.ttc",
    "/System/Library/Fonts/Helvetica.ttc",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
]:
    if os.path.exists(_fp):
        font_manager.fontManager.addfont(_fp)
        _prop = font_manager.FontProperties(fname=_fp)
        plt.rcParams["font.family"] = _prop.get_name()
        break

matplotlib.rcParams.update({
    "font.size":           11,
    "figure.facecolor":    "white",
    "axes.facecolor":      "white",
    "axes.spines.top":     False,
    "axes.spines.right":   False,
    "text.color":          "#222222",
    "axes.labelcolor":     "#555555",
    "xtick.color":         "#888888",
    "ytick.color":         "#888888",
    "axes.grid":           False,
})

# ── Load data ─────────────────────────────────────────────────────────────────
print(f"Loading {SITES_JSON} …")
with open(SITES_JSON) as f:
    sites = json.load(f)
# Sites to skip without re-running find_sites.py
DISABLED_SITES = {"nairobi", "lusail"}
sites = [s for s in sites if s["name"] not in DISABLED_SITES]
print(f"  {len(sites)} sites loaded (disabled: {DISABLED_SITES}).")

# One colour per site — Flat UI palette (warm/cool contrast, print-safe)
PALETTE = ["#000000", "#99d0f5", "#abeb83", "#f1a26d"]

# Sites excluded from line charts (still rendered in 02_ strips)
CHART_EXCLUDE = {"gaza"}
YEARS_STR = [str(y) for y in range(2017, 2026)]

# ── Helpers ───────────────────────────────────────────────────────────────────
def get_embedding_array(site: dict) -> tuple[list[str], np.ndarray]:
    """Return (sorted_years, array of shape (T, 64)) for a site."""
    yrs = sorted(y for y in YEARS_STR if y in site["embeddings"])
    X   = np.array([site["embeddings"][y] for y in yrs], dtype=np.float64)
    return yrs, X


def save_fig(fig: plt.Figure, name: str) -> None:
    path = OUT_DIR / name
    fig.savefig(path, bbox_inches="tight", dpi=200, facecolor="white")
    plt.close(fig)
    print(f"  ✓ {path}")


# ─────────────────────────────────────────────────────────────────────────────
# 01 — Cumulative distance from 2017
# ─────────────────────────────────────────────────────────────────────────────
print("\nRendering 01_distance_over_time.png …")

fig, ax = plt.subplots(figsize=(9, 5))

for site, color in zip(sites, PALETTE):
    if site["name"] in CHART_EXCLUDE:
        continue
    dist = site.get("distance_from_2017", {})
    if not dist:
        continue
    yrs  = sorted(dist.keys())
    vals = [dist[y] for y in yrs]
    xi   = [int(y) for y in yrs]
    ax.plot(xi, vals, color=color, lw=2, marker="o", ms=5,
            label=site["label"], zorder=3)

ax.set_xlabel("Year")
ax.set_ylabel("Cosine distance from 2017 embedding")
ax.set_title("Satellite Embedding Change Detection (2017-2025)", fontsize=13, loc="left",
             pad=10, color="#111111")
leg = ax.legend(fontsize=8.5, frameon=False, loc="upper left")
for h in leg.legend_handles:
    h.set_marker("")
ax.set_xticks(range(2017, 2026))
ax.tick_params(axis="both", length=3)

save_fig(fig, "01_distance_over_time.png")


# ─────────────────────────────────────────────────────────────────────────────
# 05 — Consecutive year-to-year delta
# ─────────────────────────────────────────────────────────────────────────────
print("Rendering 01_delta_over_time.png …")

fig, ax = plt.subplots(figsize=(9, 5))

for site, color in zip(sites, PALETTE):
    if site["name"] in CHART_EXCLUDE:
        continue
    yrs_list, X = get_embedding_array(site)
    if len(X) < 2:
        continue
    deltas   = [float(cosine(X[i], X[i + 1])) for i in range(len(X) - 1)]
    midyears = [int(yrs_list[i]) + 0.5 for i in range(len(yrs_list) - 1)]
    ax.plot(midyears, deltas, color=color, lw=2, marker="s", ms=4,
            label=site["label"], zorder=3)

ax.set_xlabel("Year interval")
ax.set_ylabel("Year-to-year cosine distance")
ax.set_title("Annual rate of embedding change", fontsize=13, loc="left",
             pad=10, color="#111111")
leg = ax.legend(fontsize=8.5, frameon=False, loc="upper right")
for h in leg.legend_handles:
    h.set_marker("")
ax.set_xticks([y + 0.5 for y in range(2017, 2025)])
ax.set_xticklabels([f"{y}–{y+1}" for y in range(2017, 2025)],
                   rotation=30, ha="right")

save_fig(fig, "01_delta_over_time.png")


# ─────────────────────────────────────────────────────────────────────────────
# 02 — Annual comparison strips: true color / embedding A00–A02 / B&W distance
#      Three rows per site × one column per year
# ─────────────────────────────────────────────────────────────────────────────
TILE_PX           = 220   # pixels per thumbnail
RADIUS_M          = 8000  # buffer radius (metres) — ~8 km half-scale view
VIS_EMB_MIN       = -0.3  # AlphaEarth embedding colour stretch
VIS_EMB_MAX       =  0.3
UMAP_SCALE_M      = 160   # pixel resolution (m) used when sampling embeddings for UMAP
GRID_PAD          = 0.3   # uniform gap between image tiles — adjust to taste (tight_layout w_pad = h_pad)
CLOUD_PCT_FILTER  = 80    # max CLOUDY_PIXEL_PERCENTAGE for S2 scene pre-filter (per-pixel masking still applies)
CLOUD_PROB_MAX    = 30    # s2cloudless probability threshold (0–100); pixels above this are masked
TILE_INCH         = 1.6   # side length (inches) per image tile — equal w/h keeps grid gaps symmetric


def _mask_s2_clouds(img):
    qa = img.select("QA60")
    return img.updateMask(
        qa.bitwiseAnd(1 << 10).eq(0).And(qa.bitwiseAnd(1 << 11).eq(0))
    )


def _s2_tile_url(lat, lon, year, bands, vis_min, vis_max, region=None):
    """Sentinel-2 cloud-free median composite using QA60 + s2cloudless probability mask."""
    if region is None:
        region = _proj_region(lat, lon)
    start, end = f"{year}-01-01", f"{year + 1}-01-01"
    s2   = (ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
            .filterDate(start, end)
            .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", CLOUD_PCT_FILTER)))
    s2cp = ee.ImageCollection("COPERNICUS/S2_CLOUD_PROBABILITY").filterDate(start, end)
    joined = ee.ImageCollection(
        ee.Join.saveFirst("cloud_prob").apply(
            primary=s2, secondary=s2cp,
            condition=ee.Filter.equals(
                leftField="system:index", rightField="system:index"),
        )
    )
    def _mask_cp(img):
        qa    = img.select("QA60")
        qa_ok = qa.bitwiseAnd(1 << 10).eq(0).And(qa.bitwiseAnd(1 << 11).eq(0))
        cp    = ee.Image(img.get("cloud_prob")).select("probability")
        return img.updateMask(qa_ok).updateMask(cp.lt(CLOUD_PROB_MAX))
    # Fill any remaining masked pixels (persistent cloud gaps) with a QA60-only fallback
    qa60_only = s2.map(_mask_s2_clouds).median().select(bands)
    composite = joined.map(_mask_cp).median().select(bands).unmask(qa60_only)
    return composite.getThumbURL({
        "region":     region,
        "dimensions": f"{TILE_PX}x{TILE_PX}",
        "format":     "png",
        "min":        vis_min,
        "max":        vis_max,
    })


def _fetch_umap_tiles(
    site: dict, yrs_list: list[str], clat: float, clon: float, coord_tag: str,
    region=None,
) -> dict:
    """Return {yr_str: PIL.Image} UMAP-3D-coloured embedding tiles.

    UMAP is fit on all years combined so the 3-D colour space is consistent
    across time.  Results are cached per-year as PNGs; raw pixel arrays are
    cached as .npy files so UMAP need not be re-run when only new years are
    added.
    """
    import umap as umap_lib  # umap-learn

    _bands  = [f"A{i:02d}" for i in range(64)]
    _region = region if region is not None else _proj_region(clat, clon)
    _col    = ee.ImageCollection("GOOGLE/SATELLITE_EMBEDDING/V1/ANNUAL")
    _name   = site["name"]

    # Return immediately if every year is already cached as a PNG
    umap_paths = {yr: TILE_DIR / f"umap_{_name}_{coord_tag}_s{UMAP_SCALE_M}_{yr}.png"
                  for yr in yrs_list}
    if all(p.exists() for p in umap_paths.values()):
        return {yr: Image.open(umap_paths[yr]).convert("RGB") for yr in yrs_list}

    # ── Step 1: fetch raw 64-dim pixel arrays ──────────────────────────────
    raw: dict[str, np.ndarray | None] = {}
    for yr in yrs_list:
        yr_i     = int(yr)
        npy_path = TILE_DIR / f"raw_{_name}_{coord_tag}_s{UMAP_SCALE_M}_{yr_i}.npy"
        if npy_path.exists():
            raw[yr] = np.load(str(npy_path))
            continue
        img_ee = (
            _col.filterDate(f"{yr_i}-01-01", f"{yr_i + 1}-01-01")
            .mosaic()
            .select(_bands)
            .reproject(crs="EPSG:3857", scale=UMAP_SCALE_M)
        )
        try:
            props = img_ee.sampleRectangle(
                region=_region, defaultValue=0
            ).getInfo()["properties"]
            h   = len(props["A00"])
            w   = len(props["A00"][0])
            arr = np.zeros((h, w, 64), dtype=np.float32)
            for bi, b in enumerate(_bands):
                arr[:, :, bi] = np.array(props[b], dtype=np.float32)
            np.save(str(npy_path), arr)
            raw[yr] = arr
        except Exception as exc:
            print(f" [umap-fetch-err {yr_i}: {exc}]", end="", flush=True)
            raw[yr] = None

    valid_yrs = [y for y in yrs_list if raw.get(y) is not None]
    if not valid_yrs:
        return {yr: None for yr in yrs_list}

    h, w = raw[valid_yrs[0]].shape[:2]

    # ── Step 2: fit UMAP on 2017 reference year, transform all others ──────
    # Anchors cluster colours to 2017 so colour changes mean land-cover change.
    ref_yr = "2017" if "2017" in valid_yrs else valid_yrs[0]
    ref_px = raw[ref_yr].reshape(-1, 64)
    print(f" [umap fit on {ref_yr}, {ref_px.shape[0]}px…]", end="", flush=True)
    reducer = umap_lib.UMAP(
        n_components=3, n_neighbors=15, min_dist=0.1,
        random_state=42, verbose=False,
    )
    ref_emb = reducer.fit_transform(ref_px)

    # Normalise against reference-year percentile range (fixed for all years)
    lo = np.percentile(ref_emb, 1, axis=0)
    hi = np.percentile(ref_emb, 99, axis=0)

    # ── Step 3: embed each year, normalise, save ───────────────────────────
    results: dict[str, Image.Image] = {}
    for yr in valid_yrs:
        px  = raw[yr].reshape(-1, 64)
        emb = ref_emb if yr == ref_yr else reducer.transform(px)
        norm  = np.clip((emb - lo) / np.maximum(hi - lo, 1e-9), 0, 1)
        chunk = norm.reshape(h, w, 3)
        rgb   = (chunk * 255).astype(np.uint8)
        pil   = Image.fromarray(rgb).resize((TILE_PX, TILE_PX), Image.LANCZOS)
        pil.save(str(umap_paths[yr]))
        results[yr] = pil

    return {yr: results.get(yr) for yr in yrs_list}


# Maximum cosine distance shown as black in the distance row (clip above this).
DIST_VIS_MAX = 0.35


def _proj_region(clat: float, clon: float) -> ee.Geometry:
    """Square EPSG:3857 bounding box ± RADIUS_M metres around (clat, clon).

    Using a projected square (not a geographic buffer+bounds) ensures that
    getThumbURL, sampleRectangle and the UMAP fetch all cover exactly the
    same footprint regardless of latitude.
    """
    x = clon * math.pi * 6_378_137 / 180
    y = math.log(math.tan((90 + clat) * math.pi / 360)) * 6_378_137
    return ee.Geometry.Rectangle(
        coords=[x - RADIUS_M, y - RADIUS_M, x + RADIUS_M, y + RADIUS_M],
        proj=ee.Projection("EPSG:3857"),
        geodesic=False,
    )


def _fetch_tile(url: str, cache_path: Path) -> Image.Image | None:
    """Download a tile, caching to disk on first fetch."""
    if cache_path.exists():
        return Image.open(cache_path).convert("RGB")
    try:
        resp = requests.get(url, timeout=120)
        if resp.status_code == 200:
            img = Image.open(io.BytesIO(resp.content)).convert("RGB")
            img.save(cache_path)
            return img
        print(f"      HTTP {resp.status_code}")
    except Exception as exc:
        print(f"      error: {exc}")
    return None


def _cosine_dist_url(lat: float, lon: float, year: int, ref_year: int = 2017,
                     region=None) -> str:
    """Per-pixel cosine distance image between ref_year and year AlphaEarth embeddings.

    Rendered as greyscale: white = 0 distance, black = DIST_VIS_MAX.
    """
    bands  = [f"A{i:02d}" for i in range(64)]
    region = region if region is not None else _proj_region(lat, lon)
    col    = ee.ImageCollection("GOOGLE/SATELLITE_EMBEDDING/V1/ANNUAL")

    emb_ref = col.filterDate(f"{ref_year}-01-01", f"{ref_year + 1}-01-01").mosaic().select(bands)
    emb_cur = col.filterDate(f"{year}-01-01",     f"{year + 1}-01-01").mosaic().select(bands)

    dot      = emb_ref.multiply(emb_cur).reduce(ee.Reducer.sum())
    norm_ref = emb_ref.pow(2).reduce(ee.Reducer.sum()).sqrt()
    norm_cur = emb_cur.pow(2).reduce(ee.Reducer.sum()).sqrt()
    cos_dist = ee.Image(1).subtract(dot.divide(norm_ref.multiply(norm_cur)))

    return cos_dist.getThumbURL({
        "region":     region,
        "dimensions": f"{TILE_PX}x{TILE_PX}",
        "format":     "png",
        "min":        0,
        "max":        DIST_VIS_MAX,
        "palette":    ["ffffff", "000000"],
    })


def _top3band_tile_url(lat: float, lon: float, year: int, band_indices: list[int],
                       region=None) -> str:
    """RGB tile using the three AlphaEarth band indices that changed most 2017→2024."""
    bands  = [f"A{i:02d}" for i in band_indices[:3]]
    region = region if region is not None else _proj_region(lat, lon)
    col    = ee.ImageCollection("GOOGLE/SATELLITE_EMBEDDING/V1/ANNUAL")
    img    = col.filterDate(f"{year}-01-01", f"{year + 1}-01-01").mosaic().select(bands)
    return img.getThumbURL({
        "region":     region,
        "dimensions": f"{TILE_PX}x{TILE_PX}",
        "format":     "png",
        "min":        VIS_EMB_MIN,
        "max":        VIS_EMB_MAX,
    })


ROW_LABELS   = [
    "Satellite Image\nTrue color\nSentinel-2",
    # "Embedding\nfalse color\n(UMAP 3D)",   # UMAP row disabled — uncomment to re-enable
    "",        # filled in dynamically per-site (top-3-bands label)
    "Satellite Embedding\nChange Detection\n(Cosine Distance)",
]
SPARSE_YEARS = ["2018", "2020", "2022", "2024"]


def _render_strip(
    site: dict,
    clat: float,
    yrs_to_show: list[str],
    rows_all: list[list],
    all_yrs: list[str],
    dist: dict,
    suffix: str,
    top3_label: str = "",
) -> None:
    """Build and save one falsecolor strip for the given year subset."""
    import textwrap
    n = len(yrs_to_show)
    if n == 0:
        return

    n_rows = len(rows_all)   # 4 rows: TC / UMAP / top-3 RGB / dist

    # Scale bar: 5 km ground distance, converted to projected (Web Mercator) fraction.
    cos_lat  = math.cos(math.radians(abs(clat)))
    bar_proj = 5_000 / max(cos_lat, 0.1)
    sb_frac  = bar_proj / (2 * RADIUS_M)
    if sb_frac > 0.25:
        bar_proj = 2_000 / max(cos_lat, 0.1)
        sb_frac  = bar_proj / (2 * RADIUS_M)
        sb_label = "2 km"
    else:
        sb_label = "5 km"

    # Top margin: title + subtitle + story + scale/N-arrow block
    TOP_MARGIN   = 0.15
    # Bottom margin: colorbar sits just below last image row
    BOT_MARGIN   = 0.06

    fig, axes = plt.subplots(
        n_rows, n,
        figsize=(n * TILE_INCH, n_rows * TILE_INCH + 1.2),
        dpi=200,
        squeeze=False,
    )
    fig.patch.set_facecolor("white")

    row_labels = list(ROW_LABELS)
    if top3_label:
        row_labels[1] = top3_label  # index 1 = top-3-band row (UMAP row is commented out)

    for row_i in range(n_rows):
        for ci, yr in enumerate(yrs_to_show):
            orig_ci = all_yrs.index(yr)
            ax      = axes[row_i, ci]
            tile    = rows_all[row_i][orig_ci]
            if tile is not None:
                ax.imshow(np.array(tile))
            else:
                ax.set_facecolor("#dddddd")
            ax.set_axis_off()

            if row_i == 0:
                ax.set_title(yr, fontsize=7.5, pad=3, color="#333333")

            if row_i == n_rows - 1:   # distance row — show delta value
                d_val = float(dist.get(yr, 0.0))
                d_str = "baseline" if yr == "2017" else f"\u0394={d_val:.3f}"
                ax.text(0.5, -0.04, d_str, transform=ax.transAxes,
                        fontsize=6.5, ha="center", va="top", color="#555555")

            if ci == 0:
                ax.text(
                    -0.06, 0.5, row_labels[row_i],
                    transform=ax.transAxes,
                    fontsize=7.5, va="center", ha="right",
                    color="#555555", linespacing=1.4,
                )

    # ── Layout: images fill [0, BOT_MARGIN, 1, 1-TOP_MARGIN] ─────────────
    plt.tight_layout(pad=0.3, w_pad=GRID_PAD, h_pad=GRID_PAD,
                     rect=[0, BOT_MARGIN, 1, 1.0 - TOP_MARGIN])

    # ── Post-process: equalise column/row gaps to the same absolute size ──
    _fw, _fh = fig.get_size_inches()
    _tw = axes[0, 0].get_position().width
    _th = axes[0, 0].get_position().height
    _cg = ((axes[0, 1].get_position().x0 - axes[0, 0].get_position().x1) * _fw
           ) if n > 1 else 0.0
    _rg = ((axes[0, 0].get_position().y0 - axes[1, 0].get_position().y1) * _fh
           ) if n_rows > 1 else 0.0
    _gs = [g for g in [_cg, _rg] if g > 0]
    if _gs:
        _gap_abs  = 0.15          # target gap in inches — equal horizontal + vertical
        _eq_col   = _gap_abs / _fw
        _eq_row   = _gap_abs / _fh
        _x0_ref   = axes[0, 0].get_position().x0
        _y1_ref   = axes[0, 0].get_position().y1
        for _ri in range(n_rows):
            for _ci in range(n):
                _x0 = _x0_ref + _ci * (_tw + _eq_col)
                _y1 = _y1_ref - _ri * (_th + _eq_row)
                axes[_ri, _ci].set_position([_x0, _y1 - _th, _tw, _th])

    # ── Colorbar — immediately below the last image row ───────────────────
    x0_cb  = axes[0,  0].get_position().x0
    x1_cb  = axes[0, -1].get_position().x1
    y_bot_last = axes[-1, 0].get_position().y0   # bottom of last image row
    cb_h   = 0.012
    cb_gap = 0.008
    cax = fig.add_axes([x0_cb, y_bot_last - cb_gap - cb_h,
                        x1_cb - x0_cb, cb_h])
    cb  = matplotlib.colorbar.ColorbarBase(
        cax,
        cmap=matplotlib.cm.get_cmap("gray_r"),
        norm=matplotlib.colors.Normalize(vmin=0, vmax=DIST_VIS_MAX),
        orientation="horizontal",
    )
    cb.outline.set_linewidth(0.4)
    cb.set_label(
        f"Cosine distance from 2017. Clipped at {DIST_VIS_MAX}. Darker = more change.",
        fontsize=6.5, color="#555555", loc="left",
    )
    cb.ax.tick_params(labelsize=6, colors="#888888", length=2)

    # ── Title/subtitle/story text — anchored from figure top ────────────────
    story_wrapped = textwrap.fill(site.get("story", ""), width=max(80, n * 18))
    clon = site.get("pinned_lon", site.get("found_lon", site["lon"]))
    lat_label = (
        f"{abs(clat):.4f}\N{DEGREE SIGN}{'N' if clat >= 0 else 'S'},"
        f"  {abs(clon):.4f}\N{DEGREE SIGN}{'E' if clon >= 0 else 'W'}"
    )
    y_title = 1.0 - 0.010
    fig.text(x0_cb, y_title, site["label"],
             fontsize=9.5, ha="left", va="top", color="#111111",
             fontweight="bold")
    fig.text(x0_cb, y_title - 0.033, lat_label,
             fontsize=7.0, ha="left", va="top", color="#888888")
    fig.text(x0_cb, y_title - 0.060, story_wrapped,
             fontsize=6.5, ha="left", va="top", color="#555555",
             linespacing=1.35)

    # ── Scale bar + north arrow — right side of the top margin ────────────
    # North arrow is to the LEFT of the scale bar.
    # Scale bar RIGHT edge is flush with the rightmost image column.
    x_right = x1_cb
    y_mid   = 1.0 - TOP_MARGIN / 2.0   # midpoint of top margin band

    # Scale bar dimensions in figure coords
    tile_w_fig = axes[0, 0].get_position().width
    sb_fig  = sb_frac * tile_w_fig * 2
    xs1     = x_right          # right end flush with image grid
    xs0     = xs1 - sb_fig     # left end
    y_elem  = y_mid

    # North arrow (centred to the left of the scale bar)
    x_arrow = xs0 - 0.055
    fig.patches.append(
        mpatches.FancyArrowPatch(
            (x_arrow, y_elem - 0.040), (x_arrow, y_elem + 0.040),
            transform=fig.transFigure,
            arrowstyle="-|>",
            color="#333333", lw=1.3,
            mutation_scale=8,
            clip_on=False,
        )
    )
    fig.text(x_arrow, y_elem + 0.047, "N",
             ha="center", va="bottom", fontsize=7,
             color="#333333", fontweight="bold",
             transform=fig.transFigure, clip_on=False)

    # Scale bar
    fig.lines.append(
        mlines.Line2D([xs0, xs1], [y_elem, y_elem],
                      transform=fig.transFigure,
                      color="#333333", lw=2,
                      solid_capstyle="butt", clip_on=False)
    )
    fig.lines.append(
        mlines.Line2D([xs0, xs0], [y_elem - 0.012, y_elem + 0.012],
                      transform=fig.transFigure,
                      color="#333333", lw=1.5, clip_on=False)
    )
    fig.lines.append(
        mlines.Line2D([xs1, xs1], [y_elem - 0.012, y_elem + 0.012],
                      transform=fig.transFigure,
                      color="#333333", lw=1.5, clip_on=False)
    )
    fig.text((xs0 + xs1) / 2, y_elem + 0.018, sb_label,
             ha="center", va="bottom", fontsize=6,
             color="#333333", transform=fig.transFigure, clip_on=False)

    save_fig(fig, f"02_{site['name']}_change{suffix}.png")


print("\nRendering 02_falsecolor_*.png  (tiles cached after first run) …")

for site in sites:
    name     = site["name"]
    # Prefer pinned coords (exact user-specified location), then found centroid, then seed
    clat     = site.get("pinned_lat", site.get("found_lat", site["lat"]))
    clon     = site.get("pinned_lon", site.get("found_lon", site["lon"]))
    yrs_list, _ = get_embedding_array(site)
    dist     = site.get("distance_from_2017", {})
    if not yrs_list:
        continue

    n_years = len(yrs_list)
    print(f"  {name}  ({n_years} years)  center=({clat:.4f}, {clon:.4f})")

    # Encode center coords in cache filenames so a changed found_lat/found_lon
    # automatically busts the cache (no stale tiles from a previous seed point).
    coord_tag = f"{clat:.4f}_{clon:.4f}".replace("-", "n")

    # One consistent projected region for all three tile types — avoids
    # scale mismatches between getThumbURL and sampleRectangle.
    site_region = _proj_region(clat, clon)

    # Pre-compute UMAP embedding tiles — disabled; uncomment the block below to re-enable.
    # print(f"    UMAP…", end="", flush=True)
    # umap_tiles = _fetch_umap_tiles(site, yrs_list, clat, clon, coord_tag, region=site_region)
    # print(" done")

    # Compute top-3 bands (highest |2024 - 2017|) for the false-color row.
    emb_2017 = np.array(site["embeddings"].get("2017", [0.0] * 64))
    emb_2024 = np.array(site["embeddings"].get("2024", emb_2017))
    band_diffs = np.abs(emb_2024 - emb_2017)
    top3_idx   = sorted(np.argsort(band_diffs)[-3:].tolist())
    top3_names = [f"A{i:02d}" for i in top3_idx]
    top3_label = f"Satellite Embedding\nFalse Color\nTop 3 band change\n({', '.join(top3_names)})\nAlphaEarth"

    # rows[0] = true color, rows[1] = top-3-band RGB, rows[2] = B&W distance
    # (UMAP row was rows[1] — re-add rows[1].append(umap_tiles.get(yr)) and
    #  change rows back to [[], [], [], []] to restore it)
    rows: list[list] = [[], [], []]
    for yr in yrs_list:
        yr_i = int(yr)
        print(f"    {yr_i}", end="", flush=True)

        # Row 0 — Sentinel-2 true color (B4=R, B3=G, B2=B)
        # Cache key includes cloud filter so stale black tiles are bypassed.
        print(" [tc]", end="", flush=True)
        rows[0].append(_fetch_tile(
            _s2_tile_url(clat, clon, yr_i, ["B4", "B3", "B2"], 0, 3000, region=site_region),
            TILE_DIR / f"tc_{name}_{coord_tag}_cp{CLOUD_PROB_MAX}fb_{yr_i}.png",
        ))

        # Row 1 — UMAP 3D embedding false color — disabled
        # print(" [umap]", end="", flush=True)
        # rows[1].append(umap_tiles.get(yr))   # re-enable with rows = [[], [], [], []]

        # Row 1 — top-3-band RGB false color (bands ranked by 2017→2024 change)
        print(" [top3]", end="", flush=True)
        rows[1].append(_fetch_tile(
            _top3band_tile_url(clat, clon, yr_i, top3_idx, region=site_region),
            TILE_DIR / f"top3_{name}_{coord_tag}_{yr_i}.png",
        ))

        # Row 2 — per-pixel cosine distance map from 2017 (spatial, 10 m/px)
        print(" [dist]", end="", flush=True)
        rows[2].append(_fetch_tile(
            _cosine_dist_url(clat, clon, yr_i, region=site_region),
            TILE_DIR / f"dist_{name}_{coord_tag}_{yr_i}.png",
        ))
        print(" ok")

    # Full time-series strip (all available years)
    _render_strip(site, clat, yrs_list, rows, yrs_list, dist, "", top3_label)

    # Sparse strip: 2018, 2020, 2022, 2024
    sparse = [yr for yr in SPARSE_YEARS if yr in yrs_list]
    if sparse:
        _render_strip(site, clat, sparse, rows, yrs_list, dist, "_sparse", top3_label)


# ─────────────────────────────────────────────────────────────────────────────
# 03 — Per-site 2-D PCA embedding trajectory
# ─────────────────────────────────────────────────────────────────────────────
print("\nRendering 03_trajectory_*.png …")

for site, color in zip(sites, PALETTE):
    yrs_list, X = get_embedding_array(site)
    if len(X) < 3:
        print(f"  ⚠ {site['name']}: fewer than 3 years, skipping trajectory.")
        continue

    pca    = PCA(n_components=2, random_state=42)
    coords = pca.fit_transform(X)   # (T, 2)

    fig, ax = plt.subplots(figsize=(6, 6), dpi=130)
    fig.patch.set_facecolor("white")

    # Arrows connecting successive years
    for i in range(len(coords) - 1):
        ax.annotate(
            "",
            xy=(coords[i + 1, 0], coords[i + 1, 1]),
            xytext=(coords[i, 0], coords[i, 1]),
            arrowprops=dict(
                arrowstyle="-|>", color=color, lw=1.6,
                mutation_scale=14, alpha=0.8,
            ),
        )

    # Points coloured by time (light → dark using the site colour at varying alpha)
    n = len(yrs_list)
    for i, yr in enumerate(yrs_list):
        alpha = 0.4 + 0.6 * (i / max(n - 1, 1))
        ax.scatter(coords[i, 0], coords[i, 1],
                   color=color, alpha=alpha, s=55, zorder=5,
                   edgecolors="white", lw=0.7)
        ax.annotate(yr, (coords[i, 0], coords[i, 1]),
                    xytext=(5, 5), textcoords="offset points",
                    fontsize=8.5, color="#333333")

    ev = pca.explained_variance_ratio_
    ax.set_xlabel(f"PC1  ({ev[0]:.1%} variance)", labelpad=6)
    ax.set_ylabel(f"PC2  ({ev[1]:.1%} variance)", labelpad=6)
    ax.set_title(site["label"], fontsize=11, loc="left", pad=10, color="#111111")
    ax.tick_params(length=3)
    plt.tight_layout()
    save_fig(fig, f"03_trajectory_{site['name']}.png")


# ─────────────────────────────────────────────────────────────────────────────
# 04 — Combined trajectory (shared PCA across all sites)
# ─────────────────────────────────────────────────────────────────────────────
print("Rendering 04_trajectory_combined.png …")

# Build one big array so UMAP is fit on all sites simultaneously (excluding
# chart-excluded sites like Gaza so the trajectory space isn't dominated by outliers)
segments: list[dict] = []
for site, color in zip(sites, PALETTE):
    if site["name"] in CHART_EXCLUDE:
        continue
    yrs_list, X = get_embedding_array(site)
    if len(X) < 2:
        continue
    segments.append({"site": site, "color": color, "yrs": yrs_list, "X": X})

if segments:
    import umap as umap_lib
    all_X      = np.vstack([s["X"] for s in segments])
    reducer_g  = umap_lib.UMAP(n_components=2, n_neighbors=15, min_dist=0.1,
                               random_state=42, verbose=False)
    all_coords = reducer_g.fit_transform(all_X)

    # Split back per site
    idx = 0
    for seg in segments:
        n = len(seg["yrs"])
        seg["coords"] = all_coords[idx : idx + n]
        idx += n

    fig, ax = plt.subplots(figsize=(9, 8), dpi=130)
    fig.patch.set_facecolor("white")

    for seg in segments:
        color  = seg["color"]
        coords = seg["coords"]
        yrs    = seg["yrs"]
        nt     = len(yrs)

        for i in range(nt - 1):
            ax.annotate(
                "",
                xy=(coords[i + 1, 0], coords[i + 1, 1]),
                xytext=(coords[i, 0], coords[i, 1]),
                arrowprops=dict(
                    arrowstyle="-|>", color=color, lw=1.4,
                    mutation_scale=7, alpha=0.7,
                    shrinkA=0, shrinkB=5,
                ),
            )

        for i, yr in enumerate(yrs):
            ax.scatter(coords[i, 0], coords[i, 1],
                       color=color, s=45, zorder=5,
                       edgecolors="white", lw=0.5)

        # Label only first and last year for clarity
        for i in [0, -1]:
            ax.annotate(yrs[i], (coords[i, 0], coords[i, 1]),
                        xytext=(4, 4), textcoords="offset points",
                        fontsize=7.5, color=color, alpha=0.95)

    # Legend patches
    handles = [
        mpatches.Patch(color=seg["color"], label=seg["site"]["label"])
        for seg in segments
    ]
    ax.legend(handles=handles, fontsize=8, frameon=False, loc="upper right")
    ax.set_xlabel("UMAP dimension 1", labelpad=6)
    ax.set_ylabel("UMAP dimension 2", labelpad=6)
    ax.set_title("Projected Satellite Embedding Distance Trajectories (2017-2025)",
                 fontsize=12, loc="left", pad=10, color="#111111")
    ax.tick_params(length=3)
    plt.tight_layout()
    save_fig(fig, "04_trajectory_combined.png")

    # ── 04 sparse — same UMAP projection, 2018/2020/2022/2024 points only ─
    _SPARSE_YRS = {"2018", "2020", "2022", "2024"}
    print("\nRendering 04_trajectory_sparse.png …")
    fig_s, ax_s = plt.subplots(figsize=(9, 8), dpi=130)
    fig_s.patch.set_facecolor("white")
    for seg in segments:
        c_s = seg["color"]; co_s = seg["coords"]; ys_s = seg["yrs"]
        sparse_idx = [i for i, yr in enumerate(ys_s) if yr in _SPARSE_YRS]
        for k in range(len(sparse_idx) - 1):
            i1, i2 = sparse_idx[k], sparse_idx[k + 1]
            ax_s.annotate(
                "",
                xy=(co_s[i2, 0], co_s[i2, 1]),
                xytext=(co_s[i1, 0], co_s[i1, 1]),
                arrowprops=dict(
                    arrowstyle="-|>", color=c_s, lw=1.4,
                    mutation_scale=7, alpha=0.7,
                    shrinkA=0, shrinkB=5,
                ),
            )
        for i, yr in enumerate(ys_s):
            if yr not in _SPARSE_YRS:
                continue
            ax_s.scatter(co_s[i, 0], co_s[i, 1],
                         color=c_s, s=55, zorder=5,
                         edgecolors="white", lw=0.5)
            ax_s.annotate(yr, (co_s[i, 0], co_s[i, 1]),
                          xytext=(4, 4), textcoords="offset points",
                          fontsize=7.5, color=c_s, alpha=0.95)
    handles_s = [
        mpatches.Patch(color=seg["color"], label=seg["site"]["label"])
        for seg in segments
    ]
    ax_s.legend(handles=handles_s, fontsize=8, frameon=False, loc="upper right")
    ax_s.set_xlabel("UMAP dimension 1", labelpad=6)
    ax_s.set_ylabel("UMAP dimension 2", labelpad=6)
    ax_s.set_title("Embedding positions 2018 · 2020 · 2022 · 2024  (shared UMAP space)",
                   fontsize=12, loc="left", pad=10, color="#111111")
    ax_s.tick_params(length=3)
    plt.tight_layout()
    save_fig(fig_s, "04_trajectory_sparse.png")
else:
    print("  ⚠ not enough data for combined trajectory.")


# ─────────────────────────────────────────────────────────────────────────────
# 06 — Embedding profile (64-dim vector per year)
# ─────────────────────────────────────────────────────────────────────────────
# print("\nRendering 06_embedding_profile_*.png …")  # disabled

_sites_06 = []  # change to `sites` to re-enable embedding profile plots
for site in _sites_06:
    yrs_list, X = get_embedding_array(site)
    if len(X) == 0:
        continue

    n  = len(yrs_list)
    x  = np.arange(64)
    cm = plt.cm.plasma

    fig, ax = plt.subplots(figsize=(10, 4), dpi=130)
    fig.patch.set_facecolor("white")

    for i, yr in enumerate(yrs_list):
        c = cm(i / max(n - 1, 1))
        ax.plot(x, X[i], color=c, lw=1.2, alpha=0.85)

    # Shade the range envelope (min–max across years)
    ax.fill_between(x, X.min(axis=0), X.max(axis=0),
                    color="#dddddd", alpha=0.4, zorder=0)

    ax.set_xlabel("Embedding dimension")
    ax.set_ylabel("Value")
    ax.set_title(f"Annual embedding vectors — {site['label']}",
                 fontsize=11, loc="left", pad=8)
    ax.set_xticks(range(0, 64, 4))
    ax.tick_params(length=3)

    # Colorbar legend: maps the plasma gradient to calendar years
    yr_ints = [int(y) for y in yrs_list]
    sm = matplotlib.cm.ScalarMappable(
        cmap=cm,
        norm=matplotlib.colors.Normalize(vmin=yr_ints[0], vmax=yr_ints[-1]),
    )
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, orientation="vertical", fraction=0.025, pad=0.02)
    cbar.set_label("Year", fontsize=9, color="#555555")
    cbar.set_ticks(yr_ints)
    cbar.ax.tick_params(labelsize=7.5, colors="#888888")

    plt.tight_layout()
    save_fig(fig, f"06_embedding_profile_{site['name']}.png")


print("\nAll done.")
