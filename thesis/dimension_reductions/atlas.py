"""atlas.py — outputs two 2×2 grid PNGs to atlas/

  projections.png  — 4 cartographic projections (Mercator, Robinson, Equal Earth, Mollweide)
  embeddings.png   — 4 embedding reductions as density maps (PCA, t-SNE, UMAP, LLE)

Both images share identical layout: CELL×CELL square cells, same label strip,
same font size, same gap, same canvas dimensions.
"""

from __future__ import annotations
import os, json, base64, tempfile
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from PIL import Image, ImageDraw, ImageFont
from scipy.ndimage import binary_dilation

# ── Paths ──────────────────────────────────────────────────────────────────────
SCRIPT_DIR = Path(__file__).parent.resolve()
GEOJSON    = (SCRIPT_DIR / "../../dimension-reduction/data/20000_sampled_classified_embeddings.geojson").resolve()
OUTPUT_DIR = SCRIPT_DIR / "atlas"
OUTPUT_DIR.mkdir(exist_ok=True)

# ── Layout ─────────────────────────────────────────────────────────────────────
CELL    = 720   # square cell side (px) — used for both grids
LABEL_H = 52    # label strip height below each row of maps (px)
GAP     = 16    # gap between cells (px)
COLS    = 2
ROWS    = 2
BG      = "white"
FG      = "black"

CANVAS_W = COLS * CELL + (COLS - 1) * GAP
CANVAS_H = ROWS * (CELL + LABEL_H) + (ROWS - 1) * GAP

# ── Font ───────────────────────────────────────────────────────────────────────
FONT_SIZE = 21
_ROBOTO_CANDIDATES = [
    os.path.expanduser("~/Library/Fonts/Roboto-Regular.ttf"),
    os.path.expanduser("~/Library/Fonts/Roboto.ttf"),
    "/usr/share/fonts/truetype/roboto/Roboto-Regular.ttf",
]
FONT = None
for _p in _ROBOTO_CANDIDATES:
    if os.path.exists(_p):
        try:
            FONT = ImageFont.truetype(_p, FONT_SIZE)
            break
        except OSError:
            pass
if FONT is None:
    print("  ⚠ Roboto not found, using default font")
    FONT = ImageFont.load_default()

# ── Canvas helpers ─────────────────────────────────────────────────────────────
def cell_pos(i: int, cell_h: int = CELL) -> tuple[int, int]:
    col = i % COLS
    row = i // COLS
    return col * (CELL + GAP), row * (cell_h + LABEL_H + GAP)


def new_canvas(w: int = CANVAS_W, h: int = CANVAS_H) -> tuple[Image.Image, ImageDraw.ImageDraw]:
    canvas = Image.new("RGB", (w, h), BG)
    return canvas, ImageDraw.Draw(canvas)


def paste_cell(
    canvas: Image.Image,
    draw: ImageDraw.ImageDraw,
    img: Image.Image,
    label: str,
    i: int,
    cell_h: int = CELL,
) -> None:
    """Paste img (already sized to CELL × cell_h) onto canvas at grid position i,
    draw label centred in the strip below."""
    img = img.convert("RGB")
    x, y = cell_pos(i, cell_h)
    canvas.paste(img, (x, y))

    bb = draw.textbbox((0, 0), label, font=FONT)
    tw, th = bb[2] - bb[0], bb[3] - bb[1]
    tx = x + CELL // 2 - tw // 2
    ty = y + cell_h + (LABEL_H - th) // 2
    draw.text((tx, ty), label, fill=FG, font=FONT)


# ══════════════════════════════════════════════════════════════════════════════
# 1. projections.png
# ══════════════════════════════════════════════════════════════════════════════
PROJECTIONS = [
    (ccrs.Mercator(min_latitude=-80, max_latitude=85), "Mercator"),
    (ccrs.Robinson(),                                   "Robinson"),
    (ccrs.EqualEarth(),                                 "Equal Earth"),
    (ccrs.Mollweide(),                                  "Mollweide"),
]


def proj_to_pil(proj) -> Image.Image:
    """Render a cartopy projection scaled to CELL wide, natural height — no cropping."""
    fig = plt.figure(figsize=(12, 12), dpi=150)
    ax  = fig.add_subplot(1, 1, 1, projection=proj)
    ax.set_global()
    ax.set_facecolor(BG)
    fig.patch.set_facecolor(BG)
    ax.coastlines(color=FG, linewidth=0.6)
    ax.add_feature(cfeature.LAND,  facecolor=FG, edgecolor="none")
    ax.add_feature(cfeature.OCEAN, facecolor=BG, edgecolor="none")
    ax.set_axis_off()
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
        fig.savefig(tmp.name, bbox_inches="tight", pad_inches=0.0, facecolor=BG)
        tmp_path = tmp.name
    plt.close(fig)
    img = Image.open(tmp_path).copy()
    os.unlink(tmp_path)
    scale = CELL / img.width
    new_h = max(1, round(img.height * scale))
    return img.resize((CELL, new_h), Image.LANCZOS)


ROW_GAP = 80   # extra space between the two rows of maps

print("Rendering projections...")
proj_imgs = []
for proj, name in PROJECTIONS:
    print(f"  {name}...")
    proj_imgs.append((proj_to_pil(proj), name))

# Per-row max content height (shorter maps get white padding within the row)
row_content_h = [
    max(proj_imgs[r * COLS + c][0].height
        for c in range(COLS) if r * COLS + c < len(proj_imgs))
    for r in range(ROWS)
]

proj_canvas_w = COLS * CELL + (COLS - 1) * GAP
natural_h     = sum(row_content_h) + ROWS * LABEL_H + (ROWS - 1) * ROW_GAP
proj_canvas_h = max(proj_canvas_w, natural_h)   # square overall canvas
v_offset      = (proj_canvas_h - natural_h) // 2  # top margin to centre content

proj_canvas = Image.new("RGB", (proj_canvas_w, proj_canvas_h), BG)
proj_draw   = ImageDraw.Draw(proj_canvas)

y0 = v_offset
for r in range(ROWS):
    ch = row_content_h[r]
    for c in range(COLS):
        img, name = proj_imgs[r * COLS + c]
        x = c * (CELL + GAP)
        img_y = y0 + (ch - img.height) // 2
        proj_canvas.paste(img.convert("RGB"), (x, img_y))
        bb = proj_draw.textbbox((0, 0), name, font=FONT)
        tw, th = bb[2] - bb[0], bb[3] - bb[1]
        proj_draw.text(
            (x + CELL // 2 - tw // 2, y0 + ch + (LABEL_H - th) // 2),
            name, fill=FG, font=FONT,
        )
    y0 += ch + LABEL_H + ROW_GAP

proj_out = OUTPUT_DIR / "projections.png"
proj_canvas.save(proj_out)
print(f"  ✓ {proj_out}")


# ══════════════════════════════════════════════════════════════════════════════
# 2. embeddings.png
# ══════════════════════════════════════════════════════════════════════════════
RENDER_SCALE    = 4   # Render at this multiple of CELL before downsampling
DILATION_RADIUS = 8   # Dilation radius in high-res pixels (≈ 2px at output — sub-pixel fill)


def coords_to_density(coords: np.ndarray) -> Image.Image:
    """Render points at 4× resolution, dilate to fill intra-cluster gaps
    (radius is tiny at output scale), downsample with LANCZOS for smooth
    anti-aliased edges, then hard-threshold back to binary — no blur."""
    res_hi = CELL * RENDER_SCALE

    xmin, xmax = coords[:, 0].min(), coords[:, 0].max()
    ymin, ymax = coords[:, 1].min(), coords[:, 1].max()
    half = max(xmax - xmin, ymax - ymin) / 2
    cx   = (xmin + xmax) / 2
    cy   = (ymin + ymax) / 2
    pad  = half * 0.08
    x0, x1 = cx - half - pad, cx + half + pad
    y0, y1 = cy - half - pad, cy + half + pad

    xi = np.clip(((coords[:, 0] - x0) / (x1 - x0) * (res_hi - 1)).astype(int), 0, res_hi - 1)
    yi = np.clip(((coords[:, 1] - y0) / (y1 - y0) * (res_hi - 1)).astype(int), 0, res_hi - 1)
    yi  = (res_hi - 1) - yi   # flip: row 0 = top = high y

    grid = np.zeros((res_hi, res_hi), dtype=bool)
    grid[yi, xi] = True

    r = DILATION_RADIUS
    yy, xx = np.ogrid[-r:r + 1, -r:r + 1]
    struct = (xx ** 2 + yy ** 2) <= r ** 2
    filled = binary_dilation(grid, structure=struct)

    pixels = np.where(filled, np.uint8(0), np.uint8(255))
    img_hi  = Image.fromarray(pixels, mode="L")

    # Downsample: LANCZOS anti-aliases the 4×-scaled edges naturally
    img_out = img_hi.resize((CELL, CELL), Image.LANCZOS)

    # Hard threshold → clean binary (no soft grey values)
    arr = np.array(img_out)
    arr = np.where(arr < 128, np.uint8(0), np.uint8(255))
    return Image.fromarray(arr, mode="L").convert("RGB")


ALGORITHMS = [
    "Principal Component Analysis (PCA)",
    "t-Distributed Stochastic Neighbor Embedding (t-SNE)",
    "Uniform Manifold Approximation and Projection (UMAP)",
    "Locally Linear Embedding (LLE)",
]

print("\nLoading embeddings...")
with open(GEOJSON) as f:
    geo = json.load(f)
X = np.array(
    [json.loads(base64.b64decode(ft["properties"]["embedding"]).decode())
     for ft in geo["features"]],
    dtype=np.float32,
)
print(f"  {len(X)} points × {X.shape[1]} dims")

print("Running PCA...")
from sklearn.decomposition import PCA
pca_xy = PCA(n_components=2, random_state=42).fit_transform(X)

print("Running t-SNE...")
from sklearn.manifold import TSNE
tsne_xy = TSNE(n_components=2, perplexity=40, max_iter=1000, random_state=42).fit_transform(X)

print("Running UMAP...")
import umap as _umap
umap_xy = _umap.UMAP(n_components=2, n_neighbors=30, min_dist=0.1, random_state=42).fit_transform(X)

print("Running LLE...")
from sklearn.manifold import LocallyLinearEmbedding
lle_xy = LocallyLinearEmbedding(n_components=2, n_neighbors=15, random_state=42).fit_transform(X)

emb_canvas, emb_draw = new_canvas(CANVAS_W, CANVAS_H)
for i, (xy, label) in enumerate(zip([pca_xy, tsne_xy, umap_xy, lle_xy], ALGORITHMS)):
    paste_cell(emb_canvas, emb_draw, coords_to_density(xy), label, i, cell_h=CELL)

emb_out = OUTPUT_DIR / "embeddings.png"
emb_canvas.save(emb_out)
print(f"  ✓ {emb_out}")

print("\nDone.")
