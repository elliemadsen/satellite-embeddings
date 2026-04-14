"""
Three side-by-side panels for a 4×4 AlphaEarth embedding grid:
  1. 4×4 cells — each shows all 64 embedding values in an 8×8 text layout
  2. 4×4 cells — each shows only the first 3 values (R, G, B) vertically centred
  3. 4×4 'pixels' scaled up — each coloured by the first 3 values as RGB
"""

import ee
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from pathlib import Path
import os

# ── Earth Engine ───────────────────────────────────────────────────────────────
try:
    ee.Initialize(project="gsapp-map")
except Exception:
    ee.Authenticate()
    ee.Initialize(project="gsapp-map")

YEAR      = 2024
# LAT       = 35.23639729402402
# LON       = 139.10999866436205
LAT = 37.01
LON = 120.99
LOC_NAME  = "Shandong, China"
SCALE     = 10     # metres per embedding pixel (AlphaEarth 10m)
N     = 4          # grid dimension

BANDS  = [f"A{i:02d}" for i in range(64)]
OUTPUT = "output.png"

# ── Fetch 100×100 overview grid then slice center 4×4 for the closeup ─────────
N2       = 100
PIXEL_SZ = 8   # screen pixels per embedding pixel

embed_img = (
    ee.ImageCollection("GOOGLE/SATELLITE_EMBEDDING/V1/ANNUAL")
    .filterDate(f"{YEAR}-01-01", f"{YEAR+1}-01-01")
    .mosaic()
    .select(BANDS)
    .reproject(crs="EPSG:3857", scale=SCALE)
)

print(f"Fetching {N2}×{N2} embedding pixels …")
region2 = ee.Geometry.Point(LON, LAT).buffer(SCALE * N2 / 2).bounds()
sample2 = embed_img.sampleRectangle(region=region2, defaultValue=0).getInfo()

grid2 = np.zeros((N2, N2, 64), dtype=np.float32)
for bi, bname in enumerate(BANDS):
    arr = np.array(sample2["properties"][bname], dtype=np.float32)
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)
    r2 = min(arr.shape[0], N2)
    c2 = min(arr.shape[1], N2)
    grid2[:r2, :c2, bi] = arr[:r2, :c2]

# Slice the center N×N — guaranteed identical pixels to the overview highlight
CR = grid2.shape[0] // 2
CC = grid2.shape[1] // 2
grid = grid2[CR - N//2 : CR + N//2, CC - N//2 : CC + N//2, :].copy()
print(f"  grid2 shape: {grid2.shape}  closeup slice: rows {CR-N//2}:{CR+N//2}, cols {CC-N//2}:{CC+N//2}")
print(f"  closeup range: [{grid.min():.4f}, {grid.max():.4f}]")

# Overview normalisation — shared by overview image AND the 4×4 closeup panels
_rgb2_raw = grid2[:, :, :3]
_v2min, _v2max = _rgb2_raw.min(), _rgb2_raw.max()
rgb2_norm = ((_rgb2_raw - _v2min) / max(_v2max - _v2min, 1e-6) * 255).clip(0, 255).astype(np.uint8)


# ── Font helper ────────────────────────────────────────────────────────────────
def load_font(size):
    # Prefer Roboto if installed, then fall back to common macOS/Linux system fonts
    paths = [
        # Roboto (if installed)
        str(Path.home() / "Library/Fonts/Roboto-Regular.ttf"),
        "/usr/share/fonts/truetype/roboto/Roboto-Regular.ttf",
        "/System/Library/Fonts/Supplemental/Roboto-Regular.ttf",
        # macOS system sans-serif fonts (always present)
        "/System/Library/Fonts/SFNS.ttf",               # SF Pro
        "/System/Library/Fonts/HelveticaNeue.ttc",
        "/System/Library/Fonts/Helvetica.ttc",
        "/System/Library/Fonts/Avenir Next.ttc",
        # Linux fallbacks
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
    ]
    for p in paths:
        if os.path.exists(p):
            return ImageFont.truetype(p, size)
    return ImageFont.load_default()


FONT_NUM   = load_font(11)   # for 64 numbers per cell
FONT_CAP   = load_font(14)   # for 3-value mapping panel
FONT_TITLE = load_font(20)   # per-panel title below each grid
FONT_SMALL = load_font(14)   # subcaption in overview
FONT_GEN   = load_font(30)   # general bottom caption

# ── Measure helpers ────────────────────────────────────────────────────────────
_td = ImageDraw.Draw(Image.new("RGB", (4, 4)))

def _measure(text, font):
    b = _td.textbbox((0, 0), text, font=font)
    return b[2] - b[0], b[3] - b[1]

ARROW_TXT = "→"  # unicode right arrow rendered by the font

# ── Cell / canvas dimensions ───────────────────────────────────────────────────
NW, NH = _measure("+0.12", FONT_NUM)
NW += 1; NH += 1

ARROW_W = _measure(ARROW_TXT, FONT_CAP)[0]

CELL_W    = max(NW * 8 + 8, NH * 8 + 8)   # square cell
CELL_H    = CELL_W
CELL_GAP  = 6
CELL_PAD  = 6

STEP_X = (CELL_W - 8) / 8
STEP_Y = (CELL_H - 8) / 8

_, _title_h  = _measure("Ag", FONT_TITLE)
PANEL_TITLE_H = _title_h + 10  # space below each full panel grid for its title

P1_W = N * CELL_W + (N - 1) * CELL_GAP
P1_H = N * CELL_H + (N - 1) * CELL_GAP

P2_W = P1_W
P2_H = P1_H

PS   = CELL_W
P3_W = N * PS + (N - 1) * CELL_GAP
P3_H = N * PS + (N - 1) * CELL_GAP

GUTTER = 50
MARGIN = 36

_, _gen_h  = _measure("x", FONT_GEN)
GENERAL_CAP_H = _gen_h + 24   # space at the very bottom of canvas

BG  = (255, 255, 255)
FG  = (30, 30, 30)
FG2 = (120, 120, 120)

CANVAS_W = P1_W + P2_W + P3_W + 2 * GUTTER + 2 * MARGIN
CANVAS_H = max(P1_H, P2_H, P3_H) + PANEL_TITLE_H + 2 * MARGIN + GENERAL_CAP_H

canvas = Image.new("RGB", (CANVAS_W, CANVAS_H), BG)
draw   = ImageDraw.Draw(canvas)

# ── RGB normalisation — use overview min/max so closeup colours match overview ─
_rgb_raw     = grid[:, :, :3]
rgb_norm_all = ((_rgb_raw - _v2min) / max(_v2max - _v2min, 1e-6) * 255).clip(0, 255)
rgb_norm     = rgb_norm_all.astype(np.uint8)


# ── Panel 1: 4×4 cells, all 64 values in 8×8 text layout ─────────────────────
X1 = MARGIN
Y1 = MARGIN

for row in range(N):
    for col in range(N):
        cx = X1 + col * (CELL_W + CELL_GAP)
        cy = Y1 + row * (CELL_H + CELL_GAP)
        vals = grid[row, col]
        for i, v in enumerate(vals):
            r8 = i // 8
            c8 = i % 8
            draw.text(
                (cx + 4 + round(c8 * STEP_X), cy + 4 + round(r8 * STEP_Y)),
                f"{v:+.2f}",
                font=FONT_NUM, fill=FG,
            )

# Panel 1 title
p1_title = "64D embedding:  A00–A63"
p1_tw, _ = _measure(p1_title, FONT_TITLE)
draw.text((X1 + (P1_W - p1_tw) // 2, Y1 + P1_H + 6), p1_title, font=FONT_TITLE, fill=FG)


# ── Panel 2: 4×4 cells, float → RGB mapping ───────────────────────────────────
X2 = X1 + P1_W + GUTTER
Y2 = MARGIN

CH_LABELS = ["R", "G", "B"]
_, LH = _measure("+0.0000", FONT_CAP)
LH += 5

_label_w  = max(_measure(f"{c}:", FONT_CAP)[0] for c in CH_LABELS)
_num_w    = max(_measure(str(v), FONT_CAP)[0] for v in range(256))
FIXED_GAP = 4

for row in range(N):
    for col in range(N):
        cx = X2 + col * (CELL_W + CELL_GAP)
        cy = Y2 + row * (CELL_H + CELL_GAP)
        block_h = 3 * LH
        top = cy + (CELL_H - block_h) // 2
        for k in range(3):
            v       = grid[row, col, k]
            rv      = int(round(rgb_norm_all[row, col, k]))
            dec_txt = f"{v:+.4f}"
            dec_w   = _measure(dec_txt, FONT_CAP)[0]
            ty      = top + k * LH
            content_w = (_label_w + FIXED_GAP + dec_w + FIXED_GAP +
                         ARROW_W  + FIXED_GAP + _num_w)
            x0 = cx + (CELL_W - content_w) // 2
            x1 = x0 + _label_w + FIXED_GAP
            x2 = x1 + dec_w    + FIXED_GAP
            x3 = x2 + ARROW_W  + FIXED_GAP
            draw.text((round(x0), ty), f"{CH_LABELS[k]}:", font=FONT_CAP, fill=FG)
            draw.text((round(x1), ty), dec_txt,            font=FONT_CAP, fill=FG)
            draw.text((round(x2), ty), ARROW_TXT,          font=FONT_CAP, fill=FG)
            draw.text((round(x3), ty), str(rv),            font=FONT_CAP, fill=FG)

# Panel 2 title
p2_title = "3D embedding:  A00–A02  →  RGB"
p2_tw, _ = _measure(p2_title, FONT_TITLE)
draw.text((X2 + (P2_W - p2_tw) // 2, Y2 + P2_H + 6), p2_title, font=FONT_TITLE, fill=FG)


# ── Panel 3: 4×4 pixels coloured by first 3 embedding values ─────────────────
X3 = X2 + P2_W + GUTTER
Y3 = MARGIN

for row in range(N):
    for col in range(N):
        r, g, b = (int(rgb_norm[row, col, k]) for k in range(3))
        px = X3 + col * (PS + CELL_GAP)
        py = Y3 + row * (PS + CELL_GAP)
        draw.rectangle([px, py, px + PS - 1, py + PS - 1], fill=(r, g, b))

# Panel 3 title
p3_title = "RGB Pixels"
p3_tw, _ = _measure(p3_title, FONT_TITLE)
draw.text((X3 + (P3_W - p3_tw) // 2, Y3 + P3_H + 6), p3_title, font=FONT_TITLE, fill=FG)


# ── General caption ────────────────────────────────────────────────────────────
gen_txt = (f"{LOC_NAME}    ·    "
           f"{LAT:.5f}\u00b0N, {LON:.5f}\u00b0E    ·    "
           f"{SCALE}m/pixel    ·    "
           f"AlphaEarth Foundations {YEAR}    ·    "
           f"min-max stretch over A00\u2013A02")
gw, _ = _measure(gen_txt, FONT_GEN)
draw.text(
    ((CANVAS_W - gw) // 2, CANVAS_H - GENERAL_CAP_H + 8),
    gen_txt, font=FONT_GEN, fill=FG2,
)


canvas.save(OUTPUT, dpi=(300, 300))
print(f"✓ Saved → {OUTPUT}  ({CANVAS_W}×{CANVAS_H} px)")


# ── Second output: 100×100 pixel overview with 4×4 highlight ──────────────────
OUTPUT2  = "output_overview.png"
# grid2, PIXEL_SZ, rgb2_norm already computed above

# Caption lines: first line is title, rest are subcaption
cap_title = LOC_NAME
cap_sub_lines = [
    f"{LAT:.5f}\u00b0N, {LON:.5f}\u00b0E",
    f"{SCALE} m / pixel",
    f"AlphaEarth Foundations {YEAR}",
]
_, _cl_h_title = _measure("Ag", FONT_TITLE)
_, _cl_h_sub   = _measure("Ag", FONT_SMALL)
CAP_PAD   = 14
GAP_AFTER_TITLE = 10
cap_total_h = (CAP_PAD + _cl_h_title + GAP_AFTER_TITLE
               + len(cap_sub_lines) * (_cl_h_sub + 4) + CAP_PAD)

ov_w = N2 * PIXEL_SZ
ov_h = N2 * PIXEL_SZ + cap_total_h
ov_canvas = Image.new("RGB", (ov_w, ov_h), BG)
ov_draw   = ImageDraw.Draw(ov_canvas)

# Draw pixels
for row in range(N2):
    for col in range(N2):
        r2, g2, b2 = int(rgb2_norm[row, col, 0]), int(rgb2_norm[row, col, 1]), int(rgb2_norm[row, col, 2])
        px = col * PIXEL_SZ
        py = row * PIXEL_SZ
        ov_draw.rectangle([px, py, px + PIXEL_SZ - 1, py + PIXEL_SZ - 1], fill=(r2, g2, b2))

# White outline around central 4×4 closeup region
c0r = CR - N // 2
c0c = CC - N // 2
ox0 = c0c * PIXEL_SZ
oy0 = c0r * PIXEL_SZ
ox1 = (c0c + N) * PIXEL_SZ - 1
oy1 = (c0r + N) * PIXEL_SZ - 1
ov_draw.rectangle([ox0, oy0, ox1, oy1], outline=(255, 255, 255), width=1)

# Caption: site name in title font, subcaption lines smaller + grey
cy_text = N2 * PIXEL_SZ + CAP_PAD
tw, _ = _measure(cap_title, FONT_TITLE)
ov_draw.text(((ov_w - tw) // 2, cy_text), cap_title, font=FONT_TITLE, fill=FG)
cy_sub = cy_text + _cl_h_title + GAP_AFTER_TITLE
for i, line in enumerate(cap_sub_lines):
    tw, _ = _measure(line, FONT_SMALL)
    ov_draw.text(((ov_w - tw) // 2, cy_sub + i * (_cl_h_sub + 4)), line, font=FONT_SMALL, fill=FG2)

ov_canvas.save(OUTPUT2, dpi=(300, 300))
print(f"✓ Saved → {OUTPUT2}  ({ov_w}×{ov_h} px)")
