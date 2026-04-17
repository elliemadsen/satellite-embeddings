"""
Combined illustration:
  • Embedding values arranged in a circle (outer ring).
  • Band names (A00…A63) on an inner ring.
  • 21 false-colour composites (A00-02, A03-05, …, A60-62) positioned
    around the outside, each next to its three bands on the circle.
  • Faint lines connecting each image to its three band positions.
Output: combined.png
"""

import math, os
import numpy as np
import ee
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from PIL import Image, ImageDraw

# ── Font ──────────────────────────────────────────────────────────────────────
_available = {f.name for f in fm.fontManager.ttflist}
FONT = "Roboto" if "Roboto" in _available else "DejaVu Sans"
plt.rcParams["font.family"]     = "sans-serif"
plt.rcParams["font.sans-serif"] = [FONT, "DejaVu Sans"]

# ── Earth Engine ──────────────────────────────────────────────────────────────
try:
    ee.Initialize(project="gsapp-map")
except Exception:
    ee.Authenticate()
    ee.Initialize(project="gsapp-map")

# ── Location ─────────────────────────────────────────────────────────────────
HERE      = os.path.dirname(os.path.abspath(__file__))
LAT       = -(41 + 13/60 + 27.38/3600)   # 41°13'27.38"S
LON       = -(71 + 38/60 + 29.36/3600)   # 71°38'29.36"W
place   = "Patagonia"
country = "Argentina"
YEAR      = 2024
N_BANDS   = 63   # A00-A62; drop A63 so all belong to a triplet

# ── Fetch all 64 bands from EE ────────────────────────────────────────────────
SCALE     = 10     # m/px
N_PX      = 100    # 100 × 10 m = 1 km across
ALL_BANDS = [f"A{i:02d}" for i in range(N_BANDS)]

embed_img = (
    ee.ImageCollection("GOOGLE/SATELLITE_EMBEDDING/V1/ANNUAL")
    .filterDate(f"{YEAR}-01-01", f"{YEAR+1}-01-01")
    .mosaic()
    .select(ALL_BANDS)
    .reproject(crs="EPSG:4326", scale=SCALE)
)

half_m    = SCALE * N_PX / 2
region_ee = ee.Geometry.Point(LON, LAT).buffer(half_m).bounds()

print(f"Fetching {N_PX}×{N_PX}×{N_BANDS} from EE …")
sample = embed_img.sampleRectangle(region=region_ee, defaultValue=0).getInfo()

grid = np.zeros((N_PX, N_PX, N_BANDS), dtype=np.float32)
for bi, bname in enumerate(ALL_BANDS):
    arr = np.array(sample["properties"][bname], dtype=np.float32)
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)
    r = min(arr.shape[0], N_PX)
    c = min(arr.shape[1], N_PX)
    grid[:r, :c, bi] = arr[:r, :c]

print("Fetch done.")

# Embedding values = centre pixel of the fetched grid
embedding = grid[N_PX // 2, N_PX // 2, :].tolist()

# ── Build 21 triplet composites (bands 0-62; band 63 unused) ──────────────────
N_TRIP   = 21
triplets = [(i * 3, i * 3 + 1, i * 3 + 2) for i in range(N_TRIP)]
CHIP_SZ  = 256

def make_circular_rgba(rgb_u8, size=CHIP_SZ):
    img  = Image.fromarray(rgb_u8).resize((size, size), Image.LANCZOS)
    mask = Image.new("L", (size, size), 0)
    ImageDraw.Draw(mask).ellipse((0, 0, size - 1, size - 1), fill=255)
    out          = np.zeros((size, size, 4), dtype=np.uint8)
    out[:, :, :3] = np.array(img)
    out[:, :, 3]  = np.array(mask)
    return out

composites = []
for r_i, g_i, b_i in triplets:
    chip = np.zeros((N_PX, N_PX, 3), dtype=np.float32)
    for ch, bi in enumerate([r_i, g_i, b_i]):
        ch_d = grid[:, :, bi]
        lo, hi = ch_d.min(), ch_d.max()
        chip[:, :, ch] = (ch_d - lo) / max(hi - lo, 1e-6)
    composites.append(make_circular_rgba((chip * 255).clip(0, 255).astype(np.uint8)))

# ── Geometry helpers ──────────────────────────────────────────────────────────
def band_angle_rad(i):
    """Clockwise from top; returns radians."""
    return math.radians(90 - i * (360 / N_BANDS))

def polar(r, a):
    return r * math.cos(a), r * math.sin(a)

CH_ANGLE = 0.009  # (kept for reference; per-image labels use ch_step)

def draw_curved_label_local(ax, text, cx, cy, r_local, a_center,
                            fontsize, color, zorder, ch_step=0.048):
    """Curve *text* around a small circle centred at (cx, cy).
    a_center: angle (radians) pointing outward – i.e. the arc midpoint.
    ch_step:  angular gap between successive characters (adjust to taste).
    """
    n      = len(text)
    bottom = math.sin(a_center) < 0   # flip chars on lower half so they stay upright
    for i, ch in enumerate(text):
        a_ch = a_center + (i - (n - 1) / 2.0) * ch_step
        x    = cx + r_local * math.cos(a_ch)
        y    = cy + r_local * math.sin(a_ch)
        rot  = math.degrees(a_ch) + (90 if bottom else -90)
        ax.text(x, y, ch, rotation=rot,
                ha='center', va='center', fontsize=fontsize, color=color,
                rotation_mode='anchor', zorder=zorder)

# ── Radii ────────────────────────────────────────────────────────────────────
R_VALUE = 5.1    # embedding value ring
R_LABEL = 5.45   # outer band-name ring (outside values)
R_IMG   = 7.5    # satellite image centres
IMG_H   = 0.90   # image half-size in data units

# ── Figure ────────────────────────────────────────────────────────────────────
FG   = "#111111"
GRAY = "#888888"
LGRY = "#cccccc"

fig, ax = plt.subplots(figsize=(16, 16), facecolor="white")
ax.set_facecolor("white")
ax.set_aspect("equal")
ax.set_xlim(-10.5, 10.5)
ax.set_ylim(-10.5, 10.5)
ax.axis("off")

# ── Embedding values ──────────────────────────────────────────────────────────
for i, val in enumerate(embedding[:N_BANDS]):
    a      = band_angle_rad(i)
    vx, vy = polar(R_VALUE, a)
    ax.text(vx, vy, f"{val:+.4f}",
            ha="center", va="center",
            fontsize=5.0, color=FG, fontweight="medium",
            zorder=5)

# ── Outer band labels (outside value ring, white bg so lines don't show through)
for i in range(N_BANDS):
    a      = band_angle_rad(i)
    lx, ly = polar(R_LABEL, a)
    ax.text(lx, ly, f"A{i:02d}",
            ha="center", va="center",
            fontsize=3.8, color="#c0c0c0",
            bbox=dict(boxstyle="square,pad=0.15", fc="white", ec="none"),
            zorder=6)

# ── Satellite composites + connecting lines ───────────────────────────────────
for ti, (r_i, g_i, b_i) in enumerate(triplets):
    a_img  = band_angle_rad(g_i)          # centred on middle band
    cx, cy = polar(R_IMG, a_img)

    # Faint lines: from just outside value text → image centre
    # (band labels at R_LABEL are drawn on top of the lines, zorder=6)
    for bi in (r_i, g_i, b_i):
        ba     = band_angle_rad(bi)
        sx, sy = polar(R_VALUE + 0.30, ba)
        ax.plot([sx, cx], [sy, cy],
                color="#e0e0e0", lw=0.45, solid_capstyle="round", zorder=1)

    # Circular false-colour image
    ax.imshow(composites[ti],
              extent=[cx - IMG_H, cx + IMG_H, cy - IMG_H, cy + IMG_H],
              origin="upper", interpolation="bilinear", zorder=10)

    # Thin circle border
    ax.add_patch(plt.Circle((cx, cy), IMG_H,
                             color=LGRY, fill=False, lw=0.6, zorder=11))

    # Band label curved around the outer edge of this small image
    draw_curved_label_local(ax, f"[A{r_i:02d}, A{g_i:02d}, A{b_i:02d}]",
                            cx, cy, IMG_H + 0.22, a_img,
                            fontsize=3.8, color=GRAY, zorder=12)

# ── Caption ───────────────────────────────────────────────────────────────────
ax.text(0, -9.8, f"[{LAT:.4f}°,  {LON:.4f}°]",
        ha="center", va="center", fontsize=9, color=GRAY, zorder=5)
place_parts = [p for p in [place, country] if p and p != "Unknown"]
if place_parts:
    ax.text(0, -10.2, ",  ".join(place_parts),
            ha="center", va="center", fontsize=6.5, color="#bbbbbb", zorder=5)
ax.text(0, -10.52, f"{SCALE} m/px  ·  {SCALE * N_PX / 1000:.1f} km",
        ha="center", va="center", fontsize=6.5, color="#bbbbbb", zorder=5)

# ── Save ──────────────────────────────────────────────────────────────────────
out_dir = os.path.join(HERE, "output")
os.makedirs(out_dir, exist_ok=True)
out_path = os.path.join(out_dir, f"circle_{int(round(LAT))}_{int(round(LON))}.png")
fig.savefig(out_path, dpi=200, bbox_inches="tight", facecolor="white")
plt.close(fig)
print(f"Saved → {out_path}")
