"""
Vector vs Raster — thesis illustration.
Left:  vector representations — point, line, polygon.
Right: raster representations — pixelated equivalents.
Style: white background, black/gray, Roboto, minimal text.
Square cells guaranteed by physically-square axes placement.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from matplotlib.path import Path
import matplotlib.font_manager as fm
from scipy.ndimage import distance_transform_edt

# ── Font ───────────────────────────────────────────────────────────────────────
_available = {f.name for f in fm.fontManager.ttflist}
FONT = 'Roboto' if 'Roboto' in _available else 'DejaVu Sans'
plt.rcParams['font.family'] = FONT

# ── Palette ────────────────────────────────────────────────────────────────────
BG       = 'white'
FG       = '#111111'   # lines, outlines
FG2      = '#888888'   # vertex dots, grid ghost
FILL_V   = '#d8d8d8'   # polygon fill (vector)
PIX_ON   = '#222222'   # active raster pixel
PIX_MID  = '#999999'   # polygon interior pixels
PIX_OFF  = 'white'     # empty pixel
GRID_R   = '#cccccc'   # raster pixel border
BORDER   = '#dddddd'   # cell border

# ── Geometry (normalised [0,1], y=0 bottom) ────────────────────────────────────
MARGIN = 0.10

pt_norm = np.array([0.50, 0.50])

line_norm = np.array([
    [0.12, 0.80],
    [0.35, 0.35],
    [0.63, 0.65],
    [0.88, 0.18],
])

poly_norm = np.array([
    [0.32, 0.83],
    [0.68, 0.80],
    [0.86, 0.52],
    [0.72, 0.20],
    [0.36, 0.18],
    [0.14, 0.48],
])

def to_ax(pts):
    lo, hi = MARGIN, 1 - MARGIN
    pts = np.atleast_2d(pts)
    return pts * (hi - lo) + lo

def to_grid(pts, N):
    pts = np.atleast_2d(pts)
    col = pts[:, 0] * (N - 1)
    row = (1 - pts[:, 1]) * (N - 1)
    return np.column_stack([col, row])

# ── Rasterisation helpers ──────────────────────────────────────────────────────
def bresenham_multi(pts, N):
    pixels = set()
    for i in range(len(pts) - 1):
        x0, y0 = int(round(pts[i][0])),   int(round(pts[i][1]))
        x1, y1 = int(round(pts[i+1][0])), int(round(pts[i+1][1]))
        ddx, ddy = abs(x1 - x0), abs(y1 - y0)
        sx = 1 if x0 < x1 else -1
        sy = 1 if y0 < y1 else -1
        err = ddx - ddy
        while True:
            if 0 <= x0 < N and 0 <= y0 < N:
                pixels.add((x0, y0))
            if x0 == x1 and y0 == y1:
                break
            e2 = 2 * err
            if e2 > -ddy: err -= ddy; x0 += sx
            if e2 < ddx:  err += ddx; y0 += sy
    return pixels

def fill_polygon_grid(vertices, N):
    closed = list(vertices) + [vertices[0]]
    path = Path(closed)
    cx, cy = np.meshgrid(np.arange(N) + 0.5, np.arange(N) + 0.5)
    pts = np.column_stack([cx.ravel(), cy.ravel()])
    return path.contains_points(pts).reshape(N, N)

def outline_polygon_grid(vertices, N):
    closed = list(vertices) + [vertices[0]]
    return bresenham_multi([(int(round(x)), int(round(y))) for x, y in closed], N)

# ── Figure layout: physically-square cells ────────────────────────────────────
N   = 16    # raster grid resolution
DPI = 160

CELL_IN    = 2.8    # each cell is CELL_IN × CELL_IN inches → pixels are square
COL_GAP_IN = 0.50
ROW_GAP_IN = 0.30
LBL_W_IN   = 0.55   # row-label strip on left
FTR_H_IN   = 0.50   # column header strip (below cells)
PAD        = 0.25   # outer padding (all sides)

FIG_W = PAD + LBL_W_IN + 2 * CELL_IN + COL_GAP_IN + PAD
FIG_H = PAD + FTR_H_IN + 3 * CELL_IN + 2 * ROW_GAP_IN + PAD

fig = plt.figure(figsize=(FIG_W, FIG_H), facecolor=BG, dpi=DPI)

def frac(x_in, y_in, w_in, h_in):
    """Convert physical inches to figure-fraction [l, b, w, h]."""
    return [x_in / FIG_W, y_in / FIG_H, w_in / FIG_W, h_in / FIG_H]

# Column left edges (inches from left of figure)
col_left = [PAD + LBL_W_IN,
            PAD + LBL_W_IN + CELL_IN + COL_GAP_IN]

# Row bottom edges (inches from bottom of figure), row 0 = point (top)
# Rows sit above the footer strip
row_bottom = [
    PAD + FTR_H_IN + (2 - i) * (CELL_IN + ROW_GAP_IN)
    for i in range(3)
]

ROWS  = ['point', 'line', 'polygon']
SIDES = ['vector', 'raster']

axes = {}
for ri, row in enumerate(ROWS):
    for ci, side in enumerate(SIDES):
        ax = fig.add_axes(frac(col_left[ci], row_bottom[ri], CELL_IN, CELL_IN))
        ax.set_facecolor(BG)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        for sp in ax.spines.values():
            sp.set_color(BORDER)
            sp.set_linewidth(0.5)
        ax.set_xticks([])
        ax.set_yticks([])
        axes[(row, side)] = ax

# ── Pixel-grid renderer ───────────────────────────────────────────────────────
def draw_pixel_grid(ax, intensity, N):
    """intensity: NxN float [0..1]. 0=white, 1=black."""
    d = 1.0 / N
    for r in range(N):
        for c in range(N):
            x0, y0 = c * d, 1.0 - (r + 1) * d
            v = float(intensity[r, c])
            if v > 0:
                g = 1.0 - v * 0.88   # v=1 → g=0.12 (near-black), v→0 → white
                fc = (g, g, g)
            else:
                fc = PIX_OFF
            ax.add_patch(plt.Rectangle((x0, y0), d, d,
                                       fc=fc, ec=GRID_R, lw=0.3, zorder=2))

# ─────────────────────────────────────────────────────────────────────────────
# ── POINT ─────────────────────────────────────────────────────────────────────

# vector
ax = axes[('point', 'vector')]
# faint grid
for v in np.linspace(0, 1, 7):
    ax.axvline(v, color=BORDER, lw=0.4, zorder=1)
    ax.axhline(v, color=BORDER, lw=0.4, zorder=1)
px, py = to_ax(pt_norm)[0]
ax.add_patch(Circle((px, py), 0.030, fc=FG, ec='none', zorder=4))

# raster
ax = axes[('point', 'raster')]
pt_g = to_grid(pt_norm, N)[0]
pc, pr = int(round(pt_g[0])), int(round(pt_g[1]))
ipt = np.zeros((N, N))
ipt[pr, pc] = 1.0
draw_pixel_grid(ax, ipt, N)

# ─────────────────────────────────────────────────────────────────────────────
# ── LINE ──────────────────────────────────────────────────────────────────────

# vector
ax = axes[('line', 'vector')]
for v in np.linspace(0, 1, 7):
    ax.axvline(v, color=BORDER, lw=0.4, zorder=1)
    ax.axhline(v, color=BORDER, lw=0.4, zorder=1)
ln = to_ax(line_norm)
ax.plot(ln[:, 0], ln[:, 1], color=FG, lw=1.6,
        solid_capstyle='round', solid_joinstyle='round', zorder=3)
ax.scatter(ln[:, 0], ln[:, 1], s=28, color=FG, zorder=5, linewidths=0)

# raster
ax = axes[('line', 'raster')]
lg = to_grid(line_norm, N)
lpx = [(int(round(c)), int(round(r))) for c, r in lg]
lit = bresenham_multi(lpx, N)
# fade out towards both endpoints
ep0 = np.array(lpx[0],  dtype=float)
ep1 = np.array(lpx[-1], dtype=float)
half = np.linalg.norm(ep1 - ep0) / 2.0 + 1e-6
iln = np.zeros((N, N))
for c, r in lit:
    p = np.array([c, r], dtype=float)
    d_near = min(np.linalg.norm(p - ep0), np.linalg.norm(p - ep1))
    t = np.clip(d_near / half, 0.0, 1.0)
    iln[r, c] = 0.20 + 0.80 * t   # 0.20 at endpoints, 1.0 at midpoint
draw_pixel_grid(ax, iln, N)

# ─────────────────────────────────────────────────────────────────────────────
# ── POLYGON ───────────────────────────────────────────────────────────────────

# vector
ax = axes[('polygon', 'vector')]
for v in np.linspace(0, 1, 7):
    ax.axvline(v, color=BORDER, lw=0.4, zorder=1)
    ax.axhline(v, color=BORDER, lw=0.4, zorder=1)
pg = to_ax(poly_norm)
ax.add_patch(plt.Polygon(pg, closed=True,
                         fc=FILL_V, ec=FG, lw=1.6, zorder=3))
ax.scatter(pg[:, 0], pg[:, 1], s=24, color=FG, zorder=5, linewidths=0)

# raster — distance-based gradient: dark center, lighter near edges
ax = axes[('polygon', 'raster')]
poly_g  = to_grid(poly_norm, N)
poly_px = [(c, r) for c, r in poly_g.tolist()]
fill_m  = fill_polygon_grid(poly_px, N)
dist    = distance_transform_edt(fill_m)
ipoly   = (dist / dist.max()) if dist.max() > 0 else fill_m.astype(float)
draw_pixel_grid(ax, ipoly, N)

# ─────────────────────────────────────────────────────────────────────────────
# ── Labels ────────────────────────────────────────────────────────────────────

# Column headers — centred over each column, placed in footer strip below cells
ftr_y = (PAD + FTR_H_IN * 0.5) / FIG_H
for ci, label in enumerate(['vector', 'raster']):
    cx = (col_left[ci] + CELL_IN / 2) / FIG_W
    fig.text(cx, ftr_y, label,
             color=FG2, fontsize=10,
             ha='center', va='center',
             fontfamily=FONT)

# Row labels — rotated, centred on each row
lbl_cx = (PAD + LBL_W_IN * 0.45) / FIG_W
for ri, label in enumerate(ROWS):
    cy = (row_bottom[ri] + CELL_IN / 2) / FIG_H
    fig.text(lbl_cx, cy, label,
             color=FG2, fontsize=9,
             ha='center', va='center',
             fontfamily=FONT, rotation=90)

# ─────────────────────────────────────────────────────────────────────────────
# ── Save ──────────────────────────────────────────────────────────────────────

OUT = "output.png"
fig.savefig(OUT, dpi=DPI, bbox_inches='tight', facecolor=BG)
print(f"Saved → {OUT}")
