"""
Environment toy embeddings — thesis illustration.
One PNG: environment_embeddings_3d.png
Graph (left) + vector table (right).
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

# ── Font ──────────────────────────────────────────────────────────────────────
_available = {f.name for f in fm.fontManager.ttflist}
FONT = "Roboto" if "Roboto" in _available else "DejaVu Sans"
plt.rcParams["font.family"] = FONT
_mono_candidates = ["Roboto Mono", "Courier New", "DejaVu Sans Mono", "Courier"]
MONO = next((n for n in _mono_candidates if n in _available), "monospace")

# ── Palette ───────────────────────────────────────────────────────────────────
BG     = "white"
FG     = "#111111"
FG2    = "#999999"
AXIS_C = "#aaaaaa"
GRID_C = "#e8e8e8"
GRID_N = "#cccccc"
EDGE_C = "#9bb4c8"

# ── Environment vectors: [elevation, humidity, tree_coverage] ─────────────────
locations = {
    "Atacama Desert":    np.array([0.92, 0.05, 0.01]),
    "Amazon Rainforest": np.array([0.35, 0.94, 0.98]),
    "Alpine Meadow":     np.array([0.88, 0.62, 0.42]),
    "Coastal Mangroves": np.array([0.08, 0.91, 0.76]),
}

# Dot colours — one per location
DOT_COLORS = {
    "Atacama Desert":    "#c8a46e",   # sandy tan
    "Amazon Rainforest": "#4caf6b",   # forest green
    "Alpine Meadow":     "#7badd4",   # alpine blue
    "Coastal Mangroves": "#5d9e8c",   # teal
}

OUT_DIR = os.path.dirname(os.path.abspath(__file__))
DPI     = 200
PAD     = 0.55
GAP     = 0.60


# ── Table helper ──────────────────────────────────────────────────────────────
def draw_table(fig, figW, figH, title, headers, rows, x_cols_in, y_center_in,
               row_h=0.26, fs=9, fs_mono=8.5):
    n     = len(rows)
    total = 0.44 + 0.14 + (n + 1) * row_h
    y_top = y_center_in + total / 2
    fig.text(x_cols_in[0] / figW, (y_top + 0.14) / figH,
             title, fontsize=fs + 2, color=FG, fontweight="bold",
             ha="left", va="bottom", fontfamily=FONT)
    # bottom-align all column headers: last line of every header sits on the same baseline
    max_hdr_lines = max(len(hdr.split("\n")) for hdr in headers)
    line_h_fig = (row_h * 0.55) / figH
    y_hdr_bot = y_top / figH - (max_hdr_lines - 1) * line_h_fig
    for ci, hdr in enumerate(headers):
        lines = list(reversed(hdr.split("\n")))
        for li, line in enumerate(lines):
            fig.text(x_cols_in[ci] / figW, y_hdr_bot + li * line_h_fig,
                     line, fontsize=fs, color=FG, fontweight="bold",
                     ha="left", va="bottom", fontfamily=FONT)
    for ri, row in enumerate(rows):
        ry = (y_top - (ri + 1.5) * row_h) / figH
        for ci, cell in enumerate(row):
            if ci == 0:
                fig.text(x_cols_in[ci] / figW, ry, cell,
                         fontsize=fs, color=FG2, ha="left", va="top",
                         fontfamily=FONT)
            else:
                fig.text(x_cols_in[ci] / figW, ry, cell,
                         fontsize=fs_mono, color="#444444", ha="left", va="top",
                         fontfamily=MONO)


# ── 3-D figure ────────────────────────────────────────────────────────────────
GW3     = 5.0
LIST_W3 = 5.2
TITLE_H = 0.45
FIG_W3  = PAD + GW3 + GAP + LIST_W3 + PAD
FIG_H3  = PAD + GW3 + TITLE_H + PAD

fig3 = plt.figure(figsize=(FIG_W3, FIG_H3), facecolor=BG, dpi=DPI)

def f3(x, y, w, h):
    return [x / FIG_W3, y / FIG_H3, w / FIG_W3, h / FIG_H3]

ax3 = fig3.add_axes(f3(PAD, PAD, GW3, GW3), projection="3d")
ax3.set_facecolor(BG)
ax3.patch.set_alpha(0)

ax3.set_xlim(0, 1)
ax3.set_ylim(0, 1)
ax3.set_zlim(0, 1)

ax3.set_xlabel("elevation",     fontsize=8, color=FG2, labelpad=-4)
ax3.set_ylabel("humidity",      fontsize=8, color=FG2, labelpad=-4)
ax3.set_zlabel("tree coverage", fontsize=8, color=FG2, labelpad=-4)

for pane in [ax3.xaxis.pane, ax3.yaxis.pane, ax3.zaxis.pane]:
    pane.fill = False
    pane.set_facecolor((1, 1, 1, 0))
    pane.set_edgecolor(GRID_C)

ax3.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
ax3.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
ax3.set_zticks([0, 0.25, 0.5, 0.75, 1.0])
ax3.tick_params(labelsize=6, colors=GRID_N, length=2, pad=-2)
ax3.xaxis.line.set_color(AXIS_C)
ax3.yaxis.line.set_color(AXIS_C)
ax3.zaxis.line.set_color(AXIS_C)
ax3.grid(True, color=GRID_C, linewidth=0.5)

for _ax in [ax3.xaxis, ax3.yaxis, ax3.zaxis]:
    _ax._axinfo["grid"]["color"] = "#f2f2f2"
    _ax._axinfo["grid"]["linewidth"] = 0.4

ax3.view_init(elev=28, azim=30)

# Draw lines from origin to each point
for name, vec in locations.items():
    ax3.plot([0, vec[0]], [0, vec[1]], [0, vec[2]],
             color=DOT_COLORS[name], lw=1.2, alpha=0.8, zorder=3)

# Label offsets (x, y, z) tuned per point
label_offsets = {
    "Atacama Desert":    ( 0.04, -0.06,  0.06),
    "Amazon Rainforest": (-0.04,  0.04,  0.06),
    "Alpine Meadow":     ( 0.04,  0.04, -0.08),
    "Coastal Mangroves": (-0.06,  0.04, -0.12),
}

for name, vec in locations.items():
    ax3.scatter([vec[0]], [vec[1]], [vec[2]],
                color=DOT_COLORS[name], s=36, zorder=5, edgecolors=FG, linewidths=0.5)
    ox, oy, oz = label_offsets.get(name, (0.04, 0.04, 0.04))
    ax3.text(vec[0] + ox, vec[1] + oy, vec[2] + oz,
             name, fontsize=7.5, color=FG, fontweight="bold", ha="center", zorder=6)

# Title
fig3.text((PAD + GW3 / 2) / FIG_W3, (PAD + GW3 + 0.08) / FIG_H3,
          "3-D Environmental Embedding Space",
          ha="center", va="bottom", fontsize=11, color=FG, fontweight="bold")

# Vector table
tbl_x3 = PAD + GW3 + GAP
mid_y3 = PAD + GW3 / 2
rows3 = [
    (name, f"{vec[0]:.2f}", f"{vec[1]:.2f}", f"{vec[2]:.2f}")
    for name, vec in locations.items()
]
draw_table(fig3, FIG_W3, FIG_H3,
           "Vectors",
           ["location", "elevation", "humidity", "tree\ncoverage"],
           rows3,
           [tbl_x3, tbl_x3 + 1.80, tbl_x3 + 2.72, tbl_x3 + 3.60],
           mid_y3)

out_path = os.path.join(OUT_DIR, "environment_embeddings_3d.png")
fig3.savefig(out_path, bbox_inches="tight", pad_inches=0.25, facecolor=BG, dpi=DPI)
print(f"Saved {out_path}")
plt.close(fig3)
