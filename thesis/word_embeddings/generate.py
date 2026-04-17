"""
Word embeddings — thesis illustration.
Two PNGs: word_embeddings_2d.png  /  word_embeddings_3d.png
Each:  graph (left)  +  vector list (right)
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
DOT_C  = "#111111"
EDGE_C = "#9bb4c8"   # muted blue edge lines
AXIS_C = "#aaaaaa"   # axis cross-hair
GRID_C = "#e8e8e8"   # light-grey grid
GRID_N = "#cccccc"   # faint grid number colour

# ── Word vectors — exact constant components so box edges are straight ────────
# gender: -0.90 = male,  +0.90 = female
# age:    -0.85 = child, +0.85 = adult
# royalty: -0.60 = commoner, +0.85 = royal

words_2d = {
    "man":   np.array([-0.90,  0.85]),
    "woman": np.array([ 0.90,  0.85]),
    "boy":   np.array([-0.90, -0.85]),
    "girl":  np.array([ 0.90, -0.85]),
}

words_3d = {
    "man":      np.array([-0.90,  0.85, -0.60]),
    "woman":    np.array([ 0.90,  0.85, -0.60]),
    "boy":      np.array([-0.90, -0.85, -0.60]),
    "girl":     np.array([ 0.90, -0.85, -0.60]),
    "king":     np.array([-0.90,  0.85,  0.90]),
    "queen":    np.array([ 0.90,  0.85,  0.90]),
    "prince":   np.array([-0.90, -0.85,  0.90]),
    "princess": np.array([ 0.90, -0.85,  0.90]),
}

# ── Box edges (parallelogram in 2-D / cube in 3-D) ───────────────────────────
edges_2d = [
    ("man", "woman"),   # top
    ("boy", "girl"),    # bottom
    ("man", "boy"),     # left
    ("woman", "girl"),  # right
]

edges_3d = [
    # royalty axis (commoner -> royal)
    ("man", "king"), ("woman", "queen"), ("boy", "prince"), ("girl", "princess"),
    # gender axis
    ("man", "woman"), ("king", "queen"), ("boy", "girl"), ("prince", "princess"),
    # age axis
    ("man", "boy"), ("woman", "girl"), ("king", "prince"), ("queen", "princess"),
]

OUT_DIR = os.path.dirname(os.path.abspath(__file__))
DPI     = 200
PAD     = 0.55
GAP     = 0.70


# ── Aligned table helper ─────────────────────────────────────────────────────
def draw_table(fig, figW, figH, title, headers, rows, x_cols_in, y_center_in,
               row_h=0.22, fs=9, fs_mono=8.5):
    """Vertically-centred table. Col 0 = word (Roboto, grey). Col 1+ = numbers (monospace)."""
    n     = len(rows)
    total = 0.44 + 0.14 + (n + 1) * row_h
    y_top = y_center_in + total / 2
    fig.text(x_cols_in[0] / figW, (y_top + 0.14) / figH,
             title, fontsize=fs + 2, color=FG, fontweight="bold",
             ha="left", va="bottom", fontfamily=FONT)
    for ci, hdr in enumerate(headers):
        fig.text(x_cols_in[ci] / figW, y_top / figH,
                 hdr, fontsize=fs, color=FG, fontweight="bold",
                 ha="left", va="top", fontfamily=FONT)
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


# ─────────────────────────────────────────────────────────────────────────────
# 2-D figure
# ─────────────────────────────────────────────────────────────────────────────
GW2     = 4.6
LIST_W2 = 4.0
TITLE_H = 0.45
FIG_W2  = PAD + GW2 + GAP + LIST_W2 + PAD
FIG_H2  = PAD + GW2 + TITLE_H + PAD

fig2 = plt.figure(figsize=(FIG_W2, FIG_H2), facecolor=BG, dpi=DPI)

def f2(x, y, w, h):
    return [x/FIG_W2, y/FIG_H2, w/FIG_W2, h/FIG_H2]

ax2 = fig2.add_axes(f2(PAD, PAD, GW2, GW2))
ax2.set_facecolor(BG)
ax2.set_xlim(-1.25, 1.25)
ax2.set_ylim(-1.25, 1.25)
ax2.set_aspect("equal")

# Light-grey grid + faint tick numbers
for v in np.arange(-1.0, 1.01, 0.5):
    ax2.axhline(v, color=GRID_C, lw=0.7, zorder=0)
    ax2.axvline(v, color=GRID_C, lw=0.7, zorder=0)
    lbl = f"{v:+.1f}" if v != 0 else "0"
    ax2.text(v, -1.25, lbl, fontsize=6, color=GRID_N,
             ha="center", va="top", clip_on=False)
    ax2.text(-1.25, v, lbl, fontsize=6, color=GRID_N,
             ha="right",  va="center", clip_on=False)

# Axis lines through origin
ax2.axhline(0, color=AXIS_C, lw=1.0, zorder=1)
ax2.axvline(0, color=AXIS_C, lw=1.0, zorder=1)

# Pole labels — italic, muted
ax2.text( 1.20,  0.05, "female", fontsize=7, color=AXIS_C, ha="right", va="bottom", style="italic")
ax2.text(-1.20,  0.05, "male",   fontsize=7, color=AXIS_C, ha="left",  va="bottom", style="italic")
ax2.text( 0.05,  1.20, "adult",  fontsize=7, color=AXIS_C, ha="left",  va="top",    style="italic")
ax2.text( 0.05, -1.20, "child",  fontsize=7, color=AXIS_C, ha="left",  va="bottom", style="italic")

# Axis titles — upright, black, distinct from pole labels
ax2.text(0,    -1.40, "gender", fontsize=8.5, color=FG, fontweight="bold", ha="center", va="top")
ax2.text(-1.40, 0,    "age",    fontsize=8.5, color=FG, fontweight="bold", ha="center", va="center", rotation=90)

# Box connecting lines
for w1, w2 in edges_2d:
    p1, p2 = words_2d[w1], words_2d[w2]
    ax2.plot([p1[0], p2[0]], [p1[1], p2[1]], color=EDGE_C, lw=1.1, zorder=2)

# Dots + word labels (no origin vectors)
lbl2 = {
    "man":   ( 0.12, -0.14),
    "woman": (-0.12, -0.14),
    "boy":   ( 0.12,  0.14),
    "girl":  (-0.12,  0.14),
}
for word, vec in words_2d.items():
    ax2.scatter(*vec, color=DOT_C, s=28, zorder=5)
    ox, oy = lbl2[word]
    ax2.text(vec[0]+ox, vec[1]+oy, word,
             fontsize=10, color=FG, fontweight="bold", ha="center", va="center", zorder=6)

for sp in ax2.spines.values():
    sp.set_visible(False)
ax2.set_xticks([])
ax2.set_yticks([])

# Title
fig2.text((PAD + GW2/2) / FIG_W2, (PAD + GW2 + 0.08) / FIG_H2,
          "2-D Word Embedding Space",
          ha="center", va="bottom", fontsize=11, color=FG, fontweight="bold")

# Vector table (right panel) — word | gender | age, centred on graph midpoint
tbl_x2 = PAD + GW2 + GAP
mid_y2 = PAD + GW2 / 2
rows2  = [(w, f"{v[0]:+.2f}", f"{v[1]:+.2f}") for w, v in words_2d.items()]
draw_table(fig2, FIG_W2, FIG_H2,
           "Vectors",
           ["word", "gender", "age"],
           rows2,
           [tbl_x2, tbl_x2 + 1.05, tbl_x2 + 2.00],
           mid_y2)

fig2.savefig(os.path.join(OUT_DIR, "word_embeddings_2d.png"),
             bbox_inches="tight", facecolor=BG, dpi=DPI)
print("Saved word_embeddings_2d.png")
plt.close(fig2)


# ─────────────────────────────────────────────────────────────────────────────
# 3-D figure
# ─────────────────────────────────────────────────────────────────────────────
GW3     = 5.0
LIST_W3 = 4.4
FIG_W3  = PAD + GW3 + GAP + LIST_W3 + PAD
FIG_H3  = PAD + GW3 + TITLE_H + PAD

fig3 = plt.figure(figsize=(FIG_W3, FIG_H3), facecolor=BG, dpi=DPI)

def f3(x, y, w, h):
    return [x/FIG_W3, y/FIG_H3, w/FIG_W3, h/FIG_H3]

ax3 = fig3.add_axes(f3(PAD, PAD, GW3, GW3), projection="3d")
ax3.set_facecolor(BG)
ax3.patch.set_alpha(0)

ax3.set_xlim(-1.2, 1.2)
ax3.set_ylim(-1.2, 1.2)
ax3.set_zlim(-1.2, 1.2)

ax3.set_xlabel("gender",  fontsize=8, color=FG2, labelpad=-4)
ax3.set_ylabel("age",     fontsize=8, color=FG2, labelpad=-4)
ax3.set_zlabel("royalty", fontsize=8, color=FG2, labelpad=-4)

for pane in [ax3.xaxis.pane, ax3.yaxis.pane, ax3.zaxis.pane]:
    pane.fill = False
    pane.set_facecolor((1, 1, 1, 0))
    pane.set_edgecolor(GRID_C)

ax3.set_xticks([-1, -0.5, 0, 0.5, 1])
ax3.set_yticks([-1, -0.5, 0, 0.5, 1])
ax3.set_zticks([-1, -0.5, 0, 0.5, 1])
ax3.tick_params(labelsize=6, colors=GRID_N, length=2, pad=-2)
ax3.xaxis.line.set_color(AXIS_C)
ax3.yaxis.line.set_color(AXIS_C)
ax3.zaxis.line.set_color(AXIS_C)
ax3.grid(True, color=GRID_C, linewidth=0.5)

# Make the interior 3D gridlines very faint (lighter than the pane edges)
_3d_grid = "#f2f2f2"
for _ax in [ax3.xaxis, ax3.yaxis, ax3.zaxis]:
    _ax._axinfo["grid"]["color"] = _3d_grid
    _ax._axinfo["grid"]["linewidth"] = 0.4

ax3.view_init(elev=20, azim=-50)

# Cube connecting lines
for w1, w2 in edges_3d:
    p1, p2 = words_3d[w1], words_3d[w2]
    ax3.plot([p1[0], p2[0]], [p1[1], p2[1]], [p1[2], p2[2]],
             color=EDGE_C, lw=0.9, zorder=2)

# Dots + word labels (no origin vectors)
lbl3 = {
    # commoners: push outward in x and below in z
    "man":      (-0.16,  0.00, -0.20),
    "woman":    ( 0.16,  0.00, -0.20),
    "boy":      (-0.16,  0.00, -0.20),
    "girl":     ( 0.16,  0.00, -0.20),
    # royals: push outward in x and above in z
    "king":     (-0.16,  0.00,  0.20),
    "queen":    ( 0.16,  0.00,  0.20),
    "prince":   (-0.16,  0.00,  0.20),
    "princess": ( 0.16,  0.00,  0.20),
}
for word, vec in words_3d.items():
    gx, gy, gz = vec
    ax3.scatter([gx], [gy], [gz], color=DOT_C, s=20, zorder=5)
    ox, oy, oz = lbl3.get(word, (0.05, 0.05, 0.05))
    ax3.text(gx+ox, gy+oy, gz+oz, word,
             fontsize=7.5, color=FG, fontweight="bold", ha="center", zorder=6)

# Title
fig3.text((PAD + GW3/2) / FIG_W3, (PAD + GW3 + 0.08) / FIG_H3,
          "3-D Word Embedding Space",
          ha="center", va="bottom", fontsize=11, color=FG, fontweight="bold")

# Vector table (right panel) — word | gender | age | royalty, centred on graph midpoint
tbl_x3 = PAD + GW3 + GAP
mid_y3 = PAD + GW3 / 2
rows3  = [(w, f"{v[0]:+.2f}", f"{v[1]:+.2f}", f"{v[2]:+.2f}")
          for w, v in words_3d.items()]
draw_table(fig3, FIG_W3, FIG_H3,
           "Vectors",
           ["word", "gender", "age", "royalty"],
           rows3,
           [tbl_x3, tbl_x3 + 1.05, tbl_x3 + 2.00, tbl_x3 + 2.95],
           mid_y3)

fig3.savefig(os.path.join(OUT_DIR, "word_embeddings_3d.png"),
             bbox_inches="tight", facecolor=BG, dpi=DPI)
print("Saved word_embeddings_3d.png")
plt.close(fig3)
