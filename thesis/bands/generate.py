"""
Bands illustration — thesis figure.

Layout: [LS stack] [LS table] gap [EM table] [EM stack]

- Two fully independent tables, each filling the full available height.
- LS table: 9 data rows, taller per row.
- EM table: 64 data rows (A00–A63, no dots), more compressed.
- No per-row gridlines — only top border + header rules.
- EM layer chips: light grey.
- All text: Roboto.
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon as MPoly
import matplotlib.font_manager as fm
import numpy as np
import os

OUT_DIR = os.path.dirname(os.path.abspath(__file__))

# ── Fonts ─────────────────────────────────────────────────────────────────────
_avail = {f.name for f in fm.fontManager.ttflist}
FONT   = "Roboto" if "Roboto" in _avail else "DejaVu Sans"
plt.rcParams["font.family"] = FONT

# ── Palette ───────────────────────────────────────────────────────────────────
BG   = "white"
FG   = "#111111"
FG2  = "#777777"
SEP  = "#dddddd"
CONN = "#cccccc"

# Landsat 9 layer colours (very muted, wavelength-inspired)
LS_FC = [
    "#b0b0d8",  # B1 coastal/aerosol
    "#a0b4d4",  # B2 blue
    "#96bda0",  # B3 green
    "#d4a8a8",  # B4 red
    "#c8a0a0",  # B5 NIR
    "#d0bc96",  # B6 SWIR1
    "#c0b090",  # B7 SWIR2
    "#c4c4c4",  # B8 panchromatic
    "#b8cdd6",  # B9 cirrus
]

# AlphaEarth 64-band: light grey
EM_FC = ["#dde0e4"] * 64

# ── Band data ─────────────────────────────────────────────────────────────────
LS_BANDS = [
    ("1 – Coastal/Aerosol", "0.43–0.45 µm", "30 m"),
    ("2 – Blue",            "0.45–0.51 µm", "30 m"),
    ("3 – Green",           "0.53–0.59 µm", "30 m"),
    ("4 – Red",             "0.64–0.67 µm", "30 m"),
    ("5 – Near-infrared",   "0.85–0.88 µm", "30 m"),
    ("6 – SWIR 1",          "1.57–1.65 µm", "30 m"),
    ("7 – SWIR 2",          "2.11–2.29 µm", "30 m"),
    ("8 – Panchromatic",    "0.50–0.68 µm", "15 m"),
    ("9 – Cirrus",          "1.36–1.38 µm", "30 m"),
]

# ── Figure size ───────────────────────────────────────────────────────────────
DPI  = 150
FH   = 11.5
MT   = 0.55   # top margin
MB   = 0.50   # bottom margin
AVAIL_H = FH - MT - MB   # 8.55 inches

# ── Stack geometry ────────────────────────────────────────────────────────────
ML, MR    = 0.30, 0.30
LS_FW     = 1.10      # Landsat layer face width
EM_FW     = 1.10      # AlphaEarth layer face width
SKX_LS    = 0.22;  SKY_LS = 0.12
SKX_EM    = 0.12;  SKY_EM = 0.04

GAP_STCK  = 0.24      # gap: stack ↔ table
TABLE_GAP = 0.52      # gap between the two tables

# ── Table column offsets (relative to each table's left edge) ─────────────────
LS_T_W   = 3.40
_CL_B    = 0.00       # Band
_CL_W    = 1.42       # Wavelength (µm)
_CL_S    = 2.72       # Scale (m)  — pushed right to avoid overlap

EM_T_W   = 2.20
_CE_B    = 0.00       # Band
_CE_R    = 0.56       # Value Range
_CE_S    = 1.72       # Scale — pushed right to avoid overlap

# ── Derived X coordinates ─────────────────────────────────────────────────────
LS_X0  = ML
LS_X1  = LS_X0 + LS_FW
LS_T_L = LS_X1 + SKX_LS + GAP_STCK
LS_T_R = LS_T_L + LS_T_W

EM_T_L = LS_T_R + TABLE_GAP
EM_T_R = EM_T_L + EM_T_W

EM_X0  = EM_T_R + GAP_STCK
EM_X1  = EM_X0 + EM_FW

FW = EM_X1 + SKX_EM + MR

CL_BAND  = LS_T_L + _CL_B
CL_WAVE  = LS_T_L + _CL_W
CL_SCALE = LS_T_L + _CL_S
CE_BAND  = EM_T_L + _CE_B
CE_RANGE = EM_T_L + _CE_R
CE_SCALE = EM_T_L + _CE_S

# ── Figure ────────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(FW, FH), facecolor=BG, dpi=DPI)
ax.set_xlim(0, FW)
ax.set_ylim(0, FH)
ax.axis("off")
ax.set_facecolor(BG)

# ── Landsat row geometry — absolute header heights, data fills the rest ───────
LS_HDR_H  = 0.68
LS_CHDR_H = 0.46
LS_HALF   = AVAIL_H * 0.28
LS_DATA_H = (LS_HALF - LS_HDR_H - LS_CHDR_H) / 9

ls_rows = []
y = FH - MT
for kind, data in ([('lhdr', None), ('lcolhdr', None)] +
                   [('ldata', LS_BANDS[i]) for i in range(9)]):
    h = {'lhdr': LS_HDR_H, 'lcolhdr': LS_CHDR_H, 'ldata': LS_DATA_H}[kind]
    ls_rows.append({'kind': kind, 'data': data,
                    'top': y, 'center': y - h/2, 'bot': y - h, 'h': h})
    y -= h

# ── AlphaEarth row geometry — absolute header heights, data fills the rest ───
EM_HDR_H  = 0.72
EM_CHDR_H = 0.50
EM_DATA_H = (AVAIL_H - EM_HDR_H - EM_CHDR_H) / 64

em_rows = []
y = FH - MT
for kind, em_idx in ([('ehdr', None), ('ecolhdr', None)] +
                     [('edata', i) for i in range(64)]):
    h = {'ehdr': EM_HDR_H, 'ecolhdr': EM_CHDR_H, 'edata': EM_DATA_H}[kind]
    em_rows.append({'kind': kind, 'em_idx': em_idx,
                    'top': y, 'center': y - h/2, 'bot': y - h, 'h': h})
    y -= h

ls_data_rows = [r for r in ls_rows if r['kind'] == 'ldata']
em_data_rows = [r for r in em_rows if r['kind'] == 'edata']
ls_top = ls_rows[0]['top']
ls_bot = ls_rows[-1]['bot']
em_top = em_rows[0]['top']
em_bot = em_rows[-1]['bot']

FS_S = 12.0  # section title
FS_C = 9.5   # column header
FS_D = 7.5   # data
FS_E = 6.8   # EM data (smaller for dense rows)


# ── Helpers ───────────────────────────────────────────────────────────────────
def T(x, yy, s, fs=FS_D, ha='left', va='center', color=FG, bold=False):
    ax.text(x, yy, s, fontsize=fs, color=color, ha=ha, va=va,
            fontfamily=FONT, fontweight=('bold' if bold else 'normal'))


# ── Draw Landsat table ────────────────────────────────────────────────────────
for r in ls_rows:
    kind = r['kind']
    yc   = r['center']

    if kind == 'lhdr':
        T(CL_BAND, yc + r['h']*0.13, "Satellite Image", FS_S, bold=True)
        T(CL_BAND, yc - r['h']*0.24, "(Landsat 9)", FS_S - 2.5, color=FG2)
    elif kind == 'lcolhdr':
        T(CL_BAND,  yc, "Band",       FS_C, bold=True, color=FG2)
        T(CL_WAVE,  yc, "Wavelength",  FS_C, bold=True, color=FG2)
        T(CL_SCALE, yc, "Scale",       FS_C, bold=True, color=FG2)
    elif kind == 'ldata':
        d = r['data']
        T(CL_BAND,  yc, d[0], FS_D)
        T(CL_WAVE,  yc, d[1], FS_D - 0.3, color=FG2)
        T(CL_SCALE, yc, d[2], FS_D - 0.3, color=FG2)


# ── Draw AlphaEarth table ─────────────────────────────────────────────────────
for r in em_rows:
    kind = r['kind']
    yc   = r['center']

    if kind == 'ehdr':
        T(CE_BAND, yc + r['h']*0.13, "Satellite Embedding", FS_S, bold=True)
        T(CE_BAND, yc - r['h']*0.24, "(AlphaEarth Foundations V1)", FS_S - 2.5, color=FG2)
    elif kind == 'ecolhdr':
        T(CE_BAND,  yc, "Band",        FS_C, bold=True, color=FG2)
        T(CE_RANGE, yc, "Value Range", FS_C, bold=True, color=FG2)
        T(CE_SCALE, yc, "Scale",       FS_C, bold=True, color=FG2)
    elif kind == 'edata':
        ei = r['em_idx']
        T(CE_BAND,  yc, f"A{ei:02d}", FS_E)
        T(CE_RANGE, yc, "[-1, 1]",    FS_E - 0.3, color=FG2)
        T(CE_SCALE, yc, "10 m",       FS_E - 0.3, color=FG2)


# ── Isometric layer helper ────────────────────────────────────────────────────
def draw_layer(x0, x1, yc, h, face_color, zorder, skx, sky,
               edge_color="#999999", edge_lw=0.4, alpha=1.0):
    fh = h * 0.85
    fc = np.clip(np.array(matplotlib.colors.to_rgb(face_color)), 0, 1)

    face = np.array([[x0, yc-fh/2], [x1, yc-fh/2],
                     [x1, yc+fh/2], [x0, yc+fh/2]])
    ax.add_patch(MPoly(face, closed=True,
                       facecolor=(*fc, alpha), edgecolor=edge_color,
                       lw=edge_lw, zorder=zorder))

    lighter = np.clip(fc * 1.22 + 0.06, 0, 1)
    top = np.array([[x0, yc+fh/2], [x1, yc+fh/2],
                    [x1+skx, yc+fh/2+sky], [x0+skx, yc+fh/2+sky]])
    ax.add_patch(MPoly(top, closed=True,
                       facecolor=(*lighter, alpha), edgecolor=edge_color,
                       lw=edge_lw, zorder=zorder))

    darker = np.clip(fc * 0.76, 0, 1)
    right = np.array([[x1, yc-fh/2], [x1+skx, yc-fh/2+sky],
                      [x1+skx, yc+fh/2+sky], [x1, yc+fh/2]])
    ax.add_patch(MPoly(right, closed=True,
                       facecolor=(*darker, alpha), edgecolor=edge_color,
                       lw=edge_lw, zorder=zorder))


# ── Landsat 9 stack — bottom→top draw order ───────────────────────────────────
for i in reversed(range(9)):
    r = ls_data_rows[i]
    draw_layer(LS_X0, LS_X1, r['center'], r['h'] * 0.82, LS_FC[i],
               zorder=10 + (8 - i), skx=SKX_LS, sky=SKY_LS)

# ── AlphaEarth stack — bottom→top draw order ─────────────────────────────────
for bi in reversed(range(64)):
    r = em_data_rows[bi]
    draw_layer(EM_X0, EM_X1, r['center'], r['h'] * 0.82, EM_FC[bi],
               zorder=100 + (63 - bi), skx=SKX_EM, sky=SKY_EM,
               edge_color="#c0c4c8", edge_lw=0.25)

# ── Connecting lines ──────────────────────────────────────────────────────────
# LS stack right-top edge → LS table left edge
for r in ls_data_rows:
    yc = r['center']
    ax.plot([LS_X1 + SKX_LS, LS_T_L - 0.05], [yc, yc],
            color=CONN, lw=0.5, linestyle='--', dashes=(3, 3), zorder=1)

# EM stack left edge → EM table right edge (all 64, very fine)
for r in em_data_rows:
    yc = r['center']
    ax.plot([EM_X0, EM_T_R + 0.05], [yc, yc],
            color=CONN, lw=0.3, linestyle='--', dashes=(2, 3), zorder=1)

# ── Save ──────────────────────────────────────────────────────────────────────
out = os.path.join(OUT_DIR, "output.png")
fig.savefig(out, dpi=DPI, bbox_inches='tight', facecolor=BG)
print(f"Saved → {out}")
