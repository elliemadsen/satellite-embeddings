"""
Circle embedding illustration.
Reads the first feature from the GeoJSON, decodes its 64-dim embedding,
and renders the values arranged in a circle.
Output: circle.png
"""

import json
import base64
import math
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

# ── Font ──────────────────────────────────────────────────────────────────────
_available = {f.name for f in fm.fontManager.ttflist}
FONT = "Roboto" if "Roboto" in _available else "DejaVu Sans"
plt.rcParams["font.family"] = FONT

# ── Data ──────────────────────────────────────────────────────────────────────
HERE = os.path.dirname(os.path.abspath(__file__))
GEOJSON = os.path.join(HERE, "../../dimension-reduction/data/1000_sampled_classified_embeddings.geojson")

with open(GEOJSON) as f:
    data = json.load(f)

feature = data["features"][0]
props   = feature["properties"]

lat     = props["lat"]
lon     = props["lon"]
country = props.get("SOV_A3", "")
region  = props.get("subregion_name", "")

raw_b64   = props["embedding"]
embedding = json.loads(base64.b64decode(raw_b64).decode())  # list of 64 floats
n         = len(embedding)                                   # 64

# ── Layout ────────────────────────────────────────────────────────────────────
FIG_SIZE   = 10          # inches square
RADIUS     = 3.6         # circle radius in data units (axes go -5 to 5)
TICK_IN    = 0.18        # tick mark length toward center
TICK_OUT   = 0.06        # tick mark length outward
IDX_R      = RADIUS - 0.46  # radial position of index label
VAL_R      = RADIUS + 0.52  # radial position of value label

BG  = "white"
FG  = "#111111"
POS = "#1a6e3c"   # positive value colour
NEG = "#b22222"   # negative value colour
TCK = "#aaaaaa"   # tick colour

fig, ax = plt.subplots(figsize=(FIG_SIZE, FIG_SIZE), facecolor=BG)
ax.set_facecolor(BG)
ax.set_aspect("equal")
ax.set_xlim(-5.5, 5.5)
ax.set_ylim(-5.5, 5.5)
ax.axis("off")



# ── Values around the circle ──────────────────────────────────────────────────
for i, val in enumerate(embedding):
    # angle: start at top (90°), go clockwise
    angle_deg = 90 - i * (360 / n)
    angle_rad = math.radians(angle_deg)
    cos_a, sin_a = math.cos(angle_rad), math.sin(angle_rad)

    # Value label
    vx = VAL_R * cos_a
    vy = VAL_R * sin_a
    ax.text(vx, vy, f"{val:+.4f}",
            ha="center", va="center",
            fontsize=5.8, color=FG,
            fontweight="medium",
            zorder=4)

# ── Caption ───────────────────────────────────────────────────────────────────
line1 = f"[{lat:.4f}°,  {lon:.4f}°]"
place_parts = [p for p in [country, region] if p and p != "Unknown"]
line2 = ",  ".join(place_parts) if place_parts else ""

ax.text(0, -4.85, line1,
        ha="center", va="center",
        fontsize=8, color="#999999",
        zorder=5)
if line2:
    ax.text(0, -5.2, line2,
            ha="center", va="center",
            fontsize=8, color="#999999",
            zorder=5)

# ── Save ──────────────────────────────────────────────────────────────────────
out_path = os.path.join(HERE, "circle.png")
fig.savefig(out_path, dpi=200, bbox_inches="tight", facecolor=BG)
plt.close(fig)
print(f"Saved → {out_path}")
