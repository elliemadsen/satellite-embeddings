"""
generate.py  —  Book-illustration style charts for distance-correlation thesis section.

Produces:
  embeddings_comparison_far_similar.png
  geographic_vs_embedding_distance_scatter.png
  geographic_vs_embedding_distance_heatmap.png

Intermediate cache: pairs_data.json  (skipped if already present)
"""

import base64
import json
import os
import numpy as np
import geopandas as gpd
from scipy.spatial.distance import cosine
from scipy.ndimage import gaussian_filter
from geopy.distance import geodesic
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib import font_manager
from pathlib import Path

# ── Font ───────────────────────────────────────────────────────────────────────
for p in [str(Path.home() / "Library/Fonts/Roboto-Regular.ttf"),
          str(Path.home() / "Library/Fonts/Roboto-Light.ttf"),
          "/System/Library/Fonts/SFNS.ttf"]:
    if os.path.exists(p):
        font_manager.fontManager.addfont(p)
        prop = font_manager.FontProperties(fname=p)
        plt.rcParams["font.family"] = prop.get_name()
        break

matplotlib.rcParams.update({
    "font.size": 11,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.spines.left": False,
    "axes.spines.bottom": False,
    "axes.grid": False,
    "xtick.bottom": False,
    "ytick.left": False,
    "xtick.color": "#aaaaaa",
    "ytick.color": "#aaaaaa",
    "axes.labelcolor": "#888888",
    "text.color": "#333333",
    "figure.facecolor": "#ffffff",
    "axes.facecolor": "#ffffff",
})

# ── Palette ────────────────────────────────────────────────────────────────────
AMAZON_COLOR    = "#6cbb6c"
CONGO_COLOR     = "#639ec6"
FILL_COLOR      = "#8B948E"

import copy
from matplotlib.colors import LinearSegmentedColormap

# Scatter: low-density → steel blue, high-density → warm amber
SCATTER_CMAP_T = LinearSegmentedColormap.from_list(
    "density_scatter",
    ["#c6d9f0", "#5a9fd4", "#f0a830", "#b85c00"],
)

# Density heatmap cmap: zero bins → white, then warm yellow → dark brown
HEATMAP_CMAP_OBJ = copy.copy(matplotlib.colormaps["YlOrBr"])
HEATMAP_CMAP_OBJ.set_under("white")

# ── Locations ─────────────────────────────────────────────────────────────────
LOCATIONS = {
    "amazon":    {"lat": -5.8461,  "lon": -70.5832},
    "congolian": {"lat": -0.5579,  "lon":  23.4778},
}

DATA_PATH  = Path(__file__).parent / "../../dimension-reduction/data/5000_sampled_classified_embeddings.geojson"
CACHE_PATH = Path(__file__).parent / "pairs_data.json"

# ── Helpers ───────────────────────────────────────────────────────────────────
def decode_embedding(v):
    return np.array(json.loads(base64.b64decode(v)), dtype=np.float32)

def find_nearest(gdf, lat, lon, tol=0.05):
    mask = (
        gdf["lat"].between(lat - tol, lat + tol) &
        gdf["lon"].between(lon - tol, lon + tol)
    )
    hits = gdf[mask]
    if hits.empty:
        raise ValueError(f"No point found near ({lat}, {lon})")
    return hits.iloc[0]

# ── Load data ─────────────────────────────────────────────────────────────────
print("Loading GeoJSON …")
gdf = gpd.read_file(DATA_PATH)
gdf["embedding"] = gdf["embedding"].apply(decode_embedding)
print(f"  {len(gdf):,} samples loaded")

# ── Find focus points ─────────────────────────────────────────────────────────
amazon    = find_nearest(gdf, **LOCATIONS["amazon"])
congolian = find_nearest(gdf, **LOCATIONS["congolian"])

emb_amazon    = amazon["embedding"]
emb_congolian = congolian["embedding"]

geo_dist_focus = geodesic(
    (amazon["lat"], amazon["lon"]),
    (congolian["lat"], congolian["lon"])
).kilometers
sim_focus = 1 - cosine(emb_amazon, emb_congolian)

print(f"  Amazon    ({amazon['lat']:.4f}, {amazon['lon']:.4f})")
print(f"  Congolian ({congolian['lat']:.4f}, {congolian['lon']:.4f})")
print(f"  Distance: {geo_dist_focus:.0f} km  |  Similarity: {sim_focus:.4f}")

# ── Build / load pairs cache ──────────────────────────────────────────────────
if CACHE_PATH.exists():
    print(f"\nLoading pairs from cache {CACHE_PATH} …")
    with open(CACHE_PATH) as f:
        cache = json.load(f)
    geo_dist_sample = np.array(cache["geo_dist"])
    emb_sim_sample  = np.array(cache["emb_sim"])
else:
    print("\nComputing pairwise distances (this may take a minute) …")
    np.random.seed(42)
    sample_size = 1000
    idx = np.random.choice(len(gdf), sample_size, replace=False)
    embeddings = np.stack(gdf.iloc[idx]["embedding"].values)
    coords     = gdf.iloc[idx][["lat", "lon"]].values

    upper = np.triu_indices(sample_size, k=1)
    n_pairs = len(upper[0])
    chosen  = np.random.choice(n_pairs, min(5000, n_pairs), replace=False)

    geo_dist_sample = []
    emb_sim_sample  = []
    for k, c in enumerate(chosen):
        i, j = upper[0][c], upper[1][c]
        geo_dist_sample.append(geodesic(coords[i], coords[j]).kilometers)
        emb_sim_sample.append(1 - cosine(embeddings[i], embeddings[j]))
        if (k + 1) % 500 == 0:
            print(f"  {k + 1}/{len(chosen)}")

    geo_dist_sample = np.array(geo_dist_sample)
    emb_sim_sample  = np.array(emb_sim_sample)

    with open(CACHE_PATH, "w") as f:
        json.dump({"geo_dist": geo_dist_sample.tolist(),
                   "emb_sim": emb_sim_sample.tolist()}, f)
    print(f"  Cached to {CACHE_PATH}")

# ─────────────────────────────────────────────────────────────────────────────
# CHART 1: Embedding comparison (far but similar)
# ─────────────────────────────────────────────────────────────────────────────
print("\nRendering embeddings_comparison_far_similar.png …")

fig, ax = plt.subplots(figsize=(8, 4.2))
x = np.arange(64)

ax.fill_between(x, emb_congolian, emb_amazon, alpha=0.18, color=FILL_COLOR)
ax.plot(x, emb_congolian, color=CONGO_COLOR, linewidth=1.8,
        label="Congolian rainforest", alpha=0.95)
ax.plot(x, emb_amazon,    color=AMAZON_COLOR, linewidth=1.8,
        label="Amazon rainforest",    alpha=0.95)

ax.set_xlabel("Embedding dimension", fontsize=10, labelpad=6)
ax.set_ylabel("Embedding value", fontsize=10, labelpad=6)

# Tick styling — label every dimension, rotated to fit
ax.xaxis.set_ticks(range(64))
ax.xaxis.set_ticklabels([str(i) for i in range(64)], rotation=90, fontsize=6.5)
ax.tick_params(axis="x", length=2, pad=2)
ax.tick_params(axis="y", length=0, pad=5)

# Remove all spines
for spine in ax.spines.values():
    spine.set_visible(False)

# Title block
ax.set_title(
    f"Distant locations, near embeddings\n"
    f"physical distance: {geo_dist_focus:,.0f} km  ·  cosine similarity: {sim_focus:.3f}",
    fontsize=12, fontweight="normal", loc="left", pad=14, color="#222222",
    linespacing=2.0,
)
ax.title.set_fontsize(12)
# Second line smaller + grey — override by splitting into two artists
ax.set_title("Distant locations, near embeddings",
             fontsize=12, fontweight="normal", loc="left", pad=14, color="#222222")
fig.text(0.0, 1.05,
         f"physical distance: {geo_dist_focus:,.0f} km  ·  cosine similarity: {sim_focus:.3f}",
         ha="left", va="top", fontsize=9, color="#888888",
         transform=ax.transAxes)
# Legend — slightly below top-right
from matplotlib.lines import Line2D
for ypos, color, label in [
        (0.88, CONGO_COLOR, "Congolian rainforest"),
        (0.84, AMAZON_COLOR, "Amazon rainforest"),
]:
    fig.add_artist(Line2D([0.72, 0.738], [ypos, ypos],
                          transform=fig.transFigure,
                          color=color, linewidth=2, solid_capstyle="round"))
    fig.text(0.742, ypos, label, ha="left", va="center", fontsize=9,
             color="#444444", transform=fig.transFigure)

fig.tight_layout(rect=[0, 0, 1, 0.93])
fig.savefig(Path(__file__).parent / "embeddings_comparison_far_similar.png", dpi=200, bbox_inches="tight")
plt.close(fig)
print("  ✓ saved")

# ─────────────────────────────────────────────────────────────────────────────
# CHART 2: Scatter — geographic distance vs embedding similarity
# ─────────────────────────────────────────────────────────────────────────────
print("Rendering geographic_vs_embedding_distance_scatter.png …")

fig, ax = plt.subplots(figsize=(7.5, 5))

ax.scatter(
    geo_dist_sample, emb_sim_sample,
    color="#333333", s=6, alpha=0.25, linewidths=0, rasterized=True
)

# Trend line
m, b = np.polyfit(geo_dist_sample, emb_sim_sample, 1)
x_line = np.array([geo_dist_sample.min(), geo_dist_sample.max()])
ax.plot(x_line, m * x_line + b, color="#99be2a", linewidth=1.8, zorder=5)

ax.set_xlabel("Geographic distance (km)", fontsize=10, labelpad=6)
ax.set_ylabel("Embedding similarity", fontsize=10, labelpad=6)
ax.tick_params(axis="both", length=0, pad=5)

for spine in ax.spines.values():
    spine.set_visible(False)

_t = ax.set_title("Geographic distance vs. embedding similarity",
                  fontsize=12, fontweight="normal", x=0.05, pad=12, color="#222222")
_t.set_ha("left")

corr = np.corrcoef(geo_dist_sample, emb_sim_sample)[0, 1]
fig.text(0.05, 1.02,
         f"{len(geo_dist_sample):,} randomly sampled pairs  ·  Pearson correlation: {corr:.3f} (moderate negative linear relationship)",
         ha="left", va="top", fontsize=9, color="#888888",
         transform=ax.transAxes)

fig.tight_layout()
fig.savefig(Path(__file__).parent / "geographic_vs_embedding_distance_scatter.png", dpi=200, bbox_inches="tight")
plt.close(fig)
print("  ✓ saved")

# ─────────────────────────────────────────────────────────────────────────────
# CHART 3: Heatmap — density of geographic distance vs embedding similarity
# ─────────────────────────────────────────────────────────────────────────────
print("Rendering geographic_vs_embedding_distance_heatmap.png …")

x_min, x_max = geo_dist_sample.min(), geo_dist_sample.max()
y_min, y_max = emb_sim_sample.min(), emb_sim_sample.max()
x_pad = (x_max - x_min) * 0.02
y_pad = (y_max - y_min) * 0.02

h, xedges, yedges = np.histogram2d(
    geo_dist_sample, emb_sim_sample,
    bins=60,
    range=[[x_min - x_pad, x_max + x_pad],
           [y_min - y_pad, y_max + y_pad]]
)
# Smooth the heatmap slightly for a cleaner look
h_smooth = gaussian_filter(h, sigma=1.2)

fig, ax = plt.subplots(figsize=(7.5, 5))

extent = [x_min - x_pad, x_max + x_pad, y_min - y_pad, y_max + y_pad]
im = ax.imshow(
    h_smooth.T, origin="lower", aspect="auto", extent=extent,
    cmap=HEATMAP_CMAP_OBJ, vmin=0.001, interpolation="bilinear"
)

cbar = fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
cbar.set_label("Pair density", fontsize=9, color="#888888")
cbar.ax.tick_params(labelsize=8)
cbar.outline.set_visible(False)

ax.set_xlabel("Geographic distance (km)", fontsize=10, labelpad=6)
ax.set_ylabel("Embedding similarity", fontsize=10, labelpad=6)
ax.tick_params(axis="both", length=0, pad=5)

for spine in ax.spines.values():
    spine.set_visible(False)

ax.set_title("Density of distance-similarity pairs",
             fontsize=12, fontweight="normal", loc="left", pad=12, color="#222222")

fig.text(0.13, 0.96,
         f"{len(geo_dist_sample):,} pairs, 60×60 bins",
         ha="left", va="top", fontsize=9, color="#888888",
         transform=fig.transFigure)

fig.tight_layout(rect=[0, 0, 1, 0.93])
fig.savefig(Path(__file__).parent / "geographic_vs_embedding_distance_heatmap.png", dpi=200, bbox_inches="tight")
plt.close(fig)
print("  ✓ saved")

print("\nAll charts saved.")
