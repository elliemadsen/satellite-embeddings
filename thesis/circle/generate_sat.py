"""
Fetch the satellite embedding false-colour image (bands A00, A01, A02)
for the first GeoJSON feature location at ~1 km resolution,
then render it cropped to a circle.
Output: circle_sat.png
"""

import json
import base64
import os
import math

import numpy as np
import ee
from PIL import Image, ImageDraw

# ── Earth Engine ───────────────────────────────────────────────────────────────
try:
    ee.Initialize(project="gsapp-map")
except Exception:
    ee.Authenticate()
    ee.Initialize(project="gsapp-map")

# ── Data ──────────────────────────────────────────────────────────────────────
HERE = os.path.dirname(os.path.abspath(__file__))
GEOJSON = os.path.join(HERE, "../../dimension-reduction/data/1000_sampled_classified_embeddings.geojson")

with open(GEOJSON) as f:
    data = json.load(f)

feature = data["features"][0]
props   = feature["properties"]
LAT     = props["lat"]
LON     = props["lon"]
YEAR    = 2024

# ── Fetch ──────────────────────────────────────────────────────────────────────
SCALE   = 10       # metres per pixel  (10 m native resolution)
N       = 100      # output pixels → 100 × 10 m = 1 km across
BANDS   = ["A01", "A30", "A60"]

embed_img = (
    ee.ImageCollection("GOOGLE/SATELLITE_EMBEDDING/V1/ANNUAL")
    .filterDate(f"{YEAR}-01-01", f"{YEAR+1}-01-01")
    .mosaic()
    .select(BANDS)
    .reproject(crs="EPSG:4326", scale=SCALE)
)

half_m = SCALE * N / 2
region = ee.Geometry.Point(LON, LAT).buffer(half_m).bounds()

print(f"Fetching {N}×{N} pixels at {SCALE} m/px around [{LAT:.4f}, {LON:.4f}] …")
sample = embed_img.sampleRectangle(region=region, defaultValue=0).getInfo()

grid = np.zeros((N, N, 3), dtype=np.float32)
for bi, bname in enumerate(BANDS):
    arr = np.array(sample["properties"][bname], dtype=np.float32)
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)
    r = min(arr.shape[0], N)
    c = min(arr.shape[1], N)
    grid[:r, :c, bi] = arr[:r, :c]

print(f"  raw value range: [{grid.min():.4f}, {grid.max():.4f}]")
print(f"  per-band: R=[{grid[:,:,0].min():.4f},{grid[:,:,0].max():.4f}]  G=[{grid[:,:,1].min():.4f},{grid[:,:,1].max():.4f}]  B=[{grid[:,:,2].min():.4f},{grid[:,:,2].max():.4f}]")

# ── Normalize each band independently ────────────────────────────────────────
rgb = np.zeros_like(grid)
for bi in range(3):
    ch = grid[:, :, bi]
    cmin, cmax = ch.min(), ch.max()
    rgb[:, :, bi] = (ch - cmin) / max(cmax - cmin, 1e-6)
rgb = (rgb * 255).clip(0, 255).astype(np.uint8)

img = Image.fromarray(rgb, mode="RGB").resize((512, 512), Image.NEAREST)

# ── Circular crop ─────────────────────────────────────────────────────────────
size = img.size[0]
mask = Image.new("L", (size, size), 0)
draw = ImageDraw.Draw(mask)
draw.ellipse((0, 0, size, size), fill=255)

out = Image.new("RGBA", (size, size), (255, 255, 255, 0))
out.paste(img.convert("RGBA"), mask=mask)

# ── Save ──────────────────────────────────────────────────────────────────────
out_path = os.path.join(HERE, "circle_sat.png")
out.save(out_path)
print(f"Saved → {out_path}")
