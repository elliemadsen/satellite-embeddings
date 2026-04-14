"""
find_sites.py — Script 1
~~~~~~~~~~~~~~~~~~~~~~~~
Fetches annual AlphaEarth (GOOGLE/SATELLITE_EMBEDDING/V1/ANNUAL) embeddings
for a curated set of known land-change sites, 2017–2025.

Computes cosine distance from the 2017 baseline and saves everything to
sites.json for use by visualize.py.

High-change sites across multiple change types:
  • Conflict / destruction — Gaza, Palestine
  • Deforestation          — Amazon frontier (Pará, Brazil)
  • Deforestation          — Padre Marquez Mennonite colony (Ucayali, Peru)
  • Wildfire scar          — Dixie Fire, CA
  • Rapid urbanisation     — Nairobi east suburbs, Kenya
  • Desert urbanisation    — Lusail, Qatar

Usage:
    conda run -n geo python find_sites.py
"""

import ee
import json
import numpy as np
from pathlib import Path
from scipy.spatial.distance import cosine

# ── Earth Engine ──────────────────────────────────────────────────────────────
try:
    ee.Initialize(project="gsapp-map")
except Exception:
    ee.Authenticate()
    ee.Initialize(project="gsapp-map")

# ── Constants ─────────────────────────────────────────────────────────────────
YEARS            = list(range(2017, 2026))
BANDS            = [f"A{i:02d}" for i in range(64)]
SCALE            = 10          # AlphaEarth native resolution (metres)
SEARCH_RADIUS_M  = 20_000      # radius of coarse spatial search around seed point
SEARCH_SCALE_M   = 500         # pixel size for the coarse search grid
EMBED_BUFFER_M   = 400         # buffer radius for mean-embedding aggregation
SEARCH_YEAR      = 2024        # year to compare against 2017 in the search
OUTPUT_PATH      = Path(__file__).parent / "sites.json"

# ── Candidate sites ───────────────────────────────────────────────────────────

SITES = [
    {
        "name":  "gaza",
        "label": "Gaza City, Palestine",
        "story": (
            "Dense urban area subjected to severe conflict-related destruction "
            "from late 2023; embeddings should show a sharp structural discontinuity."
        ),
        "lat":  31.42,
        "lon":  34.37,
    },
    {
        "name":  "amazon",
        "label": "Amazon deforestation frontier (Pará, Brazil)",
        "story": (
            "Active arc-of-deforestation in the eastern Amazon; primary forest "
            "cleared for soy and cattle, with accelerating rates post-2019."
        ),
        "lat":  -8.50,
        "lon": -55.50,
    },
    {
        "name":  "padre_marquez",
        "label": "Padre Marquez Mennonite colony (Ucayali, Peru)",
        "story": (
            "Approximately 365 hectares of Amazonian tropical forest were cleared in 2021 "
            "to establish a new Mennonite agricultural colony near the town of Padre Marquez "
            "in the Ucayali region of Peru; an abrupt single-year deforestation signal "
            "with no prior disturbance history."
        ),
        "lat":       -8.0052,
        "lon":      -74.9429,
        # Exact clearing center (8°00'18.84"S 74°56'34.60"W) — skip spatial search
        "pinned_lat": -8.00523,
        "pinned_lon": -74.94294,
    },
    {
        "name":  "dixie_fire",
        "label": "Dixie Fire scar (Northern California, USA)",
        "story": (
            "Largest single wildfire in California history (Jul–Oct 2021, ~390 k ha); "
            "abrupt 2021 disturbance signal followed by post-fire recovery."
        ),
        "lat":  40.10,
        "lon": -121.20,
    },
    # nairobi and lusail disabled — uncomment to re-enable
    # {
    #     "name":  "nairobi",
    #     "label": "Nairobi eastern suburbs (Kenya)",
    #     "story": ("Rapid informal and formal urban expansion eastward from the CBD "
    #                "(Embakasi / Ruai corridor); substantial land-cover conversion each year."),
    #     "lat":  -1.28, "lon":  37.00,
    # },
    # {
    #     "name":  "lusail",
    #     "label": "Lusail City (Qatar)",
    #     "story": ("Purpose-built city on reclaimed desert north of Doha; construction "
    #                "peaked for the 2022 FIFA World Cup and continued through 2025."),
    #     "lat":  25.43, "lon":  51.49,
    # },
]


# ── Helpers ───────────────────────────────────────────────────────────────────
def cosine_dist(a: list[float], b: list[float]) -> float:
    return float(cosine(np.array(a, dtype=np.float64), np.array(b, dtype=np.float64)))


def find_high_change_centroid(
    lat: float, lon: float, latest_year: int = SEARCH_YEAR
) -> tuple[float, float]:
    """Search a coarse grid within SEARCH_RADIUS_M for the sub-area with the
    highest cosine distance from 2017 to latest_year.

    Uses the top-1% of pixels as the "high change" cluster and returns their
    centroid, so a single noisy pixel does not dominate.
    """
    search_region = ee.Geometry.Point(lon, lat).buffer(SEARCH_RADIUS_M).bounds()
    col = ee.ImageCollection("GOOGLE/SATELLITE_EMBEDDING/V1/ANNUAL")

    emb_ref = (
        col.filterDate("2017-01-01", "2018-01-01").mosaic().select(BANDS)
    )
    emb_cur = (
        col.filterDate(f"{latest_year}-01-01", f"{latest_year + 1}-01-01")
        .mosaic()
        .select(BANDS)
    )

    dot      = emb_ref.multiply(emb_cur).reduce(ee.Reducer.sum())
    norm_ref = emb_ref.pow(2).reduce(ee.Reducer.sum()).sqrt()
    norm_cur = emb_cur.pow(2).reduce(ee.Reducer.sum()).sqrt()
    cos_dist_img = (
        ee.Image(1)
        .subtract(dot.divide(norm_ref.multiply(norm_cur)))
        .rename("dist")
        .reproject(crs="EPSG:3857", scale=SEARCH_SCALE_M)
    )

    lon_lat  = ee.Image.pixelLonLat().reproject(crs="EPSG:3857", scale=SEARCH_SCALE_M)
    grid_img = cos_dist_img.addBands(lon_lat)
    try:
        props    = grid_img.sampleRectangle(region=search_region, defaultValue=0).getInfo()["properties"]
        dist_arr = np.array(props["dist"],      dtype=np.float32).ravel()
        lon_arr  = np.array(props["longitude"], dtype=np.float32).ravel()
        lat_arr  = np.array(props["latitude"],  dtype=np.float32).ravel()
    except Exception as exc:
        print(f"    ✗ search failed: {exc} — using seed point")
        return lat, lon

    threshold = np.nanpercentile(dist_arr, 99)
    mask      = dist_arr >= threshold
    best_lat  = float(np.nanmean(lat_arr[mask]))
    best_lon  = float(np.nanmean(lon_arr[mask]))
    best_val  = float(np.nanmean(dist_arr[mask]))
    print(f"    → found ({best_lat:.4f}, {best_lon:.4f})  "
          f"mean dist@top1%={best_val:.4f}  "
          f"(seed was {lat:.4f}, {lon:.4f})")
    return best_lat, best_lon


def fetch_mean_embedding(lat: float, lon: float, year: int) -> list[float] | None:
    """Mean AlphaEarth embedding over a EMBED_BUFFER_M-radius buffer."""
    region = ee.Geometry.Point(lon, lat).buffer(EMBED_BUFFER_M)
    img = (
        ee.ImageCollection("GOOGLE/SATELLITE_EMBEDDING/V1/ANNUAL")
        .filterDate(f"{year}-01-01", f"{year + 1}-01-01")
        .mosaic()
        .select(BANDS)
    )
    try:
        result = img.reduceRegion(
            reducer=ee.Reducer.mean(),
            geometry=region,
            scale=SCALE,
            maxPixels=1_000_000,
            bestEffort=True,
        ).getInfo()
        vals = [result.get(b) for b in BANDS]
        if any(v is None for v in vals):
            return None
        return [float(v) for v in vals]
    except Exception as exc:
        print(f"    ✗ {exc}")
        return None


# ── Main loop ─────────────────────────────────────────────────────────────────
results = []

for site in SITES:
    print(f"\n── {site['label']}")

    # Step 1: spatial search for highest-change sub-area (skipped if pinned)
    if "pinned_lat" in site:
        found_lat, found_lon = site["pinned_lat"], site["pinned_lon"]
        print(f"  Using pinned location ({found_lat:.5f}, {found_lon:.5f})")
    else:
        print(f"  Searching {SEARCH_RADIUS_M/1000:.0f} km radius for max change "
              f"(2017 → {SEARCH_YEAR}) at {SEARCH_SCALE_M} m scale …")
        found_lat, found_lon = find_high_change_centroid(site["lat"], site["lon"])

    # Step 2: fetch mean embedding over EMBED_BUFFER_M at found location
    annual_embeddings: dict[str, list[float]] = {}
    for year in YEARS:
        print(f"  {year} …", end=" ", flush=True)
        emb = fetch_mean_embedding(found_lat, found_lon, year)
        if emb is not None:
            annual_embeddings[str(year)] = emb
            print("ok")
        else:
            print("(no data)")

    # Step 3: cosine distances from 2017 mean baseline
    distances: dict[str, float] = {}
    if "2017" in annual_embeddings:
        base = annual_embeddings["2017"]
        for yr, emb in annual_embeddings.items():
            distances[yr] = cosine_dist(base, emb)
        max_d = max(distances.values())
        print(
            f"  → max distance from 2017: {max_d:.4f}  "
            f"({len(annual_embeddings)} years fetched)"
        )
    else:
        print("  ⚠ no 2017 data — distances not computed.")

    results.append({
        **site,
        "seed_lat":            site["lat"],
        "seed_lon":            site["lon"],
        "found_lat":           found_lat,
        "found_lon":           found_lon,
        "embeddings":          annual_embeddings,
        "distance_from_2017":  distances,
    })

# ── Save ──────────────────────────────────────────────────────────────────────
with open(OUTPUT_PATH, "w") as f:
    json.dump(results, f, indent=2)

print(f"\n✓ Wrote {len(results)} sites → {OUTPUT_PATH}")
