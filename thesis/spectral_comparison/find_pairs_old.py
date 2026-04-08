"""
find_pairs.py  —  Step 1 of 3

Pipeline:
  1. find_pairs.py      → candidates.json
  2. download_images.py → output/*.png  +  results.json
  3. format_output.py   → output/case{1,2}_grid.png

Search the 5000-point dataset to identify candidate location pairs for two
thesis comparison cases:

  Case 1: Similar AlphaEarth embeddings, visually different satellite images
          (ontologically similar, optically different)

  Case 2: Visually similar satellite images, different AlphaEarth embeddings
          (optically similar, ontologically different)

Visual similarity  = cosine similarity of mean Sentinel-2 B4/B3/B2 per location
                     (640 m window, 10 m/px, median 2024 composite)
Ontological sim    = cosine similarity of 64-dim AlphaEarth embeddings

Workflow:
  1.  Sample N_SAMPLE locations from the dataset.
  2.  Pre-fetch one Sentinel-2 mean-RGB signature per location via GEE.
  3.  Compute all pairwise spectral and embedding similarities.
  4.  Print ranked candidates for both cases.

Run this script to explore the data and identify good pairs.
Hard-code the selected pairs into compare.py for full tile downloads and figures.
"""

import base64
import json
import time
from pathlib import Path

import ee
import numpy as np
import geopandas as gpd

DATA_PATH  = "../../dimension-reduction/data/5000_sampled_classified_embeddings.geojson"

# ── config ────────────────────────────────────────────────────────────────────
N_SAMPLE   = 300          # locations to pre-fetch (each = 1 GEE call; ~5 min)
YEAR       = 2024
S2_BUFFER  = 1250         # half-width in metres — MUST match compare.py BUFFER_M
                          # so spec_sim is computed on the same footprint as the
                          # rendered S2 tiles (2500 m × 2500 m)
S2_SCALE   = 10           # metres/pixel for reduceRegion (native S2 resolution)

TOP_N      = 12           # how many candidates to print per case
OUTPUT_PATH = "candidates.json"  # written next to this script

REQUIRE_DIFFERENT_CONTINENT = True   # avoids adjacent-pixel / same-region pairs
CASE1_EMB_SIM_MIN  = 0.75  # minimum embedding cosine sim for Case 1 candidates
CASE1_SPEC_SIM_MAX = 0.97  # maximum spectral cosine sim for Case 1 candidates
CASE1_MAX_APPEARANCES = 1  # each location may appear in at most this many
                           # output pairs — prevents one "hub" dominating
CASE2_MAX_APPEARANCES = 1  # same dedup for Case 2
CASE2_SPEC_SIM_MIN   = 0.90  # minimum spectral cosine sim for Case 2 candidates
CASE2_CLOUD_STRICT   = 20   # for Case 2, both sigs must come from scenes with
                            # <20% cloud coverage — ensures spec_sim reflects
                            # actual land surface, not cloud coincidence
S2_CLOUD_MAX = 80          # max CLOUDY_PIXEL_PERCENTAGE for S2 composite
               # matches compare.py — intentionally allows partly-cloudy scenes
               # so a chronically overcast site gets a bright/white signature,
               # creating genuine optical contrast with a clear-sky site
S2_BANDS = ["B4", "B3", "B2", "B8"]  # include NIR so cloud (flat spectrum) and
               # vegetation (high NIR) are spectrally distinguishable

# CGLS-LC100 class names (Copernicus Global Land Service 100 m Land Cover)
CLASS_NAMES = {
    0:   "Bare/sparse vegetation",
    20:  "Shrubland",
    30:  "Grassland",
    40:  "Cropland",
    60:  "Sparse vegetation",
    90:  "Herbaceous wetland",
    111: "Closed forest — evergreen needle-leaf",
    112: "Closed forest — evergreen broad-leaf (tropical)",
    113: "Closed forest — deciduous needle-leaf (boreal)",
    114: "Closed forest — deciduous broad-leaf (temperate)",
    115: "Closed forest — mixed",
    116: "Closed forest — unknown",
    121: "Open forest — evergreen needle-leaf",
    122: "Open forest — evergreen broad-leaf",
    124: "Open forest — deciduous broad-leaf",
    125: "Open forest — mixed",
    126: "Open forest — unknown",
}


# ── Earth Engine ──────────────────────────────────────────────────────────────
try:
    ee.Initialize(project="gsapp-map")
except Exception:
    ee.Authenticate()
    ee.Initialize(project="gsapp-map")

_s2 = ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")


# ── data loading ──────────────────────────────────────────────────────────────
def load_data(path=DATA_PATH):
    gdf = gpd.read_file(path)
    gdf["embedding"] = gdf["embedding"].apply(
        lambda v: np.array(json.loads(base64.b64decode(v)))
    )
    return gdf


# ── Sentinel-2 fetching ───────────────────────────────────────────────────────
def fetch_s2_signature(lat, lon):
    """
    Compute mean Sentinel-2 reflectance over the S2_BUFFER window (matching
    compare.py's tile footprint exactly) and return a 2-tuple:
        (L2-normalised ndarray of shape (len(S2_BANDS),), cloud_pct_used)

    cloud_pct_used is the scene-level cloud filter that produced a valid result
    (S2_CLOUD_MAX first, then 101 as last-resort fallback).  Callers can use
    this to decide whether the signature is trustworthy for Case 2 matching.

    Returns None if GEE returns no data after all fallbacks.
    """
    point  = ee.Geometry.Point(lon, lat)
    region = point.buffer(S2_BUFFER).bounds()
    for cloud_pct in [S2_CLOUD_MAX, 101]:   # strict first, then no filter
        try:
            col = (_s2
                   .filterDate(f"{YEAR}-01-01", f"{YEAR + 1}-01-01")
                   .filterBounds(region))
            if cloud_pct <= 100:
                col = col.filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", cloud_pct))
            result = (
                col.select(S2_BANDS)
                .median()
                .reduceRegion(
                    reducer=ee.Reducer.mean(),
                    geometry=region,
                    scale=S2_SCALE,
                    maxPixels=int(1e6),
                )
                .getInfo()
            )
            vals = [result.get(b) for b in S2_BANDS]
            if any(v is None for v in vals):
                continue
            sig = np.array(vals, dtype=np.float32)
            norm = np.linalg.norm(sig)
            if norm > 1e-9:
                return sig / norm, cloud_pct
        except Exception:
            continue
    return None


def prefetch_signatures(sample):
    """Pre-fetch S2 signatures for all rows in sample.
    Returns dict idx -> (sig_vector, cloud_pct_used), or idx -> None on failure.
    """
    print(f"Pre-fetching Sentinel-2 signatures for {len(sample)} locations")
    print(f"  Window  : {2 * S2_BUFFER} m × {2 * S2_BUFFER} m (buffer={S2_BUFFER} m, matches compare.py)")
    print(f"  Bands   : {', '.join(S2_BANDS)}")
    print(f"  Metric  : cosine similarity of mean per-band DN over the footprint\n")
    sigs = {}
    t0 = time.time()
    for i, row in sample.iterrows():
        sigs[i] = fetch_s2_signature(row["lat"], row["lon"])
        done = len(sigs)
        if done % 20 == 0 or done == len(sample):
            n_ok     = sum(v is not None for v in sigs.values())
            n_strict = sum(v is not None and v[1] <= S2_CLOUD_MAX for v in sigs.values())
            print(f"  {done:3d}/{len(sample)}  ({n_ok} ok, {n_strict} strict-cloud)  {time.time() - t0:.0f}s")
    n_ok = sum(v is not None for v in sigs.values())
    print(f"\nFetched {n_ok}/{len(sample)} signatures  ({time.time() - t0:.0f}s total)\n")
    return sigs


# ── pairwise similarity ───────────────────────────────────────────────────────
def compute_pairwise(sample, sigs):
    """
    Compute spectral and embedding cosine similarities for all valid pairs.
    Returns a list of dicts.
    """
    pairs = []
    indices = [i for i, v in sigs.items() if v is not None]
    for a in range(len(indices)):
        ia = indices[a]
        ra = sample.loc[ia]
        sig_a, cloud_a = sigs[ia]
        for b in range(a + 1, len(indices)):
            ib = indices[b]
            rb = sample.loc[ib]
            sig_b, cloud_b = sigs[ib]
            spec_sim = float(np.dot(sig_a, sig_b))
            ua, ub   = ra["embedding"], rb["embedding"]
            emb_sim  = float(np.dot(ua, ub) / (np.linalg.norm(ua) * np.linalg.norm(ub)))
            pairs.append({
                "ia": ia, "ib": ib,
                "lat_a": float(ra["lat"]),   "lon_a": float(ra["lon"]),
                "lat_b": float(rb["lat"]),   "lon_b": float(rb["lon"]),
                "class_a": int(ra["classification"]),
                "class_b": int(rb["classification"]),
                "lc_a": CLASS_NAMES.get(int(ra["classification"]), str(ra["classification"])),
                "lc_b": CLASS_NAMES.get(int(rb["classification"]), str(rb["classification"])),
                "continent_a": ra.get("CONTINENT") or None,
                "continent_b": rb.get("CONTINENT") or None,
                "region_a": ra.get("subregion_name") or None,
                "region_b": rb.get("subregion_name") or None,
                "cloud_a":   int(cloud_a),   # cloud threshold used for sig A
                "cloud_b":   int(cloud_b),   # cloud threshold used for sig B
                "spec_sim": float(spec_sim),
                "emb_sim":  float(emb_sim),
            })
    return pairs


def _diff_continent(p):
    return p["continent_a"] != p["continent_b"]


def _fmt(p, rank):
    return "\n".join([
        f"  [{rank}] spec_sim={p['spec_sim']:.3f}  emb_sim={p['emb_sim']:.3f}",
        f"       A: ({p['lat_a']:.2f}, {p['lon_a']:.2f})  "
        f"{p['continent_a']} / {p['region_a']}  [cloud<{p['cloud_a']}]",
        f"          {p['lc_a']}  (cls {p['class_a']})",
        f"       B: ({p['lat_b']:.2f}, {p['lon_b']:.2f})  "
        f"{p['continent_b']} / {p['region_b']}  [cloud<{p['cloud_b']}]",
        f"          {p['lc_b']}  (cls {p['class_b']})",
    ])


# ── case searches ─────────────────────────────────────────────────────────────
def find_case1(pairs):
    """
    Case 1: high embedding similarity (ontologically similar),
            low spectral similarity (visually different).
    Scored by emb_sim - spec_sim.
    Returns the ranked candidate list.
    """
    print("=" * 70)
    print("CASE 1 — Ontologically similar, optically different")
    print("         (high AlphaEarth embedding sim, low Sentinel-2 spectral sim)")
    print("=" * 70)

    # ── diagnostic: show spec_sim distribution of high-emb_sim pairs ────────
    pool_cross = [
        p for p in pairs
        if p["emb_sim"] >= CASE1_EMB_SIM_MIN
        and (not REQUIRE_DIFFERENT_CONTINENT or _diff_continent(p))
    ]
    if pool_cross:
        specs = sorted(p["spec_sim"] for p in pool_cross)
        pcts  = [0, 10, 25, 50, 75, 90, 100]
        qs    = np.percentile(specs, pcts)
        print(f"  Pairs with emb_sim >= {CASE1_EMB_SIM_MIN} (cross-continental): {len(pool_cross)}")
        print(f"  spec_sim distribution of that pool:")
        for pct, q in zip(pcts, qs):
            print(f"    {pct:3d}th percentile: {q:.3f}")
        print(f"  ↳ CASE1_SPEC_SIM_MAX = {CASE1_SPEC_SIM_MAX}  "
              f"(keeps {sum(s < CASE1_SPEC_SIM_MAX for s in specs)}/{len(specs)} pairs)")
        print()
    else:
        print(f"  !! No pairs at all with emb_sim >= {CASE1_EMB_SIM_MIN} (cross-continental).")
        print(f"     Total pairs in pool: {len(pairs)}")
        print(f"     Max emb_sim seen:    {max(p['emb_sim'] for p in pairs):.3f}")
        print(f"     ↳ Lower CASE1_EMB_SIM_MIN or increase N_SAMPLE.")
        print()
    # ─────────────────────────────────────────────────────────────────────────

    candidates = [
        p for p in pool_cross
        if p["spec_sim"] < CASE1_SPEC_SIM_MAX
    ]
    candidates.sort(key=lambda p: p["emb_sim"] - p["spec_sim"], reverse=True)

    # Greedy dedup: each location may appear at most CASE1_MAX_APPEARANCES times
    # so one high-emb_sim "hub" doesn’t consume all TOP_N slots.
    appearances: dict = {}
    deduped = []
    for p in candidates:
        ka = (round(p["lat_a"], 3), round(p["lon_a"], 3))
        kb = (round(p["lat_b"], 3), round(p["lon_b"], 3))
        if (appearances.get(ka, 0) < CASE1_MAX_APPEARANCES
                and appearances.get(kb, 0) < CASE1_MAX_APPEARANCES):
            deduped.append(p)
            appearances[ka] = appearances.get(ka, 0) + 1
            appearances[kb] = appearances.get(kb, 0) + 1

    suffix = " (cross-continental)" if REQUIRE_DIFFERENT_CONTINENT else ""
    print(f"Found {len(candidates)} pairs with emb_sim >= {CASE1_EMB_SIM_MIN}"
          f" and spec_sim < {CASE1_SPEC_SIM_MAX}{suffix}")
    print(f"After dedup (max {CASE1_MAX_APPEARANCES} appearance(s) per location): "
          f"{len(deduped)} → printing top {min(TOP_N, len(deduped))}\n")
    for i, p in enumerate(deduped[:TOP_N], 1):
        print(_fmt(p, i))
        print()
    return deduped


def find_case2(pairs):
    """
    Case 2: high spectral similarity (visually similar),
            low embedding similarity (ontologically different).
    Scored by spec_sim - emb_sim.
    Returns the ranked candidate list.
    """
    print("=" * 70)
    print("CASE 2 — Optically similar, ontologically different")
    print("         (high Sentinel-2 spectral sim, low AlphaEarth embedding sim)")
    print("=" * 70)

    # Case 2 spec_sim is only trustworthy when both sigs came from clear-sky
    # scenes (cloud_pct <= CASE2_CLOUD_STRICT).  Pairs where one or both sigs
    # relied on the fallback (cloud_pct=101) may just be two cloudy patches
    # that coincidentally have the same cloud-contaminated mean reflectance.
    all_cross = [
        p for p in pairs
        if p["spec_sim"] >= CASE2_SPEC_SIM_MIN
        and (not REQUIRE_DIFFERENT_CONTINENT or _diff_continent(p))
    ]
    candidates = [
        p for p in all_cross
        if p["cloud_a"] <= CASE2_CLOUD_STRICT
        and p["cloud_b"] <= CASE2_CLOUD_STRICT
    ]
    n_cloudy_filtered = len(all_cross) - len(candidates)

    candidates.sort(key=lambda p: p["spec_sim"] - p["emb_sim"], reverse=True)

    # Greedy dedup: same logic as Case 1 — each location at most
    # CASE2_MAX_APPEARANCES times so one "hub" doesn't dominate.
    appearances: dict = {}
    deduped = []
    for p in candidates:
        ka = (round(p["lat_a"], 3), round(p["lon_a"], 3))
        kb = (round(p["lat_b"], 3), round(p["lon_b"], 3))
        if (appearances.get(ka, 0) < CASE2_MAX_APPEARANCES
                and appearances.get(kb, 0) < CASE2_MAX_APPEARANCES):
            deduped.append(p)
            appearances[ka] = appearances.get(ka, 0) + 1
            appearances[kb] = appearances.get(kb, 0) + 1

    suffix = " (cross-continental)" if REQUIRE_DIFFERENT_CONTINENT else ""
    print(f"Found {len(all_cross)} pairs with spec_sim >= {CASE2_SPEC_SIM_MIN}{suffix}")
    print(f"  {n_cloudy_filtered} removed: one or both sigs required cloud fallback "
          f"(cloud>{CASE2_CLOUD_STRICT}%)")
    print(f"  {len(candidates)} remain with strict clear-sky sigs on both sides")
    print(f"  After dedup (max {CASE2_MAX_APPEARANCES} appearance(s) per location): "
          f"{len(deduped)} → printing top {min(TOP_N, len(deduped))}\n")
    for i, p in enumerate(deduped[:TOP_N], 1):
        print(_fmt(p, i))
        print()
    return deduped


# ── main ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("Loading dataset...")
    gdf = load_data()
    print(f"Loaded {len(gdf)} points with {len(gdf.iloc[0]['embedding'])}-dim embeddings\n")

    np.random.seed(42)
    idx    = np.random.choice(len(gdf), min(N_SAMPLE, len(gdf)), replace=False)
    sample = gdf.iloc[idx].copy()

    sigs = prefetch_signatures(sample)

    print("Computing pairwise similarities...")
    pairs     = compute_pairwise(sample, sigs)
    spec_vals = [p["spec_sim"] for p in pairs]
    emb_vals  = [p["emb_sim"]  for p in pairs]
    print(f"  {len(pairs)} valid pairs")
    print(f"  Spectral sim  — mean: {np.mean(spec_vals):.3f}  "
          f"range: [{min(spec_vals):.3f}, {max(spec_vals):.3f}]")
    print(f"  Embedding sim — mean: {np.mean(emb_vals):.3f}  "
          f"range: [{min(emb_vals):.3f}, {max(emb_vals):.3f}]\n")

    case1 = find_case1(pairs)
    case2 = find_case2(pairs)

    # ── write candidates to JSON ───────────────────────────────────────────────
    # Each entry contains lat/lon, land-cover names, and both similarity scores.
    # Embeddings are excluded (large arrays); load from the GeoJSON if needed.
    output = {
        "config": {
            "n_sample": N_SAMPLE,
            "year": YEAR,
            "s2_buffer_m": S2_BUFFER,
            "s2_window_m": 2 * S2_BUFFER,
            "s2_bands": S2_BANDS,
            "s2_cloud_max": S2_CLOUD_MAX,
            "case2_cloud_strict": CASE2_CLOUD_STRICT,
            "require_different_continent": REQUIRE_DIFFERENT_CONTINENT,
            "case1_emb_sim_min": CASE1_EMB_SIM_MIN,
            "case1_spec_sim_max": CASE1_SPEC_SIM_MAX,
            "case1_max_appearances": CASE1_MAX_APPEARANCES,
            "case2_spec_sim_min": CASE2_SPEC_SIM_MIN,
            "n_pairs_evaluated": len(pairs),
            "spectral_sim_mean": round(float(np.mean(spec_vals)), 4),
            "spectral_sim_range": [round(float(min(spec_vals)), 4), round(float(max(spec_vals)), 4)],
            "embedding_sim_mean": round(float(np.mean(emb_vals)), 4),
            "embedding_sim_range": [round(float(min(emb_vals)), 4), round(float(max(emb_vals)), 4)],
        },
        "case1": [
            {k: v for k, v in p.items() if k not in ("ia", "ib")}
            for p in case1[:TOP_N]
        ],
        "case2": [
            {k: v for k, v in p.items() if k not in ("ia", "ib")}
            for p in case2[:TOP_N]
        ],
    }

    out_path = Path(__file__).parent / OUTPUT_PATH
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"Candidates written to {out_path}")
