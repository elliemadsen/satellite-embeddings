"""
find_pairs.py  —  Step 1 of 3

Pipeline:
  1. find_pairs.py      → candidates.json          (seconds, no GEE)
  2. download_images.py → output/*.png + results.json
  3. format_output.py   → output/case{1,2}_grid.png

Reads the 1024-point web_grid_data_1024.geojson dataset.  Embeddings are
looked up from the parent 2000-point GeoJSON (matched by lat/lon), and
spectral signatures are derived from the cached satellite PNG tiles in
distance_experiment_web/tiles/.  No Earth Engine calls are needed — the
entire script runs in seconds.

  Case 1: Similar AlphaEarth embeddings, visually different satellite images
          (ontologically similar, optically different)

  Case 2: Visually similar satellite images, different AlphaEarth embeddings
          (optically similar, ontologically different)

Visual similarity  = cosine similarity of per-channel RGB histograms from
                     cached S2 PNG tiles (640 m window, 64 px, median 2024 composite).
                     Using histograms (16 bins × 3 channels = 48-dim) instead of
                     mean RGB gives much better discrimination of visual texture.
Ontological sim    = cosine similarity of 64-dim AlphaEarth embeddings
"""

import base64
import json
import time
from pathlib import Path

import numpy as np
import reverse_geocoder as rg
from PIL import Image

# ── paths ─────────────────────────────────────────────────────────────────────
SCRIPT_DIR   = Path(__file__).parent
WEB_DATA_DIR = SCRIPT_DIR / ".." / ".." / "distance_experiment_web"
GRID_GEOJSON = WEB_DATA_DIR / "data" / "web_grid_data_1024.geojson"
TILES_DIR    = WEB_DATA_DIR / "tiles"

# Source dataset with embeddings (base64-encoded 64-dim vectors)
EMB_GEOJSON  = SCRIPT_DIR / ".." / ".." / "dimension-reduction" / "data" / "2000_sampled_classified_embeddings.geojson"

OUTPUT_PATH = SCRIPT_DIR / "candidates.json"

# ── config ────────────────────────────────────────────────────────────────────
TOP_N      = 30           # candidates to output per case
HIST_BINS  = 16           # bins per channel for histogram signature

REQUIRE_DIFFERENT_SUBREGION = True   # use subregion (no continent field)
CASE1_EMB_SIM_MIN   = 0.75
CASE1_SPEC_SIM_MAX  = 0.97
CASE1_MAX_APPEARANCES = 1
CASE1_MAX_PER_LC_PAIR = 3    # max pairs sharing the same (class_a, class_b) combo
CASE2_SPEC_SIM_MIN  = 0.90
CASE2_MAX_APPEARANCES = 1

# CGLS-LC100 class names
CLASS_NAMES = {
    0:   "Bare/sparse vegetation",
    20:  "Shrubland",
    30:  "Grassland",
    40:  "Cropland",
    60:  "Sparse vegetation",
    90:  "Herbaceous wetland",
    100: "Moss and lichen",
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

# Rough subregion → continent mapping for cross-continental filtering
SUBREGION_CONTINENT = {
    "Northern America":                       "North America",
    "Latin America and the Caribbean":        "South America",
    "Northern Europe":                        "Europe",
    "Western Europe":                         "Europe",
    "Eastern Europe":                         "Europe",
    "Southern Europe":                        "Europe",
    "Northern Africa":                        "Africa",
    "Sub-Saharan Africa":                     "Africa",
    "Western Asia":                           "Asia",
    "Central Asia":                           "Asia",
    "Southern Asia":                          "Asia",
    "Eastern Asia":                           "Asia",
    "South-eastern Asia":                     "Asia",
    "Melanesia":                              "Oceania",
    "Australia and New Zealand":              "Oceania",
    "Unknown":                                "Unknown",
}

# ISO-3166-1 alpha-2 → UN subregion name  (covers all codes reverse_geocoder
# is likely to return; extend as needed)
CC_TO_SUBREGION: dict[str, str] = {
    # Northern America
    "US": "Northern America", "CA": "Northern America", "MX": "Northern America",
    "GL": "Northern America",
    # Latin America & Caribbean
    "BR": "Latin America and the Caribbean",
    "AR": "Latin America and the Caribbean",
    "CL": "Latin America and the Caribbean",
    "CO": "Latin America and the Caribbean",
    "PE": "Latin America and the Caribbean",
    "VE": "Latin America and the Caribbean",
    "EC": "Latin America and the Caribbean",
    "BO": "Latin America and the Caribbean",
    "PY": "Latin America and the Caribbean",
    "UY": "Latin America and the Caribbean",
    "GY": "Latin America and the Caribbean",
    "SR": "Latin America and the Caribbean",
    "GF": "Latin America and the Caribbean",
    "PA": "Latin America and the Caribbean",
    "CR": "Latin America and the Caribbean",
    "NI": "Latin America and the Caribbean",
    "HN": "Latin America and the Caribbean",
    "GT": "Latin America and the Caribbean",
    "BZ": "Latin America and the Caribbean",
    "SV": "Latin America and the Caribbean",
    "CU": "Latin America and the Caribbean",
    "JM": "Latin America and the Caribbean",
    "HT": "Latin America and the Caribbean",
    "DO": "Latin America and the Caribbean",
    # Northern Europe
    "GB": "Northern Europe", "IE": "Northern Europe",
    "SE": "Northern Europe", "NO": "Northern Europe",
    "FI": "Northern Europe", "DK": "Northern Europe",
    "IS": "Northern Europe", "EE": "Northern Europe",
    "LV": "Northern Europe", "LT": "Northern Europe",
    # Western Europe
    "FR": "Western Europe", "DE": "Western Europe",
    "NL": "Western Europe", "BE": "Western Europe",
    "LU": "Western Europe", "AT": "Western Europe",
    "CH": "Western Europe", "MC": "Western Europe",
    "LI": "Western Europe",
    # Eastern Europe
    "RU": "Eastern Europe", "UA": "Eastern Europe",
    "PL": "Eastern Europe", "CZ": "Eastern Europe",
    "SK": "Eastern Europe", "HU": "Eastern Europe",
    "RO": "Eastern Europe", "BG": "Eastern Europe",
    "BY": "Eastern Europe", "MD": "Eastern Europe",
    # Southern Europe
    "IT": "Southern Europe", "ES": "Southern Europe",
    "PT": "Southern Europe", "GR": "Southern Europe",
    "HR": "Southern Europe", "RS": "Southern Europe",
    "BA": "Southern Europe", "AL": "Southern Europe",
    "MK": "Southern Europe", "ME": "Southern Europe",
    "SI": "Southern Europe", "MT": "Southern Europe",
    "TR": "Western Asia",
    # Northern Africa
    "EG": "Northern Africa", "LY": "Northern Africa",
    "TN": "Northern Africa", "DZ": "Northern Africa",
    "MA": "Northern Africa", "SD": "Northern Africa",
    # Sub-Saharan Africa
    "NG": "Sub-Saharan Africa", "ET": "Sub-Saharan Africa",
    "CD": "Sub-Saharan Africa", "ZA": "Sub-Saharan Africa",
    "KE": "Sub-Saharan Africa", "TZ": "Sub-Saharan Africa",
    "UG": "Sub-Saharan Africa", "GH": "Sub-Saharan Africa",
    "MZ": "Sub-Saharan Africa", "MG": "Sub-Saharan Africa",
    "CM": "Sub-Saharan Africa", "CI": "Sub-Saharan Africa",
    "AO": "Sub-Saharan Africa", "ML": "Sub-Saharan Africa",
    "NE": "Sub-Saharan Africa", "BF": "Sub-Saharan Africa",
    "MW": "Sub-Saharan Africa", "ZM": "Sub-Saharan Africa",
    "SN": "Sub-Saharan Africa", "TD": "Sub-Saharan Africa",
    "SO": "Sub-Saharan Africa", "ZW": "Sub-Saharan Africa",
    "GN": "Sub-Saharan Africa", "RW": "Sub-Saharan Africa",
    "BJ": "Sub-Saharan Africa", "BI": "Sub-Saharan Africa",
    "SS": "Sub-Saharan Africa", "TG": "Sub-Saharan Africa",
    "SL": "Sub-Saharan Africa", "LR": "Sub-Saharan Africa",
    "CF": "Sub-Saharan Africa", "MR": "Sub-Saharan Africa",
    "ER": "Sub-Saharan Africa", "NA": "Sub-Saharan Africa",
    "GM": "Sub-Saharan Africa", "BW": "Sub-Saharan Africa",
    "GA": "Sub-Saharan Africa", "LS": "Sub-Saharan Africa",
    "GW": "Sub-Saharan Africa", "GQ": "Sub-Saharan Africa",
    "SZ": "Sub-Saharan Africa", "DJ": "Sub-Saharan Africa",
    "KM": "Sub-Saharan Africa", "CG": "Sub-Saharan Africa",
    # Western Asia
    "SA": "Western Asia", "IQ": "Western Asia",
    "IR": "Western Asia", "SY": "Western Asia",
    "YE": "Western Asia", "JO": "Western Asia",
    "IL": "Western Asia", "PS": "Western Asia",
    "LB": "Western Asia", "KW": "Western Asia",
    "AE": "Western Asia", "OM": "Western Asia",
    "QA": "Western Asia", "BH": "Western Asia",
    "GE": "Western Asia", "AM": "Western Asia",
    "AZ": "Western Asia", "CY": "Western Asia",
    # Central Asia
    "KZ": "Central Asia", "UZ": "Central Asia",
    "TM": "Central Asia", "TJ": "Central Asia",
    "KG": "Central Asia",
    # Southern Asia
    "IN": "Southern Asia", "PK": "Southern Asia",
    "BD": "Southern Asia", "AF": "Southern Asia",
    "NP": "Southern Asia", "LK": "Southern Asia",
    "BT": "Southern Asia", "MV": "Southern Asia",
    # Eastern Asia
    "CN": "Eastern Asia", "JP": "Eastern Asia",
    "KR": "Eastern Asia", "KP": "Eastern Asia",
    "MN": "Eastern Asia", "TW": "Eastern Asia",
    # South-eastern Asia
    "ID": "South-eastern Asia", "PH": "South-eastern Asia",
    "VN": "South-eastern Asia", "TH": "South-eastern Asia",
    "MM": "South-eastern Asia", "MY": "South-eastern Asia",
    "KH": "South-eastern Asia", "LA": "South-eastern Asia",
    "SG": "South-eastern Asia", "TL": "South-eastern Asia",
    "BN": "South-eastern Asia",
    # Melanesia
    "PG": "Melanesia", "FJ": "Melanesia",
    "SB": "Melanesia", "VU": "Melanesia", "NC": "Melanesia",
    # Australia & New Zealand
    "AU": "Australia and New Zealand",
    "NZ": "Australia and New Zealand",
}


def _resolve_unknown_subregions(locations: list[dict]) -> None:
    """Batch reverse-geocode locations whose subregion is 'Unknown' and fill in
    the subregion + continent fields in-place."""
    unknowns = [(i, loc) for i, loc in enumerate(locations)
                if loc["subregion"] == "Unknown"]
    if not unknowns:
        return
    coords = [(loc["lat"], loc["lon"]) for _, loc in unknowns]
    results = rg.search(coords)
    resolved = 0
    for (idx, loc), hit in zip(unknowns, results):
        cc = hit.get("cc", "")
        subregion = CC_TO_SUBREGION.get(cc)
        if subregion:
            loc["subregion"] = subregion
            loc["continent"] = SUBREGION_CONTINENT.get(subregion, "Unknown")
            resolved += 1
    if resolved:
        print(f"  ✓ Resolved {resolved}/{len(unknowns)} unknown regions via reverse geocoding")
    remaining = len(unknowns) - resolved
    if remaining:
        print(f"  ⚠ {remaining} locations still have Unknown region")


# ── data loading ──────────────────────────────────────────────────────────────

def load_data():
    """
    Load the 1024-point grid dataset with:
      - lat, lon, classification, subregion_name, tile_dir, index
      - embedding (64-dim float32) matched from the 2000-point source
      - spec_sig (L2-normalised per-channel RGB histogram from cached S2 PNG)
    Returns a list of dicts, one per location.
    """
    print("Loading grid dataset …")
    with open(GRID_GEOJSON) as f:
        grid = json.load(f)
    print(f"  {len(grid['features'])} locations from {GRID_GEOJSON.name}")

    # Build embedding lookup from 2000-point source (matched by rounded lat/lon)
    print(f"Loading embeddings from {EMB_GEOJSON.name} …")
    with open(EMB_GEOJSON) as f:
        src = json.load(f)
    emb_lookup: dict[tuple, np.ndarray] = {}
    for feat in src["features"]:
        p = feat["properties"]
        coords = feat["geometry"]["coordinates"]
        key = (round(coords[1], 6), round(coords[0], 6))
        emb_lookup[key] = np.array(json.loads(base64.b64decode(p["embedding"])),
                                   dtype=np.float32)
    print(f"  {len(emb_lookup)} embeddings indexed\n")

    # Build location records
    locations = []
    n_emb_miss = 0
    n_tile_miss = 0
    for feat in grid["features"]:
        p = feat["properties"]
        lat, lon = p["lat"], p["lon"]
        idx = p["index"]
        tile_dir = p["tile_dir"]                 # e.g. "tiles/0042"
        sat_path = TILES_DIR.parent / tile_dir / f"{idx:04d}_satellite.png"

        # Embedding
        key = (round(lat, 6), round(lon, 6))
        emb = emb_lookup.get(key)
        if emb is None:
            n_emb_miss += 1
            continue

        # Visual signature from cached satellite PNG (per-channel histogram)
        if not sat_path.exists():
            n_tile_miss += 1
            continue
        arr = np.array(Image.open(sat_path).convert("RGB"), dtype=np.float32)
        # Build per-channel histogram → 48-dim vector (16 bins × 3 channels)
        hists = []
        for ch in range(3):
            h, _ = np.histogram(arr[:, :, ch], bins=HIST_BINS, range=(0, 255))
            hists.append(h.astype(np.float32))
        hist_vec = np.concatenate(hists)
        norm = np.linalg.norm(hist_vec)
        spec_sig = hist_vec / norm if norm > 1e-9 else None
        if spec_sig is None:
            continue

        subregion = p.get("subregion_name") or "Unknown"
        continent = SUBREGION_CONTINENT.get(subregion, "Unknown")

        locations.append({
            "idx":            idx,
            "lat":            float(lat),
            "lon":            float(lon),
            "classification": int(p["classification"]),
            "lc":             CLASS_NAMES.get(int(p["classification"]),
                                              str(p["classification"])),
            "subregion":      subregion,
            "continent":      continent,
            "tile_dir":       tile_dir,
            "embedding":      emb,
            "spec_sig":       spec_sig,
        })

    if n_emb_miss:
        print(f"  ⚠ {n_emb_miss} locations without embedding match (skipped)")
    if n_tile_miss:
        print(f"  ⚠ {n_tile_miss} locations without satellite tile (skipped)")
    print(f"  {len(locations)} locations ready")

    # Resolve any Unknown subregions via reverse geocoding
    _resolve_unknown_subregions(locations)
    print()

    return locations


# ── pairwise similarity ───────────────────────────────────────────────────────

def compute_pairwise(locations):
    """All-pairs cosine similarities (spectral and embedding)."""
    n = len(locations)
    pairs = []
    for a in range(n):
        la = locations[a]
        for b in range(a + 1, n):
            lb = locations[b]
            spec_sim = float(np.dot(la["spec_sig"], lb["spec_sig"]))
            ua, ub = la["embedding"], lb["embedding"]
            emb_sim = float(np.dot(ua, ub) / (np.linalg.norm(ua) * np.linalg.norm(ub)))
            pairs.append({
                "ia": la["idx"], "ib": lb["idx"],
                "lat_a": la["lat"],   "lon_a": la["lon"],
                "lat_b": lb["lat"],   "lon_b": lb["lon"],
                "class_a": la["classification"],
                "class_b": lb["classification"],
                "lc_a": la["lc"],     "lc_b": lb["lc"],
                "continent_a": la["continent"],
                "continent_b": lb["continent"],
                "region_a": la["subregion"],
                "region_b": lb["subregion"],
                "tile_dir_a": la["tile_dir"],
                "tile_dir_b": lb["tile_dir"],
                "spec_sim": spec_sim,
                "emb_sim":  emb_sim,
            })
    return pairs


def _diff_continent(p):
    return p["continent_a"] != p["continent_b"]


def _fmt(p, rank):
    return "\n".join([
        f"  [{rank}] spec_sim={p['spec_sim']:.3f}  emb_sim={p['emb_sim']:.3f}",
        f"       A: ({p['lat_a']:.2f}, {p['lon_a']:.2f})  "
        f"{p['continent_a']} / {p['region_a']}",
        f"          {p['lc_a']}  (cls {p['class_a']})  tile={p['tile_dir_a']}",
        f"       B: ({p['lat_b']:.2f}, {p['lon_b']:.2f})  "
        f"{p['continent_b']} / {p['region_b']}",
        f"          {p['lc_b']}  (cls {p['class_b']})  tile={p['tile_dir_b']}",
    ])


def _greedy_dedup(candidates, max_appearances, max_per_lc_pair=None):
    """Return candidates where each location appears at most max_appearances times,
    and optionally each (class_a, class_b) combo appears at most max_per_lc_pair times."""
    appearances: dict = {}
    lc_counts: dict = {}
    out = []
    for p in candidates:
        ka = (round(p["lat_a"], 3), round(p["lon_a"], 3))
        kb = (round(p["lat_b"], 3), round(p["lon_b"], 3))
        if (appearances.get(ka, 0) >= max_appearances
                or appearances.get(kb, 0) >= max_appearances):
            continue
        if max_per_lc_pair is not None:
            lc_key = tuple(sorted([p["class_a"], p["class_b"]]))
            if lc_counts.get(lc_key, 0) >= max_per_lc_pair:
                continue
            lc_counts[lc_key] = lc_counts.get(lc_key, 0) + 1
        out.append(p)
        appearances[ka] = appearances.get(ka, 0) + 1
        appearances[kb] = appearances.get(kb, 0) + 1
    return out


# ── case searches ─────────────────────────────────────────────────────────────

def find_case1(pairs):
    """
    Case 1: high embedding similarity, low spectral similarity.
    Scored by emb_sim - spec_sim.
    """
    print("=" * 70)
    print("CASE 1 — Ontologically similar, optically different")
    print("         (high embedding sim, low spectral sim)")
    print("=" * 70)

    pool = [
        p for p in pairs
        if p["emb_sim"] >= CASE1_EMB_SIM_MIN
        and (not REQUIRE_DIFFERENT_SUBREGION or _diff_continent(p))
    ]

    # Diagnostic: spec_sim distribution
    if pool:
        specs = sorted(p["spec_sim"] for p in pool)
        pcts = [0, 10, 25, 50, 75, 90, 100]
        qs = np.percentile(specs, pcts)
        print(f"  Pool (emb_sim >= {CASE1_EMB_SIM_MIN}, cross-continental): {len(pool)}")
        print(f"  spec_sim percentiles:")
        for pct, q in zip(pcts, qs):
            print(f"    {pct:3d}th: {q:.3f}")
        kept = sum(s < CASE1_SPEC_SIM_MAX for s in specs)
        print(f"  ↳ CASE1_SPEC_SIM_MAX={CASE1_SPEC_SIM_MAX} keeps {kept}/{len(specs)}")
        print()
    else:
        print(f"  !! No cross-continental pairs with emb_sim >= {CASE1_EMB_SIM_MIN}")
        if pairs:
            print(f"     max emb_sim seen: {max(p['emb_sim'] for p in pairs):.3f}")
        print()

    candidates = [p for p in pool if p["spec_sim"] < CASE1_SPEC_SIM_MAX]
    candidates.sort(key=lambda p: p["emb_sim"] - p["spec_sim"], reverse=True)
    deduped = _greedy_dedup(candidates, CASE1_MAX_APPEARANCES,
                            max_per_lc_pair=CASE1_MAX_PER_LC_PAIR)

    print(f"Candidates: {len(candidates)}  →  after dedup: {len(deduped)}"
          f"  →  top {min(TOP_N, len(deduped))}\n")
    for i, p in enumerate(deduped[:TOP_N], 1):
        print(_fmt(p, i))
        print()
    return deduped


def find_case2(pairs):
    """
    Case 2: high spectral similarity, low embedding similarity.
    Scored by spec_sim - emb_sim.
    """
    print("=" * 70)
    print("CASE 2 — Optically similar, ontologically different")
    print("         (high spectral sim, low embedding sim)")
    print("=" * 70)

    candidates = [
        p for p in pairs
        if p["spec_sim"] >= CASE2_SPEC_SIM_MIN
        and (not REQUIRE_DIFFERENT_SUBREGION or _diff_continent(p))
    ]
    candidates.sort(key=lambda p: p["spec_sim"] - p["emb_sim"], reverse=True)
    deduped = _greedy_dedup(candidates, CASE2_MAX_APPEARANCES)

    print(f"Candidates: {len(candidates)}  →  after dedup: {len(deduped)}"
          f"  →  top {min(TOP_N, len(deduped))}\n")
    for i, p in enumerate(deduped[:TOP_N], 1):
        print(_fmt(p, i))
        print()
    return deduped


# ── main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    t0 = time.time()
    locations = load_data()

    print("Computing all pairwise similarities …")
    pairs = compute_pairwise(locations)
    spec_vals = [p["spec_sim"] for p in pairs]
    emb_vals  = [p["emb_sim"]  for p in pairs]
    print(f"  {len(pairs)} pairs")
    print(f"  Spectral sim  — mean: {np.mean(spec_vals):.3f}  "
          f"range: [{min(spec_vals):.3f}, {max(spec_vals):.3f}]")
    print(f"  Embedding sim — mean: {np.mean(emb_vals):.3f}  "
          f"range: [{min(emb_vals):.3f}, {max(emb_vals):.3f}]\n")

    case1 = find_case1(pairs)
    case2 = find_case2(pairs)

    # ── write candidates.json ──────────────────────────────────────────────
    output = {
        "config": {
            "dataset": str(GRID_GEOJSON.name),
            "n_locations": len(locations),
            "require_different_subregion": REQUIRE_DIFFERENT_SUBREGION,
            "case1_emb_sim_min":    CASE1_EMB_SIM_MIN,
            "case1_spec_sim_max":   CASE1_SPEC_SIM_MAX,
            "case1_max_appearances": CASE1_MAX_APPEARANCES,
            "case2_spec_sim_min":   CASE2_SPEC_SIM_MIN,
            "case2_max_appearances": CASE2_MAX_APPEARANCES,
            "n_pairs_evaluated": len(pairs),
            "spectral_sim_mean":  round(float(np.mean(spec_vals)), 4),
            "spectral_sim_range": [round(float(min(spec_vals)), 4),
                                   round(float(max(spec_vals)), 4)],
            "embedding_sim_mean":  round(float(np.mean(emb_vals)), 4),
            "embedding_sim_range": [round(float(min(emb_vals)), 4),
                                    round(float(max(emb_vals)), 4)],
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

    with open(OUTPUT_PATH, "w") as f:
        json.dump(output, f, indent=2)

    elapsed = time.time() - t0
    print(f"Candidates written to {OUTPUT_PATH}  ({elapsed:.1f}s)")
