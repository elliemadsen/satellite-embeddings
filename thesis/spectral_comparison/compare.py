"""
compare.py

Downloads Sentinel-2 and AlphaEarth tiles for candidate pairs in two thesis cases,
then produces a browseable grid figure per case so you can pick the most visually
compelling examples.

Case 1 — Ontologically similar, optically different
  Sites with HIGH AlphaEarth image-level embedding cosine similarity but clearly
  different-looking satellite images.  The embedding captures ecological/semantic
  similarity that isn't obvious from RGB alone.

Case 2 — Optically similar, different embedding
  Sites whose Sentinel-2 RGB looks nearly identical but that AlphaEarth assigns
  to different regions of embedding space, showing the model resolves context
  beyond raw spectral appearance.

Candidate pairs were identified by:
  - Case 1: searching the 5000-point dataset for cross-continental pairs with
    high embedding cosine similarity but contrasting land-cover classes.
  - Case 2: pairing same-class sites from very different geographic contexts
    whose first-3-band AlphaEarth proxy (spectral proxy) is similar while their
    full 64-dim embedding diverges.

Output (written to output/):
  case1_grid.png, case2_grid.png   — browseable grid figures (main output)
  case{1,2}_pair{N}_{site}_s2.png  — individual Sentinel-2 tiles
  case{1,2}_pair{N}_{site}_ae.png  — individual AlphaEarth tiles

Usage:
  python compare.py                 # download + render all pairs
  python compare.py --case 1       # only case 1
  python compare.py --skip-download # use cached PNGs, re-render grid only
"""

import argparse
import json
import sys
import time
from pathlib import Path
from io import BytesIO

import ee
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import requests
from PIL import Image
from scipy.spatial.distance import cosine as cosine_distance

matplotlib.rcParams["font.family"] = "Courier New"

# ── Earth Engine ──────────────────────────────────────────────────────────────
ee.Initialize(project="gsapp-map")

# ── Parameters ────────────────────────────────────────────────────────────────
BUFFER_M     = 1250    # half-width of tile in metres → 2.5 km × 2.5 km patch
IMG_PX       = 256     # download resolution (pixels per side)
YEAR         = 2024
S2_MAX       = 3000    # Sentinel-2 SR display ceiling (digital numbers)
S2_CLOUD_MAX = 80      # max scene-level CLOUDY_PIXEL_PERCENTAGE to include
               # 80 % is permissive enough for high-latitude and consistently
               # cloudy sites; the median composite will still look partly
               # cloudy for locations where most scenes are overcast, which
               # is intentional for Case 1 (optical difference via clouds).

OUTPUT_DIR = Path(__file__).parent / "output"
OUTPUT_DIR.mkdir(exist_ok=True)

# ── EE datasets ───────────────────────────────────────────────────────────────
alphaearth = ee.ImageCollection("GOOGLE/SATELLITE_EMBEDDING/V1/ANNUAL")

s2_rgb = (
    ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
    .filter(ee.Filter.calendarRange(YEAR, YEAR, "year"))
    .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", S2_CLOUD_MAX))
    .select(["B4", "B3", "B2"])
    .median()
)

AE_BANDS = [f"A{i:02d}" for i in range(64)]   # A00 … A63
AE_VIS   = ["A00", "A01", "A02"]              # first 3 bands → RGB

# ── Candidate pairs — loaded from candidates.json ───────────────────────────
# Run find_pairs.py to regenerate candidates.json.

CANDIDATES_FILE = Path(__file__).parent / "candidates.json"


def _pair_from_candidate(p: dict) -> dict:
    """Convert a find_pairs.py candidate dict to the compare.py pair format."""
    cont_a = p["continent_a"] or "Unknown"
    cont_b = p["continent_b"] or "Unknown"
    return {
        "label": (
            f"{p['lc_a']}  ({p['region_a']})\n"
            f"vs  {p['lc_b']}  ({p['region_b']})"
        ),
        "point_emb_sim":  p["emb_sim"],
        "point_spec_sim": p["spec_sim"],
        "a": {
            "name": f"{p['region_a']}\n{cont_a}",
            "lat":  p["lat_a"],
            "lon":  p["lon_a"],
            "note": f"cls {p['class_a']} · {p['lc_a']}",
        },
        "b": {
            "name": f"{p['region_b']}\n{cont_b}",
            "lat":  p["lat_b"],
            "lon":  p["lon_b"],
            "note": f"cls {p['class_b']} · {p['lc_b']}",
        },
    }


if not CANDIDATES_FILE.exists():
    sys.exit(f"ERROR: {CANDIDATES_FILE} not found. Run find_pairs.py first.")

with open(CANDIDATES_FILE) as _f:
    _candidates = json.load(_f)

CASE1_PAIRS = [_pair_from_candidate(p) for p in _candidates["case1"]]
CASE2_PAIRS = [_pair_from_candidate(p) for p in _candidates["case2"]]

if False:  # dead code kept only to satisfy linters; never executed
    CASE1_PAIRS = [
    # Siberian closed boreal forest vs Arctic-Canadian bare/open tundra
    # Dense dark-green conifer canopy vs open lake-and-tundra mosaic
    {
        "label": "Siberian boreal forest  vs  Arctic-Canadian tundra/lakes",
        "point_emb_sim": 0.936,
        "a": {"name": "Siberian boreal\nRussia", "lat": 65.90, "lon": 117.99,
              "note": "cls 113 · closed deciduous needle-leaf (larch taiga)"},
        "b": {"name": "Arctic tundra/lakes\nNW Canada", "lat": 67.68, "lon": -122.18,
              "note": "cls 0 · bare/sparse (lake & open tundra complex)"},
    },
    # Amazon tropical rainforest vs Congo basin tropical rainforest
    # Same class, same high similarity — but different continent / landscape texture
    {
        "label": "Amazon rainforest  vs  Congo basin rainforest",
        "point_emb_sim": 0.917,
        "a": {"name": "Amazon forest\nBrazil", "lat": -5.85, "lon": -70.58,
              "note": "cls 112 · closed evergreen broad-leaf (tropical)"},
        "b": {"name": "Congo basin forest\nDR Congo", "lat": -1.43, "lon": 15.92,
              "note": "cls 112 · closed evergreen broad-leaf (tropical)"},
    },
    # Amazon/Cerrado ecotone (Brazil) vs Miombo open woodland (Tanzania)
    # Patchy forest fragment vs African savanna-woodland
    {
        "label": "Brazilian cerrado forest  vs  Tanzanian miombo woodland",
        "point_emb_sim": 0.871,
        "a": {"name": "Tropical forest fragment\nMaranhão, Brazil", "lat": -5.46, "lon": -44.89,
              "note": "cls 112 · Amazon/Cerrado ecotone forest"},
        "b": {"name": "Miombo open woodland\nTanzania", "lat": -9.63, "lon": 37.18,
              "note": "cls 124 · open deciduous broad-leaf (miombo)"},
    },
    # Saharan sand desert vs equatorial African bare (burn scar / rock exposure)
    # Both class 0 — hot sandy desert vs dark tropical bare ground
    {
        "label": "Sahara desert  vs  Equatorial African bare ground",
        "point_emb_sim": 0.854,
        "a": {"name": "Sahara desert\nLibya", "lat": 30.90, "lon": 20.05,
              "note": "cls 0 · hot sandy desert (hyperarid)"},
        "b": {"name": "Bare ground / clearing\nGabon", "lat": -0.55, "lon": 9.13,
              "note": "cls 0 · equatorial bare (burn scar or rock exposure)"},
    },
    # Siberia boreal vs Iowa 'bare' (likely fallow/harvested cropland)
    # Dense forest vs open flat agricultural plain
    {
        "label": "Siberian boreal  vs  North American bare/cropland",
        "point_emb_sim": 0.930,
        "a": {"name": "Siberian larch taiga\nRussia", "lat": 65.90, "lon": 117.99,
              "note": "cls 113 · closed deciduous needle-leaf"},
        "b": {"name": "Open plain\nCanada NWT", "lat": 65.35, "lon": -96.80,
              "note": "cls 0 · bare/sparse (boreal transition or tundra)"},
    },
    # Iranian plateau (rocky, high-altitude) vs Australian outback (red sand)
    # Both class 0, same overall appearance but different texture
    {
        "label": "Iranian plateau (rocky)  vs  Australian outback (red sand)",
        "point_emb_sim": 0.492,
        "a": {"name": "Iranian plateau\nIran", "lat": 34.30, "lon": 56.50,
              "note": "cls 0 · rocky high-altitude plateau"},
        "b": {"name": "Australian outback\nWestern Australia", "lat": -24.90, "lon": 123.10,
              "note": "cls 0 · red-sand arid interior"},
    },
]  # end dead code CASE1_PAIRS

# ── Helpers ───────────────────────────────────────────────────────────────────

def tile_region(lat: float, lon: float, buf: int = BUFFER_M) -> ee.Geometry:
    return ee.Geometry.Point([lon, lat]).buffer(buf).bounds()


def fetch_png(url: str, retries: int = 3) -> Image.Image:
    for attempt in range(retries):
        try:
            resp = requests.get(url, timeout=120)
            resp.raise_for_status()
            return Image.open(BytesIO(resp.content)).convert("RGB")
        except Exception as exc:
            if attempt == retries - 1:
                raise RuntimeError(f"download failed after {retries} attempts: {exc}")
            print(f"      retry {attempt + 1}: {exc}")
            time.sleep(5)


def download_s2(lat: float, lon: float) -> Image.Image:
    region = tile_region(lat, lon)
    url = s2_rgb.getThumbURL({
        "region": region.getInfo(),
        "dimensions": f"{IMG_PX}x{IMG_PX}",
        "format": "png",
        "min": 0, "max": S2_MAX,
        "bands": ["B4", "B3", "B2"],
        "gamma": 1.4,
    })
    return fetch_png(url)


def download_alphaearth(lat: float, lon: float) -> Image.Image:
    region = tile_region(lat, lon)
    url = (
        alphaearth.filterBounds(region).mosaic()
        .select(AE_VIS)
        .getThumbURL({
            "region": region.getInfo(),
            "dimensions": f"{IMG_PX}x{IMG_PX}",
            "format": "png",
            "min": -0.3, "max": 0.3,
        })
    )
    return fetch_png(url)


def image_embedding(lat: float, lon: float) -> np.ndarray:
    """Mean AlphaEarth embedding over the tile (64-dim)."""
    region = tile_region(lat, lon)
    result = (
        alphaearth.filterBounds(region).mosaic()
        .select(AE_BANDS)
        .reduceRegion(ee.Reducer.mean(), geometry=region, scale=10, maxPixels=int(1e9))
        .getInfo()
    )
    return np.array([result.get(b, np.nan) for b in AE_BANDS])


def s2_mean_rgb(lat: float, lon: float) -> np.ndarray:
    """Mean Sentinel-2 B4/B3/B2 over the tile."""
    region = tile_region(lat, lon)
    result = (
        s2_rgb.reduceRegion(ee.Reducer.mean(), geometry=region, scale=10, maxPixels=int(1e9))
        .getInfo()
    )
    return np.array([result.get("B4", 0.0), result.get("B3", 0.0), result.get("B2", 0.0)])


def cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    mask = ~(np.isnan(a) | np.isnan(b))
    if mask.sum() < 2:
        return float("nan")
    return float(1.0 - cosine_distance(a[mask], b[mask]))


def slug(name: str) -> str:
    return name.split("\n")[0].lower().replace(" ", "_").replace("/", "-")


# ── Download one pair ─────────────────────────────────────────────────────────

def process_pair(case_name: str, pair_idx: int, pair: dict, skip_download: bool) -> dict:
    """Download tiles + compute metrics. Returns dict with images and sims."""
    results = {}
    for side in ("a", "b"):
        site = pair[side]
        tag = f"{case_name}_pair{pair_idx+1}_{side}_{slug(site['name'])}"

        path_s2 = OUTPUT_DIR / f"{tag}_s2.png"
        path_ae = OUTPUT_DIR / f"{tag}_ae.png"

        if skip_download and path_s2.exists() and path_ae.exists():
            img_s2 = Image.open(path_s2).convert("RGB")
            img_ae = Image.open(path_ae).convert("RGB")
            print(f"    [{side.upper()}] loaded from cache: {tag}")
        else:
            print(f"    [{side.upper()}] {site['name'].replace(chr(10),' ')}  ({site['lat']}, {site['lon']})")
            print(f"         downloading S2 …")
            img_s2 = download_s2(site["lat"], site["lon"])
            img_s2.save(path_s2)

            print(f"         downloading AlphaEarth …")
            img_ae = download_alphaearth(site["lat"], site["lon"])
            img_ae.save(path_ae)

        print(f"         computing image-level embedding …")
        emb = image_embedding(site["lat"], site["lon"])

        print(f"         computing S2 spectral mean …")
        spec = s2_mean_rgb(site["lat"], site["lon"])

        results[side] = {"img_s2": img_s2, "img_ae": img_ae, "emb": emb, "spec": spec}

    emb_sim  = cosine_sim(results["a"]["emb"],  results["b"]["emb"])
    spec_sim = cosine_sim(results["a"]["spec"], results["b"]["spec"])

    print(f"    ↳ AE image-level sim = {emb_sim:.3f}  |  S2 spectral sim = {spec_sim:.3f}")
    results["emb_sim"]  = emb_sim
    results["spec_sim"] = spec_sim
    return results


# ── Grid figure ───────────────────────────────────────────────────────────────

def render_grid(case_name: str, case_title: str, pairs: list, all_results: list):
    """
    Rows = candidate pairs.  Columns = [Site A S2 | Site A AE | Site B S2 | Site B AE].
    """
    n = len(pairs)
    fig_h = 3.8 * n + 1.2
    fig, axes = plt.subplots(n, 4, figsize=(14, fig_h))
    if n == 1:
        axes = axes[np.newaxis, :]

    col_headers = ["Site A · Sentinel-2", "Site A · AlphaEarth", "Site B · Sentinel-2", "Site B · AlphaEarth"]

    for col, hdr in enumerate(col_headers):
        axes[0, col].set_title(hdr, fontsize=9, pad=6, color="#444444")

    for row, (pair, res) in enumerate(zip(pairs, all_results)):
        for col, (side, key) in enumerate([("a","img_s2"),("a","img_ae"),("b","img_s2"),("b","img_ae")]):
            ax = axes[row, col]
            ax.imshow(res[side][key])
            ax.set_xticks([])
            ax.set_yticks([])
            for sp in ax.spines.values():
                sp.set_linewidth(0.5)
                sp.set_color("#cccccc")

        # Row label on the left
        axes[row, 0].set_ylabel(
            f"pair {row+1}\n{pair['a']['name'].split(chr(10))[0]}\nvs\n{pair['b']['name'].split(chr(10))[0]}",
            fontsize=7.5, rotation=0, labelpad=80, va="center", color="#333333",
        )

        # Similarity annotation below each row
        sim_txt = (
            f"AE img-level sim={res['emb_sim']:.3f}  "
            f"(point emb sim={pair['point_emb_sim']:.3f})   "
            f"S2 spectral sim={res['spec_sim']:.3f}  "
            f"(point spec sim={pair.get('point_spec_sim', float('nan')):.3f})"
        )
        axes[row, 1].text(
            1.02, -0.06, sim_txt,
            transform=axes[row, 1].transAxes,
            fontsize=7.5, color="#555555", ha="center",
        )

    fig.suptitle(case_title, fontsize=13, y=1.01)
    plt.tight_layout(rect=[0.12, 0, 1, 1])
    out = OUTPUT_DIR / f"{case_name}_grid.png"
    fig.savefig(out, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"\n  ✓ Grid saved: {out}")


# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--case", type=int, choices=[1, 2], help="Run only case 1 or case 2")
    parser.add_argument("--skip-download", action="store_true",
                        help="Load cached PNG files instead of re-downloading from GEE")
    args = parser.parse_args()

    run_cases = {
        1: ("case1", "Case 1 · Similar AlphaEarth Embedding, Visually Different Images", CASE1_PAIRS),
        2: ("case2", "Case 2 · Visually Similar Images (optically similar; check embedding divergence)", CASE2_PAIRS),
    }
    if args.case:
        run_cases = {args.case: run_cases[args.case]}

    for case_num, (case_name, case_title, pairs) in run_cases.items():
        print(f"\n{'='*70}")
        print(f"  CASE {case_num}: {case_title}")
        print(f"{'='*70}\n")

        all_results = []
        for i, pair in enumerate(pairs):
            print(f"  Pair {i+1}/{len(pairs)}: {pair['label']}")
            res = process_pair(case_name, i, pair, args.skip_download)
            all_results.append(res)
            print()

        render_grid(case_name, case_title, pairs, all_results)

    print("\n✓ Done.")


if __name__ == "__main__":
    main()
