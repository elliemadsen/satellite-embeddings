"""
download_images.py  —  Step 2 of 3

Pipeline:
  1. find_pairs.py      → candidates.json          (seconds, no GEE)
  2. download_images.py → output/pairs/*.png + results.json  (seconds, no GEE)
  3. format_output.py   → output/case{1,2}_grid.png

Reads candidates.json, copies cached Sentinel-2 and AlphaEarth tiles from
distance_experiment_web/tiles/ into output/ with pipeline naming, computes
image-level cosine similarities from pixel data, and writes results.json.

No Earth Engine calls — all data comes from the pre-downloaded tile cache.

Usage:
  python download_images.py           # process all pairs
  python download_images.py --case 1  # only case 1
"""

import argparse
import json
import sys
from pathlib import Path
from shutil import copy2

import numpy as np
from PIL import Image

# ── Paths ─────────────────────────────────────────────────────────────────────
SCRIPT_DIR      = Path(__file__).parent
WEB_DATA_DIR    = SCRIPT_DIR / ".." / ".." / "distance_experiment_web"
TILES_DIR       = WEB_DATA_DIR / "tiles"

CANDIDATES_FILE = SCRIPT_DIR / "candidates.json"
RESULTS_FILE    = SCRIPT_DIR / "results.json"
OUTPUT_DIR      = SCRIPT_DIR / "output"
PAIRS_DIR       = OUTPUT_DIR / "pairs"
OUTPUT_DIR.mkdir(exist_ok=True)
PAIRS_DIR.mkdir(exist_ok=True)

DISPLAY_PX = 256  # upscale cached 64×64 tiles for display
# AE_BANDS   = "A30_A31_A32"  # which 3-band combo to show for AlphaEarth tiles
AE_BANDS   = "A61_A62_A63"  # which 3-band combo to show for AlphaEarth tiles

# ── Load candidates ───────────────────────────────────────────────────────────
if not CANDIDATES_FILE.exists():
    sys.exit(f"ERROR: {CANDIDATES_FILE} not found.  Run find_pairs.py first.")

with open(CANDIDATES_FILE) as _f:
    _candidates = json.load(_f)


def _pair_from_candidate(p: dict) -> dict:
    cont_a = p.get("continent_a") or "Unknown"
    cont_b = p.get("continent_b") or "Unknown"
    return {
        "label": (
            f"{p['lc_a']}  ({p['region_a']})\n"
            f"vs  {p['lc_b']}  ({p['region_b']})"
        ),
        "point_emb_sim":  p["emb_sim"],
        "point_spec_sim": p["spec_sim"],
        "a": {
            "name":     f"{p['region_a']}\n{cont_a}",
            "lat":      p["lat_a"],
            "lon":      p["lon_a"],
            "note":     f"cls {p['class_a']} · {p['lc_a']}",
            "tile_dir": p["tile_dir_a"],
        },
        "b": {
            "name":     f"{p['region_b']}\n{cont_b}",
            "lat":      p["lat_b"],
            "lon":      p["lon_b"],
            "note":     f"cls {p['class_b']} · {p['lc_b']}",
            "tile_dir": p["tile_dir_b"],
        },
    }


CASE1_PAIRS = [_pair_from_candidate(p) for p in _candidates["case1"]]
CASE2_PAIRS = [_pair_from_candidate(p) for p in _candidates["case2"]]


# ── Helpers ───────────────────────────────────────────────────────────────────

def slug(name: str) -> str:
    return name.split("\n")[0].lower().replace(" ", "_").replace("/", "-")


def _tile_index(tile_dir: str) -> str:
    """Extract the 4-digit index from 'tiles/0042' → '0042'."""
    return Path(tile_dir).name


def _load_and_upscale(src: Path, dst: Path) -> Image.Image:
    """Load a cached 64×64 tile, upscale to DISPLAY_PX, and save to output."""
    img = Image.open(src).convert("RGB")
    if img.size != (DISPLAY_PX, DISPLAY_PX):
        img = img.resize((DISPLAY_PX, DISPLAY_PX), Image.NEAREST)
    img.save(dst)
    return img


def _mean_rgb(img: Image.Image) -> np.ndarray:
    """Mean RGB as a float array (3,)."""
    return np.array(img, dtype=np.float32).mean(axis=(0, 1))


def cosine_sim(a: np.ndarray, b: np.ndarray) -> float | None:
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na < 1e-9 or nb < 1e-9:
        return None
    return round(float(np.dot(a, b) / (na * nb)), 6)


# ── Process one pair ──────────────────────────────────────────────────────────

def process_pair(case_name: str, pair_idx: int, pair: dict) -> dict:
    """
    Copy cached tiles into output/ and compute image-level spectral similarity
    from the S2 pixel data.
    """
    specs: dict[str, np.ndarray] = {}
    tags:  dict[str, str] = {}

    for side in ("a", "b"):
        site     = pair[side]
        tile_idx = _tile_index(site["tile_dir"])
        tag      = f"{case_name}_pair{pair_idx + 1}_{side}_{slug(site['name'])}"
        tags[side] = tag

        # Source paths in the tile cache
        src_s2 = TILES_DIR / tile_idx / f"{tile_idx}_satellite.png"
        src_ae = TILES_DIR / tile_idx / f"{tile_idx}_{AE_BANDS}.png"

        # Destination paths in output/pairs/
        dst_s2 = PAIRS_DIR / f"{tag}_s2.png"
        dst_ae = PAIRS_DIR / f"{tag}_ae.png"

        if not src_s2.exists():
            print(f"    ⚠ S2 tile not found: {src_s2}")
        if not src_ae.exists():
            print(f"    ⚠ AE tile not found: {src_ae}")

        print(f"    [{side.upper()}] {site['name'].replace(chr(10), ' ')}"
              f"  ({site['lat']:.3f}, {site['lon']:.3f})  ← {site['tile_dir']}")

        s2_img = _load_and_upscale(src_s2, dst_s2)
        _load_and_upscale(src_ae, dst_ae)

        specs[side] = _mean_rgb(s2_img)

    img_spec_sim = cosine_sim(specs["a"], specs["b"])

    # Embedding similarity: use the point-level value from find_pairs.py
    # (the cached tiles cover the same 640 m footprint that was used to
    #  compute the embedding similarity in step 1).
    img_emb_sim = pair["point_emb_sim"]

    print(f"    ↳ emb_sim={img_emb_sim:.4f}  |  S2 spectral sim={img_spec_sim}")
    return {
        "pair_idx":     pair_idx,
        "tag_a":        tags["a"],
        "tag_b":        tags["b"],
        "img_emb_sim":  img_emb_sim,
        "img_spec_sim": img_spec_sim,
    }


# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Copy cached S2 + AE tiles to output/ and compute metrics."
    )
    parser.add_argument("--case", type=int, choices=[1, 2],
                        help="Process only case 1 or case 2")
    args = parser.parse_args()

    run_cases = {
        1: ("case1", CASE1_PAIRS),
        2: ("case2", CASE2_PAIRS),
    }
    if args.case:
        run_cases = {args.case: run_cases[args.case]}

    # Load existing results so a --case run doesn't wipe the other case
    results: dict = {"case1": [], "case2": []}
    if RESULTS_FILE.exists():
        with open(RESULTS_FILE) as f:
            results.update(json.load(f))

    for case_num, (case_name, pairs) in run_cases.items():
        print(f"\n{'='*70}")
        print(f"  CASE {case_num}  ({len(pairs)} pairs)")
        print(f"{'='*70}\n")
        case_results = []
        for i, pair in enumerate(pairs):
            print(f"  Pair {i+1}/{len(pairs)}: {pair['label'].replace(chr(10), ' ')}")
            m = process_pair(case_name, i, pair)
            case_results.append(m)
            print()
        results[case_name] = case_results

    with open(RESULTS_FILE, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n✓ Metrics written to {RESULTS_FILE}")
    print("  Run format_output.py to generate grid figures.\n")


if __name__ == "__main__":
    main()
