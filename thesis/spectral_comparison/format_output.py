"""
format_output.py  —  Step 3 of 3

Pipeline:
  1. find_pairs.py      → candidates.json
  2. download_images.py → output/pairs/*.png  +  results.json
  3. format_output.py   → output/case{1,2}_grid.png

Reads results.json (image-level metrics + tag names), loads cached PNG tiles
from output/, and renders a grid figure per case.

Usage:
  python format_output.py           # render both cases
  python format_output.py --case 1  # only case 1
"""

import argparse
import json
import sys
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

matplotlib.rcParams["font.family"] = "Roboto"

# ── Paths ─────────────────────────────────────────────────────────────────────
CANDIDATES_FILE = Path(__file__).parent / "candidates.json"
RESULTS_FILE    = Path(__file__).parent / "results.json"
OUTPUT_DIR      = Path(__file__).parent / "output"
PAIRS_DIR       = OUTPUT_DIR / "pairs"

for _required in (CANDIDATES_FILE, RESULTS_FILE):
    if not _required.exists():
        sys.exit(f"ERROR: {_required} not found.  Run the previous pipeline step first.")

with open(CANDIDATES_FILE) as _f:
    _candidates = json.load(_f)

with open(RESULTS_FILE) as _f:
    _results = json.load(_f)

# ── Hard-coded pair indices (0-based into candidates.json lists) ──────────────
# Update these to cherry-pick which pairs appear in each grid.
CASE1_INDICES = [21, 27, 11, 2]
CASE2_INDICES = [10, 11, 3, 20]


# ── Build pair dicts from candidates.json ────────────────────────────────────

def _pair_from_candidate(p: dict) -> dict:
    cont_a = p.get("continent_a") or "Unknown"
    cont_b = p.get("continent_b") or "Unknown"
    return {
        "point_emb_sim":  p["emb_sim"],
        "point_spec_sim": p["spec_sim"],
        "a": {
            "region":    p["region_a"],
            "continent": cont_a,
            "lat":       p["lat_a"],
            "lon":       p["lon_a"],
            "lc":        p["lc_a"],
            "cls":       p["class_a"],
        },
        "b": {
            "region":    p["region_b"],
            "continent": cont_b,
            "lat":       p["lat_b"],
            "lon":       p["lon_b"],
            "lc":        p["lc_b"],
            "cls":       p["class_b"],
        },
    }


ALL_CASE1 = [_pair_from_candidate(p) for p in _candidates["case1"]]
ALL_CASE2 = [_pair_from_candidate(p) for p in _candidates["case2"]]

# ── Colours ───────────────────────────────────────────────────────────────────
CLR_TITLE   = "#222222"
CLR_HEADER  = "#555555"
CLR_LABEL   = "#333333"
CLR_META    = "#999999"
CLR_SIM     = "#555555"
CLR_BORDER  = "#cccccc"
BG_COLOR    = "#ffffff"


# ── Grid rendering ────────────────────────────────────────────────────────────

def _draw_vertical_text(canvas, text, x_center, y_center, font, fill):
    """Draw text rotated 90° CCW (reading bottom→top), centered at (x_center, y_center)."""
    from PIL import ImageDraw as _ID
    # Render text onto a temporary image, then rotate
    tmp = Image.new("RGBA", (800, 50), (0, 0, 0, 0))
    td  = _ID.Draw(tmp)
    bbox = td.textbbox((0, 0), text, font=font)
    tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
    tmp2 = Image.new("RGBA", (tw + 4, th + 4), (0, 0, 0, 0))
    _ID.Draw(tmp2).text((2, 2), text, fill=fill, font=font)
    rotated = tmp2.rotate(90, expand=True, resample=Image.BICUBIC)
    px = x_center - rotated.width // 2
    py = y_center - rotated.height // 2
    canvas.paste(rotated, (px, py), rotated)


def render_grid(case_name: str, case_title: str, indices: list[int],
                all_pairs: list, all_results: list):
    """
    Rows = selected pairs.
    Columns (L→R): Site A S2 | Site B S2 | Site A AE | Site B AE
    Vertical similarity labels to the left of each pair of columns.
    Below each image: location info.
    """
    pairs   = [all_pairs[i]   for i in indices]
    results = [all_results[i] for i in indices]
    n = len(pairs)
    if n == 0:
        print(f"  {case_name}: no pairs to render.")
        return

    from PIL import ImageDraw, ImageFont
    TITLE_FONT_SIZE = 40
    try:
        font_header = ImageFont.truetype("Roboto", 14)
        font_label  = ImageFont.truetype("Roboto", 11)
        font_sim    = ImageFont.truetype("Roboto", 12)
        font_title  = ImageFont.truetype("Roboto", TITLE_FONT_SIZE)
    except OSError:
        font_header = ImageFont.load_default()
        font_label  = font_header
        font_sim    = font_header
        font_title  = font_header

    # ── Dimensions (pixels) ───────────────────────────────────────────────
    TILE      = 256           # tile image size
    COL_GAP   = 36            # gap between adjacent columns
    GROUP_GAP = 56            # wider gap between S2 pair and AE pair
    SIM_W     = 24            # width for vertical similarity label
    MARGIN_L  = 24            # left canvas margin
    MARGIN_R  = 16            # right canvas margin
    TOP_H     = 132           # more space for title + column headers (prevents overlap)
    TEXT_H    = 56            # space below image for location text (3 lines)
    ROW_GAP   = 40            # vertical gap between rows
    BOT_PAD   = 10

    # X layout:  MARGIN | SIM_W | gap4 | TileA_S2 | COL_GAP | TileB_S2
    #            | GROUP_GAP | SIM_W | gap4 | TileA_AE | COL_GAP | TileB_AE | MARGIN
    gap4 = 6  # small gap between sim label and first tile
    x_spec_label = MARGIN_L                             # center of spectral sim label
    x0 = MARGIN_L + SIM_W + gap4                        # Site A S2
    x1 = x0 + TILE + COL_GAP                            # Site B S2
    x_emb_label = x1 + TILE + GROUP_GAP                 # center of embedding sim label
    x2 = x_emb_label + SIM_W + gap4                     # Site A AE
    x3 = x2 + TILE + COL_GAP                            # Site B AE
    col_x = [x0, x1, x2, x3]

    row_h   = TILE + TEXT_H + ROW_GAP
    total_w = x3 + TILE + MARGIN_R
    total_h = TOP_H + n * row_h + BOT_PAD

    canvas = Image.new("RGB", (total_w, total_h), BG_COLOR)
    draw   = ImageDraw.Draw(canvas)

    # ── Title ─────────────────────────────────────────────────────────────
    # Left part aligned to first S2 image, right part aligned to end of last AE image
    title_left, title_right = case_title
    title_y = 10
    draw.text((x0, title_y), title_left, fill=CLR_TITLE, font=font_title)
    rb = draw.textbbox((0, 0), title_right, font=font_title)
    rw = rb[2] - rb[0]
    draw.text((x3 + TILE - rw, title_y), title_right, fill=CLR_TITLE, font=font_title)

    # ── Column headers ────────────────────────────────────────────────────
    # Place headers below the title, with a fixed gap
    HEADER_GAP = 40
    header_y = title_y + font_title.size + HEADER_GAP

    # Compute where the image grid should start (always below header row)
    GRID_GAP = 16  # vertical gap between header row and first image row
    grid_start_y = header_y + font_header.size + GRID_GAP
    headers = [
        ("Sentinel-2", "Site A"),
        ("Sentinel-2", "Site B"),
        ("AlphaEarth", "Site A"),
        ("AlphaEarth", "Site B"),
    ]
    for i, (left, right) in enumerate(headers):
        cx = col_x[i]
        draw.text((cx, header_y), left, fill=CLR_HEADER, font=font_header)
        rb = draw.textbbox((0, 0), right, font=font_header)
        rw = rb[2] - rb[0]
        draw.text((cx + TILE - rw, header_y), right, fill=CLR_HEADER, font=font_header)

    # ── Rows ──────────────────────────────────────────────────────────────
    for row, (pair, res) in enumerate(zip(pairs, results)):
        y_img = grid_start_y + row * row_h

        tag_a = res["tag_a"]
        tag_b = res["tag_b"]

        imgs = [
            Image.open(PAIRS_DIR / f"{tag_a}_s2.png").convert("RGB").resize((TILE, TILE), Image.NEAREST),
            Image.open(PAIRS_DIR / f"{tag_b}_s2.png").convert("RGB").resize((TILE, TILE), Image.NEAREST),
            Image.open(PAIRS_DIR / f"{tag_a}_ae.png").convert("RGB").resize((TILE, TILE), Image.NEAREST),
            Image.open(PAIRS_DIR / f"{tag_b}_ae.png").convert("RGB").resize((TILE, TILE), Image.NEAREST),
        ]

        # Site info per column: [A, B, A, B]
        sites = [pair["a"], pair["b"], pair["a"], pair["b"]]

        for ci in range(4):
            cx = col_x[ci]
            # Paste image
            canvas.paste(imgs[ci], (cx, y_img))
            # Border
            draw.rectangle([cx, y_img, cx + TILE - 1, y_img + TILE - 1],
                           outline=CLR_BORDER, width=1)

            # Text below image: region, classification, lat/lon
            s = sites[ci]
            ty = y_img + TILE + 4
            draw.text((cx, ty),      s["region"],               fill=CLR_LABEL, font=font_label)
            draw.text((cx, ty + 14), s["lc"],                   fill=CLR_META,  font=font_label)
            draw.text((cx, ty + 28), f"({s['lat']:.2f}, {s['lon']:.2f})",
                      fill=CLR_META, font=font_label)

        # Vertical similarity labels
        y_mid = y_img + TILE // 2
        _draw_vertical_text(
            canvas,
            f"spectral similarity: {pair['point_spec_sim']:.3f}",
            x_spec_label + SIM_W // 2, y_mid,
            font_sim, CLR_SIM,
        )
        _draw_vertical_text(
            canvas,
            f"embedding similarity: {pair['point_emb_sim']:.3f}",
            x_emb_label + SIM_W // 2, y_mid,
            font_sim, CLR_SIM,
        )

    out = OUTPUT_DIR / f"{case_name}_grid.png"
    canvas.save(out)
    print(f"  ✓ {out}  ({total_w}×{total_h})")


# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Render grid figures from cached PNG tiles."
    )
    parser.add_argument("--case", type=int, choices=[1, 2],
                        help="Render only case 1 or case 2")
    args = parser.parse_args()

    run_cases = {
        1: ("case1",
            ("Different Satellite Image", "Similar Satellite Embedding"),
            CASE1_INDICES, ALL_CASE1),
        2: ("case2",
            ("Similar Satellite Image", "Different Satellite Embedding"),
            CASE2_INDICES, ALL_CASE2),
    }
    if args.case:
        run_cases = {args.case: run_cases[args.case]}

    for case_num, (case_name, case_title, indices, all_pairs) in run_cases.items():
        all_results = _results.get(case_name, [])
        if not all_results:
            print(f"  Case {case_num}: no entries in results.json — run download_images.py first.")
            continue
        print(f"\nRendering {case_name}: {case_title[0]}, {case_title[1]}")
        render_grid(case_name, case_title, indices, all_pairs, all_results)

    print("\n✓ Done.")


if __name__ == "__main__":
    main()
