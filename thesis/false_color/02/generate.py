"""
For a random world location, produce one PNG per radius showing 64 false-color
composites of the AlphaEarth embedding arranged in an 8×8 grid.

False-color bands cycle: (0,1,2), (1,2,3), …, (61,62,63), (62,63,0), (63,0,1).
Caption below each cell: RGB(A00, A01, A02) style.
"""

import ee
import os
import requests
from PIL import Image, ImageDraw, ImageFont
from pathlib import Path

# ── Earth Engine ───────────────────────────────────────────────────────────────
try:
    ee.Initialize(project="gsapp-map")
except Exception:
    ee.Authenticate()
    ee.Initialize(project="gsapp-map")

# ── Config ─────────────────────────────────────────────────────────────────────
YEAR = 2024

# Random but visually interesting location: Mekong Delta, Vietnam
LAT =  10.0341
LON = 105.7878

# (radius_m, label, pixels_for_thumb)  — pixels kept constant so cell size is uniform
RADII = [
    (100,    "100m",  128),
    (1_000,  "1km",   128),
    (5_000,  "5km",   128),
    (10_000, "10km",  128),
]

BANDS = [f"A{i:02d}" for i in range(64)]
BAND_COMBOS = [(i, (i + 1) % 64, (i + 2) % 64) for i in range(64)]   # 64 cyclic triples

CELL_PX   = 128    # each false-color tile
CAPTION_H = 22     # pixels reserved below each tile for band label
GRID_COLS = 8
GRID_ROWS = 8
PADDING   = 6      # gap between cells
VIS_MIN   = -0.3
VIS_MAX   =  0.3

OUTPUT_DIR = "outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)


# ── Font ───────────────────────────────────────────────────────────────────────
def _load_font(size):
    search_paths = [
        str(Path.home() / "Library/Fonts/Roboto-Regular.ttf"),
        "/usr/share/fonts/truetype/roboto/Roboto-Regular.ttf",
        "/System/Library/Fonts/Supplemental/Roboto-Regular.ttf",
        # macOS system sans-serif (always present)
        "/System/Library/Fonts/SFNS.ttf",
        "/System/Library/Fonts/HelveticaNeue.ttc",
        "/System/Library/Fonts/Helvetica.ttc",
        "/System/Library/Fonts/Avenir Next.ttc",
        # Linux fallbacks
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
    ]
    for p in search_paths:
        if os.path.exists(p):
            return ImageFont.truetype(p, size)
    return ImageFont.load_default()


# ── GEE helpers ────────────────────────────────────────────────────────────────
def get_region(lat, lon, radius_m):
    return ee.Geometry.Point(lon, lat).buffer(radius_m).bounds()


def download_tile(embed_img, region, b0, b1, b2, pixels, filepath):
    """Download a 3-band false-color tile via getThumbURL; skip if already cached."""
    if os.path.exists(filepath):
        return True
    url = embed_img.select([BANDS[b0], BANDS[b1], BANDS[b2]]).getThumbURL({
        "region":     region,
        "dimensions": f"{pixels}x{pixels}",
        "format":     "png",
        "min":        VIS_MIN,
        "max":        VIS_MAX,
    })
    try:
        resp = requests.get(url, timeout=120)
        if resp.status_code == 200:
            with open(filepath, "wb") as f:
                f.write(resp.content)
            return True
        print(f"    HTTP {resp.status_code}")
    except Exception as e:
        print(f"    {e}")
    return False


# ── Grid compositor ─────────────────────────────────────────────────────────────
def make_grid(tile_paths, output_path, font, font_title, font_sub,
              loc_name, lat, lon, radius_label, year):
    cell_h   = CELL_PX + CAPTION_H
    total_w  = PADDING + GRID_COLS * (CELL_PX + PADDING)
    grid_h   = PADDING + GRID_ROWS * (cell_h  + PADDING)

    # ── Measure caption height ─────────────────────────────────────────────────
    _td = ImageDraw.Draw(Image.new("RGB", (4, 4)))
    def _meas(txt, f):
        b = _td.textbbox((0, 0), txt, font=f)
        return b[2] - b[0], b[3] - b[1]

    CAP_PAD        = 18
    GAP_AFTER_TITLE = 8
    LINE_GAP        = 5
    _, th = _meas("Ag", font_title)
    _, sh = _meas("Ag", font_sub)
    sub_lines = [
        f"{lat:.5f}\u00b0N, {lon:.5f}\u00b0E",
        f"AlphaEarth Foundations {year}",
        f"10m/pixel  \u00b7  {radius_label} \u00d7 {radius_label}",
    ]
    cap_h = CAP_PAD + th + GAP_AFTER_TITLE + len(sub_lines) * (sh + LINE_GAP) + CAP_PAD

    total_h = grid_h + cap_h
    canvas = Image.new("RGB", (total_w, total_h), (255, 255, 255))
    draw   = ImageDraw.Draw(canvas)

    for idx in range(64):
        row = idx // GRID_COLS
        col = idx % GRID_COLS
        x = PADDING + col * (CELL_PX + PADDING)
        y = PADDING + row * (cell_h  + PADDING)

        # Tile image
        if os.path.exists(tile_paths[idx]):
            tile = Image.open(tile_paths[idx]).convert("RGB").resize(
                (CELL_PX, CELL_PX), Image.LANCZOS
            )
            canvas.paste(tile, (x, y))
        else:
            # Grey placeholder
            canvas.paste(Image.new("RGB", (CELL_PX, CELL_PX), (200, 200, 200)), (x, y))

        # Caption
        b0, b1, b2 = BAND_COMBOS[idx]
        caption = f"{BANDS[b0]}, {BANDS[b1]}, {BANDS[b2]}"
        bbox = draw.textbbox((0, 0), caption, font=font)
        tw   = bbox[2] - bbox[0]
        draw.text(
            (x + (CELL_PX - tw) // 2, y + CELL_PX + 3),
            caption, font=font, fill=(30, 30, 30),
        )

    # ── Draw caption ──────────────────────────────────────────────────────────
    FG  = (30, 30, 30)
    FG2 = (120, 120, 120)
    cy = grid_h + CAP_PAD
    tw, _ = _meas(loc_name, font_title)
    draw.text(((total_w - tw) // 2, cy), loc_name, font=font_title, fill=FG)
    cy += th + GAP_AFTER_TITLE
    for line in sub_lines:
        tw, _ = _meas(line, font_sub)
        draw.text(((total_w - tw) // 2, cy), line, font=font_sub, fill=FG2)
        cy += sh + LINE_GAP

    canvas.save(output_path, dpi=(150, 150))
    print(f"  ✓ Saved grid → {output_path}")


# ── Square grid (no bottom caption block) ─────────────────────────────────────
def make_square_grid(tile_paths, output_path, cell_px=None):
    """8×8 grid with per-cell band captions; outer margin equal on all four sides,
    canvas is a perfect square.  Font and caption height scale with cell_px.

    Key insight: rows are taller than cols are wide (by caption_h per row).
    To balance: solve for GAP_X so total content width == total content height:
      8*cell_px + 7*GAP_X  =  8*(cell_px+caption_h) + 7*GAP_Y
      → GAP_X = (8*caption_h + 7*GAP_Y) / 7
    With equal outer margin M on all four sides the canvas is then exactly square.
    """
    if cell_px is None:
        cell_px = CELL_PX

    scale     = cell_px / CELL_PX
    caption_h = round(CAPTION_H * scale)          # scale caption area with cell
    font_size = max(9, round(9 * scale))
    g_font    = _load_font(font_size)

    GAP_Y = round(PADDING * scale)                # vertical gap between rows (px)
    GAP_X = (8 * caption_h + 7 * GAP_Y) / 7      # solve for equal H/V margin
    M     = round(PADDING * scale)                # equal outer margin on all four sides

    size = round(8 * cell_px + 7 * GAP_X) + 2 * M

    canvas = Image.new("RGB", (size, size), (255, 255, 255))
    draw   = ImageDraw.Draw(canvas)

    for idx in range(64):
        row = idx // GRID_COLS
        col = idx % GRID_COLS
        x = M + round(col * (cell_px + GAP_X))
        y = M + row * (cell_px + caption_h + GAP_Y)

        if os.path.exists(tile_paths[idx]):
            tile = Image.open(tile_paths[idx]).convert("RGB").resize(
                (cell_px, cell_px), Image.LANCZOS
            )
            canvas.paste(tile, (x, y))
        else:
            canvas.paste(Image.new("RGB", (cell_px, cell_px), (200, 200, 200)), (x, y))

        b0, b1, b2 = BAND_COMBOS[idx]
        caption = f"{BANDS[b0]}, {BANDS[b1]}, {BANDS[b2]}"
        bbox = draw.textbbox((0, 0), caption, font=g_font)
        tw   = bbox[2] - bbox[0]
        draw.text(
            (x + (cell_px - tw) // 2, y + cell_px + round(3 * scale)),
            caption, font=g_font, fill=(30, 30, 30),
        )

    canvas.save(output_path, dpi=(150, 150))
    print(f"  ✓ Saved square grid → {output_path}")


# ── Main ───────────────────────────────────────────────────────────────────────
embed_col = ee.ImageCollection("GOOGLE/SATELLITE_EMBEDDING/V1/ANNUAL")
embed_img = (
    embed_col
    .filterDate(f"{YEAR}-01-01", f"{YEAR + 1}-01-01")
    .mosaic()
)

font       = _load_font(9)
font_title = _load_font(20)
font_sub   = _load_font(14)

print(f"Location: {LAT}, {LON}  (Mekong Delta, Vietnam)\n")

for radius_m, label, pixels in RADII:
    print(f"── Radius {label} ──────────────────────────────────────")
    region    = get_region(LAT, LON, radius_m)
    tile_dir  = os.path.join(OUTPUT_DIR, label)
    os.makedirs(tile_dir, exist_ok=True)

    tile_paths = []
    for i, (b0, b1, b2) in enumerate(BAND_COMBOS):
        fp = os.path.join(tile_dir, f"combo_{i:02d}.png")
        tile_paths.append(fp)
        print(f"  [{i + 1:02d}/64] RGB({BANDS[b0]}, {BANDS[b1]}, {BANDS[b2]})", end=" … ")
        ok = download_tile(embed_img, region, b0, b1, b2, pixels, fp)
        print("✓" if ok else "✗ FAILED")

    grid_path = os.path.join(OUTPUT_DIR, f"false_color_{label}.png")
    make_grid(tile_paths, grid_path, font, font_title, font_sub,
              "Mekong Delta, Vietnam", LAT, LON, label, YEAR)

print("\nDone.")

# ── Atlas outputs ──────────────────────────────────────────────────────────────
# Combo 62 and 63 at all radii + 100 km, plus a full 8×8 square grid.
# All atlas tiles are downloaded at ATLAS_PIXELS resolution (higher quality).
# Individual combo PNGs are saved at full download resolution (no resize).
# Square grids display cells at ATLAS_GRID_CELL_PX.
# No caption block on any image; captions go to atlas/captions.txt.

ATLAS_DIR          = os.path.join(OUTPUT_DIR, "atlas")
ATLAS_COMBOS       = [62, 63]
ATLAS_PIXELS       = 512   # download resolution for all atlas tiles
ATLAS_GRID_CELL_PX = 256   # per-cell display size in the square grid
ATLAS_RADII        = [(r, lbl) for r, lbl, _ in RADII] + [(100_000, "100km")]

os.makedirs(ATLAS_DIR, exist_ok=True)

caption_lines = []

print("\n── Atlas ──────────────────────────────────────────────────────────────")
for radius_m, label in ATLAS_RADII:
    print(f"\n  Radius {label}")
    region         = get_region(LAT, LON, radius_m)
    atlas_tile_dir = os.path.join(OUTPUT_DIR, label + "_atlas")
    os.makedirs(atlas_tile_dir, exist_ok=True)

    # Download all 64 tiles at atlas resolution (cached per atlas tile dir)
    all_tile_paths = []
    for i, (b0, b1, b2) in enumerate(BAND_COMBOS):
        fp = os.path.join(atlas_tile_dir, f"combo_{i:02d}.png")
        all_tile_paths.append(fp)
        if not os.path.exists(fp):
            print(f"    [{i + 1:02d}/64] RGB({BANDS[b0]}, {BANDS[b1]}, {BANDS[b2]})", end=" … ")
            ok = download_tile(embed_img, region, b0, b1, b2, ATLAS_PIXELS, fp)
            print("✓" if ok else "✗ FAILED")
        else:
            print(f"    [{i + 1:02d}/64] cached")

    # ── Individual combo tiles — full download resolution, no text ─────────
    for combo_idx in ATLAS_COMBOS:
        src = os.path.join(atlas_tile_dir, f"combo_{combo_idx:02d}.png")
        dst = os.path.join(ATLAS_DIR, f"combo{combo_idx:02d}_{label}.png")
        if os.path.exists(src):
            img = Image.open(src).convert("RGB")
            img.save(dst, dpi=(150, 150))
            print(f"    ✓ {os.path.basename(dst)}  ({img.size[0]}×{img.size[1]} px)")
        else:
            print(f"    ✗ Missing tile: {src}")

        b0, b1, b2 = BAND_COMBOS[combo_idx]
        caption_lines.append(
            f"{os.path.basename(dst)}\n"
            f"{BANDS[b0]}, {BANDS[b1]}, {BANDS[b2]}\n"
            f"Mekong Delta, Vietnam\n"
            f"{LAT:.5f}\u00b0N, {LON:.5f}\u00b0E\n"
            f"AlphaEarth Foundations {YEAR}\n"
            f"{label} \u00d7 {label}"
        )

    # ── Square grid ────────────────────────────────────────────────────────
    grid_out = os.path.join(ATLAS_DIR, f"grid_{label}.png")
    make_square_grid(all_tile_paths, grid_out, cell_px=ATLAS_GRID_CELL_PX)
    caption_lines.append(
        f"{os.path.basename(grid_out)}\n"
        f"8\u00d78 false-color composite grid\n"
        f"Mekong Delta, Vietnam\n"
        f"{LAT:.5f}\u00b0N, {LON:.5f}\u00b0E\n"
        f"AlphaEarth Foundations {YEAR}\n"
        f"{label} \u00d7 {label}"
    )

captions_path = os.path.join(ATLAS_DIR, "captions.txt")
with open(captions_path, "w") as f:
    f.write("\n\n\n".join(caption_lines) + "\n")
print(f"\n✓ Captions written → {captions_path}")
print("Atlas done.")
