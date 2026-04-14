import ee
import os
import requests
from PIL import Image, ImageDraw, ImageFont
from pathlib import Path

# Initialize Earth Engine
try:
    ee.Initialize(project="gsapp-map")
except Exception:
    ee.Authenticate()
    ee.Initialize(project="gsapp-map")

# Configuration
YEAR = 2024
PIXELS = 512        # Output image size in pixels
SCALE = 20          # Meters per pixel (Sentinel-2 native: 10m, using 20 for wider coverage)
OUTPUT_DIR = "img"

os.makedirs(OUTPUT_DIR, exist_ok=True)

LOCATIONS = {
    "amazon":   {"lat": -5.8461,  "lon": -70.5832},
    "congolian": {"lat": -0.5579,  "lon":  23.4778},
}

satellite_dataset = ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")


def load_font(size):
    """Load Roboto if available, otherwise fall back to PIL default."""
    import subprocess
    search_paths = [
        "/usr/share/fonts/truetype/roboto/Roboto-Regular.ttf",
        "/System/Library/Fonts/Supplemental/Roboto-Regular.ttf",
        str(Path.home() / "Library/Fonts/Roboto-Regular.ttf"),
    ]
    for p in search_paths:
        if os.path.exists(p):
            return ImageFont.truetype(p, size)
    try:
        result = subprocess.run(
            ["fc-list", ":family=Roboto", "--format=%{file}\n"],
            capture_output=True, text=True
        )
        for line in result.stdout.splitlines():
            line = line.strip()
            if line and os.path.exists(line):
                return ImageFont.truetype(line, size)
    except Exception:
        pass
    return ImageFont.load_default()


def get_region(lat, lon, pixels, scale):
    point = ee.Geometry.Point(lon, lat)
    return point.buffer(scale * pixels / 2).bounds()


def download_rgb(name, lat, lon):
    filepath = os.path.join(OUTPUT_DIR, f"{name}_rgb.png")
    if os.path.exists(filepath):
        print(f"  ↩ Skipping RGB (already exists): {filepath}")
        return True

    region = get_region(lat, lon, PIXELS, SCALE)

    # Try with cloud filter first, then without
    for cloud_threshold in [20, 50, 100]:
        try:
            collection = (
                satellite_dataset
                .filterDate(f"{YEAR}-01-01", f"{YEAR + 1}-01-01")
                .filterBounds(region)
            )
            if cloud_threshold < 100:
                collection = collection.filter(
                    ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", cloud_threshold)
                )
            img = collection.median()
            band_names = img.bandNames().getInfo()
            if not band_names:
                continue
            rgb = img.select(["B4", "B3", "B2"]).visualize(min=0, max=3000)
            url = rgb.getThumbURL({
                "region": region,
                "dimensions": f"{PIXELS}x{PIXELS}",
                "format": "png",
            })
            response = requests.get(url, timeout=120)
            if response.status_code == 200:
                with open(filepath, "wb") as f:
                    f.write(response.content)
                print(f"  ✓ Saved RGB: {filepath} (cloud threshold: {cloud_threshold}%)")
                return True
            else:
                print(f"  ✗ HTTP {response.status_code} for {name} (cloud threshold: {cloud_threshold}%)")
        except Exception as e:
            print(f"  ✗ Error for {name} (cloud threshold: {cloud_threshold}%): {e}")

    print(f"  ✗ Failed to download RGB for {name}")
    return False


def download_false_color(name, lat, lon):
    """Download false-color composite (NIR/Red/Green) to highlight vegetation."""
    filepath = os.path.join(OUTPUT_DIR, f"{name}_false_color.png")
    if os.path.exists(filepath):
        print(f"  ↩ Skipping false color (already exists): {filepath}")
        return True

    region = get_region(lat, lon, PIXELS, SCALE)

    for cloud_threshold in [20, 50, 100]:
        try:
            collection = (
                satellite_dataset
                .filterDate(f"{YEAR}-01-01", f"{YEAR + 1}-01-01")
                .filterBounds(region)
            )
            if cloud_threshold < 100:
                collection = collection.filter(
                    ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", cloud_threshold)
                )
            img = collection.median()
            band_names = img.bandNames().getInfo()
            if not band_names:
                continue
            # B8=NIR, B4=Red, B3=Green — classic vegetation false color
            false_color = img.select(["B8", "B4", "B3"]).visualize(min=0, max=4000)
            url = false_color.getThumbURL({
                "region": region,
                "dimensions": f"{PIXELS}x{PIXELS}",
                "format": "png",
            })
            response = requests.get(url, timeout=120)
            if response.status_code == 200:
                with open(filepath, "wb") as f:
                    f.write(response.content)
                print(f"  ✓ Saved false color: {filepath} (cloud threshold: {cloud_threshold}%)")
                return True
        except Exception as e:
            print(f"  ✗ Error for {name} false color (cloud threshold: {cloud_threshold}%): {e}")

    print(f"  ✗ Failed to download false color for {name}")
    return False


def download_ndvi(name, lat, lon):
    """Download NDVI image (Normalized Difference Vegetation Index)."""
    filepath = os.path.join(OUTPUT_DIR, f"{name}_ndvi.png")
    if os.path.exists(filepath):
        print(f"  ↩ Skipping NDVI (already exists): {filepath}")
        return True

    region = get_region(lat, lon, PIXELS, SCALE)

    for cloud_threshold in [20, 50, 100]:
        try:
            collection = (
                satellite_dataset
                .filterDate(f"{YEAR}-01-01", f"{YEAR + 1}-01-01")
                .filterBounds(region)
            )
            if cloud_threshold < 100:
                collection = collection.filter(
                    ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", cloud_threshold)
                )
            img = collection.median()
            band_names = img.bandNames().getInfo()
            if not band_names:
                continue
            ndvi = img.normalizedDifference(["B8", "B4"]).rename("NDVI")
            ndvi_vis = ndvi.visualize(min=-0.2, max=0.8, palette=["white", "yellow", "green", "darkgreen"])
            url = ndvi_vis.getThumbURL({
                "region": region,
                "dimensions": f"{PIXELS}x{PIXELS}",
                "format": "png",
            })
            response = requests.get(url, timeout=120)
            if response.status_code == 200:
                with open(filepath, "wb") as f:
                    f.write(response.content)
                print(f"  ✓ Saved NDVI: {filepath} (cloud threshold: {cloud_threshold}%)")
                return True
        except Exception as e:
            print(f"  ✗ Error for {name} NDVI (cloud threshold: {cloud_threshold}%): {e}")

    print(f"  ✗ Failed to download NDVI for {name}")
    return False


def download_embedding(name, lat, lon):
    """Download satellite embedding bands as an RGB image."""
    filepath = os.path.join(OUTPUT_DIR, f"{name}_embedding.png")
    if os.path.exists(filepath):
        print(f"  ↩ Skipping embedding (already exists): {filepath}")
        return True

    region = get_region(lat, lon, PIXELS, SCALE)

    embedding_dataset = ee.ImageCollection("GOOGLE/SATELLITE_EMBEDDING/V1/ANNUAL")
    embedding_filtered = embedding_dataset.filterDate(f"{YEAR}-01-01", f"{YEAR + 1}-01-01")

    try:
        embed_img = (
            embedding_filtered
            .filterBounds(region)
            .mosaic()
            .select(["A05", "A06", "A07"])
        )
        url = embed_img.getThumbURL({
            "region": region,
            "dimensions": f"{PIXELS}x{PIXELS}",
            "format": "png",
            "min": -0.3,
            "max": 0.3,
        })
        filepath = os.path.join(OUTPUT_DIR, f"{name}_embedding.png")
        response = requests.get(url, timeout=120)
        if response.status_code == 200:
            with open(filepath, "wb") as f:
                f.write(response.content)
            print(f"  ✓ Saved embedding: {filepath}")
            return True
        else:
            print(f"  ✗ HTTP {response.status_code} for {name} embedding")
    except Exception as e:
        print(f"  ✗ Error for {name} embedding: {e}")

    print(f"  ✗ Failed to download embedding for {name}")
    return False


def compose_grid(locations, output_dir, pixels, output_path="rainforests_grid.png"):
    """Compose a 2-row × 3-column grid: rows=locations, cols=image types."""
    columns = [
        ("rgb",         "True Color"),
        ("false_color", "False Color"),
        ("ndvi",        "NDVI"),
        ("embedding",   "Embedding"),
    ]
    row_labels = {
        "amazon":    "Amazon Rainforest",
        "congolian": "Congolian Rainforest",
    }

    PADDING     = 24   # px between cells
    LABEL_H     = 52   # height reserved for column header
    ROW_LABEL_W = 240  # width reserved for row label on the left
    FONT_SIZE   = 26
    BG_COLOR    = (255, 255, 255)
    TEXT_COLOR  = (30, 30, 30)

    n_rows = len(locations)
    n_cols = len(columns)

    total_w = ROW_LABEL_W + n_cols * pixels + (n_cols + 1) * PADDING
    total_h = LABEL_H   + n_rows * pixels + (n_rows + 1) * PADDING

    canvas = Image.new("RGB", (total_w, total_h), BG_COLOR)
    draw   = ImageDraw.Draw(canvas)

    font_bold   = load_font(FONT_SIZE)
    font_normal = load_font(FONT_SIZE - 4)

    def centered_text(draw, text, font, box_x, box_y, box_w, box_h, color):
        bbox = draw.textbbox((0, 0), text, font=font)
        tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
        draw.text(
            (box_x + (box_w - tw) // 2, box_y + (box_h - th) // 2),
            text, font=font, fill=color
        )

    # Column headers
    for ci, (_, col_label) in enumerate(columns):
        cell_x = ROW_LABEL_W + PADDING + ci * (pixels + PADDING)
        centered_text(draw, col_label, font_bold,
                      cell_x, 0, pixels, LABEL_H, TEXT_COLOR)

    # Rows
    for ri, (name, _) in enumerate(locations.items()):
        cell_y = LABEL_H + PADDING + ri * (pixels + PADDING)

        # Row label: title + lat/lon subtitle
        coords = locations[name]
        subtitle = f"({coords['lat']}, {coords['lon']})"
        title_text = row_labels[name]
        t_bbox = draw.textbbox((0, 0), title_text, font=font_normal)
        s_bbox = draw.textbbox((0, 0), subtitle, font=load_font(FONT_SIZE - 8))
        t_h = t_bbox[3] - t_bbox[1]
        s_h = s_bbox[3] - s_bbox[1]
        gap = 8
        block_h = t_h + gap + s_h
        block_y = cell_y + (pixels - block_h) // 2
        # Title
        t_w = t_bbox[2] - t_bbox[0]
        draw.text((ROW_LABEL_W // 2 - t_w // 2, block_y), title_text,
                  font=font_normal, fill=TEXT_COLOR)
        # Subtitle
        s_w = s_bbox[2] - s_bbox[0]
        draw.text((ROW_LABEL_W // 2 - s_w // 2, block_y + t_h + gap), subtitle,
                  font=load_font(FONT_SIZE - 8), fill=(120, 120, 120))

        for ci, (suffix, _) in enumerate(columns):
            cell_x = ROW_LABEL_W + PADDING + ci * (pixels + PADDING)
            img_path = os.path.join(output_dir, f"{name}_{suffix}.png")
            if os.path.exists(img_path):
                tile = Image.open(img_path).convert("RGB").resize((pixels, pixels))
                canvas.paste(tile, (cell_x, cell_y))
            else:
                # Grey placeholder
                placeholder = Image.new("RGB", (pixels, pixels), (200, 200, 200))
                canvas.paste(placeholder, (cell_x, cell_y))
                draw.text((cell_x + 10, cell_y + pixels // 2), "missing",
                          font=font_normal, fill=(120, 120, 120))

    canvas.save(output_path)
    print(f"\n✓ Grid saved to: {output_path}")


def compose_true_color(locations, output_dir, pixels, output_path="rainforests_true_color.png"):
    """Side-by-side true color images with per-image metadata captions below."""
    PADDING      = 40
    CAPTION_GAP  = 16   # gap between image bottom and first caption line
    BG_COLOR     = (255, 255, 255)
    TEXT_COLOR   = (30, 30, 30)
    SUB_COLOR    = (100, 100, 100)
    FONT_TITLE   = 22
    FONT_SUB     = 16
    FONT_SMALL   = 14
    LINE_GAP     = 8
    TITLE_AFTER  = 8    # extra space below title line

    n_cols = len(locations)

    font_title = load_font(FONT_TITLE)
    font_sub   = load_font(FONT_SUB)
    font_small = load_font(FONT_SMALL)

    LOCATION_META = {
        "amazon":    "Amazon Rainforest",
        "congolian": "Congolian Rainforest",
    }

    # Measure caption block height (same structure for every column)
    _tmp_draw = ImageDraw.Draw(Image.new("RGB", (1, 1)))
    _sample_name, _sample_coords = next(iter(locations.items()))
    _lines_spec = [
        (LOCATION_META[_sample_name], font_title),
        (f"({_sample_coords['lat']}, {_sample_coords['lon']})", font_sub),
        ("Copernicus Sentinel-2",        font_small),
        ("True Color RGB: (B4, B3, B2)", font_small),
        (f"{YEAR} median composite",     font_small),
    ]
    caption_h = LINE_GAP
    for li, (text, font) in enumerate(_lines_spec):
        bbox = _tmp_draw.textbbox((0, 0), text, font=font)
        caption_h += (bbox[3] - bbox[1]) + LINE_GAP
        if li == 0:
            caption_h += TITLE_AFTER

    total_w = PADDING + n_cols * pixels + (n_cols - 1) * PADDING + PADDING
    total_h = PADDING + pixels + CAPTION_GAP + caption_h + PADDING

    canvas = Image.new("RGB", (total_w, total_h), BG_COLOR)
    draw   = ImageDraw.Draw(canvas)

    for ci, (name, coords) in enumerate(locations.items()):
        cell_x = PADDING + ci * (pixels + PADDING)
        lat, lon = coords["lat"], coords["lon"]

        # Image at top
        img_path = os.path.join(output_dir, f"{name}_rgb.png")
        if os.path.exists(img_path):
            tile = Image.open(img_path).convert("RGB").resize((pixels, pixels))
        else:
            tile = Image.new("RGB", (pixels, pixels), (200, 200, 200))
            ImageDraw.Draw(tile).text((10, pixels // 2), "missing", font=font_small, fill=(120, 120, 120))
        canvas.paste(tile, (cell_x, PADDING))

        # Caption below
        lines = [
            (LOCATION_META[name],            font_title, TEXT_COLOR),
            (f"({lat}, {lon})",              font_sub,   SUB_COLOR),
            ("Copernicus Sentinel-2",         font_small, SUB_COLOR),
            ("True Color RGB: (B4, B3, B2)", font_small, SUB_COLOR),
            (f"{YEAR} median composite",      font_small, SUB_COLOR),
        ]
        line_y = PADDING + pixels + CAPTION_GAP
        for li, (text, font, color) in enumerate(lines):
            bbox = draw.textbbox((0, 0), text, font=font)
            tw   = bbox[2] - bbox[0]
            draw.text((cell_x + (pixels - tw) // 2, line_y), text, font=font, fill=color)
            line_y += (bbox[3] - bbox[1]) + LINE_GAP
            if li == 0:
                line_y += TITLE_AFTER

    canvas.save(output_path)
    print(f"✓ True color composite saved to: {output_path}")


def compose_analysis(locations, output_dir, pixels, output_path="rainforests_analysis.png"):
    """2-row x 3-col grid (false color / NDVI / embedding) with rich column headers."""
    columns = [
        ("false_color", [
            "Color Infrared (CIR)",
            "Copernicus Sentinel-2",
            "False Color RGB: (B8, B4, B3)",
            f"{YEAR} median composite",
        ]),
        ("ndvi", [
            "Normalized Difference Vegetation Index (NDVI)",
            "Copernicus Sentinel-2",
            "False Color RGB: (B8 \u2013 B4) / (B8 + B4)",
            f"{YEAR} median composite",
        ]),
        ("embedding", [
            "Satellite Embedding",
            "AlphaEarth Foundations V1",
            "False Color RGB: (A50, A51, A52)",
            f"{YEAR}",
        ]),
    ]
    row_labels = {
        "amazon":    "Amazon Rainforest",
        "congolian": "Congolian Rainforest",
    }

    PADDING       = 24
    HEADER_BOTTOM = 14   # padding below last header line and image top
    ROW_LABEL_W   = 240
    BG_COLOR      = (255, 255, 255)
    TEXT_COLOR    = (30, 30, 30)
    SUB_COLOR     = (100, 100, 100)
    FONT_HDR      = 20
    FONT_SUB      = 14
    FONT_ROW      = 18
    FONT_COORD    = 13
    LINE_GAP      = 6
    TITLE_AFTER   = 8    # extra space after bold title lines

    n_rows = len(locations)
    n_cols = len(columns)

    font_hdr   = load_font(FONT_HDR)
    font_sub   = load_font(FONT_SUB)
    font_row   = load_font(FONT_ROW)
    font_coord = load_font(FONT_COORD)

    # Compute HEADER_H tightly from actual text
    _tmp_draw = ImageDraw.Draw(Image.new("RGB", (1, 1)))
    max_header_h = 0
    for _, header_lines in columns:
        lh = LINE_GAP
        for li, text in enumerate(header_lines):
            font = font_hdr if li == 0 else font_sub
            bbox = _tmp_draw.textbbox((0, 0), text, font=font)
            lh  += (bbox[3] - bbox[1]) + LINE_GAP
            if li == 0:
                lh += TITLE_AFTER
        max_header_h = max(max_header_h, lh)
    HEADER_H = max_header_h + HEADER_BOTTOM

    total_w = ROW_LABEL_W + n_cols * pixels + (n_cols + 1) * PADDING
    total_h = HEADER_H    + n_rows * pixels + (n_rows + 1) * PADDING

    canvas = Image.new("RGB", (total_w, total_h), BG_COLOR)
    draw   = ImageDraw.Draw(canvas)

    # Column headers (multiline)
    for ci, (_, header_lines) in enumerate(columns):
        cell_x = ROW_LABEL_W + PADDING + ci * (pixels + PADDING)
        line_y = LINE_GAP
        for li, text in enumerate(header_lines):
            font  = font_hdr if li == 0 else font_sub
            color = TEXT_COLOR if li == 0 else SUB_COLOR
            bbox  = draw.textbbox((0, 0), text, font=font)
            tw    = bbox[2] - bbox[0]
            draw.text((cell_x + (pixels - tw) // 2, line_y), text, font=font, fill=color)
            line_y += (bbox[3] - bbox[1]) + LINE_GAP
            if li == 0:
                line_y += TITLE_AFTER

    # Rows
    for ri, (name, _) in enumerate(locations.items()):
        cell_y = HEADER_H + PADDING + ri * (pixels + PADDING)
        coords = locations[name]
        lat, lon = coords["lat"], coords["lon"]

        # Row label: name + coords (with extra gap after name)
        label_lines = [
            (row_labels[name], font_row,   TEXT_COLOR),
            (f"({lat}, {lon})", font_coord, SUB_COLOR),
        ]
        lh_total = 0
        for li, (t, f, _) in enumerate(label_lines):
            bbox = _tmp_draw.textbbox((0, 0), t, font=f)
            lh_total += (bbox[3] - bbox[1]) + LINE_GAP
            if li == 0:
                lh_total += TITLE_AFTER
        ly = cell_y + (pixels - lh_total) // 2
        for li, (text, font, color) in enumerate(label_lines):
            bbox = draw.textbbox((0, 0), text, font=font)
            tw   = bbox[2] - bbox[0]
            draw.text((ROW_LABEL_W // 2 - tw // 2, ly), text, font=font, fill=color)
            ly  += (bbox[3] - bbox[1]) + LINE_GAP
            if li == 0:
                ly += TITLE_AFTER

        for ci, (suffix, _) in enumerate(columns):
            cell_x   = ROW_LABEL_W + PADDING + ci * (pixels + PADDING)
            img_path = os.path.join(output_dir, f"{name}_{suffix}.png")
            if os.path.exists(img_path):
                tile = Image.open(img_path).convert("RGB").resize((pixels, pixels))
            else:
                tile = Image.new("RGB", (pixels, pixels), (200, 200, 200))
                ImageDraw.Draw(tile).text((10, pixels // 2), "missing", font=font_sub, fill=(120, 120, 120))
            canvas.paste(tile, (cell_x, cell_y))

    canvas.save(output_path)
    print(f"✓ Analysis composite saved to: {output_path}")


if __name__ == "__main__":
    for name, coords in LOCATIONS.items():
        lat, lon = coords["lat"], coords["lon"]
        print(f"\n{'='*50}")
        print(f"Processing: {name.upper()}  (lat={lat}, lon={lon})")
        print(f"{'='*50}")
        download_rgb(name, lat, lon)
        download_false_color(name, lat, lon)
        download_ndvi(name, lat, lon)
        download_embedding(name, lat, lon)

    compose_grid(LOCATIONS, OUTPUT_DIR, PIXELS, output_path="rainforests_grid.png")
    compose_true_color(LOCATIONS, OUTPUT_DIR, PIXELS, output_path="rainforests_true_color.png")
    compose_analysis(LOCATIONS, OUTPUT_DIR, PIXELS, output_path="rainforests_analysis.png")
    print("Done.")
