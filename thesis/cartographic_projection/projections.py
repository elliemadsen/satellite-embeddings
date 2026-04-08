import os
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import numpy as np
from PIL import Image, ImageDraw, ImageFont

# Projections to visualize — top 4: conformal, two compromises, equal-area
# Mercator is capped at ±80° latitude (poles are at infinity in Mercator)
PROJECTIONS = [
    (ccrs.Mercator(min_latitude=-80, max_latitude=85), "Mercator"),
    (ccrs.Robinson(),                                   "Robinson"),
    (ccrs.EqualEarth(),                                 "Equal Earth"),
    (ccrs.Mollweide(),                                  "Mollweide"),
]

# Output size
WIDTH, HEIGHT = 1200, 600

# Roboto font path
ROBOTO_PATH = os.path.expanduser("~/Library/Fonts/Roboto.ttf")

# Helper to render and save a projection at its natural aspect ratio
def render_projection(proj, name, land_white=True, out_path=None):
    # Render large at natural aspect ratio — let bbox_inches='tight' decide size
    fig = plt.figure(figsize=(14, 10), dpi=150)
    ax = fig.add_subplot(1, 1, 1, projection=proj)
    ax.set_global()
    land_color = "white" if land_white else "black"
    sea_color  = "black" if land_white else "white"
    ax.set_facecolor(sea_color)
    fig.patch.set_facecolor(sea_color)
    ax.coastlines(color=land_color, linewidth=0.7)
    ax.add_feature(ccrs.cartopy.feature.LAND,  facecolor=land_color, edgecolor="none")
    ax.add_feature(ccrs.cartopy.feature.OCEAN, facecolor=sea_color,  edgecolor="none")
    ax.set_axis_off()
    plt.savefig(out_path, bbox_inches='tight', pad_inches=0.05,
                facecolor=sea_color)
    plt.close(fig)


def fit_into_cell(img: Image.Image, cell_w: int, cell_h: int,
                  bg: str = "white") -> Image.Image:
    """Scale img to fit within cell_w×cell_h preserving aspect ratio, center on bg."""
    img.thumbnail((cell_w, cell_h), Image.LANCZOS)
    cell = Image.new("RGB", (cell_w, cell_h), bg)
    ox = (cell_w - img.width)  // 2
    oy = (cell_h - img.height) // 2
    cell.paste(img, (ox, oy))
    return cell

# Render only black land on white sea
img_files = []
for proj, name in PROJECTIONS:
    out_path = f"{name.replace(' ', '_').lower()}_black.png"
    render_projection(proj, name, land_white=False, out_path=out_path)
    img_files.append((out_path, name))

# Combine into a 2×2 grid with titles
COLS = 2
TITLE_H = 100      # vertical space reserved below each image for the title
GAP = 20          # gap between cells
canvas_w = COLS * WIDTH + (COLS - 1) * GAP
canvas_h = 2 * (HEIGHT + TITLE_H) + GAP
canvas = Image.new("RGB", (canvas_w, canvas_h), "white")
TITLE_FONT_SIZE = 36
try:
    font = ImageFont.truetype(ROBOTO_PATH, TITLE_FONT_SIZE)
except OSError as e:
    print(f"  ⚠ Could not load Roboto ({e}), using default font")
    font = ImageFont.load_default()
draw = ImageDraw.Draw(canvas)

for i, (out_path, name) in enumerate(img_files):
    col = i % COLS
    row = i // COLS
    x = col * (WIDTH + GAP)
    y = row * (HEIGHT + TITLE_H + GAP)
    img = fit_into_cell(Image.open(out_path), WIDTH, HEIGHT)
    canvas.paste(img, (x, y))
    canvas.paste(img, (x, y))
    title = name
    bbox = draw.textbbox((0, 0), title, font=font)
    tw = bbox[2] - bbox[0]
    draw.text((x + WIDTH // 2 - tw // 2, y + HEIGHT + 12), title, fill="black", font=font)

canvas.save("all_projections.png")
print("Done. Individual PNGs and all_projections.png created.")
