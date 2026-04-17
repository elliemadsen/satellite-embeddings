import os
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

# Load Roboto
_roboto = str(os.path.expanduser("~/Library/Fonts/Roboto.ttf"))
if os.path.exists(_roboto):
    fm.fontManager.addfont(_roboto)
    plt.rcParams["font.family"] = fm.FontProperties(fname=_roboto).get_name()

# -------------------------------
# Data
# -------------------------------
years = [
    2002,
    2003,
    2004,
    2005,
    2006,
    2007,
    2008,
    2009,
    2010,
    2011,
    2012,
    2013,
    2014,
    2015,
    2016,
    2017,
    2018,
    2019,
    2020,
    2021,
    2022,
    2023,
    2024,
    2025,
]

counts = [
    4,
    8,
    11,
    7,
    5,
    4,
    4,
    13,
    9,
    12,
    10,
    13,
    14,
    19,
    29,
    25,
    27,
    27,
    38,
    42,
    83,
    128,
    263,
    358,
]

# -------------------------------
# Plot styling
# -------------------------------
plt.style.use("default")
fig, ax = plt.subplots(figsize=(12, 6), dpi=100)

# White background
fig.patch.set_facecolor("white")
ax.set_facecolor("white")

# Bars: black fill, dark-gray edge
ax.bar(years, counts, color="#111111", edgecolor="#444444")

# Title
ax.set_title(
    "Remote Sensing Foundation Model Papers Published",
    color="#111111",
    fontsize=18,
    pad=20,
)

# Axis labels
ax.set_xlabel("Year", color="#444444", fontsize=14)
ax.set_ylabel("Number of Publications", color="#444444", fontsize=14)

# Axis ticks
ax.tick_params(colors="#444444", labelsize=10)

# Spines: only bottom + left, dark gray
for spine in ["top", "right"]:
    ax.spines[spine].set_visible(False)
for spine in ["bottom", "left"]:
    ax.spines[spine].set_color("#444444")
    ax.spines[spine].set_linewidth(0.8)

# Data source annotation
fig.text(0.99, 0.01, "Data Source: webofscience.com",
         ha="right", va="bottom", fontsize=8, color="#888888",
         transform=fig.transFigure)

# Tight layout for nicer spacing
plt.tight_layout()

# -------------------------------
# Export as SVG
# -------------------------------
output_file = "remote_sensing_papers.svg"
plt.savefig(output_file, format="svg", transparent=False, facecolor="white")
print(f"SVG saved as: {output_file}")

output_png = "remote_sensing_papers.png"
plt.savefig(output_png, format="png", dpi=200, transparent=False, facecolor="white")
print(f"PNG saved as: {output_png}")