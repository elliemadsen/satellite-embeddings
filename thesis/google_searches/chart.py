import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib import font_manager
from scipy.ndimage import gaussian_filter1d
import os
from pathlib import Path

# ── Font ───────────────────────────────────────────────────────────────────────
font_paths = [
    str(Path.home() / "Library/Fonts/Roboto-Regular.ttf"),
    "/System/Library/Fonts/SFNS.ttf",
    "/System/Library/Fonts/HelveticaNeue.ttc",
]
for p in font_paths:
    if os.path.exists(p):
        font_manager.fontManager.addfont(p)
        prop = font_manager.FontProperties(fname=p)
        plt.rcParams["font.family"] = prop.get_name()
        break

plt.rcParams.update({
    "font.size":        11,
    "axes.spines.top":  False,
    "axes.spines.right":False,
    "axes.spines.left": False,
    "axes.grid":        False,
    "xtick.bottom":     False,
    "ytick.left":       False,
    "xtick.color":      "#888888",
    "ytick.color":      "#888888",
    "axes.labelcolor":  "#888888",
    "text.color":       "#333333",
})

# ── Data ───────────────────────────────────────────────────────────────────────
df = pd.read_csv("time_series_Worldwide_20210410-1258_20260410-1258.csv",
                 parse_dates=["Time"])
df = df.set_index("Time")
df = df[df.index >= "2022-01-01"]

COLORS = ["#7aa4ff", "#9fdc4f", "#ffa361", "#fff56f"]

# ── Plot ───────────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(9, 4.5))

for col, color in zip(df.columns, COLORS):
    smoothed = gaussian_filter1d(df[col].astype(float), sigma=1.5)
    ax.plot(df.index, smoothed, color=color, linewidth=2, label=col)

# ── Axes decoration ────────────────────────────────────────────────────────────
ax.set_ylim(0, 110)
ax.yaxis.set_ticks([0, 25, 50, 75, 100])
ax.yaxis.set_ticklabels(["0", "25", "50", "75", "100"])

# One tick per year + the last data point month
ax.xaxis.set_major_locator(mdates.YearLocator())
ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))

ax.set_ylabel("Relative Search interest", fontsize=10, labelpad=8, color="#888888")
ax.legend(loc="upper left", frameon=False, fontsize=10,
          labelcolor="#333333", handlelength=1.4, handletextpad=0.6)

ax.set_title("Google Search Trends — Worldwide", fontsize=13, fontweight="normal",
             loc="left", pad=12, color="#222222")

caption = (
    "Search interest over a specific time period, displayed on a relative scale from 0 to 100, "
    "where 100 signifies the peak interest for the time period of the chart. "
)
fig.text(0.5, -0.04, caption, ha="center", va="top", fontsize=8,
         color="#888888", wrap=True,
         transform=fig.transFigure,
         multialignment="center")

fig.tight_layout()
fig.savefig("chart.png", dpi=200, bbox_inches="tight")
print("✓ Saved chart.png")
