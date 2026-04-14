"""
Embedding-Space World Map
=========================
Countries grouped by AlphaEarth embedding-space similarity, not geography.

Outputs
-------
  output/embedding_world_map_voronoi.png   - weighted Voronoi territories
  output/embedding_world_map_borders.png   - scaled actual country outlines
"""

import json
import base64
import math
from collections import defaultdict, Counter as _Counter
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.patheffects as pe
from matplotlib.patches import PathPatch, Ellipse
from matplotlib.path import Path as MplPath
from matplotlib.colors import to_rgba
import matplotlib.font_manager as _fm

from shapely.geometry import (
    MultiPolygon, Polygon, Point, GeometryCollection
)
from shapely.affinity import translate, scale as affine_scale
from shapely.ops import unary_union

import pyproj
import geopandas as gpd
from scipy.spatial import Voronoi, ConvexHull
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import umap

# -- Font
_fnames = {f.name for f in _fm.fontManager.ttflist}
plt.rcParams["font.family"] = "Roboto" if "Roboto" in _fnames else "sans-serif"
plt.rcParams["font.weight"] = "regular"

# -- Paths
_here      = Path(__file__).parent
DATA_FILES = [
    _here.parent.parent / "dimension-reduction/data/20000_sampled_classified_embeddings.geojson",
    _here.parent.parent / "dimension-reduction/data/10000_sampled_classified_embeddings.geojson",
]
SHAPEFILE  = _here.parent.parent / "dimension-reduction/data/raw/ne_110m_admin_0_countries/ne_110m_admin_0_countries.shp"
OUT_DIR    = _here / "output"
OUT_DIR.mkdir(exist_ok=True)

# -- Config
MIN_SAMPLES      = 5
N_CLUSTERS       = 8
RANDOM_STATE     = 42
LOCAL_UMAP_KW    = dict(n_components=2, n_neighbors=6,  min_dist=0.30, random_state=RANDOM_STATE)
GLOBAL_UMAP_KW   = dict(n_components=2, n_neighbors=10, min_dist=0.40, random_state=RANDOM_STATE)
CONTINENT_SPREAD = 3.0
CLUSTER_SCALE    = 12.0
MIN_RADIUS   = 0.12
MAX_RADIUS   = 0.70
ISLAND_FRAC  = 0.04

CLUSTER_COLORS = [
    "#f5c09a",  # pale terracotta / peach
    "#b3d9f0",  # pale steel blue
    "#bde6aa",  # pale sage green
    "#f0afc3",  # pale rose
    "#d3c5ee",  # pale violet
    "#a3d9cb",  # pale mint / teal
    "#f5e49a",  # pale golden
    "#bccee0",  # pale slate blue
]

BG_COLOR     = "#ffffff"
OCEAN_COLOR  = "#ffffff"   # white oval interior
GRID_COLOR   = "#888888"
BORDER_COLOR = "#333333"
TEXT_COLOR   = "#111111"
SUB_COLOR    = "#555555"
COUNTRY_LABEL_SIZE = 10.0



# =============================================================================
#  HELPERS
# =============================================================================

def keep_significant(geom, min_frac=ISLAND_FRAC):
    if geom is None or geom.is_empty:
        return None
    if geom.geom_type == "Polygon":
        return geom
    polys = [g for g in geom.geoms if g.geom_type == "Polygon"]
    if not polys:
        return None
    mx   = max(p.area for p in polys)
    kept = [p for p in polys if p.area >= min_frac * mx]
    if not kept:
        return polys[0]
    return MultiPolygon(kept) if len(kept) > 1 else kept[0]


def normalize_geom(geom):
    if geom is None or geom.is_empty:
        return None
    cx, cy = geom.centroid.x, geom.centroid.y
    g      = translate(geom, -cx, -cy)
    minx, miny, maxx, maxy = g.bounds
    ext = max(maxx - minx, maxy - miny)
    if ext < 1e-9:
        return None
    return affine_scale(g, 1.0 / ext, 1.0 / ext, origin=(0, 0))


def geom_to_patch(geom, **kw):
    verts, codes = [], []

    def add_ring(ring):
        c = list(ring.coords)
        if len(c) < 2:
            return
        verts.extend(c)
        codes.append(MplPath.MOVETO)
        codes.extend([MplPath.LINETO] * (len(c) - 2))
        codes.append(MplPath.CLOSEPOLY)

    def add_poly(p):
        add_ring(p.exterior)
        for interior in p.interiors:
            add_ring(interior)

    if geom.geom_type == "Polygon":
        add_poly(geom)
    elif geom.geom_type in ("MultiPolygon", "GeometryCollection"):
        for g in geom.geoms:
            if g.geom_type == "Polygon":
                add_poly(g)
    else:
        return None
    if not verts:
        return None
    return PathPatch(MplPath(np.array(verts, dtype=float), codes), **kw)


def finite_voronoi_cells(points, clip_poly):
    """Finite Voronoi cells clipped to clip_poly using 16 far mirror points."""
    n = len(points)
    if n == 0:
        return []
    if n == 1:
        return [clip_poly]
    center = points.mean(0)
    ext    = max(
        np.ptp(points, axis=0).max() + 1.0,
        clip_poly.bounds[2] - clip_poly.bounds[0],
        clip_poly.bounds[3] - clip_poly.bounds[1],
    ) * 20.0
    angles  = np.linspace(0, 2 * math.pi, 16, endpoint=False)
    mirrors = center + ext * np.c_[np.cos(angles), np.sin(angles)]
    all_pts = np.vstack([points, mirrors])
    vor     = Voronoi(all_pts)
    cells   = []
    for i in range(n):
        region = vor.regions[vor.point_region[i]]
        if not region or -1 in region:
            cells.append(clip_poly)
            continue
        poly = Polygon(vor.vertices[region])
        if not poly.is_valid:
            poly = poly.buffer(0)
        clipped = poly.intersection(clip_poly)
        cells.append(None if clipped.is_empty else clipped)
    return cells


def voronoi_cells_weighted(positions, area_weights, clip_poly):
    """
    Approximate area-proportional Voronoi:
    each country seeds 1-7 representative points (log-scaled by area);
    cells from the same country are unioned.
    """
    n = len(positions)
    if n == 0:
        return []
    if n == 1:
        return [clip_poly]
    w      = np.array(area_weights, dtype=float)
    w_log  = np.log1p(w)
    w_norm = (w_log - w_log.min()) / (w_log.max() - w_log.min() + 1e-9)
    reps   = np.round(1 + w_norm * 6).astype(int)   # 1..7
    base_r    = 0.08
    rep_pos   = []
    rep_owner = []
    for i, (pos, k) in enumerate(zip(positions, reps)):
        if k == 1:
            rep_pos.append(pos)
            rep_owner.append(i)
        else:
            r    = base_r * math.sqrt(k)
            angs = np.linspace(0, 2 * math.pi, k, endpoint=False)
            for ang in angs:
                rep_pos.append(pos + r * np.array([math.cos(ang), math.sin(ang)]))
                rep_owner.append(i)
    rep_pos   = np.array(rep_pos)
    raw_cells = finite_voronoi_cells(rep_pos, clip_poly)
    by_country = [[] for _ in range(n)]
    for j, owner_idx in enumerate(rep_owner):
        if raw_cells[j] is not None and not raw_cells[j].is_empty:
            by_country[owner_idx].append(raw_cells[j])
    merged = []
    for cell_list in by_country:
        if not cell_list:
            merged.append(None)
        elif len(cell_list) == 1:
            merged.append(cell_list[0])
        else:
            try:
                merged.append(unary_union(cell_list))
            except Exception:
                merged.append(cell_list[0])
    return merged


def jitter_poly(geom, amplitude=0.04, density=0.09, seed=0):
    """Add organic noise to polygon boundary for a naturalistic border look."""
    rng = np.random.default_rng(abs(int(seed)) % (2 ** 31))

    def jitter_ring(coords):
        pts = np.array(coords[:-1])  # drop closing duplicate
        n   = len(pts)
        if n < 3:
            return list(coords)
        dense = []
        for i in range(n):
            dense.append(pts[i])
            nxt     = pts[(i + 1) % n]
            seg_len = np.linalg.norm(nxt - pts[i])
            n_mid   = max(0, int(seg_len / density) - 1)
            for k in range(1, n_mid + 1):
                dense.append(pts[i] + (k / (n_mid + 1)) * (nxt - pts[i]))
        dense = np.array(dense)
        noise = rng.normal(0, amplitude, dense.shape)
        dense += noise
        # 3-point wrap-around smoothing
        d2 = np.roll(dense, -1, axis=0)
        d0 = np.roll(dense,  1, axis=0)
        dense = (d0 + 2 * dense + d2) / 4.0
        return [tuple(p) for p in dense] + [tuple(dense[0])]

    def jitter_polygon(p):
        ext = jitter_ring(list(p.exterior.coords))
        try:
            result = Polygon(ext)
            if not result.is_valid:
                result = result.buffer(0)
            return result if (result.is_valid and not result.is_empty) else p
        except Exception:
            return p

    try:
        if geom.geom_type == "Polygon":
            result = jitter_polygon(geom)
        elif geom.geom_type == "MultiPolygon":
            parts  = [jitter_polygon(p) for p in geom.geoms]
            result = MultiPolygon([p for p in parts
                                   if p.is_valid and not p.is_empty])
        else:
            return geom
        if result is None or result.is_empty:
            return geom
        return result
    except Exception:
        return geom


# =============================================================================
#  NatEarth projection — precomputed once, shared by draw_natearth_map + globe
# =============================================================================
_NE_PROJ  = "+proj=natearth +lon_0=0 +datum=WGS84 +units=m"
_NE_TRANS = pyproj.Transformer.from_crs("EPSG:4326", _NE_PROJ, always_xy=True)
# Boundary: right meridian → flat top → left meridian → flat bottom
_ne_sl    = np.linspace(-89.9, 89.9, 400)
_ne_bxR, _ne_byR = _NE_TRANS.transform(np.full_like(_ne_sl,  179.99), _ne_sl)
_ne_bxL, _ne_byL = _NE_TRANS.transform(np.full_like(_ne_sl, -179.99), _ne_sl)
_NE_FLAT_Y = float(abs(_NE_TRANS.transform(0.0, 89.9)[1]))
_NE_BND_X  = np.concatenate([_ne_bxR, np.linspace(_ne_bxR[-1], _ne_bxL[-1], 60),
                               _ne_bxL[::-1], np.linspace(_ne_bxL[0], _ne_bxR[0], 60)])
_NE_BND_Y  = np.concatenate([_ne_byR, np.full(60, _NE_FLAT_Y),
                               _ne_byL[::-1], np.full(60, -_NE_FLAT_Y)])
_NE_RX = float(_NE_BND_X.max())   # projection half-width (at equator)
_NE_RY = _NE_FLAT_Y               # projection half-height (at pole)
# Graticule — every 15° lat and lon, 400 pts each
_NE_GRAT_LAT = [_NE_TRANS.transform(np.linspace(-179.9, 179.9, 400), np.full(400, float(g)))
                for g in np.arange(-75, 76, 15)]
_NE_GRAT_LON = [_NE_TRANS.transform(np.full(400, float(g)), np.linspace(-89.9, 89.9, 400))
                for g in np.arange(-180, 181, 15)]


def draw_oval_map(ax, cx, cy, rx, ry, n_lat=9, n_lon=11):
    """Robinson-style oval: ocean fill, graticule, white outside mask, border."""
    ax.add_patch(Ellipse((cx, cy), 2 * rx, 2 * ry,
                         facecolor=OCEAN_COLOR, edgecolor="none", zorder=0.5))
    for t in np.linspace(-0.88, 0.88, n_lat):
        y_line = cy + t * ry
        x_ext  = rx * math.sqrt(max(0.0, 1.0 - t ** 2)) * 0.999
        ax.plot([cx - x_ext, cx + x_ext], [y_line, y_line],
                color=GRID_COLOR, lw=0.4, zorder=0.6, solid_capstyle="round")
    for frac in np.linspace(-0.88, 0.88, n_lon):
        y_pts = np.linspace(-ry * 0.999, ry * 0.999, 160)
        t_pts = y_pts / ry
        x_pts = cx + frac * rx * np.sqrt(np.maximum(0.0, 1.0 - t_pts ** 2))
        ax.plot(x_pts, cy + y_pts,
                color=GRID_COLOR, lw=0.4, zorder=0.6, solid_capstyle="round")
    # White mask outside oval (outer rect with oval hole)
    theta   = np.linspace(0, 2 * math.pi, 300)
    ov_x    = cx + rx * np.cos(theta)
    ov_y    = cy + ry * np.sin(theta)
    pad     = max(rx, ry) * 0.35
    rect_v  = np.array([[cx-rx-pad, cy-ry-pad], [cx+rx+pad, cy-ry-pad],
                         [cx+rx+pad, cy+ry+pad], [cx-rx-pad, cy+ry+pad],
                         [cx-rx-pad, cy-ry-pad]])
    oval_v  = np.c_[ov_x, ov_y][::-1]
    oval_v  = np.vstack([oval_v, oval_v[:1]])
    mask_v  = np.vstack([rect_v, oval_v])
    n_r     = len(rect_v)
    n_o     = len(oval_v)
    mask_c  = (
        [MplPath.MOVETO] + [MplPath.LINETO] * (n_r - 2) + [MplPath.CLOSEPOLY] +
        [MplPath.MOVETO] + [MplPath.LINETO] * (n_o - 2) + [MplPath.CLOSEPOLY]
    )
    ax.add_patch(PathPatch(MplPath(mask_v, mask_c),
                           facecolor=BG_COLOR, edgecolor="none", zorder=8))
    ax.add_patch(Ellipse((cx, cy), 2 * rx, 2 * ry,
                         facecolor="none", edgecolor=BORDER_COLOR,
                         linewidth=0.75, zorder=9))


def draw_natearth_map(ax, cx, cy, rx, ry):
    """
    Identical frame to world_map_globe: uses the same precomputed NE boundary
    and graticule vectors, scaled from projection space to data space.
    """
    def _s(px, py):
        return cx + np.asarray(px) / _NE_RX * rx, cy + np.asarray(py) / _NE_RY * ry

    bx, by = _s(_NE_BND_X, _NE_BND_Y)
    bv = np.c_[bx, by]
    bc = [MplPath.MOVETO] + [MplPath.LINETO] * (len(bx) - 2) + [MplPath.CLOSEPOLY]
    # Ocean fill
    ax.add_patch(PathPatch(MplPath(bv, bc), facecolor=OCEAN_COLOR, edgecolor="none", zorder=0.5))
    # Graticule
    for gx, gy in _NE_GRAT_LAT:
        dx, dy = _s(gx, gy)
        ax.plot(dx, dy, color=GRID_COLOR, lw=0.4, zorder=0.6, solid_capstyle="round")
    for gx, gy in _NE_GRAT_LON:
        dx, dy = _s(gx, gy)
        ax.plot(dx, dy, color=GRID_COLOR, lw=0.4, zorder=0.6, solid_capstyle="round")
    # White mask outside NE shape
    pad    = max(rx, ry) * 0.35
    rect_v = np.array([[cx-rx-pad, cy-ry-pad], [cx+rx+pad, cy-ry-pad],
                        [cx+rx+pad, cy+ry+pad], [cx-rx-pad, cy+ry+pad],
                        [cx-rx-pad, cy-ry-pad]])
    ne_v   = np.c_[bx, by][::-1]
    ne_v   = np.vstack([ne_v, ne_v[:1]])
    mask_v = np.vstack([rect_v, ne_v])
    n_r, n_ne = len(rect_v), len(ne_v)
    mask_c = (
        [MplPath.MOVETO] + [MplPath.LINETO] * (n_r - 2) + [MplPath.CLOSEPOLY] +
        [MplPath.MOVETO] + [MplPath.LINETO] * (n_ne - 2) + [MplPath.CLOSEPOLY]
    )
    ax.add_patch(PathPatch(MplPath(mask_v, mask_c), facecolor=BG_COLOR, edgecolor="none", zorder=8))
    # Border outline
    ax.add_patch(PathPatch(MplPath(bv, bc), facecolor="none", edgecolor=BORDER_COLOR,
                           linewidth=0.75, zorder=9))


def draw_compass(ax, x, y, size=0.35):
    ax.annotate("", xy=(x, y + size), xytext=(x, y),
                arrowprops=dict(arrowstyle="-|>", color="#444", lw=1.1), zorder=11)
    ax.text(x, y + size + 0.10, "N", ha="center", va="bottom",
            fontsize=7, color="#444", fontweight="bold", zorder=11)


def climate_zone(abs_lat):
    if abs_lat < 15:  return "Equatorial"
    if abs_lat < 25:  return "Tropical"
    if abs_lat < 35:  return "Subtropical"
    if abs_lat < 50:  return "Temperate"
    if abs_lat < 62:  return "Continental"
    return "Boreal / Polar"


def est_temp_c(mean_lat):
    return int(round(27.0 - 0.52 * abs(mean_lat)))


# =============================================================================
#  1  LOAD
# =============================================================================

print("Loading data ...")
country_embeddings = defaultdict(list)
country_meta       = {}
country_lats       = defaultdict(list)

for _data_file in DATA_FILES:
    with open(_data_file) as f:
        data = json.load(f)
    for feat in data["features"]:
        p    = feat["properties"]
        iso  = p.get("ISO_A3")
        name = p.get("NAME")
        if not iso or iso == "-99" or not name:
            continue
        raw_emb = p.get("embedding")
        if not raw_emb:
            continue
        try:
            emb = np.array(json.loads(base64.b64decode(raw_emb).decode()), dtype=np.float32)
        except Exception:
            continue
        country_embeddings[iso].append(emb)
        country_lats[iso].append(float(p.get("lat", 0.0)))
        if iso not in country_meta:
            country_meta[iso] = {
                "name":      name,
                "continent": p.get("CONTINENT", ""),
                "subregion": p.get("SUBREGION", ""),
            }


# =============================================================================
#  2  AGGREGATE
# =============================================================================

print("Aggregating ...")
isos, names, mean_embs, mean_lats = [], [], [], []
for iso, embs in country_embeddings.items():
    if len(embs) < MIN_SAMPLES:
        continue
    isos.append(iso)
    names.append(country_meta[iso]["name"])
    mean_embs.append(np.mean(embs, axis=0))
    mean_lats.append(float(np.mean(country_lats[iso])))
mean_embs = np.array(mean_embs)
mean_lats = np.array(mean_lats)
print(f"  {len(isos)} countries retained")


# =============================================================================
#  3  K-MEANS
# =============================================================================

print(f"Clustering into {N_CLUSTERS} groups ...")
scaler   = StandardScaler()
X_scaled = scaler.fit_transform(mean_embs)
km     = KMeans(n_clusters=N_CLUSTERS, random_state=RANDOM_STATE, n_init=20)
labels = km.fit_predict(X_scaled)
for c in range(N_CLUSTERS):
    in_c = [names[i] for i in range(len(names)) if labels[i] == c]
    print(f"  [{c}] {len(in_c):3d}: {', '.join(sorted(in_c)[:6])}{'...' if len(in_c)>6 else ''}")


# =============================================================================
#  4  CLUSTER STATISTICS
# =============================================================================

cluster_stats = {}
for c in range(N_CLUSTERS):
    mask     = np.where(labels == c)[0]
    all_lats = []
    for i in mask:
        all_lats.extend(country_lats[isos[i]])
    mean_lat = float(np.mean(all_lats)) if all_lats else 0.0
    conts    = [country_meta[isos[i]]["continent"] for i in mask]
    dom_cont = _Counter(conts).most_common(1)[0][0] if conts else ""
    cluster_stats[c] = {
        "n":         len(mask),
        "mean_lat":  mean_lat,
        "climate":   climate_zone(abs(mean_lat)),
        "temp":      est_temp_c(mean_lat),
        "continent": dom_cont,
    }


# =============================================================================
#  5  UMAP
# =============================================================================

print("Running global UMAP ...")
reducer_g = umap.UMAP(**GLOBAL_UMAP_KW)
coords_g  = reducer_g.fit_transform(X_scaled)

print("Running per-cluster UMAP ...")
cens   = np.array([coords_g[labels == c].mean(0) for c in range(N_CLUSTERS)])
cx_arr = cens[:, 0]
cy_arr = cens[:, 1]
cx_arr = (cx_arr - cx_arr.min()) / (cx_arr.max() - cx_arr.min() + 1e-9) * CLUSTER_SCALE
cy_arr = (cy_arr - cy_arr.min()) / (cy_arr.max() - cy_arr.min() + 1e-9) * CLUSTER_SCALE
cluster_positions = np.stack([cx_arr, cy_arr], axis=1)

final_coords = np.zeros((len(isos), 2))
for c in range(N_CLUSTERS):
    mask  = np.where(labels == c)[0]
    cpos  = cluster_positions[c]
    if len(mask) == 1:
        final_coords[mask[0]] = cpos
        continue
    n_nbrs = max(2, min(LOCAL_UMAP_KW["n_neighbors"], len(mask) - 1))
    lr     = umap.UMAP(**{**LOCAL_UMAP_KW, "n_neighbors": n_nbrs})
    lc     = lr.fit_transform(X_scaled[mask])
    for d in range(2):
        rng = lc[:, d].max() - lc[:, d].min() + 1e-9
        lc[:, d] = (lc[:, d] - lc[:, d].min()) / rng - 0.5
    final_coords[mask] = cpos + lc * CONTINENT_SPREAD


# =============================================================================
#  6  SHAPEFILE  +  NORMALISED GEOMETRIES
# =============================================================================

print("Loading shapefile ...")
gdf        = gpd.read_file(SHAPEFILE)
geo_lookup = {}
for _, row in gdf.iterrows():
    for col in ("ISO_A3", "SOV_A3", "ADM0_A3"):
        val = str(row.get(col, "")).strip()
        if val and val != "-99" and val not in geo_lookup:
            geo_lookup[val] = row

geo_areas = np.array([
    geo_lookup[iso].geometry.area
    if (iso in geo_lookup
        and geo_lookup[iso].geometry is not None
        and not geo_lookup[iso].geometry.is_empty)
    else 0.0
    for iso in isos
])
sqrt_areas   = np.sqrt(np.maximum(geo_areas, 1e-9))
sa_min, sa_max = sqrt_areas.min(), sqrt_areas.max()
target_radii = (MIN_RADIUS
                + (MAX_RADIUS - MIN_RADIUS)
                * (sqrt_areas - sa_min) / (sa_max - sa_min + 1e-9))

normed_geoms = {}
for iso in isos:
    row = geo_lookup.get(iso)
    if row is None or row.geometry is None or row.geometry.is_empty:
        normed_geoms[iso] = None
        continue
    geom_wgs = keep_significant(row.geometry)
    if geom_wgs is None:
        normed_geoms[iso] = None
        continue
    # Reproject to azimuthal equal-area centered on this country's centroid
    # so shapes appear as viewed from directly above on a globe (no lat/lon stretch).
    _clon = geom_wgs.centroid.x
    _clat = geom_wgs.centroid.y
    _laea = f"+proj=laea +lon_0={_clon:.4f} +lat_0={_clat:.4f} +datum=WGS84 +units=m"
    try:
        _tr  = pyproj.Transformer.from_crs("EPSG:4326", _laea, always_xy=True)
        import shapely.ops as _sops
        geom_ea = _sops.transform(_tr.transform, geom_wgs)
        if geom_ea is None or geom_ea.is_empty or not geom_ea.is_valid:
            geom_ea = geom_ea.buffer(0)
        normed_geoms[iso] = normalize_geom(geom_ea)
    except Exception:
        normed_geoms[iso] = normalize_geom(geom_wgs)


# =============================================================================
#  7  SHARED LAYOUT
# =============================================================================

PAD_X   = 1.8
PAD_Y   = 1.6
_data_xlim = (final_coords[:, 0].min() - PAD_X, final_coords[:, 0].max() + PAD_X)
_data_ylim = (final_coords[:, 1].min() - PAD_Y, final_coords[:, 1].max() + PAD_Y)

# Horizontally stretched oval
data_cx = (_data_xlim[0] + _data_xlim[1]) / 2
data_cy = (_data_ylim[0] + _data_ylim[1]) / 2
data_rx = (_data_xlim[1] - _data_xlim[0]) / 2
data_ry = (_data_ylim[1] - _data_ylim[0]) / 2
HSTRETCH = 1.45
oval_cx = data_cx
oval_cy = data_cy
oval_rx = data_rx * HSTRETCH
oval_ry = data_ry

# Expand axis limits to accommodate stretched oval
xlim = (oval_cx - oval_rx - 0.6, oval_cx + oval_rx + 0.6)
ylim = (oval_cy - oval_ry - 0.4, oval_cy + oval_ry + 0.4)

FIG_W, FIG_H = 30, 18

# Shapely ellipse used for cluster Voronoi tiling
oval_shapely = affine_scale(
    Point(oval_cx, oval_cy).buffer(1.0),
    oval_rx, oval_ry, origin=(oval_cx, oval_cy)
)


def compute_cluster_clips(all_pts_per_cluster, cpos):
    """
    Per-cluster clip shapes that are flush where clusters would overlap but
    leave white space between distant clusters.

    1. Build an inflated convex hull per cluster.
    2. Voronoi-partition the UNION of those hulls seeded at cluster centroids.
    3. Each clip = hull ∩ voronoi_cell  →  compact shape, no overlaps.
    """
    hulls = []
    for c, pts in enumerate(all_pts_per_cluster):
        if len(pts) >= 3:
            try:
                hull     = ConvexHull(pts)
                hull_pts = pts[hull.vertices]
                centroid = hull_pts.mean(0)
                inflated = centroid + (hull_pts - centroid) * 1.42
                shp = Polygon(inflated).buffer(0.28, cap_style=1, join_style=1)
                if not shp.is_valid:
                    shp = shp.buffer(0)
            except Exception:
                shp = Point(cpos[c]).buffer(CONTINENT_SPREAD)
        elif len(pts) == 2:
            shp = (Point(pts[0]).buffer(CONTINENT_SPREAD * 0.7)
                   .union(Point(pts[1]).buffer(CONTINENT_SPREAD * 0.7)))
        else:
            shp = Point(pts[0]).buffer(CONTINENT_SPREAD * 0.7)
        hulls.append(shp)

    total_region = unary_union(hulls)
    vtiles = finite_voronoi_cells(cpos, total_region)

    clips = []
    for c in range(len(hulls)):
        vtile = vtiles[c] if vtiles[c] is not None else Polygon()
        clip  = hulls[c].intersection(vtile)
        if not clip.is_valid:
            clip = clip.buffer(0)
        clips.append(clip)
    return clips


def setup_ax(fig, ax):
    ax.set_facecolor(BG_COLOR)
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.set_aspect("equal")
    ax.axis("off")
    draw_oval_map(ax, oval_cx, oval_cy, oval_rx, oval_ry)


def setup_ax_ne(fig, ax):
    """Like setup_ax but uses the flat-top Natural-Earth-style frame."""
    ax.set_facecolor(BG_COLOR)
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.set_aspect("equal")
    ax.axis("off")
    draw_natearth_map(ax, oval_cx, oval_cy, oval_rx, oval_ry)


def save_legend_png(entries, path, fig_w=8, row_h=0.5, bsz=0.28, gap=0.12,
                    left=0.08, top_pad=0.55):
    """Save the embedding-cluster legend as a standalone PNG."""
    n      = len(entries)
    fig_h  = top_pad + n * row_h + 0.25
    fig, ax = plt.subplots(figsize=(fig_w, fig_h), facecolor=BG_COLOR)
    ax.set_facecolor(BG_COLOR)
    ax.set_xlim(0, fig_w)
    ax.set_ylim(0, fig_h)
    ax.axis("off")
    # Header
    ax.text(left, fig_h - 0.12, "Embedding Cluster",
            ha="left", va="top",
            fontsize=14, color=SUB_COLOR, fontweight="bold")
    for i, e in enumerate(entries):
        y_ctr = fig_h - top_pad - i * row_h - row_h / 2
        rect  = mpatches.Rectangle(
            (left, y_ctr - bsz / 2), bsz, bsz,
            facecolor=e["color"], edgecolor="#000000", linewidth=0.3)
        ax.add_patch(rect)
        tx = left + bsz + gap
        ax.text(tx, y_ctr + row_h * 0.16, e["label_top"],
                ha="left", va="center",
                fontsize=12, color=TEXT_COLOR)
        ax.text(tx, y_ctr - row_h * 0.16, e["label_bot"],
                ha="left", va="center",
                fontsize=10, color="#888888")
    plt.tight_layout(pad=0.3)
    fig.savefig(path, dpi=220, bbox_inches="tight", facecolor=BG_COLOR)
    plt.close(fig)


def add_legend(ax, entries, *, bot=-0.28, row_h=0.033, bsz=0.022):
    """Manually drawn legend: square colored box + dark cluster name + gray stats."""
    trans = ax.transAxes
    left  = 0.018
    gap   = 0.010
    n     = len(entries)

    ax.text(left, bot + n * row_h + 0.010, "Embedding Cluster",
            transform=trans, ha="left", va="bottom",
            fontsize=11.0, color=SUB_COLOR, fontweight="bold", zorder=14,
            clip_on=False)

    for i, e in enumerate(reversed(entries)):
        y    = bot + i * row_h
        rect = mpatches.Rectangle(
            (left, y + (row_h - bsz) / 2), bsz, bsz,
            transform=trans, clip_on=False,
            facecolor=e["color"],
            edgecolor="#000000", linewidth=0.35,
            zorder=14,
        )
        ax.add_patch(rect)
        tx = left + bsz + gap
        ax.text(tx, y + row_h * 0.65, e["label_top"],
                transform=trans, ha="left", va="center",
                fontsize=10.0, color=TEXT_COLOR, zorder=14, clip_on=False)
        ax.text(tx, y + row_h * 0.35, e["label_bot"],
                transform=trans, ha="left", va="center",
                fontsize=8.5, color="#888888", zorder=14, clip_on=False)


def add_titles(ax, subtitle_extra=""):
    ax.set_title(
        "A World Map of Embedded Earth",
        color=TEXT_COLOR, fontsize=20, fontweight="bold", pad=16,
    )
    cap_lines = [
        f"Each country is represented by the mean of 30,000 randomly sampled AlphaEarth satellite embeddings (64-dimensional).",
        f"Countries are grouped into {N_CLUSTERS} clusters using K-means; cluster positions are arranged by a global UMAP of country mean embeddings, with repulsion added between clusters for visual balance.",
        (f"Within each cluster, relative positions are refined by a local UMAP pass.  {subtitle_extra}." if subtitle_extra
         else f"Within each cluster, relative positions are refined by a local UMAP pass."),
    ]
    ax.text(0.5, -0.022, "\n".join(cap_lines),
            transform=ax.transAxes, ha="center", va="top",
            fontsize=7.5, color=SUB_COLOR, linespacing=1.6)


# =============================================================================
#  8  VERSION 1  --  WEIGHTED VORONOI
# =============================================================================

print("Computing cluster clip shapes ...")
_pts_per_cluster = [
    final_coords[np.where(labels == c)[0]] for c in range(N_CLUSTERS)
]
cluster_clips = compute_cluster_clips(_pts_per_cluster, cluster_positions)

print("Drawing Voronoi version ...")
fig1, ax1 = plt.subplots(figsize=(FIG_W, FIG_H), facecolor=BG_COLOR)
setup_ax(fig1, ax1)
legend_entries1 = []

for c in range(N_CLUSTERS):
    mask  = np.where(labels == c)[0]
    color = CLUSTER_COLORS[c]
    pts   = final_coords[mask]
    tile  = cluster_clips[c] if cluster_clips[c] is not None else Polygon()
    # Organic borders: jitter seed positions before Voronoi so shared edges stay flush
    _rng_org = np.random.default_rng(RANDOM_STATE + c * 997)
    pts_j = pts + _rng_org.normal(0, 0.13, pts.shape)
    cells = voronoi_cells_weighted(pts_j, geo_areas[mask], tile)

    for idx, i in enumerate(mask):
        cell = cells[idx]
        if cell is None or cell.is_empty:
            ax1.plot(*final_coords[i], "o", color=color,
                     markersize=4, markeredgewidth=0.5,
                     markeredgecolor="#000000", zorder=3)
            lx, ly = final_coords[i]
        else:
            shade = 0.10 + (idx % 4) * 0.04
            patch = geom_to_patch(cell,
                                  facecolor=to_rgba(color, shade),
                                  edgecolor="#000000",
                                  linewidth=0.5, zorder=2)
            if patch:
                ax1.add_patch(patch)
            try:
                lx, ly = cell.centroid.x, cell.centroid.y
            except Exception:
                lx, ly = final_coords[i]

        ax1.text(lx, ly, names[i],
                 ha="center", va="center",
                 fontsize=COUNTRY_LABEL_SIZE, color=TEXT_COLOR, zorder=6)

    # Thick black border around cluster tile
    if not tile.is_empty:
        bp = geom_to_patch(tile, facecolor="none", edgecolor="#000000",
                           linewidth=1.8, zorder=7)
        if bp:
            ax1.add_patch(bp)

    st = cluster_stats[c]
    lat  = st["mean_lat"]
    sign = "N" if lat >= 0 else "S"
    legend_entries1.append({
        "color":     color,
        "label_top": f"Cluster {c+1}",
        "label_bot": f"{st['climate']}  \u00b7  avg {abs(lat):.0f}\u00b0{sign} / ~{st['temp']}\u00b0C  \u00b7  {st['n']} countries",
    })

add_titles(ax1, "Voronoi territories weighted by country area")
draw_compass(ax1, xlim[1] - 0.75, ylim[0] + 0.45)

plt.tight_layout(pad=1.2)
p1 = OUT_DIR / "embedding_world_map_voronoi.png"
fig1.savefig(p1, dpi=220, bbox_inches="tight", facecolor=BG_COLOR)
print(f"Saved -> {p1}")
plt.close(fig1)


# =============================================================================
#  9  VERSION 2  --  PACKED COUNTRY OUTLINES  (island puzzle layout)
# =============================================================================

CLUSTER_REPULSION = 0.0001   # force multiplier for cluster-centre repulsion pass
COUNTRY_REPULSION = 0.7   # force multiplier for country-level packing

def _pack_circles(positions, radii, anchors, n_iter=800,
                  oval_cx=0.0, oval_cy=0.0, oval_rx=1e9, oval_ry=1e9):
    """Vectorised force-directed circle packing with cluster anchors + oval boundary."""
    pos = positions.copy().astype(float)
    n   = len(pos)
    for it in range(n_iter):
        # Cooling spring toward cluster anchor (fades to zero after ~60% of iters)
        spring = 0.012 * max(0.0, 1.0 - 1.5 * it / n_iter)
        pos   += (anchors - pos) * spring
        # Vectorised pairwise repulsion (allow touching, not overlapping)
        d      = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]   # (n,n,2)
        dist   = np.sqrt((d ** 2).sum(axis=2))                    # (n,n)
        np.fill_diagonal(dist, 1e9)
        min_sep = (radii[:, np.newaxis] + radii[np.newaxis, :]) * 0.5
        ov     = np.maximum(0.0, min_sep - dist)
        safe   = np.where(dist < 1e-9, 1.0, dist)
        force  = (ov[:, :, np.newaxis] * d / safe[:, :, np.newaxis]).sum(axis=1) * COUNTRY_REPULSION
        pos   += force
        # Clamp each center inside the oval (accounting for country radius)
        dx  = pos[:, 0] - oval_cx
        dy  = pos[:, 1] - oval_cy
        erx = np.maximum(oval_rx - radii, 0.1)
        ery = np.maximum(oval_ry - radii, 0.1)
        f   = np.sqrt((dx / erx) ** 2 + (dy / ery) ** 2)
        outside = f > 1.0
        if outside.any():
            sf = np.where(outside, 1.0 / np.maximum(f, 1e-9), 1.0)
            pos[:, 0] = oval_cx + dx * sf
            pos[:, 1] = oval_cy + dy * sf
    return pos


BORDER_SCALE = 8.0
_pack_r      = target_radii * BORDER_SCALE * 0.72   # bounding-circle estimate
_pack_anch   = np.array([cluster_positions[labels[i]] for i in range(len(isos))])

# --- Phase 0: repel cluster centroids from each other before country packing ---
def _repel_cluster_centers(cpos, cluster_radii, n_iter=300,
                           oval_cx=0.0, oval_cy=0.0, oval_rx=1e9, oval_ry=1e9):
    """Push cluster centres apart (treating each as a circle of its member spread)."""
    pos = cpos.copy().astype(float)
    for it in range(n_iter):
        d    = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]
        dist = np.sqrt((d ** 2).sum(axis=2))
        np.fill_diagonal(dist, 1e9)
        sep  = (cluster_radii[:, np.newaxis] + cluster_radii[np.newaxis, :]) * 0.8
        ov   = np.maximum(0.0, sep - dist)
        safe = np.where(dist < 1e-9, 1.0, dist)
        force = (ov[:, :, np.newaxis] * d / safe[:, :, np.newaxis]).sum(axis=1) * CLUSTER_REPULSION
        pos  += force
        # Clamp inside oval
        dx = pos[:, 0] - oval_cx
        dy = pos[:, 1] - oval_cy
        f  = np.sqrt((dx / oval_rx) ** 2 + (dy / oval_ry) ** 2)
        out = f > 0.92
        if out.any():
            sf = np.where(out, 0.92 / np.maximum(f, 1e-9), 1.0)
            pos[:, 0] = oval_cx + dx * sf
            pos[:, 1] = oval_cy + dy * sf
    return pos

# Cluster radius = spread of member country positions
_cluster_radii = np.array([
    (_pack_r[labels == c].sum() if (labels == c).any() else 0.5)
    for c in range(N_CLUSTERS)
])
print("Repelling cluster centres ...")
_spread_cpos = _repel_cluster_centers(
    cluster_positions, _cluster_radii,
    oval_cx=oval_cx, oval_cy=oval_cy, oval_rx=oval_rx * 0.80, oval_ry=oval_ry * 0.80
)
# Shift each country anchor to the new cluster position
_cpos_delta = _spread_cpos - cluster_positions
_pack_anch  = _pack_anch + np.array([_cpos_delta[labels[i]] for i in range(len(isos))])

print("Packing country positions ...")
packed_coords = _pack_circles(final_coords, _pack_r, _pack_anch,
                              oval_cx=oval_cx, oval_cy=oval_cy,
                              oval_rx=oval_rx, oval_ry=oval_ry)

print("Drawing country-borders version ...")
fig2, ax2 = plt.subplots(figsize=(FIG_W, FIG_H), facecolor=BG_COLOR)
setup_ax_ne(fig2, ax2)
legend_entries2 = []

# Collect all countries sorted largest-first so smaller countries draw on top
_all_countries2 = []
for c in range(N_CLUSTERS):
    for i in np.where(labels == c)[0]:
        _all_countries2.append((target_radii[i] * BORDER_SCALE, int(i), c))
_all_countries2.sort(key=lambda x: -x[0])   # largest first

for _draw_rank, (radius, i, c) in enumerate(_all_countries2):
    color  = CLUSTER_COLORS[c]
    iso    = isos[i]
    px, py = packed_coords[i]
    geom   = normed_geoms.get(iso)
    z      = 2 + _draw_rank   # largest=2 (bottom), smallest=2+N (top)

    if geom is not None and not geom.is_empty:
        placed = translate(
            affine_scale(geom, radius, radius, origin=(0, 0)),
            px, py)
        patch = geom_to_patch(placed,
                              facecolor=color,
                              edgecolor="#555555",
                              linewidth=0.3, zorder=z)
        if patch:
            ax2.add_patch(patch)
        try:
            rp = placed.representative_point()
            lx, ly = rp.x, rp.y
        except Exception:
            lx, ly = px, py
    else:
        ax2.plot(px, py, "s", color=color,
                 markersize=6, markeredgewidth=0.6,
                 markeredgecolor="#000000", zorder=z)
        lx, ly = px, py

    ax2.text(lx, ly, names[i],
             ha="center", va="center",
             fontsize=COUNTRY_LABEL_SIZE, color=TEXT_COLOR, zorder=z + 300)

for c in range(N_CLUSTERS):
    st   = cluster_stats[c]
    lat  = st["mean_lat"]
    sign = "N" if lat >= 0 else "S"
    legend_entries2.append({
        "color":     CLUSTER_COLORS[c],
        "label_top": f"Cluster {c+1}",
        "label_bot": f"{st['climate']}  \u00b7  avg {abs(lat):.0f}\u00b0{sign} / ~{st['temp']}\u00b0C  \u00b7  {st['n']} countries",
    })

add_titles(ax2, "Country outlines use per-country azimuthal equal-area (LAEA) projection, scaled to geographic area")
draw_compass(ax2, xlim[1] - 0.75, ylim[0] + 0.45)

plt.tight_layout(pad=1.2)
p2 = OUT_DIR / "embedding_world_map_borders.png"
fig2.savefig(p2, dpi=220, bbox_inches="tight", facecolor=BG_COLOR)
print(f"Saved -> {p2}")
plt.close(fig2)

# Legend as standalone PNG (uses borders legend which is identical to voronoi)
print("Saving legend ...")
p_legend = OUT_DIR / "embedding_world_map_legend.png"
save_legend_png(legend_entries2, p_legend)
print(f"Saved -> {p_legend}")


# =============================================================================
#  10  VERSION 3  --  EQUAL-AREA WORLD MAP  (Mollweide projection)
# =============================================================================

print("Drawing equal-area world map ...")
iso_to_cluster = {isos[i]: int(labels[i]) for i in range(len(isos))}

_gdf_world = gdf.to_crs(_NE_PROJ)

fig3, ax3 = plt.subplots(figsize=(FIG_W, FIG_H * 0.62), facecolor=BG_COLOR)
ax3.set_facecolor(BG_COLOR)
ax3.axis("off")
ax3.set_aspect("equal")

# Graticule lines (same precomputed vectors as draw_natearth_map)
for _gxs, _gys in _NE_GRAT_LAT:
    ax3.plot(_gxs, _gys, color=GRID_COLOR, lw=0.4, zorder=1, solid_capstyle="round")
for _gxs, _gys in _NE_GRAT_LON:
    ax3.plot(_gxs, _gys, color=GRID_COLOR, lw=0.4, zorder=1, solid_capstyle="round")

# Countries
for _, _wrow in _gdf_world.iterrows():
    _wgeom = _wrow.geometry
    if _wgeom is None or _wgeom.is_empty:
        continue
    _wiso = None
    for _wcol in ("ISO_A3", "SOV_A3", "ADM0_A3"):
        _wv = str(_wrow.get(_wcol, "")).strip()
        if _wv and _wv != "-99":
            _wiso = _wv
            break
    _wc     = iso_to_cluster.get(_wiso)
    _wcolor = CLUSTER_COLORS[_wc] if _wc is not None else "#cccccc"
    _wpatch = geom_to_patch(_wgeom,
                            facecolor=_wcolor,
                            edgecolor="#555555", linewidth=0.3, zorder=2)
    if _wpatch:
        ax3.add_patch(_wpatch)

ax3.autoscale_view()

# Natural Earth boundary outline (same precomputed vectors as draw_natearth_map)
ax3.add_patch(PathPatch(
    MplPath(np.c_[_NE_BND_X, _NE_BND_Y],
            [MplPath.MOVETO] + [MplPath.LINETO] * (len(_NE_BND_X) - 2) + [MplPath.CLOSEPOLY]),
    facecolor="none", edgecolor=BORDER_COLOR, linewidth=0.75, zorder=20, clip_on=False))

legend_entries3 = []
for c in range(N_CLUSTERS):
    st   = cluster_stats[c]
    lat  = st["mean_lat"]
    sign = "N" if lat >= 0 else "S"
    legend_entries3.append({
        "color":     CLUSTER_COLORS[c],
        "label_top": f"Cluster {c+1}",
        "label_bot": f"{st['climate']}  \u00b7  avg {abs(lat):.0f}\u00b0{sign} / ~{st['temp']}\u00b0C  \u00b7  {st['n']} countries",
    })

ax3.set_title(
    "Countries Clustered by Satellite Embedding Similarity",
    color=TEXT_COLOR, fontsize=18, fontweight="bold", pad=14,
)
_globe_cap = (
    f"Each country is represented by the mean of 30,000 randomly sampled AlphaEarth satellite embeddings (64-dimensional).\n"
    f"Countries are grouped into {N_CLUSTERS} clusters using K-means.  Natural Earth projection.  Countries without sufficient data are shown in grey."
)
ax3.text(0.5, -0.04, _globe_cap,
         transform=ax3.transAxes, ha="center", va="top",
         fontsize=8.5, color=SUB_COLOR, linespacing=1.6)

plt.tight_layout(pad=1.2)
p3 = OUT_DIR / "embedding_world_map_globe.png"
fig3.savefig(p3, dpi=220, bbox_inches="tight", facecolor=BG_COLOR)
print(f"Saved -> {p3}")
plt.close(fig3)

print("Done.")
