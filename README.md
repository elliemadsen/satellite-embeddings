# Satellite Embeddings

A series of experiments exploring Google DeepMind's AlphaEarth satellite embedding dataset to understand patterns of environmental change through multi-dimensional embedding space.

## Live Projects

- **[Embedding Topography Interactive Visualization](https://elliemadsen.github.io/satellite-embeddings/dimension-reduction/point_cloud.html)** – Latent space explorer, globally sampled embeddings reduced to 3d with UMAP or t-SNE.
- **[False Color Embedding Globe](https://elliemadsen.github.io/satellite-embeddings/web-maps/globe.html)** – Interactive 3D globe visualization of satellite embeddings
- **[Earth Engine App](https://gsapp-map.projects.earthengine.app/view/sat-embeddings)** – Interactive web app for real-time analysis

---

## Project Structure

### **dimension-reduction/** – Global Sampling & UMAP Projection

Generates a global dataset of stratified satellite embeddings and reduces them to 2D for exploration.

- Stratifies global sampling by land classification (snow/ice, agriculture, urban, forest, water) using Copernicus land cover
- Samples 1000–5000 locations globally with equal representation per category
- Extracts 64D embedding vectors for each location using Earth Engine
- Applies UMAP dimensionality reduction to project embeddings to 2D
- Creates interactive 3D point cloud visualization with Plotly

**Inputs:** Google Satellite Embedding V1 (2024), Copernicus Global Land Cover, UN shapefile  
**Outputs:** GeoJSON files with embeddings + UMAP coords, HTML point cloud visualization, classified samples by land type

![image](dimension-reduction/output/website-screenshot.png)

---

### **distance_experiment/** – Temporal & Spatial Embedding Distance

Compares embedding spaces across years and generates large-scale similarity grids.

- **global-similarity-grid.ipynb**: Extends to global scale using pre-computed UMAP coordinates from dimension-reduction
- **us-similarity-grid.ipynb**: Creates US-wide grid of sample points, extracts embeddings for multiple years (2021–2024), computes Euclidean distances between year pairs
- **gen-us-samples.ipynb**: Generates random coordinate samples for the US grid
- Outputs satellite patches and metadata for each location

**Inputs:** Random or stratified coordinates, Google Satellite Embeddings (multi-year)  
**Outputs:** CSV files with embedding distances, GeoJSON with sample geometries, satellite patch image files, UMAP grid layouts

![image](distance_experiment/output/400_64_10_2024/400_64_10_2024_sentinel_grid.png)

---

### **web-maps/** – Global Web Visualization

Generates web tiles and interactive globe for global embedding exploration.

- **gen-tiles.ipynb**: Exports global embedding data as GeoTIFFs (4 regional exports) from Earth Engine to Google Drive, then locally:
  - Merges 4 GeoTIFFs into single global dataset
  - Converts from 32-bit to 8-bit and EPSG:4326 → EPSG:3857
  - Generates tile pyramid (zoom 0–8) in XYZ tile structure
- Creates interactive globe.html with Three.js or Cesium viewer

**Inputs:** Google Satellite Embedding V1 (2024), global geometry  
**Outputs:** Tile pyramid (xyz structure for zoom 0–8), globe HTML viewer

![image](web-maps/screenshot.png)

---

### **two-point-projection/** – Two-Point Equidistant Projection

Creates a cartographic projection centered on two geographic points using D3.js.

- Implements d3.js `geoTwoPointEquidistant` projection with two anchor points from near locations in embedding space (Alaska and Greenland)
- Interactive HTML/SVG visualization

**Inputs:** `land-50m.json` (TopoJSON land geometry)  
**Outputs:** Interactive HTML map with two-point equidistant projection, downloadable SVG

![image](two-point-projection/projection.svg)

---

### **poster/** – Drawing Generation

Generates visualizations for poster/publication materials. Also includes Adobe Illustrator poster file.

![image](poster/poster.png)

---

### **element-embeddings/** – Element Analysis

Creates drawing comparing four elemental typologies: forest, water, ice, desert.

**Inputs:** Element GeoJSON with locations  
**Outputs:** Embedding vectors per element, satellite/drawing comparisons

---

### **watershed/** – Temporal Embedding Change Analysis - NYC Watershed, Catskills

Analyzes satellite embedding changes in NYC watershed between 2017 and 2024.

- Filters embeddings to 50km watershed region
- Compares 2017 vs 2024 embeddings via K-means clustering (6, 10, 20 clusters)
- Computes per-pixel Euclidean distance between years
- Identifies and visualizes the 3 embedding bands with greatest change
- Maps results interactively

**Inputs:** Google Satellite Embeddings (2017, 2024), NYC watershed geometry  
**Outputs:** Geemap interactive layers (clustering, distance, band differences)

![image](watershed/img/clusters.png)

<!-- ![image](watershed/img/rgb.png) -->

---

### **greenland/** – Glacier Change Detection via Embeddings

Analyzes Greenland glacier regions (Eqip Sermia) using embedding vectors to detect temporal changes.

- Compares embeddings from 2017 and 2024 for glacier regions
- Computes per-pixel Euclidean distances in embedding space
- Visualizes distance heatmaps to identify areas of significant environmental change
- Generates reprojected TIF outputs for further geospatial analysis

**Inputs:** Google Satellite Embeddings (2017, 2024), glacier region coordinates  
**Outputs:** GeoTIFF files with RGB embeddings and Euclidean distance rasters

![image](greenland/maps/eqip_sermia/eqip-poster.png)

---

### **bay-area/** – Embedding Similarity & Coastal Topology

Explores spatial patterns in the Bay Area's satellite embedding space through k-nearest neighbor networks.

- Samples locations in San Francisco and the Bay Area using both random and grid-based methods
- Constructs k-NN networks in 64D embedding space using cosine similarity
- Compares embedding distances with geographic distances to identify environmental patterns
- Analyzes edge angles in the similarity network to discover that similar conditions cluster at specific orientations (125°), suggesting topographic influence

**Inputs:** Google Satellite Embedding V1 Annual dataset (2024), Bay Area/SF geometry  
**Outputs:** Folium maps with network visualizations, angle distribution histograms

![image](bay-area/img/nearest-neighbors.png)
![image](bay-area/img/angle-histogram.png)

---

### **palestine/** – Regional Embedding Analysis

Analyzes embedding patterns for Palestine region.

**Inputs:** Google Satellite Embeddings, Palestine geometry  
**Outputs:** Interactive maps with embedding visualizations

![image](palestine/change-17-24.png)

---

### **wisconsin/** – Multi-band Satellite Data Export

Exports all 64 embedding bands as RGB GeoTIFFs for a Wisconsin region.

- Iterates through all 64 embedding bands in groups of 3 (R,G,B)
- Downloads each triple as a separate GeoTIFF at 500m and 5000m resolution
- Stores as individual band-indexed TIFF files

**Inputs:** Google Satellite Embedding V1 (2024), Wisconsin region boundary  
**Outputs:** 21 GeoTIFF files (~500m resolution) + 21 files (~5000m resolution) organized by band groups

![image](wisconsin/data/animation_5000m_i3.gif)

---

## Getting Started

### Prerequisites

```bash
pip install -r requirements.txt
```

### Earth Engine Setup

1. Authenticate: `ee.Authenticate()`
2. Initialize with your project: `ee.Initialize(project="your-project")`

### Running Analysis Notebooks

Each directory contains standalone Jupyter notebooks. Run them sequentially.

---

## Data Attribution

Uses Google Earth Engine's AlphaEarth satellite embedding dataset. See individual notebooks for detailed citations.
