# Distance Experiment Web Visualization

Interactive web visualization of satellite embedding dimensionality reduction results with 1D, 2D, and 3D grid layouts.

## Setup

### 1. Generate the data

Run the notebook to process embeddings and download satellite patches:

```bash
jupyter notebook generate_web_data.ipynb
```

This will:

- Load samples from `../dimension-reduction/data/2000_sampled_classified_embeddings.geojson`
- Generate grid layouts for 1D, 2D (perfect squares), and 3D (perfect cubes) using Hungarian algorithm
- Process both UMAP and t-SNE dimension reduction results
- Download two types of satellite imagery for each sample:
  - Embedding visualization (Google Satellite Embedding RGB bands 1-3)
  - True color satellite (Sentinel-2 or Landsat 8 with fallback strategies)
- Output multiple GeoJSON files for different N values:
  - 2D squares: 64, 144, 256, 576, 900, 1024 samples
  - 3D cubes: 64, 125, 216, 512, 1000 samples

The notebook implements fallback strategies for satellite imagery:

1. Sentinel-2 with <20% cloud cover
2. Sentinel-2 without cloud filter
3. Sentinel-2 from 2022-2025 (3 years) with <30% clouds
4. Landsat 8 from 2022-2025 with <30% clouds

### 2. Run the visualization

```bash
# Python 3
python -m http.server 8000
```

Then open `http://localhost:8000` in your browser.

## Features

### Interactive Controls

- **Samples Slider**:

  - 1D mode: Continuous from 1-100 samples
  - 2D mode: Discrete steps at 64, 144, 256, 576, 900, 1024 (perfect squares)
  - 3D mode: Discrete steps at 64, 125, 216, 512, 1000 (perfect cubes)

- **Dimension Toggle**: Switch between 1D line, 2D grid, or 3D cube layouts

- **Algorithm Toggle**: Switch between UMAP and t-SNE dimension reduction results

- **Image Type Toggle**:
  - Embedding: Visualization of the embedding RGB bands (A01, A02, A03)
  - Satellite: True color satellite imagery from Sentinel-2 or Landsat 8

### Interaction

- **Hover**: View location (lat/lon), land classification, and region name in info panel
- **Click**: Open enlarged view of satellite patch with info panel overlay
- **3D Controls**:
  - Click and drag to rotate
  - Scroll to zoom
  - Right-click drag to pan
  - Hover highlights with soft drop shadow effect
  - Camera position preserved when changing settings

### Performance

- **HTTP Caching**: Images and data cached by browser for 1 hour
- **Image Preloading**: Background preloading of both image types when dataset loads
- **Data Caching**: GeoJSON files cached in memory to avoid reloading

## Visualization Modes

### 1D

Images arranged in a horizontal line based on 1D embedding coordinate. Samples are sorted and optimally matched to linear grid positions using the Hungarian algorithm.

- Continuous slider from 1-100 samples
- Images scaled to fit screen width

### 2D

Images arranged in a square grid minimizing distance from embedding space to grid positions using the Hungarian algorithm.

- Grid dimensions automatically calculated: ceil(√N) × ceil(√N)
- Discrete N values that form perfect squares
- Fixed 40px buffer maintained at all sample counts
- Images scale down for larger grids while maintaining aspect ratio
- Maximum image size: 100px

### 3D

Images arranged in a 3D cube with Three.js, supporting full orbit controls.

- Cube dimensions: ceil(∛N) × ceil(∛N) × ceil(∛N)
- Discrete N values that form perfect cubes
- Interactive camera with orbit controls (rotate, zoom, pan)
- Hover highlighting with blurred square gradient shadow (512×512 texture)
- Cursor changes to pointer on hover
- Polygon offset rendering prevents z-fighting artifacts
- Camera position preserved across setting changes

## File Structure

```
distance_experiment_web/
├── index.html                    # Main HTML page
├── app.js                        # JavaScript visualization logic
├── generate_web_data.ipynb       # Data processing notebook
├── README.md                     # This file
├── data/
│   ├── web_grid_data_64.geojson
│   ├── web_grid_data_125.geojson
│   ├── web_grid_data_144.geojson
│   └── ...                       # GeoJSON for each N value
└── sat_patches/
    ├── patch_0000_embed.png      # Embedding visualization
    ├── patch_0000_rgb.png        # True color satellite
    ├── patch_0001_embed.png
    ├── patch_0001_rgb.png
    └── ...
```

## Requirements

### Python

- `ee` - Google Earth Engine API
- `geopandas` - Geospatial data handling
- `scipy` - Hungarian algorithm (linear_sum_assignment)
- `numpy`, `pandas` - Data processing
- `Pillow` - Image handling
- `requests` - HTTP requests for image downloads

### Earth Engine

- Google Earth Engine account and project
- Project ID configured in notebook (`gsapp-map`)
- Access to:
  - `GOOGLE/SATELLITE_EMBEDDING/V1/ANNUAL`
  - `COPERNICUS/S2_SR_HARMONIZED` (Sentinel-2)
  - `LANDSAT/LC08/C02/T1_L2` (Landsat 8, fallback)
