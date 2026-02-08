# Distance Experiment Web Visualization

Interactive web visualization of satellite embedding dimensionality reduction results with 1D, 2D, and 3D grid layouts. Features a three-column layout with parameter controls, visualization canvas, and an interactive band selector for exploring different false color band combinations.

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
- Download satellite imagery for each sample:
  - **Embedding tiles**: 62 false color band combinations (bands 0-1-2 through 61-62-63) from Google Satellite Embedding
  - **True color satellite**: RGB imagery from Sentinel-2 or Landsat 8
- Output multiple GeoJSON files for different N values:
  - 2D squares: 64, 144, 256, 576, 900, 1024 samples
  - 3D cubes: 64, 125, 216, 512, 1000 samples
- Tile structure: `tiles/{index:04d}/{index}_A{b1}_A{b2}_A{b3}.png` for embeddings, `{index}_satellite.png` for RGB

The notebook implements:

- **Parallel processing**: 3 concurrent locations, 20 concurrent downloads per location
- **Retry logic**: Up to 3 attempts per download with exponential backoff
- **Multi-pass system**: Automatically retries failed downloads across 5 passes
- **Fallback strategies** for true color imagery:
  1. Sentinel-2 with <20% cloud cover
  2. Sentinel-2 without cloud filter
  3. Sentinel-2 from 2022-2025 (3 years) with <30% clouds
  4. Landsat 8 from 2022-2025 with <30% clouds

### 2. Run the visualization

Open `index.html` in your browser.

## Features

### Layout

The application uses a three-column flexbox layout:

- **Left column (320px)**: Parameter controls and location info popup
- **Middle column (flexible)**: Visualization canvas and modal image viewer
- **Right column (180px)**: Interactive band selector for false color combinations

### Interactive Controls

- **Samples Slider**:

  - 1D mode: Continuous from 1-100 samples
  - 2D mode: Discrete steps at 64, 144, 256, 576, 900, 1024 (perfect squares)
  - 3D mode: Discrete steps at 64, 125, 216, 512, 1000 (perfect cubes)

- **Dimension Toggle**: Switch between 1D line, 2D grid, or 3D cube layouts

- **Algorithm Toggle**: Switch between UMAP and t-SNE dimension reduction results

- **Image Type Toggle**:
  - **Satellite**: True color RGB satellite imagery from Sentinel-2 or Landsat 8
  - **Embedding**: False color visualization using selected bands from Google Satellite Embedding (64 bands total, A00-A63)

### Band Selector

When viewing embedding images, an interactive vertical band selector appears in the right column:

- **64 band ticks**: Zero-padded band numbers (00-63) arranged vertically
- **Draggable handle**: Select 3 consecutive bands for RGB visualization
- **RGB labels**: indicators show which band maps to each color channel
- **Modal integration**: Band selector remains visible and functional during image closeup

The band selector automatically hides when viewing true color satellite images.

### Interaction

- **Hover**: View location (lat/lon), land classification, and region name in info panel (left column bottom)
- **Click**: Open enlarged modal view of satellite patch
- **3D Controls**:
  - Click and drag to rotate
  - Scroll to zoom
  - Right-click drag to pan
  - Hover highlights with soft drop shadow effect
  - Camera position preserved when changing settings

### Performance

- **HTTP Caching**: Images and data cached by browser
- **Tile-based storage**: Images organized in directories (`tiles/0000/`, `tiles/0001/`, etc.)
- **Dynamic image loading**: Only loads the currently selected band combination
- **Data Caching**: GeoJSON files cached in memory to avoid reloading
- **Three.js memory management**: Proper disposal of geometries, materials, and textures to prevent memory leaks

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
├── styles.css                    # Stylesheet
├── app.js                        # JavaScript visualization logic
├── generate_web_data.ipynb       # Data processing notebook
├── README.md                     # This file
├── data/
│   ├── web_grid_data_64.geojson
│   ├── web_grid_data_125.geojson
│   ├── web_grid_data_144.geojson
│   └── ...                       # GeoJSON for each N value
└── tiles/
    ├── 0000/
    │   ├── 0000_A00_A01_A02.png  # Band combination 0-1-2
    │   ├── 0000_A01_A02_A03.png  # Band combination 1-2-3
    │   ├── ...                   # All 62 band combinations
    │   ├── 0000_A61_A62_A63.png  # Band combination 61-62-63
    │   └── 0000_satellite.png    # True color RGB
    ├── 0001/
    │   ├── 0001_A00_A01_A02.png
    │   └── ...
    └── ...
```

## Technical Details

### Band Combinations

The Google Satellite Embedding provides 64 spectral bands (A00-A63). The visualization generates all possible consecutive triplet combinations:

- 62 total combinations: (0,1,2), (1,2,3), ..., (61,62,63)
- Each combination saved as a separate PNG file
- User can interactively select which 3 consecutive bands to visualize as RGB

### Grid Positioning

All grid layouts use the Hungarian algorithm (scipy.optimize.linear_sum_assignment) to minimize total distance:

- Cost matrix: Euclidean distance from embedding coordinates to grid positions
- Optimal one-to-one assignment between samples and grid cells
- Preserves relative proximity from high-dimensional embedding space

## Requirements

### Python

- `ee` - Google Earth Engine API
- `geopandas` - Geospatial data handling
- `scipy` - Hungarian algorithm (linear_sum_assignment)
- `numpy`, `pandas` - Data processing
- `requests` - HTTP requests for image downloads
- `concurrent.futures` - Parallel processing

### Earth Engine

- Google Earth Engine account and project
- Project ID configured in notebook
- Access to:
  - `GOOGLE/SATELLITE_EMBEDDING/V1/ANNUAL`
  - `COPERNICUS/S2_SR_HARMONIZED` (Sentinel-2)
  - `LANDSAT/LC08/C02/T1_L2` (Landsat 8, fallback)
