"""
Get AlphaEarth embeddings from Ecuador location
Location: 0°16'20.29"N 78°12'53.29"W
Uses first three bands (A0, A1, A2) as RGB
"""

import ee
import requests
from pathlib import Path

# Initialize Earth Engine
ee.Initialize(project='gsapp-map')

# AlphaEarth dataset
dataset = ee.dataset = ee.ImageCollection("GOOGLE/SATELLITE_EMBEDDING/V1/ANNUAL")

# Output directory
OUTPUT_DIR = Path(__file__).parent
OUTPUT_DIR.mkdir(exist_ok=True)

# Coordinates
LON = -78.2148
LAT = 0.2723

# Create point and region
point = ee.Geometry.Point(LON, LAT)
region = point.buffer(5000).bounds()

print(f"Location: {LAT}°N, {abs(LON)}°W")
print(f"Downloading AlphaEarth embeddings...\n")

# Get AlphaEarth image
image = dataset.filterBounds(region).mosaic().clip(region)

# Visualize three axes of the embedding space as an RGB
vis_params = {'min': -0.3, 'max': 0.3, 'bands': ['A20', 'A21', 'A22']}

# Download
print("=" * 60)
print("AlphaEarth Embeddings (First 3 Bands as RGB)")
print("=" * 60)

try:
    rgb = image.visualize(**vis_params)
    
    url = rgb.getDownloadURL({
        'region': region,
        'dimensions': 1024,
        'format': 'png'
    })
    
    filepath = OUTPUT_DIR / 'alphaearth_A20-A22.png'
    
    response = requests.get(url)
    if response.status_code == 200:
        with open(filepath, 'wb') as f:
            f.write(response.content)
        print(f"✓ Downloaded: alphaearth_A0-A2.png")
    else:
        print(f"✗ Failed to download")
        
except Exception as e:
    print(f"Error: {e}")

print("\n" + "=" * 60)
print("✓ Complete!")
print("=" * 60)
