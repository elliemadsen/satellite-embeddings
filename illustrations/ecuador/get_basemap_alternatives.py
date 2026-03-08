"""
Get cloud-free imagery from GEE's pre-made composites
Location: 0°16'20.29"N 78°12'53.29"W

Note: The exact Google Earth basemap is not available as a dataset.
These are the closest alternatives - pre-made cloud-free composites.
"""

import ee
import requests
from pathlib import Path

# Initialize Earth Engine
ee.Initialize(project='gsapp-map')

# Output directory
OUTPUT_DIR = Path(__file__).parent
OUTPUT_DIR.mkdir(exist_ok=True)

# Coordinates
LON = -78.2148
LAT = 0.2723

# Create point of interest
point = ee.Geometry.Point([LON, LAT])
region = point.buffer(5000).bounds()

print(f"Location: {LAT}°N, {abs(LON)}°W")
print(f"Downloading pre-made cloud-free composites...\n")

def download_image(image, filename, bands, vis_params):
    """Download image to file"""
    rgb = image.select(bands).visualize(**vis_params)
    
    url = rgb.getDownloadURL({
        'region': region,
        'dimensions': 1024,
        'format': 'png'
    })
    
    filepath = OUTPUT_DIR / filename
    
    response = requests.get(url)
    if response.status_code == 200:
        with open(filepath, 'wb') as f:
            f.write(response.content)
        print(f"✓ Downloaded: {filename}")
        return True
    else:
        print(f"✗ Failed: {filename}")
        return False

# 1. Sentinel-2 median composite (simple cloud-free composite)
print("=" * 60)
print("Sentinel-2 Median Composite (2024)")
print("=" * 60)
try:
    s2_composite = ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED') \
        .filterBounds(point) \
        .filterDate('2024-01-01', '2024-12-31') \
        .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 20)) \
        .median()
    
    vis_params = {
        'min': 0,
        'max': 3000,
        'gamma': 1.4
    }
    
    download_image(s2_composite, 'sentinel2_median_2024.png', ['B4', 'B3', 'B2'], vis_params)
except Exception as e:
    print(f"Error: {e}")

# 2. Latest clear Sentinel-2 image (single clearest image, not composite)
print("\n" + "=" * 60)
print("Single Clearest Sentinel-2 Image")
print("=" * 60)
try:
    clear_image = ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED') \
        .filterBounds(point) \
        .filterDate('2023-01-01', '2025-12-31') \
        .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 5)) \
        .sort('CLOUDY_PIXEL_PERCENTAGE') \
        .first()
    
    info = clear_image.getInfo()
    date = info['properties']['system:time_start']
    from datetime import datetime
    date_obj = datetime.fromtimestamp(date / 1000)
    
    print(f"Date: {date_obj.strftime('%Y-%m-%d')}")
    print(f"Cloud: {info['properties']['CLOUDY_PIXEL_PERCENTAGE']:.1f}%")
    
    vis_params = {
        'min': 0,
        'max': 3000,
        'gamma': 1.4
    }
    
    download_image(clear_image, f"sentinel2_clear_{date_obj.strftime('%Y%m%d')}.png", ['B4', 'B3', 'B2'], vis_params)
except Exception as e:
    print(f"Error: {e}")

# 3. NICFI Planet basemap (high-res cloud-free, available in tropics)
print("\n" + "=" * 60)
print("Planet NICFI Basemap (High-Resolution Tropical)")
print("=" * 60)
try:
    nicfi = ee.ImageCollection('projects/planet-nicfi/assets/basemaps/americas') \
        .filterBounds(point) \
        .sort('system:time_start', False) \
        .first()
    
    info = nicfi.getInfo()
    date = info['properties']['system:time_start']
    date_obj = datetime.fromtimestamp(date / 1000)
    
    print(f"Date: {date_obj.strftime('%Y-%m')}")
    
    vis_params = {
        'min': 64,
        'max': 5454,
        'gamma': 1.4
    }
    
    download_image(nicfi, f"planet_nicfi_{date_obj.strftime('%Y%m')}.png", ['R', 'G', 'B'], vis_params)
except Exception as e:
    print(f"Error: {e}")

print("\n" + "=" * 60)
print("✓ Complete!")
print("Note: The Google Earth basemap itself is not available as a GEE dataset.")
print("These are pre-made cloud-free alternatives.")
print("=" * 60)
