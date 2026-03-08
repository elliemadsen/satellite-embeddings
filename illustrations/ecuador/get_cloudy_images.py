"""
Get high cloud coverage images from Ecuador location using Google Earth Engine
Location: 0°16'20.29"N 78°12'53.29"W
"""

import ee
import requests
from pathlib import Path
from datetime import datetime

# Initialize Earth Engine
ee.Initialize(project='gsapp-map')

# Output directory
OUTPUT_DIR = Path(__file__).parent
OUTPUT_DIR.mkdir(exist_ok=True)

# Convert coordinates: 0°16'20.29"N 78°12'53.29"W
# 0 + 16/60 + 20.29/3600 = 0.2723°N
# 78 + 12/60 + 53.29/3600 = 78.2148°W
LON = -78.2148
LAT = 0.2723

# Create point of interest
point = ee.Geometry.Point([LON, LAT])

# Buffer around point (e.g., 5km radius for image)
region = point.buffer(5000).bounds()

print(f"Location: {LAT}°N, {abs(LON)}°W")
print(f"Searching for high cloud coverage images...\n")

def get_sentinel2_cloudy_images(min_cloud=50, max_cloud=100, limit=10):
    """Get Sentinel-2 images with high cloud coverage"""
    
    # Sentinel-2 Surface Reflectance
    collection = ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED') \
        .filterBounds(point) \
        .filter(ee.Filter.gte('CLOUDY_PIXEL_PERCENTAGE', min_cloud)) \
        .filter(ee.Filter.lte('CLOUDY_PIXEL_PERCENTAGE', max_cloud)) \
        .sort('CLOUDY_PIXEL_PERCENTAGE', False) \
        .limit(limit)
    
    return collection

def get_landsat8_cloudy_images(min_cloud=50, max_cloud=100, limit=10):
    """Get Landsat 8 images with high cloud coverage"""
    
    # Landsat 8 Surface Reflectance
    collection = ee.ImageCollection('LANDSAT/LC08/C02/T1_L2') \
        .filterBounds(point) \
        .filter(ee.Filter.gte('CLOUD_COVER', min_cloud)) \
        .filter(ee.Filter.lte('CLOUD_COVER', max_cloud)) \
        .sort('CLOUD_COVER', False) \
        .limit(limit)
    
    return collection

def download_image(image, satellite_name, date_str, cloud_pct, bands, vis_params):
    """Download image to file"""
    
    # Select bands and visualize
    rgb = image.select(bands).visualize(**vis_params)
    
    # Get download URL
    url = rgb.getDownloadURL({
        'region': region,
        'dimensions': 1024,
        'format': 'png'
    })
    
    # Download
    filename = f"{satellite_name}_{date_str}_{cloud_pct:.1f}pct.png"
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

# Get Sentinel-2 images
print("=" * 60)
print("SENTINEL-2 Images (High Cloud Coverage)")
print("=" * 60)

s2_collection = get_sentinel2_cloudy_images(min_cloud=60, max_cloud=90, limit=5)
s2_info = s2_collection.getInfo()

if s2_info['features']:
    for feature in s2_info['features']:
        props = feature['properties']
        image_id = feature['id']
        date = props['system:time_start']
        date_obj = datetime.fromtimestamp(date / 1000)
        date_str = date_obj.strftime('%Y%m%d')
        cloud_pct = props['CLOUDY_PIXEL_PERCENTAGE']
        
        print(f"\nDate: {date_obj.strftime('%Y-%m-%d')}")
        print(f"Cloud coverage: {cloud_pct:.1f}%")
        print(f"Image ID: {image_id}")
        
        image = ee.Image(image_id)
        
        # True color RGB
        vis_params = {
            'min': 0,
            'max': 3000,
            'gamma': 1.4
        }
        
        download_image(image, 'sentinel2', date_str, cloud_pct, ['B4', 'B3', 'B2'], vis_params)
else:
    print("No Sentinel-2 images found with specified cloud coverage")

# Get Landsat 8 images
# print("\n" + "=" * 60)
# print("LANDSAT 8 Images (High Cloud Coverage)")
# print("=" * 60)

# l8_collection = get_landsat8_cloudy_images(min_cloud=60, max_cloud=90, limit=5)
# l8_info = l8_collection.getInfo()

# if l8_info['features']:
#     for feature in l8_info['features']:
#         props = feature['properties']
#         image_id = feature['id']
#         date = props['system:time_start']
#         date_obj = datetime.fromtimestamp(date / 1000)
#         date_str = date_obj.strftime('%Y%m%d')
#         cloud_pct = props['CLOUD_COVER']
        
#         print(f"\nDate: {date_obj.strftime('%Y-%m-%d')}")
#         print(f"Cloud coverage: {cloud_pct:.1f}%")
#         print(f"Image ID: {image_id}")
        
#         image = ee.Image(image_id)
        
#         # Apply scale factors for Landsat Collection 2
#         def apply_scale_factors(image):
#             optical = image.select('SR_B.').multiply(0.0000275).add(-0.2)
#             return image.addBands(optical, None, True)
        
#         image = apply_scale_factors(image)
        
#         # True color RGB
#         vis_params = {
#             'min': 0.0,
#             'max': 0.3,
#             'gamma': 1.4
#         }
        
#         download_image(image, 'landsat8', date_str, cloud_pct, ['SR_B4', 'SR_B3', 'SR_B2'], vis_params)
# else:
#     print("No Landsat 8 images found with specified cloud coverage")

print("\n" + "=" * 60)
print("✓ Complete! Images saved to:", OUTPUT_DIR)
print("=" * 60)
