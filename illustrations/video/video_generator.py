"""
Video Generator for Glitchy False Color Satellite Embeddings
Creates MP4 videos with temporal effects and transitions
"""

import os
import json
import base64
import random
import argparse
import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageChops, ImageEnhance
import geopandas as gpd
from pathlib import Path
from tqdm import tqdm
from scipy.spatial.distance import cdist
import glob
import math
import subprocess

# Setup directories
BASE_DIR = Path(__file__).parent.parent
OUTPUT_DIR = Path(__file__).parent / "output"
FRAMES_DIR = OUTPUT_DIR / "frames"
FRAMES_DIR.mkdir(exist_ok=True, parents=True)

# Data paths
GEOJSON_PATH = BASE_DIR / "dimension-reduction" / "data" / "5000_sampled_classified_embeddings.geojson"
TILES_PATH = BASE_DIR / "distance_experiment_web" / "tiles"


def load_embedding_data():
    """Load and decode embedding data from geojson"""
    print("Loading embedding data...")
    gdf = gpd.read_file(GEOJSON_PATH)
    
    embeddings = []
    for emb_str in tqdm(gdf['embedding'], desc="Decoding embeddings"):
        decoded = base64.b64decode(emb_str)
        emb_list = json.loads(decoded)
        embeddings.append(emb_list)
    
    gdf['embedding_array'] = embeddings
    return gdf


def get_tile_samples(n=300, color_mode='embedding'):
    """Get tile image samples
    
    Args:
        n: Number of samples
        color_mode: 'embedding', 'satellite', or 'black'
    """
    print(f"Loading {n} tile samples (mode: {color_mode})...")
    all_tiles = []
    
    tile_dirs = list(TILES_PATH.glob("*/"))
    sample_dirs = random.sample(tile_dirs, min(n, len(tile_dirs)))
    
    for tile_dir in tqdm(sample_dirs, desc="Scanning tiles"):
        tile_num = tile_dir.name
        
        if color_mode == 'satellite':
            tile_path = tile_dir / f"{tile_num}_satellite.png"
        elif color_mode == 'embedding':
            tile_path = tile_dir / f"{tile_num}_A61_A62_A63.png"
        else:  # black
            tile_path = None  # Will create black tiles on-the-fly
            
        if tile_path and tile_path.exists():
            all_tiles.append(tile_path)
        elif color_mode == 'black':
            all_tiles.append(tile_dir)  # Store directory for black tiles
    
    return all_tiles[:n]


def get_tile_image(tile_ref, size, color_mode='embedding'):
    """Get a tile image, either from file or create black tile
    
    Args:
        tile_ref: Path to tile file or directory
        size: Tuple of (width, height)
        color_mode: 'embedding', 'satellite', or 'black'
    """
    if color_mode == 'black':
        return Image.new('RGB', size, (0, 0, 0))
    else:
        return Image.open(tile_ref).resize(size)


def create_cluster_morph(embeddings_data, tiles, frame_num, total_frames, size=(1920, 1080), color_mode='embedding'):
    """Morph between UMAP and TSNE cluster visualizations"""
    canvas = Image.new('RGB', size, (255, 255, 255))
    
    phase = frame_num / total_frames
    
    # Interpolate between UMAP and TSNE
    x_umap = embeddings_data['umap_2d_x'].values
    y_umap = embeddings_data['umap_2d_y'].values
    x_tsne = embeddings_data['tsne_2d_x'].values
    y_tsne = embeddings_data['tsne_2d_y'].values
    
    # Smooth interpolation with easing
    t = (math.sin(phase * math.pi * 2 - math.pi/2) + 1) / 2
    
    x_coords = x_umap * (1 - t) + x_tsne * t
    y_coords = y_umap * (1 - t) + y_tsne * t
    
    # Normalize
    x_min, x_max = x_coords.min(), x_coords.max()
    y_min, y_max = y_coords.min(), y_coords.max()
    
    tile_size = 12
    
    for idx in range(min(800, len(embeddings_data))):
        row = embeddings_data.iloc[idx]
        
        x = int(((x_coords[idx] - x_min) / (x_max - x_min)) * (size[0] - tile_size))
        y = int(((y_coords[idx] - y_min) / (y_max - y_min)) * (size[1] - tile_size))
        
        tile_path = tiles[idx % len(tiles)]
        tile = get_tile_image(tile_path, (tile_size, tile_size), color_mode)
        
        canvas.paste(tile, (x, y))
    
    return canvas


def create_tsne_rotation(embeddings_data, tiles, frame_num, total_frames, size=(1920, 1080), color_mode='embedding'):
    """Rotate TSNE cluster in a circle"""
    canvas = Image.new('RGB', size, (255, 255, 255))
    
    phase = frame_num / total_frames
    rotation_angle = phase * math.pi * 2  # Full rotation
    
    # Get TSNE coordinates
    x_tsne = embeddings_data['tsne_2d_x'].values
    y_tsne = embeddings_data['tsne_2d_y'].values
    
    # Center the coordinates
    x_centered = x_tsne - x_tsne.mean()
    y_centered = y_tsne - y_tsne.mean()
    
    # Apply rotation
    x_rotated = x_centered * math.cos(rotation_angle) - y_centered * math.sin(rotation_angle)
    y_rotated = x_centered * math.sin(rotation_angle) + y_centered * math.cos(rotation_angle)
    
    # Normalize to canvas
    x_min, x_max = x_rotated.min(), x_rotated.max()
    y_min, y_max = y_rotated.min(), y_rotated.max()
    
    tile_size = 12
    
    for idx in range(min(800, len(embeddings_data))):
        row = embeddings_data.iloc[idx]
        
        x = int(((x_rotated[idx] - x_min) / (x_max - x_min)) * (size[0] - tile_size))
        y = int(((y_rotated[idx] - y_min) / (y_max - y_min)) * (size[1] - tile_size))
        
        tile_path = tiles[idx % len(tiles)]
        tile = get_tile_image(tile_path, (tile_size, tile_size), color_mode)
        
        canvas.paste(tile, (x, y))
    
    return canvas


def create_umap_rotation(embeddings_data, tiles, frame_num, total_frames, size=(1920, 1080), color_mode='embedding'):
    """Rotate UMAP cluster in a circle"""
    canvas = Image.new('RGB', size, (255, 255, 255))
    
    phase = frame_num / total_frames
    rotation_angle = phase * math.pi * 2  # Full rotation
    
    # Get UMAP coordinates
    x_umap = embeddings_data['umap_2d_x'].values
    y_umap = embeddings_data['umap_2d_y'].values
    
    # Center the coordinates
    x_centered = x_umap - x_umap.mean()
    y_centered = y_umap - y_umap.mean()
    
    # Apply rotation
    x_rotated = x_centered * math.cos(rotation_angle) - y_centered * math.sin(rotation_angle)
    y_rotated = x_centered * math.sin(rotation_angle) + y_centered * math.cos(rotation_angle)
    
    # Normalize to canvas
    x_min, x_max = x_rotated.min(), x_rotated.max()
    y_min, y_max = y_rotated.min(), y_rotated.max()
    
    tile_size = 12
    
    for idx in range(min(800, len(embeddings_data))):
        row = embeddings_data.iloc[idx]
        
        x = int(((x_rotated[idx] - x_min) / (x_max - x_min)) * (size[0] - tile_size))
        y = int(((y_rotated[idx] - y_min) / (y_max - y_min)) * (size[1] - tile_size))
        
        tile_path = tiles[idx % len(tiles)]
        tile = get_tile_image(tile_path, (tile_size, tile_size), color_mode)
        
        canvas.paste(tile, (x, y))
    
    return canvas


def create_cluster_zoom(embeddings_data, tiles, frame_num, total_frames, size=(1920, 1080), color_mode='embedding'):
    """Fly through cluster, moving camera to zoom into different tiles"""
    canvas = Image.new('RGB', size, (255, 255, 255))
    
    # Use UMAP coordinates for navigation
    x_coords = embeddings_data['umap_2d_x'].values
    y_coords = embeddings_data['umap_2d_y'].values
    
    # Normalize coordinates to 0-1 range
    x_min, x_max = x_coords.min(), x_coords.max()
    y_min, y_max = y_coords.min(), y_coords.max()
    x_norm = (x_coords - x_min) / (x_max - x_min)
    y_norm = (y_coords - y_min) / (y_max - y_min)
    
    # Determine number of tiles to visit
    tiles_per_visit = 120  # 4 seconds per tile at 30fps (2s moving + 2s pause)
    total_visits = max(1, total_frames // tiles_per_visit)
    current_visit = min(total_visits - 1, frame_num // tiles_per_visit)
    visit_progress = (frame_num % tiles_per_visit) / tiles_per_visit
    
    # Build a path through the cluster
    num_samples = min(800, len(embeddings_data))
    if not hasattr(create_cluster_zoom, 'path_cache'):
        # Create path with medium-distance jumps
        coords = np.column_stack([x_norm[:num_samples], y_norm[:num_samples]])
        
        # Start with random point
        path = [random.randint(0, num_samples - 1)]
        visited = set(path)
        
        # Build path by selecting medium-distance tiles
        for _ in range(min(total_visits, num_samples - 1)):
            current_idx = path[-1]
            current_pos = coords[current_idx]
            
            # Calculate distances to all tiles
            distances = cdist([current_pos], coords)[0]
            
            # Mark visited as infinite distance
            distances[list(visited)] = np.inf
            
            # Find tiles in medium distance range (30th to 70th percentile)
            valid_distances = distances[distances != np.inf]
            if len(valid_distances) == 0:
                break
            
            dist_10 = np.percentile(valid_distances, 10)
            dist_50 = np.percentile(valid_distances, 50)
            
            # Get indices of medium-distance tiles
            medium_dist_mask = (distances >= dist_10) & (distances <= dist_50)
            medium_dist_indices = np.where(medium_dist_mask)[0]
            
            if len(medium_dist_indices) == 0:
                # Fallback to any unvisited tile
                unvisited = [i for i in range(num_samples) if i not in visited]
                if not unvisited:
                    break
                next_idx = random.choice(unvisited)
            else:
                # Randomly pick from medium-distance tiles
                next_idx = random.choice(medium_dist_indices)
            
            path.append(next_idx)
            visited.add(next_idx)
        
        create_cluster_zoom.path_cache = path
    
    path = create_cluster_zoom.path_cache
    
    # Get current and next tile in path
    current_idx = path[min(current_visit, len(path) - 1)]
    next_idx = path[min(current_visit + 1, len(path) - 1)]
    
    # Smooth easing for camera movement with pause
    t = visit_progress
    
    # Split visit into movement (first half) and pause (second half)
    movement_duration = 0.5
    
    if t < movement_duration:
        # Flying phase - interpolate position
        t_move = t / movement_duration
        t_smooth = (math.sin((t_move - 0.5) * math.pi) + 1) / 2
    else:
        # Pause phase - stay at destination
        t_smooth = 1.0
    
    # Camera position interpolates between tiles
    camera_x = x_norm[current_idx] * (1 - t_smooth) + x_norm[next_idx] * t_smooth
    camera_y = y_norm[current_idx] * (1 - t_smooth) + y_norm[next_idx] * t_smooth
    
    # Camera zoom: stay constant to avoid abrupt changes between visits
    # Could add slight zoom out during movement for drama, but stay continuous
    if t < movement_duration:
        # Slight zoom out during middle of movement
        t_move = t / movement_duration
        zoom_dip = math.sin(t_move * math.pi) * 0.1  # Small dip in the middle
        zoom_curve = 1.0 - zoom_dip  # 1.0 -> 0.9 -> 1.0
    else:
        # Stay fully zoomed in during pause
        zoom_curve = 1.0
    
    base_zoom = 15.0  # Base zoom level
    camera_zoom = base_zoom * zoom_curve
    
    # Draw all tiles with camera transformation
    for idx in range(num_samples):
        # Get tile position in normalized coordinates
        tile_x = x_norm[idx]
        tile_y = y_norm[idx]
        
        # Transform by camera (pan and zoom)
        # Center camera on target position
        rel_x = tile_x - camera_x
        rel_y = tile_y - camera_y
        
        # Apply zoom
        screen_x = size[0] / 2 + rel_x * size[0] * camera_zoom
        screen_y = size[1] / 2 + rel_y * size[1] * camera_zoom
        
        # Calculate tile size based on distance from camera center
        distance = math.sqrt(rel_x**2 + rel_y**2)
        
        # Tiles closer to camera center are larger
        base_tile_size = 200  # Maximum tile size at camera center
        tile_size = int(base_tile_size / (1 + distance * camera_zoom * 5))
        tile_size = max(4, min(400, tile_size))  # Clamp size
        
        # Skip tiles that are off-screen
        if (screen_x < -tile_size or screen_x > size[0] + tile_size or
            screen_y < -tile_size or screen_y > size[1] + tile_size):
            continue
        
        # Load and draw tile
        tile_path = tiles[idx % len(tiles)]
        tile = get_tile_image(tile_path, (tile_size, tile_size), color_mode)
        
        # Fade tiles based on distance from camera
        brightness = max(0.3, 1.0 - distance * 2)
        if brightness < 1.0:
            tile = ImageEnhance.Brightness(tile).enhance(brightness)
        
        # Center tile on position
        paste_x = int(screen_x - tile_size / 2)
        paste_y = int(screen_y - tile_size / 2)
        
        canvas.paste(tile, (paste_x, paste_y))
    
    return canvas


def generate_video_sequence(video_type, tiles, embeddings_data, duration_sec=10, fps=30, color_mode='embedding'):
    """Generate a video sequence"""
    total_frames = duration_sec * fps
    size = (1920, 1080)
    
    print(f"Generating {video_type} video ({total_frames} frames @ {fps}fps)...")
    
    # Clear frames directory
    for f in FRAMES_DIR.glob("*.png"):
        f.unlink()
    
    # Generate frames
    for frame_num in tqdm(range(total_frames), desc=f"Rendering {video_type}"):
        if video_type == 'cluster':
            frame = create_cluster_morph(embeddings_data, tiles, frame_num, total_frames, size, color_mode)
        elif video_type == 'tsne_rotation':
            frame = create_tsne_rotation(embeddings_data, tiles, frame_num, total_frames, size, color_mode)
        elif video_type == 'umap_rotation':
            frame = create_umap_rotation(embeddings_data, tiles, frame_num, total_frames, size, color_mode)
        elif video_type == 'cluster_zoom':
            frame = create_cluster_zoom(embeddings_data, tiles, frame_num, total_frames, size, color_mode)
        
        frame.save(FRAMES_DIR / f"frame_{frame_num:04d}.png")
    
    # Compile to video with ffmpeg
    output_path = OUTPUT_DIR / f"{video_type}_video_{color_mode}.mp4"
    
    print(f"Compiling video with ffmpeg...")
    cmd = [
        'ffmpeg', '-y',
        '-framerate', str(fps),
        '-i', str(FRAMES_DIR / 'frame_%04d.png'),
        '-c:v', 'libx264',
        '-pix_fmt', 'yuv420p',
        '-crf', '18',
        str(output_path)
    ]
    
    try:
        subprocess.run(cmd, check=True, capture_output=True)
        print(f"✓ Video saved: {output_path}")
        
        # Clean up frames
        for f in FRAMES_DIR.glob("*.png"):
            f.unlink()
            
    except subprocess.CalledProcessError as e:
        print(f"Error compiling video: {e}")
        print("Frames saved in:", FRAMES_DIR)
    except FileNotFoundError:
        print("ffmpeg not found. Please install ffmpeg to generate videos.")
        print("Frames saved in:", FRAMES_DIR)


def main():
    """Generate video sequences"""
    parser = argparse.ArgumentParser(description='Generate satellite embedding videos')
    parser.add_argument('--color', choices=['embedding', 'satellite', 'black'], 
                        default='embedding',
                        help='Color mode: embedding (A61-A63 bands), satellite (true color), or black')
    parser.add_argument('--zoom', action='store_true',
                        help='Generate cluster zoom video (navigates through cluster)')
    args = parser.parse_args()
    
    print("=== Glitchy False Color Video Generator ===\n")
    print(f"Color mode: {args.color}\n")
    
    # Load data
    embeddings_data = load_embedding_data()
    tiles = get_tile_samples(n=300, color_mode=args.color)
    
    print(f"\nLoaded {len(embeddings_data)} embeddings and {len(tiles)} tiles\n")
    
    # Generate videos
    if args.zoom:
        video_types = [
            ('cluster_zoom', 30),    # 30 second cluster zoom navigation
        ]
    else:
        video_types = [
            ('cluster', 20),         # 20 second cluster morph
            ('tsne_rotation', 15),   # 15 second TSNE rotation
            ('umap_rotation', 15),   # 15 second UMAP rotation
        ]
    
    for video_type, duration in video_types:
        print(f"\n{'='*60}")
        print(f"Generating {video_type} video ({duration}s)")
        print('='*60)
        try:
            generate_video_sequence(video_type, tiles, embeddings_data, 
                                   duration_sec=duration, fps=30, color_mode=args.color)
        except Exception as e:
            print(f"Error generating {video_type}: {e}")
            continue
    
    print(f"\n{'='*60}")
    print("✓ Video generation complete!")
    print(f"Output directory: {OUTPUT_DIR}")
    print('='*60)


if __name__ == "__main__":
    main()
