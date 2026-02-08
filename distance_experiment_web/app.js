// Global state
const SQUARE_VALUES = [64, 144, 256, 576, 900, 1024]; // Perfect squares for 2D
const CUBE_VALUES = [64, 125, 216, 512, 1000];   // Perfect cubes for 3D
let dataCache = {}; // Cache loaded datasets
let imageCache = new Map(); // Cache preloaded images
let currentN = 64;
let currentDimension = 2;
let currentAlgorithm = 'umap';
let currentImageType = 'embed'; // 'embed' or 'rgb'
let currentBandStart = 0; // Starting band for embedding visualization (0-61)
let currentModalSample = null; // Currently displayed sample in modal
let scene, camera, renderer, controls;
let modalOpen = false; // Track if modal is open
// Land classification legend
const LAND_CLASSIFICATION_LEGEND = {
        0: ["#949C9F", "Unknown / No Data"],
        20: ["#5C4772", "Shrubs"],
        30: ["#CB98B9", "Herbaceous Vegetation"],
        40: ["#F5C138", "Cultivated / Agriculture"],
        50: ["#F18F01", "Urban / Built-up"],
        60: ["#F9F26F", "Bare / Sparse Vegetation"],
        70: ["#9A5A66", "Snow and Ice"],
        80: ["#71D7F0", "Permanent Water Bodies"],
        90: ["#ADCAD6", "Herbaceous Wetland"],
        100: ["#C9DA5E", "Moss & Lichen"],
        111: ["#58481f", "Closed Forest (evergreen needleleaf)"],
        112: ["#009900", "Closed Forest (evergreen broadleaf)"],
        113: ["#70663e", "Closed Forest (deciduous needleleaf)"],
        114: ["#00cc00", "Closed Forest (deciduous broadleaf)"],
        115: ["#4e751f", "Closed Forest (mixed)"],
        116: ["#007800", "Closed Forest"],
        121: ["#666000", "Open Forest (evergreen needleleaf)"],
        122: ["#8db400", "Open Forest (evergreen broadleaf)"],
        123: ["#8d7400", "Open Forest (deciduous needleleaf)"],
        124: ["#a0dc00", "Open Forest (deciduous broadleaf)"],
        125: ["#929900", "Open Forest (mixed)"],
        126: ["#648c00", "Open Forest"],
        200: ["#006E90", "Oceans / Seas"],
        cf: ["#425C1A", "Closed Forest"],
        of: ["#99C24D", "Open Forest"],
      };


// Modal functions
function openModal(imageSrc, sample) {
    const modal = document.getElementById('modal-overlay');
    const img = document.getElementById('modal-image');
    img.src = imageSrc;
    modal.classList.add('active');
    
    // If showing satellite image, cover the right column too
    if (currentImageType === 'rgb') {
        modal.classList.add('fullwidth');
    } else {
        modal.classList.remove('fullwidth');
    }
    
    modalOpen = true;
    currentModalSample = sample;
    
    // Keep info visible and update it for the clicked image
    if (sample) {
        showInfo(sample);
        // Increase z-index to be above modal
        document.getElementById('info').style.zIndex = '2001';
    }
}

function closeModal() {
    const modal = document.getElementById('modal-overlay');
    modal.classList.remove('active');
    modalOpen = false;
    currentModalSample = null;
    
    // Reset info z-index and hide it
    const info = document.getElementById('info');
    info.style.zIndex = '1000';
    hideInfo();
}

// Modal event listeners
document.getElementById('modal-close').addEventListener('click', closeModal);
document.getElementById('modal-overlay').addEventListener('click', (e) => {
    if (e.target.id === 'modal-overlay') {
        closeModal();
    }
});

// Close modal on Escape key
document.addEventListener('keydown', (e) => {
    if (e.key === 'Escape') {
        closeModal();
    }
});

// Preload image into cache
function preloadImage(src) {
    if (imageCache.has(src)) {
        return Promise.resolve(imageCache.get(src));
    }
    
    return new Promise((resolve, reject) => {
        const img = new Image();
        img.onload = () => {
            imageCache.set(src, img);
            resolve(img);
        };
        img.onerror = reject;
        img.src = src;
    });
}

// Preload all images for a dataset
async function preloadImagesForData(data, imageType, bandStart = 0) {
    const imagePromises = data.map(sample => {
        const imageSrc = getImageSrc(sample, imageType, bandStart);
        return preloadImage(imageSrc);
    });
    
    try {
        await Promise.all(imagePromises);
        console.log(`Preloaded ${imagePromises.length} images`);
    } catch (error) {
        console.warn('Some images failed to preload:', error);
    }
}

// Load GeoJSON data for specific N
async function loadData(n) {
    // Check cache first
    if (dataCache[n]) {
        return dataCache[n];
    }
    
    try {
        const response = await fetch(`data/web_grid_data_${n}.geojson`);
        const geojson = await response.json();
        const data = geojson.features.map(feat => ({
            ...feat.properties,
            lon: feat.geometry.coordinates[0],
            lat: feat.geometry.coordinates[1]
        }));
        dataCache[n] = data;
        
        // Preload images in background for both image types
        setTimeout(() => {
            preloadImagesForData(data, 'embed');
            preloadImagesForData(data, 'rgb');
        }, 100);
        
        return data;
    } catch (error) {
        console.error(`Error loading data for N=${n}:`, error);
        throw error;
    }
}

// Find closest N value
function findClosestN(value) {
    return N_VALUES.reduce((prev, curr) => 
        Math.abs(curr - value) < Math.abs(prev - value) ? curr : prev
    );
}

// Get available N values for current dimension
function getAvailableNValues() {
    if (currentDimension === 1) {
        return null; // 1D uses continuous slider
    } else if (currentDimension === 2) {
        return SQUARE_VALUES;
    } else if (currentDimension === 3) {
        return CUBE_VALUES;
    }
}

// Update slider to match current dimension
function updateSlider() {
    const slider = document.getElementById('samples-slider');
    const availableValues = getAvailableNValues();
    
    if (currentDimension === 1) {
        // 1D: continuous slider from 1 to 100
        slider.min = 1;
        slider.max = 100;
        slider.value = Math.min(currentN, 100);
        slider.step = 1;
    } else {
        // 2D/3D: discrete positions
        const currentIndex = availableValues.indexOf(currentN);
        slider.min = 0;
        slider.max = availableValues.length - 1;
        slider.value = currentIndex >= 0 ? currentIndex : 0;
        slider.step = 1;
        
        // If current N not in available values, switch to first available
        if (currentIndex < 0) {
            currentN = availableValues[0];
        }
    }
}

// Initialize - load first dataset
async function initialize() {
    try {
        // Set up slider for initial dimension (2D by default)
        updateSlider();
        
        await loadData(currentN);
        document.getElementById('loading').style.display = 'none';
        render();
    } catch (error) {
        document.getElementById('loading').textContent = 'Error loading data';
    }
}

// Event listeners
document.getElementById('samples-slider').addEventListener('input', async (e) => {
    const sliderValue = parseInt(e.target.value);
    const availableValues = getAvailableNValues();
    
    let requestedN;
    if (currentDimension === 1) {
        // 1D: direct value from slider
        requestedN = sliderValue;
    } else {
        // 2D/3D: map slider index to discrete value
        requestedN = availableValues[sliderValue];
    }
    
    // For 1D, always use largest dataset and slice
    const datasetN = currentDimension === 1 ? 1000 : requestedN;
    
    // Only reload if dataset changed
    if (datasetN !== currentN && currentDimension !== 1) {
        currentN = requestedN;
        document.getElementById('loading').style.display = 'block';
        try {
            await loadData(currentN);
            document.getElementById('loading').style.display = 'none';
        } catch (error) {
            document.getElementById('loading').textContent = 'Error loading data';
            return;
        }
    } else if (currentDimension === 1) {
        // Load 1000 dataset if not already loaded
        if (!dataCache[1000]) {
            document.getElementById('loading').style.display = 'block';
            try {
                await loadData(1000);
                document.getElementById('loading').style.display = 'none';
            } catch (error) {
                document.getElementById('loading').textContent = 'Error loading data';
                return;
            }
        }
        currentN = requestedN; // Store the actual display count for 1D
    }
    
    // Update display
    document.getElementById('samples-value').textContent = requestedN;
    render();
});

// Dimension toggle buttons
document.querySelectorAll('[data-dimension]').forEach(btn => {
    btn.addEventListener('click', async () => {
        document.querySelectorAll('[data-dimension]').forEach(b => b.classList.remove('active'));
        btn.classList.add('active');
        const newDimension = parseInt(btn.dataset.dimension);
        
        // Store old dimension
        const oldDimension = currentDimension;
        currentDimension = newDimension;
        
        // Update slider for new dimension
        updateSlider();
        
        // Determine which dataset to load
        let datasetN;
        if (currentDimension === 1) {
            datasetN = 1000; // Always use largest for 1D
            currentN = Math.min(currentN, 100); // Clamp to 100 for display
        } else {
            const availableValues = getAvailableNValues();
            // Try to keep similar N, or use first available
            const closestIndex = availableValues.reduce((prev, curr, idx) => 
                Math.abs(curr - currentN) < Math.abs(availableValues[prev] - currentN) ? idx : prev
            , 0);
            currentN = availableValues[closestIndex];
            datasetN = currentN;
        }
        
        // Load data if needed
        if (!dataCache[datasetN]) {
            document.getElementById('loading').style.display = 'block';
            try {
                await loadData(datasetN);
                document.getElementById('loading').style.display = 'none';
            } catch (error) {
                document.getElementById('loading').textContent = 'Error loading data';
                return;
            }
        }
        
        document.getElementById('samples-value').textContent = currentN;
        render();
    });
});

// Algorithm toggle buttons
document.querySelectorAll('[data-algo]').forEach(btn => {
    btn.addEventListener('click', () => {
        document.querySelectorAll('[data-algo]').forEach(b => b.classList.remove('active'));
        btn.classList.add('active');
        currentAlgorithm = btn.dataset.algo;
        render();
    });
});

// Image type toggle buttons
document.querySelectorAll('[data-imgtype]').forEach(btn => {
    btn.addEventListener('click', () => {
        document.querySelectorAll('[data-imgtype]').forEach(b => b.classList.remove('active'));
        btn.classList.add('active');
        currentImageType = btn.dataset.imgtype;
        updateBandSelectorVisibility();
        render();
    });
});

// Helper function to get correct image source based on image type and band selection
function getImageSrc(sample, imageType = currentImageType, bandStart = currentBandStart) {
    const tileDir = sample.tile_dir || `tiles/${String(sample.index).padStart(4, '0')}`;
    const index = String(sample.index).padStart(4, '0');
    
    if (imageType === 'embed') {
        const b1 = bandStart;
        const b2 = bandStart + 1;
        const b3 = bandStart + 2;
        return `${tileDir}/${index}_A${b1}_A${b2}_A${b3}.png`;
    } else {
        return `${tileDir}/${index}_satellite.png`;
    }
}

// Show info on hover
function showInfo(sample) {
    const info = document.getElementById('info');
    document.getElementById('info-location').textContent = 
        `${sample.lat.toFixed(4)}, ${sample.lon.toFixed(4)}`;
    
    // Look up classification in legend
    const classValue = sample.classification;
    const classEntry = LAND_CLASSIFICATION_LEGEND[classValue];
    const className = classEntry ? classEntry[1] : (classValue || 'N/A');
    document.getElementById('info-class').textContent = className;
    
    document.getElementById('info-region').textContent = sample.subregion_name || 'Unknown';
    info.classList.add('visible');
}

function hideInfo() {
    // Don't hide info if modal is open
    if (!modalOpen) {
        document.getElementById('info').classList.remove('visible');
    }
}

// Render based on current settings
async function render() {
    // Hide all containers
    document.getElementById('grid-1d').style.display = 'none';
    document.getElementById('grid-2d').style.display = 'none';
    document.getElementById('grid-3d').style.display = 'none';
    
    // Get data for current N
    let data;
    if (currentDimension === 1) {
        // For 1D, use 1000 dataset and slice to currentN
        data = dataCache[1000];
        if (!data) {
            console.error('Data not loaded for N = 1024');
            return;
        }
    } else {
        // For 2D/3D, use exact dataset
        data = dataCache[currentN];
        if (!data) {
            console.error('Data not loaded for N =', currentN);
            return;
        }
    }
    
    // Determine samples to use
    const samples = currentDimension === 1 ? data.slice(0, currentN) : data;
    document.getElementById('samples-value').textContent = currentN;
    
    if (currentDimension === 1) {
        render1D(samples);
    } else if (currentDimension === 2) {
        render2D(samples);
    } else if (currentDimension === 3) {
        render3D(samples);
    }
}

// 1D Rendering - horizontal line
function render1D(samples) {
    const container = document.getElementById('grid-1d');
    container.style.display = 'block';
    container.innerHTML = '';
    
    const xKey = `grid_${currentAlgorithm}_1d_x`;
    const middleColumn = document.getElementById('middle-column');
    const containerWidth = middleColumn.clientWidth;
    const containerHeight = middleColumn.clientHeight;
    const imageSize = Math.min(80, containerWidth / samples.length);
    const gap = 5;
    const totalWidth = samples.length * (imageSize + gap);
    const startX = (containerWidth - totalWidth) / 2;
    const centerY = containerHeight / 2;
    
    // Sort by grid position
    const sorted = samples.sort((a, b) => a[xKey] - b[xKey]);
    
    sorted.forEach((sample, i) => {
        const img = document.createElement('img');
        img.src = getImageSrc(sample);
        img.className = 'grid-image';
        img.style.width = `${imageSize}px`;
        img.style.height = `${imageSize}px`;
        img.style.left = `${startX + i * (imageSize + gap)}px`;
        img.style.top = `${centerY - imageSize/2}px`;
        
        img.addEventListener('mouseenter', () => showInfo(sample));
        img.addEventListener('mouseleave', hideInfo);
        img.addEventListener('click', () => openModal(getImageSrc(sample), sample));
        
        container.appendChild(img);
    });
}

// 2D Rendering - grid layout
function render2D(samples) {
    const container = document.getElementById('grid-2d');
    container.style.display = 'block';
    container.innerHTML = '';
    
    const xKey = `grid_${currentAlgorithm}_2d_x`;
    const yKey = `grid_${currentAlgorithm}_2d_y`;
    
    // Find grid bounds
    const gridSize = Math.ceil(Math.sqrt(samples.length));
    const middleColumn = document.getElementById('middle-column');
    const containerWidth = middleColumn.clientWidth;
    const containerHeight = middleColumn.clientHeight;
    const buffer = 20; // Fixed buffer around grid
    const gap = 3;
    
    // Calculate image size to fit grid with buffer
    const availableWidth = containerWidth - buffer * 2;
    const availableHeight = containerHeight - buffer * 2;
    const imageSize = Math.min(100, Math.min(availableWidth, availableHeight) / gridSize - gap);
    
    const totalGridWidth = gridSize * (imageSize + gap) - gap;
    const totalGridHeight = gridSize * (imageSize + gap) - gap;
    const offsetX = (containerWidth - totalGridWidth) / 2;
    const offsetY = (containerHeight - totalGridHeight) / 2;
    
    samples.forEach(sample => {
        const gridX = sample[xKey];
        const gridY = sample[yKey];
        
        const img = document.createElement('img');
        img.src = getImageSrc(sample);
        img.className = 'grid-image';
        img.style.width = `${imageSize}px`;
        img.style.height = `${imageSize}px`;
        img.style.left = `${offsetX + gridX * (imageSize + gap)}px`;
        img.style.top = `${offsetY + gridY * (imageSize + gap)}px`;
        
        img.addEventListener('mouseenter', () => showInfo(sample));
        img.addEventListener('mouseleave', hideInfo);
        img.addEventListener('click', () => openModal(getImageSrc(sample), sample));
        
        container.appendChild(img);
    });
}

// 3D Rendering - Three.js cube
function render3D(samples) {
    const canvas = document.getElementById('grid-3d');
    canvas.style.display = 'block';
    
    // Save previous camera position and target if they exist
    let savedCameraPosition = null;
    let savedControlsTarget = null;
    if (camera && controls) {
        savedCameraPosition = camera.position.clone();
        savedControlsTarget = controls.target.clone();
    }
    
    // Clean up previous scene properly
    if (scene) {
        // Dispose of all geometries, materials, and textures
        scene.traverse((object) => {
            if (object.geometry) {
                object.geometry.dispose();
            }
            if (object.material) {
                if (Array.isArray(object.material)) {
                    object.material.forEach(material => {
                        if (material.map) material.map.dispose();
                        material.dispose();
                    });
                } else {
                    if (object.material.map) object.material.map.dispose();
                    object.material.dispose();
                }
            }
        });
        // Clear the scene
        while(scene.children.length > 0) {
            scene.remove(scene.children[0]);
        }
    }
    
    if (renderer) {
        renderer.dispose();
        canvas.innerHTML = '';
    }
    
    if (controls) {
        controls.dispose();
    }
    
    const xKey = `grid_${currentAlgorithm}_3d_x`;
    const yKey = `grid_${currentAlgorithm}_3d_y`;
    const zKey = `grid_${currentAlgorithm}_3d_z`;
    
    // Get container dimensions
    const middleColumn = document.getElementById('middle-column');
    const containerWidth = middleColumn.clientWidth;
    const containerHeight = middleColumn.clientHeight;
    
    // Setup Three.js scene
    scene = new THREE.Scene();
    scene.background = new THREE.Color(0xffffff);
    
    camera = new THREE.PerspectiveCamera(
        75,
        containerWidth / containerHeight,
        0.1,
        10000
    );
    
    renderer = new THREE.WebGLRenderer({ canvas, antialias: true });
    renderer.setSize(containerWidth, containerHeight);
    
    // Find cube dimensions - fixed total size, variable image size
    const cubeSize = Math.ceil(Math.pow(samples.length, 1/3));
    const fixedTotalSize = 20; // Fixed cube size in 3D space
    const spacing = fixedTotalSize / cubeSize; // Spacing decreases as N increases
    const imageSize = spacing * 0.75; // Image size is 75% of spacing

    const initialZoom = 1.3;
    
    // Restore saved camera position or use default
    if (savedCameraPosition && savedControlsTarget) {
        camera.position.copy(savedCameraPosition);
        camera.lookAt(savedControlsTarget);
    } else {
        camera.position.set(fixedTotalSize * initialZoom, fixedTotalSize * initialZoom * 0.7, fixedTotalSize * initialZoom);
        camera.lookAt(fixedTotalSize / 2, fixedTotalSize / 2 - 2, fixedTotalSize / 2);
    }
    
    // Add orbit controls
    controls = new THREE.OrbitControls(camera, renderer.domElement);
    if (savedControlsTarget) {
        controls.target.copy(savedControlsTarget);
    } else {
        controls.target.set(fixedTotalSize / 2, fixedTotalSize / 2 - 2, fixedTotalSize / 2);
    }
    controls.enableDamping = true;
    
    // Add lights
    const ambientLight = new THREE.AmbientLight(0xffffff, 0.6);
    scene.add(ambientLight);
    
    const directionalLight = new THREE.DirectionalLight(0xffffff, 0.8);
    directionalLight.position.set(fixedTotalSize, fixedTotalSize, fixedTotalSize);
    scene.add(directionalLight);
    
    // Create image planes
    const loader = new THREE.TextureLoader();
    const raycaster = new THREE.Raycaster();
    const mouse = new THREE.Vector2();
    let hoveredObject = null;
    let hoverOutline = null; // Store the outline mesh for cleanup
    
    samples.forEach(sample => {
        const gridX = sample[xKey];
        const gridY = sample[yKey];
        const gridZ = sample[zKey];
        
        loader.load(getImageSrc(sample), (texture) => {
            const geometry = new THREE.PlaneGeometry(imageSize, imageSize);
            const material = new THREE.MeshBasicMaterial({ 
                map: texture,
                side: THREE.DoubleSide
            });
            const mesh = new THREE.Mesh(geometry, material);
            
            mesh.position.set(
                gridX * spacing,
                gridY * spacing,
                gridZ * spacing
            );
            
            // Store sample data
            mesh.userData = sample;
            
            // Ensure image renders on top of shadow
            mesh.renderOrder = 1;
            
            scene.add(mesh);
        });
    });
    

    
    // Mouse interaction
    function onMouseMove(event) {
        const middleColumn = document.getElementById('middle-column');
        const rect = middleColumn.getBoundingClientRect();
        const containerWidth = middleColumn.clientWidth;
        const containerHeight = middleColumn.clientHeight;
        
        mouse.x = ((event.clientX - rect.left) / containerWidth) * 2 - 1;
        mouse.y = -((event.clientY - rect.top) / containerHeight) * 2 + 1;
        
        raycaster.setFromCamera(mouse, camera);
        const intersects = raycaster.intersectObjects(scene.children.filter(obj => obj.userData.lat));
        
        if (intersects.length > 0 && intersects[0].object.userData.lat) {
            const obj = intersects[0].object;
            
            // Change cursor to pointer
            renderer.domElement.style.cursor = 'pointer';
            
            if (hoveredObject !== obj) {
                hoveredObject = obj;
                showInfo(obj.userData);
            }
        } else {
            // Reset cursor
            renderer.domElement.style.cursor = 'default';
            
            if (hoveredObject) {
                hoveredObject = null;
                hideInfo();
            }
        }
    }
    
    function onMouseClick(event) {
        const middleColumn = document.getElementById('middle-column');
        const rect = middleColumn.getBoundingClientRect();
        const containerWidth = middleColumn.clientWidth;
        const containerHeight = middleColumn.clientHeight;
        
        mouse.x = ((event.clientX - rect.left) / containerWidth) * 2 - 1;
        mouse.y = -((event.clientY - rect.top) / containerHeight) * 2 + 1;
        
        raycaster.setFromCamera(mouse, camera);
        const intersects = raycaster.intersectObjects(scene.children);
        
        if (intersects.length > 0 && intersects[0].object.userData.lat) {
            const sample = intersects[0].object.userData;
            openModal(getImageSrc(sample), sample);
        }
    }
    
    renderer.domElement.addEventListener('mousemove', onMouseMove);
    renderer.domElement.addEventListener('click', onMouseClick);
    
    // Animation loop
    function animate() {
        requestAnimationFrame(animate);
        controls.update();
        renderer.render(scene, camera);
    }
    
    animate();
}

// Handle window resize
window.addEventListener('resize', () => {
    if (currentDimension === 3 && camera) {
        const middleColumn = document.getElementById('middle-column');
        const containerWidth = middleColumn.clientWidth;
        const containerHeight = middleColumn.clientHeight;
        camera.aspect = containerWidth / containerHeight;
        camera.updateProjectionMatrix();
        renderer.setSize(containerWidth, containerHeight);
    } else {
        render();
    }
});

// Band selector functions
function initBandSelector() {
    const selector = document.getElementById('band-selector');
    const scale = document.getElementById('band-scale');
    const handle = document.getElementById('band-handle');
    const bandValue = document.getElementById('band-value');
    
    const padding = 20; // Top/bottom padding from CSS
    const availableHeight = 800 - (padding * 2); // 760px usable space
    const tickSpacing = availableHeight / 63; // Space between each band number
    
    // Create tick marks for 0-63
    const tickElements = [];
    for (let i = 0; i <= 63; i++) {
        const tick = document.createElement('div');
        tick.className = 'band-tick';
        tick.textContent = String(i).padStart(2, '0');
        tick.style.top = `${padding + (i * tickSpacing)}px`;
        scale.appendChild(tick);
        tickElements.push(tick);
    }
    
    // Create RGB labels
    const rgbLabels = ['R', 'G', 'B'];
    const rgbElements = [];
    rgbLabels.forEach((label, idx) => {
        const elem = document.createElement('div');
        elem.className = `rgb-label ${label.toLowerCase()}`;
        elem.textContent = label;
        scale.appendChild(elem);
        rgbElements.push(elem);
    });
    
    let isDragging = false;
    const minBand = 0;
    const maxBand = 61; // 0-61 allows A0-A1-A2 through A61-A62-A63
    const handleHeight = tickSpacing * 3; // Height to cover exactly 3 bands
    
    function updateHandlePosition(band) {
        // Position handle so it centers on the 3 selected bands
        const position = padding + (band * tickSpacing);
        handle.style.top = `${position}px`;
        handle.style.height = `${handleHeight}px`;
        
        // Update RGB label positions to align with selected bands
        rgbElements.forEach((elem, idx) => {
            elem.style.top = `${padding + ((band + idx) * tickSpacing)}px`;
        });
        
        // Highlight selected band numbers
        tickElements.forEach((tick, idx) => {
            if (idx >= band && idx <= band + 2) {
                tick.classList.add('selected');
            } else {
                tick.classList.remove('selected');
            }
        });
        
        // Update modal image if modal is open
        if (modalOpen && currentModalSample) {
            const modalImg = document.getElementById('modal-image');
            modalImg.src = getImageSrc(currentModalSample);
        }
    }
    
    function handleMove(clientY) {
        const rect = selector.getBoundingClientRect();
        let y = clientY - rect.top - padding; // Adjust for padding
        const maxY = maxBand * tickSpacing; // Allow handle to reach band 61
        y = Math.max(0, Math.min(y, maxY));
        
        const band = Math.round(y / tickSpacing);
        currentBandStart = Math.max(minBand, Math.min(band, maxBand));
        
        updateHandlePosition(currentBandStart);
        render();
    }
    
    handle.addEventListener('mousedown', (e) => {
        isDragging = true;
        e.preventDefault();
    });
    
    document.addEventListener('mousemove', (e) => {
        if (isDragging) {
            handleMove(e.clientY);
        }
    });
    
    document.addEventListener('mouseup', () => {
        isDragging = false;
    });
    
    selector.addEventListener('click', (e) => {
        if (e.target !== handle && !handle.contains(e.target)) {
            handleMove(e.clientY);
        }
    });
    
    updateHandlePosition(currentBandStart);
}

function updateBandSelectorVisibility() {
    const bandSelector = document.getElementById('band-selector');
    if (currentImageType === 'embed') {
        bandSelector.style.display = 'block';
    } else {
        bandSelector.style.display = 'none';
    }
}

// Initialize
initialize();
initBandSelector();
updateBandSelectorVisibility();
