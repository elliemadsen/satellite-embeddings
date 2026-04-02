import * as THREE from "three";
import { OrbitControls } from "https://cdn.jsdelivr.net/npm/three@0.164/examples/jsm/controls/OrbitControls.js";

let features, N;
let positions, colors;
let geometry, points;
let geojson;

// Raycaster for hover detection
const raycaster = new THREE.Raycaster();
const mouse = new THREE.Vector2();
let mouseInitialized = false;
raycaster.params.Points.threshold =0.5;

Promise.all([
  fetch("data/20000_sampled_classified_embeddings.geojson").then((r) => r.json()),
  // fetch("data/10000_sampled_classified_embeddings.geojson").then((r) => r.json()),
]).then(([d1]) => {
  geojson = d1;
    // geojson = { ...d1, features: [...d1.features, ...d2.features] };
    initPointCloud();

    updateDimReductionVisibility();

    document
      .querySelectorAll(
        "input[name='dim_reduction'], input[name='color'], input[name='projection']"
      )
      .forEach((el) =>
        el.addEventListener("change", () => {
          updateDimReductionVisibility();
          updatePointCloud();
          updateLegend();
        })
      );
  });

const LAND_CLASSIFICATION_LEGEND = {
  0: ["#949C9F", "Unknown / No Data"],
  20: ["#5C4772", "Shrubs"],
  30: ["#CB98B9", "Herbaceous vegetation"],
  40: ["#F5C138", "Cultivated / Agriculture"],
  50: ["#F18F01", "Urban / Built-up"],
  60: ["#F9F26F", "Bare / Sparse vegetation"],
  70: ["#9A5A66", "Snow and Ice"],
  80: ["#71D7F0", "Permanent water bodies"],
  90: ["#ADCAD6", "Herbaceous wetland"],
  100: ["#C9DA5E", "Moss & Lichen"],
  200: ["#006E90", "Oceans / Seas"],
  cf: ["#425C1A", "Closed Forest"],
  of: ["#99C24D", "Open Forest"],
};

//   0: ["#9e9e9e", "Unknown / No Data"],
//   20: ["#c4a35a", "Shrubs"],
//   30: ["#a8d57a", "Herbaceous vegetation"],
//   40: ["#f5cb45", "Cultivated / Agriculture"],
//   50: ["#d9534f", "Urban / Built-up"],
//   60: ["#e8d4a0", "Bare / Sparse vegetation"],
//   70: ["#daeef7", "Snow and Ice"],
//   80: ["#4a9eda", "Permanent water bodies"],
//   90: ["#5bbfb5", "Herbaceous wetland"],
//   100: ["#b5ce52", "Moss & Lichen"],
//   200: ["#1a5f8a", "Oceans / Seas"],
//   cf: ["#2d7a2d", "Closed Forest"],
//   of: ["#6ab54d", "Open Forest"],
// };


const SUBREGION_LEGEND = {
  0: ["#949C9F", "Unknown / No Data"],
  15.0: ["#F18F01", "Northern Africa"],
  202.0: ["#F5C138", "Sub-Saharan Africa"],
  419.0: ["#425C1A", "Latin America and the Caribbean"],
  21.0: ["#99C24D", "Northern America"],
  143.0: ["#71D7F0", "Central Asia"],
  30.0: ["#006E90", "Eastern Asia"],
  35.0: ["#62a8c4", "South-eastern Asia"],
  34.0: ["#355a9e", "Southern Asia"],
  145.0: ["#1e98c0", "Western Asia"],
  151.0: ["#9A5A66", "Eastern Europe"],
  154.0: ["#CB98B9", "Northern Europe"],
  39.0: ["#5C4772", "Southern Europe"],
  155.0: ["#D96C65", "Western Europe"],
  53.0: ["#E1BE41", "Australia and New Zealand"],
  54.0: ["#F9F26F", "Melanesia"],
  57.0: ["#F9F26F", "Micronesia"],
  61.0: ["#F9F26F", "Polynesia"],
};

//   0: ["#9e9e9e", "Unknown / No Data"],
//   15.0: ["#e07c35", "Northern Africa"],
//   202.0: ["#f2b134", "Sub-Saharan Africa"],
//   419.0: ["#56ab2f", "Latin America and the Caribbean"],
//   21.0: ["#2980b9", "Northern America"],
//   143.0: ["#8e44ad", "Central Asia"],
//   30.0: ["#c0392b", "Eastern Asia"],
//   35.0: ["#16a085", "South-eastern Asia"],
//   34.0: ["#d35400", "Southern Asia"],
//   145.0: ["#c0417b", "Western Asia"],
//   151.0: ["#7f8c8d", "Eastern Europe"],
//   154.0: ["#5dade2", "Northern Europe"],
//   39.0: ["#a569bd", "Southern Europe"],
//   155.0: ["#2471a3", "Western Europe"],
//   53.0: ["#82b944", "Australia and New Zealand"],
//   54.0: ["#48c9b0", "Melanesia"],
//   57.0: ["#85c1e9", "Micronesia"],
//   61.0: ["#f0b27a", "Polynesia"],
// };

let radius = 40;

function latLonToSphere(lat, lon, r = radius) {
  const phi = (90 - lat) * (Math.PI / 180);
  const theta = (180 - lon) * (Math.PI / 180);

  const x = r * Math.sin(phi) * Math.cos(theta);
  const y = r * Math.cos(phi);
  const z = r * Math.sin(phi) * Math.sin(theta);

  return { x, y, z };
}

const container = document.getElementById("container");

const scene = new THREE.Scene();
scene.background = new THREE.Color(0xffffff);

const camera = new THREE.PerspectiveCamera(
  60,
  (window.innerWidth - 260) / window.innerHeight,
  0.1,
  2000
);
camera.position.set(50, 50, 50);

const renderer = new THREE.WebGLRenderer({ antialias: true });
renderer.setSize(window.innerWidth - 260, window.innerHeight);
container.appendChild(renderer.domElement);

const controls = new OrbitControls(camera, renderer.domElement);
controls.enableDamping = true;

window.addEventListener("resize", () => {
  renderer.setSize(window.innerWidth - 260, window.innerHeight);
  camera.aspect = (window.innerWidth - 260) / window.innerHeight;
  camera.updateProjectionMatrix();
});

// Mouse move handler for hover detection
renderer.domElement.addEventListener("mousemove", onMouseMove);

function onMouseMove(event) {
  // Calculate mouse position in normalized device coordinates
  const rect = renderer.domElement.getBoundingClientRect();
  mouse.x = ((event.clientX - rect.left) / rect.width) * 2 - 1;
  mouseInitialized = true;
  mouse.y = -((event.clientY - rect.top) / rect.height) * 2 + 1;
}

function initPointCloud() {
  features = geojson.features;

  N = features.length;
  positions = new Float32Array(N * 3);
  colors = new Float32Array(N * 3);

  // Store original embedding positions + globe positions
  window.embeddingPositions = new Float32Array(N * 3);
  window.globePositions = new Float32Array(N * 3);

  // Fill both arrays after reading features
  for (let i = 0; i < N; i++) {
    const p = features[i].properties;

    // store embedding
    embeddingPositions[i * 3 + 0] = p.tsne_3d_x;
    embeddingPositions[i * 3 + 1] = p.tsne_3d_y;
    embeddingPositions[i * 3 + 2] = p.tsne_3d_z;

    // compute geo sphere position
    const { x, y, z } = latLonToSphere(p.lat, p.lon);
    globePositions[i * 3 + 0] = x;
    globePositions[i * 3 + 1] = y;
    globePositions[i * 3 + 2] = z;
  }

  geometry = new THREE.BufferGeometry();

  geometry.setAttribute("position", new THREE.BufferAttribute(positions, 3));
  geometry.setAttribute("color", new THREE.BufferAttribute(colors, 3));

  const material = new THREE.PointsMaterial({
    size: 0.5,
    vertexColors: true,
    sizeAttenuation: true,
  });

  points = new THREE.Points(geometry, material);
  scene.add(points);

  updatePointCloud();
  updateLegend();
}

function remapForestClass(c) {
  if (c >= 111 && c <= 116) return "cf";
  if (c >= 121 && c <= 126) return "of";
  return c;
}

function updatePointCloud(duration = 1200) {
  if (!features) return;

  const dim_reduction = document.querySelector(
    'input[name="dim_reduction"]:checked'
  ).value;
  const projection = document.querySelector(
    'input[name="projection"]:checked'
  ).value;

  const colorMode = document.querySelector(
    'input[name="color"]:checked'
  ).value;

  // Compute target positions based on projection
  const targets = [];
  for (let i = 0; i < N; i++) {
    const p = features[i].properties;

    // --- COLOR ---
    let hex;
    if (colorMode === "classification") {
      const key = remapForestClass(p.classification);
      hex = (LAND_CLASSIFICATION_LEGEND[key] || ["#949C9F"])[0];
    } else {
      const key = p.subregion_code;
      hex = (SUBREGION_LEGEND[key] || ["#949C9F"])[0];
    }
    const c = new THREE.Color(hex);
    colors[i * 3 + 0] = c.r;
    colors[i * 3 + 1] = c.g;
    colors[i * 3 + 2] = c.b;

    // --- TARGET POSITIONS ---
    let tx, ty, tz;
    if (projection === "embedding") {
      if (dim_reduction === "tsne") {
        tx = p.tsne_3d_x;
        ty = p.tsne_3d_y;
        tz = p.tsne_3d_z;
      } else {
        tx = p.umap_3d_x;
        ty = p.umap_3d_y;
        tz = p.umap_3d_z;
      }
    } else {
      const spherePos = latLonToSphere(p.lat, p.lon);
      tx = spherePos.x;
      ty = spherePos.y;
      tz = spherePos.z;
    }

    targets.push({ x: tx, y: ty, z: tz });
  }

  geometry.attributes.color.needsUpdate = true;

  // --- ANIMATION ---
  const startPositions = [];
  for (let i = 0; i < N; i++) {
    startPositions.push({
      x: positions[i * 3 + 0],
      y: positions[i * 3 + 1],
      z: positions[i * 3 + 2],
    });
  }

  const startTime = performance.now();
  let animating = true;

  function frame(now) {
    const t = Math.min((now - startTime) / duration, 1);
    const eased = t < 0.5 ? 2 * t * t : -1 + (4 - 2 * t) * t;

    for (let i = 0; i < N; i++) {
      positions[i * 3 + 0] = THREE.MathUtils.lerp(
        startPositions[i].x,
        targets[i].x,
        eased
      );
      positions[i * 3 + 1] = THREE.MathUtils.lerp(
        startPositions[i].y,
        targets[i].y,
        eased
      );
      positions[i * 3 + 2] = THREE.MathUtils.lerp(
        startPositions[i].z,
        targets[i].z,
        eased
      );
    }

    geometry.attributes.position.needsUpdate = true;
    geometry.computeBoundingSphere();

    if (t < 1) {
      requestAnimationFrame(frame);
    } else {
      animating = false;
    }
  }

  requestAnimationFrame(frame);
}

function updateDimReductionVisibility() {
  const projection = document.querySelector('input[name="projection"]:checked').value;
  const section = document.getElementById("dim-reduction-section");
  section.style.display = projection === "embedding" ? "" : "none";
}

function updateLegend() {
  const colorMode = document.querySelector('input[name="color"]:checked').value;
  const legendTitle = document.getElementById("legend-title");
  legendTitle.innerHTML =
    colorMode === "classification" ? "Land Classification" : "Region";

  const legendBox = document.getElementById("legend");

  const data =
    colorMode === "classification"
      ? LAND_CLASSIFICATION_LEGEND
      : SUBREGION_LEGEND;

  legendBox.innerHTML = "";

  for (const key in data) {
    const [hex, name] = data[key];

    const item = document.createElement("div");
    item.className = "legend-item";

    item.innerHTML = `
      <div class="legend-color" style="background:${hex};"></div>
      <div>${name}</div>
    `;

    legendBox.appendChild(item);
  }
}

function checkHover() {
  if (!points || !geometry.attributes.position || !mouseInitialized) {
    return;
  }

  raycaster.setFromCamera(mouse, camera);
  const intersects = raycaster.intersectObject(points);

  // Debug: log every 100 frames
  if (Math.random() < 0.01) {
    console.log('Mouse:', mouse.x.toFixed(2), mouse.y.toFixed(2), 
                'Intersects:', intersects.length, 
                'Threshold:', raycaster.params.Points.threshold);
  }

  const tooltip = document.getElementById("tooltip");

  if (intersects.length > 0) {
    const intersect = intersects[0];
    const index = intersect.index;
    
    console.log('✓ Hovering point:', index, 'Distance:', intersect.distance.toFixed(2));
    
    // Safety check
    if (!features[index]) {
      console.warn('Invalid index:', index);
      return;
    }
    
    const props = features[index].properties;

    // Get classification name
    const classKey = remapForestClass(props.classification);
    const classificationName =
      (LAND_CLASSIFICATION_LEGEND[classKey] || ["", "Unknown"])[1];

    // Get region name
    const regionName =
      (SUBREGION_LEGEND[props.subregion_code] || ["", "Unknown"])[1];

    document.getElementById("tooltip-location").textContent = `${props.lat.toFixed(4)}°, ${props.lon.toFixed(4)}°`;
    document.getElementById("tooltip-classification").textContent = classificationName;
    document.getElementById("tooltip-region").textContent = regionName;

    tooltip.classList.add("visible");
    renderer.domElement.style.cursor = "pointer";
  } else {
    tooltip.classList.remove("visible");
    renderer.domElement.style.cursor = "default";
  }
}

function animate() {
  requestAnimationFrame(animate);

  controls.update();
  checkHover();
  renderer.render(scene, camera);
}

animate();
