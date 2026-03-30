const YEAR = 2023;
const SCALE = 1000;

const TILE_URL = `https://media.githubusercontent.com/media/elliemadsen/satellite-embeddings/refs/heads/main/web-maps/tiles/${YEAR}_${SCALE}/{z}/{x}/{y}.png`;

// ── Band selector state (re-enable when triplet tiles are ready) ──
// let currentBandStart = 0;
// let mapReady = false;
//
// function getTileUrl(bandStart) {
//   const triplet = `${bandStart}_${bandStart + 1}_${bandStart + 2}`;
//   return `tiles/${triplet}/{z}/{x}/{y}.png`;
// }
//
// function updateTileSource() {
//   const url = getTileUrl(currentBandStart);
//   console.log("[Globe] Updating tiles → band", currentBandStart, "URL:", url);
//   if (map.getLayer("alpha-earth-layer")) map.removeLayer("alpha-earth-layer");
//   if (map.getSource("alpha-earth")) map.removeSource("alpha-earth");
//   map.addSource("alpha-earth", { type: "raster", tiles: [url], tileSize: 256, maxzoom: 8 });
//   map.addLayer({ id: "alpha-earth-layer", type: "raster", source: "alpha-earth", paint: { "raster-fade-duration": 150 } });
// }

const map = new maplibregl.Map({
  container: "map",
  style: { version: 8, sources: {}, layers: [] },
  center: [0, 0],
  zoom: 2.4,
  pitch: 0,
  bearing: 0,
  antialias: true,
  attributionControl: false,
});

map.on("style.load", () => {
  map.setProjection({ type: "globe" });
  // mapReady = true;  // re-enable with band selector

  map.addSource("alpha-earth", {
    type: "raster",
    tiles: [TILE_URL],
    tileSize: 256,
    maxzoom: 8,
  });

  map.addLayer({
    id: "alpha-earth-layer",
    type: "raster",
    source: "alpha-earth",
    paint: { "raster-fade-duration": 150 },
  });

  map.getStyle().layers.forEach((layer) => {
    if (layer.type === "symbol")
      map.setLayoutProperty(layer.id, "visibility", "none");
  });
});

map.on("error", (e) => {
  console.error("[Globe] Map error:", e.error);
});

// ── Band selector interaction — re-enable when triplet tiles are ready ──
// (function initBandSelector() {
//   const selector = document.getElementById("band-selector");
//   const scale = document.getElementById("band-scale");
//   const handle = document.getElementById("band-handle");
//   ... (see git history for full implementation)
// })();


