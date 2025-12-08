const YEAR = 2023;
const SCALE = 1000;

const map = new maplibregl.Map({
  container: "map",
  style: { version: 8, sources: {}, layers: [] },
  center: [0, 0],
  zoom: 1.5,
  pitch: 0,
  bearing: 0,
  antialias: true,
  attributionControl: false,
});

map.on("style.load", () => {
  map.setProjection({ type: "globe" });

  // Add custom raster tile source
  map.addSource("alpha-earth", {
    type: "raster",
    tiles: [`tiles/${YEAR}_${SCALE}/{z}/{x}/{y}.png`],
    tileSize: 256,
    maxzoom: 8,
  });

  map.addLayer({
    id: "alpha-earth-layer",
    type: "raster",
    source: "alpha-earth",
    paint: { "raster-fade-duration": 150 },
  });

  // Hide any symbol layers
  map.getStyle().layers.forEach((layer) => {
    if (layer.type === "symbol")
      map.setLayoutProperty(layer.id, "visibility", "none");
  });
});
