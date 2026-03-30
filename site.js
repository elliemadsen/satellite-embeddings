document.addEventListener("DOMContentLoaded", () => {
  const tabs = document.querySelectorAll(".tab");
  const frames = document.querySelectorAll(".tab-frame");

  tabs.forEach((tab) => {
    tab.addEventListener("click", () => {
      const target = tab.dataset.tab;

      // Update active tab button
      tabs.forEach((t) => t.classList.remove("active"));
      tab.classList.add("active");

      // Show matching iframe, hide others
      frames.forEach((f) => {
        f.classList.toggle("active", f.id === `frame-${target}`);
      });

      // Close info popup when switching tabs
      document.getElementById("info-popup").classList.remove("visible");
      document.getElementById("info-btn").classList.remove("active");
    });
  });

  // Info popup toggle
  const infoBtn = document.getElementById("info-btn");
  const infoPopup = document.getElementById("info-popup");

  infoBtn.addEventListener("click", (e) => {
    e.stopPropagation();
    infoPopup.classList.toggle("visible");
    infoBtn.classList.toggle("active");
  });

  // Close popup when clicking outside
  document.addEventListener("click", (e) => {
    if (!infoPopup.contains(e.target) && e.target !== infoBtn) {
      infoPopup.classList.remove("visible");
      infoBtn.classList.remove("active");
    }
  });
});
