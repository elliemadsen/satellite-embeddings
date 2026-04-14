"""
Compress 64-dim satellite embeddings to 2D with multiple algorithms
and save one clean scatter plot PNG per algorithm to outputs/.
"""

import json
import base64
import os
import argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib import font_manager
from pathlib import Path

# ── CLI flags ─────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument("--bw",      action="store_true", help="Black & white scatter (ignore class colours)")
parser.add_argument("--n",       type=int, default=None, help="Number of points to use (default: all)")
parser.add_argument("--cluster", action="store_true", help="Color points by k-means cluster instead of land-cover class")
parser.add_argument("--k",       type=int, default=10,  help="Number of k-means clusters (default: 10)")
args = parser.parse_args()

# ── data ──────────────────────────────────────────────────────────────────────
GEOJSON = "../../dimension-reduction/data/20000_sampled_classified_embeddings.geojson"
OUTPUT_DIR = "outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

with open(GEOJSON) as f:
    data = json.load(f)

embeddings, labels = [], []
for feat in data["features"]:
    props = feat["properties"]
    emb = json.loads(base64.b64decode(props["embedding"]).decode())
    embeddings.append(emb)
    labels.append(props["classification"])

X = np.array(embeddings, dtype=np.float32)
y = np.array(labels)

# Apply --n
if args.n is not None:
    X = X[:args.n]
    y = y[:args.n]

# Output subdir encodes the run config
_mode = "bw" if args.bw else (f"cluster{args.k}" if args.cluster else "color")
_tag = _mode + (f"_{args.n}" if args.n else "_all")
OUTPUT_DIR = os.path.join("outputs", _tag)
os.makedirs(OUTPUT_DIR, exist_ok=True)
classes = sorted(set(y.tolist()))
class_to_idx = {c: i for i, c in enumerate(classes)}
y_idx = np.array([class_to_idx[c] for c in y])

# Colour palette — one colour per land-cover class
cmap_color = plt.cm.get_cmap("tab20", len(classes))

# K-means clustering on raw embeddings (when --cluster is set)
if args.cluster:
    from sklearn.cluster import KMeans
    print(f"Running k-means (k={args.k}) on raw embeddings...")
    _km = KMeans(n_clusters=args.k, random_state=42, n_init="auto")
    cluster_labels = _km.fit_predict(X)
    cmap_cluster = plt.cm.get_cmap("tab20", args.k)
    print(f"  k-means done.")

print(f"Loaded {len(X)} samples, {X.shape[1]}-dim embeddings, {len(classes)} classes")
print(f"Mode: {'black & white' if args.bw else ('cluster k=' + str(args.k) if args.cluster else 'color')}, n={len(X)}")

# ── Roboto font ───────────────────────────────────────────────────────────────
def _find_roboto():
    import subprocess
    search_paths = [
        "/usr/share/fonts/truetype/roboto/Roboto-Regular.ttf",
        "/System/Library/Fonts/Supplemental/Roboto-Regular.ttf",
        str(Path.home() / "Library/Fonts/Roboto-Regular.ttf"),
    ]
    for p in search_paths:
        if os.path.exists(p):
            return p
    try:
        result = subprocess.run(
            ["fc-list", ":family=Roboto", "--format=%{file}\n"],
            capture_output=True, text=True
        )
        for line in result.stdout.splitlines():
            line = line.strip()
            if line and os.path.exists(line):
                return line
    except Exception:
        pass
    return None

_roboto_path = _find_roboto()
if _roboto_path:
    font_manager.fontManager.addfont(_roboto_path)
    LABEL_FONT = font_manager.FontProperties(fname=_roboto_path, size=13)
else:
    LABEL_FONT = font_manager.FontProperties(size=13)

# ── plot helper ───────────────────────────────────────────────────────────────
DOT   = 4      # marker size
ALPHA = 0.7
LABEL_PAD = 0.02   # axes fraction below scatter for label

def save_scatter(coords_2d, name, label):
    fig = plt.figure(figsize=(7, 7.4), dpi=150)
    # Scatter occupies top portion; leave room at bottom for label
    ax = fig.add_axes([0, 0.06, 1, 0.94])

    if args.bw:
        ax.scatter(
            coords_2d[:, 0], coords_2d[:, 1],
            c="black", s=DOT, alpha=ALPHA, linewidths=0,
        )
    elif args.cluster:
        ax.scatter(
            coords_2d[:, 0], coords_2d[:, 1],
            c=cluster_labels, cmap=cmap_cluster, s=DOT, alpha=ALPHA, linewidths=0,
        )
    else:
        ax.scatter(
            coords_2d[:, 0], coords_2d[:, 1],
            c=y_idx, cmap=cmap_color, s=DOT, alpha=ALPHA, linewidths=0,
        )

    ax.set_axis_off()
    fig.patch.set_facecolor("white")

    # Algorithm label below the scatter
    fig.text(
        0.5, 0.02, label,
        ha="center", va="bottom",
        fontproperties=LABEL_FONT,
        color="#1e1e1e",
    )

    out = os.path.join(OUTPUT_DIR, f"{name}.png")
    fig.savefig(out, bbox_inches="tight", pad_inches=0.05)
    plt.close(fig)
    print(f"  ✓ {out}")

# ── algorithms ────────────────────────────────────────────────────────────────

# 1. PCA
print("Running PCA...")
from sklearn.decomposition import PCA
pca = PCA(n_components=2, random_state=42)
save_scatter(pca.fit_transform(X), "pca", "Principal Component Analysis (PCA)")

# 2. t-SNE
print("Running t-SNE...")
from sklearn.manifold import TSNE
tsne = TSNE(n_components=2, perplexity=40, max_iter=1000, random_state=42)
save_scatter(tsne.fit_transform(X), "tsne", "t-Distributed Stochastic Neighbor Embedding (t-SNE)")

# 3. UMAP
print("Running UMAP...")
import umap
reducer = umap.UMAP(n_components=2, n_neighbors=30, min_dist=0.1, random_state=42)
save_scatter(reducer.fit_transform(X), "umap", "Uniform Manifold Approximation and Projection (UMAP)")

# 4. Isomap
print("Running Isomap...")
from sklearn.manifold import Isomap
iso = Isomap(n_components=2, n_neighbors=15)
save_scatter(iso.fit_transform(X), "isomap", "Isometric Mapping (Isomap)")

# 5. Locally Linear Embedding (LLE)
print("Running LLE...")
from sklearn.manifold import LocallyLinearEmbedding
lle = LocallyLinearEmbedding(n_components=2, n_neighbors=15, random_state=42)
save_scatter(lle.fit_transform(X), "lle", "Locally Linear Embedding (LLE)")

# 6. Spectral Embedding
print("Running Spectral Embedding...")
from sklearn.manifold import SpectralEmbedding
se = SpectralEmbedding(n_components=2, n_neighbors=15, random_state=42)
save_scatter(se.fit_transform(X), "spectral", "Laplacian Eigenmap (Spectral Embedding)")

# 7. Landmark MDS — classical MDS on k landmarks, Nyström extension to all N points
print("Running MDS...")

def landmark_mds(X, n_landmarks=1000, random_state=42):
    rng = np.random.default_rng(random_state)
    X = X.astype(np.float64)
    n = len(X)
    lm_idx = rng.choice(n, size=min(n_landmarks, n), replace=False)
    X_lm = X[lm_idx]

    # L2-normalise so dot product == cosine similarity
    def _l2norm(A):
        norms = np.linalg.norm(A, axis=1, keepdims=True)
        return A / np.where(norms > 1e-10, norms, 1.0)

    X_n  = _l2norm(X)
    Xlm_n = _l2norm(X_lm)

    # Squared cosine distance between landmarks: (1 - cos)²
    cos_lm = np.clip(Xlm_n @ Xlm_n.T, -1.0, 1.0)
    D2_lm = (1.0 - cos_lm) ** 2

    # Double-centre to get the Gram matrix
    mu = D2_lm.mean(axis=0)
    grand_mean = mu.mean()
    B = -0.5 * (D2_lm - mu[None, :] - mu[:, None] + grand_mean)

    # Top-2 eigenpairs
    eigvals, eigvecs = np.linalg.eigh(B)
    top2 = np.argsort(eigvals)[::-1][:2]
    lam = eigvals[top2]
    V = eigvecs[:, top2]
    sqrt_lam = np.sqrt(np.maximum(lam, 0.0))
    inv_sqrt_lam = np.where(sqrt_lam > 1e-10, 1.0 / sqrt_lam, 0.0)

    # Nyström projection for all N points using cosine distance to landmarks
    cos_all = np.clip(X_n @ Xlm_n.T, -1.0, 1.0)   # (N, k)
    D2_all = (1.0 - cos_all) ** 2
    Y = 0.5 * ((mu[None, :] - D2_all) @ V) * inv_sqrt_lam[None, :]
    return Y

print(f"  (landmark MDS: {min(1000, len(X))} landmarks, projecting all {len(X)} points)")
save_scatter(landmark_mds(X, n_landmarks=1000, random_state=42), "mds", "Multidimensional Scaling (MDS)")

# 8. Truncated SVD (LSA-style, no centering)
print("Running Truncated SVD...")
from sklearn.decomposition import TruncatedSVD
svd = TruncatedSVD(n_components=2, random_state=42)
save_scatter(svd.fit_transform(X), "truncated_svd", "Truncated Singular Value Decomposition (SVD)")

# 9. Kernel PCA (RBF)
print("Running Kernel PCA...")
from sklearn.decomposition import KernelPCA
kpca = KernelPCA(n_components=2, kernel="rbf", random_state=42)
save_scatter(kpca.fit_transform(X), "kernel_pca", "Kernel Principal Component Analysis (RBF)")

print("\nDone.")
