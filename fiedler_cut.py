import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_openml
from sklearn.neighbors import NearestNeighbors, radius_neighbors_graph
from scipy.linalg import eigh
from scipy.sparse.linalg import eigsh
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
from skimage.feature import hog
from skimage.color import rgb2gray
from sklearn.preprocessing import StandardScaler
from scipy.sparse.csgraph import laplacian

# ======================================================
# HOG Feature Extraction
# ======================================================
def extract_hog_features(X_flat):
    hog_features = []
    for i in range(X_flat.shape[0]):
        img = X_flat[i].reshape(28, 28)
        img = rgb2gray(img) if img.ndim == 3 else img
        features = hog(
            img,
            orientations=9,
            pixels_per_cell=(4, 4),
            cells_per_block=(2, 2),
            block_norm='L2-Hys',
            feature_vector=True
        )
        hog_features.append(features)
    hog_features = np.array(hog_features)
    return StandardScaler().fit_transform(hog_features)

# ======================================================
# Load MNIST Dataset
# ======================================================
mnist = fetch_openml('mnist_784', version=1, as_frame=False)
X, y = mnist['data'], mnist['target'].astype(int)

# Using first 1000 images for faster computation
X_orig = X[:1000]  # keep original images for visualization
X = extract_hog_features(X_orig)  # extract HOG features
y = y[:1000]

# ======================================================
# Build kNN Graph
# ======================================================
def build_knn_graph(X, k=10, sigma=None):
    """
    Build a weighted kNN graph using Gaussian kernel.
    - X: feature matrix
    - k: number of neighbors
    - sigma: kernel width; if None, use median distance
    Returns symmetric weighted adjacency matrix
    """
    nn = NearestNeighbors(n_neighbors=k+1)  # include self
    nn.fit(X)
    distances, indices = nn.kneighbors(X)
    
    n = X.shape[0]
    A = np.zeros((n, n))
    
    # Determine sigma if not provided
    if sigma is None:
        # take median of all distances (excluding self=0)
        sigma = np.median(distances[:, 1:])
    
    for i in range(n):
        for idx, j in enumerate(indices[i][1:]):  # skip self
            d = distances[i][idx+1]  # corresponding distance
            weight = np.exp(-d**2 / (2 * sigma**2))  # Gaussian kernel
            A[i, j] = weight
            A[j, i] = weight  # symmetric

    return A

# ======================================================
# Conductance Computation
# ======================================================
def compute_conductance(A_sub, mask1, mask2):
    """
    Compute conductance of a cut defined by mask1 and mask2.
    
    Conductance = cut(S, S') / min(vol(S), vol(S'))
    """
    cut_weight = np.sum(A_sub[np.ix_(mask1, mask2)])
    vol1 = np.sum(A_sub[mask1, :])
    vol2 = np.sum(A_sub[mask2, :])
    min_vol = min(vol1, vol2)
    if min_vol == 0:
        return float('inf')
    return cut_weight / min_vol

# ======================================================
# Recursive Fiedler Segmentation
# ======================================================

def recursive_fiedler(A, nodes=None, normalized=False, lambda2_thresh=None, conductance_thresh=0.3, verbose=False):
    """
    Minimal recursive Fiedler bisection.
    Stopping rule: λ2 > lambda2_thresh or conductance > conductance_thresh
    """
    if nodes is None:
        nodes = np.arange(A.shape[0])
    nodes = np.asarray(nodes)
    n_nodes = len(nodes)
    
    # Stop trivially small clusters
    if n_nodes <= 2:
        return [nodes]
    
    # Subgraph
    A_sub = A[np.ix_(nodes, nodes)]
    degs = A_sub.sum(axis=1)
    
    # Laplacian
    if not normalized:
        L = np.diag(degs) - A_sub
    else:
        with np.errstate(divide='ignore'):
            inv_sqrt = 1.0 / np.sqrt(degs + 1e-12)
        inv_sqrt[~np.isfinite(inv_sqrt)] = 0.0
        D_inv_sqrt = np.diag(inv_sqrt)
        L = np.eye(n_nodes) - D_inv_sqrt @ A_sub @ D_inv_sqrt
    
    # Eigen decomposition
    vals, vecs = eigh(L)
    lambda2 = vals[1]
    
    if verbose:
        print(f"n={n_nodes} λ2={lambda2:.4f}")
    
    # Stop if λ2 is too large (cluster is tight)
    if lambda2_thresh:
        if lambda2 > lambda2_thresh:
            return [nodes]
    
    # Split by Fiedler sign
    fiedler = vecs[:, 1]
    mask1 = fiedler > 0
    mask2 = fiedler <= 0
    cluster1 = nodes[mask1]
    cluster2 = nodes[mask2]
    
    # Stop if split is trivial
    if len(cluster1) == 0 or len(cluster2) == 0:
        return [nodes]
    
    # Check conductance if threshold provided
    if conductance_thresh:
        conductance = compute_conductance(A_sub, mask1, mask2)
        if verbose:
            print(f"  conductance={conductance:.4f}")
        if conductance > conductance_thresh:
            return [nodes]
    
    # Recurse
    return recursive_fiedler(A, cluster1, normalized, lambda2_thresh, conductance_thresh, verbose) + \
           recursive_fiedler(A, cluster2, normalized, lambda2_thresh, conductance_thresh, verbose)

# ======================================================
# Evaluate Clustering
# ======================================================
def evaluate_clustering(X, y, clusters, title=None):
    labels_pred = np.zeros(X.shape[0], dtype=int)
    for i, cluster in enumerate(clusters):
        labels_pred[cluster] = i
    ari = adjusted_rand_score(y, labels_pred)
    nmi = normalized_mutual_info_score(y, labels_pred)
    if title:
        print(f"\n--- {title} ---")
    print(f"Number of clusters: {len(clusters)}")
    print(f"Adjusted Rand Index (ARI): {ari:.4f}")
    print(f"Normalized Mutual Information (NMI): {nmi:.4f}")
    return ari, nmi

# ======================================================
# Compute Fiedler Value
# ======================================================
def compute_fiedler_value(A, normalized=True):
    """
    Compute the Fiedler (2nd smallest) eigenvalue of the Laplacian of a graph.

    Parameters
    ----------
    A : np.ndarray
        Adjacency matrix (symmetric, non-negative)
    normalized : bool
        Whether to use the normalized Laplacian

    Returns
    -------
    lambda2 : float
        The second-smallest eigenvalue (Fiedler value)
    """
    if A.shape[0] < 2:  # trivial case
        return np.nan

    L = laplacian(A, normed=normalized)
    vals = eigh(L, eigvals_only=True)
    return vals[1]  # λ₂