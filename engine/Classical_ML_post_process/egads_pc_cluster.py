"""Principal-component clustering + deviation metric for anomalous-series
detection, following EGADS Section 3.3 and the threshold selection of Section
4.1.

    Laptev, Amizadeh & Flint, KDD 2015 (DOI 10.1145/2783258.2788611).

EGADS Section 3.3 ("Detecting Anomalous Time-series"):

    "In EGADS our current approach involves clustering the time-series into a
     set of clusters C based on various time-series features including trend &
     seasonality, spectral entropy, autocorrelation, average Euclidean distance
     etc. After clustering we perform intra or inter-cluster time-series anomaly
     detection by measuring the deviation within or among the cluster centroids
     and the time-series (i)."

This module receives the feature matrix F (one row per series/window, columns =
the ``egads_features`` catalogue) and:

    1. standardises the features (z-score) so no single feature dominates,
    2. projects them onto their leading principal components -- the "PC space"
       in which clustering is done (dimension reduction on the feature matrix,
       as in EGADS reference [29]),
    3. clusters the projected points with k-means (k chosen by silhouette),
    4. computes the *deviation metric*: each point's distance to its own
       cluster centroid (intra-cluster) and, optionally, the gap to the nearest
       *other* centroid (inter-cluster),
    5. turns the deviation metric into a binary anomaly flag with the two
       Section 4.1 threshold rules: parametric K-sigma and non-parametric
       density (Local Outlier Factor).

The parametric K-sigma rule is EGADS' "three-sigma rule": flag a series when
its deviation metric lies more than K standard deviations above the mean
deviation. The density rule uses LOF (EGADS cites Breunig et al. LOF, [3]) to
find low-density points in PC space directly.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.neighbors import LocalOutlierFactor


# ------------------------------------------------------------------- k choice
def choose_k(pcs, k_range=range(2, 8), random_state=0):
    """Pick the number of clusters by maximising the silhouette score.

    Returns k=1 when there are too few points to form two meaningful clusters.
    """
    n = pcs.shape[0]
    best_k, best_s = 1, -1.0
    for k in k_range:
        if k >= n:
            break
        labels = KMeans(n_clusters=k, n_init=10,
                        random_state=random_state).fit_predict(pcs)
        if len(np.unique(labels)) < 2:
            continue
        s = silhouette_score(pcs, labels)
        if s > best_s:
            best_k, best_s = k, s
    return best_k


@dataclass
class PCClusterAnomalyModel:
    """EGADS anomalous-series detector: PCA -> k-means -> centroid deviation.

    Parameters
    ----------
    n_components : int or float
        PCA dimensionality. An int keeps that many PCs; a float in (0, 1] keeps
        enough PCs to explain that fraction of variance.
    k : int or None
        Number of k-means clusters; ``None`` selects it by silhouette.
    inter_cluster : bool
        If True the deviation metric adds the inverse margin to the nearest
        *other* centroid, so a point sitting between clusters also scores high
        (EGADS "inter-cluster" deviation). If False, pure intra-cluster
        distance to the assigned centroid is used.
    k_sigma : float
        K in the parametric K-sigma threshold rule (Section 4.1).
    lof_neighbors : int
        Neighbourhood size for the density (LOF) threshold rule.
    """

    n_components: float = 0.9
    k: int | None = None
    inter_cluster: bool = True
    k_sigma: float = 3.0
    lof_neighbors: int = 20
    lof_contamination: float | str = "auto"
    random_state: int = 0

    # fitted state
    mean_: np.ndarray = field(default=None, repr=False)
    std_: np.ndarray = field(default=None, repr=False)
    pca_: PCA = field(default=None, repr=False)
    kmeans_: KMeans = field(default=None, repr=False)
    pcs_: np.ndarray = field(default=None, repr=False)
    deviation_: np.ndarray = field(default=None, repr=False)
    k_used_: int = field(default=None, repr=False)

    # ----------------------------------------------------------------- fitting
    def _standardize(self, F, fit):
        if fit:
            self.mean_ = np.nanmean(F, axis=0)
            self.std_ = np.nanstd(F, axis=0)
            self.std_[self.std_ < 1e-12] = 1.0
        Z = (F - self.mean_) / self.std_
        return np.nan_to_num(Z, nan=0.0, posinf=0.0, neginf=0.0)

    def fit(self, F):
        """Fit standardisation, PCA, and k-means on the feature matrix F."""
        F = np.asarray(F, dtype=float)
        Z = self._standardize(F, fit=True)

        max_pc = min(Z.shape)
        if isinstance(self.n_components, float) and 0 < self.n_components <= 1:
            ncomp = self.n_components
        else:
            ncomp = min(int(self.n_components), max_pc)
        self.pca_ = PCA(n_components=ncomp, random_state=self.random_state)
        self.pcs_ = self.pca_.fit_transform(Z)

        self.k_used_ = self.k or choose_k(self.pcs_,
                                          random_state=self.random_state)
        if self.k_used_ < 1:
            self.k_used_ = 1
        self.kmeans_ = KMeans(n_clusters=self.k_used_, n_init=10,
                              random_state=self.random_state).fit(self.pcs_)
        self.deviation_ = self._deviation(self.pcs_,
                                          self.kmeans_.predict(self.pcs_))
        return self

    # ------------------------------------------------------- deviation metric
    def _deviation(self, pcs, labels):
        """Distance to the assigned centroid (+ inter-cluster margin)."""
        centroids = self.kmeans_.cluster_centers_
        own = np.linalg.norm(pcs - centroids[labels], axis=1)
        if not self.inter_cluster or self.k_used_ < 2:
            return own
        # distance to every centroid, then the nearest *other* one
        dall = np.linalg.norm(pcs[:, None, :] - centroids[None, :, :], axis=2)
        dall_sorted = np.sort(dall, axis=1)
        nearest_other = dall_sorted[:, 1]
        # a point deep inside its cluster has own << nearest_other -> low score;
        # a point stranded between clusters has own ~ nearest_other -> high.
        margin = own / (nearest_other + 1e-12)
        return own * (1.0 + margin)

    def deviation_metric(self, F):
        """Deviation metric for new rows F under the fitted model."""
        Z = self._standardize(np.asarray(F, dtype=float), fit=False)
        pcs = self.pca_.transform(Z)
        labels = self.kmeans_.predict(pcs)
        return self._deviation(pcs, labels)

    # ----------------------------------------------------- Section 4.1 thresholds
    def ksigma_threshold(self, deviation=None):
        """Parametric three-sigma rule: mean + K * std of the deviation metric."""
        d = self.deviation_ if deviation is None else deviation
        return float(d.mean() + self.k_sigma * d.std())

    def flag_ksigma(self, F=None):
        """Binary anomaly flags via the K-sigma rule (Section 4.1a)."""
        d = self.deviation_ if F is None else self.deviation_metric(F)
        return d > self.ksigma_threshold(self.deviation_)

    def flag_density(self, F=None):
        """Binary anomaly flags via the density rule (LOF, Section 4.1b).

        LOF is fit in the PCA space; points whose local density is much lower
        than their neighbours' are flagged (negative_outlier_factor below the
        LOF offset). Returns flags plus the LOF scores (higher = more outlying).
        """
        if F is None:
            pcs = self.pcs_
        else:
            Z = self._standardize(np.asarray(F, dtype=float), fit=False)
            pcs = self.pca_.transform(Z)
        n_neigh = min(self.lof_neighbors, max(2, pcs.shape[0] - 1))
        lof = LocalOutlierFactor(n_neighbors=n_neigh,
                                 contamination=self.lof_contamination)
        pred = lof.fit_predict(pcs)          # -1 outlier, +1 inlier
        scores = -lof.negative_outlier_factor_
        return pred == -1, scores

    # --------------------------------------------------------------- convenience
    def fit_flag(self, F, rule="ksigma"):
        """Fit and return anomaly flags + the continuous deviation score.

        rule : "ksigma" (parametric) or "density" (LOF).
        """
        self.fit(F)
        if rule == "density":
            flags, scores = self.flag_density()
            return flags, scores
        return self.flag_ksigma(), self.deviation_

    def explained_variance(self):
        """Fraction of feature variance captured by the retained PCs."""
        return self.pca_.explained_variance_ratio_
