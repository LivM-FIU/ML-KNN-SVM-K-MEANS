from pathlib import Path
import sys

import numpy as np
from sklearn.cluster import KMeans

sys.path.append(str(Path(__file__).resolve().parents[1]))

from HW4 import trimmed_inlier_mask


def test_trimmed_inlier_mask_marks_far_points_as_outliers():
    rng = np.random.default_rng(0)
    inliers = rng.normal(0, 0.3, size=(20, 2))
    outlier = np.array([[5.0, 5.0]])
    X = np.vstack([inliers, outlier])

    mask = trimmed_inlier_mask(X, fraction_to_trim=0.05)

    # Only one point should be trimmed and it must be the far outlier
    assert mask.sum() == X.shape[0] - 1
    assert not bool(mask[-1])


def test_kmeans_centroid_is_closer_to_origin_after_trimming_outliers():
    rng = np.random.default_rng(1)
    inliers = rng.normal(0, 0.4, size=(50, 2))
    outlier = np.array([[6.0, 6.0]])
    X = np.vstack([inliers, outlier])

    mask = trimmed_inlier_mask(X, fraction_to_trim=0.05)

    km_all = KMeans(n_clusters=1, n_init=10, random_state=0).fit(X)
    km_trimmed = KMeans(n_clusters=1, n_init=10, random_state=0).fit(X[mask])

    center_all = km_all.cluster_centers_[0]
    center_trimmed = km_trimmed.cluster_centers_[0]
    true_center = inliers.mean(axis=0)

    dist_all = np.linalg.norm(center_all - true_center)
    dist_trimmed = np.linalg.norm(center_trimmed - true_center)

    assert not bool(mask[-1])  # the single outlier should be removed
    assert dist_trimmed < dist_all
