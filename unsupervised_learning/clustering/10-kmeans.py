#!/usr/bin/env python3
"""
a function def kmeans(X, k): that performs K-means on a dataset
"""
import sklearn.cluster


def kmeans(X, k):
    """
    Perform K-means clustering on a dataset.

    Args:
        X (numpy.ndarray): Input dataset of
        shape (n_samples, n_features).
        k (int): Number of clusters to form.

    Returns:
        Tuple[numpy.ndarray, numpy.ndarray]:
            - C (numpy.ndarray): Centroid means for each
            cluster of shape (k, n_features).
            - clss (numpy.ndarray): Index of the cluster in C that
            each data point belongs to, shape (n_samples,).
    """
    kmeans = \
        sklearn.cluster.KMeans(n_clusters=k)
    kmeans.fit(X)
    C = kmeans.cluster_centers_
    clss = kmeans.labels_

    return C, clss
