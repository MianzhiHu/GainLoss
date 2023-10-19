import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans


def wcss(X, labels):
    """Compute within-cluster sum of squares."""
    n_clusters = len(np.unique(labels))
    cluster_means = np.array([X[labels == i].mean(axis=0) for i in range(n_clusters)])
    return np.sum([(X[i] - cluster_means[labels[i]]) ** 2 for i in range(X.shape[0])])


def gap_statistic(X, max_clusters, b=1000, tol=1e-4):
    """Compute the Gap statistic for a range of cluster numbers."""
    gaps = []
    sks = []

    for k in range(1, max_clusters + 1):
        # Actual data clustering
        kmeans = KMeans(n_clusters=k, n_init='auto', tol=tol).fit(X)
        actual_log_w = np.log(wcss(X, kmeans.labels_))

        # Random data clustering
        random_ws = []
        for _ in range(b):
            print(_)
            random_data = np.random.random_sample(size=X.shape)
            random_kmeans = KMeans(n_clusters=k, n_init='auto', tol=tol).fit(random_data)
            random_ws.append(np.log(wcss(random_data, random_kmeans.labels_)))

        E_log_w = np.mean(random_ws)
        sk = np.std(random_ws) * np.sqrt(1 + 1/b)
        sks.append(sk)

        gap = E_log_w - actual_log_w
        gaps.append(gap)

    optimal_k = None
    for i in range(1, len(gaps) - 1):  # Start from 1 to avoid k=1
        if gaps[i] >= gaps[i + 1] - sks[i + 1]:
            optimal_k = i + 1  # +1 because the cluster numbers start from 1
            break

    return optimal_k, gaps, sks