import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score


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


def permutation_test_ari(labels_method_1, labels_method_2, n_permutations=100000):
    # Compute observed ARI
    observed_ari = adjusted_rand_score(labels_method_1, labels_method_2)

    permuted_aris = []
    count_greater = 0
    for _ in range(n_permutations):
        print(_)
        # Shuffle labels from one method
        permuted_labels = np.random.permutation(labels_method_1)
        # Compute ARI with permuted labels and second method
        permuted_ari = adjusted_rand_score(permuted_labels, labels_method_2)
        permuted_aris.append(permuted_ari)

        if permuted_ari >= observed_ari:
            count_greater += 1

    p_value = count_greater / n_permutations
    return observed_ari, p_value, permuted_aris