import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from utilities.utility_k_means import gap_statistic

# Read in the data
data = pd.read_csv('./data/data.csv')
CAoptimal = data[data['ChoiceSet'] == 'CA']['PropOptimal'].to_numpy().reshape(-1, 1)
trimodal_assignments_CA = pd.read_csv('./data/trimodal_assignments_CA.csv')['assignments'].to_numpy()

# # first, we need to find the optimal number of clusters
# # as expected, the optimal number of clusters is 3
#
# # we first examine the elbow method and silhouette scores
# WCSS = []  # Within-Cluster Sum of Squares
# silhouette_scores = []
#
# # Evaluate k-means for k from 1 to 10
# for k in range(1, 11):
#     kmeans = KMeans(n_clusters=k, n_init='auto', tol=1e-10, random_state=42).fit(CAoptimal)
#     WCSS.append(kmeans.inertia_)
#
#     if k > 1:
#         silhouette_scores.append(silhouette_score(CAoptimal, kmeans.labels_))
#
# # visualize the results
# plt.figure(figsize=(12, 6))
#
# plt.subplot(1, 2, 1)
# plt.plot(range(1, 11), WCSS, marker='o', linestyle='--')
# plt.title('Elbow Method')
# plt.xlabel('Number of clusters')
# plt.ylabel('WCSS')
#
# plt.subplot(1, 2, 2)
# plt.plot(range(2, 11), silhouette_scores, marker='o', linestyle='--')
# plt.title('Silhouette Scores')
# plt.xlabel('Number of clusters')
# plt.ylabel('Silhouette Score')
#
# plt.tight_layout()
# plt.show()
#
# # use the gap statistic proposed by Tibshirani et al. (2001)
# optimal_k, gaps, sks = gap_statistic(CAoptimal, max_clusters=10, tol=1e-10)
# print(f"Optimal number of clusters based on Gap statistic: {optimal_k}")


# now we know the optimal number of clusters is 3, we can fit the model
kmeans = KMeans(n_clusters=3, n_init='auto', tol=1e-10, random_state=42).fit(CAoptimal)

# compare the labels with the results from the EM algorithm
kmean_labels = kmeans.labels_ + 1
print(kmeans.cluster_centers_)
print(kmeans.inertia_)

# we can see that group 2 and 3 are switched
# we can switch them back
kmean_labels[kmean_labels == 2], kmean_labels[kmean_labels == 3] = -1, 2  # temporarily set group 2 to -1
kmean_labels[kmean_labels == -1] = 3  # set group -1 to 3

# combine into a dataframe
# we can see that the conclusion is generally the same, albeit some minor differences (22%)
kmean_results = pd.DataFrame({'PropOptimal': CAoptimal.flatten(), 'trimodal_assignments': trimodal_assignments_CA,
                                'kmean_labels': kmean_labels})

divergence = np.sum(kmean_results['trimodal_assignments'] != kmean_results['kmean_labels']) / len(kmean_results)
print(f"Proportion of divergent results: {divergence}")








# # now let's incorporate the RT
# # this is bad as hell
# CAoptimal_RT = data[data['ChoiceSet'] == 'CA'][['PropOptimal', 'RT']].to_numpy().reshape(-1, 2)
#
# # # holy shit, the optimal number of clusters is 5
# # optimal_k_RT, gaps_RT, sks_RT = gap_statistic(CAoptimal_RT, max_clusters=10, tol=1e-10)
# # print(f"Optimal number of clusters based on Gap statistic: {optimal_k_RT}")
#
# kmeans_dual = KMeans(n_clusters=5, n_init='auto', tol=1e-10, random_state=42).fit(CAoptimal_RT)
#
# print(kmeans_dual.cluster_centers_)
# print(kmeans_dual.inertia_)
#
# kmeans_dual_labels = kmeans_dual.labels_
#
# # Scatter plot
# for cluster_num in range(5):
#     cluster_data = CAoptimal_RT[kmeans_dual.labels_ == cluster_num]
#     plt.scatter(cluster_data[:, 0], cluster_data[:, 1], label=f'Cluster {cluster_num + 1}')
#
# # Mark the cluster centers
# centers = kmeans_dual.cluster_centers_
# plt.scatter(centers[:, 0], centers[:, 1], c='red', marker='X', s=100, label='Cluster Centers')
#
# plt.title("K-means Clustering with 2 Variables")
# plt.xlabel('Variable 1')
# plt.ylabel('Variable 2')
# plt.legend()
# plt.grid(True)
# plt.show()

