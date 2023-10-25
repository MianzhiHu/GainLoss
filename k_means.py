import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, adjusted_rand_score
from utilities.utility_k_means import gap_statistic, permutation_test_ari
import scipy.stats as stats

# Read in the data
data = pd.read_csv('./data/data.csv')
CAoptimal = data[data['ChoiceSet'] == 'CA']['PropOptimal'].to_numpy().reshape(-1, 1)
BDoptimal = data[data['ChoiceSet'] == 'BD']['PropOptimal'].to_numpy().reshape(-1, 1)
trimodal_assignments_CA = pd.read_csv('./data/trimodal_assignments_CA.csv')['assignments'].to_numpy()

# first, we need to find the optimal number of clusters
# as expected, the optimal number of clusters is 3 for both CA and BD

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
#
# # visualize the gap statistic
# plt.plot(range(1, 11), gaps, marker='o', linestyle='--')
# plt.title('Gap Statistic')
# plt.xlabel('Number of clusters')
# plt.ylabel('Gap')
#
# plt.tight_layout()
# plt.show()


# now we know the optimal number of clusters is 3, we can fit the model
kmeans_CA = KMeans(n_clusters=3, n_init='auto', tol=1e-10, random_state=42).fit(CAoptimal)
kmeans_BD = KMeans(n_clusters=3, n_init='auto', tol=1e-10, random_state=42).fit(BDoptimal)

# compare the labels with the results from the EM algorithm
kmean_labels_CA = kmeans_CA.labels_ + 1
kmean_labels_BD = kmeans_BD.labels_ + 1

print(kmeans_BD.cluster_centers_)
print(kmeans_CA.inertia_)

# we can see that group 2 and 3 are switched
# we can switch them back
kmean_labels_CA[kmean_labels_CA == 2], kmean_labels_CA[kmean_labels_CA == 3] = -1, 2  # temporarily set group 2 to -1
kmean_labels_CA[kmean_labels_CA == -1] = 3  # set group -1 to 3

# switch for BD
kmean_labels_BD[kmean_labels_BD == 1] = -1
kmean_labels_BD[kmean_labels_BD == 2] = 1
kmean_labels_BD[kmean_labels_BD == 3] = 2
kmean_labels_BD[kmean_labels_BD == -1] = 3

# combine into a dataframe
# we can see that the conclusion is generally the same, albeit some minor differences (22%)
kmean_results = pd.DataFrame({'CAOptimal': CAoptimal.flatten(), 'trimodal_assignments': trimodal_assignments_CA,
                                'kmean_labels_CA': kmean_labels_CA, 'BDOptimal': BDoptimal.flatten(), 'kmean_labels_BD': kmean_labels_BD})

divergence = np.sum(kmean_results['kmean_labels_CA'] != kmean_results['trimodal_assignments']) / len(kmean_results)
print(f"Proportion of divergent results: {divergence}")

# check the percentage of each group
print(kmean_results['kmean_labels_CA'].value_counts() / len(kmean_results))
print(kmean_results['kmean_labels_BD'].value_counts() / len(kmean_results))
print(kmean_results['trimodal_assignments'].value_counts() / len(kmean_results))

# # conduct a permutation test on ari to see if the difference is significant
# # ari = 0.411, p-value = 0.000, with 100k permutations
# observed_ari, p_value, permuted_results = permutation_test_ari(kmean_results['kmean_labels_CA'],
#                                                                kmean_results['kmean_labels_BD'])
#
# print(f"Observed ARI: {observed_ari}")
# print(f"P-value: {p_value}")





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

