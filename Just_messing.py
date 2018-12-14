from sklearn.cluster import KMeans
import numpy as np
import torch
import metrics
import data_in_out as io


def cluster(labels, features):

    # num_clusters = np.unique(labels).shape[0]
    num_clusters = labels.shape[1]

    kMeans = KMeans(n_clusters=num_clusters, init=labels.transpose(), n_init=10, algorithm="auto")
    kMeans = kMeans.fit(features.transpose())
    clusters = kMeans.cluster_centers_
    cluster_labels = kMeans.labels_

    return clusters.transpose(), cluster_labels


def kmeans_accuracy(gallery_labels, query_labels, cluster_labels, feature_c_labels):

    score = []
    for i in range(cluster_labels.shape[0]):
        match_mask = gallery_labels[np.argwhere(feature_c_labels == cluster_labels[i])] == query_labels[i]
        score.append(np.sum(match_mask.astype(np.uint8), axis=0)/match_mask.shape[0])
        print (score[i])
    return score

if __name__ == '__main__':
    features = np.memmap('PR_data/features', mode='r', shape=(14096, 2048), dtype=np.float64)
    features = features.transpose()

    ground_truth = io.get_ground_truth()
    cam_ids = io.get_cam_ids()

    training_indices = io.get_training_indexes()
    training_features = features[:, training_indices]
    training_labels = ground_truth[training_indices]

    query_indices = io.get_query_indexes()
    query_features = features[:, query_indices]
    query_labels = ground_truth[query_indices]

    gallery_indices = io.get_gallery_indexes()

    gallery_features = features[:, gallery_indices]
    gallery_labels = ground_truth[gallery_indices]

    clusters, feature_c_labels = cluster(query_features, gallery_features)
    cluster_labels = np.arange(0, clusters.shape[1], 1)

    print(clusters, clusters.shape)
    print(feature_c_labels, feature_c_labels.shape)

    kmeans_accuracy(gallery_labels, query_labels, cluster_labels, feature_c_labels)

    print('Hello')