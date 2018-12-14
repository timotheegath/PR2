from sklearn.cluster import KMeans
import numpy as np
import torch
import metrics
import data_in_out as io
import evaluation as eval


def cluster(labels, features, initclusters='k-means++'):

    if initclusters == 'k-means++':
        num_clusters = np.unique(query_labels)
        num_clusters = num_clusters.shape[0]
    else:
        num_clusters = labels.shape[1]

    kMeans = KMeans(n_clusters=num_clusters, init=initclusters, n_init=10, algorithm="auto")
    kMeans = kMeans.fit(features.transpose())
    clusters = kMeans.cluster_centers_
    cluster_labels = kMeans.labels_

    return clusters.transpose(), cluster_labels


def kmeans_accuracy(query_labels, gallery_labels, cluster_labels, feature_c_labels, query_cam_ids, gallery_cam_ids):

    score = []
    for i in range(cluster_labels.shape[0]):
        local_neigh = np.argwhere(feature_c_labels == cluster_labels[i])
        match_mask = (gallery_labels[local_neigh] == query_labels[i]) & \
                     (gallery_cam_ids[local_neigh] != query_cam_ids[i])

        excluded = np.sum(((gallery_labels[local_neigh] == query_labels[i]) &
                    (gallery_cam_ids[local_neigh] == query_cam_ids[i])).astype(np.uint8), axis=0)

        score.append(np.sum(match_mask.astype(np.uint8), axis=0)/(match_mask.shape[0]-excluded))
        print(score[i])
    return score

# def kmeans_acc(query_features, clusters):



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
    query_cam_ids = cam_ids[query_indices]

    gallery_indices = io.get_gallery_indexes()
    gallery_features = features[:, gallery_indices]
    gallery_labels = ground_truth[gallery_indices]
    gallery_cam_ids = cam_ids[gallery_indices]

    clust_method = None

    if clust_method == 'query':
        clusters, feature_c_labels = cluster(query_features, gallery_features, initclusters=query_features.transpose())
        cluster_labels = np.arange(0, clusters.shape[1], 1)
        kmeans_accuracy(query_labels, gallery_labels, cluster_labels, feature_c_labels, query_cam_ids, gallery_cam_ids)
    else:
        clusters, feature_c_labels = cluster(query_labels, gallery_features)
        cluster_labels = np.arange(0, clusters.shape[1], 1)
        query_to_cluster_dist = metrics.minkowski_metric(query_features, 2, features_compare=clusters)

        distances = query_to_cluster_dist[:, feature_c_labels]
        _, score = eval.evaluate(1, distances, query_indices, gallery_indices, clusters=True)
    print(score)


    print(clusters, clusters.shape)
    print(feature_c_labels, feature_c_labels.shape)


    print('Hello')