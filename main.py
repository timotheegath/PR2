import numpy as np
import data_in_out as io
from scipy.io import loadmat
from sklearn.neighbors import KNeighborsClassifier as KNNC
import evaluation as eval
import metrics


def minkowski_metric(x, y, p):

    distances = - (y - x[:, None])
    distances = (distances ** p)
    distances = np.sum(distances, axis=0)
    # distances = (distances ** 1/p)

    return distances


def KNN_classifier(features, gallery_indices, query_indices, gallery_mask):

    features_classify = features[:, gallery_indices]
    features_query = features[:, query_indices]
    query_distances = np.zeros((query_indices.shape[0], gallery_indices.shape[0]))

    for i in range(query_indices.shape[0]):
        print('HERE: ', i)
        gallery_mask_temp = np.repeat(gallery_mask[i, None], features.shape[0], axis=0)
        query_distances[i, :] = minkowski_metric(features_query[:, i], np.ma.masked_where(gallery_mask_temp, features_classify), 2)
    query_distances = np.ma.masked_where(gallery_mask, query_distances)

    return query_distances

# def KNN_classifier(features, gallery_indices, query_indices, gallery_mask):
#
#     gallery_mindices = np.repeat(gallery_indices[None, :], query_indices.shape[0], axis=0)
#     gallery_mindices = np.ma.masked_where(gallery_mask, gallery_mindices)
#
#     features_classify = features[gallery_indices]
#     features_classify = np.repeat(features_classify.transpose()[None, ...], query_indices.shape[0], axis=0)
#     features_classify = features_classify[gallery_mindices[..., None]]
#
#     #labels_classify = labels[gallery_indices]
#     features_query = features[query_indices]
#     print(features_query.shape, features_classify.shape)
#
#     #distance = minkowski_metric(features_query, features_classify, 2)
#     return None

if __name__ == '__main__':

    features = np.memmap('PR_data/features', mode='r', shape=(14096, 2048), dtype=np.float64)
    features = features.transpose()

    # data = loadmat('PR_data/cuhk03_new_protocol_config_labeled.mat')
    cam_ids = io.get_cam_ids()
    query_indices = io.get_query_indexes()
    gallery_indices = io.get_gallery_indexes()
    g_ts = io.get_ground_truth()

    gallery_mask = eval.get_to_remove_mask(cam_ids, query_indices, gallery_indices, g_ts)
    distances_query = KNN_classifier(features, gallery_indices, query_indices, gallery_mask)
    print(distances_query.shape)
