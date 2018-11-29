import numpy as np
import data_in_out as io
import torch
from scipy.io import loadmat
from sklearn.neighbors import KNeighborsClassifier as KNNC
import evaluation as eval

if torch.cuda.is_available():
    Tensor = torch.cuda.FloatTensor
else:
    Tensor = torch.FloatTensor

def minkowski_metric(x, y, p):

    x = x.unsqueeze(dim=1)
    
    distances = (y - x)
    distances = torch.pow(distances, p)
    distances = torch.sum(distances, dim=0)
    # distances = (distances ** 1/p)

    return distances


def KNN_classifier(features, gallery_indices, query_indices, gallery_mask):

    features_classify = torch.from_numpy(features[:, gallery_indices]).type(Tensor)
    features_query = torch.from_numpy(features[:, query_indices]).type(Tensor)
    query_distances = torch.zeros((query_indices.shape[0], gallery_indices.shape[0])).type(Tensor)
    gallery_mask = torch.from_numpy(1 - gallery_mask.astype(np.uint8))
    if torch.cuda.is_available():
        gallery_mask = gallery_mask.cuda()


    for i in range(query_indices.shape[0]):
        print(i)
        # gallery_mask_temp = np.repeat(gallery_mask[i, None], features.shape[0], axis=0)
        out_d = minkowski_metric(features_query[:, i],
                         torch.index_select(features_classify, 1, gallery_mask[i].nonzero()[:, 0]).type(Tensor), 2)
        query_distances[i, gallery_mask[i].nonzero()[:, 0]] = out_d
    query_distances = np.ma.masked_where(gallery_mask.cpu().numpy(), query_distances.cpu().numpy())

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
    # print(distances_query)
