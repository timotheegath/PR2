import numpy as np
import data_in_out as io
import torch
import evaluation as eval
import metrics
import matplotlib.pyplot as plt
from OPtim import mahalanobis_metric

if torch.cuda.is_available():
    Tensor = torch.cuda.FloatTensor
else:
    Tensor = torch.FloatTensor

# def build_histogram(features):
#
#     dimensions = features.shape[0]
#     hist = np.histogramdd(features.transpose(), density=True)
#     print(hist.shape)
#
#     return hist
#
# def build_covariance(features):
#
#     cov = np.corrcoef(features)
#     plt.imshow(cov)
#     plt.waitforbuttonpress()



# def KNN_classifier(features, gallery_indices, query_indices, gallery_mask):
#
#     features_classify = torch.from_numpy(features[:, gallery_indices]).type(Tensor)
#     features_query = torch.from_numpy(features[:, query_indices]).type(Tensor)
#     query_distances = torch.zeros((query_indices.shape[0], gallery_indices.shape[0])).type(Tensor)
#     gallery_mask_t = torch.from_numpy(1 - gallery_mask.astype(np.uint8))
#     if torch.cuda.is_available():
#         gallery_mask_t = gallery_mask_t.cuda()
#
#     print('Calculating nearest neighbours:')
#     for i in range(query_indices.shape[0]):
#         io.loading_bar(i, query_indices.shape[0])
#         # gallery_mask_temp = np.repeat(gallery_mask[i, None], features.shape[0], axis=0)
#         out_d = metrics.minkowski_metric(features_query[:, i],
#                          torch.index_select(features_classify, 1, gallery_mask_t[i].nonzero()[:, 0]).type(Tensor), 2)
#         query_distances[i, gallery_mask_t[i].nonzero()[:, 0]] = out_d
#     query_distances = np.ma.masked_where(gallery_mask, query_distances.cpu().numpy())
#
#     return query_distances


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

    removal_mask = eval.get_to_remove_mask(cam_ids, query_indices, gallery_indices, ground_truth)

    # test_distances = metrics.minkowski_metric(query_features, p=2, features_compare=gallery_features)

    parameters = torch.rand((training_features.shape[0], training_features.shape[0]), requires_grad=True)
    parameters.data = torch.from_numpy(np.linalg.inv(np.cov(training_features))).type(Tensor)
    parameters = torch.tril(parameters).view(-1)

    test_distances = mahalanobis_metric(parameters, query_features, features_compare=gallery_features)


    rank = 10
    ranked_inds_test, _ = eval.rank(rank, test_distances.clone().detach().numpy(), gallery_indices, removal_mask=removal_mask)
    total_score, query_scores = eval.compute_mAP(rank, ground_truth, ranked_inds_test, query_indices)
    print(query_scores[456],query_scores[122], query_scores[186], total_score)
    display_inds = np.array([456, 122, 186])
    io.display_ranklist(query_indices, ranked_inds_test, rank, 3, override_choice=display_inds)

