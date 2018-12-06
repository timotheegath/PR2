import numpy as np
import data_in_out as io
import torch
import evaluation as eval
import metrics
import matplotlib.pyplot as plt

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
    ground_truth = io.get_ground_truth()
    features = features.transpose()

    cam_ids = io.get_cam_ids()
    query_indices = io.get_query_indexes()
    gallery_indices = io.get_gallery_indexes()
    training_index = io.get_training_indexes() - 1
    g_ts = io.get_ground_truth()
    metrics.optimize_torch(features, training_index, ground_truth, 1000)

    gallery_mask = eval.get_to_remove_mask(cam_ids, query_indices, gallery_indices, g_ts)
    distances_query = eval.KNN_classifier(features, gallery_indices, query_indices, gallery_mask)
    ranked_winners, ranked_distances = eval.rank(10, distances_query, gallery_indices)
    io.display_ranklist(query_indices, ranked_winners, 10, 3)

    l_score, score = eval.compute_score(10, ground_truth, ranked_winners, query_indices)
    print(l_score, score)
