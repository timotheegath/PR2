import numpy as np
import torch
import metrics
import data_in_out as io

if torch.cuda.is_available():
    Tensor = torch.cuda.FloatTensor
else:
    Tensor = torch.FloatTensor

# def KNN_classifier(features, gallery_indices, query_indices, gallery_mask):
#
#     features_classify = features[:, gallery_indices]
#     features_query = features[:, query_indices]
#     query_distances = np.zeros((query_indices.shape[0], gallery_indices.shape[0]))
#
#     for i in range(query_indices.shape[0]):
#         print('HERE: ', i)
#         gallery_mask_temp = np.repeat(gallery_mask[i, None], features.shape[0], axis=0)
#         query_distances[i, :] = minkowski_metric(features_query[:, i], np.ma.masked_where(gallery_mask_temp, features_classify), 2)
#     query_distances = np.ma.masked_where(gallery_mask, query_distances)
#
#     return query_distances

def KNN_classifier(features, gallery_indices, query_indices, gallery_mask):

    features_classify = torch.from_numpy(features[:, gallery_indices]).type(Tensor)
    features_query = torch.from_numpy(features[:, query_indices]).type(Tensor)
    query_distances = torch.zeros((query_indices.shape[0], gallery_indices.shape[0])).type(Tensor)
    gallery_mask_t = torch.from_numpy(1 - gallery_mask.astype(np.uint8))
    if torch.cuda.is_available():
        gallery_mask_t = gallery_mask_t.cuda()

    print('Calculating nearest neighbours:')
    for i in range(query_indices.shape[0]):
        io.loading_bar(i, query_indices.shape[0])
        # gallery_mask_temp = np.repeat(gallery_mask[i, None], features.shape[0], axis=0)
        out_d = metrics.minkowski_metric(features_query[:, i],
                         torch.index_select(features_classify, 1, gallery_mask_t[i].nonzero()[:, 0]).type(Tensor), 2)
        query_distances[i, gallery_mask_t[i].nonzero()[:, 0]] = out_d
    query_distances = np.ma.masked_where(gallery_mask, query_distances.cpu().numpy())

    return query_distances


def rank(r, distance_array, gallery_indexes, removal_mask=None):
    # expect distance_array query_images X gallery_images, numpy masked array

    if removal_mask is not None:
        if isinstance(removal_mask, torch.Tensor):
            removal_mask = removal_mask.data.numpy()

        distance_array_m = np.ma.masked_where(removal_mask, distance_array)
    else:
        distance_array_m = distance_array
    ranked_winners = np.argsort(distance_array_m, axis=1)[:, :r]  # sort from smaller to bigger
    ranked_distances = np.sort(distance_array_m, axis=1)[:, :r]
    ranked_winners_true_ix = gallery_indexes[ranked_winners]
    return ranked_winners_true_ix, ranked_distances


def get_to_remove_mask(cam_id, query_indexes, gallery_index, g_t):
    query_cam_id = cam_id[query_indexes][:, None]
    gallery_cam_id = cam_id[gallery_index][None, :]
    query_label = g_t[query_indexes][:, None]
    gallery_label = g_t[gallery_index][None, :]

    to_remove = (gallery_label == query_label) & (query_cam_id == gallery_cam_id)
    # print(query_indexes.shape, gallery_index.shape)
    # print(to_remove.shape)
    return to_remove


def compute_score(rank, ground_truth, winner_indexes, query_indexes):

    query_labels = ground_truth[query_indexes]
    winner_labels = ground_truth[winner_indexes[:, :rank]]
    zero_if_match = winner_labels - query_labels[:, None]
    zero_bool = (zero_if_match==0).astype(np.uint8)
    number_of_positives = np.sum(zero_bool, axis=1)
    score_by_query = number_of_positives/rank
    total_score = np.mean(score_by_query)

    return score_by_query, total_score


def compute_mAP(rank, ground_truth, ranked_inds, query_inds):

    query_labels = ground_truth[query_inds]
    ranked_labels = ground_truth[ranked_inds[:, :rank]]
    match_mask = ranked_labels == query_labels[:, None]
    num_of_correct = np.sum((match_mask == 1).astype(np.uint8), axis=1)

    seen_imgs = np.arange(1, rank+1)
    seen_imgs = 1/seen_imgs

    match_coordinates = np.argwhere(match_mask).transpose()

    score_mask = np.copy(match_mask).astype(np.uint8)
    score_mask[1 - match_mask] = 0
    score_mask[match_coordinates[0], match_coordinates[1]] = match_coordinates[1]

    scores = score_mask * seen_imgs[None, :]
    query_scores = np.divide(np.sum(scores, axis=1), num_of_correct, out=np.zeros(num_of_correct.shape),
                             where=num_of_correct != 0)
    total_score = np.sum(query_scores)/query_scores.shape

    return total_score, query_scores


if __name__ == '__main__':
    cam_ids = io.get_cam_ids()
    query_indices = io.get_query_indexes()
    gallery_indices = io.get_gallery_indexes()
    g_ts = io.get_ground_truth()
    get_to_remove_mask(cam_ids, query_indices, gallery_indices, g_ts)
