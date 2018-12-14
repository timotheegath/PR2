import numpy as np
import torch
import data_in_out as io

if torch.cuda.is_available():
    Tensor = torch.cuda.FloatTensor
else:
    Tensor = torch.FloatTensor

def to_numpy(*arrays):
    np_arrays = []
    for array in arrays:
        if isinstance(array, torch.Tensor):
            np_arrays.append(array.clone().detach().cpu().numpy())
        else:
            np_arrays.append(array)
    if len(arrays) is 1:
        np_arrays = np_arrays[0]
    return np_arrays

def rank(r, distance_array, gallery_indexes, removal_mask=None, flip=False):
    # expect distance_array query_images X gallery_images, numpy masked array

    if removal_mask is not None:
        if isinstance(removal_mask, torch.Tensor):
            removal_mask = removal_mask.data.numpy()

        distance_array_m = np.ma.masked_where(removal_mask, distance_array)
    else:
        distance_array_m = distance_array

    if flip is False:
        ranked_winners = np.argsort(distance_array_m, axis=1)   # sort from smaller to bigger
        ranked_distances = np.sort(distance_array_m, axis=1)

    if flip is True:
        ranked_winners = np.argsort(distance_array_m*(-1), axis=1)
        ranked_distances = - np.sort(distance_array_m*(-1), axis=1)

    ranked_winners = ranked_winners[:, :r]
    ranked_distances = ranked_distances[:, :r]

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
    query_correct = np.cumsum(match_mask.astype(np.uint8), axis=1)

    num_of_correct = np.max(query_correct, axis=1)
    seen_imgs = np.arange(1, rank+1)
    seen_imgs = 1/seen_imgs

    # match_coordinates = np.argwhere(match_mask).transpose()
    # score_mask = np.copy(match_mask).astype(np.uint8)
    # score_mask[1 - match_mask] = 0
    # score_mask[match_coordinates[0], match_coordinates[1]] = match_coordinates[1]
    score_mask = np.copy(query_correct)
    score_mask[match_mask == 0] = 0
    scores = score_mask * seen_imgs[None, :]
    query_scores = np.divide(np.sum(scores, axis=1), num_of_correct, out=np.zeros(num_of_correct.shape),
                             where=num_of_correct != 0)
    total_score = np.sum(query_scores)/query_scores.shape

    return total_score, query_scores

def evaluate(r, distances, query_ind, gallery_ind, clusters=False, flip=False):

    distances, query_ind, gallery_ind = to_numpy(distances, query_ind, gallery_ind)
    query_ind, gallery_ind = query_ind.astype(np.int32), gallery_ind.astype(np.int32)
    cam_ids = io.get_cam_ids()
    g_t = io.get_ground_truth()
    to_remove_mask = get_to_remove_mask(cam_ids, query_ind, gallery_ind, g_t)
    distances = np.ma.masked_where(to_remove_mask, distances)
    if flip is False:
        ranked_winners = np.argsort(distances, axis=1)   # sort from smaller to bigger
        ranked_distances = np.sort(distances, axis=1)

    if flip is True:
        ranked_winners = np.argsort(distances*(-1), axis=1)
        ranked_distances = - np.sort(distances*(-1), axis=1)

    ranked_local_winners = ranked_winners[:, :r]
    # ranked_distances = ranked_distances[:, :r]

    ranked_winners = gallery_ind[ranked_winners]

    query_labels = g_t[query_ind]
    if not clusters:
        ranked_labels = g_t[ranked_winners[:, :r]]
    else:
        ranked_labels = np.ma.masked_where(ranked_distances != ranked_distances[:, 0, None], g_t[ranked_winners])
    match_mask = ranked_labels == query_labels[:, None]
    query_correct = np.cumsum(match_mask.astype(np.uint8), axis=1)

    num_of_correct = np.max(query_correct, axis=1)
    seen_imgs = np.arange(1, r + 1)
    seen_imgs = 1 / seen_imgs

    score_mask = np.copy(query_correct)
    score_mask[match_mask == 0] = 0
    scores = score_mask * seen_imgs[None, :]
    query_scores = np.divide(np.sum(scores, axis=1), num_of_correct, out=np.zeros(num_of_correct.shape),
                             where=num_of_correct != 0)
    total_score = np.sum(query_scores) / query_scores.shape

    return ranked_winners, total_score

