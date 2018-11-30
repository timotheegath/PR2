import numpy as np


def rank(r, distance_array, gallery_indexes):
    # expect distance_array query_images X gallery_images, numpy masked array

    ranked_winners = np.argsort(distance_array, axis=1)[:, :r]  # sort from smaller to bigger
    ranked_distances = np.sort(distance_array, axis=1)[:, :r]
    ranked_winners_true_ix = gallery_indexes[ranked_winners]
    return ranked_winners_true_ix, ranked_distances


def get_to_remove_mask(cam_id, query_indexes, gallery_index, g_t):
    query_cam_id = cam_id[query_indexes][:, None]
    gallery_cam_id = cam_id[gallery_index][None, :]
    query_label = g_t[query_indexes][:, None]
    gallery_label = g_t[gallery_index][None, :]

    to_remove = (gallery_label == query_label) & (query_cam_id == gallery_cam_id)
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
