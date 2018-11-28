import numpy as np


def rank(r, distance_array):
    # expect distance_array query_images X gallery_images, numpy array
    ranked_winners = np.argsort(distance_array, axis=0)[:, :r]  # sort from smaller to bigger
    return ranked_winners