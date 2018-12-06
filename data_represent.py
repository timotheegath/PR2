import numpy as np
import matplotlib.pyplot as plt
import torch


def build_histogram(features):

    dimensions = features.shape[0]
    hist = np.histogramdd(features.transpose(), density=True)
    print(hist.shape)

    return hist


def build_covariance(features):

    cov = np.corrcoef(features)
    plt.imshow(cov)
    plt.waitforbuttonpress()


def normalise_features(features):

    norms = torch.norm(features, 2, 1, keepdim=True)
    return features/norms
