import torch
import numpy as np

if torch.cuda.is_available():
    Tensor = torch.cuda.FloatTensor
else:
    Tensor = torch.FloatTensor

def to_torch(*arrays):
    torch_arrays = []
    for array in arrays:
        if isinstance(array, np.ndarray):
            torch_arrays.append(torch.from_numpy(array).type(Tensor))
        else:
            torch_arrays.append(array)
    if len(arrays) is 1:
        torch_arrays = torch_arrays[0]
    return torch_arrays


def compute_vector_terms(features, features_compare=None):

    features_norm = (features ** 2).sum(0).view(1, -1)

    if features_compare is not None:

        features_compare_norm = (features_compare ** 2).sum(0).view(-1, 1)

    else:

        features_compare = features
        features_compare_norm = features_norm.view(-1, 1)

    features_mm = torch.mm(features_compare.transpose(1, 0), features)

    return features_norm, features_compare_norm, features_mm


"""TRAINABLE metrics"""


def mahalanobis_metric(parameters, features, features_compare=None):

    L = torch.triu(parameters['L'])
    L, features, features_compare = to_torch(L, features, features_compare)

    L_features = torch.mm(L, features)
    if features_compare is not None:
        L_features_compare = torch.mm(L, features_compare)
    else:
        L_features_compare = None

    L_features_norm, L_features_compare_norm, L_features_mm = compute_vector_terms(L_features, L_features_compare)
    # Decomposition of the Mahalanobis distance
    distances = L_features_norm + L_features_compare_norm - 2.0 * L_features_mm

    if features_compare is None: # Enforce 0 distance on diagonal
        distances = distances - torch.diag(distances.diag())

    return distances.transpose(1, 0)


def gaussian_Maha(parameters, features, features_compare=None):
    L = torch.triu(parameters['L'])
    sigma = parameters['sigma']
    denom = 2 * sigma ** 2

    L, features, features_compare = to_torch(L, features, features_compare)
    print(type(features))
    L_features = torch.mm(L, features)
    if features_compare is not None:
        L_features_compare = torch.mm(L, features_compare)
    else:
        L_features_compare = None

    L_features_norm, L_features_compare_norm, L_features_mm = compute_vector_terms(L_features, L_features_compare)

    distances = 2 - 2 * (torch.exp(-(L_features_norm + L_features_compare_norm - 2 * L_features_mm)/denom)) # RBF kernel

    if features_compare is None:

        distances = distances - torch.diag(distances.diag())

    return distances.transpose(1, 0)


def poly_Maha(parameters, features, features_compare=None):
    L = torch.triu(parameters['L'])
    p = parameters['p']
    L, features, features_compare = to_torch(L, features, features_compare)

    L_features = torch.mm(L, features)

    if features_compare is not None:
        L_features_compare = torch.mm(L, features_compare)
    else:
        L_features_compare = None

    L_features_norm, L_features_compare_norm, L_features_mm = compute_vector_terms(L_features, L_features_compare)
    distances = L_features_norm ** p + L_features_compare_norm ** p - 2*(L_features_mm ** p)  # polynomial kernel
    if features_compare is None:
        distances = distances - torch.diag(distances.diag())

    return distances.transpose(1, 0)

"""Non-trainable metrics"""

def BilinearSimilarity(parameters, features, features_compare=None):

    A, features, features_compare = to_torch(parameters['A'], features, features_compare)
    if features_compare is None:
        features_compare = features

    similarity = torch.mm(torch.mm(features.transpose(1, 0), A), features_compare)
    distances = 1 - (similarity/torch.max(similarity.view(-1)))
    return distances

def cosine_similarity(features, features_compare=None):

    if isinstance(features, np.ndarray):
        features = torch.from_numpy(features).type(Tensor)
    if isinstance(features_compare, np.ndarray):
        features_compare = torch.from_numpy(features_compare).type(Tensor)

    if features_compare is None:
        features_compare = features

    Kcos = torch.mm(features.transpose(1, 0), features_compare)
    features_norm = torch.norm(features, 2, dim=1, keepdim=True)
    features_compare_norm = torch.norm(features_compare, 2, dim=1, keepdim=True)
    cross_norm_matrix = torch.mm(features_norm.transpose(1, 0), features_compare_norm)
    distances = Kcos/cross_norm_matrix

    return 1 - distances/torch.max(distances.view(-1))


def cross_correlation(features, features_compare=None):

    if isinstance(features, np.ndarray):
        features = torch.from_numpy(features).type(Tensor)
    if isinstance(features_compare, np.ndarray):
        features_compare = torch.from_numpy(features_compare).type(Tensor)

    if features_compare is None:
        features_compare = features

    features_norm = features/torch.norm(features, 2, dim=1, keepdim=True)
    features_compare_norm = features_compare/torch.norm(features_compare, 2, dim=1, keepdim=True)

    distances = torch.mm(features_norm.transpose(1, 0), features_compare_norm)
    return distances


def minkowski_metric(features, p=1, features_compare=None, max=False):

    if isinstance(features, np.ndarray):
        features = torch.from_numpy(features).type(Tensor)
    if isinstance(features_compare, np.ndarray):
        features_compare = torch.from_numpy(features_compare).type(Tensor)

    if features_compare is None:
        features_compare = features

    distances = torch.empty(features.shape[1], features_compare.shape[1])

    if max is False:
        for i in range(features.shape[1]):
            distance = torch.pow(torch.abs(features[:, i].view(-1, 1) - features_compare), p)
            distances[i, :] = distance.sum(0)
    if max is True:
        distance = torch.abs(features[:, i] - features_compare)
        distances[i, :] = distance.max(0)

    return distances



