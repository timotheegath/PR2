import numpy as np
import data_in_out as io
import torch
from scipy.spatial.distance import mahalanobis
from scipy import optimize
from scipy.optimize import minimize

if torch.cuda.is_available():
    Tensor = torch.cuda.FloatTensor
else:
    Tensor = torch.FloatTensor


def objective_function(parameters, features, labels):

    if isinstance(parameters, np.ndarray):
        parameters = torch.from_numpy(parameters).type(Tensor)
    if isinstance(features, np.ndarray):
        features = torch.from_numpy(features).type(Tensor)
    if isinstance(labels, np.ndarray):
        labels = torch.from_numpy(labels.astype(dtype=np.int32)).type(Tensor)

    shape = features.shape[0]
    L = parameters.view(shape, shape)
    # print(features.shape, L.shape)

    class_values, class_counts = np.unique(labels, return_counts=True)

    L_features = torch.mm(L, features)
    L_features_norm = (L_features**2).sum(0).view(1, -1)
    L_features_norm_t = L_features_norm.view(-1, 1)
    L_features_mm = torch.mm(L_features.transpose(1, 0), L_features)
    # print(L_features.shape, L_features_norm.shape, L_features_norm_t.shape, L_features_mm.shape)

    distances = L_features_norm + L_features_norm_t - 2.0 * L_features_mm
    distances = distances - torch.diag(distances.diag())

    print(parameters)
    label_mask = labels.view(1, -1) == labels.view(-1, 1)
    Dw = torch.masked_select(distances, label_mask)
    Db = torch.masked_select(distances, 1 - label_mask)
    objective = torch.sum(Dw) - torch.sum(Db)
    return objective


def optim_call(parameters):

    print(parameters)


if __name__ == '__main__':
    features = np.memmap('PR_data/features', mode='r', shape=(14096, 2048), dtype=np.float64)
    features = features.transpose()
    labels = io.get_ground_truth()

    train_ind = io.get_training_indexes() - 1
    training_features = features[:, train_ind]
    training_labels = labels[train_ind]
    values, counts = np.unique(training_labels, return_counts=True)
    parameters = torch.rand((training_features.shape[0], training_features.shape[0]), requires_grad=True)




    # objective_function(parameters, training_features, training_labels)

    optimizer = torch.optim.Adam([parameters], lr=0.1)

    for it in range(500):
        parameters_ = torch.tril(parameters).view(-1)
        loss = objective_function(parameters_, training_features, training_labels)
        print(loss)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


    # Opt = minimize(objective_function, parameters, (training_features, training_labels), callback=optim_call)





# import numpy as np
# from scipy.optimize import minimize
#
#
# def objective_function(parameters, features, labels):
#
#     dim = features.shape[0]
#
#     mask = labels[:, None] == labels[None, :]
#     same_class_features = np.argwhere(labels)
#     diff_class_features = np.argwhere(labels == False)
#
#     model_matrix = parameters.reshape([dim, dim])
#     distances_sum = 0
#     for i in range(same_class_features.shape[0]):
#         feature_distance = features[same_class_features[i, 0]] - features[same_class_features[i, 1]]
#         distances_sum += np.matmul(np.matmul(feature_distance.transpose, model_matrix, feature_distance))
#
#     return distances_sum
#
# def constraints(parameters, features, labels):
