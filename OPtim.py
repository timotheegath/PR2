import numpy as np
import data_in_out as io
import evaluation as eval
import torch
import metrics
import data_represent as dare

if torch.cuda.is_available():
    Tensor = torch.cuda.FloatTensor
else:
    Tensor = torch.FloatTensor

# ----------------------------------------------------------------------------------------------------------------------

def mahalanobis_metric(parameters, features, features_compare = None):
    parameters = parameters[0]
    if isinstance(parameters, np.ndarray):
        parameters = torch.from_numpy(parameters).type(Tensor)
    if isinstance(features, np.ndarray):
        features = torch.from_numpy(features).type(Tensor)

    if isinstance(features_compare, np.ndarray):
        features_compare = torch.from_numpy(features_compare).type(Tensor)

    shape = features.shape[0]

    L = parameters.view(shape, shape)

    L_features = torch.mm(L, features)

    L_features_norm = (L_features ** 2).sum(0).view(1, -1)

    if features_compare is not None:

        L_features_compare = torch.mm(L, features_compare)

        L_features_compare_t = torch.transpose(L_features_compare, 1, 0)

        L_features_compare_norm = (L_features_compare ** 2).sum(0).view(-1, 1)

    else:

        L_features_compare_t = torch.transpose(L_features, 1, 0)

        L_features_compare_norm = L_features_norm.view(-1, 1)

    L_features_mm = torch.mm(L_features_compare_t, L_features)

    distances = L_features_norm + L_features_compare_norm - 2.0 * L_features_mm

    if features_compare is None:
        distances = distances - torch.diag(distances.diag())

    return distances.transpose(1, 0)

def gaussian_Maha(parameters, features, features_compare=None):

    if isinstance(parameters[0], np.ndarray):
        parameters[0] = torch.from_numpy(parameters[0]).type(Tensor)
    if isinstance(features, np.ndarray):
        features = torch.from_numpy(features).type(Tensor)

    if isinstance(features_compare, np.ndarray):
        features_compare = torch.from_numpy(features_compare).type(Tensor)
    sigma = parameters[1]
    shape = features.shape[0]
    denom = torch.sqrt((2 * sigma ** 2).unsqueeze(1))
    L = torch.mm(parameters[0].view(shape, shape), torch.eye(shape, shape) / denom)

    L_features = torch.mm(L, features)

    L_features_norm = (L_features ** 2).sum(0).view(1, -1)

    if features_compare is not None:

        L_features_compare = torch.mm(L, features_compare)

        L_features_compare_t = torch.transpose(L_features_compare, 1, 0)

        L_features_compare_norm = (L_features_compare ** 2).sum(0).view(-1, 1)

    else:

        L_features_compare_t = torch.transpose(L_features, 1, 0)

        L_features_compare_norm = L_features_norm.view(-1, 1)

    L_features_mm = torch.mm(L_features_compare_t, L_features)
    distances = 2 - 2 * (torch.exp(-(L_features_norm + L_features_compare_norm - 2 * L_features_mm)))  # RBF kernel
    if features_compare is None:
        distances = distances - torch.diag(distances.diag())

    return distances.transpose(1, 0)

def poly_Maha(parameters, features, features_compare=None):

    if isinstance(parameters[0], np.ndarray):
        parameters[0] = torch.from_numpy(parameters[0]).type(Tensor)
    if isinstance(features, np.ndarray):
        features = torch.from_numpy(features).type(Tensor)

    if isinstance(features_compare, np.ndarray):
        features_compare = torch.from_numpy(features_compare).type(Tensor)
    p = parameters[1]
    shape = features.shape[0]

    L = parameters[0].view(shape, shape)

    L_features = torch.mm(L, features)

    L_features_norm = (L_features ** 2).sum(0).view(1, -1)

    if features_compare is not None:

        L_features_compare = torch.mm(L, features_compare)

        L_features_compare_t = torch.transpose(L_features_compare, 1, 0)

        L_features_compare_norm = (L_features_compare ** 2).sum(0).view(-1, 1)

    else:

        L_features_compare_t = torch.transpose(L_features, 1, 0)

        L_features_compare_norm = L_features_norm.view(-1, 1)

    L_features_mm = torch.mm(L_features_compare_t, L_features)
    distances = L_features_norm ** p + L_features_compare_norm ** p - 2*(L_features_mm ** p) #polynomial kernel
    if features_compare is None:
        distances = distances - torch.diag(distances.diag())

    return distances.transpose(1, 0)

def objective_function(parameters, lagrangian, features, labels = None, features_compare = None, kernel=None):
    if kernel is not None:
        param2 = parameters[1]

    if isinstance(parameters[0], np.ndarray):
        parameters[0] = torch.from_numpy(parameters[0]).type(Tensor)

    if isinstance(features, np.ndarray):
        features = torch.from_numpy(features).type(Tensor)
    if isinstance(features_compare, np.ndarray):
        features_compare = torch.from_numpy(features_compare).type(Tensor)


    if isinstance(labels, np.ndarray):
        labels = torch.from_numpy(labels.astype(dtype=np.int32)).type(Tensor)


    if kernel is 'RBF':
        distances  = gaussian_Maha(parameters, features, features_compare)
    elif kernel is 'poly':
        distances = poly_Maha(parameters, features, features_compare)
    else:
        distances = mahalanobis_metric(parameters, features, features_compare=None)



    distances = distances - torch.diag(distances.diag())
    label_mask = labels.view(1, -1) == labels.view(-1, 1)
    Dw = torch.masked_select(distances, label_mask) - 0.1
    Db = torch.masked_select(distances, 1 - label_mask)

    # lagrangian = torch.masked_select(lagrangian, 1 - label_mask)
    objective = lagrangian*torch.sum(Dw) - torch.sum(torch.sqrt(Db))
    return objective, distances.clone().detach().cpu().numpy()



def optim_call(parameters):

    print(parameters)


def constraint_distances(features):

    if isinstance(features, np.ndarray):
        features = torch.from_numpy(features).type(Tensor)

    shape0 = features.shape[0]
    shape1 = features.shape[1]
    print('Here1')
    differences = torch.empty(shape0, shape1, shape1)
    print('Here2')
    for i in range(shape1):
        differences[:, 0:i, i] = 0
        differences[:, i:, i] = features[:, i].view(-1, 1) - features[:, i:]
        print(differences.shape)
    return differences

# ----------------------------------------------------------------------------------------------------------------------



if __name__ == '__main__':
    BATCHIFY = True
    KERNEL = 'RBF'
    batch_size = 2000
    # Feature loading
    features = np.memmap('PR_data/features', mode='r', shape=(14096, 2048), dtype=np.float64)
    features = features.transpose()

    ground_truth = io.get_ground_truth()
    cam_ids = io.get_cam_ids()
    train_ind = io.get_training_indexes()
    if not BATCHIFY:

        training_features = features[:, train_ind]
        training_labels = ground_truth[train_ind]

    query_ind = io.get_query_indexes()
    query_features = features[:, query_ind]
    query_labels = ground_truth[query_ind]

    gallery_ind = io.get_gallery_indexes()
    gallery_features = features[:, gallery_ind]
    gallery_labels = ground_truth[gallery_ind]

    removal_mask = eval.get_to_remove_mask(cam_ids, query_ind, gallery_ind, ground_truth)

    parameters = []
    # PARAMETERS DEFINITION
    matrix = torch.zeros((features.shape[0], features.shape[0]), requires_grad=True)
    matrix.data = torch.from_numpy(np.linalg.inv(np.cov(features[:, train_ind]))* np.random.rand(features.shape[0], features.shape[0])).type(Tensor)
    parameters.append(matrix)
    # For gaussian kernel
    if KERNEL is 'RBF':
        param2 = torch.rand((features.shape[0],), requires_grad=True)
        param2.data = (param2.data+0.5) * 3000
        parameters.append(param2)
    elif KERNEL is 'poly':
        param2 = torch.full((1,), 1, requires_grad=True)
        parameters.append(param2)



    lagrangian = torch.full((1,), 10, requires_grad=True)

    optimizer = torch.optim.Adagrad(parameters, lr=2)

    for it in range(500):


        # print('Sigma: ', sigma_)
        if BATCHIFY:
            temp_index = np.arange(0, train_ind.shape[0]).astype(np.int32)

            np.random.shuffle(temp_index)
            temp_index = temp_index[:batch_size]
            train_ix = train_ind[temp_index].astype(np.int32)
            # lagrangian_ = lagrangian[temp_index, temp_index]
            training_features = torch.from_numpy(features[:, train_ix]).type(Tensor)

            training_labels = ground_truth[train_ix]
        else:
            train_ix = train_ind

        parameters[0] = torch.tril(parameters[0])

        loss, distances = objective_function(parameters, lagrangian, training_features, labels=training_labels, kernel=KERNEL)
        optimizer.zero_grad()
        training_features.cpu()
        torch.cuda.empty_cache()
        loss.backward()
        optimizer.step()
        with torch.no_grad():
            if KERNEL is 'RBF':
                test_distances = gaussian_Maha(parameters, query_features, features_compare=gallery_features)
            elif KERNEL is 'poly':
                test_distances = poly_Maha(parameters, query_features, features_compare=gallery_features)
            else:
                test_distances = mahalanobis_metric(parameters, query_features, features_compare=gallery_features)
            ranked_idx_train, _ = eval.rank(10, distances, train_ix)
            ranked_idx_test, _ = eval.rank(10, test_distances.clone().detach().cpu().numpy(), gallery_ind, removal_mask=removal_mask)

            total_score_t, query_scores_t = eval.compute_mAP(10, ground_truth, ranked_idx_train, train_ix)
            total_score, query_scores = eval.compute_mAP(10, ground_truth, ranked_idx_test, query_ind)

        print(loss)
        print(total_score_t, total_score)
        # print('p:', sigma)
