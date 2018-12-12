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

    # L = parameters.view(shape, shape)
    L = parameters

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
    denom = 2 * sigma**2
    # denom = torch.sqrt((2 * sigma ** 2).unsqueeze(1))
    # L = torch.mm(parameters[0].view(shape, shape), torch.eye(shape, shape) / denom)
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
    # distances = 2 - 2 * (torch.exp(-(L_features_norm + L_features_compare_norm - 2 * L_features_mm)))  # RBF kernel
    distances = 2 - 2 * (torch.exp(-(L_features_norm + L_features_compare_norm - 2 * L_features_mm)/denom))  # RBF kernel

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


def objective_function(parameters, lagrangian, features, slack = 0, labels = None, features_compare = None, kernel=None):

    if kernel is not None:
        min_distance = 0.1
    else:
        min_distance = 1

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

    objective = lossC(distances, labels, lagrangian, slack)

    return objective, distances.clone().detach().cpu().numpy()

def lossA(distances, labels):
    distances = distances - torch.diag(distances.diag())
    label_mask = labels.view(1, -1) == labels.view(-1, 1)
    Dw = torch.masked_select(distances, label_mask) - 1
    Db = torch.masked_select(distances, 1 - label_mask)

    objective = lagrangian*torch.sum(Dw) - torch.sum(torch.sqrt(Db))
    return objective, distances.clone().detach().cpu().numpy()



def lossC(distances, labels, l, slack):
    rows_to_pick = 300
    distances = distances - torch.diag(distances.diag())
    label_mask = labels.view(1, -1) == labels.view(-1, 1)
    constraint = torch.zeros((1,))
    chosen_rows = np.arange(0, distances.shape[0])
    np.random.shuffle(chosen_rows)
    chosen_rows = chosen_rows[:min(rows_to_pick, distances.shape[0])].tolist()
    for i in chosen_rows:

        same_label_candidates = torch.masked_select(distances[i, :],  label_mask[i, :])
        different_label_candidates = torch.masked_select(distances[i, :],  1 - label_mask[i, :])
        try:
            pair = different_label_candidates[0] - same_label_candidates[0]
            constraint += pair - 1
        except IndexError:

            pass
    # Transform maximisation into minimisation

    constraint = constraint

    same_distances = torch.masked_select(torch.triu(distances), label_mask)

    loss = torch.sum(same_distances) - torch.abs(l)*constraint
    # print('Constraint', torch.sum(same_distances))

    return loss

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
    KERNEL = None
    BATCH_SIZE = 2000
    SKIP_STEP = 2
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
    test_removal_mask = eval.get_to_remove_mask(cam_ids, query_ind, gallery_ind, ground_truth)


    parameters = []
    # PARAMETERS DEFINITION
    matrix = torch.zeros((features.shape[0], features.shape[0]), requires_grad=True)

    matrix.data = (torch.from_numpy(np.linalg.cholesky(np.linalg.inv(np.cov(features[:, train_ind]))).transpose()).type(Tensor))
    # matrix = torch.eye(features.shape[0], requires_grad=True)
    # matrix.data = torch.tril(matrix)

    print(matrix)

    parameters.append(matrix)
    # For gaussian kernel
    if KERNEL is 'RBF':
        # param2 = torch.rand((features.shape[0],), requires_grad=True)
        param2 = torch.rand((1,), requires_grad=True)
        # param2.data = (param2.data+0.5) * 1000
        param2.data = torch.full((1,), 300)
        parameters.append(param2)
    elif KERNEL is 'poly':
        param2 = torch.full((1,), 0.1, requires_grad=True)
        parameters.append(param2)




    slack = torch.full((1,), 0.5, requires_grad=True)
    lagrangian = torch.full((1,), 1, requires_grad=True)
    # parameters.append(lagrangian)
    # parameters.append(slack)
    optimizer = torch.optim.SGD(parameters, lr=0.0000001)
    l_optim = torch.optim.SGD([lagrangian, slack], lr=0.00007)

    recorder = io.Recorder('loss', 'test_mAp', 'train_mAp')
    for it in range(1000):
        m_loss = 0

        for _ in range(SKIP_STEP):
            if BATCHIFY:
                temp_index = np.arange(0, train_ind.shape[0]).astype(np.int32)

                np.random.shuffle(temp_index)
                temp_index = temp_index[:BATCH_SIZE]
                train_ix = train_ind[temp_index].astype(np.int32)
                training_features = torch.from_numpy(features[:, train_ix]).type(Tensor)

                training_labels = ground_truth[train_ix]
            else:
                train_ix = train_ind

            loss, distances = objective_function([torch.triu(parameters[0])], lagrangian, training_features, slack=slack, labels=training_labels, kernel=KERNEL)
            # print(distances)
            m_loss += loss.clone().detach().cpu().numpy()/SKIP_STEP
            # training_features.cpu()
            torch.cuda.empty_cache()
            # print(distances)

            loss.backward()

        if not it == 0:
            optimizer.step()
            l_optim.step()
            # parameters[0].data = torch.triu(parameters[0]).data
            print('distances', distances)
            print('Optimized')
            optimizer.zero_grad()
        with torch.no_grad():
            if KERNEL is 'RBF':
                test_distances = gaussian_Maha(parameters, query_features, features_compare=gallery_features)
            elif KERNEL is 'poly':
                test_distances = poly_Maha(parameters, query_features, features_compare=gallery_features)
            else:
                test_distances = mahalanobis_metric(parameters, query_features, features_compare=gallery_features)
            removal_mask = eval.get_to_remove_mask(cam_ids, train_ix, train_ix, ground_truth)
            ranked_idx_train, _ = eval.rank(10, distances, train_ix, removal_mask=removal_mask)

            ranked_idx_test, _ = eval.rank(10, test_distances.clone().detach().cpu().numpy(), gallery_ind, removal_mask=test_removal_mask)

            total_score_t, query_scores_t = eval.compute_mAP(10, ground_truth, ranked_idx_train, train_ix)
            total_score, query_scores = eval.compute_mAP(10, ground_truth, ranked_idx_test, query_ind)
        recorder.update('no_kernel_lagrag_lr0-1', loss=m_loss, test_mAp=total_score, train_mAp=total_score_t)

        print(m_loss)

        print('slack', slack.data)
        print('lagrangian', lagrangian.data)
        print(total_score_t, total_score)


