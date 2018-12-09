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



def objective_function(parameters, features, labels):


    if isinstance(labels, np.ndarray):
        labels = torch.from_numpy(labels.astype(dtype=np.int32)).type(Tensor)

    class_values, class_counts = np.unique(labels, return_counts=True)

    distances = mahalanobis_metric(parameters, features, features_compare=None)
    label_mask = labels.view(1, -1) == labels.view(-1, 1)
    Dw = torch.masked_select(distances, label_mask)
    Db = torch.masked_select(distances, 1 - label_mask)
    objective = torch.sum(Dw) - torch.sum(Db)
    return objective, distances


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


def optimize_torch(features, training_indexes, ground_truth, iterations, batch_size=200):
    L = ((torch.rand((2048, 2048), requires_grad=True) - 0.5)*100).retain_grad()
    lagrangian = torch.full((1, ), 200, requires_grad=True)
    def get_batch():
        np.random.shuffle(training_indexes)
        batch_index = training_indexes[:batch_size]
        batch_features = features[:, batch_index]
        batch_labels = ground_truth[batch_index]

        return batch_index, batch_features, batch_labels
    training_indexes = training_indexes.astype(np.int32)
    with torch.no_grad():
        if isinstance(features, np.ndarray):
            features = torch.from_numpy(features).cpu()
        features = dare.normalise_features(features)
    optimizer = torch.optim.Adadelta(params=[L], lr=0.1)

    for n in range(iterations):

        L_new = torch.tril(L)

        Lagrag_new = torch.clamp(lagrangian, min=10)

        batch_index, batch_features, batch_labels = get_batch()
        batch_features = batch_features.type(Tensor)
        distances = torch.empty((batch_size, batch_size)).type(Tensor)

        for i in range(batch_size):

            distances[i, :] = metrics.new_Maha(batch_features[:, i], batch_features, L_new, torch.full((1,), 1))

            # io.loading_bar(i, batch_size)

        if (distances < 0).any():
            print('Negative distances detected')
        loss = loss1(distances, batch_labels, batch_index, Lagrag_new)
        print(loss, lagrangian)

        loss.backward()
        print(L)
        optimizer.step()

        optimizer.zero_grad()
        ranked_idx, _ = eval.rank(10, distances.clone().detach(), batch_index)
        score_by_query, total_score = eval.compute_score(10, ground_truth, ranked_idx, batch_index)
        # io.display_ranklist(training_indexes[:batch_size],ranked_idx, rank=10, N=3)
        print(total_score)



def loss1(distances, labels, training_index, lagrangian):

    same_person = torch.from_numpy((labels[:, None] == labels[None, :]).astype(np.int32) - np.eye(labels.shape[0])).cuda().byte()

    # Actual objective function to minimize
    same_distance_score = torch.masked_select(distances, same_person)

    same_distance_score = lagrangian * (torch.sum(same_distance_score) - 1)  # Dividing by 2 since the matrix is symmetrical ?

    different_distance = -torch.sum(torch.sqrt(torch.masked_select(distances, 1 - same_person)))



    # Lagrangian trick, Going from maximize to minimize

    # How do you enforce positive definite matrices ?
    loss = same_distance_score + different_distance
    # loss = same_distance_score
    return loss


if __name__ == '__main__':

    features = np.memmap('PR_data/features', mode='r', shape=(14096, 2048), dtype=np.float64)
    features = features.transpose()

    ground_truth = io.get_ground_truth()
    cam_ids = io.get_cam_ids()

    train_ind = io.get_training_indexes()
    training_features = features[:, train_ind]
    training_labels = ground_truth[train_ind]
    # values, counts = np.unique(training_labels, return_counts=True)

    query_ind = io.get_query_indexes()
    query_features = features[:, query_ind]
    query_labels = ground_truth[query_ind]

    gallery_ind = io.get_gallery_indexes()
    gallery_features = features[:, gallery_ind]
    gallery_labels = ground_truth[gallery_ind]

    removal_mask = eval.get_to_remove_mask(cam_ids, query_ind, gallery_ind, ground_truth)
    # removal_mask = torch.from_numpy(removal_mask.astype(dtype=np.uint8))

    parameters = torch.rand((training_features.shape[0], training_features.shape[0]), requires_grad=True)
    parameters.data = torch.from_numpy(np.linalg.inv(np.cov(training_features))).type(Tensor)
    optimizer = torch.optim.Adam([parameters], lr=0.1)

    # constraint_distances(training_features)

    for it in range(500):
        parameters_ = torch.tril(parameters).view(-1)
        loss, distances = objective_function(parameters_, training_features, training_labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        test_distances = mahalanobis_metric(parameters_, query_features, features_compare=gallery_features)

        ranked_idx_train, _ = eval.rank(10, distances.clone().detach(), train_ind)
        ranked_idx_test, _ = eval.rank(10, test_distances.clone().detach().numpy(), gallery_ind, removal_mask=removal_mask)
        total_score_t, query_scores_t = eval.compute_mAP(10, ground_truth, ranked_idx_train, train_ind)
        total_score, query_scores = eval.compute_mAP(10, ground_truth, ranked_idx_test, query_ind)
        print(loss)
        print(total_score_t, total_score)
