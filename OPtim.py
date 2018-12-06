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

def objective_function(parameters, features, labels = None, features_compare = None):
    sigma = 2
    if isinstance(parameters, np.ndarray):
        parameters = torch.from_numpy(parameters).type(Tensor)
    if isinstance(features, np.ndarray):
        features = torch.from_numpy(features).type(Tensor)
    if isinstance(labels, np.ndarray):
        labels = torch.from_numpy(labels.astype(dtype=np.int32)).type(Tensor)
    if isinstance(features_compare, np.ndarray):
        features_compare = torch.from_numpy(features_compare).type(Tensor)

    shape = features.shape[0]
    L = parameters.view(shape, shape)
    # print(features.shape, L.shape)

    class_values, class_counts = np.unique(labels, return_counts=True)

    L_features = torch.mm(L, features)
    L_features_norm = (L_features**2).sum(0).view(1, -1)

    if features_compare is not None:
        L_features_compare = torch.mm(L, features_compare)
        L_features_compare_t = torch.transpose(L_features_compare, 1, 0)
        L_features_compare_norm = (L_features_compare**2).sum(0).view(-1, 1)
    else:
        L_features_compare_t = torch.transpose(L_features, 1, 0)
        L_features_compare_norm = L_features_norm.view(-1, 1)

    # L_features_norm_t = L_features_norm.view(-1, 1)
    # L_features_mm = torch.mm(L_features.transpose(1, 0), L_features)
    L_features_mm = torch.mm(L_features_compare_t, L_features)
    # print(L_features.shape, L_features_norm.shape, L_features_norm_t.shape, L_features_mm.shape)
    # print(parameters)

    distances = 2 - 2*(torch.exp(-L_features_norm/(2*sigma**2)) *
                       torch.exp(-L_features_compare_norm/(2*sigma**2)) *
                       torch.exp(2*L_features_mm/(2*sigma**2)))

    if features_compare is not None:
        return distances.transpose(1, 0)

    else:
        distances = distances - torch.diag(distances.diag())
        label_mask = labels.view(1, -1) == labels.view(-1, 1)
        Dw = torch.masked_select(distances, label_mask)
        Db = torch.masked_select(distances, 1 - label_mask)
        objective = torch.sum(Dw) - torch.sum(Db)
    return objective, distances


def optim_call(parameters):

    print(parameters)

# ----------------------------------------------------------------------------------------------------------------------


def optimize_torch(features, full_training_indexes, full_ground_truth, iterations, batch_size=200):
    L = ((torch.rand((2048, 2048), requires_grad=True) - 0.5)*100).retain_grad()
    lagrangian = torch.full((1, ), 200, requires_grad=True)
    training_indexes = np.copy(full_training_indexes)
    np.random.shuffle(training_indexes)
    training_indexes[:batch_size]
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
    labels = io.get_ground_truth()
    ground_truth = io.get_ground_truth()
    cam_ids = io.get_cam_ids()

    train_ind = io.get_training_indexes() - 1

    values, counts = np.unique(training_labels, return_counts=True)

    query_ind = io.get_query_indexes()
    query_features = features[:, query_ind]
    query_labels = labels[query_ind]
    gallery_ind = io.get_gallery_indexes()
    gallery_features = features[:, gallery_ind]
    gallery_labels = labels[gallery_ind]
    removal_mask = eval.get_to_remove_mask(cam_ids, query_ind, gallery_ind, ground_truth)
    # removal_mask = torch.from_numpy(removal_mask.astype(dtype=np.uint8))

    parameters = torch.ones((features.shape[0], features.shape[0]), requires_grad=True)
    optimizer = torch.optim.Adam([parameters], lr=0.1)

    batch_size = 200
    for it in range(500):
        temp_index = np.copy(train_ind)
        np.random.shuffle(temp_index)
        temp_index = temp_index[:batch_size]
        training_features = features[:, temp_index]
        training_labels = labels[temp_index]
        parameters_ = torch.tril(parameters).view(-1)
        loss, distances = objective_function(parameters_, training_features, labels=training_labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        test_distances = objective_function(parameters_, query_features, features_compare=gallery_features )

        ranked_idx_train, _ = eval.rank(10, distances.clone().detach(), temp_index)
        ranked_idx_test, _ = eval.rank(10, test_distances.clone().detach().numpy(), gallery_ind, removal_mask=removal_mask)
        score_by_query_t, total_score_t = eval.compute_score(10, ground_truth, ranked_idx_train, temp_index)
        score_by_query, total_score = eval.compute_score(10, ground_truth, ranked_idx_test, query_ind)
        print(loss)
        print(total_score_t, total_score)
