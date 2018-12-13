import torch
import numpy as np
import data_in_out as io
import metrics as metrics
import evaluation as eval
from scipy.io import savemat

if torch.cuda.is_available():
    Tensor = torch.cuda.FloatTensor
else:
    Tensor = torch.FloatTensor
torch.set_default_tensor_type(Tensor)

if __name__ == '__main__':

    recorder = io.Recorder('rank', 'testm1_mAp', 'testm2_mAp', 'testcc_mAp', 'testcos_mAp', 'testbis_mAp', 'testmah_mAp')

    features = np.memmap('PR_data/features', mode='r', shape=(14096, 2048), dtype=np.float64)
    features = features.transpose()

    ground_truth = io.get_ground_truth()
    cam_ids = io.get_cam_ids()

    training_indices = io.get_training_indexes()
    training_features = features[:, training_indices]
    query_indices = io.get_query_indexes()
    query_features = features[:, query_indices]
    gallery_indices = io.get_gallery_indexes()
    gallery_features = features[:, gallery_indices]

    cossim = metrics.BilinearSimilarity(torch.eye(query_features.shape[0]), cosine=True)
    bisim = metrics.BilinearSimilarity(torch.eye(query_features.shape[0], query_features.shape[0]), cosine=False)

    parameters = torch.rand((training_features.shape[0], training_features.shape[0]), requires_grad=True)
    parameters.data = torch.from_numpy(np.linalg.inv(np.cov(training_features))).type(Tensor)
    parameters.data = torch.potrf(parameters.data)

    # parameters = torch.tril(parameters).view(-1)
    #
    # test_distances = mahalanobis_metric(parameters, query_features, features_compare=gallery_features)

    # rank = 10
    # ranked_inds_test, _ = eval.rank(rank, test_distances.clone().detach().numpy(), gallery_indices,
    #                                 removal_mask=removal_mask)
    # total_score, query_scores = eval.compute_mAP(rank, ground_truth, ranked_inds_test, query_indices)
    # print(query_scores[456], query_scores[122], query_scores[186], total_score)
    # display_inds = np.array([456, 122, 186])
    # io.display_ranklist(query_indices, ranked_inds_test, rank, 3, override_choice=display_inds)

    # test_distances_m1 = metrics.minkowski_metric(query_features, p=1, features_compare=gallery_features)
    # test_distances_m2 = metrics.minkowski_metric(query_features, p=2, features_compare=gallery_features)
    # test_distances_cc = metrics.cross_correlation(query_features, features_compare=gallery_features)
    # test_distances_cos = cossim(query_features, gallery_features)
    # test_distances_bis = bisim(query_features, gallery_features)
    test_distances_mah1 = metrics.mahalanobis_metric({'L': parameters}, query_features, features_compare=gallery_features)
    # distances = {'mink1': test_distances_m1, 'mink2': test_distances_m2, 'cc': test_distances_cc,
    #              'cos': test_distances_cos, 'bi': test_distances_bis,'mah': test_distances_mah}
    parameters.data = torch.eye(features.shape[0])
    test_distances_mah12 = metrics.mahalanobis_metric({'L': parameters}, query_features, gallery_features)
    distances = {'cov_Maha': test_distances_mah1, 'eye_Maha': test_distances_mah12}


    data = dict.fromkeys(distances.keys())
    for k in distances.keys():
        data[k] = {'score': [], 'ranked': []}
    for i in range(1, 15, 1):

        rank = i
        for k in distances.keys():
            if k is 'cc' or k is 'bi' or k is 'cos':
                flip = True
            else:
                flip = False
            ranked, score = eval.evaluate(rank, distances[k], query_indices, gallery_indices, flip=flip)
            data[k]['score'].append(score)
            # data[k]['ranked'].append(ranked)

    savemat('Results/baseline_Maha', data)

    print(data)