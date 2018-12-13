import numpy as np
import torch
from Old import evaluation as eval, data_in_out as io, metrics
from Old.OPtim import mahalanobis_metric

if torch.cuda.is_available():
    Tensor = torch.cuda.FloatTensor
else:
    Tensor = torch.FloatTensor

# def build_histogram(features):
#
#     dimensions = features.shape[0]
#     hist = np.histogramdd(features.transpose(), density=True)
#     print(hist.shape)
#
#     return hist
#
# def build_covariance(features):
#
#     cov = np.corrcoef(features)
#     plt.imshow(cov)
#     plt.waitforbuttonpress()



# def KNN_classifier(features, gallery_indices, query_indices, gallery_mask):
#
#     features_classify = torch.from_numpy(features[:, gallery_indices]).type(Tensor)
#     features_query = torch.from_numpy(features[:, query_indices]).type(Tensor)
#     query_distances = torch.zeros((query_indices.shape[0], gallery_indices.shape[0])).type(Tensor)
#     gallery_mask_t = torch.from_numpy(1 - gallery_mask.astype(np.uint8))
#     if torch.cuda.is_available():
#         gallery_mask_t = gallery_mask_t.cuda()
#
#     print('Calculating nearest neighbours:')
#     for i in range(query_indices.shape[0]):
#         io.loading_bar(i, query_indices.shape[0])
#         # gallery_mask_temp = np.repeat(gallery_mask[i, None], features.shape[0], axis=0)
#         out_d = metrics.minkowski_metric(features_query[:, i],
#                          torch.index_select(features_classify, 1, gallery_mask_t[i].nonzero()[:, 0]).type(Tensor), 2)
#         query_distances[i, gallery_mask_t[i].nonzero()[:, 0]] = out_d
#     query_distances = np.ma.masked_where(gallery_mask, query_distances.cpu().numpy())
#
#     return query_distances


if __name__ == '__main__':

    recorder = io.Recorder('rank', 'testm1_mAp', 'testm2_mAp', 'testcc_mAp', 'testcos_mAp', 'testbis_mAp', 'testmah_mAp')

    features = np.memmap('PR_data/features', mode='r', shape=(14096, 2048), dtype=np.float64)
    features = features.transpose()

    ground_truth = io.get_ground_truth()
    cam_ids = io.get_cam_ids()

    training_indices = io.get_training_indexes()
    training_features = features[:, training_indices]
    training_labels = ground_truth[training_indices]

    query_indices = io.get_query_indexes()
    query_features = features[:, query_indices]
    query_labels = ground_truth[query_indices]

    gallery_indices = io.get_gallery_indexes()

    gallery_features = features[:, gallery_indices]
    gallery_labels = ground_truth[gallery_indices]

    removal_mask = eval.get_to_remove_mask(cam_ids, query_indices, gallery_indices, ground_truth)

    cossim = metrics.BilinearSimilarity(torch.eye(query_features.shape[0]), cosine=True)
    bisim = metrics.BilinearSimilarity(torch.eye(query_features.shape[0], query_features.shape[0]), cosine=False)
    # test_distances = metrics.minkowski_metric(query_features, p=2, features_compare=gallery_features)

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

    test_distances_m1 = metrics.minkowski_metric(query_features, p=1, features_compare=gallery_features)
    test_distances_m2 = metrics.minkowski_metric(query_features, p=2, features_compare=gallery_features)
    test_distances_cc = metrics.cross_correlation(query_features, features_compare=gallery_features)
    test_distances_cos = cossim(query_features, gallery_features)
    test_distances_bis = bisim(query_features, gallery_features)
    test_distances_mah = mahalanobis_metric(parameters, query_features, features_compare=gallery_features)

    for i in range(1, 21, 1):
        rank = i

        ranked_inds_m1, _ = eval.rank(rank, test_distances_m1.clone().detach().numpy(), gallery_indices,
                                      removal_mask=removal_mask)
        total_score_m1, query_scores_m1 = eval.compute_mAP(rank, ground_truth, ranked_inds_m1, query_indices)

        ranked_inds_m2, _ = eval.rank(rank, test_distances_m2.clone().detach().numpy(), gallery_indices,
                                      removal_mask=removal_mask)
        total_score_m2, query_scores_m2 = eval.compute_mAP(rank, ground_truth, ranked_inds_m2, query_indices)

        ranked_inds_cc, _ = eval.rank(rank, test_distances_cc.clone().detach().numpy(), gallery_indices,
                                      removal_mask=removal_mask, flip=True)
        total_score_cc, query_scores_cc = eval.compute_mAP(rank, ground_truth, ranked_inds_cc, query_indices)

        ranked_inds_cos, _ = eval.rank(rank, test_distances_cos.clone().detach().numpy(), gallery_indices,
                                      removal_mask=removal_mask, flip=True)
        total_score_cos, query_scores_cos = eval.compute_mAP(rank, ground_truth, ranked_inds_cos, query_indices)

        ranked_inds_bis, ranked_distances = eval.rank(rank, test_distances_bis.clone().detach().numpy(), gallery_indices,
                                      removal_mask=removal_mask, flip=True)
        total_score_bis, query_scores_bis = eval.compute_mAP(rank, ground_truth, ranked_inds_bis, query_indices)

        ranked_inds_mah, _ = eval.rank(rank, test_distances_mah.clone().detach().numpy(), gallery_indices,
                                      removal_mask=removal_mask)
        total_score_mah, query_scores_mah = eval.compute_mAP(rank, ground_truth, ranked_inds_mah, query_indices)


        print(total_score_m1, total_score_m2, total_score_cc, total_score_cos, total_score_bis, total_score_mah)
        # print(total_score_m2)
        # display_inds = np.array([456, 122, 186])
        # io.display_ranklist(query_indices, ranked_inds_bis, rank, 3, override_choice=display_inds)

        # recorder.update('Baselines', rank=rank, testm1_mAp=total_score_m1, testm2_mAp=total_score_m2,
        #                    testcc_mAp=total_score_cc, testcos_mAp=total_score_cos, testbis_mAp=total_score_bis,
        #                    testmah_mAp=total_score_mah)
        # io.recorder.save('Baselines')
