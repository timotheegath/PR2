import numpy as np
import data_in_out as io
from scipy.io import loadmat
from sklearn.neighbors import KNeighborsClassifier as KNNC

features = np.memmap('PR_data/features', mode='r', shape=(14096, 2048), dtype=np.float64)
data = loadmat('PR_data/cuhk03_new_protocol_config_labeled.mat')


def process_gallery(gallery_indexes, ):

    gallery_indexes


def KNN_classifier(features, labels, gallery_indices, query_indices):

    features_classify = features[gallery_indices]
    labels_classify = labels[gallery_indices]
    classifier = KNNC(n_neighbors=1)
    classifier.fit(features_classify.transpose(), labels_classify)

    features_query = features[query_indices]
    return classifier.kneighbors(features_query, return_distance=True)

query = io.get_query_indexes()
print(query.shape)