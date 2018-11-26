import numpy
import json
from scipy.io import loadmat
import cv2
import os.path


def load_features():

    with open('PR_data/feature_data.json', 'r') as f:
        features = json.load(f)

    return features

def get_image(camId, filename=None, index=None, phase='training', labels=None):
    choice_of_index = 0


    # Make all relevant arguments lists
    if not isinstance(filename, list) & filename is not None:
        filename = list(filename)
        choice_of_index = 1
    elif not isinstance(index, list) & index is not None:
        index = list(index)
        choice_of_index = 2
    elif not isinstance(labels, list) & labels is not None:
        labels = list(labels)
        choice_of_index = 3

    if choice_of_index is 0:
        return
    elif choice_of_index is 1:
        images = []
        for f in filename:
            images.append(cv2.imread(os.path.join('PR_data/')))




    info = loadmat('PR_data/ cuhk03_new_protocol_config_labeled.mat')
    defaults = {}

