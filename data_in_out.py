import numpy
import json
from scipy.io import loadmat



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



    info = loadmat('PR_data/ cuhk03_new_protocol_config_labeled.mat')
    defaults = {}

