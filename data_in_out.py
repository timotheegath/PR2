import numpy as np
import json
from scipy.io import loadmat
import cv2
import os.path


def load_features():

    with open('PR_data/feature_data.json', 'r') as f:
        features = json.load(f)

    return features


#   Function to easily interact with our image bank. Choose one out of three ways to get image(s): by filename, by index 
#   Or by label (returns all images of same label). The function returns the images, their ground_truth and their cam_id
#   The arguments can be single integers or a list. If opening by filename, specify if the image is part of query or
#   training. If images are not needed, return_im=False (saves time)
def get_image(filename=None, index=None, phase='training', labels=None, return_im=True):
    chosen_way = 0
    all_g_t = loadmat('PR_data/cuhk03_new_protocol_config_labeled.mat', variable_names=['labels'])['labels'].flatten()
    filenames = loadmat('PR_data/cuhk03_new_protocol_config_labeled.mat', variable_names=['filelist'])['filelist'].flatten()
    cam_ids = loadmat('PR_data/cuhk03_new_protocol_config_labeled.mat', variable_names=['camId'])['camId'].flatten()
    # Make all relevant arguments lists
    # And identify which way of indexing has been selected
    if filename is not None:
        if not isinstance(filename, list):
            filename = [filename]
        chosen_way = 1
    elif index is not None:
        if not isinstance(index, list):
            index = [index]
        chosen_way = 2
    elif (labels is not None):
        if not isinstance(labels, list):
            desired_labels = [labels]
        chosen_way = 3

    if chosen_way is 0:
        return

    elif chosen_way is 1:
        images = []
        g_t = []
        cam_id = []
        for f in filename:
            idx = filenames.index(f)
            if return_im:
                images.append(cv2.imread(os.path.join('PR_data/images_cuhk03/', filename)))
            g_t.append(all_g_t[idx])
            cam_id.append(cam_ids[idx])
        return images, g_t, cam_id
    elif chosen_way is 2:
        images = []
        cam_id = []
        g_t = []
        if phase is 'training':
            train_idxs = loadmat('PR_data/cuhk03_new_protocol_config_labeled.mat', variable_names=['train_idx'])['train_idx'].flatten()

            idx = train_idxs[index]
        else:
            query_idxs = loadmat('PR_data/cuhk03_new_protocol_config_labeled.mat', variable_names=['query_idx'])['query_idx'].flatten()
            idx = query_idxs[index]

        for i in idx:

            cam_id.append(cam_ids[i])
            filename = np.array2string(filenames[i])[2:-2]
            if return_im:
                images.append(cv2.imread('PR_data/images_cuhk03/' + filename))
            g_t.append(all_g_t[idx])

        return images, g_t, cam_id
    elif chosen_way is 3:
        images = []
        g_t = []
        cam_id = []
        for l in desired_labels:
            idx = np.argwhere(all_g_t == l).reshape(-1).tolist()
            for i in idx:

                cam_id.append(cam_ids[i])
                filename = np.array2string(filenames[i])[2:-2]
                if return_im:
                    images.append(cv2.imread('PR_data/images_cuhk03/' + filename))
                g_t.append(all_g_t[i])
        return images, g_t, cam_id














