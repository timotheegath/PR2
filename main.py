import numpy as np
import data_in_out as io

features = np.memmap('PR_data/features', mode='r', dtype=np.float64)
features = features.reshape(14096, 2048)


