import numpy as np
import data_in_out as io


features = np.memmap('PR_data/features', mode='r', shape=(14096, 2048), dtype=np.float64)

