import numpy as np
import data_in_out as io

f = io.load_features()
features = np.memmap('features', mode='w+', shape=f.shape)
features[:] = f[:]
print(features.shape)

