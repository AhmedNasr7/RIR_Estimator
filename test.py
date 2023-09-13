import h5py
import numpy as np

path = "./data/IRs_HL00W.mat"


with h5py.File(path, 'r') as file:
    my_array = np.array(file["data"][:])
    # min_, max_= np.min(my_array), np.max(my_array)

    print(my_array.shape)
