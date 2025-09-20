import numpy as np
import h5py

# 假设 data 是你的数组，dtype 可选 float32/float64/...
data = np.random.rand(201, 201, 423, 3).astype(np.float64)

with h5py.File("mydata.h5", "w") as f:
    f.create_dataset("4d array", data=data,
                     compression="gzip", compression_opts=4)