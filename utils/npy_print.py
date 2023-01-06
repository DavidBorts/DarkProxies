import numpy as np
import sys

npy_path = sys.argv[1]

params = np.load(npy_path)
np.set_printoptions(threshold=sys.maxsize)
print(params)
