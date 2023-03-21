import sys
import numpy as np
import Constants as c
from npy_convert import convert


proxy_type, param = sys.argv[1].split('_')
num = int(sys.argv[2])
imgNum = int(sys.argv[3])
output_path = sys.argv[4]

possible_values = getattr(c.POSSIBLE_VALUES(), proxy_type + '_' + param)
min = possible_values[0][0]
max = possible_values[0][1]

# List of all slider values
vals = []

for img in range(imgNum):
    for value in np.linspace(min, max, int(num)):
        vals.append(float(value)) # Adding to params list

print(vals)
convert(vals, output_path)