import numpy as np
import sys

params_file = sys.argv[1]
output_file_path = sys.argv[2]

vals = []

with open(params_file, 'r') as file:
        values = file.readlines()

        for value in values:
            vals.append(float(value))

# Converting param list to numpy array and saving to file
vals = np.array(vals)
vals = np.expand_dims(vals, axis=0)
with open(output_file_path, 'wb') as f:
    np.save(f, vals)
print("Data for evaluation generated.")