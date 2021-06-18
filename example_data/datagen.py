# Example data generator

import gzip
import numpy as np
import pickle

for year in range(2018, 2020+1):
    for i in range(43):
        with gzip.open(f"radar_201806010{i//6}{i%6}0.bin.gz", "wb") as f:
            x = np.zeros(2048*2048, dtype='i2')
            f.write(x.tobytes())

example_sampled_data_inner = [(1020, 1020, 0.0), (1024, 1024, 1.5), (1028, 1028, 10.5)]
example_sampled_data = [(f"{year}06010700", example_sampled_data_inner[:]) for year in range(2018, 2020+1)]
    
with open("sampled.pkl", "wb") as f:
    pickle.dump(example_sampled_data, f)
