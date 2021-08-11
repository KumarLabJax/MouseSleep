import pandas as pd
import numpy as np
import feather as f
import os, sys, re
import time, datetime

# Read in data
all_data = f.read_dataframe('sampled_6k_per_class_dataset.feather')

# Only look at NREM
sleep_state = all_data.groupby('Sleep Stage').groups['NREM']
all_data = pd.DataFrame(all_data.iloc()[sleep_state])

# Calculate all the dx,dy data by epochs, get average per epoch
by_epoch = all_data.groupby('unique_epoch_id').apply(lambda x: np.mean(np.diff(x['x'])**2+np.diff(x['y'])**2))

# Output the quantiles
np.quantile(by_epoch.values, [0.25,0.5,0.75])
# Output: [0.03093566, 0.06007488, 0.16484085]