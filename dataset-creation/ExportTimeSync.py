import numpy as np
import pandas as pd
import os, re


flist = os.listdir('.')
flist = [x for x in flist if re.match('.*DarkMask_1.csv',x)]
results = []

for cur_file in flist:
	try:
		data = pd.read_csv(cur_file)
		alignment_index = np.max(np.where(data['m00']==0))
	except:
		alignment_index = 0
	results.append(alignment_index)

video_pattern = [x[0:len(x)-len('_DarkMask_1.csv')] for x in flist]
out_df = pd.DataFrame({'Video':video_pattern, 'TimeSyncFrame':results})
out_df.to_csv('TimeSyncFrames.csv', index=False)
