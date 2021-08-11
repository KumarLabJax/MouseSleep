import pandas as pd
import numpy as np
import datetime
import re
import feather
import os

# Helper function for saving video clips
def save_video(epoch_data):
	epoch_name = epoch_data['unique_epoch_id'].iloc()[0]
	os.system('ffmpeg -i ' + epoch_name.split(' ')[0] + '.h264 -c:v mpeg4 -q 0 -vf "select=gte(n\,' + str(epoch_data['frame_index'].iloc()[0]) + '),setpts=PTS-STARTPTS" -vframes 300 -n ' + epoch_data['Sleep Stage'].iloc()[0] + '_' + epoch_name.split(' ')[0] + '_' + str(epoch_data['frame_index'].iloc()[0]) + '.avi')

# List of epochs Mandy made plots for...
#Wake: "M_3mos_B6-W#11_PI01 2019-02-06 23:18:50"
#NREM: "M_3mos_B6-W#11_PI01 2019-02-06 16:26:00"
#REM: "M_3mos_B6-W#13_PI03 2019-07-15 17:31:30"
feather_files = ['run/user/1001/gvfs/smb-share:server=bht2stor,share=vkumar/VideosFromOtherLabs/UpennSleep_Raspi/segmentation_output/IndividualAnimalFeathers/M_3mos_B6-W#11_PI01','run/user/1001/gvfs/smb-share:server=bht2stor,share=vkumar/VideosFromOtherLabs/UpennSleep_Raspi/segmentation_output/IndividualAnimalFeathers/M_3mos_B6-W#11_PI01','run/user/1001/gvfs/smb-share:server=bht2stor,share=vkumar/VideosFromOtherLabs/UpennSleep_Raspi/segmentation_output/IndividualAnimalFeathers/M_3mos_B6-W#13_PI03']
epochs = ["M_3mos_B6-W#11_PI01 2019-02-06 23:18:50","M_3mos_B6-W#11_PI01 2019-02-06 16:26:00","M_3mos_B6-W#13_PI03 2019-07-15 17:31:30"]

for feather_file, epoch_id in zip(feather_files, epochs):
	data = feather.read_dataframe(feather_file)
	small_data = data.loc()[data['unique_epoch_id'] == epoch_id, :]
	save_video(small_data)


