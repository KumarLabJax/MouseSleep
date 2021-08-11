import pandas as pd
import numpy as np
import feather as f
import os, sys, re
import time, datetime
import os, sys, argparse
import cv2

id_pattern_string = '.*(BL|Sal|Met).*(B6|C3)-(WO?#?[0-9]+).*'
id_pattern_sub = '\\2-\\3-\\1'

def get_time_for_cover(filename, default='10:00:00'):
	try:
		with open(filename, 'r') as fp:
			ret_val = [line for line in fp if 'cage cover' in line]
		ret_val = re.sub('.*([0-9]{2}:[0-9]{2}:[0-9]{2}).*','\\1',ret_val[0])[:-1]
		return ret_val
	except:
		return default

def export_animal(args):
	# Testing pattern for any downstream matching
	test_animal_id = re.sub(id_pattern_string,id_pattern_sub,args.animal_id)
	# List files and folder for results
	if args.data_folder is not None:
		if os.path.isdir(args.data_folder):
			data_folder = args.data_folder
			animal_pattern = os.path.basename(args.animal_id)
		else:
			data_folder = os.path.dirname(args.data_folder)
			animal_pattern = os.path.basename(args.data_folder)
		data_files = os.listdir(data_folder)
		data_files = [x for x in data_files if re.search(animal_pattern, x)]
		file_df = pd.DataFrame({'chunk_idx': [int(re.sub('.*_([0-9]+)\.csv','\\1',x)) for x in data_files], 'filename': [data_folder + '/' + x for x in data_files]}).sort_values(by=['chunk_idx']).reset_index(drop=True)
	else:
		file_df = pd.DataFrame({'chunk_idx': [1], 'filename': [args.data_file]})
	# Do the same for eeg calls
	annotations_missing = False
	if args.eeg_folder is not None:
		if os.path.isdir(args.eeg_folder):
			eeg_folder = args.eeg_folder
		else:
			eeg_folder = os.path.dirname(args.eeg_folder)
		eeg_files = os.listdir(eeg_folder)
		eeg_ids = [re.sub(id_pattern_string,id_pattern_sub,x) for x in eeg_files]
		match_ids = np.where(eeg_ids == np.array(test_animal_id))[0]
		if len(match_ids)==1:
			eeg_file = eeg_folder + eeg_files[match_ids[0]]
		else:
			annotations_missing = True
	elif args.eeg_file is not None:
		eeg_file = arg.eeg_file
	else:
		annotations_missing = True
	# Time sync file (alignment to eeg/emg annotations)
	if args.timesyncfile is not None:
		sync_frames = pd.read_csv(args.timesyncfile)
		sync_tests = [re.sub(r'.*(BL|Sal|Met).*(B6|C3)-(WO?#[0-9]+).*',id_pattern_sub,x) for x in sync_frames['Video']]
		match_ids = np.where(sync_tests == np.array(test_animal_id))[0]
		if len(match_ids)==1:
			sync_skip = sync_frames['TimeSyncFrame'][match_ids[0]]
		else:
			sync_skip = 0
	# Timestamp file
	if args.timestamp_folder is not None:
		if os.path.isdir(args.timestamp_folder):
			timestamp_folder = args.timestamp_folder
		else:
			timestamp_folder = os.path.dirname(args.timestamp_folder)
		timestamp_files = os.listdir(timestamp_folder)
		# Only look at _timestamps.txt
		timestamp_files = [x for x in timestamp_files if re.search('_timestamps.txt', x)]
		timestamp_ids = [re.sub(id_pattern_string,id_pattern_sub,x) for x in timestamp_files]
		match_ids = np.where(timestamp_ids == np.array(test_animal_id))[0]
		if len(match_ids)==1:
			timestamp_file = timestamp_folder + timestamp_files[match_ids[0]]
		else:
			print('Timestamp file not linked correctly, found ' + str(len(match_ids)) + ' matching timestamp files. Use --timestamp_file to specify.')
			exit(0)
	else:
		timestamp_file = args.timestamp_file

	# Begin reading in the data
	# Time information
	timestamps = pd.read_csv(timestamp_file, header=None, names=['times'])
	if args.fragmented_clip_length > 0:
		# Assumes keyframe rate are RPi's default of 60 frames/keyframe
		to_remove = np.round(args.fragmented_clip_length*np.arange(len(timestamps)/args.fragmented_clip_length)/60)*60
		timestamps = timestamps.drop(to_remove).reset_index(drop=True)
	# Import Annotation Data
	# Insert defaults if annotations were missing
	if annotations_missing:
		try:
			# Attempt to parse the "starting timestamps" file for the notes
			eeg_record_date = datetime.datetime.strptime(pd.read_table(re.sub('timestamps.txt','StartingTimestamp.txt',timestamp_file), skiprows=3, nrows=1, header=None, sep=': ')[1][0], '%a %Y-%m-%d %H:%M:%S %Z')
			# Align to 10am
			start_time = datetime.datetime.strptime(datetime.datetime.strftime(eeg_record_date, '%Y-%m-%d ') + '10:00:00', '%Y-%m-%d %H:%M:%S')
		except:
			# Insert dummy data of Jan 1, 1970 10AM
			eeg_record_date = '1/1/1970'
			start_time = datetime.datetime(1970, 1, 1, 10, 0)
		# Populate the data as best as possible
		eeg_data = pd.DataFrame({'time_bin':[start_time + datetime.timedelta(seconds=int(i*10)) for i in np.arange(np.floor(timestamps['times'].values[-1]-timestamps['times'].values[0]))], 'Sleep Stage':'NA'})
	else:
		# StartingTimestamps is more reliable than eeg file...
		try:
			eeg_record_date = datetime.datetime.strptime(pd.read_table(re.sub('timestamps.txt','StartingTimestamp.txt',timestamp_file), skiprows=3, nrows=1, header=None, sep=': ')[1][0], '%a %Y-%m-%d %H:%M:%S %Z')
			# Align to note in starting timestamp file (default to 10AM)
			start_string = get_time_for_cover(re.sub('timestamps.txt','StartingTimestamp.txt',timestamp_file))
			start_time = datetime.datetime.strptime(datetime.datetime.strftime(eeg_record_date, '%Y-%m-%d ') + start_string, '%Y-%m-%d %H:%M:%S')
			found_start = True
		# Fall back to eeg records
		except:
			found_start = False
		if not found_start:
			eeg_record_date = pd.read_csv(eeg_file, header=None, sep='\t', skiprows=3, nrows=1)[1][0]
		# Label data
		eeg_data = pd.read_csv(eeg_file, skiprows=14, sep='\t')
		if not ('Sleep Stage' in eeg_data.keys()): # The number of skiprows was wrong... because upenn changed the format.
			eeg_data = pd.read_csv(eeg_file, skiprows=10, sep='\t')
			eeg_data['Sleep Stage'] = eeg_data['Event']
		if not found_start:
			eeg_start_time = datetime.datetime.strptime(eeg_record_date + ' ' + eeg_data['Time [hh:mm:ss]'][0], '%m/%d/%Y %H:%M:%S %p')
			start_time = eeg_start_time
		else:
			eeg_start_time = datetime.datetime.strptime(datetime.datetime.strftime(eeg_record_date, '%m/%d/%Y') + ' ' + eeg_data['Time [hh:mm:ss]'][0], '%m/%d/%Y %H:%M:%S %p')
		eeg_data['time_bin'] = [eeg_start_time + datetime.timedelta(seconds=int(i*10)) for i in np.arange(eeg_data.shape[0])]
	# Raw data
	raw_data = pd.DataFrame().append([pd.read_csv(y) for y in file_df['filename']])
	raw_data['frame_index'] = np.arange(raw_data.shape[0])
	# Clip the two to be the same shape
	min_size = np.min([raw_data.shape[0], timestamps.shape[0]])
	raw_data = raw_data.iloc[0:min_size,:]
	timestamps = timestamps.iloc[0:min_size,:]
	raw_data['frame_time'] = [start_time + datetime.timedelta(seconds=(cur_time-timestamps['times'][sync_skip])) for cur_time in timestamps['times']]
	raw_data['time_bin'] = [cur_time - datetime.timedelta(seconds=cur_time.second%10, microseconds=cur_time.microsecond) for cur_time in raw_data['frame_time']]
	# Merge eeg calls into raw_data
	raw_data = raw_data.merge(eeg_data[['time_bin','Sleep Stage']], on='time_bin')
	# Extra calculations
	raw_data['x'] = raw_data['m10']/raw_data['m00']
	raw_data['y'] = raw_data['m01']/raw_data['m00']
	raw_data['a'] = raw_data['m20']/raw_data['m00']-raw_data['x']**2
	raw_data['b'] = 2*(raw_data['m11']/raw_data['m00'] - raw_data['x']*raw_data['y'])
	raw_data['c'] = raw_data['m02']/raw_data['m00'] - raw_data['y']**2
	raw_data['w'] = np.sqrt(8*(raw_data['a']+raw_data['c']-np.sqrt(raw_data['b']**2+(raw_data['a']-raw_data['c'])**2)))/2
	raw_data['l'] = np.sqrt(8*(raw_data['a']+raw_data['c']+np.sqrt(raw_data['b']**2+(raw_data['a']-raw_data['c'])**2)))/2
	raw_data['theta'] = 1/2.*np.arctan(2*raw_data['b']/(raw_data['a']-raw_data['c']))
	# Useful new columns
	raw_data['video'] = args.animal_id
	raw_data['unique_epoch_id'] = [test_animal_id + ' ' + str(tb) for tb in raw_data['time_bin']]
	cv2_hu_moments = np.concatenate([cv2.HuMoments(row) for i,row in raw_data.iterrows()], axis=1).T
	raw_data['hu0'] = cv2_hu_moments[:,0]
	raw_data['hu1'] = cv2_hu_moments[:,1]
	raw_data['hu2'] = cv2_hu_moments[:,2]
	raw_data['hu3'] = cv2_hu_moments[:,3]
	raw_data['hu4'] = cv2_hu_moments[:,4]
	raw_data['hu5'] = cv2_hu_moments[:,5]
	raw_data['hu6'] = cv2_hu_moments[:,6]
	# Export the data
	f.write_dataframe(raw_data, args.animal_id + '.feather')

def main(argv):
	parser = argparse.ArgumentParser(description='Plots wavelet responses for sleep data')
	parser.add_argument('--animal_id', help='Input animal, corresponding to a *.feather file', required=True)
	group = parser.add_mutually_exclusive_group(required=True)
	group.add_argument('--data_folder', help='Folder with animal id data suffixed with numbers')
	group.add_argument('--data_file', help='Single segmentation file')
	group2 = parser.add_mutually_exclusive_group(required=True)
	group2.add_argument('--timestamp_folder', help='Folder containing the timestamp files')
	group2.add_argument('--timestamp_file', help='Single file containing the timestamps')
	group3 = parser.add_mutually_exclusive_group()
	group3.add_argument('--eeg_folder', help='Folder containing the eeg calls, attempts to match based on animal_id')
	group3.add_argument('--eeg_file', help='Single file containing the eeg calls')
	parser.add_argument('--timesyncfile', help='File containing synchronization frame for 10AM')
	parser.add_argument('--fragmented_clip_length', help='Videos were fragmented with ffmpeg for inference, splitting at keyframes. This number adjusts the timestamps.', default=-1, type=int)
	args = parser.parse_args()
	export_animal(args)


if __name__ == '__main__':
	main(sys.argv[1:])
