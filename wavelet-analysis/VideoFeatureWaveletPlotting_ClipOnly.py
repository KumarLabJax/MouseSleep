from scipy import signal
import pandas as pd
import numpy as np
import datetime
import re
import feather
import os, sys, argparse
import plotnine as p9
from matplotlib import gridspec
import matplotlib.pyplot as plt

# custom cwt function because signal.cwt doesn't work on morlets...
def cwt_time(data, frequencies, dt, axis, samplerate):
	import scipy
	from scipy.signal import morlet
	# wavelets can be complex so output is complex
	output = np.zeros((len(frequencies),) + data.shape, dtype=np.complex)
	# compute in time
	slices = [None for _ in data.shape]
	slices[axis] = slice(None)
	for ind, frequency in enumerate(frequencies):
		# number of points needed to capture wavelet
		n_sample_wavelet=int(20*samplerate/frequency)
		scale_factor = frequency*n_sample_wavelet/(2*5*samplerate)
		wavelet_data = morlet(n_sample_wavelet, 5, scale_factor)
		output[ind, :] = scipy.signal.fftconvolve(data,wavelet_data[tuple(slices)],mode='same')
	return output


# Plots the cwt data on ax
# Returns the cwt matrix
def plot_cwt(small_data, feature, widths, a, b, samplerate, ax, use_quant):
	# Filter and cwt transform the data
	cwtmatr = cwt_time(signal.filtfilt(a, b, small_data[feature]-np.mean(small_data[feature])), widths, 1, 0, samplerate)
	# Only need magnitude component
	signal_mag = cwtmatr.real**2+cwtmatr.imag**2
	# Plotting
	t = np.arange(small_data[feature].shape[0])
	T, S = np.meshgrid(t, widths)
	vmin = np.min(signal_mag)
	vmax = np.max(signal_mag)
	if use_quant:
		vmin, vmax = np.quantile(signal_mag, [0.001,0.99])
	c_levels = np.linspace(vmin, vmax, 50)
	_ = ax.contourf(T/samplerate, S, signal_mag, 100, levels=c_levels, extend='both')
	_ = ax.semilogy()
	_ = ax.set_ylabel('Frequency, Hz')
	_ = ax.set_title(feature)
	return cwtmatr, t


def plot_video_clip(args):
	input_file = args.input_file
	if os.path.splitext(input_file)[1] == '.feather':
		data = feather.read_dataframe(input_file)
	elif os.path.splitext(input_file)[1] == '.csv':
		data = pd.read_csv(input_file)
	else:
		print('Input file format not recognized: ' + os.path.splitext(input_file)[1])
		exit(1)
	# Run calculations, if not already present
	data['x'] = data['m10']/data['m00']
	data['y'] = data['m01']/data['m00']
	data['a'] = data['m20']/data['m00']-data['x']**2
	data['b'] = 2*(data['m11']/data['m00'] - data['x']*data['y'])
	data['c'] = data['m02']/data['m00'] - data['y']**2
	data['w'] = np.sqrt(8*(data['a']+data['c']-np.sqrt(data['b']**2+(data['a']-data['c'])**2)))/2
	data['l'] = np.sqrt(8*(data['a']+data['c']+np.sqrt(data['b']**2+(data['a']-data['c'])**2)))/2
	data['theta'] = 1/2.*np.arctan(2*data['b']/(data['a']-data['c']))
	data['aspect_w/l'] = data['w']/data['l']
	data['circularity'] = data['m00']*4*np.pi/data['perimeter']**2
	data['rectangular'] = data['m00']/(data['w']*data['l'])
	data['eccentricity'] = np.sqrt(data['w']**2 + data['l']**2)/data['l']
	data['elongation'] = (data['mu20'] + data['mu02'] + (4 * data['mu11']**2 + (data['mu20'] - data['mu02'])**2)**0.5) / (data['mu20'] + data['mu02'] - (4 * data['mu11']**2 + (data['mu20'] - data['mu02'])**2)**0.5)
	data['hu0'] = data['nu20'] + data['nu02']
	data['hu1'] = (data['nu20']-data['nu02'])**2 + 4*data['nu11']**2
	data['hu2'] = (data['nu30']-3*data['nu12'])**2 + (3*data['nu21']-data['nu03'])**2
	data['hu3'] = (data['nu30']+data['nu12'])**2 + (data['nu21']+data['nu03'])**2
	data['hu4'] = (data['nu30']-3*data['nu12'])*(data['nu30']+data['nu12'])*((data['nu30']+data['nu12'])**2-3*(data['nu21']+data['nu03'])**2) + (3*data['nu21']-data['nu03'])*(data['nu21']+data['nu03'])*(3*(data['nu30']+data['nu12'])**2-(data['nu21']+data['nu03'])**2)
	data['hu5'] = (data['nu20']-data['nu02'])*((data['nu03']+data['nu12'])**2) + 4*data['nu11']*(data['nu30']+data['nu12'])*(data['nu21']+data['nu03'])
	data['hu6'] = (3*data['nu21']-data['nu03'])*(data['nu21']+data['nu03'])*(3*(data['nu30']+data['nu12'])**2-(data['nu21']+data['nu03'])**2) + (data['nu30']-3*data['nu12'])*(data['nu21']+data['nu03'])*(3*(data['nu30']+data['nu12'])**2-(data['nu21']+data['nu03'])**2)
	#
	# Assumed capturing information
	samplerate = 30.
	# Bandpass filtering of signal
	if args.filter_breathing:
		a, b = signal.butter(7, [2*2./samplerate, 2*8./samplerate], 'bandpass')
	else:
		# Default to 0.5-14.5hz bandpass
		a, b = signal.butter(5, [1./samplerate, 29./samplerate], 'bandpass')
	widths = np.arange(1,15,0.01)
	#
	feature = args.feature
	if not (feature in data.keys()):
		print('Feature not available: ' + feature)
		print('Available features: ' + data.keys())
		exit(1)
	# Start the chunking of the data...
	# Use "epochs"
	has_found_epochs = False
	if args.use_epochs == True:
		if 'unique_epoch_id' in data.keys():
			has_found_epochs = True
			# Roughly clip chunks by epoch lengths...
			if args.num_samples > 0:
				num_epoch_per_plot = np.ceil(args.num_samples/300)
			else:
				num_epoch_per_plot = 1
			data_groups = data.groupby('unique_epoch_id').groups
			# Grab indices for groupings
			data_chunks = []
			cur_chunk_count = 0
			cur_chunks = np.array([])
			for i in data_groups:
				if cur_chunk_count == num_epoch_per_plot:
					data_chunks.append(np.reshape(cur_chunks, [-1]))
					cur_chunks = np.array([])
					cur_chunk_count = 0
				# Pull out only the indices to match old method...
				#cur_chunks = np.concatenate(cur_chunks, data_groups[i].values)
				cur_chunks = np.concatenate((cur_chunks, np.array(data_groups[i].values)))
				cur_chunk_count = cur_chunk_count + 1
	# Old method (Plot constant durations)
	if not has_found_epochs:
		if args.num_samples < samplerate:
			data_chunks = [np.arange(len(data))]
		else:
			data_chunks = [np.arange(args.num_samples)+i*args.num_samples for i,x in enumerate(np.arange(len(data)//args.num_samples))]
			if (len(data)%args.num_samples) > samplerate:
				data_chunks.append(np.arange(len(data)%args.num_samples)-1+args.num_samples*len(data_chunks))
	for i, chunk in enumerate(data_chunks):
		small_data = data.iloc[chunk,:]
		# Filter and cwt transform the data
		#fig = plt.figure(figsize=(12,9))
		fig = (p9.ggplot()+p9.geom_blank(data=pd.DataFrame())+p9.theme_void()).draw()
		gs = gridspec.GridSpec(6,6)
		ax = fig.add_subplot(gs[0:3,0:5])
		cwtmatr, t = plot_cwt(small_data, feature, widths, a, b, samplerate, ax, args.color_quant)
		# Change title if epoch id was used
		if has_found_epochs:
			if num_epoch_per_plot == 1:
				_ = ax.set_title('Feature ' + feature + ' for ' + small_data['unique_epoch_id'].values[0])
			else:
				_ = ax.set_title('Feature ' + feature + ' for epochs starting at ' + small_data['unique_epoch_id'].values[0])
		# Max amplitude over time subplot
		ax = fig.add_subplot(gs[3:6,0:5])
		signal_mag = cwtmatr.real**2+cwtmatr.imag**2
		breathing_band = widths[np.intersect1d(np.where(widths<8.5),np.where(widths>0.5))]
		breathing_data = signal_mag[np.isin(widths, breathing_band),:]
		# Mute some plotting copy warnings...
		pd.set_option('mode.chained_assignment', None)
		if has_found_epochs and num_epoch_per_plot > 1:
			idxs = np.array((np.arange(num_epoch_per_plot, dtype=np.uint16))*300+150, dtype=np.uint16)
			p_max = (p9.ggplot(pd.DataFrame({'time':t/samplerate, 'values':widths[np.argmax(breathing_data, axis=0)]}), p9.aes(x='time',y='values'))+p9.geom_vline(xintercept=(np.arange(num_epoch_per_plot-1)+1)*10, size=2, color='#ffd3d3')+p9.geom_line()+p9.theme_bw()+p9.scale_x_continuous(expand=(0,0))+p9.geom_text(pd.DataFrame({'x':(np.arange(num_epoch_per_plot)*300)/samplerate+150/samplerate, 'y':np.repeat([np.max(widths[np.argmax(breathing_data, axis=0)])-1], num_epoch_per_plot), 'label':small_data['Sleep Stage'].values[idxs]}), p9.aes(x='x', y='y', label='label'), angle=90))
		else:
			p_max = (p9.ggplot()+p9.geom_line(pd.DataFrame({'time':t/samplerate, 'values':widths[np.argmax(breathing_data, axis=0)]}), p9.aes(x='time',y='values'))+p9.theme_bw()+p9.scale_x_continuous(expand=(0,0))+p9.geom_text(pd.DataFrame({'x':[150/samplerate], 'y':[np.mean(widths[np.argmax(breathing_data, axis=0)])], 'label':small_data['Sleep Stage'].values[150]}), p9.aes(x='x', y='y', label='label')))
		_ = p_max._draw_using_figure(fig, [ax])
		pd.set_option('mode.chained_assignment', 'warn')
		ax.set_ylabel('Frequency, Hz')
		ax.set_xlabel('Time, s')
		# Histogram subplot
		ax = fig.add_subplot(gs[3:6,5])
		pd.set_option('mode.chained_assignment', None)
		p_hist = (p9.ggplot(pd.DataFrame({'time':t/samplerate, 'values':widths[np.argmax(breathing_data, axis=0)]}), p9.aes(x='values'))+p9.geom_histogram(bins=20)+p9.coord_flip()+p9.theme_bw())
		_ = p_hist._draw_using_figure(fig, [ax])
		pd.set_option('mode.chained_assignment', 'warn')
		ax.set_ylabel('Frequency, Hz')
		ax.set_xlabel('Counts')
		#fig.show()
		print('Saving: ' + os.path.splitext(input_file)[0] + '_clip' + str(i) + '_' + feature[0:3] + '...')
		output_pattern = '.png'
		if args.use_svg:
			output_pattern = '.svg'
		if has_found_epochs and num_epoch_per_plot==1:
			fig.savefig(small_data['Sleep Stage'].values[0] + '_' + os.path.splitext(input_file)[0] + '_clip' + str(i) + '_' + feature[0:3] + output_pattern, dpi=300)
		else:
			fig.savefig(os.path.splitext(input_file)[0] + '_clip' + str(i) + '_' + feature[0:3] + output_pattern, dpi=300)
		plt.close(fig)
		#print(np.mean(widths[np.argmax(breathing_data, axis=0)]))
		# Histogram plot
		#fig = plt.figure(figsize=(12,9))
		#ax = fig.add_subplot(1,1,1)
		#_ = ax.hist(widths[np.argmax(breathing_data, axis=0)])
		#fig.savefig(os.path.splitext(input_file)[0] + '_clip' + str(i) + '_' + feature[0:3] + '_HIST.png')
		#plt.close(fig)



def main(argv):
	parser = argparse.ArgumentParser(description='Plots wavelet responses for sleep data')
	parser.add_argument('--input_file', help='Input file', required=True)
	parser.add_argument('--use_epochs', help='Attempt to use epochs as grouping', default=False, action='store_true')
	parser.add_argument('--num_samples', help='Number of samples per figure (default 10s)', default=300, type=int)
	parser.add_argument('--feature', help='Featuer in table (or calculated)', default='hu4')
	parser.add_argument('--filter_breathing', help='Apply a bandpass filter for breathing frequencies', default=False, action='store_true')
	parser.add_argument('--color_quant', help='Scale color by 25-75 quantile instead of min/max', default=False, action='store_true')
	parser.add_argument('--use_svg', help='Save image as svg instead of png', default=False, action='store_true')
	args = parser.parse_args()
	plot_video_clip(args)


if __name__ == '__main__':
	main(sys.argv[1:])

