import pandas as pd
import numpy as np
import feather as f
import os, sys, re
import time, datetime
from scipy import signal
from scipy import stats
import os, sys, argparse

def export_breathing_summaries(args):
	animal_data = f.read_dataframe(args.animal_id + '.feather')
	# Add in calculated features
	animal_data['x'] = animal_data['m10']/animal_data['m00']
	animal_data['y'] = animal_data['m01']/animal_data['m00']
	animal_data['a'] = animal_data['m20']/animal_data['m00']-animal_data['x']**2
	animal_data['b'] = 2*(animal_data['m11']/animal_data['m00'] - animal_data['x']*animal_data['y'])
	animal_data['c'] = animal_data['m02']/animal_data['m00'] - animal_data['y']**2
	animal_data['w'] = np.sqrt(8*(animal_data['a']+animal_data['c']-np.sqrt(animal_data['b']**2+(animal_data['a']-animal_data['c'])**2)))/2
	animal_data['l'] = np.sqrt(8*(animal_data['a']+animal_data['c']+np.sqrt(animal_data['b']**2+(animal_data['a']-animal_data['c'])**2)))/2
	animal_data['theta'] = 1/2.*np.arctan(2*animal_data['b']/(animal_data['a']-animal_data['c']))
	animal_data['aspect_w/l'] = animal_data['w']/animal_data['l']
	animal_data['circularity'] = animal_data['m00']*4*np.pi/animal_data['perimeter']**2
	animal_data['rectangular'] = animal_data['m00']/(animal_data['w']*animal_data['l'])
	animal_data['eccentricity'] = np.sqrt(animal_data['w']**2 + animal_data['l']**2)/animal_data['l']
	animal_data['elongation'] = (animal_data['mu20'] + animal_data['mu02'] + (4 * animal_data['mu11']**2 + (animal_data['mu20'] - animal_data['mu02'])**2)**0.5) / (animal_data['mu20'] + animal_data['mu02'] - (4 * animal_data['mu11']**2 + (animal_data['mu20'] - animal_data['mu02'])**2)**0.5)
	animal_data['hu0'] = animal_data['nu20'] + animal_data['nu02']
	animal_data['hu1'] = (animal_data['nu20']-animal_data['nu02'])**2 + 4*animal_data['nu11']**2
	animal_data['hu2'] = (animal_data['nu30']-3*animal_data['nu12'])**2 + (3*animal_data['nu21']-animal_data['nu03'])**2
	animal_data['hu3'] = (animal_data['nu30']+animal_data['nu12'])**2 + (animal_data['nu21']+animal_data['nu03'])**2
	animal_data['hu4'] = (animal_data['nu30']-3*animal_data['nu12'])*(animal_data['nu30']+animal_data['nu12'])*((animal_data['nu30']+animal_data['nu12'])**2-3*(animal_data['nu21']+animal_data['nu03'])**2) + (3*animal_data['nu21']-animal_data['nu03'])*(animal_data['nu21']+animal_data['nu03'])*(3*(animal_data['nu30']+animal_data['nu12'])**2-(animal_data['nu21']+animal_data['nu03'])**2)
	animal_data['hu5'] = (animal_data['nu20']-animal_data['nu02'])*((animal_data['nu03']+animal_data['nu12'])**2) + 4*animal_data['nu11']*(animal_data['nu30']+animal_data['nu12'])*(animal_data['nu21']+animal_data['nu03'])
	animal_data['hu6'] = (3*animal_data['nu21']-animal_data['nu03'])*(animal_data['nu21']+animal_data['nu03'])*(3*(animal_data['nu30']+animal_data['nu12'])**2-(animal_data['nu21']+animal_data['nu03'])**2) + (animal_data['nu30']-3*animal_data['nu12'])*(animal_data['nu21']+animal_data['nu03'])*(3*(animal_data['nu30']+animal_data['nu12'])**2-(animal_data['nu21']+animal_data['nu03'])**2)

	epoch_dist = animal_data.groupby('unique_epoch_id').apply(lambda x: np.mean(np.diff(x['x'])**2+np.diff(x['y'])**2))

	# NREM training 6k dataset thresholds (25%,50%,75%): [0.03093566, 0.06007488, 0.16484085]
	if args.threshold_quant == 25:
		threshold = 0.03093566
	elif args.threshold_quant == 50:
		threshold = 0.06007488
	elif args.threshold_quant == 75:
		threshold = 0.16484085
	else: # Default to "all"
		threshold = -1
	# Pick based on threshold
	if threshold > 0:
		good_epochs = np.where(epoch_dist.values<threshold)[0]
	else:
		good_epochs = np.where(epoch_dist.values>threshold)[0]
	good_epoch_names = epoch_dist.keys()[good_epochs].values

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
			n_sample_wavelet=np.int(20*samplerate/frequency)
			scale_factor = frequency*n_sample_wavelet/(2*5*samplerate)
			wavelet_data = morlet(n_sample_wavelet, 5, scale_factor)
			output[ind, :] = scipy.signal.fftconvolve(data,wavelet_data[slices],mode='same')
		return output

	animal_groups = animal_data.groupby('unique_epoch_id')
	breathing_epoch_means = []
	breathing_epoch_weightedmeans_global = []
	breathing_epoch_weightedmeans_local = []
	breathing_epoch_modes = []
	breathing_epoch_stds = []
	breathing_epoch_quant25 = []
	breathing_epoch_quant50 = []
	breathing_epoch_quant75 = []
	dist_during_epoch = []
	epoch_id = []
	sleep_state = []

	for epoch in good_epoch_names:
		tmp_data = animal_data.iloc()[animal_groups.groups[epoch].values].reset_index(drop=True)
		# CWT code
		samplerate = 30.
		# Filter for breathing range
		a, b = signal.butter(7, [2*1./samplerate, 2*8./samplerate], 'bandpass')
		# Default to 0.5-14.5hz bandpass to remove DC offset
		#a, b = signal.butter(5, [1./samplerate, 29./samplerate], 'bandpass')
		widths = np.arange(1,15,0.01)
		cwtmatr = cwt_time(signal.filtfilt(a, b, tmp_data[args.feature]-np.mean(tmp_data[args.feature])), widths, 1, 0, samplerate)
		signal_mag = cwtmatr.real**2+cwtmatr.imag**2
		breathing_band = widths[np.intersect1d(np.where(widths<8.5),np.where(widths>0.5))]
		breathing_data = signal_mag[np.isin(widths, breathing_band),:]
		predicted_breathing_signal = widths[np.argmax(breathing_data, axis=0)]
		# Add the summaries to results
		breathing_epoch_means.append(np.mean(predicted_breathing_signal))
		amp_data = np.array([breathing_data[j,i] for i,j in enumerate(np.argmax(breathing_data, 0))])
		local_min_amp = np.min(amp_data)
		breathing_epoch_weightedmeans_global.append(np.average(predicted_breathing_signal, weights=amp_data))
		breathing_epoch_weightedmeans_local.append(np.average(predicted_breathing_signal, weights=amp_data-local_min_amp))
		breathing_epoch_modes.append(stats.mode(predicted_breathing_signal)[0][0])
		breathing_epoch_stds.append(np.std(predicted_breathing_signal))
		breathing_epoch_quant25.append(np.quantile(predicted_breathing_signal, [0.25])[0])
		breathing_epoch_quant50.append(np.quantile(predicted_breathing_signal, [0.5])[0])
		breathing_epoch_quant75.append(np.quantile(predicted_breathing_signal, [0.75])[0])
		# Also add in the distance
		dist_during_epoch.append(np.mean(np.diff(tmp_data['x'])**2+np.diff(tmp_data['y'])**2))
		epoch_id.append(tmp_data['unique_epoch_id'].values[0])
		sleep_state.append(tmp_data['Sleep Stage'].values[0])

	# Summary for filtered group distribution
	#output = animal_groups.apply(lambda x: x.reset_index(drop=True).iloc[0,:]['Sleep Stage'])
	#output_2 = [output[i] for i in np.arange(len(output)) if output.keys()[i] in good_epoch_names]
	#np.unique(output_2, return_counts=True)

	new_df = pd.DataFrame({'id':args.animal_id, 'unique_epoch_id':epoch_id, 'sleep_stage':sleep_state, 'means':breathing_epoch_means, 'means_globalweight':breathing_epoch_weightedmeans_global, 'means_localweight':breathing_epoch_weightedmeans_local, 'modes':breathing_epoch_modes, 'stds':breathing_epoch_stds, 'quant25':breathing_epoch_quant25, 'quant50':breathing_epoch_quant50, 'quant75':breathing_epoch_quant75, 'dist':dist_during_epoch})

	new_df.to_csv(args.animal_id + '_' + args.feature[:3] + '.csv', index=False)


def main(argv):
	parser = argparse.ArgumentParser(description='Plots wavelet responses for sleep data')
	parser.add_argument('--animal_id', help='Input animal, corresponding to a *.feather file', required=True)
	parser.add_argument('--feature', help='Featuer in table (or calculated)', default='aspect_w/l')
	parser.add_argument('--threshold_quant', help='Quantile used for distance (-1 for all epochs)', default=25, type=int, choices=[25,50,75,-1])
	args = parser.parse_args()
	export_breathing_summaries(args)


if __name__ == '__main__':
	main(sys.argv[1:])

# Running all the feather files in a folder using parallel
# ls . | grep feather | grep 'W#' | sed -e 's/\.feather//g' | parallel -j2 'python AnalyzeAnimalBreathing.py --animal_id {}' --threshold_quant -1

# Merging the outputs into 1 file
# head -n 1 M_3mos_B6-W#1_PI03_asp.csv > merged_output.csv
# ls | grep csv | grep -v merged | grep -v TimeSync | parallel -j1 "awk 'NR>1' {} >> merged_output.csv"

# Basic plotting
# from plotnine import *
# import pandas as pd
# import numpy as np
# import re
# data = pd.read_csv('merged_output.csv')
# data['strain'] = [re.sub('.*(B6|C3).*','\\1',x) for x in data['id'].values]
# data['sex'] = [re.sub('^(M|F).*','\\1',x) for x in data['id'].values]
# ggplot(data, aes(x='id',y='means',color='sex'))+geom_jitter(width=0.15,alpha=0.05)+stat_summary(color='black',fun_y=np.mean, fun_ymin=lambda x: np.mean(x)-np.std(x), fun_ymax=lambda x: np.mean(x)+np.std(x), geom='crossbar')+theme_538()+theme(axis_text_x=element_text(angle=90))+facet_grid('.~strain', scales='free_x')

# Basic plotting for "filtered after the fact" data
# from plotnine import *
# import pandas as pd
# import numpy as np
# import re
# data = pd.read_csv('merged_output.csv')
# data['strain'] = [re.sub('.*(B6|C3).*','\\1',x) for x in data['id'].values]
# data['sex'] = [re.sub('^(M|F).*','\\1',x) for x in data['id'].values]
# data = pd.DataFrame(data.groupby('id').apply(lambda x: x.iloc()[np.where(x['dist'].values<np.quantile(x['dist'],[0.1])[0])].reset_index(drop=True)).reset_index(drop=True))
# ggplot(data, aes(x='id',y='means',color='sex'))+geom_violin()+stat_summary(color='black',fun_y=np.mean, fun_ymin=lambda x: np.mean(x)-np.std(x), fun_ymax=lambda x: np.mean(x)+np.std(x), geom='crossbar')+geom_jitter(width=0.15,alpha=0.05)+theme_538()+theme(axis_text_x=element_text(angle=90))+facet_grid('.~strain', scales='free_x')
