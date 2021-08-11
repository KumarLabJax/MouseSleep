from scipy import signal
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import datetime
import re

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
		n_sample_wavelet=20*samplerate/frequency
		scale_factor = frequency*n_sample_wavelet/(2*5*samplerate)
		wavelet_data = morlet(n_sample_wavelet, 5, scale_factor)
		output[ind, :] = scipy.signal.fftconvolve(data,wavelet_data[slices],mode='same')
	return output


data = pd.read_csv('3mos_B6_EEG_vs_Video_1BL, B6_W#1_EEG_Data.txt', skiprows=3, sep='\t', header=None)

# Clip out the epoch we want as well as 3 in front and 6 afterwards surrounding ones on each side
#epoch_of_interest = ['10:34:50 AM', '10:35:00 AM']
to_keep = np.where(~(np.array([re.match('((10:34:[2345][0-9] AM)|(10:35:[012345][0-9] AM))', str(x)) for x in data.iloc()[:,0]])==None))[0]
small_data = data.iloc()[to_keep,:]
small_data.iloc()[:,0] = [datetime.datetime.strptime(re.sub('[0-9]*([0-9]{3})$','\\1',re.sub('(AM|PM):([0-9]+)','\\1:000000\\2',str(x))), '%H:%M:%S %p:%f') for x in small_data.iloc()[:,0]]
# Resample to 100Hz (as in paper)
small_data = pd.Series(np.array(small_data[1]),index=small_data[0]).resample('0.01S').first()
samplerate = 100


a, b = signal.butter(5, [0.3/samplerate, 90/samplerate], 'bandpass')
#widths = np.arange(1,samplerate/2+1)
widths = np.arange(1,15,0.1)
cwtmatr = cwt_time(signal.filtfilt(a, b, small_data.iloc()[:]-np.mean(small_data.iloc()[:])), widths, 1, 0, samplerate)

convolve_data = True
if convolve_data:
	signal_mag = signal.convolve2d(cwtmatr.real**2+cwtmatr.imag**2, [np.ones(np.int(2.5*samplerate))/np.int(2.5*samplerate)], mode='same')
else:
	signal_mag = cwtmatr.real**2+cwtmatr.imag**2

t = np.arange(small_data.iloc()[:].shape[0])
T, S = np.meshgrid(t, widths)	
fig = plt.figure()
ax = fig.add_subplot(2,1,1)
ax.contourf(T/samplerate, S, signal_mag, 100, vmax=3.5831567421787716e-06*0.5)
ax = fig.add_subplot(2,1,2)
theta_band = widths[np.intersect1d(np.where(widths<8.5),np.where(widths>3.5))]
upper_delta_band = widths[np.intersect1d(np.where(widths<3.4),np.where(widths>2))]
theta_data = signal_mag[np.isin(widths, theta_band),:]
upper_delta_data = signal_mag[np.isin(widths, upper_delta_band),:]
ax.plot(np.max(theta_data, axis=0)/np.max(upper_delta_data, axis=0))
ax.hlines(y=[1,1.5], xmin=0, xmax=t[-1])
ax.vlines(x=np.arange(0,max(t),1000), ymin=0, ymax=2.5)
fig.show()

# Convolve for 2.5s moving average
plt.contourf(T/samplerate, S, signal.convolve2d(cwtmatr.real**2+cwtmatr.imag**2, [np.ones(np.int(2.5*samplerate))/np.int(2.5*samplerate)], mode='same'), 100)
plt.show()

