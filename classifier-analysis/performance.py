
import numpy as np
import pandas as pd
from sklearn import metrics
import plotnine as p9
import matplotlib.pyplot as plt
from matplotlib import gridspec
import os, sys, argparse
from datetime import timedelta
import re
import itertools
from mizani.breaks import date_breaks
from mizani.formatters import date_format
import time
import warnings
warnings.filterwarnings("ignore")
from statsmodels.stats.multitest import multipletests
from scipy.stats import ttest_ind, sem

from types import SimpleNamespace
args = SimpleNamespace(input_folder='/home/bgeuther/Documents/video-sleep-analysis/Amphetamine/predictions_rng_1438939568')

os.environ['TZ'] = 'Etc/UTC'
os.environ['TZ'] = 'America/New_York'
time.tzset()

def plot_hourly(df, column, sigs=[]):
	plot = p9.ggplot(df)
	plot = plot + p9.geom_rect(p9.aes(xmin=11.5,ymin=0,xmax=23.5,ymax=1), fill='0.9')
	plot = plot + p9.stat_summary(p9.aes(x='time_zt', y=column, fill='is_baseline'), fun_ymin=lambda x: np.mean(x)-sem(x), fun_ymax=lambda x: np.mean(x)+sem(x), geom='ribbon', alpha=0.25)
	plot = plot + p9.stat_summary(p9.aes(x='time_zt', y=column, color='is_baseline'), fun_y=lambda x: np.mean(x), geom='point')
	plot = plot + p9.stat_summary(p9.aes(x='time_zt', y=column, color='is_baseline'), fun_y=lambda x: np.mean(x), geom='line')
	plot = plot + p9.scale_color_discrete(labels=['Methamphetamine','Baseline'])
	plot = plot + p9.scale_fill_discrete(labels=['Methamphetamine','Baseline'])
	plot = plot + p9.scale_y_continuous(labels=['0%','25%','50%','75%','100%'], breaks=[0,0.25,0.5,0.75,1.0])
	plot = plot + p9.scale_x_continuous(breaks=[0,3,6,9,12,15,18,21,24], minor_breaks=[1,2,4,5,7,8,10,11,13,14,16,17,19,20,22,23], limits=[0,24], labels=['0','3','6','9','12','15','18','21','24/0'])
	plot = plot + p9.theme_bw()
	plot = plot + p9.labs(color='Mean ± SEM', fill='')
	if len(sigs)>0:
		sig_df = pd.DataFrame({'x':sigs, 'label':'*', 'y':1})
		plot = plot + p9.geom_text(p9.aes(x='x', y='y', label='label'), data=sig_df, color='black')
	inj_arrows = pd.DataFrame({'x':[2,2,6,6], 'y':[-0.1,-0.025,-0.1,-0.025], 'group':[0,0,1,1]})
	plot = plot + p9.geom_path(p9.aes(x='x', y='y', group='group'), data=inj_arrows, arrow=p9.geoms.arrow(length=0.03))
	return plot

def plot_gtvpd_hourly(df, column_prefix='Wake_'):
	plot = p9.ggplot(df)
	plot = plot + p9.geom_rect(p9.aes(xmin=11.5,ymin=0,xmax=23.5,ymax=1), fill='0.9')
	plot = plot + p9.stat_summary(p9.aes(x='time_zt', y=column_prefix+'GT', fill='str(0)'), fun_ymin=lambda x: np.mean(x)-sem(x), fun_ymax=lambda x: np.mean(x)+sem(x), geom='ribbon', alpha=0.25)
	plot = plot + p9.stat_summary(p9.aes(x='time_zt', y=column_prefix+'Pred', fill='str(1)'), fun_ymin=lambda x: np.mean(x)-sem(x), fun_ymax=lambda x: np.mean(x)+sem(x), geom='ribbon', alpha=0.25)
	plot = plot + p9.stat_summary(p9.aes(x='time_zt', y=column_prefix+'GT', color='str(0)'), fun_y=lambda x: np.mean(x), geom='line')
	plot = plot + p9.stat_summary(p9.aes(x='time_zt', y=column_prefix+'Pred', color='str(1)'), fun_y=lambda x: np.mean(x), geom='line')
	plot = plot + p9.stat_summary(p9.aes(x='time_zt', y=column_prefix+'GT', color='str(0)'), fun_y=lambda x: np.mean(x), geom='point')
	plot = plot + p9.stat_summary(p9.aes(x='time_zt', y=column_prefix+'Pred', color='str(1)'), fun_y=lambda x: np.mean(x), geom='point')
	plot = plot + p9.scale_color_manual(labels=['EEG/EMG','Visual Prediction'], values=['#984ea3', '#ff7f00'])
	plot = plot + p9.scale_fill_manual(labels=['EEG/EMG','Visual Prediction'], values=['#984ea3', '#ff7f00'])
	plot = plot + p9.theme_bw()
	plot = plot + p9.labs(color='Mean ± SEM',fill='')
	plot = plot + p9.scale_y_continuous(labels=['0%','25%','50%','75%','100%'], breaks=[0,0.25,0.5,0.75,1.0])
	plot = plot + p9.scale_x_continuous(breaks=[0,3,6,9,12,15,18,21,24], minor_breaks=[1,2,4,5,7,8,10,11,13,14,16,17,19,20,22,23], limits=[0,24], labels=['0','3','6','9','12','15','18','21','24/0'])
	inj_arrows = pd.DataFrame({'x':[2,2,6,6], 'y':[-0.1,-0.025,-0.1,-0.025], 'group':[0,0,1,1]})
	plot = plot + p9.geom_path(p9.aes(x='x', y='y', group='group'), data=inj_arrows, arrow=p9.geoms.arrow(length=0.03))
	return plot

def print_differences(df, stage='Wake'):
	pval = ttest_ind(df[df.is_baseline][stage + '_GT'],df[~df.is_baseline][stage + '_GT']).pvalue
	print(stage + ': EEG/EMG Baseline to Methamphetamine, pval' + str(pval))
	pval = ttest_ind(df[df.is_baseline][stage + '_Pred'],df[~df.is_baseline][stage + '_Pred']).pvalue
	print(stage + ': Visual Prediction Baseline to Methamphetamine, pval=' + str(pval))
	pval = ttest_ind(df[df.is_baseline][stage + '_GT'],df[df.is_baseline][stage + '_Pred']).pvalue
	print(stage + ': EEG/EMG Baseline to Visual Prediction Baseline, pval=' + str(pval))
	pval = ttest_ind(df[~df.is_baseline][stage + '_GT'],df[~df.is_baseline][stage + '_Pred']).pvalue
	print(stage + ': EEG/EMG Methamphetamine to Visual Prediction Methamphetamine, pval=' + str(pval))

def plot_ethogram(df):
	plot = p9.ggplot(data=df)
	plot = plot + p9.geom_point(p9.aes(x='time_zt', y='label', color='str(label)'), shape='|')
	plot = plot + p9.geom_point(p9.aes(x='time_zt', y='prediction+0.25', color='str(prediction)'), shape='|')
	plot = plot + p9.theme_bw()
	plot = plot + p9.scale_color_discrete(labels=['gt','prediction'])
	plot = plot + p9.scale_y_continuous(breaks=[0,0.25,1,1.25,2,2.25], labels=['Wake EEG/EMG', 'Wake Visual Prediction', 'NREM EEG/EMG', 'NREM Visual Prediction', 'REM EEG/EMG', 'REM Visual Prediction'])
	plot = plot + p9.scale_x_datetime(date_breaks='1 hours', date_labels='%m/%d/%Y %H')
	plot = plot + p9.labs(color='')
	plot = plot + p9.theme(axis_text_x=p9.element_text(rotation=90, ha='center'))
	return plot

def read_data(file):
	try:
		data = pd.read_csv(file)
	except:
		print('File ' + file + ' not found, exiting')
		exit(1)
	data['file'] = [x.split(' ')[0] for x in data.unique_epoch_id]
	data['file'] = [re.sub('.*(BL|Meth).*#([0-9]+).*','\\1-\\2',x) for x in data['file']]
	data['datetime'] = pd.to_datetime([' '.join(x.split(' ')[1:]) for x in data.unique_epoch_id], utc=False)
	data['time_zt'] = data['datetime']-timedelta(hours=10)
	data['light'] = np.logical_and(data['datetime'].dt.hour>=10, data['datetime'].dt.hour<20)
	return data

def plot_single_file(args):
	data = read_data(args.input_file)
	if 'prediction' in data.keys():
		print('Results for: ' + os.path.basename(args.input_file))
		print('Accuracy: ' + str(metrics.accuracy_score(data.label, data.prediction)))
		print('Precision: ' + str(metrics.precision_score(data.label, data.prediction, average=None)))
		print('Recall: ' + str(metrics.recall_score(data.label, data.prediction, average=None)))
		(plot_ethogram(data)).draw()
	else:
		fig = (p9.ggplot()+p9.geom_blank(data=data)+p9.theme_void()).draw()
		gs = gridspec.GridSpec(1,2)
		ax1 = fig.add_subplot(gs[0,0])
		ax2 = fig.add_subplot(gs[0,1])
		p1 = p9.ggplot(data=data)+p9.geom_histogram(p9.aes(x='Stage'))+p9.theme_bw()
		p2 = p9.ggplot(data=data)+p9.geom_histogram(p9.aes(x='m00__Ave_Signal'))+p9.theme_bw()
		_ = p1._draw_using_figure(fig, [ax1])
		_ = p2._draw_using_figure(fig, [ax2])
	plt.show(block=True)

def plot_multi_file(args):
	if os.path.isdir(args.input_folder):
		folder = args.input_folder
	else:
		folder = os.path.dirname(args.input_folder)
	files = os.listdir(folder)
	data_list = []
	for ifile in files:
		data_list.append(read_data(folder + '/' + ifile))
	data_list = pd.concat(data_list)
	(p9.ggplot(data_list)+p9.stat_summary(p9.aes(x='label-0.125', y='label', fill='str(0)'), fun_y=lambda x: len(x)*10/60, geom='bar', width=0.25)+p9.stat_summary(p9.aes(x='prediction+0.125', y='prediction', fill='str(1)'), fun_y=lambda x: len(x)*10/60, geom='bar', width=0.25)+p9.facet_wrap('file')+p9.theme_bw()+p9.labs(fill='', x='', y='duration, m')+p9.scale_fill_discrete(labels=['gt','prediction'])+p9.scale_x_continuous(breaks=[0,1,2], labels=['Wake','NREM','REM'])).draw()
	print('Bulk results for: ' + folder)
	print('Accuracy: ' + str(metrics.accuracy_score(data_list.label, data_list.prediction)))
	print('Precision: ' + str(metrics.precision_score(data_list.label, data_list.prediction, average=None)))
	print('Recall: ' + str(metrics.recall_score(data_list.label, data_list.prediction, average=None)))
	print('F1 Score: ' + str(metrics.f1_score(data_list.label, data_list.prediction, average=None)))
	# bout analysis
	gt_bouts = data_list.groupby('file').apply(lambda x: [[k,len(list(l))] for k,l in itertools.groupby(x['label'])])
	pred_bouts = data_list.groupby('file').apply(lambda x: [[k,len(list(l))] for k,l in itertools.groupby(x['prediction'])])
	bout_df = pd.DataFrame({'video':[], 'is_pred':[], 'state':[], 'bout_count':[], 'bout_duration':[], 'longest_bout':[]})
	for vid_id in list(pred_bouts.keys()):
		# GT
		tmp_df = pd.DataFrame(data=np.array(gt_bouts[vid_id]), columns=['state','length'])
		states = pd.DataFrame(tmp_df.groupby('state').apply(lambda x: sum(x['length']))).reset_index()['state'].values
		durations = pd.DataFrame(tmp_df.groupby('state').apply(lambda x: sum(x['length']))).reset_index()[0].values
		longest = pd.DataFrame(tmp_df.groupby('state').apply(lambda x: max(x['length']))).reset_index()[0].values
		counts = pd.DataFrame(tmp_df.groupby('state').apply(lambda x: len(x['length']))).reset_index()[0].values
		bout_df = bout_df.append(pd.DataFrame({'video':vid_id,'is_pred':False,'state':states,'bout_count':counts,'bout_duration':durations,'longest_bout':longest}))
		# Pred
		tmp_df = pd.DataFrame(data=np.array(pred_bouts[vid_id]), columns=['state','length'])
		states = pd.DataFrame(tmp_df.groupby('state').apply(lambda x: sum(x['length']))).reset_index()['state'].values
		durations = pd.DataFrame(tmp_df.groupby('state').apply(lambda x: sum(x['length']))).reset_index()[0].values
		longest = pd.DataFrame(tmp_df.groupby('state').apply(lambda x: max(x['length']))).reset_index()[0].values
		counts = pd.DataFrame(tmp_df.groupby('state').apply(lambda x: len(x['length']))).reset_index()[0].values
		bout_df = bout_df.append(pd.DataFrame({'video':vid_id,'is_pred':True,'state':states,'bout_count':counts,'bout_duration':durations,'longest_bout':longest}))
	state_keys = {0:'Wake',1:'NREM',2:'REM'}
	bout_df['state'] = [state_keys[x] for x in bout_df['state']]
	fig = (p9.ggplot()+p9.geom_blank(data=bout_df)+p9.theme_void()).draw()
	gs = gridspec.GridSpec(3,3)
	for i in np.arange(3):
		lb_plot = p9.ggplot(bout_df[bout_df['state']==state_keys[i]])+p9.geom_bar(p9.aes(x='video', y='longest_bout*10/60/60', fill='factor(is_pred)'), stat='identity', position=p9.position_dodge(width=1))+p9.theme_bw()+p9.labs(fill='', x='animal',y='longest bout, m')+p9.theme(axis_text_x=p9.element_text(rotation=90, ha='center'))
		ad_plot = p9.ggplot(bout_df[bout_df['state']==state_keys[i]])+p9.geom_bar(p9.aes(x='video', y='bout_duration/bout_count*10/60/60', fill='factor(is_pred)'), stat='identity', position=p9.position_dodge(width=1))+p9.theme_bw()+p9.labs(fill='', x='animal',y='average bout duration, m')+p9.theme(axis_text_x=p9.element_text(rotation=90, ha='center'))
		nb_plot = p9.ggplot(bout_df[bout_df['state']==state_keys[i]])+p9.geom_bar(p9.aes(x='video', y='bout_count', fill='factor(is_pred)'), stat='identity', position=p9.position_dodge(width=1))+p9.theme_bw()+p9.labs(fill='', x='animal',y='number bouts')+p9.theme(axis_text_x=p9.element_text(rotation=90, ha='center'))
		ax1 = fig.add_subplot(gs[0,i])
		ax2 = fig.add_subplot(gs[1,i])
		ax3 = fig.add_subplot(gs[2,i])
		_ = ad_plot._draw_using_figure(fig, [ax1])
		_ = nb_plot._draw_using_figure(fig, [ax2])
		_ = lb_plot._draw_using_figure(fig, [ax3])
		_ = ax1.set_ylabel('Average Bout Duration, m')
		_ = ax2.set_ylabel('Numer Bouts')
		_ = ax3.set_ylabel('Longest Bout, m')
		_ = ax1.set_title(state_keys[i])
	hourly_df = pd.DataFrame(data_list.groupby([data_list.file, data_list.datetime.dt.day,data_list.datetime.dt.hour]).apply(lambda x: np.mean(x['label']==0)))
	hourly_df.index.names = ['file','day','hour']
	hourly_df = hourly_df.reset_index()
	hourly_df = hourly_df.rename(columns={0:'Wake_GT'})
	hourly_df['Wake_Pred'] = pd.DataFrame(data_list.groupby([data_list.file, data_list.datetime.dt.day,data_list.datetime.dt.hour]).apply(lambda x: np.mean(x['prediction']==0))).values
	hourly_df['NREM_GT'] = pd.DataFrame(data_list.groupby([data_list.file, data_list.datetime.dt.day,data_list.datetime.dt.hour]).apply(lambda x: np.mean(x['label']==1))).values
	hourly_df['NREM_Pred'] = pd.DataFrame(data_list.groupby([data_list.file, data_list.datetime.dt.day,data_list.datetime.dt.hour]).apply(lambda x: np.mean(x['prediction']==1))).values
	hourly_df['REM_GT'] = pd.DataFrame(data_list.groupby([data_list.file, data_list.datetime.dt.day,data_list.datetime.dt.hour]).apply(lambda x: np.mean(x['label']==2))).values
	hourly_df['REM_Pred'] = pd.DataFrame(data_list.groupby([data_list.file, data_list.datetime.dt.day,data_list.datetime.dt.hour]).apply(lambda x: np.mean(x['prediction']==2))).values
	hourly_df['time'] = np.reshape(pd.DataFrame(data_list.groupby([data_list.file, data_list.datetime.dt.day,data_list.datetime.dt.hour]).apply(lambda x: x['datetime'].head(1))).values, [-1])
	hour_align_df = pd.DataFrame(hourly_df.groupby('file').apply(lambda x: (hourly_df['time'].head(1).values.astype('datetime64[D]')-x['time'].head(1).values.astype('datetime64[D]'))[0])).reset_index()
	hourly_df['time'] = [(x['time']+hour_align_df[hour_align_df['file']==x['file']][0].values)[0] for i,x in hourly_df.iterrows()]
	hourly_df['time_zt'] = hourly_df['time']-timedelta(hours=10)
	hourly_df = hourly_df[~(hourly_df['time']>='2020-03-04 10:00:00')]
	hourly_df['is_baseline'] = [re.search('Meth',x)==None for x in hourly_df['file']]
	hourly_df.time_zt = hourly_df.time_zt.dt.hour
	# Compare GT and Pred
	fig = (p9.ggplot()+p9.geom_blank(data=hourly_df)+p9.theme_void()).draw()
	gs = gridspec.GridSpec(3,2)
	ax1 = fig.add_subplot(gs[0,0])
	ax2 = fig.add_subplot(gs[1,0])
	ax3 = fig.add_subplot(gs[2,0])
	ax4 = fig.add_subplot(gs[0,1])
	ax5 = fig.add_subplot(gs[1,1])
	ax6 = fig.add_subplot(gs[2,1])
	_ = plot_gtvpd_hourly(hourly_df[hourly_df.is_baseline], 'Wake_')._draw_using_figure(fig, [ax1])
	_ = plot_gtvpd_hourly(hourly_df[hourly_df.is_baseline], 'NREM_')._draw_using_figure(fig, [ax2])
	_ = plot_gtvpd_hourly(hourly_df[hourly_df.is_baseline], 'REM_')._draw_using_figure(fig, [ax3])
	_ = plot_gtvpd_hourly(hourly_df[~hourly_df.is_baseline], 'Wake_')._draw_using_figure(fig, [ax4])
	_ = plot_gtvpd_hourly(hourly_df[~hourly_df.is_baseline], 'NREM_')._draw_using_figure(fig, [ax5])
	_ = plot_gtvpd_hourly(hourly_df[~hourly_df.is_baseline], 'REM_')._draw_using_figure(fig, [ax6])
	_ = ax3.set_xlabel('ZT (Hours)')
	_ = ax6.set_xlabel('ZT (Hours)')
	_ = ax2.set_ylabel('Percent time spent during hour')
	_ = ax1.set_title('Baseline\nWake')
	_ = ax2.set_title('NREM')
	_ = ax3.set_title('REM')
	_ = ax4.set_title('Methamphetamine\nWake')
	_ = ax5.set_title('NREM')
	_ = ax6.set_title('REM')
	# Compare effect of Meth
	# Grab significances
	wake_gt_pvals = []
	nrem_gt_pvals = []
	rem_gt_pvals = []
	wake_pd_pvals = []
	nrem_pd_pvals = []
	rem_pd_pvals = []
	for group, group_df in hourly_df.groupby('time_zt'):
		bl = group_df[group_df.is_baseline]
		me = group_df[~group_df.is_baseline]
		wake_gt_pvals.append(ttest_ind(bl['Wake_GT'],me['Wake_GT'])[1])
		nrem_gt_pvals.append(ttest_ind(bl['NREM_GT'],me['NREM_GT'])[1])
		rem_gt_pvals.append(ttest_ind(bl['REM_GT'],me['REM_GT'])[1])
		wake_pd_pvals.append(ttest_ind(bl['Wake_Pred'],me['Wake_Pred'])[1])
		nrem_pd_pvals.append(ttest_ind(bl['NREM_Pred'],me['NREM_Pred'])[1])
		rem_pd_pvals.append(ttest_ind(bl['REM_Pred'],me['REM_Pred'])[1])
	wake_gt_sigs = np.where(multipletests(wake_gt_pvals, 0.05)[0])[0]%24
	wake_pd_sigs = np.where(multipletests(wake_pd_pvals, 0.05)[0])[0]%24
	nrem_gt_sigs = np.where(multipletests(nrem_gt_pvals, 0.05)[0])[0]%24
	nrem_pd_sigs = np.where(multipletests(nrem_pd_pvals, 0.05)[0])[0]%24
	rem_gt_sigs = np.where(multipletests(rem_gt_pvals, 0.05)[0])[0]%24
	rem_pd_sigs = np.where(multipletests(rem_pd_pvals, 0.05)[0])[0]%24
	fig = (p9.ggplot()+p9.geom_blank(data=hourly_df)+p9.theme_void()).draw()
	gs = gridspec.GridSpec(3,2)
	ax1 = fig.add_subplot(gs[0,0])
	ax2 = fig.add_subplot(gs[1,0])
	ax3 = fig.add_subplot(gs[2,0])
	ax4 = fig.add_subplot(gs[0,1])
	ax5 = fig.add_subplot(gs[1,1])
	ax6 = fig.add_subplot(gs[2,1])
	_ = plot_hourly(hourly_df, 'Wake_GT', sigs=wake_gt_sigs)._draw_using_figure(fig, [ax1])
	_ = plot_hourly(hourly_df, 'NREM_GT', sigs=nrem_gt_sigs)._draw_using_figure(fig, [ax2])
	_ = plot_hourly(hourly_df, 'REM_GT', sigs=rem_gt_sigs)._draw_using_figure(fig, [ax3])
	_ = plot_hourly(hourly_df, 'Wake_Pred', sigs=wake_pd_sigs)._draw_using_figure(fig, [ax4])
	_ = plot_hourly(hourly_df, 'NREM_Pred', sigs=nrem_pd_sigs)._draw_using_figure(fig, [ax5])
	_ = plot_hourly(hourly_df, 'REM_Pred', sigs=rem_pd_sigs)._draw_using_figure(fig, [ax6])
	_ = ax3.set_xlabel('ZT (Hours)')
	_ = ax6.set_xlabel('ZT (Hours)')
	_ = ax2.set_ylabel('Percent time spent during hour')
	_ = ax1.set_title('EEG/EMG\nWake')
	_ = ax2.set_title('NREM')
	_ = ax3.set_title('REM')
	_ = ax4.set_title('Visual Prediction\nWake')
	_ = ax5.set_title('NREM')
	_ = ax6.set_title('REM')
	# Stats for only 2 hours after injection
	df_post_inj_wide = hourly_df[np.isin(hourly_df.time_zt, [2,3,6,7])]
	df_post_inj = pd.wide_to_long(df_post_inj_wide, stubnames=['Wake','NREM','REM'], i=['file','time_zt'], j='pred', sep='_', suffix='(Pred|GT)').reset_index(drop=False)
	# Make an easier to read x-axis
	df_post_inj['Group'] = [re.sub('GT','EEG/EMG',re.sub('Pred','Visual Prediction',x)) + '\n' + ['Methamphetamine','Baseline'][int(y)] for x,y in zip(df_post_inj['pred'], df_post_inj['is_baseline'])]
	fig = (p9.ggplot()+p9.geom_blank(data=df_post_inj)+p9.theme_void()).draw()
	gs = gridspec.GridSpec(3,1)
	ax1 = fig.add_subplot(gs[0,0])
	ax2 = fig.add_subplot(gs[1,0])
	ax3 = fig.add_subplot(gs[2,0])
	_ = (p9.ggplot(p9.aes(x='Group', y='Wake'), data=df_post_inj)+p9.stat_summary(fun_y=lambda x: np.mean(x), geom='bar')+p9.stat_summary(fun_ymin=lambda x: np.mean(x)-sem(x), fun_ymax=lambda x: np.mean(x)+sem(x), geom='linerange')+p9.theme_bw()+p9.labs(x='',y='')+p9.scale_x_discrete(limits=['EEG/EMG\nBaseline','Visual Prediction\nBaseline','EEG/EMG\nMethamphetamine','Visual Prediction\nMethamphetamine']))._draw_using_figure(fig, [ax1])
	_ = (p9.ggplot(p9.aes(x='Group', y='NREM'), data=df_post_inj)+p9.stat_summary(fun_y=lambda x: np.mean(x), geom='bar')+p9.stat_summary(fun_ymin=lambda x: np.mean(x)-sem(x), fun_ymax=lambda x: np.mean(x)+sem(x), geom='linerange')+p9.theme_bw()+p9.labs(x='',y='')+p9.scale_x_discrete(limits=['EEG/EMG\nBaseline','Visual Prediction\nBaseline','EEG/EMG\nMethamphetamine','Visual Prediction\nMethamphetamine']))._draw_using_figure(fig, [ax2])
	_ = (p9.ggplot(p9.aes(x='Group', y='REM'), data=df_post_inj)+p9.stat_summary(fun_y=lambda x: np.mean(x), geom='bar')+p9.stat_summary(fun_ymin=lambda x: np.mean(x)-sem(x), fun_ymax=lambda x: np.mean(x)+sem(x), geom='linerange')+p9.theme_bw()+p9.labs(x='',y='')+p9.scale_x_discrete(limits=['EEG/EMG\nBaseline','Visual Prediction\nBaseline','EEG/EMG\nMethamphetamine','Visual Prediction\nMethamphetamine']))._draw_using_figure(fig, [ax3])
	_ = ax2.set_ylabel('Percent time spent 2 Hours Post Injection')
	_ = ax1.set_title('Wake')
	_ = ax2.set_title('NREM')
	_ = ax3.set_title('REM')
	fig.subplots_adjust(left=0.1, bottom=0.07, right=0.95, top=0.95, wspace=0.2, hspace=0.37)
	print_differences(df_post_inj_wide, 'Wake')
	print_differences(df_post_inj_wide, 'NREM')
	print_differences(df_post_inj_wide, 'REM')
	plt.show(block=True)

def main(argv):
	parser = argparse.ArgumentParser(description='Reports information on sleep results')
	group = parser.add_mutually_exclusive_group(required=True)
	group.add_argument('--input_file', help='Input dataset to analyze')
	group.add_argument('--input_folder', help='Input folder with multiple datasets to analyze')
	args = parser.parse_args()
	if args.input_file is not None:
		plot_single_file(args)
	else:
		plot_multi_file(args)

if __name__ == '__main__':
	main(sys.argv[1:])
