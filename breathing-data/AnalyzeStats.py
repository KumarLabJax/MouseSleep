from plotnine import *
import pandas as pd
import numpy as np
import re
import statsmodels.api as sm
import statsmodels.formula.api as smf
from sklearn.preprocessing import power_transform
import matplotlib.pyplot as plt

data_all = pd.read_csv('breathing_aspect.csv')
data_all['strain'] = [re.sub('.*(B6|C3).*','\\1',x) for x in data_all['id'].values]
data_all['sex'] = [re.sub('.*/(M|F).*','\\1',x) for x in data_all['id'].values]
data_all['mouse_num'] = [re.sub('[^#]*(#[0-9]+)_PI[0-9]+.*','\\1',x) for x in data_all['id'].values]
data_all['arena_num'] = [re.sub('[^#]*#[0-9]+_(PI[0-9]+).*','\\1',x) for x in data_all['id'].values]
data = data_all.groupby('id').apply(lambda x: x[x['dist']<np.quantile(x['dist'],[0.1])[0]]).reset_index(drop=True)

data_b6 = data[data['strain'].eq('B6')].reset_index(drop=True)
# Count the sleep stages after filter
# data_b6.groupby('sleep_stage')['sleep_stage'].count()/data_b6['sleep_stage'].count()

# Analyze histogram of distances
#data_all_b6 = data_all[data_all['strain'].eq('B6')].reset_index(drop=True)
#data_all_b6 = data_all_b6[data_all_b6['dist'].lt(0.25)].reset_index(drop=True)
#ggplot(data_all_b6, aes(x='dist', fill='sleep_stage'))+geom_histogram(bins=40)+facet_wrap('~mouse_num', scales='free_x')

def get_epoch_percents(input_df, low=0.0, high=1.0):
	df_subset = input_df[input_df['strain'].eq('B6')].reset_index(drop=True)
	df_subset = df_subset.groupby('id').apply(lambda x: x.iloc()[np.intersect1d(np.where(x['dist']<np.quantile(x['dist'],[high])[0]), np.where(x['dist']>np.quantile(x['dist'],[low])[0]))]).reset_index(drop=True)
	return (df_subset.groupby('sleep_stage')['sleep_stage'].count()/df_subset['sleep_stage'].count()).values

sleep_dist_df = pd.DataFrame({'Sleep Stage':['NREM','REM','Wake']}).join(pd.DataFrame(np.transpose(np.stack([get_epoch_percents(data_all, x-0.025, x) for x in np.arange(0.025,1.025,0.025)])), columns=['quant_' + "{:.2f}".format(x-0.0125) for x in np.arange(0.025,1.025,0.025)]))
sleep_dist_df = sleep_dist_df.melt('Sleep Stage')
sleep_dist_df['quantile'] = [np.float(re.sub('quant_([0-9.]+)','\\1',x)) for x in sleep_dist_df['variable'].values]
(ggplot(sleep_dist_df, aes(x='quantile',y='value',color='Sleep Stage'))+geom_line()+geom_vline(aes(xintercept=0.1))+theme_538()+labs(x='Epochs in 0.025 Quantile Range',y='Percent',title='Per-animal Distance Filter')).draw().show()


#(ggplot(data_b6, aes(x='id',y='stds',color='sleep_stage'))+geom_boxplot()+theme_538()+theme(axis_text_x=element_text(angle=90))+facet_grid('.~strain', scales='free_x')).draw().show()

data_sleep = data_all[data_all['sleep_stage'].isin(['REM','NREM'])].reset_index(drop=True)
#(ggplot(data_sleep, aes(x='id',y='stds',color='sleep_stage'))+geom_violin()+stat_summary(fun_y=np.mean, fun_ymin=lambda x: np.mean(x)-np.std(x), fun_ymax=lambda x: np.mean(x)+np.std(x), geom='pointrange', position=position_dodge(0.9))+theme_538()+theme(axis_text_x=element_text(angle=90))+facet_grid('.~strain', scales='free_x')).draw().show()
#(ggplot(data_sleep, aes(x='id',y='means',color='sleep_stage'))+geom_violin()+stat_summary(fun_y=np.mean, fun_ymin=lambda x: np.mean(x)-np.std(x), fun_ymax=lambda x: np.mean(x)+np.std(x), geom='pointrange', position=position_dodge(0.9))+theme_538()+theme(axis_text_x=element_text(angle=90))+facet_grid('.~strain', scales='free_x')).draw().show()

(ggplot(data, aes(x='id',y='means',color='sex'))+geom_violin()+stat_summary(fun_y=np.mean, fun_ymin=lambda x: np.mean(x)-np.std(x), fun_ymax=lambda x: np.mean(x)+np.std(x), geom='pointrange', position=position_dodge(0.9))+theme_538()+theme(axis_text_x=element_text(angle=90))+facet_grid('.~strain', scales='free_x')).draw().show()

# Grouped Trend
(ggplot(data_sleep, aes(x='sleep_stage',y='stds',color='id'))+stat_summary(fun_y=np.mean, geom='point')+stat_summary(aes(group='id'), fun_y=np.mean, geom='line')+theme_538()+theme(axis_text_x=element_text(angle=90))+facet_grid('.~strain', scales='free_x')).draw().show()
# Density Plot
#(ggplot(data_sleep, aes(x='stds', color='id'))+geom_density()+theme_538()+facet_grid('.~sleep_stage')).draw().show()

(ggplot(data_sleep, aes(x='stds', color='sleep_stage', group='sleep_stage'))+stat_bin(aes(y='stat(density)'),geom='line', bins=50)+theme_538()).draw().show()

# Stats comparisons
# Standard deviation REM vs NREM
data_sleep = data_sleep.join(pd.DataFrame(power_transform(data_sleep[['means','stds']], method='box-cox', standardize=True), columns=['means_norm','stds_norm']))
md = smf.mixedlm("stds_norm ~ sleep_stage", data_sleep, groups=data_sleep["id"]).fit(reml=False)
#sm.qqplot(md.resid, line='q')
print(md.summary())

# Means B6 vs C3H
data = data.join(pd.DataFrame(np.concatenate(data.groupby('id').apply(lambda x: power_transform(x[['means','stds']], method='box-cox', standardize=False))), columns=['means_norm','stds_norm']))
md = smf.mixedlm("means ~ strain", data, groups=data["id"]).fit()
print(md.summary())

# Means C3H M vs F
data_c3 = data[data['strain'].eq('C3')].reset_index(drop=True)
data_c3 = data_c3.join(pd.DataFrame(np.concatenate(data_c3.groupby('id').apply(lambda x: power_transform(x[['means','stds']], method='box-cox', standardize=False))), columns=['means_norm','stds_norm']))
md = smf.mixedlm("means ~ sex", data_c3, groups=data_c3["id"]).fit(reml=False)
print(md.summary())
