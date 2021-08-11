from sklearn.metrics import confusion_matrix
from sklearn.feature_selection import RFE
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn import metrics
import time
import pandas as pd
import warnings
warnings.filterwarnings("ignore")
import scipy as sp
import scipy.fftpack
from scipy.signal import welch
from scipy import signal
import numpy as np
from sklearn.metrics import confusion_matrix, make_scorer, classification_report
import os, sys, argparse
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression
from hmm_filter.hmm_filter import HMMFilter

# Constants for states
WAKE_STATE = 0
NREM_STATE = 1
REM_STATE = 2

def get_info(df):
	x= df.groupby('Stage').apply(lambda x: x.shape[0])
	return x

def get_balanced_data(df):
	df=df.set_index('unique_epoch_id', drop=False)
	to_sample = df.groupby(['Stage']).groups
	num_to_sample = min(get_info(df))
	samples = np.concatenate([np.random.choice(to_sample[(1)], num_to_sample, False),np.random.choice(to_sample[(2)], num_to_sample, False),np.random.choice(to_sample[(0)], num_to_sample, False)])
	sampled_data = df[df['unique_epoch_id'].isin(samples)].reset_index(drop=True)
	print(get_info(sampled_data))
	return sampled_data

def svm(X_train, X_test, Y_train, Y_test):
	clf = SVC(kernel='rbf')
	clf.fit(X_train, Y_train)
	y_pred=clf.predict(X_test)
	toc = time.time()
	training_accuracy = metrics.accuracy_score(Y_train, clf.predict(X_train))
	test_accuracy = metrics.accuracy_score(Y_test, y_pred)
	print("Training accuracy", training_accuracy)
	print("Test Accuracy:", test_accuracy)
	print(classification_report(Y_test, y_pred))
	return clf, training_accuracy, test_accuracy

def random_forest(X_train, X_test, Y_train, Y_test):
	clf = RandomForestClassifier(n_estimators=100)
	clf.fit(X_train,Y_train)
	y_pred=clf.predict(X_test)  
	toc = time.time()
	training_accuracy = metrics.accuracy_score(Y_train, clf.predict(X_train))
	test_accuracy = metrics.accuracy_score(Y_test, y_pred)
	print("Training accuracy", training_accuracy)
	print("Test Accuracy:", test_accuracy)
	print(classification_report(Y_test, y_pred))
	return clf, training_accuracy, test_accuracy

def XGBoost(X_train, X_test, Y_train, Y_test):
	clf = XGBClassifier()
	clf.fit(X_train, Y_train)  
	y_pred=clf.predict(X_test) 
	toc = time.time()
	training_accuracy = metrics.accuracy_score(Y_train, clf.predict(X_train))
	test_accuracy = metrics.accuracy_score(Y_test, y_pred)
	print("Training accuracy", training_accuracy)
	print("Test Accuracy:", test_accuracy)
	print(classification_report(Y_test, y_pred))
	return clf, training_accuracy, test_accuracy

def neural_network(X_train, X_test, Y_train, Y_test):
	clf = MLPClassifier(random_state=1)
	clf.fit(X_train, Y_train)  
	y_pred=clf.predict(X_test) 
	toc = time.time()
	training_accuracy = metrics.accuracy_score(Y_train, clf.predict(X_train))
	test_accuracy = metrics.accuracy_score(Y_test, y_pred)
	print("Training accuracy", training_accuracy)
	print("Test Accuracy:", test_accuracy)
	print(classification_report(Y_test, y_pred)) 
	return clf, training_accuracy, test_accuracy

def logistic_regression(X_train, X_test, Y_train, Y_test):
	clf = LogisticRegression()
	clf.fit(X_train, Y_train)
	y_pred=clf.predict(X_test)
	toc = time.time()
	training_accuracy = metrics.accuracy_score(Y_train, clf.predict(X_train))
	test_accuracy = metrics.accuracy_score(Y_test, y_pred)
	print("Training accuracy", training_accuracy)
	print("Test Accuracy:", test_accuracy)
	print(classification_report(Y_test, y_pred))
	return clf, training_accuracy, test_accuracy

def hmm_filter(clf, t, X_train, Y_train, X_test, Y_test, S_train, S_test):
	y_pred_train=clf.predict(X_train)
	train_data=pd.DataFrame(Y_train,columns=["Stage"])
	train_data["predict_stage"]=y_pred_train
	train_data['video'] = S_train
	hmmfilter = HMMFilter()
	hmmfilter.A = t
	hmmfilter.fit(train_data, session_column="video", prediction_column="predict_stage")
	d = pd.DataFrame.from_records(clf.predict_proba(X_test), columns=clf.classes_).to_dict(orient="records")
	test_data=pd.DataFrame(Y_test,columns=["Stage"])
	test_data["predict_stage"]=clf.predict(X_test)
	test_data["probabs"] = [{ k:v for k,v in r.items() if v > 0} for r in d ]
	test_data["index"] = np.arange(0, len(test_data))
	test_data['video'] = S_test
	df_hmm = hmmfilter.predict(test_data, session_column='video', probabs_column="probabs", prediction_column='predict_stage')
	df_hmm=df_hmm.set_index("index")
	df_hmm = df_hmm.sort_values(by=['index'])
	classifiers_accuracy = len(test_data[test_data.Stage == test_data.predict_stage]) / len(test_data)
	hmm_accuracy = len(df_hmm[df_hmm.Stage == df_hmm.predict_stage]) / len(df_hmm)
	return [classifiers_accuracy, hmm_accuracy, df_hmm]

def get_t_matrix(df):
	df_temp=df[["unique_epoch_id","video","Stage"]]
	df_temp=df_temp.drop_duplicates()
	df_temp["previous_stage"] = df_temp["Stage"].shift()
	df_temp = df_temp.reset_index()
	df_temp["previous_stage"].iloc[0] = df_temp["Stage"].iloc[0] 
	df_temp['previous_stage'] = df_temp['previous_stage'].astype(int)
	w2w = len(df_temp[(df_temp["Stage"]==WAKE_STATE) & (df_temp["previous_stage"]==WAKE_STATE)])
	w2n = len(df_temp[(df_temp["Stage"]==NREM_STATE) & (df_temp["previous_stage"]==WAKE_STATE)])
	w2r = len(df_temp[(df_temp["Stage"]==REM_STATE) & (df_temp["previous_stage"]==WAKE_STATE)])
	n2w = len(df_temp[(df_temp["Stage"]==WAKE_STATE) & (df_temp["previous_stage"]==NREM_STATE)])
	n2n = len(df_temp[(df_temp["Stage"]==NREM_STATE) & (df_temp["previous_stage"]==NREM_STATE)])
	n2r = len(df_temp[(df_temp["Stage"]==REM_STATE) & (df_temp["previous_stage"]==NREM_STATE)])
	r2w = len(df_temp[(df_temp["Stage"]==WAKE_STATE) & (df_temp["previous_stage"]==REM_STATE)])
	r2n = len(df_temp[(df_temp["Stage"]==NREM_STATE) & (df_temp["previous_stage"]==REM_STATE)])
	r2r = len(df_temp[(df_temp["Stage"]==REM_STATE) & (df_temp["previous_stage"]==REM_STATE)])
	from_w = w2w+w2n+w2r
	from_n = n2w+n2n+n2r
	from_r = r2w+r2n+r2r
	t={(0,0):w2w/from_w,(0,1):w2n/from_w,(0,2):w2r/from_w,
	   (1,0):n2w/from_n,(1,1):n2n/from_n,(1,2):n2r/from_n,
	   (2,0):r2w/from_r,(2,1):r2n/from_r,(2,2):r2r/from_r}
	transition_matrix=pd.DataFrame([[w2w,w2n,w2r],[n2w,n2n,n2r],[r2w,r2n,r2r]],columns=["wake","nrem","rem"],index=["wake","nrem","rem"])
	transition_matrix = transition_matrix/transition_matrix.sum()
	return t

def train_classifier(args):
	# Assign the random seed for reproducibility
	if args.use_random_seed is not None:
		np.random.seed(args.use_random_seed)
	else:
		seed = np.random.randint(4294967295)
		np.random.seed(seed)
		print('Random seed used in training: ' + str(seed))
	df_features_full = pd.read_csv(args.train_dataset)
	df_features_balanced = get_balanced_data(df_features_full)
	# Splint train/test
	mouse_train, mouse_test=train_test_split(df_features_balanced["video"].unique(), test_size=0.3)
	signals = ['m00', 'perimeter','w', 'l', "wl_ratio", "dx", "dy", "dx2_plus_dy2",'hu0', 'hu1', 'hu2', 'hu3', 'hu4', 'hu5', 'hu6']
	colnames_surfix=["__k", "__k_psd", "__s_psd", "__MPL_1", "__MPL_3", "__MPL_5", "__MPL_8","__MPL_15", "__Tot_PSD", "__Max_PSD", "__Ave_PSD", "__Std_PSD", "__Ave_Signal", "__Std_Signal","__Top_Signal"]
	features=[s + c for s in signals for c in colnames_surfix]
	df_train = df_features_balanced[df_features_balanced["video"].isin(mouse_train)]
	df_test = df_features_full[df_features_full["video"].isin(mouse_test)]
	X_train = df_train[features].values
	X_test =  df_test[features].values
	Y_train = df_train["Stage"].values
	Y_test = df_test["Stage"].values
	S_train = df_train['video'].values
	S_test = df_test['video'].values
	# Train the base classifier
	if args.classifier == 'svm':
		classifier, train_acc, test_acc = svm(X_train, X_test, Y_train, Y_test)
	elif args.classifier == 'rf':
		classifier, train_acc, test_acc = random_forest(X_train, X_test, Y_train, Y_test)
	elif args.classifier == 'xgb':
		classifier, train_acc, test_acc = XGBoost(X_train, X_test, Y_train, Y_test)
	elif args.classifier == 'nn':
		classifier, train_acc, test_acc = neural_network(X_train, X_test, Y_train, Y_test)
	elif args.classifier == 'lr':
		classifier, train_acc, test_acc = logistic_regression(X_train, X_test, Y_train, Y_test)
	# Apply optional HMM
	if args.add_hmm:
		t_mat = get_t_matrix(df_features_full[df_features_full["video"].isin(mouse_train)])
		pre_hmm_acc, hmm_acc, hmm_df = hmm_filter(classifier, t_mat, X_train, Y_train, X_test, Y_test, S_train, S_test)
		predictions = hmm_df.predict_stage
		print('HMM increased accuracy in training data from ' + str(pre_hmm_acc) + ' to ' + str(hmm_acc))
	else:
		predictions = classifier.predict(X_test)
	print('Overall classifier test set performance: ')
	print('Accuracy: ' + str(metrics.accuracy_score(Y_test, predictions)))
	print('Precision: ' + str(metrics.precision_score(Y_test, predictions, average=None)))
	print('Recall: ' + str(metrics.recall_score(Y_test, predictions, average=None)))
	# Predict on dataset
	if args.predict_dataset is not None:
		df_infer_dataset = pd.read_csv(args.predict_dataset)
		X_infer =  df_infer_dataset[features].values
		Y_infer = df_infer_dataset["Stage"].values
		if args.add_hmm:
			pre_hmm_acc, hmm_acc, hmm_df = hmm_filter(classifier, t_mat, X_train, Y_train, X_infer, Y_infer, S_train, df_infer_dataset['video'])
			print('HMM increased accuracy in inference data from ' + str(pre_hmm_acc) + ' to ' + str(hmm_acc))
			Y_infer_pred = hmm_df.predict_stage

		else:
			Y_infer_pred = classifier.predict(X_infer)
		# Write out results
		out_fname = os.path.splitext(args.predict_dataset)[0] + '_predictions.csv'
		results = pd.DataFrame({'unique_epoch_id': df_infer_dataset['unique_epoch_id'].values, 'label':Y_infer, 'prediction':Y_infer_pred}).to_csv(out_fname, index=False)

def main(argv):
	parser = argparse.ArgumentParser(description='Trains a classifier for sleep state prediction')
	parser.add_argument('--train_dataset', help='Training dataset', required=True)
	parser.add_argument('--classifier', help='Classifier to use', required=True, choices=['svm','rf','xgb','nn','lr'])
	parser.add_argument('--add_hmm', help='Add HMM after classification', default=False, action='store_true')
	parser.add_argument('--use_random_seed', help='Select random seed', default=None, type=np.uint32)
	parser.add_argument('--predict_dataset', help='Optional data to predict on', default=None)
	args = parser.parse_args()
	print('Args used: ' + str(args))
	train_classifier(args)

if __name__ == '__main__':
	main(sys.argv[1:])
