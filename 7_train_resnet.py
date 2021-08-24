#!/usr/bin/python

# Matthew Leming, MGH Center for Systems Biology, 2021
# Trains a 3D ResNet on a matched dataset.
# Parameters for which variables are used to balance, and the ranges of these
# variables, are in the covariate pairings file. This script iteratively loads
# .npy files in from the balanced filename list and trains them on the model
# with only 500 in memory at a given time.

import os,sys,glob,json,argparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
working_dir = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))

parser = argparse.ArgumentParser(description = "Used for testing models")
parser.add_argument('--test_variable',default='Test_Control')
parser.add_argument('--gpu',default="0")
parser.add_argument('--num_iters',default=100,type=int)
parser.add_argument('--load_only',action='store_true',default=False)
parser.add_argument('--meta',type=str,default="")
parser.add_argument('--no_train_if_exists',default=False,action='store_true')
parser.add_argument('--var_file',default=os.path.join(working_dir,'json','covariate_pairings','covariate_pairings.json'),type=str)
parser.add_argument('--get_all_test_set',action='store_true',default=False)
parser.add_argument('--test_predictions_filename',default='test_predictions.json',type=str)
parser.add_argument('--total_size_limit',default=None,type=int)
parser.add_argument('--test_count',default=500,type=int,help='Number of .npy files in each class to load into memory at a time (2 classes * 500 = 1000)')
args = parser.parse_args()

os.environ["CUDA_VISIBLE_DEVICES"]=args.gpu
from resnet3d import Resnet3DBuilder
from sklearn import preprocessing
from get_balanced_training_set import *
from get_3D_models import *
from tensorflow.keras.models import load_model
from sklearn.metrics import roc_curve, auc

var_dict = json.load(open(args.var_file,'r'))

# Functions used for string matching on ProtocolName directly if that is preferred
# over ProtocolNameSimplified
t1_func = lambda k: "t1" in k.lower() or np.all([ kk not in k.lower() for kk in ["t2","flair","dwi","fraction","swi","zero_b"]])
t2_func = lambda k: "t2" in k.lower()# or "flair" in k.lower()
flair_func = lambda k: "flair" in k.lower()
mprage_func = lambda k: "mprage" in k.lower()

if "ProtocolName" in var_dict:
	func_dict = {"flair_func":flair_func,"t1_func":t1_func,"t2_func":t2_func,"mprage_func":mprage_func}
	func_description = var_dict["value_ranges"]["ProtocolName"]
	var_dict["value_ranges"]["ProtocolName"] = func_dict[func_description]

test_variable = args.test_variable

confounds = [s for s in var_dict["value_ranges"]]
value_ranges = [var_dict["value_ranges"][s] for s in confounds]
for i in range(len(value_ranges)):
	if isinstance(value_ranges[i],list) and len(value_ranges[i]) == 2:
		if np.all([isinstance(k,int) or isinstance(k,float) for k in value_ranges[i]]):
			value_ranges[i] = tuple(value_ranges[i])

state = {}

models_dir = os.path.join(working_dir,'models')
if not os.path.isdir(models_dir):
	os.makedirs(models_dir)

current_model_dir = os.path.join(models_dir,test_variable)
if args.meta != "":
	current_model_dir = os.path.join(models_dir,test_variable,"%s_%s" % (test_variable,args.meta))

best_model_dir            = os.path.join(current_model_dir,'model')
best_model_state          = os.path.join(current_model_dir,'state.json')
parameters_state          = os.path.join(current_model_dir,'parameters.json')
np_dir                    = os.path.join(current_model_dir,'npy')
output_covars_savepath    = os.path.join(current_model_dir,'cache','%s_balanced.pkl'%test_variable)
output_selection_savepath = os.path.join(current_model_dir,'cache','%s_balanced.npy'%test_variable)
output_test_predictions   = os.path.join(current_model_dir,args.test_predictions_filename)

imsize = (96,96,96)
test_count = args.test_count
if not os.path.isdir(np_dir): os.makedirs(np_dir)

# Note: look at PatientClassDSC to consider injury likelihood -- inpatient versus outpatient versus emergency

print("Test variable: %s" % test_variable)
if test_variable in confounds: # Used to avoid rewriting arrays every time the test variable is changed
	i = confounds.index(test_variable)
	print("Deleting %s from confounds" % confounds[i])
	del confounds[i]
	del value_ranges[i]

if test_variable in var_dict["exclusions"]:
	for con in var_dict["exclusions"][test_variable]:
		if con in confounds:
			i = confounds.index(con)
			print("Removing %s" % confounds[i])
			del confounds[i]
			del value_ranges[i]
	if '*' in var_dict["exclusions"][test_variable]:
		print("Excluding all")
		confounds = []
		value_ranges = []

assert(test_variable in var_dict["labels"])
training_labels = var_dict["labels"][test_variable]

assert(len(confounds) == len(value_ranges))
assert(test_variable not in confounds)

parameters = {}
for i in range(len(confounds)):
	if confounds[i] == "ProtocolName":
		parameters[confounds[i]] = func_description
	else:
		parameters[confounds[i]] = value_ranges[i]
parameters["labels"] = training_labels
parameters["test_variable"] = test_variable

json.dump(parameters,open(parameters_state,'w'),indent=4)
setdict = {}
label_iterator = {"train":{},"test":{},"valid":{}}
for setname in label_iterator:
	for label in training_labels:
		if label not in label_iterator[setname]:
			label_iterator[setname][label] = 0
print("Training labels: %s" % str(training_labels))
train_now = True
if os.path.isfile(best_model_state):
	state = json.load(open(best_model_state,'r'))

	if "complete_test_set" in state:
		if not (args.get_all_test_set and state["complete_test_set"] == True):
			exit()
		elif args.get_all_test_set and state["complete_test_set"] == "all":
			exit()
	X_train = np.load(state["X_train"],allow_pickle=True)
	X_valid = np.load(state["X_valid"],allow_pickle=True)
	Y_train = np.load(state["Y_train"],allow_pickle=True)
	Y_valid = np.load(state["Y_valid"],allow_pickle=True)
	if args.get_all_test_set and \
		os.path.isfile(state["X_test"].replace('X_test.npy','X_test_all.npy')) and \
		os.path.isfile(state["Y_test"].replace('Y_test.npy','Y_test_all.npy')):
		X_test  = np.load(state["X_test"].replace('X_test.npy','X_test_all.npy'),allow_pickle=True)
		Y_test  = np.load(state["Y_test"].replace('Y_test.npy','Y_test_all.npy'),allow_pickle=True)
	else:
		X_test  = np.load(state["X_test"],allow_pickle=True)
		Y_test  = np.load(state["Y_test"],allow_pickle=True)
else:
	[X_train,X_test,X_valid],[Y_train,Y_test,Y_valid] = \
		get_balanced_filename_list(test_variable,confounds,
		selection_ratios = [0.66,0.16,0.16],
		selection_limits = [np.Inf,np.Inf,np.Inf],
		value_ranges = value_ranges,
		output_covars_savepath = output_covars_savepath,
		test_value_ranges = training_labels,
		output_selection_savepath = output_selection_savepath,
		get_all_test_set=args.get_all_test_set,
		total_size_limit = args.total_size_limit)
	#[X_train,X_test,X_valid],[Y_train,Y_test,Y_valid] = [None,None,None],[None,None,None]
	state["total_data_size"] = int(np.sum([len(y) for y in [Y_train,Y_test,Y_valid]]))
	for varname,var in [("X_train",X_train),("X_test",X_test),("X_valid",X_valid),("Y_train",Y_train),("Y_test",Y_test),("Y_valid",Y_valid)]:
		state[varname] = os.path.join(np_dir,'%s.npy' % varname)
		np.save(os.path.join(np_dir,'%s.npy' % varname),var)
	json.dump(state,open(best_model_state,'w'),indent=4)
print(X_train.shape)
if X_train.shape[0] == 0:
	print("No data returned from query. Check that test variables are not mutual with one or more confounds")
	exit(0)

test_total = len(Y_test)
print("test_variable: %s" %str(test_variable))
print("confounds: %s" % str(confounds))
print("value_ranges: %s" % str(value_ranges))
print("X_train.shape: %s"%str(X_train.shape))
print("X_test.shape: %s"%str(X_test.shape))
print("X_valid.shape: %s"%str(X_valid.shape))
if args.load_only:
	exit()

balanced_labels = {"train": {} , "test" : {},"valid" :{} }

for setname in [k for k in balanced_labels]:
	if setname == "train": X,Y = X_train,Y_train
	elif setname == "test": X,Y = X_test,Y_test
	elif setname == "valid": X,Y = X_valid,Y_valid
	for i in range(len(Y)):
		label = Y[i]
		if label not in balanced_labels[setname]:
			balanced_labels[setname][label] = []
		balanced_labels[setname][label].append(X[i])

def get_nifti_data_balanced(count=20,setname="train",training_labels=[],get_all = False,return_filestubs=False,exclusion_dict=None):
	actual_count = np.sum([len(balanced_labels[setname][label]) for label in training_labels])
	#if exclusion_dict is not None and (len(exclusion_dict) + (count * len(training_labels)) >= actual_count)
	#	count = int()
	if count * len(training_labels) > actual_count:
		get_all = True
	if get_all:
		total_sizes = list(map(lambda k: len(k), [balanced_labels[setname][label] for label in balanced_labels[setname]]))
		total_size = int(np.sum(total_sizes))
		X = np.zeros((total_size,imsize[0],imsize[1],imsize[2],1))
	else:
		X = np.zeros((count * len(training_labels),imsize[0],imsize[1],imsize[2],1))
	Y = []
	i = 0
	epass = 0
	if return_filestubs:
		teh_filestubs = []
	for label in training_labels:
		if get_all:
			count = len(balanced_labels[setname][label])
		for _ in range(count):
			while True:
				filebase = balanced_labels[setname][label][label_iterator[setname][label]]
				partial_path = '%s_resized_%d.npy' % (filebase,imsize[0])
				while partial_path[0] == "/": partial_path = partial_path[1:]
				filepath = os.path.join(working_dir,partial_path)
				folder = os.path.basename(os.path.dirname(filepath))				
				label_iterator[setname][label] = (label_iterator[setname][label] + 1) % len(balanced_labels[setname][label])
				try:
					if exclusion_dict is not None and filebase in exclusion_dict and epass < actual_count:
						epass += 1
						continue
					arr = np.load(filepath)
					break
				except:
					continue
			X[i,:,:,:,:] = np.expand_dims(arr,axis=3)
			Y.append(label)
			if return_filestubs:
				teh_filestubs.append(filebase)
			i += 1
	Y = [str_to_list(k) for k in Y]
	if return_filestubs:
		return X,Y,teh_filestubs
	else:
		return X,Y


# train

lb = preprocessing.MultiLabelBinarizer()
X_test,Y_test,test_filestubs = get_nifti_data_balanced(count = test_count, setname="test",training_labels = training_labels,get_all=False,return_filestubs=True)
lb.fit(Y_test)
parameters["labels"] = [str(_) for _ in lb.classes_]
json.dump(parameters,open(parameters_state,'w'),indent=4)

Y_test = np.array(lb.transform(Y_test))
print(Y_test.shape)
#model = Resnet3DBuilder.build_resnet_18((imsize[0], imsize[1], imsize[2], 1), len(training_labels))
assert(np.all(np.sum(Y_test,axis=1) == 1))

if os.path.isdir(best_model_dir):
	model = load_model(best_model_dir)
else:
	model = get_3D_CNN_model(imsize[0], imsize[1], imsize[2],num_output = Y_test.shape[1])
	model.compile(loss='categorical_crossentropy', optimizer="rmsprop",metrics=["accuracy"])
model.summary()

num_iters = args.num_iters

X_valid,Y_valid = get_nifti_data_balanced(count = test_count, setname="valid",training_labels = training_labels,get_all=False)
Y_valid = np.array(lb.transform(Y_valid))

if "best_valid_prediction" in state:
	best_valid_prediction = state["best_valid_prediction"]
else:
	best_valid_prediction = 0.0

if "label_iterator" in state:
	label_iterator = state["label_iterator"]
else:
	state["label_iterator"] = label_iterator

if "epoch" in state and state["epoch"] > 10 and args.no_train_if_exists:
	train_now = False

if "epoch" not in state:
	state["epoch"] = 0


def get_acc(X,Y,model,get_auc=False,output_test_predictions = None,
	auroc_graph_output_file=None,labels_prediction_files=None,
	filenames=None):
	prediction = model.predict(X)
	if output_test_predictions is not None and filenames is not None:
		if os.path.isfile(output_test_predictions):
			d = json.load(open(output_test_predictions,'r'))
		else:
			d = {}
		for i in range(len(filenames)):
			d[filenames[i]] = [list(prediction[i,:].astype(float)),list(Y[i,:].astype(float))]
		json.dump(d,open(output_test_predictions,'w'),indent=4)
	pred_bin = np.zeros(prediction.shape).astype(bool)
	am = np.argmax(prediction,axis=1)
	for i in range(pred_bin.shape[0]):
		pred_bin[i,am[i]] = True
	accs = []
	for i in range(Y.shape[1]):
		accs.append(np.mean(np.abs(Y[:,i].astype(bool) == pred_bin[:,i])))
	if get_auc:
		labels = lb.classes_
		aucs = []
		for i in range(Y.shape[1]):
			fpr, tpr, threshold = roc_curve(Y[:,i], prediction[:,i])
			roc_auc = auc(fpr, tpr)
			aucs.append(roc_auc)
			plt.plot(fpr, tpr, label = '%s (AUC = %0.3f)' % (labels[i].upper(),roc_auc))
		plt.legend(loc = 'lower right')
		plt.plot([0, 1], [0, 1],'r--')
		plt.xlim([0, 1])
		plt.ylim([0, 1])
		plt.ylabel('True Positive Rate')
		plt.xlabel('False Positive Rate')
		plt.title("%s (n = %d)" % (test_variable,int(Y.shape[0])))
		if not auroc_graph_output_file is None:
			plt.savefig(auroc_graph_output_file)
		plt.clf()
		if not (labels_prediction_files is None):
			for i in range(prediction.shape[1]):
				l = {}
				label = "%s_%s_predictions" % (test_variable,labels[i])
				l["discrete"] = False
				l["prediction"] = True
				l[label] = []
				for j in range(prediction.shape[0]):
					filestub = filenames[j]
					l[label].append((float(prediction[j,i]),filestub))
				json.dump(l,open(os.path.join(labels_prediction_files,"%s.json" % label),'w'),indent=4)
		
		return np.mean(accs),aucs
	else:
		return np.mean(accs)

if train_now:	
	for i in range(state["epoch"],num_iters):
		print("%d/%d" % (i,num_iters))
		print(label_iterator)
		X_train,Y_train=get_nifti_data_balanced(setname = "train",count=test_count,training_labels = training_labels,get_all=False)
		
		Y_train = np.array(lb.transform(Y_train))
		model.fit(X_train,Y_train,batch_size=32,epochs=10,verbose=False)
		
		prediction_acc,prediction_aucs = get_acc(X_valid,Y_valid,model,get_auc=True)
		if np.mean(prediction_aucs) > best_valid_prediction:
			model.save(best_model_dir)
			best_valid_prediction = np.mean(prediction_aucs)
			state["epoch"] = i
			state["best_valid_prediction"] = best_valid_prediction
			best_test_prediction,best_test_aurocs = get_acc(X_test,
						Y_test,model,
						get_auc=True,
						filenames=test_filestubs,
						auroc_graph_output_file=os.path.join(current_model_dir,'auroc_plot.png'),
						output_test_predictions = output_test_predictions)
						#labels_prediction_files=os.path.join(working_dir,'json','labels'),
			state["best_test_prediction"] = np.mean(best_test_aurocs)
			state["best_test_auroc"] = best_test_aurocs
			state["label_iterator"] = label_iterator
			json.dump(state,open(best_model_state,'w'),indent=4)
		print("Test set prediction: %f" % prediction_acc)

if "complete_test_set" not in state or (state["complete_test_set"] == True and args.get_all_test_set):
	model = load_model(best_model_dir)
	print("test_total: %d" % test_total)
	print("test_count: %d" % test_count)
	print("int(test_total / test_count) - 1: %d" % (int(test_total / test_count) - 1))
	print(output_test_predictions)
	output_test_predictions_list = []
	#if os.path.isfile(output_test_predictions):
	#	ds = json.load(open(output_test_predictions,'r'))
	#else:
	#	ds = {}
	if "current_test_tick" not in state: state["current_test_tick"] = 0
	for i in range(0,int(test_total / test_count) - 1):
		output_test_predictions_cur_file = output_test_predictions.replace('.json','_%d.json'%i)
		output_test_predictions_list.append(output_test_predictions_cur_file)
		if i < state["current_test_tick"]:
			continue
		
		print("%d / %d" % (i,int(test_total / test_count) - 1))
		X_test,Y_test,test_filestubs = get_nifti_data_balanced(setname = "test",
			count = test_count,
			training_labels = training_labels,
			get_all = False,return_filestubs = True,
			exclusion_dict = None)
		state["label_iterator"] = label_iterator
		state["current_test_tick"] = i
		json.dump(state,open(best_model_state,'w'),indent=4)
		Y_test = np.array(lb.transform(Y_test))
		output_test_predictions_cur_file = output_test_predictions
		if args.get_all_test_set:
			if os.path.isfile(output_test_predictions_cur_file):
				try:
					json.load(open(output_test_predictions_cur_file,'r'))
					continue
				except:
					_=1
		best_test_prediction = get_acc(X_test,
					Y_test,model,
					get_auc=False,
					filenames=test_filestubs,
					output_test_predictions = output_test_predictions_cur_file)
	state["complete_test_set"] = "all" if args.get_all_test_set else True
	if args.get_all_test_set:
		ds = {}
		for json_filename in output_test_predictions_list:
			d = json.load(open(json_filename,'r'))
			for e in d:
				ds[e] = d[e]
		json.dump(ds,open(output_test_predictions,'w'),indent=4)
		if os.path.isfile(output_test_predictions):
			try:
				json.load(open(output_test_predictions,'r'))
				for json_filename in output_test_predictions_list:
					#os.remove(json_filename)
					continue
			except:
				print("%s not valid file -- keeping intermediaries" % output_test_predictions)
	json.dump(state,open(best_model_state,'w'),indent=4)
