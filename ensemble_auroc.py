#!/usr/bin/python
import json,os,argparse
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
import numpy as np
#from tensorflow.keras.models import load_model
import pandas as pd
from textwrap import wrap

parser = argparse.ArgumentParser(description = "Finds the aggregate AUROC")
parser.add_argument('json_files', metavar='N', type=str, nargs='+')
parser.add_argument('--plot',action='store_true',default=False)
parser.add_argument('--confound_var',type=str,default="")
parser.add_argument('--num_discrete',type=int,default=6)
parser.add_argument('--disc_mode',type=str,default="modality")
parser.add_argument('--min_samples',type=int,default=1)
parser.add_argument('--max_samples',type=int,default=np.Inf)
parser.add_argument('--max_std',type=float,default=np.inf)
parser.add_argument('--group_folders',default=False,action='store_true')
parser.add_argument('--group_var',default="",type=str)
parser.add_argument('--output_filename',default="",type=str)
parser.add_argument('--noshow',action='store_true',default=False)
parser.add_argument('--nonverbose',action='store_true',default=False)
parser.add_argument('--output_aurocs',action='store_true',default=False)
parser.add_argument('--title',default = "",type=str)

args = parser.parse_args()

assert(not(args.group_folders and args.group_var != ""))

json_files = args.json_files
for json_file in json_files:
	assert os.path.isfile(json_file), "%s doesn't exist" % json_file
test_variable=""
all_file = {}
labels = None

working_dir = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
pandas_cache = os.path.join(working_dir,'pandas','cache','all_vars.pkl')

def iz_nan(k):
	if k is None:
		return True
	try:
		if np.isnan(k):
			return True
		else:
			return False
	except:
		if k == np.nan:
			return True
		else:
			return False

def denanize(arr1,arr2):
	arr1nan = np.array([not iz_nan(i) for i in arr1])
	arr2nan = np.array([not iz_nan(i) for i in arr2])
	bothnan = np.logical_and(arr1nan,arr2nan)
	return arr1[bothnan],arr2[bothnan]

def convert_indices(arr,mode = "discretize"):
	if mode == "discretize":
		sorted_arr = sorted(arr)
		str_arr = []
		range_nums = [sorted_arr[i] for i in range(0,len(arr),int(len(arr)/args.num_discrete))]
		if range_nums[-1] < sorted_arr[-1]: range_nums.append(sorted_arr[-1])
		make_float = np.any(np.array(range_nums).astype(float) != np.array(range_nums).astype(int))
		for i in range(len(arr)):
			s = len(str_arr)
			for j in range(len(range_nums)-1):
				if arr[i] >= range_nums[j] and arr[i] <= range_nums[j+1]:
					if args.confound_var=="Ages":
						str_arr.append("%.1f - %.1f" % (float(range_nums[j])/12,float(range_nums[j+1])/12))
					elif make_float:
						str_arr.append("%.2f - %.2f" % (float(range_nums[j]),float(range_nums[j+1])))
					else:
						str_arr.append("%d - %d" % (int(range_nums[j]),int(range_nums[j+1])))
					break
			assert(len(str_arr) == s + 1)
	elif mode == "modality":
		str_arr = []
		for a in arr:
			if "flair" in a.lower():
				str_arr.append("FLAIR")
			elif "mprage" in a.lower():
				str_arr.append("MPRAGE")
			elif "t1" in a.lower():
				str_arr.append("T1")
			elif "t2" in a.lower():
				str_arr.append("T2")
			elif "dwi" in a.lower():
				str_arr.append("DWI")
			elif "swi" in a.lower():
				str_arr.append("SWI")
			else:
				str_arr.append("Other")
	elif mode == "angle":
		str_arr = []
		for a in arr:
			if "sag" in a.lower():
				str_arr.append("SAG")
			elif "ax" in a.lower():
				str_arr.append("AX")
			elif "cor" in a.lower():
				str_arr.append("COR")
			else:
				str_arr.append("UNKNOWN")
	elif mode == "none":
		return np.array(arr)
	else:
		print("Invalid discretization: %s" % mode)
		return np.array(arr)
	return np.array(str_arr)

for json_file in json_files:
	try:
		f = json.load(open(json_file,'r'))
	except:
		print("Corrupted json file: %s" % json_file)
		continue
	parampath = os.path.join(os.path.dirname(json_file),'parameters.json')
	if (labels is None or test_variable == "") and os.path.isfile(parampath):
		parameters = json.load(open(parampath,'r'))
		if "labels" in parameters:
			labels = parameters["labels"]
		if "test_variable" in parameters:
			test_variable = parameters["test_variable"]
	for filestub in f:
		if filestub not in all_file:
			all_file[filestub] = []
		all_file[filestub].append(f[filestub])

if args.confound_var != "" or args.group_var != "":
	assert(os.path.isfile(pandas_cache))
	all_vars = pd.read_pickle(pandas_cache)
	if args.confound_var != "" and args.confound_var not in all_vars:
		print("%s not a valid variable" % args.confound_var)
		exit()
	if args.group_var != "" and args.group_var not in all_vars:
		print("%s not valid grouping variable" % args.group_var)
		exit()
else:
	all_vars = None

disp_names = {"ProtocolNameSimplified" : args.disc_mode.capitalize(),"AlzStage" : "Alzheimer's"}
for a in [args.confound_var,args.group_var,test_variable]:
	if a not in disp_names:
		disp_names[a] = a

if args.confound_var != "":
	#indices = all_vars.index.to_numpy()
	indices = np.array(list(all_file))
	#tc = all_vars[args.confound_var].to_numpy()
	tc = all_vars.loc[indices,args.confound_var]
	indices,tc = denanize(indices,tc)
	if not isinstance(tc[0],str):
		if not args.nonverbose: print("Discretizing")
		tc = convert_indices(tc,mode="discretize")
	elif args.confound_var == "ProtocolNameSimplified":
		tc = convert_indices(tc,mode=args.disc_mode)
	assert(len(tc) == len(indices))
	unique_covars,counts = np.unique(tc,return_counts = True)
	if len(unique_covars) > args.num_discrete:
		n = sorted(counts)[-1 * args.num_discrete]
		unique_covars = unique_covars[counts > n]
	cov_dict = {}
	if args.group_folders:
		for i in range(len(indices)):
			indices[i] = os.path.dirname(indices[i])
	elif args.group_var != "":
		for i in range(len(indices)):
			indices[i] = "%s_%s"%(all_vars.loc[indices[i],args.group_var],str(all_file[indices[i]][0][1]))
	for i in range(len(tc)):
		cov_dict[indices[i]] = tc[i]
	unique_covars = sorted(unique_covars)
else:
	unique_covars = [""]

if args.group_folders or args.group_var != "":
	unique_groups = set()
	all_file_new = {}
	for filestub in all_file:
		if args.group_folders:
			f = os.path.dirname(filestub)
		elif args.group_var != "":
			f = "%s_%s" % (all_vars.loc[filestub,args.group_var],str(all_file[filestub][0][1]))
		if f not in all_file_new:
			all_file_new[f] = []
		all_file_new[f] = all_file_new[f] + all_file[filestub]
	all_file = all_file_new

for c in unique_covars:
	if not args.nonverbose and not args.output_aurocs: print(c)
	Y = []
	pred = []
	sizes = {}
	
	for filestub in all_file:
		if c != "":
			if filestub not in cov_dict:
				continue
			if cov_dict[filestub] != c:
				continue
		size = len(all_file[filestub])
		if size not in sizes:
			sizes[size] = [0,filestub]
		sizes[size][0] += 1
		if size >= args.min_samples and size <= args.max_samples:
			pred_std = np.std([k[0] for k in all_file[filestub]],axis=0)
			if np.mean(pred_std) < args.max_std:
				Y.append(np.mean([k[1] for k in all_file[filestub]],axis=0))
				pred.append(np.mean([k[0] for k in all_file[filestub]],axis=0))
	
	if c == "" and not args.nonverbose and not args.output_aurocs:
		for key in sorted(sizes):
			print("%d: %d (%s)" % (key,sizes[key][0],sizes[key][1]))
	
	Y = np.array(Y)
	pred = np.array(pred)
	aucs = []
	if True:
		pred_thresh = np.zeros(Y.shape,dtype=float)
		for i in range(Y.shape[0]):
			pred_thresh[i,pred[i,:].argmax()] = 1
		acc_account = np.mean(np.all((pred_thresh == Y),axis=1))
		print("Overall accuracy: %f"%acc_account)
		for i in range(Y.shape[1]):
			fpr, tpr, threshold = roc_curve(Y[:,i], pred[:,i])
			roc_auc = auc(fpr, tpr)
			aucs.append(roc_auc)
			if not args.output_aurocs:
				if labels is not None:
					plt.plot(fpr, tpr, label = '%s (AUC = %0.3f)' % (labels[i].upper(),roc_auc))
				else:
					plt.plot(fpr, tpr, label = 'AUC = %0.3f' % (roc_auc))
		if args.output_aurocs:
			print("%s,%d,%d," %(c,len(json_files),int(Y.shape[0])) + ",".join(["%s,%f" % (labels[i],aucs[i]) for i in range(len(aucs))]))
			continue
	else:
		if not args.output_aurocs:
			print("Messed up")
			if c == unique_covars[-1]:
				plt.show()
		continue
	plt.legend(loc = 'lower right')
	plt.axis('square')
	plt.plot([0, 1], [0, 1],'r--')
	plt.xlim([0, 1])
	plt.ylim([0, 1])
	plt.ylabel('True Positive Rate')
	plt.xlabel('False Positive Rate')
	if args.group_folders:
		group_str = ", averaged by session"
	elif args.group_var != "":
		group_str = ", averaged by %s" % disp_names[args.group_var]
	else:
		group_str = ", all files"
	if c == "":
		t = "%s classification%s (N = %d)" % (disp_names[test_variable],group_str,int(Y.shape[0]))
	else:
		t ="%s classification%s, %s = %s (N = %d)" % (disp_names[test_variable],group_str,disp_names[args.confound_var],c,int(Y.shape[0]))
	if args.title != "":
		t = "%s, (N = %d)" % (args.title,int(Y.shape[0]))
	plt.title("\n".join(wrap(t,50)))
	if args.output_filename != "":
		foo = os.path.join(working_dir,'figures',"%s%s.png" % (args.output_filename,"" if c == "" else "_" + c))
		if not os.path.isdir(os.path.dirname(foo)):
			os.makedirs(os.path.dirname(foo))
		plt.savefig(foo)
	if c == unique_covars[-1] and not args.noshow:
		plt.show()
	elif not args.noshow:
		plt.show(block=False)
		plt.figure()
	elif c != unique_covars[-1]:
		plt.figure()
