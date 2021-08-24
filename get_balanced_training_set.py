#!/usr/bin/python

import os,sys,json,glob
from general_class_balancer import *
import numpy as np
import pandas as pd
from copy import deepcopy as copy
import nibabel as nb


# Used to get a training set with equal distributions of input covariates
# Can also be used to only have certain ranges of continuous covariates,
# or certain labels of discrete covariates.

working_dir = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))

json_label_folder = os.path.join(working_dir,'json','labels')

json_label_filepaths = glob.glob(os.path.join(json_label_folder,'*.json'))

pandas_output = os.path.join(working_dir,'pandas','cache','all_vars.pkl')
json_output   = os.path.join(working_dir,'json','cache','all_vars.json')

image_dim = [96,96,96]

def str_to_list(s,nospace=False):
	if s is None or s == "": return []
	if s[0] == "[" and  s[-1] == "]":
		s = s[1:-1]
		s = s.replace("'","").replace("_","").replace("-","")
		if nospace:s=s.replace(" ","")
		return s.split(",")
	else:
		return [s]

def iz_nan(k,inc_null_str=False):
	if k is None:
		return True
	if inc_null_str and isinstance(k,str):
		if k.lower() == "null" or k.lower() == "unknown":
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

def get_data_from_filenames(filename_list,test_variable,confounds=None, return_as_strs = False,unique_test_vals = None):
	from sklearn.preprocessing import MultiLabelBinarizer
	all_vars = pd.read_pickle(pandas_output)
	X_filenames_list = [os.path.join(working_dir,"%s_resized_%d.npy" % (_[1:],image_dim[0])) for _ in filename_list]
	if confounds is not None:
		confound_strs = [[None for _ in confounds] for __ in filename_list]
	Y_strs = [None for _ in filename_list]
	X = np.zeros((len(filename_list),image_dim[0],image_dim[1],image_dim[2]))
	for i in range(len(filename_list)):
		f = X_filenames_list[i]
		f_key = filename_list[i]
		assert(os.path.isfile(f))
		X_single = np.load(f)
		X[i,:,:,:] = X_single
		Y_strs[i] = str_to_list(all_vars.loc[f_key,test_variable],nospace=True)
		if confounds is not None:
			for j in range(len(confounds)):
				confound_strs[i][j] = all_vars.loc[f_key,confounds[j]]
	if return_as_strs:
		if confounds is not None:
			return X_filenames_list,Y_strs,confound_strs
		else:
			return X_filenames_list,Y_strs
	
	mlb = MultiLabelBinarizer()
	if unique_test_vals is not None:
		mlb.fit([unique_test_vals])
	else:
		Y_strs_all = []
		for s in all_vars.loc[:,test_variable]:
			if not iz_nan(s):
				Y_strs_all.append(str_to_list(s,nospace=True))
		["" if "" not in _ else _.remove("") for _ in Y_strs_all]
		mlb.fit(Y_strs_all)
		print(mlb.classes_)
	Y = mlb.transform(Y_strs)
	if confounds is not None:
		n_buckets = 10
		uniques = {}
		for c in confounds:
			uniques[c] = {}
			lis = list(all_vars.loc[:,c])
			if np.any([isinstance(_,str) for _ in lis]):
				uniques[c]["discrete"] = True
				u = set()
				for l in lis:
					if not iz_nan(l):
						u.add(l)
				u = sorted(list(u))
				uniques[c]["unique"] = u
				n_buckets = max(n_buckets,len(u))
			else:
				uniques[c]["discrete"] = False
				max_ = -np.inf
				min_ = np.inf
				nonnan_list = []
				for l in lis:
					if not iz_nan(l):
						max_ = max(max_,l)
						min_ = min(min_,l)
						nonnan_list.append(l)
				uniques[c]["max"] = max_
				uniques[c]["min"] = min_
				uniques[c]["nonnan_list"] = sorted(nonnan_list)
		for c in confounds:
			if not uniques[c]["discrete"]:
				skips = int(len(uniques[c]["nonnan_list"])/float(n_buckets)) + 1
				uniques[c]["nonnan_list"] = uniques[c]["nonnan_list"][::skips]
				assert(len(uniques[c]["nonnan_list"]) == n_buckets)
		confound_encode = np.zeros((len(filename_list),len(confounds),n_buckets + 1))
		for j in range(len(confounds)):
			c = confounds[j]
			if uniques[c]["discrete"]:
				
				c_uniques = uniques[c]["unique"]
				for i in range(len(filename_list)):
					if iz_nan(confound_strs[i][j]):
						confound_encode[i,j,-1] = 1
					else:
						confound_encode[i,j,c_uniques.index(confound_strs[i][j])] = 1
			else:
				max_ = uniques[c]["max"]
				min_ = uniques[c]["min"]
				for i in range(len(filename_list)):
					if iz_nan(confound_strs[i][j]):
						confound_encode[i,j,-1] = 1
					else:
						for kk in range(len(uniques[c]["nonnan_list"])):
							if uniques[c]["nonnan_list"][kk] < confound_strs[i][j]:
								confound_encode[i,j,kk] =1
			
		return X,Y,confound_encode
	else:
		return X,Y

def get_MRI_variable(pandas_output,json_output):
	if os.path.isfile(pandas_output) and os.path.isfile(json_output):
		covars = json.load(open(json_output,'r'))
		covars_df = pd.read_pickle(pandas_output)
	else:
		columns = [os.path.splitext(os.path.basename(filepath))[0] for filepath in json_label_filepaths]
		covars = {}
		
		emptydict = {}
		for c in columns:
			emptydict[c] = None
		
		for filepath in json_label_filepaths:
			json_dict = json.load(open(filepath,'r'))
			key = os.path.splitext(os.path.basename(filepath))[0]
			if json_dict["discrete"]:
				c = 0
				for value in json_dict[key]:
					for dataset_name in json_dict[key][value]:
						if dataset_name not in covars:
							covars[dataset_name] = copy(emptydict)
						covars[dataset_name][key] = value
						c += 1
			else:
				c = 0
				for v in json_dict[key]:
					value,dataset_name = v
					if dataset_name not in covars:
						covars[dataset_name] = copy(emptydict)
					covars[dataset_name][key] = value
					c += 1
		covars_df = pd.DataFrame.from_dict(covars,orient='index',columns = columns)
		covars_df.to_pickle(pandas_output)
		json.dump(covars,open(json_output,'w'),indent=4)
	return covars_df,covars

from copy import deepcopy as copy

def recompute_selection_ratios(selection_ratios,selection_limits,N):
	new_selection_ratios = copy(selection_ratios)
	assert(np.any(np.isinf(selection_limits)))
	print("new_selection_ratios: %s" % str(new_selection_ratios))
	variable = [True for i in range(len(selection_ratios))]
	for i in range(len(selection_ratios)):
		if selection_ratios[i] * N > selection_limits[i]:
			new_selection_ratios[i] = selection_limits[i] / N
			variable[i] = False
		else:
			new_selection_ratios[i] = selection_ratios[i]
	vsum = 0.0
	nvsum = 0.0
	for i in range(len(selection_ratios)):
		if variable[i]: vsum += new_selection_ratios[i]
		else: nvsum += new_selection_ratios[i]
	assert(nvsum < 1)
	for i in range(len(selection_ratios)):
		if variable[i]: new_selection_ratios[i] = (new_selection_ratios[i] / vsum) * (1 - nvsum)
	return new_selection_ratios

def get_balanced_filename_list(test_variable,confounds_array,selection_ratios = [0.66,0.16,0.16],
		selection_limits = [np.Inf,np.Inf,np.Inf],
		pandas_output = pandas_output,json_output = json_output,value_ranges = [],
		output_covars_savepath = None,output_selection_savepath = None,
		test_value_ranges=None,get_all_test_set=False,total_size_limit=None):
	if len(value_ranges) == 0:
		value_ranges = [None for _ in confounds_array]
	assert(len(value_ranges) == len(confounds_array))
	
	covars_df,covars = get_MRI_variable(pandas_output,json_output)
	print("len(covars): %d" % len(covars))
	value_selection = np.ones((len(covars_df),),dtype=bool)
	confounds_array.append(test_variable)
	value_ranges.append(test_value_ranges)
	print("confounds_array: %s" % str(confounds_array))
	print("value_ranges: %s" % str(value_ranges))
	for i in range(len(confounds_array)):
		temp_value_selection = np.zeros((len(covars_df),),dtype=bool)
		c = covars_df[confounds_array[i]]
		value_range = value_ranges[i]
		if value_range is None:
			continue
		if isinstance(value_range,tuple):
			for j in range(len(c)):
				if c[j] is None:
					continue
				if c[j] >= value_range[0] and c[j] <= value_range[1]:
					temp_value_selection[j] = True
		elif callable(value_range):
			for j in range(len(c)):
				if c[j] is None:
					continue
				if value_range(c[j]):
					temp_value_selection[j] = True
		else:
			for j in range(len(c)):
				if c[j] is None:
					continue
				if c[j] in value_range:
					temp_value_selection[j] = True	
		value_selection = np.logical_and(value_selection,temp_value_selection)
	del confounds_array[-1]
	del value_ranges[-1]
	print("value_selection.sum(): %s"%str(value_selection.sum()))
	print("value_selection.shape: %s"%str(value_selection.shape))
	covars_df = covars_df[value_selection]
	covars_df = covars_df.sample(frac=1)
	test_vars = covars_df[test_variable].to_numpy(dtype=np.dtype(object))
	if output_selection_savepath is not None and os.path.isfile(output_selection_savepath):
		selection = np.load(output_selection_savepath)
	else:
		if len(confounds_array) == 0:
			print(test_value_ranges)
			#selection = class_balance(test_vars,[],unique_classes=test_value_ranges,plim=0.1)
			selection = np.ones(test_vars.shape)
		else:
			selection = class_balance(test_vars,covars_df[confounds_array].to_numpy(dtype=np.dtype(object)).T,unique_classes=test_value_ranges,plim=0.1)
		if output_covars_savepath is not None:
			if not os.path.isdir(os.path.dirname(output_covars_savepath)):
				os.makedirs(os.path.dirname(output_covars_savepath))
			covars_df[selection].to_pickle(output_covars_savepath)
		selection_ratios = recompute_selection_ratios(selection_ratios,selection_limits,np.sum(selection))
		if total_size_limit is not None:
			select_sum = selection.sum()
			rr = list(range(len(selection)))
			for i in rr:
				if select_sum <= total_size_limit:
					break
				if selection[i]:
					selection[i] = 0
					select_sum -= 1
		selection = separate_set(selection,selection_ratios,covars_df["PatientID"].to_numpy(dtype=np.dtype(object)).T)
		if output_selection_savepath is not None:
			np.save(output_selection_savepath,selection)
	all_files = (covars_df.index.values)
	if get_all_test_set:
		selection[selection == 0] = 2
	X_files = [all_files[selection == i] for i in range(1,len(selection_ratios) + 1)]
	Y_files = [test_vars[selection == i] for i in range(1,len(selection_ratios) + 1)]
	print(np.sum([len(x) for x in X_files]))
	for i in range(len(X_files)):
		rr = list(range(len(X_files[i])))
		random.shuffle(rr)
		X_files[i] = X_files[i][rr]
		Y_files[i] = Y_files[i][rr]
	return X_files,Y_files

def get_balanced_mri_data(test_variable = "Test_Control",confounds_array = ["Ages","SexDSC","BodyPartExamined"],
	selection_ratios = [0.66,0.16,0.16],
	pandas_output = pandas_output,json_output = json_output,image_dim = (32,32,32),
	value_ranges = []):
	y_list,X_filebase_list = get_balanced_filename_list(test_variable,
							confounds_array,
							selection_ratios,
							pandas_output = pandas_output,
							json_output = json_output,
							value_ranges=value_ranges) 
	X_filenames_list = [[ os.path.join(working_dir,"%_resized_%d.npy" % (f,image_dim[0])) for f in x] for x in X_filenames_list]
	X = [np.zeros(tuple([len(x)] + list(image_dim))) for x in X_filenames_list]
	for i in range(len(X)):
		for j in range(X[i].shape[0]):
			X[i][j,:,:,:] = nb.load(X_filenames_list[i][j]).get_data()
	return X,y_list

#confounds_array = ["Ages","SexDSC","BodyPartExamined"]
#confounds_array = ["SexDSC","BodyPartExamined","DeviceSerialNumber","Ages"]
#confounds_array = ["SexDSC","BodyPartExamined"]

if __name__ == '__main__':
	get_balanced_filename_list("Test_Control",confounds_array = ["SexDSC","Ages","DataSortSSIMCut","DataSort"],selection_ratios = [0.66,0.16,0.16],
		pandas_output = pandas_output,json_output = json_output,value_ranges = [["MALE","FEMALE"],(50,500),["1"],["0","1","2","3","4"]])




