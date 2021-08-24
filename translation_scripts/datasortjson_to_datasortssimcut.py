#!/usr/bin/python

import json,os,sys,csv
from get_cutoff_thresh import get_cutoff_thresh

working_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))

label_name = "DataSort"
ssim_label_name = "DataSortSSIMCut"

json_label_partition_output = os.path.join(working_dir,'json','labels','%s.json' %label_name)

ssim_label_partition_output = os.path.join(working_dir,'json','labels','%s.json' %ssim_label_name)
output_folder = os.path.join(working_dir,'data_processing','mri_clusters2')
avg_ssims_output = os.path.join(output_folder,'partitions_avg_ssims.csv')

with open(json_label_partition_output,'r') as fileobj:
	json_label = json.load(fileobj)

npy_fnames_file = os.path.join(working_dir,'txt','all_filenames.txt')

npy_fnames = []
all_avg_partition_ssims = []
with open(npy_fnames_file,'r') as fileobj:
	reader = csv.reader(fileobj)
	for row in reader:
		npy_fnames.append(row[0])

with open(avg_ssims_output,'r') as fileobj:
	reader = csv.reader(fileobj)
	for row in reader:
		all_avg_partition_ssims.append(float(row[0]))

npy_fname_stubs = [ os.sep + os.sep.join(os.path.normpath(f).split(os.sep)[5:])[:-len('_resized_96.npy')] for f in npy_fnames]


cutoff_thresh = get_cutoff_thresh(all_avg_partition_ssims)

ssim_label = {}
ssim_label["discrete"] = True
ssim_label[ssim_label_name] = {}
ssim_label[ssim_label_name]["0"] = []
ssim_label[ssim_label_name]["1"] = []
ssim_label["cutoff_thresh"] = cutoff_thresh
for i in range(len(all_avg_partition_ssims)):
	a = all_avg_partition_ssims[i]
	ssim_label[ssim_label_name]["0" if a < cutoff_thresh else "1"].append(npy_fname_stubs[i])


with open(ssim_label_partition_output,'w') as fileobj:
	json.dump(ssim_label,fileobj,indent=4)
