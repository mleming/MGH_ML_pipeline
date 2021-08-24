#!/usr/bin/python

import sys,os,glob,shutil

# Matthew Leming, MGH Center for Systems Biology, 2021
# Converts DICOM files in your data directory to NIFTI files in your
# data_processing directory. Must have dcm2niix in your path.
# Files in this case are set up as ${WORKING_DIR}/{test|control}/data/RPDR_query_data
# Converts them to .nii.gz files in ${WORKING_DIR}/{test|control}/data_processing/RPDR_nifti_conversions
# Cleans up intermediary files after the fact

working_dir = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
print(working_dir)
# Files are separated into test and control folders
for tc in ["test","control"]:
	data_dir = os.path.realpath(os.path.join(working_dir,'data','RPDR_query_data',tc))
	output_dir = os.path.realpath(os.path.join(working_dir,'data_processing','RPDR_nifti_conversions',tc))
	print(output_dir)
	if not os.path.isdir(output_dir):
		print(output_dir)
		os.makedirs(output_dir)
	data_dir_glob = glob.glob(os.path.join(data_dir,'*'))
	for i in range(len(data_dir_glob)):
		sys.stdout.write("Percent %s converted: %.3f\r" % (tc,float(i) / len(data_dir_glob) * 100))
		sys.stdout.flush()
		d = data_dir_glob[i]
		folder=os.path.basename(d)
		print(folder)
		if os.path.isdir(os.path.join(output_dir,folder)):
			#print("is folder")
			continue
		shutil.copytree(d,os.path.join(output_dir,folder))
		#print("copies")
		os.chdir(os.path.join(output_dir,folder))
		for j in glob.glob('*' + os.sep):
			os.system('dcm2niix %s >/dev/null 2>&1' % j)
			for k in glob.glob(os.path.join(j,'*.nii')):
				shutil.move(k,os.path.join(output_dir,folder,os.path.basename(k)))
				os.system("gzip '%s' >/dev/null 2>&1" % os.path.join(output_dir,folder,os.path.basename(k)))
			for k in glob.glob(os.path.join(j,'*.json')):
				shutil.move(k,os.path.join(output_dir,folder,os.path.basename(k)))
			print("File: %s"%j)
			shutil.rmtree(j)

