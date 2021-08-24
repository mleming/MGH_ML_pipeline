#!/usr/bin/python

import os,sys,glob,json,random,shutil
from time import time
import numpy as np
import nibabel as nb
from scipy import ndimage, misc

# Matthew Leming, MGH Center for Systems Biology, 2021
# Reorients .nii.gz files using fslreorient2std and then converts each to a 
# 96x96x96 .npy file, which is then accessed for use in DL tasks later.
# Also eliminates any files that are not MRIs of the brain or head.

working_dir = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))

data_dir = os.path.join(working_dir,'data_processing','RPDR_nifti_conversions')

folders = glob.glob(os.path.join(data_dir,'[ct]*','*'))

random.shuffle(folders)

output_image_dim = (96,96,96)

apply_fsl_reorient = True

eycount = {}
c = 0
for folder in folders:
	print("%d/%d (%s)" % (c,len(folders),folder))
	c += 1
	nifti_filepaths = glob.glob(os.path.join(folder,'*.nii.gz'))
	nifti_filepaths = list(filter(lambda k: not k.endswith('_reorient.nii.gz'),nifti_filepaths))
	#if len(nifti_filepaths) > 25:
	#	continue
	reorient_filepath = os.path.join(folder,'processing_status.txt')
	for nifti_filepath in nifti_filepaths:
		nifti_filepath = os.path.realpath(nifti_filepath)
		t = time()
		nifti_filename = os.path.basename(nifti_filepath)
		basename = os.path.splitext(os.path.splitext(nifti_filename)[0])[0]
		npy_output_filepath = os.path.join(folder,"%s_resized_%d.npy" % (basename,output_image_dim[0]))
		npy_output_filepath32 = os.path.join(folder,"%s_resized_%d.npy" % (basename,32))
		if os.path.isfile(npy_output_filepath32):
			os.remove(npy_output_filepath32)
		json_filename = '%s_patient.json' % basename
		json_filepath = os.path.join(folder,json_filename)
		if os.path.getsize(nifti_filepath) < 1000000:
			if os.path.isfile(npy_output_filepath):
				os.remove(npy_output_filepath)
			continue
		if os.path.isfile(json_filepath):
			try:
				with open(json_filepath,'r') as fileobj:
					patient_data = json.load(fileobj)
			except KeyboardInterrupt:
				print('Interrupted')
				try:
					sys.exit(0)
				except SystemExit:
					os._exit(0)	
			except:
				continue
			remove_and_continue = False
			if "BodyPartExamined" in patient_data:
				bodypart = patient_data["BodyPartExamined"]
				if not np.any([foo in bodypart.lower() for foo in ["head","brain"]]):
					remove_and_continue = True
			if "ProcedureNM" in patient_data:
				procedure = patient_data["ProcedureNM"]
				if not np.any([foo in procedure.lower() for foo in ["head","brain"]]):
					remove_and_continue = True
			if remove_and_continue:
				if os.path.isfile(npy_output_filepath):
					os.remove(npy_output_filepath)
				continue
		else:
			continue
		nifti_reorient_filepath = os.path.join(folder,"%s_reorient.nii.gz" % basename)
		applied_reorient = False
		if apply_fsl_reorient:
			skip=False
			if os.path.isfile(reorient_filepath):
				with open(reorient_filepath,'r') as fileobj:
					for row in fileobj.readlines():
						if "%s reoriented" % nifti_filename in row:
							skip=True
			if not skip:
				with open(reorient_filepath,'a') as fileobj:
					fileobj.writelines(["%s reoriented\n"%nifti_filename])
				try:
					print("Reorienting %s" % nifti_filepath)
					os.system("fslreorient2std '%s' '%s'" % (nifti_filepath,nifti_reorient_filepath))
					if os.path.isfile(nifti_reorient_filepath):
						print("Successfully reoriented %s" % nifti_filepath)
						shutil.move(nifti_reorient_filepath,nifti_filepath)
						applied_reorient = True
					else:
						print("Failed to orient %s" % nifti_filepath)
				except KeyboardInterrupt:
					print('Interrupted')
					try:
						sys.exit(0)
					except SystemExit:
						os._exit(0)
				except:
					print("Failed to reorient %s" % nifti_filepath)
					assert(nifti_filepath != nifti_reorient_filepath)
		if os.path.isfile(npy_output_filepath) and not applied_reorient:
			continue
		try:
			nifti_file = nb.load(nifti_filepath)
			nifti_data = nifti_file.get_fdata()
			if len(nifti_data.shape) != len(output_image_dim):
				continue
			#nifti_data = ndimage.affine_transform(nifti_data,np.linalg.inv(nifti_file.header.get_qform()[:3,:3]))
			nifti_data -= nifti_data.min()
			m = nifti_data.max()
			if m == 0:
				continue
			nifti_data = nifti_data / m
			nifti_data = nifti_data.astype(np.float32)
			zp = [output_image_dim[i]/nifti_data.shape[i] for i in range(len(output_image_dim))]
			nifti_data_zoomed = ndimage.zoom(nifti_data,zp)
			np.save(npy_output_filepath,nifti_data_zoomed)
		except KeyboardInterrupt:
			print('Interrupted')
			try:
				sys.exit(0)
			except SystemExit:
				os._exit(0)
		except:
			print("Error with %s. Deleting..." % nifti_filepath)
		if nifti_filepath == nifti_reorient_filepath:
			os.remove(nifti_reorient_filepath)
for key in keycount:
	print("%s %d" % (key,keycount[key]))
