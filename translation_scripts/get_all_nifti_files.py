#!/usr/bin/python

import os,glob

working_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
data_dir = os.path.join(working_dir,'data_processing','RPDR_nifti_conversions')
output_file = os.path.join(working_dir,'txt','all_filenames_nifti.txt')

with open(output_file,'w') as fileobj:
	for filename in glob.glob(os.path.join(data_dir,'[ct]*','*','*.nii.gz')):
		fileobj.write('%s\n' % filename)
