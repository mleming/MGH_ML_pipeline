#!/usr/bin/python

import os,sys,json,csv,glob
from datetime import datetime

#/home/mleming/Desktop/MGH_ML_pipeline/csv/edw_sql_results/all_brain_mri_test_group_diagnosiss/all_brain_mri_test_group_diagnosiss.csv
#/home/mleming/Desktop/MGH_ML_pipeline/csv/edw_sql_results/all_brain_mri_test_group_diagnosiss/patient_id_to_diagnosis.json

working_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
csv_file = os.path.join(working_dir,'csv','edw_sql_results','ICD9_to_ICD10','ICD9_to_ICD10.csv')
output_json_filename = os.path.join(working_dir,'json','translation_files','ICD9_to_ICD10.json')

d = {}
mrn_test_empty = {}
firstrow = True

key_one = 'ICD10'
key_two = 'ICD9'
key_three = 'ICD10DSC'

for row in csv.reader(open(csv_file,'r'),delimiter=",",quotechar='"'):
	if firstrow:
		header = row
		key_two_id_col = header.index(key_two)
		key_one_id_col = header.index(key_one)
		key_three_id_col = header.index(key_three)
		firstrow = False
	else:
		#print(row[patient_id_col])
		#print(row[date_id_col])
		#print(row[key_two_id_col])
		icd10 = row[key_one_id_col]
		icd9 = row[key_two_id_col]
		icd10_dsc = row[key_three_id_col]
		if icd9 not in d:
			d[icd9] = []
		d[icd9].append(icd10)

#2013-12-03 13:34:00.0000000
#print(mrn_test_empty)

#print(header)
"""
patientid:
	[[Medication,date],[Medication,date]...]
"""

json.dump(d,open(output_json_filename,'w'),indent=4)
