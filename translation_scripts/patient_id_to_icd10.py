#!/usr/bin/python

import os,sys,json,csv
from datetime import datetime

working_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
csv_dir = os.path.join(working_dir,'csv','edw_sql_results','icd10_all_codes')
csv_file =  os.path.join(csv_dir,'icd10_all_codes.csv')
output_json_filename = os.path.join(working_dir,'json','translation_files','patient_id_to_icd10.json')

patient_id_to_icd10 = {}
icd10_code_to_medication = {}

firstrow = True

icd10_code_key = 'ICD10CD'
patient_id_key = 'PatientID'
date_key = 'CalendarDTS'

i=0
for row in csv.reader(open(csv_file,'r'),delimiter=",",quotechar='"'):
	if firstrow:
		header = row
		patient_id_col = header.index(patient_id_key)
		icd10_code_id_col = header.index(icd10_code_key)
		date_id_col = header.index(date_key)
		firstrow = False
	else:
		icd10_code = row[icd10_code_id_col]
		patient_id = row[patient_id_col]
		date = row[date_id_col]
		
		if patient_id not in patient_id_to_icd10:
			patient_id_to_icd10[patient_id] = []
		app = True
		for i in range(len(patient_id_to_icd10[patient_id])):
			icd,d = patient_id_to_icd10[patient_id][i]
			if icd == icd10_code:
				if d > date:
					patient_id_to_icd10[patient_id][i] = (icd10_code,date)
				app = False
		if app:
			patient_id_to_icd10[patient_id].append((icd10_code,date))

json.dump(patient_id_to_icd10,open(output_json_filename,'w'),indent=4)
