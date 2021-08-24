#!/usr/bin/python

import os,sys,json,csv,glob
from datetime import datetime

#/home/mleming/Desktop/MGH_ML_pipeline/csv/edw_sql_results/all_brain_mri_test_group_medications/all_brain_mri_test_group_medications.csv
#/home/mleming/Desktop/MGH_ML_pipeline/csv/edw_sql_results/all_brain_mri_test_group_medications/patient_id_to_medication.json

working_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
csv_dir = os.path.join(working_dir,'csv','RPDR_supp_files')
csv_files = [os.path.join(csv_dir,'control','SD587_20201102_131239_Med.txt'),os.path.join(csv_dir,'test','SD587_20201102_100230_Med.txt')]
output_json_filename = os.path.join(working_dir,'json','translation_files','mrn_to_medication.json')

mrn_to_medication = {}
mrn_test_empty = {}
firstrow = True

mrn_key = 'EMPI'
date_key = 'Medication_Date'
medication_key = 'Medication'

med_dict = {"RIVASTIGMINE":["EXELON","RIVASTIGMINE"],
	"MEMANTINE":["MEMANTINE","NAMENDA","NAMZARIC"],
	"GALANTAMINE":["RAZADYNE","GALANTAMINE"],
	"DONEPEZIL":["DONEPEZIL","NAMZARIC","ARICEPT"]}

def med_string(med_str):
	for med in med_dict:
		for s in med_dict[med]:
			if s.lower() in med_str.lower():
				return med
	return ""
	print(med_str)
	assert(False)

for i in range(len(csv_files)):
	csv_file = csv_files[i]
	for row in csv.reader(open(csv_file,'r'),delimiter="|",quotechar='"'):
		if firstrow:
			header = row
			date_id_col = header.index(date_key)
			medication_id_col = header.index(medication_key)
			mrn_id_col = header.index(mrn_key)
			firstrow = False
		else:
			#print(row[patient_id_col])
			#print(row[date_id_col])
			#print(row[medication_id_col])
			mrn = row[mrn_id_col]
			medication = med_string(row[medication_id_col])
			if medication != "" and i == 0:
				print(mrn)
			if medication == "":
				if i == 1 and mrn not in mrn_test_empty and mrn not in mrn_to_medication:
					mrn_test_empty[mrn] = True
				continue
			date = row[date_id_col]
			if mrn not in mrn_to_medication:
				if i == 1 and mrn in mrn_test_empty:
					del mrn_test_empty[mrn]
				mrn_to_medication[mrn] = []
			if date == "NULL":
				continue
			mrn_to_medication[mrn].append((medication,date))

#2013-12-03 13:34:00.0000000
print(len(mrn_to_medication))
print(len(mrn_test_empty))
print(mrn_test_empty)
for key in mrn_to_medication:
	mrn_to_medication[key].sort(key=lambda date: datetime.strptime(date[1].split(".")[0],"%m/%d/%Y"))

#print(header)
"""
patientid:
	[[Medication,date],[Medication,date]...]
"""

json.dump(mrn_to_medication,open(output_json_filename,'w'),indent=4)
