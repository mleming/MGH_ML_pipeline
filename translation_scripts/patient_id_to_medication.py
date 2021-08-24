#!/usr/bin/python

import os,sys,json,csv
from datetime import datetime

working_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
csv_file =  os.path.join(working_dir,'csv','edw_sql_results','all_brain_mri_test_group_medications','all_brain_mri_test_group_medications.csv')
output_json_filename = os.path.join(working_dir,'json','translation_files','patient_id_to_medication.json')
output_json_filename_accession = os.path.join(working_dir,'json','translation_files','accession_nbr_to_medication.json')

patient_id_to_medication = {}
accession_nbr_to_medication = {}

firstrow = True

accession_nbr_key = 'AccessionNBR'
patient_id_key = 'PatientID'
date_key = 'OrderingDTS'
medication_key = 'MedicationDSC'

def med_string(med_str):
	for s in ["RIVASTIGMINE","MEMANTINE","GALANTAMINE","DONEPEZIL"]:
		if s.lower() in med_str.lower():
			return s
	print(med_str)
	assert(False)

i=0
for row in csv.reader(open(csv_file,'r'),delimiter=",",quotechar='"'):
	if firstrow:
		header = row
		patient_id_col = header.index(patient_id_key)
		date_id_col = header.index(date_key)
		medication_id_col = header.index(medication_key)
		accession_nbr_id_col = header.index(accession_nbr_key)
		firstrow = False
	else:
		#print(row[patient_id_col])
		#print(row[date_id_col])
		#print(row[medication_id_col])
		accession_nbr = row[accession_nbr_id_col]
		patient_id = row[patient_id_col]
		medication = med_string(row[medication_id_col])
		date = row[date_id_col]
		if patient_id not in patient_id_to_medication:
			patient_id_to_medication[patient_id] = []
		if accession_nbr not in accession_nbr_to_medication:
			accession_nbr_to_medication[accession_nbr] = []
		if date == "NULL":
			continue
		patient_id_to_medication[patient_id].append((medication,date))
		accession_nbr_to_medication[accession_nbr].append((medication,date))

#2013-12-03 13:34:00.0000000

for key in patient_id_to_medication:
	patient_id_to_medication[key].sort(key=lambda date: datetime.strptime(date[1].split(".")[0],"%Y-%m-%d %H:%M:%S"))

for key in accession_nbr_to_medication:
	accession_nbr_to_medication[key].sort(key=lambda date: datetime.strptime(date[1].split(".")[0],"%Y-%m-%d %H:%M:%S"))

#print(header)
"""
patientid:
	[[Medication,date],[Medication,date]...]
"""

json.dump(patient_id_to_medication,open(output_json_filename,'w'),indent=4)
json.dump(accession_nbr_to_medication,open(output_json_filename_accession,'w'),indent=4)
