#!/usr/bin/python

import os,sys,json,csv,glob
from datetime import datetime

working_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))

csv_dir = os.path.join(working_dir,'csv','RPDR_supp_files')
csv_files = [os.path.join(csv_dir,'control','SD587_20201102_131239_Dia.txt'),os.path.join(csv_dir,'test','SD587_20201102_100230_Dia.txt')]
output_json_filename = os.path.join(working_dir,'json','translation_files','mrn_to_icd10.json')
icd9_to_icd10_filename = os.path.join(working_dir,'json','translation_files','ICD9_to_ICD10.json')
icd9_to_icd10 = json.load(open(icd9_to_icd10_filename,'r'))

mrn_to_icd10 = json.load(open(os.path.join(working_dir,'json','translation_files','mrn_to_icd10.json'),'r'))
mrn_to_medication = json.load(open(os.path.join(working_dir,'json','translation_files','mrn_to_medication.json'),'r'))

mrn_to_diagnosis = {}

mrn_key = 'EMPI'
date_key = 'Date'
diagnosis_key = 'Code'
icd_type_key = 'Code_Type'



def med_string(med_str):
	for med in med_dict:
		for s in med_dict[med]:
			if s.lower() in med_str.lower():
				return med
	return ""
	print(med_str)
	assert(False)
header = []
unknown_icds = set()
for i in range(len(csv_files)):
	firstrow = True
	csv_file = csv_files[i]
	for row in csv.reader(open(csv_file,'r'),delimiter="|",quotechar='"'):
		if firstrow:
			header = row
			firstrow = False
		else:
			#print(row[patient_id_col])
			#print(row[date_id_col])
			#print(row[diagnosis_id_col])
			mrn = row[header.index(mrn_key)]
			icd_type = row[header.index(icd_type_key)]
			icd_code = row[header.index(diagnosis_key)]
			if icd_type == "ICD9":
				if icd_code not in icd9_to_icd10:
					icd_code = icd_code.replace(".","")
					if icd_code not in icd9_to_icd10:
						unknown_icds.add(icd_code)
						continue
				icd_code = icd9_to_icd10[icd_code][0]
			date = row[header.index(date_key)]
			if mrn not in mrn_to_diagnosis:
			#	if i == 1 and mrn in mrn_test_empty:
			#		del mrn_test_empty[mrn]
				mrn_to_diagnosis[mrn] = []
			if date == "NULL":
				continue
			mrn_to_diagnosis[mrn].append((icd_code,date))

#2013-12-03 13:34:00.0000000
#print(mrn_test_empty)
for key in mrn_to_diagnosis:
	mrn_to_diagnosis[key].sort(key=lambda date: datetime.strptime(date[1].split(".")[0],"%m/%d/%Y"))

#print(header)
"""
patientid:
	[[Medication,date],[Medication,date]...]
"""

json.dump(mrn_to_diagnosis,open(output_json_filename,'w'),indent=4)
