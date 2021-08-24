#!/usr/bin/python

# Compiles all merged .json files in each study's folder

import os,sys,glob,json,random,csv,datetime
from time import time
import numpy as np
import nibabel as nb
from scipy import ndimage, misc
from dateutil import relativedelta,parser


# Matthew Leming, MGH Center for Systems Biology, 2021
# Uses the translation files and .json files in each folder to compile labels
# in aggregate and place them in the ${WORKING_DIR}/json/labels folder.
# Also computes variables such as age and Alzheimer's stage based on medication
# history and ICD codes.

working_dir = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
data_dir = os.path.join(working_dir,'data_processing','RPDR_nifti_conversions')
translation_dir = os.path.join(working_dir,'json','translation_files')
patient_id_to_med        = json.load(open(os.path.join(translation_dir,'patient_id_to_medication.json'),'r'))
patient_id_to_icd10_code = json.load(open(os.path.join(translation_dir,'patient_id_to_icd10.json'),'r'))
mrn_to_icd10_code        = json.load(open(os.path.join(translation_dir,'mrn_to_icd10.json'),'r'))
mrn_to_medication        = json.load(open(os.path.join(translation_dir,'mrn_to_medication.json'),'r'))
excluded_keys_file = os.path.join(working_dir,'txt','excluded_labels.txt')
excluded_keys = []
if os.path.isfile(excluded_keys_file):
	with open(excluded_keys_file,'r') as fileobj:
		csv_reader = csv.reader(fileobj)
		for row in csv_reader:
			excluded_keys.append(row[0])

cachefile = os.path.join(working_dir,'json','labelscache.json')

age_start_key = 'BirthDTS'
age_end_key = 'ExamEndDTS'

outfolder = os.path.join(working_dir,'json','labels')
if not os.path.isdir(outfolder):
	os.makedirs(outfolder)

def is_float(N):
	try:
		float(N)
		return True
	except:
		return False

def list_to_str(val):
	val = str(sorted(val))
	val = val.upper()
	val = val.replace(" ","_")
	val = val.replace("-","_")
	return val

def parsedate(d,date_format="%Y-%m-%d %H:%M:%S"):
	return datetime.datetime.strptime(d.split(".")[0],date_format)

def get_meds_before_date(patientid,date,future=False):
	if patientid not in patient_id_to_med:
		return ['None']
	else:
		meds = []
		for med,meddate in patient_id_to_med[patientid]:
			meddate = parsedate(meddate)# datetime.datetime.strptime(meddate.split(".")[0],"%Y-%m-%d %H:%M:%S")
			if (meddate < date and not future) or ( meddate > date and future):
				if med not in meds:
					meds.append(med)
			else:
				break
		return sorted(meds)

def get_alz_stage_at_date(patientid,date,patient_id_to_med = None,
		patient_id_to_icd10_code = None,test_control = None,
		date_format="%m/%d/%Y",consider_meds = True):
	#datetime.timedelta(years=5)
	meds = []
	med_timing = datetime.timedelta(days=365)
	icd_trauma_timing = datetime.timedelta(days=365)
	icd_disease_timing = datetime.timedelta(days=5*365)
	relevant_icds = []
	if patientid in patient_id_to_med:
		for med,meddate in patient_id_to_med[patientid]:
			meddate = parsedate(meddate,date_format=date_format)
			if (date + med_timing) > meddate and med not in meds:
				meds.append(med)
	if patientid in patient_id_to_icd10_code:
		for icd,icddate in patient_id_to_icd10_code[patientid]:
			icddate = parsedate(icddate,date_format=date_format)
			if icd == "":
				continue
			# G30 - Alzheimer's
			elif icd.startswith("G30"):
				if (date + icd_disease_timing) > icddate:
					relevant_icds.append(icd)
			# G31 - MCI
			elif icd.startswith("G31"):
				if (date + icd_disease_timing) > icddate:
					relevant_icds.append(icd)
			# C79 - Malignant neoplasms
			elif icd.startswith("C79.31") or icd.startswith("C71.9") or icd.startswith("I63.9") or icd.startswith("D49.6") or icd.startswith("D32.0"):
				if (date + icd_disease_timing) > icddate:
					relevant_icds.append(icd)
			# F01-F99 (Mental, Behavioral and Neurodevelopmental
			# disorders)
			elif icd.startswith("F"):
				relevant_icds.append(icd)
			# G00-G47 (Inflammatory diseases of the central nervous
			# system, Systemic atrophies primarily affecting the 
			# central nervous system, Extrapyramidal and movement 
			# disorders, Other degenerative diseases of the nervous
			# system, Demyelinating diseases of the central nervous 
			# system,  Episodic and paroxysmal disorders)
			elif np.any([icd.startswith("G.2%d" % k) for k in range(48)]):
				relevant_icds.append(icd)
			# G80-G99 (Cerebral palsy and other paralytic
			# syndromes, Other disorders of the nervous system)
			elif icd.startswith("G8") or icd.startswith("G9"):
				relevant_icds.append(icd)
			# I60-I69 (Cerebrovascular diseases)
			elif icd.startswith("I6"):
				if (date + icd_disease_timing) > icddate:
					relevant_icds.append(icd)
			# Q00-Q07 (Congenital malformations of the nervous system)
			elif icd.startswith("Q0") and not (icd.startswith("Q08") or icd.startswith("Q09")):
				relevant_icds.append(icd)
			# S00-S09 (Injuries to the head)
			elif icd.startswith("S0"):
				if (date + icd_trauma_timing) > icddate:
					relevant_icds.append(icd)
	if np.any([k.startswith("S0") or k.startswith("C79.31") or k.startswith("C71.9") or k.startswith("D32.0") or k.startswith("D49.6") or k.startswith("I63.9") for k in relevant_icds]):
		stage = "EXCLUDE"
	elif len(meds) == 0 and len(relevant_icds) == 0:
		stage = "CONTROL"
	elif np.any([k.startswith("G30") for k in relevant_icds]) or ("MEMANTINE" in meds and consider_meds):
		stage = "AD"
	elif np.any([k.startswith("G31") for k in relevant_icds]) or (len(meds) > 0 and consider_meds):
		stage = "MCI"
	elif len(meds) == 0 and not consider_meds:
		stage = "CONTROL"
#	elif np.any([(k.startswith("S") or k.startswith("Q") or k.startswith("G8") or k.startswith("G9")) for k in relevant_icds]):
#		stage = "EXCLUDE"
	else:
		stage = "EXCLUDE"
	if (stage == "CONTROL" and test_control == "test"):
		stage = "EXCLUDE"
	return stage

if not os.path.isfile(cachefile):
	folders = glob.glob(os.path.join(data_dir,'[ct]*','*'))
	
	label_json = {}
	c = 0
	for folder in folders:
		print("%d/%d" % (c,len(folders)))
		c += 1
		tc = os.path.basename(os.path.dirname(folder))
		print(folder)
		nifti_filepaths = glob.glob(os.path.join(folder,'*.nii.gz'))
		basefolder = folder[len(working_dir):]
		for nifti_filepath in nifti_filepaths:
			nifti_filepath = os.path.realpath(nifti_filepath)
			nifti_filename = os.path.basename(nifti_filepath)
			basename = os.path.splitext(os.path.splitext(nifti_filename)[0])[0]
			np_filepath = os.path.join(folder,"%s_resized_%d.npy" % (basename,96))
			json_filename = '%s_patient.json' % basename
			json_filepath = os.path.join(folder,json_filename)
			if os.path.isfile(json_filepath) and os.path.isfile(np_filepath):
				try:
					json_file = json.load(open(json_filepath,'r'))
				except KeyboardInterrupt:
					print('Interrupted')
					try:
						sys.exit(0)
					except SystemExit:
						os._exit(0)
				except:
					print("%s aint no real json"%json_filepath)
					continue
				for key in json_file:
					val = json_file[key]
					if key not in label_json:
						label_json[key] = {}
					if isinstance(val,list): val = list_to_str(val)
					val = str(val)
					val = val.upper()
					val = val.replace(" ","_")
					val = val.replace("-","_")
					lt = 0
					while len(val) != lt:
						lt = len(val)
						val = val.replace("__","_")
					if val not in label_json[key]:
						label_json[key][val] = []
					label_json[key][val].append(os.path.join(basefolder,basename))
				if "Test_Control" not in label_json:
					label_json["Test_Control"] = {}
				if tc not in label_json["Test_Control"]:
					label_json["Test_Control"][tc] = []
				label_json["Test_Control"][tc].append(os.path.join(basefolder,basename))
		if c > 1000 and False:
			break
	json.dump(label_json,open(cachefile,'w'),indent=4)
else:
	print("Loading %s" % cachefile)
	label_json = json.load(open(cachefile,'r'))
	print("Loaded successfully")

# Converts birthdays to ages
ages = {}
birthdays = {}
for val in label_json[age_start_key]:
	for dataset in label_json[age_start_key][val]:
		birthdays[dataset] = val

date_format = '%Y_%m_%d_%H:%M:%S'
earliest_date = None
latest_date = None
all_dates = []
for val in label_json[age_end_key]:
	for dataset in label_json[age_end_key][val]:
		if dataset in birthdays:
			try:
				bd = parsedate(birthdays[dataset],date_format)
				ed = parsedate(val,date_format)
				all_dates.append(ed)
				if earliest_date is None or earliest_date > ed:
					earliest_date = ed
				if latest_date is None or latest_date < ed:
					latest_date = ed
				age = str((ed.year - bd.year) * 12 + (ed.month - bd.month))
				if age not in ages:
					ages[age] = []
				ages[age].append(dataset)
			except:
				continue

# This is a subroutine I used to find the full range of dates across all files
#with open('all_dates.txt','w') as fileobj:
#	for i in sorted(all_dates):
#		fileobj.write(str(i) + "\n")

label_json["Ages"] = ages

medications_json = {}

dataset_to_patient_id = {}
for val in label_json["PatientID"]:
	for dataset in label_json["PatientID"][val]:
		dataset_to_patient_id[dataset] = val

dataset_to_mrn = {}
for val in label_json["MRN"]:
	for dataset in label_json["MRN"][val]:
		dataset_to_mrn[dataset] = val

mrn_to_patient_id = {}
for dataset in dataset_to_mrn:
	if dataset in dataset_to_patient_id:
		mrn = dataset_to_mrn[dataset]
		patient_id = dataset_to_patient_id[dataset]
		mrn_to_patient_id[mrn] = patient_id
json.dump(mrn_to_patient_id,open(os.path.join(working_dir,'json','translation_files','mrn_to_patient_id.json'),'w'),indent=4)

dataset_to_test_control = {}
for val in label_json["Test_Control"]:
	for dataset in label_json["Test_Control"][val]:
		dataset_to_test_control[dataset] = val

for val in label_json[age_end_key]:
	try:
		ed = parsedate(val,date_format)#datetime.datetime.strptime(val.split(".")[0],date_format)
	except:
		continue
	for dataset in label_json[age_end_key][val]:
		patientid = dataset_to_patient_id[dataset]
		meds = get_meds_before_date(patientid,ed)
		meds = list_to_str(meds)
		
		if meds not in medications_json:
			medications_json[meds] = []
		medications_json[meds].append(dataset)

medications_ever = {}
for dataset in dataset_to_patient_id:
	patientid = dataset_to_patient_id[dataset]
	if patientid in patient_id_to_med:
		meds = []
		for med,meddate in patient_id_to_med[patientid]:
			if med not in meds:
				meds.append(med)
		meds = sorted(meds)
		meds = list_to_str(meds)
		if meds not in medications_ever:
			medications_ever[meds] = []
		medications_ever[meds].append(dataset)
	else:
		if "['None']" not in medications_ever:
			medications_ever["['None']"] = []
		medications_ever["['None']"].append(dataset)
		

alzstage = {}
for val in label_json[age_end_key]:
	try:
		ed = parsedate(val,date_format)
	except:
		continue
	for dataset in label_json[age_end_key][val]:
		patientid = dataset_to_patient_id[dataset]
		mrn = dataset_to_mrn[dataset]
		# RPDR Lookup by MRN
		f = get_alz_stage_at_date(mrn,ed,
			patient_id_to_med = mrn_to_medication,
			patient_id_to_icd10_code = mrn_to_icd10_code,
			test_control=dataset_to_test_control[dataset],
			date_format="%m/%d/%Y")
		# EDW Lookup by Patient ID
		f2 = get_alz_stage_at_date(patientid,ed,
			patient_id_to_med = patient_id_to_med,
			patient_id_to_icd10_code = patient_id_to_icd10_code,
			test_control=dataset_to_test_control[dataset],
			date_format="%Y-%m-%d %H:%M:%S")
		if f == "AD" or f2 == "AD":
			f = "AD"
		elif f == "MCI" or f2 == "MCI":
			f = "MCI"
		elif f == "CONTROL" and f2 == "CONTROL":
			f = "CONTROL"
		else:
			f = "EXCLUDE"
		if f not in alzstage:
			alzstage[f] = []
		alzstage[f].append(dataset)

alzstage_icd = {}
for val in label_json[age_end_key]:
	try:
		ed = parsedate(val,date_format)
	except:
		continue
	for dataset in label_json[age_end_key][val]:
		patientid = dataset_to_patient_id[dataset]
		mrn = dataset_to_mrn[dataset]
		# RPDR Lookup by MRN
		f = get_alz_stage_at_date(mrn,ed,
			patient_id_to_med = mrn_to_medication,
			patient_id_to_icd10_code = mrn_to_icd10_code,
			test_control=dataset_to_test_control[dataset],
			date_format="%m/%d/%Y",
			consider_meds = False)
		# EDW Lookup by Patient ID
		f2 = get_alz_stage_at_date(patientid,ed,
			patient_id_to_med = patient_id_to_med,
			patient_id_to_icd10_code = patient_id_to_icd10_code,
			test_control=dataset_to_test_control[dataset],
			date_format="%Y-%m-%d %H:%M:%S",
			consider_meds = False)
		if f == "AD" or f2 == "AD":
			f = "AD"
		elif f == "MCI" or f2 == "MCI":
			f = "MCI"
		elif f == "CONTROL" and f2 == "CONTROL":
			f = "CONTROL"
		else:
			f = "EXCLUDE"
		if f not in alzstage_icd:
			alzstage_icd[f] = []
		alzstage_icd[f].append(dataset)

label_json["AlzStage"] = alzstage
label_json["AlzStageICD"] = alzstage_icd
label_json["Medications"] = medications_json
label_json["Medications_ever"] = medications_ever

for key in label_json:
	k = {}
	if key in excluded_keys:
		continue
	if key.endswith("CD") and key.replace("CD","DSC") in label_json:
		continue
#	if key.endswith("DTS"):
#		continue
	if len(label_json[key]) < 2:
		continue
	k[key] = label_json[key]
	k["discrete"] = True
	if not (key.endswith("ID") or key.endswith("CD") or key == "MRN" or key == "CPTCodeNBR") and np.mean([is_float(i) for i in k[key]]) > 0.99:# and np.mean([len(k[key][i]) for i in k[key]]) < 4:
		k["discrete"] = False
		k[key] = [[(i,j) for j in k[key][i]] for i in k[key]]
		k[key] = [item for sublist in k[key] for item in sublist]
		k[key] = filter(lambda k: is_float(k[0]),k[key])
		k[key] = [(float(i),j) for i,j in k[key]]
		k[key] = sorted(k[key],key=lambda kk:kk[0])
		if k[key][0][0] == k[key][-1][0]:
			continue
	if k["discrete"]:
		null = [0 if i != "NULL" and i != "XXXX" else len(k[key][i]) for i in k[key]]
		num_null = np.sum(null)
		alls = [len(k[key][i]) for i in k[key]]
		num_all  = np.sum(alls)
		percentage = num_null / num_all
	if (k["discrete"] and percentage < 0.9) or (not k["discrete"]):
		json.dump(k,open(os.path.join(outfolder,'%s.json'%key),'w'),indent=4)
