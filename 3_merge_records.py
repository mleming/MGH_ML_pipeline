#!/usr/bin/python

# Matthew Leming, MGH Center for Systems Biology, 2021
# Merges information from RPDR with information from the EDW query to output
# a combined .json file in each study's folder

import os,sys,glob,csv,json
import xml.etree.ElementTree as ET
import numpy as np
import linecache

working_dir = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))

orig_data_dir = os.path.join(working_dir,'data','RPDR_query_data')
csv_base = 'all_brain_mri_procedures_w_accession'
patients_filepath = os.path.join(working_dir,'json','edw_sql_results',
	csv_base,'%s.json' % csv_base)

accession_variable = 'AccessionNBR'
accessionline_file = os.path.join(working_dir,'csv','edw_sql_results',csv_base,'%s_line.json' % accession_variable)
accessionline_csv = os.path.join(working_dir,'csv','edw_sql_results',csv_base,'%s.csv'%csv_base)

# CSVs need to be accessed this way due to memory issues (they're really big)

if not os.path.isfile(accessionline_file):
	accessionline_dict = {}
	firstrow = True
	i = 0
	for row in csv.reader(open(accessionline_csv,'r'),delimiter=",",quotechar='"'):
		if firstrow:
			firstrow = False
			csv_header = row
		else:
			accessionline_dict[i] = row[csv_header.index(accession_variable)]
	i += 1
	json.dump(accessionline_dict,open(accessionline_file,'w'),indent=4)

#patients_dict = json.load(open(patients_filepath,'r'))
accessionline_dict = json.load(open(accessionline_file,'r'))
for row in csv.reader(open(accessionline_csv,'r')):
	csv_header = row
	break


def get_accession_record(AccessionNumber):
	aline = None
	if AccessionNumber in accessionline_dict:
		aline = accessionline_dict[AccessionNumber]
	elif "E" + AccessionNumber in accessionline_dict:
		aline = accessionline_dict["E" + AccessionNumber]
	if aline is None: return None
	line = linecache.getline(accessionline_csv, aline).replace("\n","")
	#print(line)
	r = csv.reader([line],delimiter=",",quotechar='"')
	linez = []
	for row in r: linez.append(row)
	line = linez[0]
	record = {}
	if not len(line) == len(csv_header):
		print(line)
		print(csv_header)
	assert(len(line) == len(csv_header))
	for i in range(len(csv_header)):
		record[csv_header[i]] = line[i]
	return record

medim_glob = os.path.join(working_dir,'data_processing',
			'RPDR_nifti_conversions','[ct]*','*','*.nii.gz')

medim_file_list = os.path.join(working_dir,'txt','all_filenames_nifti.txt')
problem_children_path = os.path.join(working_dir,'json','problem_children.json')

replace_json_patient_files = False
if os.path.isfile(problem_children_path):
	try:
		problem_children = json.load(open(problem_children_path,'r'))
	except:
		problem_children = {"xmls" : {},"jsons" : {},"patientids" : {}, "npys" : {},"scanids" : {}}
else:
	problem_children = {"xmls" : {},"jsons" : {},"patientids" : {}, "npys" : {},"scanids" : {}}

xml_dict = {"read":{},"ids":{}}

kk = -1
for filepath in open(medim_file_list,'r').readlines():
	if filepath is None:
		#print("No file: %s" % str(filepath))
		continue
	filepath = filepath.replace("\n","")
	if not os.path.isfile(filepath):
		#print("No file: %s" % str(filepath))
		continue
	kk += 1
	#sys.stdout.write("Percent converted: %.3f\r" % (float(kk) / len(medim_file_list) * 100))
	sys.stdout.write("Num converted: %.3f\r" % (float(kk)))
	sys.stdout.flush()
	#filepath = medim_file_list[kk]
	basepath = os.path.splitext(os.path.splitext(filepath)[0])[0]
	file_basename = os.path.basename(basepath)
	folder = os.path.basename(os.path.dirname(filepath))
	cont_test_folder = os.path.basename(os.path.dirname(os.path.dirname(filepath)))
	xml_filepaths = glob.glob(os.path.join(orig_data_dir,cont_test_folder,folder,"*.xml"))
	medim_json_output_filepath = "%s_patient.json" % basepath
	medim_json_filepath = "%s.json" % basepath
#	npy_filepath = "%s_resized_%d.npy" % (basepath,image_dims[0])
	if not replace_json_patient_files and os.path.isfile(medim_json_output_filepath):
		continue
	scanid = None
	patientid = None
	for xml_filepath in xml_filepaths:
		try:
			if xml_filepath not in xml_dict["read"]:
				xml_dict["read"][xml_filepath] = True
				tree = ET.parse(xml_filepath)
				root = tree.getroot()
				for event in root.iter('event'):
					#for k in event.iter('patient_id'):
					#	patientid = k.text
					for k in event.iter('param'):
						#print(k.attrib)
						if k.attrib["name"].endswith('Instance UID'):
							scanid    = k.text
						if k.attrib["name"] == "Accession Number":
							patientid = k.text
					xml_dict["ids"][scanid] = patientid
		except:
			problem_children["xmls"][xml_filepath] = True
			json.dump(problem_children,open(problem_children_path,'w'),indent=4)
#	if not os.path.isfile(npy_filepath):
#		if npy_filepath not in problem_children["npys"]:
#			problem_children["npys"][npy_filepath] = True
#			json.dump(problem_children,open(problem_children_path,'w'),indent=4)
#		#print("%s not found" % npy_filepath)
#		continue
	if not os.path.isfile(medim_json_filepath):
		if medim_json_filepath not in problem_children["jsons"]:
			#print("%s not found" % medim_json_filepath)
			problem_children["jsons"][medim_json_filepath] = True
			json.dump(problem_children,open(problem_children_path,'w'),indent=4)
		continue
	#else:
	#	print("Found %s" % medim_json_filepath)
	try:
		medim_dict = json.load(open(medim_json_filepath,'r'))
	except:
		if medim_json_filepath not in problem_children["jsons"]:
			problem_children["jsons"][medim_json_filepath] = True
			json.dump(problem_children,open(problem_children_path,'w'),indent=4)
		continue
		
	file_basename_base2 = file_basename.replace(')','_').replace('(','_').split("_")[0]
	
	if file_basename_base2 not in xml_dict["ids"]:
		if file_basename_base2 not in problem_children["scanids"]:
			problem_children["scanids"][file_basename_base2] = (file_basename,file_basename_base2)
			json.dump(problem_children,open(problem_children_path,'w'),indent=4)
		continue
	patientid = xml_dict["ids"][file_basename_base2]
	#if patientid not in patients_dict:
	if patientid is None:
		continue
	if patientid not in accessionline_dict and ("E" + patientid) not in accessionline_dict:
		if patientid not in problem_children["patientids"]:
			#print("%s not in patients dict" % patientid)
			problem_children["patientids"][patientid] = True
			json.dump(problem_children,open(problem_children_path,'w'),indent=4)
		continue
	#patient_info = patients_dict[patientid]
	patient_info = get_accession_record(patientid)
	medim_dict.update(patient_info)
	json.dump(medim_dict,open(medim_json_output_filepath,'w'),indent=4)
	print(medim_json_output_filepath)
#json.dump(problem_children,open(problem_children_path,'w'),indent=4)
