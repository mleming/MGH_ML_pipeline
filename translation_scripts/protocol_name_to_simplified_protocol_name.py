#!/usr/bin/python

import os,sys,json,csv,glob

working_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))

json_name = "ProtocolName"
json_output_name = "ProtocolNameSimplified"

json_filename = os.path.join(working_dir,"json","labels","%s.json" %json_name)
assert(os.path.isfile(json_filename))
json_output_filename = os.path.join(working_dir,"json","labels","%s.json"%json_output_name)

label_json = json.load(open(json_filename,'r'))

label_json_output = {}
label_json_output[json_output_name] = {}
label_json_output["discrete"] = True

unknowns = []

for labelo in label_json[json_name]:
	label = labelo.lower()
	# Determine angle
	angle = "unknown"
	for l in ["ax","sag","cor"]:
		if l in label:
			angle = l
	
	# Determine modality
	mod = "unknown"
	for l in ["t1","t2","dwi","swi"]:
		if l in label:
			mod = l
	
	mprage = "_mprage" if ("mprage" in label or "mp_rage" in label) else ""
	
	flair = "_flair" if "flair" in label else ""
	
	if mod == "unknown":
		continue
	
	s = "%s_%s%s%s" % (mod,angle,flair,mprage)
	s = s.upper()
	
	if s not in label_json_output[json_output_name]:
		label_json_output[json_output_name][s] = []
	
	
	label_json_output[json_output_name][s] = label_json_output[json_output_name][s] + label_json[json_name][labelo]
		
json.dump(label_json_output,open(json_output_filename,'w'),indent=4)
