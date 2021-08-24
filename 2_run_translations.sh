#!/bin/bash

# Matthew Leming, MGH Center for Systems Biology, 2021
# Runs Python translation scripts in bulk that translate various .csv and .tsv
# Files to .json files, greatly reducing computational times later on. All files
# output to ${WORKING_DIR}/json/translation_files

#translation_scripts/datasortjson_to_datasortssimcut.py
echo "Retrieving nifti file names"
#translation_scripts/get_all_nifti_files.py
echo "Translating ICD codes"
#translation_scripts/icd9_to_icd10.py
echo "Translating MRNs to diagnoses"
translation_scripts/mrn_to_diagnosis.py
echo "Translating MRNs to ICD codes"
translation_scripts/mrn_to_icd10.py
echo "Translating MRNs to medications"
translation_scripts/mrn_to_medication.py
echo "Translating Patient IDs to ICD10 codes"
translation_scripts/patient_id_to_icd10.py
echo "Translating Patient IDs to Medications"
translation_scripts/patient_id_to_medication.py
#translation_scripts/protocol_name_to_simplified_protocol_name.py
