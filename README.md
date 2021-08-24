# MGH MRI training kit

These are Python and shell scripts used to train an extremely large clinical MRI dataset at Mass General Brigham by Alzheimer's, MCI, and healthy controls. Due to the sensitive nature of the data, including particular data files is infeasible, but this code is released for transparency of our analysis techniques. Paper under review.

## Dependencies

dcm2niix
FSL Library
Keras/Tensorflow
Pandas
Numpy

## Use

Python scripts starting with a number are run in that order, and they take raw DICOM files and convert them first to NIFTI and then to resized NUMPY files. They then combine data with covariates across .csv files taken from RPDR and EDW inquiries.

The seventh script may be run on its own with just a Pandas Dataframe (stored in ${WORKING_DIR}/pandas/cache/all_vars.pkl), with indices as a partial path to the Numpy files (for instance, /data_processing/RPDR_nifti_conversions/path/to/numpy is converted to /your/working/directory/data/RPDR_nifti_conversions/path/to/numpy_resized_96.npy) and columns as covariates, with data either being strings (for categorical data) or floats. Our default example covariate_pairings.json file is included, which tells the dataset matching algorithm which variables to match by.
