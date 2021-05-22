from dicom_csv import join_tree, aggregate_images, stack_images
import os
from pydicom import dcmread
import nibabel as nb
import numpy as np

path = "C:\\Users\\bzavo\\Documents\\MRIphantom\\MRIphantom\\MR_Phantom_scans\\DICOM_CT_contour_Copy\\"
df = join_tree(path, relative=True, verbose=False)

# # dicoms = df[df.NoError & df.HasPixelArray]
to_group = ['PatientID', 'SeriesInstanceUID', 'StudyInstanceUID', 'PathToFolder']
images = aggregate_images(df, to_group)

datasets = []
for filename in os.listdir(path):
   datasets.append(dcmread(path + filename))


pathtosave = 'C:\\Users\\bzavo\\Documents\\MRIphantom\\MRIphantom\\3dimages\\'
img = stack_images(datasets)
nb.save(nb.Nifti1Image(img, np.eye(4)), pathtosave + "file.nii")


