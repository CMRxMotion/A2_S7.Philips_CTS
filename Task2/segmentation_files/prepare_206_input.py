import os
import SimpleITK as sitk
# import matplotlib.pyplot as plt
import numpy as np
# import cv2
# import copy
import argparse
import shutil

'''
-i /home/philips/disk1/CMR/docker/task2/input
-o /home/philips/disk1/CMR/docker/task2/nnUNet_output/rename_input
'''
parser = argparse.ArgumentParser()
parser.add_argument('--path_predict', help="", required=True)
parser.add_argument("--path_img", required=True, help="")
parser.add_argument("--path_save", required=True, help="")


args = parser.parse_args()
path_predict = args.path_predict
path_img = args.path_img
path_save = args.path_save

if not os.path.exists(path_save):
    os.makedirs(path_save)

print('Running prepare_206_input')
files = os.listdir(path_img)
for file in files:
    img_file = os.path.join(path_img, file)
    pred_file = os.path.join(path_predict, file.replace('_0000.nii.gz', '.nii.gz'))
    img_sitk = sitk.ReadImage(img_file)
    img_arr = sitk.GetArrayFromImage(img_sitk)
    pred_arr = sitk.GetArrayFromImage(sitk.ReadImage(pred_file))
    img_out = img_arr * pred_arr
    img_out = sitk.GetImageFromArray(img_out)
    img_out.SetSpacing = img_sitk.GetSpacing()
    img_out.SetOrigin = img_sitk.GetOrigin()
    sitk.WriteImage(img_out, os.path.join(path_save, file))
print('Finish prepare_206_input')

