# copy input

import argparse
import os
import shutil

'''
-i /home/philips/disk1/CMR/docker/task2/input
-o /home/philips/disk1/CMR/docker/task2/nnUNet_output/rename_input
'''
parser = argparse.ArgumentParser()
parser.add_argument("-i", '--input_folder', help="", required=True)
parser.add_argument('-o', "--output_folder", required=True, help="")

args = parser.parse_args()
input_folder = args.input_folder
output_folder = args.output_folder
if not os.path.exists(output_folder):
	os.makedirs(output_folder)
paths = os.listdir(input_folder)
for path in paths:
	# print('os.path.join(input_folder, path)', os.path.join(input_folder, path))
	# print(os.path.join(output_folder, path.replace('.nii.gz', '_0000.nii.gz')))
	shutil.copy(os.path.join(input_folder, path), os.path.join(output_folder, path.replace('.nii.gz', '_0000.nii.gz')))