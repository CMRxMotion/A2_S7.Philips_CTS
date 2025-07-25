import os
import SimpleITK as sitk
import numpy as np
import matplotlib.pyplot as plt
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--path_predict_1', help="", required=True)
parser.add_argument("--path_predict_2", required=True, help="")
parser.add_argument("--path_save", required=True, help="")

# path_1 = '/home/philips/disk1/CMR/nnUNet_output/Task207_predict/labels_Va_9_4'
# path_2 = '/home/philips/disk1/CMR/nnUNet_output/Task201_predict/labels_Va_9_4'
# path_submit = '/home/philips/disk1/CMR/nnUNet_output/Task207_predict/labels_Va_207_3'
# path_ori = '/home/philips/disk1/CMR/nnUNet_output/nnUNet_raw_data/Task207_CMR/imagesVa'
args = parser.parse_args()
path_1 = args.path_predict_1
path_2 = args.path_predict_2
path_save = args.path_save

if not os.path.exists(path_save):
    os.makedirs(path_save)

pats = os.listdir(path_1)
for pat in pats:
    # pat_sub = os.path.join(path_submit, pat)
    pat_1 = os.path.join(path_1, pat)
    pat_2 = os.path.join(path_2, pat)
    # pat_ori = os.path.join(path_ori, pat.replace('.nii.gz', '_0000.nii.gz'))
    # arr_sub = sitk.GetArrayFromImage(sitk.ReadImage(pat_sub))
    img_sitk = sitk.ReadImage(pat_1)
    arr_1 = sitk.GetArrayFromImage(img_sitk)
    arr_2 = sitk.GetArrayFromImage(sitk.ReadImage(pat_2))
    arr_merge = arr_2 + arr_1
    arr_merge[arr_merge > 1] = 1
    img_out = sitk.GetImageFromArray(img_out)
    img_out.SetSpacing = img_sitk.GetSpacing()
    img_out.SetOrigin = img_sitk.GetOrigin()
    sitk.WriteImage(img_out, os.path.join(path_save, pat))


    # ori = sitk.GetArrayFromImage(sitk.ReadImage(pat_ori))
    # ori = (ori-np.min(ori))/(np.max(ori)-np.min(ori))*2
    # print('arr_sub', arr_sub.shape)
    # for i in range(arr_sub.shape[0]):
    #     minus_sub = arr_sub[i, :, :]-arr_merge[i, :, :]
    #     minus_sub[minus_sub > 0] = 1
    #     print(np.max(minus_sub), np.min(minus_sub))
    #     print(np.max(arr_sub[i, :, :]), np.max(arr_merge[i, :, :]))
    #     sums_sub = np.sum(minus_sub)
    #     if sums_sub > 500:
    #         print('pat', pat)
    #         print('sum_sub', sums_sub)
    #         plt.imshow(np.concatenate((arr_sub[i, :, :], arr_merge[i, :, :],
    #                                    ori[i, :, :], minus_sub
    #                                    ), axis=1), cmap='gray')
    #         plt.show()






