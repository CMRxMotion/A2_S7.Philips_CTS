# -*- coding: utf-8 -*-
"""
Created on Tue Aug 30 12:01:02 2022

@author: ssj
"""

import os
import argparse

import pycaret
import pandas as pd
from pycaret.classification import *


import torch
import numpy as np
import logging
import SimpleITK as sitk
from monai.transforms import *
from torchvision import transforms as ttf
from tqdm import tqdm
from efficientnet import EfficientNet

import radiomics
import six
import csv

import shutil

def Radiomics_extractor_ini():
    ###BEGIN radiomics settings  
    settings = {}
    settings['sigma'] = [1,2,3]  
    #设置了这个sigma 后面用LoG作为image type才能有结果，同时是列表，可迭代
    # settings['resampledPixelSpacing'] = None  
    # [3,3,3] is an example for defining resampling (voxels with size 3x3x3mm)
    settings['interpolator'] = sitk.sitkBSpline
    settings['normalize'] = True  #设置图像归一化，CT不进行标准化
    #settings['normalizeScale'] = 1
    # settings['removeOutliers'] = 6
    # Set to 3 to remove 3-sigma outliers
    #以下两个只要保留一个，一个是按照灰度值域划分binCount个bin，一个是按给定的Width划分bin
    #settings['binWidth'] = 16  #CT
    settings['binCount'] = 32 #MR
    settings['resampledPixelSpacing'] = [1.0,1.0,1.0] #the resample spacing
    
    extractor = radiomics.featureextractor.RadiomicsFeatureExtractor(**settings)
    
    # print(radiomics.getImageTypes())
    # print(radiomics.getFeatureClasses())
    extractor.disableAllImageTypes()
    extractor.disableAllFeatures()
    
    # extractor.enableImageTypes(Original={}, LoG={}, Wavelet={})
    extractor.enableImageTypeByName('Wavelet')
    extractor.enableImageTypeByName('LoG')
    extractor.enableImageTypeByName('Gradient')
    extractor.enableImageTypeByName('Original')
    
    extractor.enableFeatureClassByName('firstorder')
    extractor.enableFeatureClassByName('glrlm')
    extractor.enableFeatureClassByName('ngtdm')
    extractor.enableFeatureClassByName('gldm')
    extractor.enableFeatureClassByName('glcm')
    extractor.enableFeatureClassByName('glszm')
    extractor.enableFeatureClassByName('shape')
    
    # =============================================================================
    # ### 启用全部特征和滤波，可能会因为2D/3D产生warning
    # extractor.enableAllFeatures()
    # extractor.enableAllImageTypes()
    # =============================================================================
    
    # print ("Extraction parameters:\n\t", extractor.settings)
    #print ("setup filters:\n\t", extractor.enabledImagetypes)
    #print ("features to extract:\n\t", extractor.enabledFeatures)
    ###END radiomics settings 
    return extractor

def get_row_feature(extractor,name, img,roi,intype,heart = 1): #LV
    result = extractor.execute(img, roi, label=heart)
    result_keys_name=[]
    row_feature = {}
    for key, val in six.iteritems(result):
        result_keys_name.append(key)
        row_feature[key] = val
    keys_str = [x for x in result_keys_name if not 'diagnostics' in x]#
    keys_str.insert(0, 'ID')
    row_feature['ID'] = name
    keys_str.insert(1, 'label')
    
    row_feature['label'] = intype 
    
    return keys_str,row_feature

def check_spacing(opt):
    #纠正mask的spacing什么的    
    mr_folder = opt.input
    roi_folder = '/segmentation_files/nnUNet_output/seg_predict'
    files = os.listdir(mr_folder)
    X_test_cmr = [x.replace('.nii.gz','') for x in files if x.endswith('.nii.gz')]

    out_folder = opt.output+ '/nnunet_predict_check'
    if not os.path.exists(out_folder):
        os.mkdir(out_folder)
    for file in X_test_cmr:
        img = os.path.join(mr_folder,file+'.nii.gz')
        roi = os.path.join(roi_folder,file+'.nii.gz')
        out = os.path.join(out_folder,file+'.nii.gz')
        
        img = sitk.ReadImage(img, sitk.sitkInt16)    
        img_arr = sitk.GetArrayFromImage(img)
            
            
        mask = sitk.ReadImage(roi, sitk.sitkUInt8)    
        mask_arr = sitk.GetArrayFromImage(mask)
        
        new_mask = sitk.GetImageFromArray(mask_arr)
        new_mask.SetDirection(img.GetDirection())
        new_mask.SetOrigin(img.GetOrigin())
        new_mask.SetSpacing((img.GetSpacing()[0] , img.GetSpacing()[1], img.GetSpacing()[2]))
        sitk.WriteImage(new_mask, out)
    

def generate_radiomics_feature(opt):
    logger = logging.getLogger('radiomics')
    logger.setLevel(logging.ERROR)
    logger.info('start generating radiomics features')
    print('start generating radiomics features')
    check_spacing(opt)
    mr_folder = opt.input
    roi_folder = opt.output+ '/nnunet_predict_check' 
    logger.info('segmentation checked')
    image_paths = os.listdir(mr_folder)

    files = sorted(image_paths)
    X_test_cmr = [x.replace('.nii.gz','') for x in files]

    csv_out = opt.output+'/rad_data.csv'
    extractor = Radiomics_extractor_ini()
    ini_count = 0
    
    with open(csv_out, 'w',newline='') as csvfile:
        for i in range(0,len(X_test_cmr)):
            name = X_test_cmr[i]
            file = name+'.nii.gz'
            img = os.path.join(mr_folder,file)
            roi = os.path.join(roi_folder,file)
            quality = 9
            # start = time.time()
            keys_str,row_feature = get_row_feature(extractor,name, img,roi,quality,heart = 1)
            writer = csv.DictWriter(csvfile, fieldnames=keys_str, extrasaction='ignore')
            if ini_count == 0:
                writer.writeheader()
                ini_count += 1
            writer.writerow(row_feature)
            #print('{}s'.format(time.time()-start)) 
            if i % 5 == 0:
            	logger.info(f'Progress: {i / len(X_test_cmr) * 100: 02.02f}%')
            	print(f'Progress: {i / len(X_test_cmr) * 100: 02.02f}%')
    

def get_radiomics_result(opt):
    logger.info('get radiomics result')
    et = load_model('/segmentation_files/workspace/et')
    rf = load_model('/segmentation_files/workspace/rf')
    dt = load_model('/segmentation_files/workspace/dt')
    gbc = load_model('/segmentation_files/workspace/gbc')
    logger.info('radiomics models ready')
    rad_csv = opt.output+'/rad_data.csv'
    
    
    cases_val_path = rad_csv
    cases_val = pd.read_csv(cases_val_path)
    cases_val_id = cases_val['ID']
    cases_val = cases_val.drop(columns='ID') 
    
    results = pd.DataFrame()
    results['Image'] = cases_val_id
    
    
    pred_et = predict_model(et, data= cases_val)
    results['Label_et'] = pred_et['Label']+1
    results['Score_et'] = pred_et['Score']
    
    pred_rf = predict_model(rf, data= cases_val)
    results['Label_rf'] = pred_rf['Label']+1
    results['Score_rf'] = pred_rf['Score']
    
    pred_dt = predict_model(dt, data= cases_val)
    results['Label_dt'] = pred_dt['Label']+1
    results['Score_dt'] = pred_dt['Score']
    
    pred_gbc = predict_model(gbc, data= cases_val)
    results['Label_gbc'] = pred_gbc['Label']+1
    results['Score_gbc'] = pred_gbc['Score']
   
    results.to_csv(opt.output+'/results_rad.csv')
    logger.info('prediction complete')
    
    
def fuse_dl_rad(opt):
    logger.info('start fusing dl & radiomics results')
    csv_rad = opt.output+'/results_rad.csv'
    csv_eff = opt.output+'/results_eff.csv'
    df_rad = pd.read_csv(csv_rad)
    df_eff = pd.read_csv(csv_eff)
    df_eff['Label']=-1
    for i, row in df_eff.iterrows():
        list1 = [row['1'],row['2'],row['3']]
        tmp = max(list1)
        df_eff.at[i,'Score'] = tmp
        df_eff.at[i,'Label'] = list1.index(tmp)+1
        
    df_rad['Label_eff']=df_eff['Label']
    df_rad['Score_eff']=df_eff['Score']
    df_rad['Score_eff1']=df_eff['1']
    df_rad['Score_eff2']=df_eff['2']
    df_rad['Score_eff3']=df_eff['3']
    
    df=df_rad
    df['output'] = -1
    
    method = ['et','rf','dt','gbc','eff']
    for i, row in df.iterrows():
        #print(row['Image'])
        score_list=[0,0,0]
        for m in method:
            pre = row['Label_{}'.format(m)]
            #print(pre)
            score = row['Score_{}'.format(m)]       
            score_list[pre-1]+=1
       
        max_value = max(score_list)
        index = score_list.index(max_value)+1
        df.at[i,'output'] = index
    
    
    df1 = pd.DataFrame()
    df1['Image'] = df['Image']
    df1['Label'] = df['output']
    df1.to_csv(opt.output+'/output.csv',index=None)
    logger.info('fuse complete')
    
        



def get_logger():
    logger = logging.getLogger('cmr cls prediction')
    logger.setLevel(logging.DEBUG)
    ch = logging.StreamHandler()
    ch.setLevel(logging.DEBUG)
    formatter = logging.Formatter('%(asctime)s [%(levelname)s] %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
    ch.setFormatter(formatter)
    logger.addHandler(ch)
    return logger

def read_image(path):
    image = sitk.GetArrayFromImage(sitk.ReadImage(path))
    return rescale_array(image)
    


def get_eff_results(opt):
    testing = True
    weights_file = '/segmentation_files/workspace/snapshot_084_best_7823.pth.tar' #eff
    scoreout_csv = os.path.join(opt.output, 'results_eff.csv')
    out_csv = os.path.join(opt.output, 'output_eff.csv')
    src_root = opt.input
    
    topk = -1

    device_id = 0
    label_map = [1, 2, 3]
    
    


    # walk input scans
    image_paths = []
    for ds, _, fs in os.walk(src_root):
        for f in fs:
            image_paths.append(os.path.join(ds, f))
    logger.info(image_paths)
    logger.info(f"Scans len = {len(image_paths)}")

    norm_cfg = dict(type='SyncBN', eps=1e-3)
    conv_cfg = dict(type='Conv2dAdaptivePadding')
    model = EfficientNet('b0', in_channels=1, head_type='classification', num_classes=3,
                         conv_cfg=conv_cfg, norm_cfg=norm_cfg)
    # conv_cfg = dict(type='Conv2d')
    # model = ResNet(50, in_channels=1, head_type='classification', num_classes=3,
    #                conv_cfg=conv_cfg, norm_cfg=norm_cfg, deep_stem=True)

    checkpoint = torch.load(weights_file, map_location='cpu')
    model.load_state_dict(checkpoint['state_dict'])
    model.eval()
    model.cuda(device_id)

    logger.info('Model ready.')

    v_fn = Compose([
        ttf.ToTensor(),
        ResizeWithPadOrCrop((512, 512)),
    ])

    ground_truths = []
    predictions = []
    scores_record = []
    if testing:
        df = []

    for fid, f in enumerate(sorted(image_paths)):
        
        image = read_image(f)
        
        pid = os.path.basename(f).replace('.nii.gz', '')


        slice_preds = []
        for sli in image:
            
            sli = v_fn(sli).cuda(device_id).unsqueeze(0)
            with torch.no_grad():
                output = model(sli)
            output = torch.softmax(output, 1).squeeze(0)
            slice_preds.append(output.cpu().numpy())
        slice_preds = np.stack(slice_preds).transpose((1, 0))
        score_preds = []
        for cate_id, cate_preds in enumerate(slice_preds):
            cate_preds = cate_preds[np.argsort(cate_preds)[::-1]]
            if topk > 0:
                score_preds.append(np.mean(cate_preds[:topk]))
            else:
                score_preds.append(np.mean(cate_preds))

        # record preds
        predictions.append(np.argmax(score_preds))
        scores_record.append(np.array(score_preds))

        # testing
        if testing:
            df.append({"Image": pid, "Label": label_map[np.argmax(score_preds)]})

        if fid % 5== 0:
            logger.info(f'Progress: {fid / len(image_paths) * 100: 02.02f}%')

    # output
    if testing:
        df = pd.DataFrame(df)
        df.to_csv(out_csv, index=None)
        scores_series = np.stack(scores_record).T
        score_df = {'Image': df['Image']}
        for idx, i in enumerate(label_map):
            score_df[i] = pd.Series(scores_series[idx])
        score_df = pd.DataFrame(score_df)
        score_df.to_csv(scoreout_csv, index=None)
        
    return 


    

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='cmr task1')
    parser.add_argument('--input',type=str,help='input directory')
    parser.add_argument('--output',type=str,help='output directory')
    logger = get_logger()
    opt = parser.parse_args()

    output = opt.output

    opt.output = '/segmentation_files/tmp'
    if not os.path.exists(opt.output):
        os.mkdir(opt.output)
    else:
        shutil.rmtree('/segmentation_files/tmp')
        os.mkdir('/segmentation_files/tmp')
        print('clean tmp')

    
    #产生eff
    get_eff_results(opt) 
    
    #调用产生radiomics csv
    generate_radiomics_feature(opt)
    
    #radiomics模型predict
    get_radiomics_result(opt)
    
    #fuse dl and radiomics
    fuse_dl_rad(opt)

    #names = ['output','rad_data','results_eff','results_rad']
    names = ['output']

    for n in names:
        src = opt.output+'/{}.csv'.format(n)
        dst = output + '/{}.csv'.format(n)
        shutil.copy(src,dst)


    
    
    
    
    
    
    
    
