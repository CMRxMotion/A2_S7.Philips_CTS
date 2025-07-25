
python3 /segmentation_files/rename_input.py -i /input \
-o /segmentation_files/nnUNet_output/rename_input

python3 /segmentation_files/nnUNet/nnunet/inference/predict_simple.py -i /segmentation_files/nnUNet_output/rename_input \
-o /segmentation_files/nnUNet_output/207_predict \
-m 2d -f all -tr nnUNetTrainerV2 -chk model_best -t  Task201_CMR

python3 /segmentation_files/prepare_206_input.py --path_predict /segmentation_files/nnUNet_output/207_predict \
--path_img /segmentation_files/nnUNet_output/rename_input \
--path_save /segmentation_files/nnUNet_output/206_input

python3 /segmentation_files/nnUNet/nnunet/inference/predict_simple.py -i /segmentation_files/nnUNet_output/206_input \
-o /segmentation_files/nnUNet_output/seg_predict \
-m 2d -f all -tr nnUNetTrainerV2 -chk model_best -t  Task206_CMR

python3 /segmentation_files/copy_to_output.py -i /segmentation_files/nnUNet_output/seg_predict \
-o /output