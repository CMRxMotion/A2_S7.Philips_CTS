
python /home/philips/disk1/CMR/docker/task2/segmentation_files/rename_input.py -i /home/philips/disk1/CMR/docker/task2/input \
-o /home/philips/disk1/CMR/docker/task2/segmentation_files/nnUNet_output/rename_input

python /home/philips/disk1/CMR/docker/task2/segmentation_files/nnUNet/nnunet/inference/predict_simple.py -i /home/philips/disk1/CMR/docker/task2/segmentation_files/nnUNet_output/rename_input \
-o /home/philips/disk1/CMR/docker/task2/segmentation_files/nnUNet_output/207_predict \
-m 2d -f all -tr nnUNetTrainerV2 -chk model_best -t  Task207_CMR

# python prepare_206_input.py --path_predict /home/philips/disk1/CMR/docker/task2/segmentation_files/nnUNet_output/207_predict \
# --path_img /home/philips/disk1/CMR/docker/task2/segmentation_files/nnUNet_output/rename_input \
# --path_save /home/philips/disk1/CMR/docker/task2/segmentation_files/nnUNet_output/206_input

# python /home/philips/disk1/CMR/docker/task2/segmentation_files/nnUNet/nnunet/inference/predict_simple.py -i /home/philips/disk1/CMR/docker/task2/segmentation_files/nnUNet_output/206_input \
# -o /segmentation_files/nnUNet_output/seg_predict \
# -m 2d -f all -tr nnUNetTrainerV2 -chk model_best -t  Task206_CMR

