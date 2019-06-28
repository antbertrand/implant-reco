#!/bin/sh
#
# Install additional modules

pip install --upgrade pip

pip uninstall tensorflow
pip install tensorflow-gpu
pip install keras
pip install numpy
pip install opencv-python
pip install imutils
pip install Pillow


# Start tensorboard
#mkdir -p /artifacts/tensorboard_logs/
#tensorboard --logdir=/artifacts/tensorboard_logs/ --port=6006 &

#Create useful folders
# mkdir -p /artifacts/
# mkdir /storage/test/
#mkdir /storage/eurosilicone
#mkdir /storage/eurosilicone/ds_rotated
#mkdir /artifacts/logs

#rm -r /storage/eurosilicone/ds_rotated
#rm -r /storage/eurosilicone/img_corr_resized/
#rm -r /storage/eurosilicone/ds_corr_resized/
#mkdir /storage/eurosilicone/ds_rotated
#mkdir /storage/eurosilicone/ds_rotated/train
#mkdir /storage/eurosilicone/ds_rotated/val
#mkdir /storage/eurosilicone/ds_rotated/test
# mkdir /storage/test/masks/
#
#mv /paperspace/ds_corr_resized /storage/eurosilicone/


python3 test.py
#python3 create_dataset.py
python3 train_angle_resnet.py
