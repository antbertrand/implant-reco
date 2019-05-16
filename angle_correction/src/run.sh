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
#mkdir /storage/eurosilicone/ds_rotated
# mkdir /storage/test/masks/
#
#mv /paperspace/img_corr_resized /storage/eurosilicone


python3 test.py
#python3 create_dataset.py
python3 train_angle_resnet.py
