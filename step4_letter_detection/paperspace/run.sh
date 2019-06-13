#!/bin/sh
#
# Install additional modules
#apt-get install unzip

pip install --upgrade pip

#pip install zipfile36
#pip install progressbar2

#pip install keras
pip install numpy
pip install opencv-python
pip install imutils
pip install Pillow
pip install 'cython'
pip install 'keras-resnet'
pip install 'git+https://github.com/cocodataset/cocoapi.git#subdirectory=PythonAPI'
pip install 'h5py'
pip install 'keras'
pip install 'matplotlib'
pip install 'numpy>=1.14'
pip install 'opencv-python>=3.3.0'
pip install 'pillow'
pip install 'progressbar2'
pip install 'pytest-flake8'
pip install 'pytest-xdist'
pip install 'tensorflow'
pip uninstall tensorflow
pip install tensorflow-gpu

# Start tensorboard
#mkdir -p /artifacts/tensorboard_logs/
#tensorboard --logdir=/artifacts/tensorboard_logs/ --port=6006 &

#Create useful folders
# mkdir -p /artifacts/
# mkdir /storage/test/
#mkdir /storage/eurosilicone
#mkdir /storage/eurosilicone/ds_rotated
#mkdir /artifacts/models


#mkdir /storage/eurosilicone/ds_step4
#mkdir /storage/eurosilicone/ds_step4/gen_dataset_big
#mkdir /storage/eurosilicone/ds_step4/gen_dataset_big/val
#mkdir /storage/eurosilicone/ds_step4/gen_dataset_big/train
#rm -r /storage/eurosilicone/ds_rotated
#rm -r /storage/eurosilicone/img_corr_resized/
#rm -r /storage/eurosilicone/ds_corr_resized/
#mkdir /storage/eurosilicone/ds_rotated3
#mkdir /storage/eurosilicone/ds_rotated3/train
#mkdir /storage/eurosilicone/ds_rotated3/val
#mkdir /storage/eurosilicone/ds_rotated3/test
# mkdir /storage/test/masks/
#
#mv /paperspace/ds_corr_resized_uncropped /storage/eurosilicone/
#unzip /storage/eurosilicone/ds_corr_full/val/img_corr_full.zip -d /storage/eurosilicone/ds_corr_full/val/
#unzip /storage/eurosilicone/ds_corr_full/train/img_corr_full.zip -d /storage/eurosilicone/ds_corr_full/train/

#cd /paperspace/keras_retinanet

#python3 test.py
#python3 dataset_creation.py
#python3 zip.py
#python3 create_dataset.py
#keras_retinanet/bin/train.py --batch-size=1 --random-transform --compute-val-loss --steps=200 --weighted-average csv ../gen_dataset/ann_train.csv ../gen_dataset/class_mapping.csv --val-annotations ../gen_dataset/ann_test.csv
#/paperspace/keras_retinanet/bin/debug.py --annotations csv /storage/eurosilicone/ds_step4/split_dataset_big2/ann_train.csv /storage/eurosilicone/ds_step4/split_dataset_big2/class_mapping.csv

#/paperspace/keras_retinanet/bin/train.py --batch-size=1 --lr=1e-4 --random-transform --compute-val-loss --steps=800 --weighted-average --snapshot-path /artifacts/models/ csv /storage/eurosilicone/ds_step4/split_dataset_big2/ann_train.csv /storage/eurosilicone/ds_step4/split_dataset_big2/class_mapping.csv --val-annotations /storage/eurosilicone/ds_step4/split_dataset_big2/ann_test.csv

keras_retinanet/bin/train.py --batch-size=1 --lr=1e-5 --random-transform --compute-val-loss --steps=600 --backbone="mobilenet128_1.0" --weighted-average --snapshot /storage/eurosilicone/ds_step4/models/overfit_gen_mobilenet.h5 --snapshot-path /storage/eurosilicone/ds_step4/models/ csv /storage/eurosilicone/ds_step4/real_dataset/ann_train_mix.csv /storage/eurosilicone/ds_step4/real_dataset/class_mapping.csv --val-annotations /storage/eurosilicone/ds_step4/real_dataset/ann_val.csv
