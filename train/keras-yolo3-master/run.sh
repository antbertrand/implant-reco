#!/bin/sh
#
# Install additional modules
pip install --upgrade pip
pip install imgaug
#pip install tqdm
pip install awscli
pip install shapely
pip install .

python3 setup.py build_ext --inplace

# Start tensorboard
#mkdir -p /artifacts/tensorboard_logs/
#tensorboard --logdir=/artifacts/tensorboard_logs/ --port=8888 &

export AWS_ACCESS_KEY_ID=AKIAIE2OPCN2K47HTOSQ
export AWS_SECRET_ACCESS_KEY=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

mkdir -p /artifacts/

#For inference
aws s3 sync s3://cardamin-paperspace/eurosilicone/ /storage --delete

#python3 convert.py -w /storage/misc/yolov3.cfg /storage/misc/yolov3.weights /artifacts/yolo_weights.h5
#python3 convert.py -w /storage/misc/tiny_yolov3.cfg /storage/misc/tiny_yolov3.weights /artifacts/tiny_yolo_weights.h5

python3 ./convert_to_voc.py \
    /storage/dataset/lettres_equal_crop/img/ \
    /storage/dataset/lettres_equal_crop/ann/ \
    /artifacts/train.txt

#python3 train.py


# Sync the artifacts folder on s3
export AWS_ACCESS_KEY_ID=AKIAIE2OPCN2K47HTOSQ
export AWS_SECRET_ACCESS_KEY=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
#aws s3 sync /artifacts s3://cardamin-paperspace/artifacts/keras-retinanet/`date "+%Y%m%d-%H%M%S"`
