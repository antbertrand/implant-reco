# Chip detection

The goal here is to detect precisly where the chip is on the implant.

## Getting Started

The detector used is RetinaNet (Lin, Goyal et al. 2017) <br />
To learn more about the way it works : [link](https://blog.zenggyu.com/en/post/2018-12-05/retinanet-explained-and-demystified/)<br />
The implementation used is an implementation of RetinaNet in Keras : [keras_retinanet](https://github.com/fizyr/keras-retinanet)

To test the detection on some images, clone this repo on your machine.




#### DL the dataset

The dataset is stored on Azure, the URL of the blob container is the following :
*https://eurosilicone.blob.core.windows.net/dsdetection*

This will give you a dataset split in three : a train, validation and test set. The split has been done manually, to make sure very similar images do not end up in different splits, which would distort the results.

Split                      | Train    |     Validation   |     Test   
------------------------------|-------------|-------------|-------------------|
Nb of images              |   256         |   52     |   40           

You should also have an annotation folder in which there are the info of the true bouding boxes for all the images. This is only useful if you want to train a new network.

#### Downloading the models

We used a RetinaNet, once trained it needs to be converted to an inference model before being able to be tested on new images.

The inference model is stored on Azure :
*https://eurosilicone.blob.core.windows.net/weights/retinanet_detection_resnet50_inf.h5*

The raw model can be used to continue training with the adding of new images. If needed is also stored on Azure :
*https://eurosilicone.blob.core.windows.net/weights/retinanet_detection_resnet50.h5*

##### **In order to use the exact same commands showed later on, please place the downloaded models and dataset in their respecting folders of the repository.**

### Prerequisites

Things needed to run the detection :

* [Keras](https://www.pyimagesearch.com/2016/11/14/installing-keras-with-tensorflow-backend/)
* [OpenCV2](https://pypi.org/project/opencv-python/)



<br /><br />

## Testing

To test the network use the chip_detector.py file.

## Usage

To train it on a custom dataset, a CSV file can be used as a way to pass the data.
See below for more details on the format of these CSV files.

### CSV datasets
The `CSVGenerator` provides an easy way to define your own datasets.
It uses two CSV files: one file containing annotations and one file containing a class name to ID mapping.

#### Annotations format
The CSV file with annotations should contain one annotation per line.
Images with multiple bounding boxes should use one row per bounding box.
Note that indexing for pixel values starts at 0.
The expected format of each line is:
```
path/to/image.jpg,x1,y1,x2,y2,class_name
```

Some images may not contain any labeled objects.
To add these images to the dataset as negative examples,
add an annotation where `x1`, `y1`, `x2`, `y2` and `class_name` are all empty:
```
path/to/image.jpg,,,,,
```

A full example:
```
/data/imgs/img_001.jpg,837,346,981,456,cow
/data/imgs/img_002.jpg,215,312,279,391,cat
/data/imgs/img_002.jpg,22,5,89,84,bird
/data/imgs/img_003.jpg,,,,,
```

This defines a dataset with 3 images.
`img_001.jpg` contains a cow.
`img_002.jpg` contains a cat and a bird.
`img_003.jpg` contains no interesting objects/animals.


#### Class mapping format
The class name to ID mapping file should contain one mapping per line.
Each line should use the following format:
```
class_name,id
```

Indexing for classes starts at 0.
Do not include a background class as it is implicit.

For example:
```
cow,0
cat,1
bird,2
```

## Training
A model is trained using the keras_retinanet/bin/train.py script.
You have to run the script directly from inside the folder `retinanet`.

Every parameters is given through the command line as arguments. The following lists all the different arguments that can be given to the train.py script :

```python


csv_parser = subparsers.add_parser('csv')
csv_parser.add_argument('annotations', help='Path to CSV file containing annotations for training.')
csv_parser.add_argument('classes', help='Path to a CSV file containing class label mapping.')
csv_parser.add_argument('--val-annotations', help='Path to CSV file containing annotations for validation (optional).')

group = parser.add_mutually_exclusive_group()
group.add_argument('--snapshot',          help='Resume training from a snapshot.')
group.add_argument('--imagenet-weights',  help='Initialize the model with pretrained imagenet weights. This is the default behaviour.', action='store_const', const=True, default=True)
group.add_argument('--weights',           help='Initialize the model with weights from a file.')
group.add_argument('--no-weights',        help='Don\'t initialize the model with any weights.', dest='imagenet_weights', action='store_const', const=False)

parser.add_argument('--backbone',         help='Backbone model used by retinanet.', default='resnet50', type=str)
parser.add_argument('--batch-size',       help='Size of the batches.', default=1, type=int)
parser.add_argument('--gpu',              help='Id of the GPU to use (as reported by nvidia-smi).')
parser.add_argument('--multi-gpu',        help='Number of GPUs to use for parallel processing.', type=int, default=0)
parser.add_argument('--multi-gpu-force',  help='Extra flag needed to enable (experimental) multi-gpu support.', action='store_true')
parser.add_argument('--epochs',           help='Number of epochs to train.', type=int, default=50)
parser.add_argument('--steps',            help='Number of steps per epoch.', type=int, default=10000)
parser.add_argument('--lr',               help='Learning rate.', type=float, default=1e-5)
parser.add_argument('--snapshot-path',    help='Path to store snapshots of models during training (defaults to \'./snapshots\')', default='./snapshots')
parser.add_argument('--tensorboard-dir',  help='Log directory for Tensorboard output', default='./logs')
parser.add_argument('--no-snapshots',     help='Disable saving snapshots.', dest='snapshots', action='store_false')
parser.add_argument('--no-evaluation',    help='Disable per epoch evaluation.', dest='evaluation', action='store_false')
parser.add_argument('--freeze-backbone',  help='Freeze training of backbone layers.', action='store_true')
parser.add_argument('--random-transform', help='Randomly transform image and annotations.', action='store_true')
parser.add_argument('--image-min-side',   help='Rescale the image so the smallest side is min_side.', type=int, default=800)
parser.add_argument('--image-max-side',   help='Rescale the image if the largest side is larger than max_side.', type=int, default=1333)
parser.add_argument('--config',           help='Path to a configuration parameters .ini file.')
parser.add_argument('--weighted-average', help='Compute the mAP using the weighted average of precisions among classes.', action='store_true')
parser.add_argument('--compute-val-loss', help='Compute validation loss during training', dest='compute_val_loss', action='store_true')

# Fit generator arguments
parser.add_argument('--workers', help='Number of multiprocessing workers. To disable multiprocessing, set workers to 0', type=int, default=1)
parser.add_argument('--max-queue-size', help='Queue length for multiprocessing workers in fit generator.', type=int, default=10)
```

<br />

The command we used to train the model is the following:

```shell
keras_retinanet/bin/train.py --batch-size=1 --random-transform --compute-val-loss --steps=300 --weighted-average csv ../ann_train.csv ../class_mapping.csv --val-annotations ../ann_val.csv
```

#### Changing backbone architecture
The default backbone is `resnet50`. You can change this using the `--backbone=xxx` argument in the running script.
`xxx` can be one of the backbones in resnet models (`resnet50`, `resnet101`, `resnet152`), mobilenet models (`mobilenet128_1.0`, `mobilenet128_0.75`, `mobilenet160_1.0`, etc), densenet models or vgg models. The different options are defined by each model in their corresponding python scripts (`resnet.py`, `mobilenet.py`, etc).


#### Converting a training model to inference model
The training procedure of `keras-retinanet` works with *training models*. These are stripped down versions compared to the *inference model* and only contains the layers necessary for training (regression and classification values). If you wish to do inference on a model (perform object detection on an image), you need to convert the trained model to an inference model. This is done as follows:

```shell
# Running directly from the repository:
keras_retinanet/bin/convert_model.py /path/to/training/model.h5 /path/to/save/inference/model.h5

```




## Debugging
Creating your own dataset does not always work out of the box. There is a [`debug.py`](https://github.com/fizyr/keras-retinanet/blob/master/keras_retinanet/bin/debug.py) tool to help find the most common mistakes.

Particularly helpful is the `--annotations` flag which displays your annotations on the images from your dataset. Annotations are colored in green when there are anchors available and colored in red when there are no anchors available. If an annotation doesn't have anchors available, it means it won't contribute to training. It is normal for a small amount of annotations to show up in red, but if most or all annotations are red there is cause for concern. The most common issues are that the annotations are too small or too oddly shaped (stretched out).




## Model evaluation

The script `evaluate.py` can be used to evaluate a model.
It computes the mAP on the whole test set and other useful metrics.
With the argument `--save-path`, the images with the true and predicted bouding boxes can be saved in a specific folder.

```python
csv_parser = subparsers.add_parser('csv')
csv_parser.add_argument('annotations', help='Path to CSV file containing annotations for evaluation.')
csv_parser.add_argument('classes', help='Path to a CSV file containing class label mapping.')

parser.add_argument('model',              help='Path to RetinaNet model.')
parser.add_argument('--convert-model',    help='Convert the model to an inference model (ie. the input is a training model).', action='store_true')
parser.add_argument('--backbone',         help='The backbone of the model.', default='resnet50')
parser.add_argument('--gpu',              help='Id of the GPU to use (as reported by nvidia-smi).')
parser.add_argument('--score-threshold',  help='Threshold on score to filter detections with (defaults to 0.05).', default=0.05, type=float)
parser.add_argument('--iou-threshold',    help='IoU Threshold to count for a positive detection (defaults to 0.5).', default=0.5, type=float)
parser.add_argument('--max-detections',   help='Max Detections per image (defaults to 100).', default=100, type=int)
parser.add_argument('--save-path',        help='Path for saving images with detections (doesn\'t work for COCO).')
parser.add_argument('--image-min-side',   help='Rescale the image so the smallest side is min_side.', type=int, default=800)
parser.add_argument('--image-max-side',   help='Rescale the image if the largest side is larger than max_side.', type=int, default=1333)
parser.add_argument('--config',           help='Path to a configuration parameters .ini file (only used with --convert-model).')
```

The command we used to evaluate our model is the following :
```bash
keras_retinanet/bin/evaluate.py --save-path ~/Documents/NumeriCube/eurosilicone/dataset/resized/mixed/results csv ../ann_test.csv ../class_mapping.csv ../models/3rd/resnet50_csv_07_inf.h5
```

The models are evaluated on an holdout Test set.
### Current results of the model :

mAP on Test Set : 96,6%
