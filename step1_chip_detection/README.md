# Chip detection

The goal here is to detect precisly where the chip is on the implant.

## Getting Started

The detector used is RetinaNet (Lin, Goyal et al. 2017) <br />
To learn more about the way it works : [link](https://blog.zenggyu.com/en/post/2018-12-05/retinanet-explained-and-demystified/)<br />
The implementation used is an implementation of RetinaNet in Keras : [keras_retinanet](https://github.com/fizyr/keras-retinanet)




#### DL the dataset

The dataset used in this step are the raw images out of the camera.
It is stored on Azure, the URL of the blob container is the following :
*https://eurosilicone.blob.core.windows.net/dsstep1*

This will give you a dataset split in three : a train, validation and test set. The split has been done manually, to make sure very similar images do not end up in same splits, which would distort the results.

Split                      | Train    |     Validation   |     Test   
------------------------------|-------------|-------------|-------------------|
Nb of images              |   256         |   52     |   40           

You should also have an annotation folder in which there are the info of the bouding boxes of the chip for all the images. This will  only be useful if you want to train a new network.

#### DL the models

We used a RetinaNet, once trained it needs to be converted to an inference model before being able to be tested on new images.

The inference model is stored on Azure :
*https://eurosilicone.blob.core.windows.net/weights/retinanet_step1_resnet50_20190605101500.h5*

The raw model can be used to continue training with the adding of new images. If needed is also stored on Azure :
*https://eurosilicone.blob.core.windows.net/weights/retinanet_step1_resnet50_inf_20190605101500.h5*

##### **In order to use the exact same commands showed later on, please place the downloaded models and dataset in their respecting folders of the repository.**

### Prerequisites

Some packages are needed in this part :

The following commands  will install everything that is needed :
```bash
pip install imutils
pip install cython
pip install scipy
pip install keras-resnet
pip install h5py
pip install keras
pip install numpy>=1.14
pip install opencv-python>=3.3.0
pip install pillow
pip install progressbar2
```

TODO: Check more precisely what is needed and for what. Also maybe give the requirements.txt of the virtualenv.

<br /><br />


## Using it

To test the network use the chip_detector.py file.
<br /><br />
## Re-training it

### 1. Dataset preparation

#### Images
The dataset we will use are the raw images (out of the camera). The way to download it is described previously.
In order to use the same commands that will be used later, you should place the 3 splits of the dataset, in *ds_step1_chip_detector* in their corresponding folder, *img_train* / *img_val* / *img_test*.     
<br />

#### Annotations

The `CSVGenerator` of RetinaNet provides an easy way to define our own datasets.
It uses two CSV files: one file containing annotations and one file containing a class name to ID mapping.

###### Annotations format
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

###### Class mapping format
The class name to ID mapping file should contain one mapping per line.
Each line should use the following format:
```
class_name,id
```
##### Annotation conversion from supervisely
You should have labelized the whole dataset. If you did it on supervisly, we need to convert them to the csv format described previously.

To do so, use the notebook creating_labels.ipynb in the ds_step4_caracter_detector folder. (TODO: do it cleanly in a independant .py file)


<br />

### 2. Training
<br />
The model is trained using the keras_retinanet/bin/train.py script.
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

The command we used to train the model on the macchiato is the following:

```shell
keras_retinanet/bin/train.py --batch-size=1 --lr=1e-4 --random-transform --compute-val-loss --steps=260 --weighted-average --snapshot-path ./snapshots csv /home/numericube/Documents/current_projects/gcaesthetics-implantbox/dataset/ds_step4_caracter_detector/ann_train.csv /home/numericube/Documents/current_projects/gcaesthetics-implantbox/dataset/ds_step4_caracter_detector/class_mapping.csv --val-annotations /home/numericube/Documents/current_projects/gcaesthetics-implantbox/dataset/ds_step4_caracter_detector/ann_val.csv
```


#### Converting a training model to inference model
The training procedure of `keras-retinanet` works with *training models*. These are stripped down versions compared to the *inference model* and only contains the layers necessary for training (regression and classification values). Before using the model to perform the detetion on an image, we need to convert the trained model to an inference model. This is done as follows:

```shell
# Running directly from the repository:
keras_retinanet/bin/convert_model.py /home/numericube/Documents/current_projects/gcaesthetics-implantbox/step4_letter_detection/models/model_name.h5 /home/numericube/Documents/current_projects/gcaesthetics-implantbox/step4_letter_detection/models/model_name_inf.h5 --no-class-specific-filter
```
The `--no-class-specific-filter` argument is important as it will let the nms filter remove overlapping bboxes from different classes, as in our case, two caracters will never be overlapping.

<br />


### 3. Debugging (if needed)
<br />
If things do not work as expected there is a [`debug.py`](https://github.com/fizyr/keras-retinanet/blob/master/keras_retinanet/bin/debug.py) tool to help find the most common mistakes.

Particularly helpful is the `--annotations` flag which displays your annotations on the images from your dataset. Annotations are colored in green when there are anchors available and colored in red when there are no anchors available. If an annotation doesn't have anchors available, it means it won't contribute to training. It is normal for a small amount of annotations to show up in red, but if most or all annotations are red there is cause for concern. The most common issues are that the annotations are too small or too oddly shaped (stretched out).

```shell
keras_retinanet/bin/debug.py --annotations csv /home/numericube/Documents/current_projects/gcaesthetics-implantbox/dataset/ds_step4_caracter_detector/ann_train_large.csv /home/numericube/Documents/current_projects/gcaesthetics-implantbox/dataset/ds_step4_caracter_detector/class_mapping.csv
```

<br />
### 4. Evaluating the new models
<br />
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
keras_retinanet/bin/evaluate2.py --score-threshold=0.48 --save-path /home/numericube/Documents/current_projects/gcaesthetics-implantbox/step5_evaluation/test_im csv /home/numericube/Documents/current_projects/gcaesthetics-implantbox/dataset/ds_step4_caracter_detector/ann_val.csv /home/numericube/Documents/current_projects/gcaesthetics-implantbox/dataset/ds_step4_caracter_detector/class_mapping.csv /home/numericube/Documents/current_projects/gcaesthetics-implantbox/step4_letter_detection/models/model_name_inf.h5
```

The models are chosen on their performances on the Validation set.

Model version               | mAP (weighted average)   |  mAP (average) |     Nb of codes read  
------------------------------|-------------|-------------|-------------------|
retinanet_step4_resnet50_inf_20190605101500             |   69.28%         |   52     |   18 / 51  


The models are then evaluated on an holdout Test set.

Model version               | mAP (weighted average)   |  mAP (average) |     Code   
------------------------------|-------------|-------------|-------------------|
Nb of images              |   256         |   52     |   40  
