# Orientation Fix

The goal here is to correct the orientation of a chip, so that on the text will be upright.

## Getting Started

The architecture used is called a RotNet.  <br />
It is taken from this repository: [link](https://github.com/d4nst/RotNet)<br />
In the original implementation, the images are given upright and the generator creates different random rotations of those images at each epoch.
However, because the network only sees one rotation of each images per epoch, it didn't quite manage to converge efficiently. With some hindsight, increasing the number of steps per epoch would have forced the generator to send multiple rotations of the same image in one epoch and should have helped on the convergence.
But at that time we took the decision of simply creating the 360 rotations of each images, and then feeding them to the network as 360 different classes. (not a great decision wich will not scale well with more images because of the need to store on disk 360 * nb_images).
In fine, we end up only with a classic classification problem, we only kept the architecture of the RotNet not the generator ( " the "clever" part) wich was creating the rotations and thus changing the classes at each step.


To test the detection on some images, clone this repo on your machine.

## Using it


## Re-training it

### 1. Dataset preparation

#### Images
The dataset we will use is the dataset obtained after step2, wich is the circle refiner.
You can also directly download the images from Azure, from this container:
*https://eurosilicone.blob.core.windows.net/dsstep3*

In order to use the same commands that will be used later, you should place the 3 splits of the dataset, passed through step2, in *dataset/ds_step3_orientation_fixer* in their corresponding folder, in *img_train* / *img_val* / *img_test*.     
<br />



#### Angle label

We need to correct the orientation of each image before feeding it to the network.
In the actual code doing it, it first corrects it so that all image are upright, than another part rotates it in the 360 different rotations. Rotating an image involves interpolation wich leads to some loss of info. An improve would be to change the code so it reads the true rotation, stores it in the corresponding class and then rotates from this rotated image into the 358 other rotations. ( 1 rotation instead of 2)


We did the labelization using the vector tool of supervisely. The vector must be oriented vertically, with one point near the center and the other marking the top of the chip.

The code in *angle_correction.ipynb* will rotate it back to the upright position.

The code in *create_dataset.py* will do the rotations and saves the images in the 360 fodlers.




<br />

### 2. Training
<br />
The model is trained using a resnet50 network.
You have to run the script directly from inside the folder `retinanet`.

Every parameters is given through the command line as arguments. The following lists all the different arguments that can be given to the train.py script :


### 3. Debugging (if needed)
<br />
If things do not work as expected there is a [`debug.py`](https://github.com/fizyr/keras-retinanet/blob/master/keras_retinanet/bin/debug.py) tool to help find the most common mistakes.

Particularly helpful is the `--annotations` flag which displays your annotations on the images from your dataset. Annotations are colored in green when there are anchors available and colored in red when there are no anchors available. If an annotation doesn't have anchors available,* it means it won't contribute to training. It is normal for a small amount of annotations to show up in red, but if most or all annotations are red there is cause for concern. The most common issues are that the annotations are too small or too oddly shaped (stretched out).

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
