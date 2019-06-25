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

##### Preprocessing on the images

The preprocessing to add to any input images using the actual model is the following:
- equalizing its histogram
- rescaling the pixel range from [0, 255] to [0, 1]
- resizing it to (224, 224)

<br /><br />

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

<br /><br />


The code in *angle_correction.ipynb* will rotate it back to the upright position.

The code in *create_dataset.py* will do the rotations and save the images in the 360 fodlers.




<br />

### 2. Training
<br />
The model is trained using a resnet50 network.
You have to run the script in *train_angle_resnet.py*. The only thing you will have to modify is the *DATA_DIR* and the *output_folder*.

A custom metric called angle_error will compute the error on the predicted angle. The CheckPointer uses this metric , and will save the model at each epoch where the angle_error is minimum.


### 3. Testing

You can use the *classify_test.ipynb* notebook, to test the trained network on some images. It will plot the input image and its corrected rotation.


### 4. Debugging (if needed)
<br />
If things do not work as expected here are listed some issues that may happen :

- Problem with the rotation of small images. If you kept the code as it is it shoudln't happen but it is still interesting to notice that if you rescale the images to 224x224 before doing all the rotations, the model will learn to recognize some artefacts or the pixel structure that appear with the rotation. On bigger images, this issue doesn't appear.

- In this model, we rescaled the input images to [0; 1]. You must not forget to divide by 255 the pixel values of any input image, for inference or for training. You can also decide to remove that as it is not all useful, and therefore have to retrain a model without it.

<br />
### 5. Evaluating the new models
<br />

To evaluate the models you will use the img_evaluate folder that you can create using the last cell of *angle_correction.ipynb*.

The last cell of *classify_test.ipynb* will compute the mean of angle_error on a dataset (average deviation) and the nb of images classified with less than a 20 degree error. It will also give the mean after removing the extremes ( diff > 20°).

The models are chosen on their performances on the Validation set.

Model version               | angle_error   |  angle_error (diff < 20°) | nb (diff < 20°)
------------------------------|-------------|-------------|-------------------|
rotnet_step3_resnet50_20190606005100.hdf5   |   10.56 °   |   3.58° |  48/52     


The models are then evaluated on an holdout Test set.

Model version               | angle_error   |  angle_error (diff < 20°) | nb (diff < 20°)
------------------------------|-------------|-------------|-------------------|
rotnet_step3_resnet50_20190606005100.hdf5   |   15.05 °   |   2.87° |  33/39     



<br />
### 6. Creating the dataset for the next step
<br />

The step4 will need the corrected images as an input.
To create it, use the first cells of *classify_test.ipynb*.
