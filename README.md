# Gcaesthetics Implantbox

A project that scans a serial number on a breast implant


## Interesting entry points

main.py : complete app with detection until text

eurosilicone-reader.py : acquisition app (saves chips to /var/eurosilicone/acqusitions)

uploader.py : will upload acquisitions to Azure (use this in a CRON)


## Getting Started

There are 4 distinct steps in the processing of an input image :
- step1: Chip Detection <br/>
Gets the chip localization with a RetinaNet chip_detector
Done in chip_detector.py

- step2: Circle Refiner <br/>
Gets a more precise localization of the chip with the hough circle method.
Done in better_circle.py

- step3: Angle Corrector <br/>
Corrects the orientation of the implant in order to have the text upside down. Uses a RotNet ( 360 class classifier)
Done in orientation_fixer.py

- step4: Caracter Detector <br/>
Detects the different types of caracters ( 37 : 26 letters + 10 numbers + '/' + '-' )
Done in caracter_detector.py


Those four steps have to be applied on an input image in order to detect the caracters on the implant. This is done in production in the *eurosilicone_reader.py* ( reads the image from the camera and processes it).
It is easier to test it is with *eurosilicone_reader_v2.py* wichs will read the image from a folder and process them, saving on the way the images got after each steps.


## Training with additionnal data ?

If more data is accessible, training the models with this new data could lead to big improvements in performance. <br/>
You can train the models in step1, step3 and step4 again. The process of training for each step is specific and described in READMEs in their corresponding folder.


#### DL the dataset

The dataset we used is stored on Azure, the URL of the blob container is the following :
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

Install with pip install ./step1_chip_detection/retinanet/


<br /><br />

## Testing

To test the network use the chip_detector.py file.

## Usage

To train it on a custom dataset, a CSV file can be used as a way to pass the data.
See below for more details on the format of these CSV files.# gcaesthetics-implantbox
A project that scans a serial number on a breast implant
