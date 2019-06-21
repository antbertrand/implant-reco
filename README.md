# Gcaesthetics Implantbox

A project that scans a serial number on a breast implant


## Interesting entry points

main.py : complete app with detection until text

eurosilicone-reader.py : acquisition app (saves chips to /var/eurosilicone/acqusitions)

uploader.py : will upload acquisitions to Azure (use this in a CRON)


## Getting Started

There are 4 distinct steps in the processing of an input image :
- step1: Chip Detection <br/>
Gets the chip localization with a RetinaNet.
Done in chip_detector.py

- step2: Circle Refiner <br/>
Gets a more precise localization of the chip with the hough circle method.
Done in better_circle.py

- step3: Angle Corrector <br/>
Corrects the orientation of the implant in order to have the text upright. Uses a RotNet ( 360 class classifier)
Done in orientation_fixer.py

- step4: Caracter Detector <br/>
Detects the different types of caracters ( 37 : 26 letters + 10 numbers + '/' + '-' ) with a RetinaNet.
Done in caracter_detector.py


Those four steps have to be applied on an input image in order to detect the caracters on the implant. This is done in production in the *eurosilicone_reader.py* ( reads the image from the camera and processes it).
It is easier to test it is with *eurosilicone_reader_v2.py* wichs will read the image from the folder *tests/input* and process them, saving on the way the images at each steps.


## Training with additionnal data ?

If more data is accessible, training the models with this new data could lead to big improvements in performance. <br/>
You can train the models in step1, step3 and step4 again. The process of training for each step is specific and described in the READMEs in their corresponding folder.


#### DL the dataset

The dataset we used is stored on Azure, the URL of the blob container is the following :
*https://eurosilicone.blob.core.windows.net/acquisitions*

This will give you a dataset of all the images we took of implants. However some of them are not usable because of their quality or for other various reasons. The final dataset we used is in the blob container *https://eurosilicone.blob.core.windows.net/dsstep1*. The three splits train / val / test were not done randomly because we had to make sure several images of the same implant didn't end up in the same split.

Split                      | Train    |     Validation   |     Test   
------------------------------|-------------|-------------|-------------------|
Nb of images              |   256         |   52     |   40           





### Prerequisites

Things needed to run the detection :

* [Keras](https://www.pyimagesearch.com/2016/11/14/installing-keras-with-tensorflow-backend/)
* [OpenCV2](https://pypi.org/project/opencv-python/)
