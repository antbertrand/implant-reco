import os
import random

import keras
from keras_preprocessing.image import ImageDataGenerator
from keras.applications.resnet50 import ResNet50, preprocess_input

import numpy as np

# Import all the Keras machinery we need
from keras import applications
from keras import optimizers
from keras.preprocessing import image
from keras.models import Model
from keras.layers import Dense, GlobalAveragePooling2D
from keras import backend as K
from keras import metrics


DATA_DIR = '/storage/eurosilicone/ds_rotated/'



HEIGHT = 224
WIDTH = 224
batch_size = 4


# data generator for training set
train_datagen = ImageDataGenerator(
    rescale = 1./255)

# data generator for test set
test_datagen = ImageDataGenerator(rescale = 1./255)

# generator for reading train data from folder
train_generator = train_datagen.flow_from_directory(
    DATA_DIR+'train',
    target_size = (224, 224),
    color_mode = 'rgb',
    batch_size = batch_size,
    class_mode = 'categorical')

# generator for reading validation data from folder
validation_generator = test_datagen.flow_from_directory(
    DATA_DIR+'val',
    target_size = (224, 224),
    color_mode = 'rgb',
    batch_size = batch_size,
    class_mode = 'categorical')

# generator for reading test data from folder
test_generator = test_datagen.flow_from_directory(
    DATA_DIR+'test',
    target_size = (224, 224),
    color_mode = 'rgb',
    batch_size = 1,
    class_mode = 'categorical',
    shuffle = False)





def angle_difference(x, y):
    """
    Calculate minimum difference between two angles.
    """
    return 180 - abs(abs(x - y) - 180)

def angle_error(y_true, y_pred):
    """
    Calculate the mean diference between the true angles
    and the predicted angles. Each angle is represented
    as a binary vector.
    """
    diff = angle_difference(K.argmax(y_true), K.argmax(y_pred))
    print(K.argmax(y_true), K.argmax(y_pred))
    return K.mean(K.cast(K.abs(diff), K.floatx()))





# SOLUTION 3 InceptionV3 simple sans préentrainement du top classifier
from keras.optimizers import Adam, SGD, Adadelta
from keras.layers import Dense, Dropout, Flatten, Input
from keras.layers import Conv2D, MaxPooling2D
from keras.models import Model


model_name = 'rotnet_chip_resnet50'

# number of classes
nb_classes = 360
# input image shape
input_shape = (224, 224, 3)

# load base model
base_model = ResNet50(weights='imagenet', include_top=False,
                  input_shape=input_shape)

# append classification layer
x = base_model.output
x = Flatten()(x)
final_output = Dense(nb_classes, activation='softmax', name='fc360')(x)

# create the new model
model = Model(inputs=base_model.input, outputs=final_output)

model.summary()

# model compilation
model.compile(loss='categorical_crossentropy',
          optimizer=Adam(lr = 1e-5),
          metrics=[angle_error])




'''

# set the first 25 layers (up to the last conv block)
# to non-trainable (weights will not be updated)
for layer in model.layers[:10]:
    layer.trainable = False

'''



from keras.callbacks import ModelCheckpoint, EarlyStopping, TensorBoard, ReduceLROnPlateau
from time import time


# training parameters
batch_size = 4
nb_epoch = 200
n_train_samples = len(train_generator) * batch_size
n_val_samples = len(validation_generator) * batch_size


output_folder = '/artifacts/models'

if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# callbacks
monitor = 'val_angle_error'
checkpointer = ModelCheckpoint(
    filepath=os.path.join(output_folder, model_name + '.hdf5'),
    monitor=monitor,
    save_best_only=True
)
reduce_lr = ReduceLROnPlateau(monitor=monitor, factor = 0.3, patience=3)
early_stopping = EarlyStopping(monitor=monitor, patience=5)
tensorboard = TensorBoard(log_dir='/artifacts/logs/{}'.format(time()))

#K.clear_session()

history = model.fit_generator(
    train_generator,
    steps_per_epoch=n_train_samples // batch_size,
    epochs=nb_epoch,
    validation_data=validation_generator,
    validation_steps=n_val_samples // batch_size,
    verbose = 2,
    callbacks=[checkpointer, tensorboard])
