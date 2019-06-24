"""Script that will train a RotNet on our dataset. Paths are made to be used on Paperspace"""


import os
from time import time


# Import all the Keras machinery we need

from keras import backend as K
from keras_preprocessing.image import ImageDataGenerator
from keras.applications.resnet50 import ResNet50
from keras.models import Model
from keras.optimizers import Adam
from keras.layers import Dense, Flatten
from keras.callbacks import (
    ModelCheckpoint,
    EarlyStopping,
    TensorBoard,
    ReduceLROnPlateau,
)


DATA_DIR = "/storage/eurosilicone/ds_rotated/"


HEIGHT = 224
WIDTH = 224
BATCH_SIZE = 32


# data generator for training set
train_datagen = ImageDataGenerator(
    featurewise_center=True,
    featurewise_std_normalization=True,
    rescale=1 / 255
)

# data generator for test set
test_datagen = train_datagen
# generator for reading train data from folder
train_generator = train_datagen.flow_from_directory(
    DATA_DIR + "train",
    target_size=(224, 224),
    color_mode="rgb",
    batch_size=BATCH_SIZE,
    class_mode="categorical",
)

# generator for reading validation data from folder
validation_generator = test_datagen.flow_from_directory(
    DATA_DIR + "val",
    target_size=(224, 224),
    color_mode="rgb",
    batch_size=BATCH_SIZE,
    class_mode="categorical",
)

# generator for reading test data from folder
test_generator = test_datagen.flow_from_directory(
    DATA_DIR + "test",
    target_size=(224, 224),
    color_mode="rgb",
    batch_size=1,
    class_mode="categorical",
    shuffle=False,
)


# ================== Angle_error metric =========================
# Functions to compute the angle_error metric


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


# ================== Model creation =========================
# Model = Resnet50 without top classifier / without freezing layers




model_name = "rotnet_chip_resnet50_v2"

# number of classes
nb_classes = 360
# input image shape
input_shape = (224, 224, 3)

# load base model
BASE_MODEL = ResNet50(weights="imagenet", include_top=False, input_shape=input_shape)

# append classification layer
X = BASE_MODEL.output
X = Flatten()(X)
final_output = Dense(nb_classes, activation="softmax", name="fc360")(X)

# create the new model
model = Model(inputs=BASE_MODEL.input, outputs=final_output)

model.summary()

# model compilation
model.compile(
    loss="categorical_crossentropy", optimizer=Adam(lr=1e-5), metrics=[angle_error]
)


# ================== Training model =========================





# training parameters
BATCH_SIZE = 8
nb_epoch = 50
n_train_samples = len(train_generator) * BATCH_SIZE
n_val_samples = len(validation_generator) * BATCH_SIZE


output_folder = "/artifacts/models/"

if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# Callback functions

monitor = "val_angle_error"

# Always saving the model with the best val_angle_error
checkpointer = ModelCheckpoint(
    filepath=os.path.join(output_folder, model_name + ".hdf5"),
    monitor=monitor,
    verbose=2,
    mode="min",
    save_best_only=True,
)
reduce_lr = ReduceLROnPlateau(monitor=monitor, factor=0.3, patience=3)
early_stopping = EarlyStopping(monitor=monitor, patience=5)
tensorboard = TensorBoard(log_dir="/artifacts/logs/{}".format(time()))


history = model.fit_generator(
    train_generator,
    steps_per_epoch=n_train_samples // BATCH_SIZE,
    epochs=nb_epoch,
    validation_data=validation_generator,
    validation_steps=n_val_samples // BATCH_SIZE,
    verbose=2,
    callbacks=[checkpointer, tensorboard],
)
