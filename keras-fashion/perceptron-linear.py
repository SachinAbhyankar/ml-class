import numpy
from keras.datasets import fashion_mnist
import keras
import numpy as np
from keras.models import Sequential
from keras.layers import GlobalAveragePooling2D, MaxPooling2D, Input, Dense, Flatten, Dropout, BatchNormalization
from keras.utils import np_utils
import wandb
from wandb.keras import WandbCallback
from keras.layers import Conv2D, BatchNormalization, Activation
from keras.layers import AveragePooling2D, Input, Flatten
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint, LearningRateScheduler
from keras.callbacks import ReduceLROnPlateau
from keras.preprocessing.image import ImageDataGenerator
from keras.regularizers import l2
from keras import backend as K
from keras.models import Model
from keras.preprocessing.image import ImageDataGenerator

# logging code
run = wandb.init()
config = run.config
config.epochs = 10

# load data
(X_train, y_train), (X_test, y_test) = fashion_mnist.load_data()

img_width = X_train.shape[1]
img_height = X_train.shape[2]
labels =["T-shirt/top","Trouser","Pullover","Dress",
    "Coat","Sandal","Shirt","Sneaker","Bag","Ankle boot"]

# one hot encode outputs
y_train = np_utils.to_categorical(y_train)
y_test = np_utils.to_categorical(y_test)

X_train= np.expand_dims(X_train, axis=3)
X_test= np.expand_dims(X_test, axis=3)

X_train = X_train/255
X_test = X_test/255

num_classes = y_train.shape[1]
image_shape1=(img_width, img_height,1)
# create model
model=Sequential()
model.add(Conv2D(32, (3, 3), padding='same',
                 activation='relu',
                 input_shape=image_shape1))
model.add(Conv2D(32, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.4))
model.add(BatchNormalization())

model.add(Conv2D(64, (3, 3), padding='same', activation='relu'))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
#model.add(BatchNormalization())

model.add(Conv2D(128, (3, 3), padding='same', activation='relu'))
model.add(Conv2D(128, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
#model.add(BatchNormalization())

#model.add(Flatten())
model.add(GlobalAveragePooling2D())
model.add(Dense(512,activation='relu'))
#model.add(BatchNormalization())
model.add(Dropout(0.5))

model.add(Dense(100,activation='relu'))
model.add(Dropout(0.3))
#model.add(Dense(20,activation='relu'))
#model.add(Dropout(0.1))
#model.add(BatchNormalization())

#model.add(Flatten())
#model.add(Dense(128, activation='relu'))
#model.add(Dropout(0.3))
model.add(Dense(num_classes, activation='softmax'))
model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.Adadelta(),
              metrics=['accuracy'])

# Fit the model
model.fit(X_train, y_train, epochs=config.epochs, validation_data=(X_test, y_test),
                    callbacks=[WandbCallback(data_type="image", labels=labels)])



