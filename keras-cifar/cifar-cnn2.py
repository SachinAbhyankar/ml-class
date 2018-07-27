from __future__ import print_function
import keras
from keras.datasets import cifar10
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D

import os
import wandb
from wandb.wandb_keras import WandbKerasCallback

run = wandb.init()
config = run.config
class_names = ['airplane','automobile','bird','cat','deer',
               'dog','frog','horse','ship','truck']
num_classes = 10
# Training parameters
batch_size = 32  # orig paper trained all networks with batch_size=128
epochs = 200
data_augmentation = True
num_classes = 10

main_model = applications.ResNet50(include_top=False)
for layer in main_model.layers:
    layer.trainable=False

main_model = main_model(inp)
main_out = Flatten()(main_model)
main_out = Dense(512, activation='relu', name='fcc_0')(main_out)
main_out = Dense(1, activation='softmax', name='class_id')(main_out)

model = Model(input=inp, output=main_out)
model._is_graph_network = False

opt = keras.optimizers.SGD(lr=config.learn_rate)

# Let's train the model using RMSprop
model.compile(loss='categorical_crossentropy',
              optimizer=opt,
              metrics=['accuracy'])

x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255

# data prep
train_datagen = ImageDataGenerator(
        featurewise_center=False,
        samplewise_center=False,
        samplewise_std_normalization=False,
        zca_whitening=False,
        rotation_range=30,
        width_shift_range=0.2,
        height_shift_range=0.2,
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        vertical_flip=False,
        fill_mode='nearest')
test_datagen = ImageDataGenerator(rescale=1. / 255)

train_generator = train_datagen.flow_from_directory(
    x_train,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='categorical')

validation_generator = test_datagen.flow_from_directory(
    x_test,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='categorical')
val_data = validation_generator.next()

#datagen = ImageDataGenerator(
#        width_shift_range=0.1)


#datagen.fit(x_train)
    # Fit the model on the batches generated by datagen.flow().
model.fit_generator(train_generator,
                        steps_per_epoch=32 // config.batch_size,
                        epochs=config.epochs,
                        validation_data= val_data,
                        workers=4,
                        callbacks=[WandbKerasCallback(data_type="image", labels=class_names)]
   )

