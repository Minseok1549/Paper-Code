import keras, os
import numpy as np
from keras.layers import Conv2D, MaxPooling2D, Dense, Flatten
from keras.models import Sequential
from keras.preprocessing.image import ImageDataGenerator
from keras.datasets import cifar10
from keras.callbacks import ModelCheckpoint, EarlyStopping
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
(x_train, y_train), (x_test, y_test) = cifar10.load_data()
# input image dimensions (Assuming input image pixel square matrix)
image_size = x_train.shape[1]

# normalize (for speed up learning)
x_train = x_train.astype('float32') / 255
x_test = x_test.astype('float32') / 255

x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.3)

y_train = to_categorical(y_train)
y_val = to_categorical(y_val)
y_test = to_categorical(y_test)

# Define the parameters
batch_size = 100
epochs = 100

# Data Augmentation
train_generator = ImageDataGenerator(rotation_range=2, horizontal_flip=True, zoom_range=.1)
val_generator = ImageDataGenerator(rotation_range=2, horizontal_flip=True, zoom_range=.1)
test_generator = ImageDataGenerator(rotation_range=2, horizontal_flip=True, zoom_range=.1)
train_generator.fit(x_train)
val_generator.fit(x_val)
test_generator.fit(x_test)

VGG16 = Sequential()
VGG16.add(Conv2D(input_shape=(32,32,3), filters=64, kernel_size=3, padding='same', activation='relu'))
VGG16.add(Conv2D(filters=64, kernel_size=3, padding='same', activation='relu'))
VGG16.add(MaxPooling2D(pool_size=2, strides=2))

VGG16.add(Conv2D(filters=128, kernel_size=3, padding='same', activation='relu'))
VGG16.add(Conv2D(filters=128, kernel_size=3, padding='same', activation='relu'))
VGG16.add(MaxPooling2D(pool_size=2, strides=2))

VGG16.add(Conv2D(filters=256, kernel_size=3, padding='same', activation='relu'))
VGG16.add(Conv2D(filters=256, kernel_size=3, padding='same', activation='relu'))
VGG16.add(MaxPooling2D(pool_size=2, strides=2))

VGG16.add(Conv2D(filters=512, kernel_size=3, padding='same', activation='relu'))
VGG16.add(Conv2D(filters=512, kernel_size=3, padding='same', activation='relu'))
VGG16.add(Conv2D(filters=512, kernel_size=3, padding='same', activation='relu'))
VGG16.add(MaxPooling2D(pool_size=2, strides=2))

VGG16.add(Conv2D(filters=512, kernel_size=3, padding='same', activation='relu'))
VGG16.add(Conv2D(filters=512, kernel_size=3, padding='same', activation='relu'))
VGG16.add(Conv2D(filters=512, kernel_size=3, padding='same', activation='relu'))
VGG16.add(MaxPooling2D(pool_size=2, strides=2))

VGG16.add(Flatten())
VGG16.add(Dense(units=4096, activation='relu'))
VGG16.add(Dense(units=4096, activation='relu'))
VGG16.add(Dense(units=10, activation='softmax'))

VGG16.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
VGG16.summary()

checkpoint = ModelCheckpoint("vgg16_1.h5", monitor='val_acc', verbose=1, save_best_only=True, save_weights_only=False, mode='auto', period=1)
early = EarlyStopping(monitor='val_acc', min_delta=0, patience=20, verbose=1, mode='auto')

VGG16.fit(x=train_generator.flow(x_train, y_train, batch_size=batch_size), batch_size=batch_size, verbose=1, epochs=epochs, validation_data=val_generator.flow(x_val, y_val, batch_size=batch_size), steps_per_epoch=100)
scores = VGG16.evaluate(x_test, y_test, batch_size=batch_size, verbose=1)
print('Test Accuracy: ', scores[1])
