import keras
import math
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout, Flatten, Conv2D, MaxPooling2D, BatchNormalization
from keras.datasets import cifar10
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import EarlyStopping
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

# Data Augmentation
train_generator = ImageDataGenerator(rotation_range=2, horizontal_flip=True, zoom_range=.1)
val_generator = ImageDataGenerator(rotation_range=2, horizontal_flip=True, zoom_range=.1)
test_generator = ImageDataGenerator(rotation_range=2, horizontal_flip=True, zoom_range=.1)

train_generator.fit(x_train)
val_generator.fit(x_val)
test_generator.fit(x_test)

# Define the parameters
batch_size = 100
epochs = 100
learning_rate = 0.001
steps_per_epoch =  math.ceil(len(x_train) / batch_size)
np.random.seed(20211223)

# Instantiation
AlexNet = Sequential()

# 1st Conv Layer
AlexNet.add(Conv2D(filters=96, input_shape=(32, 32, 3), kernel_size=11, strides=4, padding='same'))
AlexNet.add(BatchNormalization())
AlexNet.add(Activation('relu'))
AlexNet.add(MaxPooling2D(pool_size=2, strides=2, padding='same'))

# 2nd Conv Layer
AlexNet.add(Conv2D(filters=256, kernel_size=5, strides=1, padding='same'))
AlexNet.add(BatchNormalization())
AlexNet.add(Activation('relu'))
AlexNet.add(MaxPooling2D(pool_size=2, strides=2, padding='same'))

# 3rd Conv Layer
AlexNet.add(Conv2D(filters=384, kernel_size=3, strides=1, padding='same'))
AlexNet.add(BatchNormalization())
AlexNet.add(Activation('relu'))

# 4th Conv Layer
AlexNet.add(Conv2D(filters=384, kernel_size=3, strides=1, padding='same'))
AlexNet.add(BatchNormalization())
AlexNet.add(Activation('relu'))

# 5th Conv Layer
AlexNet.add(Conv2D(filters=256, kernel_size=3, strides=1, padding='same'))
AlexNet.add(BatchNormalization())
AlexNet.add(Activation('relu'))
AlexNet.add(MaxPooling2D(pool_size=2, strides=2, padding='same'))

# FC Layer
AlexNet.add(Flatten())
AlexNet.add(Dense(4096))
AlexNet.add(BatchNormalization())
AlexNet.add(Activation('relu'))
AlexNet.add(Dropout(0.4))

# FC2 Layer
AlexNet.add(Dense(4096))
AlexNet.add(BatchNormalization())
AlexNet.add(Activation('relu'))
AlexNet.add(Dropout(0.4))

# FC3 Layer
AlexNet.add(Dense(1000))
AlexNet.add(BatchNormalization())
AlexNet.add(Activation('relu'))
AlexNet.add(Dropout(0.4))

# Output Layer
AlexNet.add(Dense(10))
AlexNet.add(BatchNormalization())
AlexNet.add(Activation('softmax'))

AlexNet.summary()
AlexNet.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# Traing model
AlexNet.fit(x=train_generator.flow(x_train, y_train, batch_size=batch_size),
            batch_size=batch_size, verbose=1, epochs=epochs, validation_data=(x_val, y_val),
            steps_per_epoch=steps_per_epoch)
scores = AlexNet.evaluate(x_test, y_test, batch_size=batch_size, verbose=1)
print('Test Accuracy: ', scores[1])
