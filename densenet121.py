from tensorflow.keras.layers import Dense, Conv2D, BatchNormalization
from tensorflow.keras.layers import MaxPooling2D, AveragePooling2D, GlobalAveragePooling2D
from tensorflow.keras.layers import Input, Flatten, Dropout
from tensorflow.keras.layers import Concatenate, Activation
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, Callback
from tensorflow.keras.callbacks import LearningRateScheduler
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Model
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.utils import plot_model
from tensorflow.keras.utils import to_categorical
from tensorflow.keras import backend as K
import cv2
import os
import numpy as np
import math
import tensorflow as tf
# resize cifar10 : memory problem spilt training data
(x_train, y_train), (x_test, y_test) = cifar10.load_data()
x_train = x_train[:10000, :, :, :]
y_train = y_train[:10000, :]
target_shape = (224,224)
def _resize_image(image, target):
    return cv2.resize(image, dsize=(224, 224), interpolation=cv2.INTER_LINEAR)

x_train = [_resize_image(image=i, target=target_shape) for i in x_train]
x_train = np.stack(x_train, axis=0)
x_train = x_train.astype('float32') / 255
y_train = to_categorical(y_train, 10)
# y_train = to_categorica(y_train, [-1])
print(y_train.shape)


def Conv_Block(x, growth_rate, activation='relu'):
    x_l = BatchNormalization()(x)
    x_l = Activation(activation)(x_l)
    x_l = Conv2D(growth_rate*4, (1, 1), padding='same', kernel_initializer='he_normal')(x_l)

    x_l = BatchNormalization()(x_l)
    x_l = Activation(activation)(x_l)
    x_l = Conv2D(growth_rate, (3, 3), padding='same', kernel_initializer='he_normal')(x_l)

    x = Concatenate()([x, x_l])

    return x

def Dense_Block(x, layers, growth_rate=32):
    for i in range(layers):
        x = Conv_Block(x, growth_rate)
    return x

def Transition_Layer(x, compression_factor=0.5, activation='relu'):
    # 분할 수행 시 입력 값의 채널 수를 반환받아 compression_factor 를 곱하는 과정 --> reduced channel
    reduced_filters = int(K.int_shape(x)[-1] * compression_factor)

    x = BatchNormalization()(x)
    x = Activation(activation)(x)
    x = Conv2D(reduced_filters, (1, 1), padding='same', kernel_initializer='he_normal')(x)

    x = AveragePooling2D((2, 2), padding='same', strides=2)(x)

    return x

layers_in_block = {'DenseNet-121' : [6, 12, 24, 16],
                   'DenseNet-169' : [6, 12, 32, 32],
                   'DenseNet-201' : [6, 12, 48, 32],
                   'DenseNet-265' : [6, 12, 64, 48]}

base_growth_rate = 32

def DenseNet(model_input, classes, densenet_type='DenseNet-121'):
    x = Conv2D(base_growth_rate*2, (7, 7), padding='same', strides=2, kernel_initializer='he_normal')(model_input) # (224x224x3) --> (112x112x64)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    x = MaxPooling2D((3, 3), padding='same', strides=2)(x) # (112x112x64) --> (56x56x64)

    x = Dense_Block(x, layers_in_block[densenet_type][0], base_growth_rate)
    x = Transition_Layer(x, compression_factor=0.5)
    x = Dense_Block(x, layers_in_block[densenet_type][1], base_growth_rate)
    x = Transition_Layer(x, compression_factor=0.5)
    x = Dense_Block(x, layers_in_block[densenet_type][2], base_growth_rate)
    x = Transition_Layer(x, compression_factor=0.5)
    x = Dense_Block(x, layers_in_block[densenet_type][3], base_growth_rate)
    x = Transition_Layer(x, compression_factor=0.5)

    x = GlobalAveragePooling2D()(x)

    model_output = Dense(classes, activation='softmax', kernel_initializer='he_normal')(x)
    model = Model(model_input, model_output, name=densenet_type)

    return model

class LearningRateSchedule(Callback):
    def __init__(self, selected_epochs=[]):
        self.selected_epochs = selected_epochs

    def on_epoch_end(self, epoch, logs=None):
        if (epoch+1) in self.selected_epochs:
            lr = K.get_value(self.model.optimizer.lr)
            K.set_value(self.model.optimizer.lr, lr*0.1)

input_shape = (224, 224, 3)

model_input = Input(shape=input_shape)

model = DenseNet(model_input, 10, 'DenseNet-121')
optimizer = SGD(lr=0.1, decay=1e-3, momentum=0.9, nesterov=True)
callbacks_list = [LearningRateSchedule([30, 60])]
model.compile(optimizer, loss='categorical_crossentropy', metrics=[tf.keras.metrics.CategoricalAccuracy()])
model.summary()
history = model.fit(x_train, y_train, batch_size=256, epochs=90, validation_split=0.2, callbacks=callbacks_list)
