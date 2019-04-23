# the code is implemented on Keras with tensorflow as backend
from keras.layers import Activation, Dropout, Flatten, Dense, Input
from keras.layers import Conv2D, MaxPooling2D, GlobalAveragePooling2D, AveragePooling2D, Cropping2D,BatchNormalization
from keras import optimizers
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import LearningRateScheduler, TensorBoard
from keras.regularizers import l2
from keras.models import Model
import os
from PIL import Image
import numpy as np
import keras
import time
import scipy.io as sio
import random
import h5py
import time
from scipy.misc import imresize
from numpy import *

batch_size = 2048
epochs = 50
iterations = 391
num_classes = 3
log_filepath = './sonar_dp_da'
weight_decay = 0.005

def get_crop_shape(target, refer):
    # width, the 3rd dimension
    cw = (target.get_shape()[2] - refer.get_shape()[2]).value
    assert (cw >= 0)
    if cw % 2 != 0:
        cw1, cw2 = int(cw/2), int(cw/2) + 1
    else:
        cw1, cw2 = int(cw/2), int(cw/2)
    # height, the 2nd dimension
    ch = (target.get_shape()[1] - refer.get_shape()[1]).value
    assert (ch >= 0)
    if ch % 2 != 0:
        ch1, ch2 = int(ch/2), int(ch/2) + 1
    else:
        ch1, ch2 = int(ch/2), int(ch/2)

    return (ch1, ch2), (cw1, cw2)

def build_model():
    # 37*37
    input_1 = Input(shape=(37, 37, 1))
    conv37_1 = Conv2D(16, (7, 7), padding='valid', activation='relu', kernel_initializer='he_normal', name='conv1')(input_1)
    #conv37_1  = BatchNormalization()(conv37_1 )
    pool37_1 = MaxPooling2D((2, 2), strides=(2, 2), padding='valid')(conv37_1)
    conv37_2 = Conv2D(32, (5, 5), padding='valid', activation='relu', kernel_initializer='he_normal', name='conv2')(pool37_1)
    #conv37_2  = BatchNormalization()(conv37_2 )
    pool37_2 = MaxPooling2D((2, 2), strides=(2, 2), padding='valid')(conv37_2)
    conv37_3 = Conv2D(32, (3, 3), padding='valid', activation='relu', kernel_initializer='he_normal', name='conv3')(pool37_2)
    #conv37_3  = BatchNormalization()(conv37_3 )
    # 25*25
    input_2 = Input(shape=(25, 25, 1))
    conv25_1 = Conv2D(16, (5, 5), padding='valid', activation='relu', kernel_initializer='he_normal', name='conv4')(input_2)
    #conv25_1  = BatchNormalization()(conv25_1 )
    pool25_1 = MaxPooling2D(pool_size=(2, 2))(conv25_1)
    conv25_2 = Conv2D(32, (5, 5), padding='valid', activation='relu', kernel_initializer='he_normal', name='conv5')(pool25_1)
    #conv25_2  = BatchNormalization()(conv25_2 )
    pool25_2 = MaxPooling2D(pool_size=(2, 2))(conv25_2)
    # 13*13
    input_3 = Input(shape=(13, 13, 1))
    conv13_1 = Conv2D(16, (3,3),padding='valid', activation = 'relu', kernel_initializer='he_normal',name = 'conv6')(input_3)
    #conv13_1  = BatchNormalization()(conv13_1)
    pool13_1 = MaxPooling2D(pool_size=(2, 2))(conv13_1)
    conv13_2 = Conv2D(32, (3,3),padding='valid', activation = 'relu', kernel_initializer='he_normal',name = 'conv7')(pool13_1)
    #conv13_2  = BatchNormalization()(conv13_2)

    conca = keras.layers.concatenate([conv37_3, pool25_2, conv13_2], axis=3,name = 'conca')
    F1 = Flatten()(conca)

    D1 = Dense(120, activation='relu', kernel_initializer='he_normal', kernel_regularizer=l2(weight_decay), name='fc1')(F1)
    #D1 = keras.layers.pooling.GlobalAveragePooling2D()(conca)
    D1 = Dropout(0.5)(D1)
    D1 = Dense(3, activation='softmax', kernel_initializer='he_normal', kernel_regularizer=l2(weight_decay))(D1)

    model = Model(inputs=[input_1, input_2, input_3], outputs=D1)
    sgd = optimizers.SGD(lr=.02, momentum=0.9, nesterov=True)
    model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])
    return model


def scheduler(epoch):
    learning_rate_init = 0.02
    if epoch >= 20:
        learning_rate_init = 0.01
    if epoch >= 40:
        learning_rate_init = 0.004
    if epoch >= 160:
        learning_rate_init = 0.0008
    return learning_rate_init


def read_image(img_name):
    im = Image.open(img_name).convert('L')
    data = np.array(im)
    return data


if __name__ == '__main__':
    f_train = h5py.File('/home/ml/SONG/JOE/datasets_h5/sonar37_train.h5','r')
    x_train = np.array(f_train['data'])
    x_train_25 = x_train[:,6:31,6:31,:]
    x_train_13 = x_train[:,12:25,12:25,:]
    f_Gtrain = h5py.File('/home/ml/SONG/JOE/datasets_h5/sonar37_Septrain5.h5','r')
    x_Gtrain = np.array(f_Gtrain['data'])
    x_Gtrain_25 = x_Gtrain[:,6:31,6:31,:]
    x_Gtrain_13 = x_Gtrain[:,12:25,12:25,:]
    
    y_train = np.array(f_train['labels'])
    y_Gtrain = np.array(f_Gtrain['labels'])
    f_tt = h5py.File('/home/ml/SONG/JOE/datasets_h5/sonar37_tt1.h5','r')
    x_test = np.array(f_tt['data'])
    x_test_25 = x_test[:,6:31,6:31,:]
    x_test_13 = x_test[:,12:25,12:25,:]

    y_test = np.array(f_tt['labels'])

    y_train = keras.utils.to_categorical(y_train,3)
    x_train = x_train.astype('float32')
    x_train_25 = x_train_25.astype('float32')
    x_train_13 = x_train_13.astype('float32')
    y_Gtrain = keras.utils.to_categorical(y_Gtrain, 3)
    x_Gtrain = x_Gtrain.astype('float32')
    x_Gtrain_25 = x_Gtrain_25.astype('float32')
    x_Gtrain_13 = x_Gtrain_13.astype('float32')

    y_test = keras.utils.to_categorical(y_test,3)
    x_test = x_test.astype('float32')
    x_test_25 = x_test_25.astype('float32')
    x_test_13 = x_test_13.astype('float32')

    #x_train_i = zeros([x_train.shape[0],41,41,1])
    #x_train_i = x_train[:,8:29,8:29,:]
    #x_test_i = zeros([x_test.shape[0],41,41,1])
    #x_test_i = x_test[:,8:29,8:29,:]
    #x_train = x_train_i
    #x_test = x_test_i
    x_train /= 255
    x_train_25 /= 255
    x_train_13 /= 255
    x_Gtrain /= 255
    x_Gtrain_25 /= 255
    x_Gtrain_13 /= 255

    x_test /= 255
    x_test_25 /= 255
    x_test_13 /= 255

    x_train_aug = np.vstack((x_train,x_Gtrain))
    x_train_aug_25 = np.vstack((x_train_25, x_Gtrain_25))
    x_train_aug_13 = np.vstack((x_train_13, x_Gtrain_13))
    y_train_aug = np.vstack((y_train,y_Gtrain))
    # build network
    model = build_model()
    print(model.summary())
    # set callback
    tb_cb = TensorBoard(log_dir=log_filepath, histogram_freq=0)
    change_lr = LearningRateScheduler(scheduler)
    cbks = [change_lr,tb_cb]

    # start training
    initial_time = time.time()
    model.fit({'input_1': x_Gtrain,'input_2': x_Gtrain_25, 'input_3': x_Gtrain_13}, y_Gtrain,batch_size=batch_size,epochs=epochs,verbose=2,
              callbacks=cbks,validation_split = 0.1), shuffle=True)
    training_time = time.time() - initial_time
    print("\ntraining Time = ", training_time)

    initial_time = time.time()
    predict = model.predict({'input_1': x_test, 'input_2': x_test_25,'input_3': x_test_13})
    predict_time = time.time() - initial_time
    print("\nPredict Time = ", predict_time)
    print(type(predict))
    sio.savemat('mscnn_5_tt1.mat', {"mscnn_5_tt1": predict})
    json_string = model.to_json()
    open('mscnn_5_weights.json', 'w').write(json_string)
    model.save_weights('mscnn_5_weights.h5', overwrite=True)
