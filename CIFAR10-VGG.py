import tensorflow as tf 
from tensorflow import keras 
import numpy as np 
import matplotlib.pyplot as plt 
import sys 

from keras.datasets import cifar10
from keras.models import Sequential 
from keras.layers import Dense, Dropout, Flatten 
from keras.layers import Conv2D, MaxPooling2D
from keras.optimizers import SGD
from keras.regularizers import l2
from keras.preprocessing.image import ImageDataGenerator
from keras.layers import BatchNormalization
from time import time 

def load_dataset():
    #load dataset
    (trainX, trainY),(testX, testY) = cifar10.load_data()
    #one hot encode the target 
    trainY = keras.utils.to_categorical(trainY)
    testY = keras.utils.to_categorical(testY)
    return trainX, trainY, testX, testY

def validation_split(testX, testY, valid_X, valid_Y, v_split):
    
    index_of_validation = int(v_split * len(testX))
    valid_X.extend(testX[-index_of_validation:])
    valid_Y.extend(testY[-index_of_validation:])
    testX = testX[:-index_of_validation]
    testY = testY[:-index_of_validation]
    return testX, testY, np.asarray(valid_X), np.asarray(valid_Y)

def normalize(train,test,valid):
    # convert from integers to float 
    train_norm = train.astype('float32')
    test_norm = test.astype('float32')
    valid_norm = valid.astype('float32')
    #normalize to range 0-1
    train_norm = train_norm / 255.0
    test_norm = test_norm / 255.0
    valid_norm = valid_norm / 255.0
    return train_norm, test_norm,valid_norm 

def define_model():
    model = Sequential()
    model.add(Conv2D(32,(3,3), activation = 'relu', kernel_initializer = 'he_uniform', padding = 'same', input_shape = (32,32,3)))
    model.add(BatchNormalization())
    model.add(Conv2D(32,(3,3), activation = 'relu', kernel_initializer = 'he_uniform', padding = 'same'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D((2,2)))
    model.add(Dropout(0.2))
    model.add(Conv2D(64,(3,3), activation = 'relu', kernel_initializer = 'he_uniform', padding = 'same'))
    model.add(BatchNormalization())
    model.add(Conv2D(64,(3,3), activation = 'relu', kernel_initializer = 'he_uniform', padding = 'same'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D((2,2)))
    model.add(Dropout(0.3))
    model.add(Conv2D(128,(3,3), activation = 'relu', kernel_initializer = 'he_uniform', padding = 'same'))
    model.add(BatchNormalization())
    model.add(Conv2D(128,(3,3), activation = 'relu', kernel_initializer = 'he_uniform', padding = 'same'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D((2,2)))
    model.add(Dropout(0.4))
   

    model.add(Flatten())
    model.add(Dense(128, activation = 'relu', kernel_initializer = 'he_uniform'))
    model.add(BatchNormalization())
    model.add(Dropout(0.5))
    model.add(Dense(10, activation = 'softmax'))
    
    #compile model 
    opt = SGD(lr = 0.001, momentum = 0.9)
    model.compile(optimizer = opt, loss = 'categorical_crossentropy', metrics = ['accuracy'])
    return model

# plot diagnostic learning curves
def summarize_diagnostics(history):
    plt.subplots(figsize = (7,7))
    # plot loss
    plt.subplot(211)
    plt.title('Cross Entropy Loss')
    plt.plot(history.history['loss'], color='blue', label='train')
    plt.plot(history.history['val_loss'], color='orange', label='test')

    # plot accuracy
    plt.subplot(212)
    plt.title('Classification Accuracy')
    plt.plot(history.history['accuracy'], color='blue', label='train')
    plt.plot(history.history['val_accuracy'], color='orange', label='test')
    plt.show()
    # save plot to file
    filename = sys.argv[0].split('/')[-1]
    plt.savefig(filename + '_plot.png')
    plt.close()

# run all the defined functions for evaluating a model
def test_model():
    # load dataset
    trainX, trainY, testX, testY = load_dataset()
    #get validation set 
    valid_X = []
    valid_Y = []
    testX, testY, validX, validY = validation_split(testX, testY, valid_X, valid_Y,v_split=0.5)
    
    # normalize the data
    trainX, testX,validX = normalize(trainX, testX,validX)
    # define model
    model = define_model()
    #create data generator 
    datagen = ImageDataGenerator(width_shift_range = 0.1, height_shift_range = 0.1, horizontal_flip = True)
    #iterator 
    train = datagen.flow(trainX, trainY, batch_size = 64)
    # fit model
    steps = int(trainX.shape[0]/ 64)
    history = model.fit_generator(train, steps_per_epoch = steps, epochs=400, validation_data=(validX, validY), verbose=0)
    # evaluate model
    _, acc = model.evaluate(testX, testY, verbose=0)
    print('> %.3f' % (acc * 100.0))
    # learning curves
    summarize_diagnostics(history)
    return history

def main():
	test_model()

if __name__ == "__main__":
    main()

