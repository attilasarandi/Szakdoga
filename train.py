import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from keras.layers import Dense, Flatten, Conv2D, Lambda, Dropout, Activation
from keras.layers import MaxPooling2D
from keras.models import Sequential
from keras.callbacks import ModelCheckpoint
import pickle
from keras.optimizers import Adam
import matplotlib.pyplot as plt
import cv2
from imgaug import augmenters as iaa


def keras_model():
    'NVIDIA MODEL'
    model = Sequential()

    # Normalizing data in range of -1 to 1 and zero-centering data
    model.add(Lambda(lambda x: x / 127.5 - 1., input_shape=(40, 40, 1)))

    # Layer 1: 5x5 Conv + RELU + + 2x2 MaxPool2D
    model.add(Conv2D(24, (5, 5), padding='same'))
    model.add(Activation('relu'))
    model.add(MaxPooling2D((2, 2), padding='valid'))

    # Layer 2: 5x5 Conv + RELU + + 2x2 MaxPool2D
    model.add(Conv2D(36, (5, 5), padding='same'))
    model.add(Activation('relu'))
    model.add(MaxPooling2D((2, 2), padding='valid'))

    # Layer 3: 5x5 Conv + RELU + 2x2 MaxPool2D
    model.add(Conv2D(48, (5, 5), padding='same'))
    model.add(Activation('relu'))
    model.add(MaxPooling2D((2, 2), padding='valid'))

    # Layer 4: 3x3 Conv + RELU
    model.add(Conv2D(64, (3, 3), padding='same'))
    model.add(Activation('relu'))

    # Layer 5: 3x3 Conv + RELU
    model.add(Conv2D(64, (3, 3), padding='same'))
    model.add(Activation('relu'))

    #Flatten layer
    model.add(Flatten())
    model.add(Dropout(0.5))
    # Layers 6-8: Fully connected
    model.add(Dense(100))
    model.add(Dense(50))
    model.add(Dense(10))
    #Output Layer
    model.add(Dense(1))
    'compiling the created module'
    model.compile(optimizer=Adam(lr=0.0001), loss="mse")
    'saving directory'
    file = "model_save.h5"
    'saving the best model'
    checkpoint = ModelCheckpoint(file, verbose=1, save_best_only=True)
    callbacks_list = [checkpoint]

    return model, callbacks_list


def loadFromPickle():
    'loading our saved pickle files'
    with open("features", "rb") as f:
        features = np.array(pickle.load(f))
    with open("labels", "rb") as f:
        labels = np.array(pickle.load(f))

    return features, labels

def augmentData(features, labels):
    'augmenting the features and labels files'
    features = np.append(features, features[:, :, ::-1], axis=0)
    labels = np.append(labels, -labels, axis=0)
    return features, labels

def main():
    'load our images and steering values'
    features, labels = loadFromPickle()
    'augmentation'
    features, labels = augmentData(features, labels)
    'shuffle for more random data'
    features, labels = shuffle(features, labels)
    'training our model'
    train_x, test_x, train_y, test_y = train_test_split(features, labels, random_state=0, test_size=0.1)
    'reshaping our model so it fits with our resized image'
    train_x = train_x.reshape(train_x.shape[0], 40, 40, 1)
    test_x = test_x.reshape(test_x.shape[0], 40, 40, 1)
    'model and callbacks list'
    model, callbacks = keras_model()
    'fit our model to the training dataset'
    history = model.fit(train_x, train_y, validation_data=(test_x, test_y), epochs=5, batch_size=64, callbacks=callbacks)
    'summary of our model'
    model.summary()
    'saving the model'
    model.save('model_save.h5')
    'plotting the losses of our model the lower it is the best'
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.legend(['Training', 'Validation'])
    plt.title('Loss')
    plt.xlabel('Epoch')
    plt.show()
main()