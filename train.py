import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from keras.layers import Dense, Activation, Flatten, Conv2D, Lambda
from keras.layers import MaxPooling2D, Dropout
from keras.models import Sequential
from keras.callbacks import ModelCheckpoint
import pickle
from keras.optimizers import Adam

def keras_model():
    'NVIDIA MODEL'
    model = Sequential()
    model.add(Lambda(lambda x: x / 127.5 - 1., input_shape=(40, 40, 1)))

    model.add(Conv2D(32, (3, 3), padding='same'))
    model.add(Activation('relu'))
    model.add(MaxPooling2D((2, 2), padding='valid'))

    model.add(Conv2D(64, (3, 3), padding='same'))
    model.add(Activation('relu'))
    model.add(MaxPooling2D((2, 2), padding='valid'))

    model.add(Conv2D(128, (3, 3), padding='same'))
    model.add(Activation('relu'))
    model.add(MaxPooling2D((2, 2), padding='valid'))

    model.add(Flatten())
    model.add(Dropout(0.5))

    model.add(Dense(128))

    model.add(Dense(64))
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
    'load'
    features, labels = loadFromPickle()
    'augmentation'
    features, labels = augmentData(features, labels)
    'shuffle'
    features, labels = shuffle(features, labels)
    'training our model'
    train_x, test_x, train_y, test_y = train_test_split(features, labels, random_state=0,
                                                        test_size=0.1)
    'reshaping our model so it fits with our resized image'
    train_x = train_x.reshape(train_x.shape[0], 40, 40, 1)
    test_x = test_x.reshape(test_x.shape[0], 40, 40, 1)
    model, callbacks = keras_model()
    'giving the predicted values to our model'
    model.fit(train_x, train_y, validation_data=(test_x, test_y), epochs=5, batch_size=64,
              callbacks=callbacks)
    'saving the model'
    model.save('model_save.h5')
main()