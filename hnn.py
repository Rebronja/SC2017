from keras.models import load_model
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import Flatten
from keras.layers.convolutional import Convolution2D
from keras.layers.convolutional import MaxPooling2D
from sklearn.neighbors import KNeighborsClassifier
from keras import backend as K
K.set_image_dim_ordering('th')


class HNN:
    def __init__(self, classifier, file, model=None):
        self.__classifier = classifier
        self.__file = file
        self.__model = model

    def train(self, data, labels, epoch, batch):
        if self.__classifier=='ann':
            self.__model.fit(data, labels, nb_epoch=epoch, batch_size=batch, verbose=0)
            self.__model.save('models/' + self.__classifier + '/' + self.__file + '.h5')
        elif self.__classifier=='cnn':
            self.__model.fit(data, labels, nb_epoch=epoch, batch_size=batch, verbose=0)
            self.__model.save('models/' + self.__classifier + '/' + self.__file + '.h5')
        elif self.__classifier=='knn':
            self.__model.fit(data,labels)



    def load(self):
        model = load_model('models/' + self.__classifier + '/' + self.__file + '.h5')
        self.__model = model

    def predict(self, img):
        prediction = self.__model.predict(img)
        return prediction

    def evaluate(self, test_data, test_labels):
        if self.__classifier == 'knn':
            evaluation = self.__model.score(test_data, test_labels, verbose=0)
        else:
            evaluation = self.__model.evaluate(test_data, test_labels, verbose=0)
        return evaluation




def ann_model(dimension,output):
    model = Sequential()
    model.add(Dense(dimension, input_dim=dimension, init='normal', activation='relu'))
    model.add(Dense(output, init='normal', activation='softmax'))
    # Compile model
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

def cnn_model(dimH,dimW,output):
    # create model
    model = Sequential()
    model.add(Convolution2D(30, 5, 5, border_mode='valid', input_shape=(1, 150, 150), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Convolution2D(15, 3, 3, activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dense(50, activation='relu'))
    model.add(Dense(74, activation='softmax'))
    # Compile model
    model.compile(loss='mse', optimizer='rmsprop', metrics=['mae', 'mape'])
    return model

def knn_model(neighbours):
    model = KNeighborsClassifier(n_neighbors = neighbours)
    return model