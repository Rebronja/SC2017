import keras.backend as K
import theano.tensor as T

import keras.applications.vgg16
import keras.models
import keras.layers
import lasagne
import lasagne.layers
import numpy as np
import os
import skimage.io
import sklearn.ensemble
import sklearn.neighbors
import sklearn.linear_model
import sklearn.svm
import sklearn.metrics
import sklearn.preprocessing
import theano

K.set_image_dim_ordering('th')


def cnn(train_data,test_data,train_labels,test_labels):
    (train_data, test_data, train_labels, test_labels) = preprocess(train_data, test_data, train_labels, test_labels)

    def larger_model():
        # create model
        model = keras.models.Sequential()
        model.add(keras.layers.Conv2D(30, (5, 5), input_shape=(1, 150, 150), activation='relu'))
        model.add(keras.layers.MaxPooling2D(pool_size=(2, 2)))
        model.add(keras.layers.Conv2D(15, (3, 3), activation='relu'))
        model.add(keras.layers.MaxPooling2D(pool_size=(2, 2)))
        model.add(keras.layers.Dropout(0.25))
        model.add(keras.layers.Flatten())
        model.add(keras.layers.Dense(128, activation='relu'))
        model.add(keras.layers.Dense(50, activation='relu'))
        model.add(keras.layers.Dense(74, activation='softmax'))
        # Compile model
        model.compile(loss='mse', optimizer='rmsprop', metrics=['mae', 'mape'])
        return model


    # build the model
    model = larger_model()
    # Fit the model
    model.fit(train_data, train_labels, epochs=300, batch_size=50, verbose=0)

    scores = model.evaluate(test_data, test_labels, verbose=0)
    print("Baseline Error: %.2f%%" % (scores[1]*100))

def knn(train_data,test_data,train_labels,test_labels):
    (train_data, test_data, train_labels, test_labels) = preprocess(train_data, test_data, train_labels, test_labels)
    nsamples, nx, ny, nz = train_data.shape
    d2_train_data = train_data.reshape((nsamples,nx*ny*nz))

    model = sklearn.neighbors.KNeighborsClassifier(n_neighbors = 1)
    model.fit(d2_train_data, train_labels)

    #printing out the score for test sample

    nsamples, nx, ny, nz = test_data.shape
    d2_test_data = test_data.reshape((nsamples,nx*ny*nz))
    prediction = model.predict(d2_test_data)
    acc_knn = sklearn.metrics.accuracy_score(test_labels, prediction)
    print('Nearest neighbours accuracy: ',acc_knn)

def lstm(train_data,test_data,train_labels,test_labels):
    (train_data, test_data, train_labels, test_labels) = preprocess(train_data, test_data, train_labels, test_labels)

    nsamples, nx, ny, nz = train_data.shape
    d3_train_data = train_data.reshape((nsamples,nx,ny*nz))

    model = keras.models.Sequential()
    model.add(keras.layers.LSTM(2, batch_input_shape=(10, 1, 22500), input_shape=(None,22500), stateful=True, return_sequences=True))
    model.add(keras.layers.LSTM(2, batch_input_shape=(10, 1, 22500), input_shape=(None,22500), stateful=True))
    model.add(keras.layers.Dense(74))
    model.compile(loss='mean_squared_error', optimizer='adam')
    model.fit(d3_train_data, train_labels, epochs=300, batch_size=10, verbose=0)

    scaler = sklearn.preprocessing.MinMaxScaler(feature_range=(0, 1))
    nsamples, nx, ny, nz = test_data.shape
    d3_test_data = test_data.reshape((nsamples,nx,ny*nz))

    testPredict = model.predict(d3_test_data, batch_size=10)
    scores = model.evaluate(d3_test_data, test_labels, verbose=0, batch_size=10)
    print("Baseline Error: %.2f%%" % (100-scores*100))

def multipleLSTM(dset):
    # Number of Units in hidden layers
    L1_UNITS = 50
    L2_UNITS = 100

    # Training Params
    LEARNING_RATE = 0.001
    N_BATCH = 10
    NUM_EPOCHS = 1500

    # Load the dataset
    print("Loading data...")
    (train,test,train_labels,test_labels) = dset.require_new(8,2, True)

    train=np.array(train)
    test=np.array(test)
    train_labels=np.array(train_labels)
    test_labels=np.array(test_labels)
    train = train.reshape(train.shape[0], 150, 150).astype('float32')
    test = test.reshape(test.shape[0], 150, 150).astype('float32')

    num_feat    = train.shape[1]
    num_classes = np.unique(test).size

    # Generate sequence masks (redundant here)
    mask_train = np.ones((train.shape[0], train.shape[1]))
    mask_test  = np.ones((test.shape[0], test.shape[1]))


    # Model
    tanh = lasagne.nonlinearities.tanh
    relu = lasagne.nonlinearities.rectify
    soft = lasagne.nonlinearities.softmax

    # Network Architecture
    l_in = lasagne.layers.InputLayer(shape=(None, None, num_feat))
    batchsize, seqlen, _ = l_in.input_var.shape

    l_noise = lasagne.layers.GaussianNoiseLayer(l_in, sigma=0.6)
    l_mask  = lasagne.layers.InputLayer(shape=(batchsize, seqlen))

    l_rnn_1 = lasagne.layers.LSTMLayer(l_noise, num_units=L1_UNITS, mask_input=l_mask)
    l_in_drop = lasagne.layers.DropoutLayer(l_rnn_1, p=0.25)
    l_rnn_2 = lasagne.layers.LSTMLayer(l_in_drop, num_units=L2_UNITS)
    l_in_drop2 = lasagne.layers.DropoutLayer(l_rnn_2, p=0.1)
    l_shp   = lasagne.layers.ReshapeLayer(l_in_drop2,(-1, L2_UNITS))
    l_dense = lasagne.layers.DenseLayer(l_shp, num_units=num_classes, nonlinearity=soft)
    l_out   = lasagne.layers.ReshapeLayer(l_dense, (batchsize, seqlen, num_classes))


    # Symbols and Cost Function
    target_values = T.ivector('target_output')

    network_output   = lasagne.layers.get_output(l_out)
    predicted_values = network_output[:, -1]
    prediction = T.argmax(predicted_values, axis=1)
    all_params = lasagne.layers.get_all_params(l_out, trainable=True)

    cost = lasagne.objectives.categorical_crossentropy(predicted_values, target_values)
    cost = cost.mean()



    # Compute SGD updates for training
    print("Computing updates ...")
    updates = lasagne.updates.rmsprop(cost, all_params, LEARNING_RATE)

    # Theano functions for training and computing cost
    print("Compiling functions ...")
    training   = theano.function(
        [l_in.input_var, target_values, l_mask.input_var], cost, updates=updates,allow_input_downcast=True)
    predict = theano.function([l_in.input_var, l_mask.input_var], prediction,allow_input_downcast=True)
    compute_cost = theano.function([l_in.input_var, target_values, l_mask.input_var], cost,allow_input_downcast=True)




    # Training
    print("Training ...")
    num_batches_train = int(np.ceil(len(train) / N_BATCH))
    train_losses = []
    valid_losses = []

    for epoch in range(NUM_EPOCHS):
        now = time.time
        losses = []

        batch_shuffle = np.random.choice(train.shape[0], train.shape[0], False)
        sequences   = train[batch_shuffle]
        labels      = train_labels[batch_shuffle]
        train_masks = mask_train[batch_shuffle]

        for batch in range(num_batches_train):
            batch_slice = slice(N_BATCH * batch,
                                N_BATCH * (batch + 1))

            X_batch = sequences[batch_slice]
            y_batch = labels[batch_slice]
            m_batch = train_masks[batch_slice]

            loss = training(X_batch, y_batch, m_batch)
            losses.append(loss)

        train_loss = np.mean(losses)
        train_losses.append(train_loss)

        valid_loss = compute_cost(test, test_labels, mask_test)
        valid_losses.append(valid_loss)

        test_pred   = predict(test, mask_test)
        accuracy = sklearn.metrics.accuracy_score(test_labels, test_pred)


        print('Current epoch:', epoch+1,'|', 'Number of Epochs:', NUM_EPOCHS,'|','Train loss:', train_loss,'|','Validation loss:', valid_loss,'|','Accuracy:', accuracy)

def neuralNet50(dataset):
    folder='images50x50'

    rel_path = 'images50x50'
    size = 2500 # 50x50 slicica

    folders = os.listdir(rel_path)

    kana_dict = dataset.characters()
    if os.path.isfile('dset50x50.hdf5'):
        os.remove('dset50x50.hdf5')
    hiragana_dataset = dataset.HiraSet('dset50x50', size)

    for folder in folders:
        entry = dataset.HiraEntry(folder, kana_dict[folder])

        files = os.listdir(rel_path + '/' + folder)
        for file in files:
            img = skimage.io.imread(rel_path + '/' + folder + '/' + file)
            re_img = np.reshape(img, size)
            flt_img = re_img / 65535.0

            entry.add(flt_img)

        hiragana_dataset.add(entry)
    dset50x50 = dataset.HiraSet('dset50x50', 2500)
    dset50x50.pull()


    (train,test,train_labels,test_labels) = dset50x50.require_new(25,20)
    train=np.array(train)
    test=np.array(test)
    train_labels=np.array(train_labels)
    test_labels=np.array(test_labels)



    model = keras.models.Sequential()
    model.add(keras.layers.Dense(2500, input_dim=2500, kernel_initializer='normal', activation='relu'))
    model.add(keras.layers.Dense(74, kernel_initializer='normal', activation='softmax'))
    # Compile model
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])


    # training
    training = model.fit(train, train_labels, epochs=300, batch_size=100, verbose=0)
    scores = model.evaluate(test, test_labels, verbose=0)
    print("Baseline Error: %.2f%%" % (100-scores[1]*100))


def preprocess(train_data,test_data,train_labels,test_labels):
    train_data=np.array(train_data)
    test_data=np.array(test_data)
    train_labels=np.array(train_labels)
    test_labels=np.array(test_labels)
    train_data = train_data.reshape(train_data.shape[0], 1, 150, 150).astype('float32')
    test_data = test_data.reshape(test_data.shape[0], 1, 150, 150).astype('float32')
    return (train_data,test_data,train_labels,test_labels)


def randomForest(train_data,test_data,train_labels,test_labels):
    train_data=np.array(train_data)
    test_data=np.array(test_data)
    train_labels=np.array(train_labels)
    test_labels=np.array(test_labels)

    model = sklearn.ensemble.RandomForestClassifier(n_estimators=50)
    model.fit(train_data, train_labels)
    prediction = model.predict(test_data)
    acc_rf = sklearn.metrics.accuracy_score(test_labels, prediction)
    print ("Random forest accuracy: ",acc_rf)

def sgd(dset):
    (train_data,test_data,train_labels,test_labels) = dset.require_new(25,20,True)
    train_data=np.array(train_data)
    test_data=np.array(test_data)
    train_labels=np.array(train_labels)
    test_labels=np.array(test_labels)


    model = sklearn.linear_model.SGDClassifier(max_iter=100,tol=0.003)
    model.fit(train_data, train_labels)
    prediction = model.predict(test_data)
    acc_sgd = sklearn.metrics.accuracy_score(test_labels, prediction)
    print ("Stochastic gradient descent accuracy: ",acc_sgd)

def svm(dset):
    (train_data,test_data,train_labels,test_labels) = dset.require_new(25,20, True)
    train_data=np.array(train_data)
    test_data=np.array(test_data)
    train_labels=np.array(train_labels)
    test_labels=np.array(test_labels)

    model = sklearn.svm.LinearSVC()
    model.fit(train_data, train_labels)
    prediction = model.predict(test_data)
    acc_svm = sklearn.metrics.accuracy_score(test_labels, prediction)
    print ("Linear SVM accuracy: ",acc_svm)

def vgg16(dset):
    #Get back the convolutional part of a VGG network trained on ImageNet
    model_vgg16_conv = keras.applications.vgg16.VGG16(weights='imagenet', include_top=False)
    model_vgg16_conv.summary()

    #Create your own input format (here 3x200x200)
    input = keras.layers.Input(shape=(3,150,150),name = 'image_input')

    #Use the generated model
    output_vgg16_conv = model_vgg16_conv(input)

    #Add the fully-connected layers
    x = keras.layers.Flatten(name='flatten')(output_vgg16_conv)
    x = keras.layers.Dense(4096, activation='relu', name='fc1')(x)
    x = keras.layers.Dense(4096, activation='relu', name='fc2')(x)
    x = keras.layers.Dense(74, activation='softmax', name='predictions')(x)

    #Create your own model
    my_model = keras.models.Model(input=input, output=x)

    #In the summary, weights and layers from VGG part will be hidden, but they will be fit during the training
    my_model.summary()
