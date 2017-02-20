import numpy as np
import time
import theano
import theano.tensor as T
import lasagne
import sklearn.metrics
import lasagne.layers as L

from lasagne.layers import InputLayer, LSTMLayer, ReshapeLayer, DenseLayer, GaussianNoiseLayer

def RNN_Compute(layer1, layer2, lrate, batch_size, epochs, train, test, train_labels, test_labels):

    # Number of Units in hidden layers
    L1_UNITS = layer1
    L2_UNITS = layer2

    # Training Params
    LEARNING_RATE = lrate
    N_BATCH = batch_size
    NUM_EPOCHS = epochs


    num_feat = train.shape[1]
    num_classes = np.unique(test).size

    # Generate sequence masks (redundant here)
    mask_train = np.ones((train.shape[0], train.shape[1]))
    mask_test = np.ones((test.shape[0], test.shape[1]))

    # Model
    tanh = lasagne.nonlinearities.tanh
    relu = lasagne.nonlinearities.rectify
    soft = lasagne.nonlinearities.softmax

    # Network Architecture
    l_in = InputLayer(shape=(None, None, num_feat))
    batchsize, seqlen, _ = l_in.input_var.shape

    l_noise = GaussianNoiseLayer(l_in, sigma=0.6)
    l_mask = InputLayer(shape=(batchsize, seqlen))

    l_rnn_1 = LSTMLayer(l_noise, num_units=L1_UNITS, mask_input=l_mask)
    l_in_drop = lasagne.layers.DropoutLayer(l_rnn_1, p=0.25)
    l_rnn_2 = LSTMLayer(l_in_drop, num_units=L2_UNITS)
    l_in_drop2 = lasagne.layers.DropoutLayer(l_rnn_2, p=0.1)
    l_shp = ReshapeLayer(l_in_drop2, (-1, L2_UNITS))
    l_dense = DenseLayer(l_shp, num_units=num_classes, nonlinearity=soft)
    l_out = ReshapeLayer(l_dense, (batchsize, seqlen, num_classes))

    # Symbols and Cost Function
    target_values = T.ivector('target_output')

    network_output = L.get_output(l_out)
    predicted_values = network_output[:, -1]
    prediction = T.argmax(predicted_values, axis=1)
    all_params = L.get_all_params(l_out, trainable=True)

    cost = lasagne.objectives.categorical_crossentropy(predicted_values, target_values)
    cost = cost.mean()

    # Compute SGD updates for training
    print("Computing updates ...")
    updates = lasagne.updates.rmsprop(cost, all_params, LEARNING_RATE)

    # Theano functions for training and computing cost
    print("Compiling functions ...")
    training = theano.function(
        [l_in.input_var, target_values, l_mask.input_var], cost, updates=updates, allow_input_downcast=True)
    predict = theano.function([l_in.input_var, l_mask.input_var], prediction, allow_input_downcast=True)
    compute_cost = theano.function([l_in.input_var, target_values, l_mask.input_var], cost, allow_input_downcast=True)

    # Training
    print("Training ...")
    num_batches_train = int(np.ceil(len(train) / N_BATCH))
    train_losses = []
    valid_losses = []

    for epoch in range(NUM_EPOCHS):
        now = time.time
        losses = []

        batch_shuffle = np.random.choice(train.shape[0], train.shape[0], False)
        sequences = train[batch_shuffle]
        labels = train_labels[batch_shuffle]
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

        test_pred = predict(test, mask_test)
        accuracy = sklearn.metrics.accuracy_score(test_labels, test_pred)

        print('Current epoch:', epoch + 1, '|', 'Number of Epochs:', NUM_EPOCHS, '|', 'Train loss:', train_loss, '|',
              'Validation loss:', valid_loss, '|', 'Accuracy:', accuracy)