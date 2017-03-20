Usage:
python predict.py -i [--image] <image_path> -c [--classifier] <classifier>

if you wish to run neural network on pretrained models like ann or cnn.

If you wish to train your own model on your own data, run something like hiraNet = hnn.HNN(classifier, filename)
and then hiraNet.train(data,labels,epoch,batch)
where data is your train data, labels are labels for the input data, epoch is number of training epochs and batch is the desired batch size for processing

It's possible to train and use K nearest neighbours, convolutional neural network, a 2 layer artificial neural network, long short term memory neural network (2 dimensional recurrent neural network), random forest, support vector machine and stochastic gradient descent.

There are training examples in the Neural_Network_Zoo.ipynb jupyter notebook, along with the comments that hold info about running the training process on the GeForce GTX 950 graphics card aswell as the accuracy achieved.

Once you have the model trained, running the hiraNet.predict(image_name) will classify the image based on chosen algorithm.

So far, we have used this in classifying hiragana characters on our custom dataset