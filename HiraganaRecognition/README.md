## About

These files are meant to create a custom dataset for detecting and separating Hiragana out of images.

## Usage

Out of 9 images located in the images directory, two are used for this purpose. It is, however, possible to change the .py files used for cropping to further the sample gathering on other images.

First run the directoryCreation.py, as it will create the train directory and subdirectories for each of the hiragana letters. After that, run either of CropDataset1.py and CropDataset1.py, or run them both, to place cropped images in directories we created before. After that, running the  runTest.py should generate test results in a h5 directory based on train and validation folders.

It is also possible to run the cropping manualy from the supplied .ipynb files. Although, due to the memory required for running each of them, it is highly suggested you open those files through web browser.

## Algorithm

Images used for creating dataset are firstly converted to grayscale through the usage of skimage.color rgb2gray function, setting the global threshold at 0.2 for the first processed image (page1_1.jpg) and 0.3 for the second one (page3_2.jpg). After that, we remove all the white noise through running the erosion for disk(5) which is followed by dilation with disk(0.5). After processing the image in such matter, we labeled image regions.

For the purpose of minimazing the error, each and every test image is cropped manually. We did so by cropping parts of grayscale processed image, plotting the figure and then, after removing axis and scaling the plotted image to 5x5 inches (or 360x360 pixels) train image is being saved as png file in its apropriate location. After each separated image, for the purpose of avoiding overfitting the neural network, every image is shifted couple of pixels up,down,left or right creating therefore 4 new train examples. This is repeted for each individual character.

