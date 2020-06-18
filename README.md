# **Traffic Sign Recognition** 


[//]: # (Image References)

[image1]: ./examples/visualization.jpg "Visualization"
[image2]: ./examples/grayscale.jpg "Grayscaling"
[image3]: ./examples/random_noise.jpg "Random Noise"
[pic1]: ./sampleImgs/pic1.png "Traffic Sign 1"
[pic2]: ./sampleImgs/pic2.png "Traffic Sign 2"
[pic3]: ./sampleImgs/pic3.png "Traffic Sign 3"
[pic4]: ./sampleImgs/pic4.png "Traffic Sign 4"
[pic5]: ./sampleImgs/pic5.png "Traffic Sign 5"
[histogram]: ./sampleImgs/histogram.png "Histogram"
[webpic1]: ./GermanTrafficSigns/pic1.png "Traffic Sign 1"
[webpic2]: ./GermanTrafficSigns/pic2.png "Traffic Sign 2"
[webpic3]: ./GermanTrafficSigns/pic3.png "Traffic Sign 3"
[webpic4]: ./GermanTrafficSigns/pic4.png "Traffic Sign 4"
[webpic5]: ./GermanTrafficSigns/pic5.png "Traffic Sign 5"


---
### Overview

Goal of this project was to use convolutional neural networks to classify traffic signs. This implied the designing the architecture of the net with tensorflow, training it with a given set of german traffic signs and testing it with a test data set and also images from the Internet.

### Data Set Summary & Exploration

#### 1. Overview on the data set

The data set used in this project consist of labeled images of german traffic signs.

* Number of training images: 34799
* Number of validation images: 4410
* Number of test images: 12630
* The shape of a traffic sign image is 32x32 RGB images
* The number of unique classes/labels in the data set is 43




#### 2. Include an exploratory visualization of the dataset.

To get a rough idea how the traffic signs to identify I plotted a small subset from the training data with their labels:

![alt text][pic1]
![alt text][pic2]
![alt text][pic3]
![alt text][pic4]
![alt text][pic5]

Also, to get a better idea on the distribution over the 43 possible labels here the histogram:

![alt text][histogram]

### Design and Test a Model Architecture

#### 1. Preprocessing
Preprocessing involved normalizing all images to have 
* Zero mean
* Values between -1 and 1

This can be achieved by `(img - 128)/128`.

#### 2. Model architecture
My initial model follows the LeNet architecture consists of the following layers:

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32x3 RGB image   							| 
| Convolution 3x3     	| 1x1 stride, same padding, outputs 28x28x12 	|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 14x14x12 				    |
| Convolution 3x3	    | 1x1 stride, same padding, outputs 10x10x16    |  									
| Max pooling    		| 2x2 stride,  outputs 5x5x16 					|
| Fully connected		| inputs 400, outputs 120       				|
| RELU					|												|
| Fully connected		| inputs 120, outputs 84						|
| RELU                  |                                               |
| Fully connected       | inputs 84, outputs 43 (num. of classes)       |
| Softmax               |                                               |


#### 3. Describe how you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

To train the model, I used the Softmax function as loss function. For the actual optimization I used the adam optimizer of tensorflow which is basically a variant of the stocasic gradient descent. 

I choose the hyper parameters initially as follows:
* learning rate: 0.0006
* batch size: 128
* number of epochs: 150

#### 4. Training the model

My results with this architecture and set of hyperparameters  were:
* training set accuracy of 0.95
* validation set accuracy of 0.93 
* test set accuracy of 0.88

With this set of hyper parameter I was able to meet the required accuracy of 0.93 barely. The convergence between epochs was extremely poor towards the end which point to overfitting.

I started experimenting with dropout layers which I added to the first fully connected layer with some success. The the dropout probability was set to 50%. 

The second thing which I did was change the input data to be grayscale. Intuitively it is clear that most information of the traffic sign type is embedded in the shape as many signs share similar color patterns. 

I also balanced the training set as some of the samples which where wrongly identified where underrepresented. I did this by duplicating random samples from the underrepresented classes to be at least 400. 

Also I did some tweaking with the hyperparameters and changed the number of epochs to 250

With said changes I was able to achieve
* training set accuracy of 0.9929
* validation set accuracy of 0.9451 
* test set accuracy of 0.9243

### Test a Model on New Images

#### 1. Traffic sign examples

To test the model I used five traffic signs which I basically used some post-processed screenshots of traffic signs from google street view (from the lovely city of Frankfurt). Also I used five random samples from the test data set. This set of traffic signs was correctly identified by the the net. 

![alt text][webpic1] ![alt text][webpic2] ![alt text][webpic3] 
![alt text][webpic4] ![alt text][webpic5]


#### 2. Prediction results

Here are the results of the prediction for the traffic signs from the web:

| Image			        |     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| Keep right     		| Keep right (prob. 1.0)									| 
| Speed limit (50km/h)			| Speed limit (50km/h) (prob. 0.9999974)										|
| General caution				| General caution (prob. 0.9997389)											|
| Priority road	      		| Priority road	(0.994601)				 				|
| Yield			| Yield (0.9999982)     							|


The model was able to correctly guess 5 of the 5 traffic signs, which gives an accuracy of 100%. This could be expected given the accuracies base on the validation and test data set. 

Some of the images of the test date set (not the web images) are quite badly lit which will make it harder to classify. On other hand it will be necessary to provide learning with different environment conditions (in this case different lighting conditions) to train the neural net to be able to classify images under changing conditions. This could be achieved by duplicating the images in the training data set with versions e.g. by converting to HSL colorspace and reducing the overall lightness. An other idea to artificially increases the variety in the data set ist to use a perspective transformation to create images from different perspectives.


#### 3. Confidence of the predictions

Calculating the softmax probability for each of the 43 possible classes gives nuber on how confident we can be on an estimate to be correct. For the given images here the tob five probabilities: 

| Probability top 5        	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 1.0000000e+00, 8.0429316e-14, 1.6923558e-18, 9.0425724e-20, 1.3638657e-20       			| Keep right    									| 
| 9.9999738e-01, 2.6758682e-06, 1.9348943e-11, 1.1417563e-11, 4.5416279e-12   				| Speed limit (50km/h) 										|
| 9.9973887e-01, 2.1376039e-04, 3.8265014e-05, 4.7601529e-06, 2.9047935e-06					| General caution											|
| 9.9460101e-01, 3.1535293e-03, 9.0386666e-04, 6.6403067e-04, 3.0325877e-04	      			| Priority road					 				|
| 9.9999821e-01, 1.2339864e-06, 6.1178520e-07, 1.9356555e-08, 7.6480502e-11				    | Yield      							|


As the probability of the top estimate is almost 1 and the other softmax values are many decades smaller we can be relatively confident on the predictions. 

