#**Traffic Sign Recognition**

**Build a Traffic Sign Recognition Project**

The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report

---
###Writeup

####1. Provide a Writeup / README that includes all the rubric points and how you addressed each one. You can submit your writeup as markdown or pdf. You can use this template as a guide for writing the report. The submission includes the project code.

Here is a link to my [project code](https://github.com/jseptienr/CarND-Traffic-Signs-P2)

###Data Set Summary & Exploration

####1. Provide a basic summary of the data set. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.

Here is a basic summary of the dataset for the project:

* The size of training set is 34799
* The size of the validation set is 4410
* The size of test set is 12630
* The shape of a traffic sign image is 32x32x3
* The number of unique classes/labels in the data set is 43

####2. Include an exploratory visualization of the dataset.

First I wanted to see what the data looked like, so I printed one sample image for each class.

[image1]: ./examples/samples.png "Samples"

This shows that the data set is composed of 43 different traffic sign images.

Next to visualize the distribution of samples, I created a histogram that shows the number of samples per class.

[image1]: ./examples/histogram.png "Histogram"

There is not an even distribution for the different classes as some classes have fewer samples. One potential solution is to create new samples to equalize the number of samples per class. For the sake of this project, I decided to test using the existing distribution since there is relatively enough samples to generalize.

###Design and Test a Model Architecture

####1. Describe how you preprocessed the image data. What techniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc. (OPTIONAL: As described in the "Stand Out Suggestions" part of the rubric, if you generated additional data for training, describe why you decided to generate additional data, how you generated the data, and provide example images of the additional data. Then describe the characteristics of the augmented training set like number of images in the set, number of images for each class, etc.)

For preprocessing, I began trying to convert images to grayscale but ended up just normalizing the samples to maintain the color channels and thus a depth of 3 for the input layer. The normalization step takes the pixel data and scale it from a range of 0 to 255 to a range of 0.1 to 0.9 to avoid problems such as dividing by zero.


####2. Describe what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

I used the LeNet architecture as a baseline. I decided to add two dropout layers after the first and second fully connected layers in order to achieve a better performance and avoid overfitting. The final architecture was:

| Layer         		    |     Description	        					|
|:---------------------:|:---------------------------------------------:|
| Input         		    | 32x32x3 RGB image   							|
| Convolution 3x3     	| 1x1 stride, valid padding, outputs 28x28x6 	|
| RELU					        |												                      |
| Max pooling	      	  | 2x2 stride,  outputs 14x14x6                |
| Convolution 3x3	      | 1x1 stride, valid padding, outputs 10x10x16 |
| Max pooling	      	  | 2x2 stride,  outputs 10x10x16               |
| Fully connected		    | 400x120 Fully connected layer      									|
| RELU					        |												                      |
| Dropout					      |	Dropout layer with 0.5 probability            |
| Fully connected		    | 120x84 Fully connected layer        									|
| RELU					        |												                      |
| Dropout					      |	Dropout layer with 0.5 probability           |
| Fully connected		    | 84x43 Fully connected layer      							|
| Softmax		            | softmax classifier     							|

Given that the baseline model had a good enough performance, I decided to keep the architecture for the convolutional and pooling layers. This means that this architecture is able to detect features and generalize to different types of images and tune the last layers for the task of classifying traffic signs. I focused on trying to improve the overal architecture and prevent overfitting by adding two dropout layers after the first two fully connected layers.

####3. Describe how you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

To train the model, cross entropy is computed with Softmax classifier applied to the outputs of the network. It uses Adam Optimizer from the lab with a learning rate of 0.001 to minimize the cross entropy loss and I used 128 as the batch size for training. I tested with different values for hyperparameters such as increasing the learning rate, but achieve a better performance by maintaining a small learning rate. In the first run, the number of epochs was set at 10 but I increased it to 30. This allowed the model to achieve better accuracy but increase the time of training.

####4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

In the first run I achieve a validation accuracy of 0.883 by just using the LeNet architecture. I achieve marginally better results by increasing the number of epochs to 20 and the learning rate to 0.005 but still did not make it to good enough validation accuracy of 0.93. I figured that there was some overfitting and decided to add the two dropout layers after the first two fully connected layers. I also increased the learning rate to 30 which yielded a much better accuracy of 0.958.

My final model results were:
* validation set accuracy of 0.958
* test set accuracy of 0.924

I used LeNet architecture as a baseline from the lab and was surprised by its performance. Since it achieve a great performance on the MNIST dataset, traffic signs would be relatively similar to the digit classification task. I figured that with some additional tuning it would meet the performance needed to classify the images. By adding the dropout layers and increasing the epochs allowed achieving a better accuracy.


###Test a Model on New Images

####1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

I used the following traffic sign images from the web:

[image1]: ./examples/test_signs.png "Test Signs"

There was nothing particularly unique about the images that would make it difficult to classify other than the resolution of the signs from a distance. This would cause a low pixel density to define the sign and thus make it difficult for the model to predict accurately. I used six images with some that are relatively similar to test how the model performed with small variation.

####2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

Here are the results of the prediction:

| Image			            |     Prediction	        					            |     
|:---------------------:|:---------------------------------------------:|
| Right-of-way     		  | Right-of-way  									              |
| Speed Limit 30  			| Speed Limit 30										            |
| No Vehicles				    | No vehicles										                |
| Double Curve	      	| Right of Way					 				                |
| Road Work		          | Road Work     							                  |
| Stop		              | Stop     							                        |


The model was able to correctly guess 5 of the 6 traffic signs, which gives an accuracy of 83%. Since there is a smaller test sample the number is lower than the test but it still accurate in classifying the signs. The only error happened with the double curve sign and classified it as a right of way sign. This is not surprising considering that the signs are somewhat similar but further testing can be done to determine the downfalls of the model on this particular class. Despite, I was surprised by how well the model performed on new data and assigned high probabilities to the true classes.

####3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

For the first image, the model is relatively sure that this is right-of-way with a probability of 0.99 and all the other probabilities are relatively low. The top five soft max probabilities are:

| Probability         	|     Prediction	        					|
|:---------------------:|:---------------------------------------------:|
| 0.99        			    | Right-of-way  									|
| 0.0022    				    | Beware of ice										|
| 4.95E-8					      | Slippery road										|
| 1.30E-8	      			  | Pedestrians					 				|
| 8.61E-9				        | Double curve     							|


For the second image, the model correctly predicted the image to be a speed limit 30km/h sign. Intuitively, the other probabilities are all speed limit signs as they are very similar, but it is able to identify with high accuracy that the sign represents 30km/h.  

| Probability         	|     Prediction	        					|
|:---------------------:|:---------------------------------------------:|
| 1.0E+00        			  | Speed Limit 30km/h   									|
| 8.73E-17    				  | Speed Limit 50km/h										|
| 3.61E-18				      | Speed Limit 20km/h											|
| 3.92E-21	      			| Speed Limit 80km/h					 				|
| 3.77E-25				      | End of speed limit      							|

For the third image, the model correctly predicted that the image is a no vehicles traffic sign with a high probability compared to all the others.

| Probability         	|     Prediction	        					|
|:---------------------:|:---------------------------------------------:|
| 1.0E+00        			  | No vehicles   									|
| 3.88E-12    				  | No passing 										|
| 2.59E-12				      | Speed Limit 70km/h											|
| 1.48E-12	      			| Speed Limit 60km/h					 				|
| 1.44E-13				      | Speed Limit 120km/h      							|

For the fourth image, the model incorrectly classified it as a right-of-way sign while the true class was a double curve. The probability for the top class was low meaning which would signal that the model had trouble classifying this image and included classes that share similar features as the true class. The actual class was included in the top five probabilities but given a low value.

| Probability         	|     Prediction	        					|
|:---------------------:|:---------------------------------------------:|
| 0.68        			    | Right-of-way 									|
| 0.15    				      | Beware of ice 										|
| 0.12				          | Wild animals											|
| 0.027	      			    | Double Curve					 				|
| 0.017				          | Pedestrians     							|

For the fifth image, the model correctly classified it as a Road work traffic sign giving it a high probability.

| Probability         	|     Prediction	        					|
|:---------------------:|:---------------------------------------------:|
| 0.99        			    | Road Work 									|
| 7.58E-03   				    | Bicycles crossing										|
| 4.44E-03				      | Wild animals											|
| 2.07E-04	      			| Bumpy road					 				|
| 9.35E-05			        | Road narrows on right     							|

For the sixth image, the model correctly classified it as a stop traffic sign giving it a high probability.

| Probability         	|     Prediction	        					|
|:---------------------:|:---------------------------------------------:|
| 1.0E+00        			  | Stop 									|
| 7.11E-09    				  | Road work 										|
| 1.28E-09				      | Bicycles crossing											|
| 7.77E-11	      			| Speed Limit 60km/h					 				|
| 1.79E-12				      | No entry     							|


Given the performance on the new images, one can conclude that the model is able to classify with high accuracy and generalize to new data. For five of the examples, the probability given to the true class was very high and there was only one image that the model could not classify correctly. Further testing can be done with new images for this class to determine if only this particular image presented a  problem or if the model is not optimized to predict this particular class.
