#**Behavioral Cloning** 

##Writeup for Alistair Kirk - Project 3 - Self Driving Car NanoDegree course

---

**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./writeup/NvidiaArch.png "Model Visualization"
[image2]: ./writeup/Graph_MSEvsEPOCHS.png "MSE vs. Epochs"
[image3]: ./writeup/center.jpg "Center Driving Image"
[image4]: ./writeup/off_center.jpg "Off C Image"
[image5]: ./writeup/off_left.jpg "Off L Image"
[image6]: ./writeup/off_right.jpg "Off R Image"
[image7]: ./writeup/flipped.jpg "Flipped Image"
[image8]: ./writeup/recover_center.jpg "Recover C Image"
[image9]: ./writeup/recover_left.jpg "Recover L Image"
[image10]: ./writeup/recover_right.jpg "Recover R Image"

## Rubric Points
###Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
###Files Submitted & Code Quality

####1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* P3 Train - ASK.ipynb containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network 
* writeup_report.md summarizing the results

####2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.h5
```

####3. Submission code is usable and readable

The P3 Train - ASK notebook file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

###Model Architecture and Training Strategy

####1. An appropriate model architecture has been employed

My final model consists of 5 convolution neural network layers. In order:
3 CNN's with 5x5 filter sizes, 2x2 stride, and increasing depths of 24, 36 and 48
Dropout Layer with 0.2 probability
1 CNN with 3x3 filter size, 1x1 stride, and depth of 64
Dropout Layer with 0.3 probability
1 CNN with 3x3 filter size, 1x1 stride, and depth of 64
Dropout Layer with 0.5 probability
Flatten Layer
Dense Layer with 1164 nodes
Dense Layer with 100 nodes
Dense Layer with 50 nodes
Dense Layer with 10 nodes
Dense Layer with 1 node

The model includes RELU layers to introduce nonlinearity, and the data is normalized in the model using a Keras lambda layer.

I also used generator fit functions to batch through the images using my local GPU for learning (NVIDIA GTX 770). Batch sizes were set at 32. Input image shape was 160x320x3 as defined by the Udacity Simulator image output.

####2. Attempts to reduce overfitting in the model

The model contains dropout layers in order to reduce overfitting. 

The model was trained and validated on different data sets to ensure that the model was not overfitting. The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

####3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually (model.py line 25). This was taken from the example classroom material.

####4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. I used a combination of center lane driving, recovering from the left and right sides of the road, and driving in the reverse direction on both tracks.

For details about how I created the training data, see the next section. 

###Model Architecture and Training Strategy

####1. Solution Design Approach

The overall strategy for deriving a model architecture was to start with a basic architecture that I knew could at least train and create an output model from previous projects. I would then progressively test and change the neural network and find a better optimized architecture, supplementing with more learning data along the way.

My first step was to use a convolution neural network model similar to the Lenet Architecture used in previous projects. I thought this model might be appropriate because the architecture proved usefule for both digit recoginition and traffic sign classification. A reasonable first step would be to try and use it for this image recognition project.

This approach was rudimentary and resulted in fairly poor driving performance. I then followed the guidance of exploring the NVIDIA self driving car architecture found here: [NVIDIA](https://devblogs.nvidia.com/parallelforall/deep-learning-self-driving-cars/)

The NVIDIA architecture performed substantially better, however the vanilla architecture as shown in their figure could be further improved upon.

In order to gauge how well the model was working, I split my image and steering angle data into a training and validation set. I found that my first model had a low mean squared error (MSE) on the training set but a high MSE on the validation set. This implied that the model was overfitting. 

To combat the overfitting, I modified the model by adding dropout between the convolutional layers. I also reduced the number of epochs from 10 to 6. Both of these techniques together seemed to reduce the MSE on the validation set.

I made sure to crop the images to only the area of interest (cropping each image to between 25% and 70% in the vertical direction) to remove the sky and front bumper area of the car. This would reduce the amount of raw data being analyzed and also remove spurious features that the architecture may pick up.

I made sure to normalize the images using a lambda function in Keras, specifically I divided each pixel by 255 and substracted 0.5 to improve the gradient decent performance by normalizing around zero.

I also augmented my datasets by copying and flipping each image horizontally, and adding the negative of the steering angle:
```sh
aug_images.append(cv2.flip(image,1))
aug_angles.append(angle*-1.0)
```
I additionally augmented the data by taking the left and right camera images and applying a scalar steering angle correction factor of 0.35:
```sh
steering_center = measurement
steering_left = steering_center + correction
steering_right = steering_center - correction
steering_correction = [steering_center, steering_left, steering_right]
```

Augmenting as above increases the number of training data by 6.

I created a Keras generator function with batch sizing of 32 to allow for processing on my local GPU. Care had to be taken to adjust the syntax from Keras 1.1 to the latest 2.0 version as they changed the sematics to steps per epoch, which required dividing by the original variable name (length of train\_samples) by the batch size.

After training I saved my model to an h5 file.

The final step was to run the simulator to see how well the car was driving around track one. There were a few spots where the vehicle fell off the track... to improve the driving behavior in these cases, I added more training data by driving in both directions on the course, and by driving on the second track provided.

At the end of the process, the vehicle is able to drive autonomously around the track without leaving the road.

####2. Final Model Architecture

The final model architecture (model.py lines 18-24) consisted of a convolution neural network with the layers and layer sizes as described above.

Here is a visualization of the architecture:

![P3ArchitectureASK][image1]

####3. Creation of the Training Set & Training Process

To capture good driving behavior, I first recorded two laps on track one using center lane driving. Here is an example image of center lane driving:

![alt text][image3]

I then recorded the vehicle recovering from the left side and right sides of the road back to center so that the vehicle would learn to correct its behaviour by turning sharply if it approached the sides too closely. These images show what a recovery looks like starting from a turn on the second track:

![alt text][image4]
![alt text][image5]
![alt text][image6]

...and followed by a few snapshots from seconds later:

![alt text][image8]
![alt text][image9]
![alt text][image10]

Then I repeated this process on track two in order to get more data points.

To augment the data set, I also flipped images and angles thinking that this would remove any dependence on directional turns like driving around a left handed track. For example, here is an image that has then been flipped:

![flipped][image7]

After the collection process, I had 19,755 \*3 = 59,265 number of data points. I then preprocessed this data by cropping the images to remove sky and foreground, and normalized the pixels to a mean of zero with std. deviation of 0.5

I finally randomly shuffled the data set and put 20% of the data into a validation set. 

I used this training data for training the model. The validation set helped determine if the model was over or under fitting. The ideal number of epochs was 4 as evidenced by the fact that continued training tended to favour the left handed turning on the first track and the model tended to edge itself to the left of center. I used an adam optimizer so that manually training the learning rate wasn't necessary.

A graph of the number of epohcs vs. MSE using the generator history is shown below:

![msevsepochs][image2]
