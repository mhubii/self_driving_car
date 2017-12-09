# Self-Driving Car
In this short project we aim to implement a self-driving car algorithm within a simulation environment.

## The Network
Our network architecture for the self-driving car is based on the [End to End Learning for Self-Driving Cars](https://arxiv.org/pdf/1604.07316.pdf "Link to Paper") paper which was released by NVIDIA. Note that all of the following pictures are taken from the paper as well.

### Training
The convolutional network is trained in a supervised manner using backpropagation. It therefore predicts the steering angle and the acceleration from the recorded input images. Then, the error between predicted actions and actions, which correspond to the recorded input images, is computed.

![Training](/img/training.png)

### Architecture

![Architecture](/img/net_architecture.png)

### Prediction

![Prediction](/img/prediction.png)

## The Environment
As environment we use a car driving simulation built by Udacity. The simulation can be found on their GitHub if you follow this [link](https://github.com/udacity/self-driving-car-sim "Link to GitHub").
