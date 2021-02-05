# 0x07. Convolutional Neural Networks

## Learning Objectives
At the end of this project, you are expected to be able to explain to anyone, without the help of Google:

## General

What is a convolutional layer?
What is a pooling layer?
Forward propagation over convolutional and pooling layers
Back propagation over convolutional and pooling layers
How to build a CNN using Tensorflow and Keras

## Requirements
## General

Allowed editors: vi, vim, emacs
All your files will be interpreted/compiled on Ubuntu 16.04 LTS using python3 (version 3.5)
Your files will be executed with numpy (version 1.15) and tensorflow (version 1.12)
All your files should end with a new line
The first line of all your files should be exactly #!/usr/bin/env python3
A README.md file, at the root of the folder of the project, is mandatory
Your code should use the pycodestyle style (version 2.4)
All your modules should have documentation (python3 -c 'print(__import__("my_module").__doc__)')
All your classes should have documentation (python3 -c 'print(__import__("my_module").MyClass.__doc__)')
All your functions (inside and outside a class) should have documentation (python3 -c 'print(__import__("my_module").my_function.__doc__)' and python3 -c 'print(__import__("my_module").MyClass.my_function.__doc__)')
Unless otherwise noted, you are not allowed to import any module
All your files must be executable
The length of your files will be tested using wc

## Tasks

0. Convolutional Forward Prop mandatory
Write a function def conv_forward(A_prev, W, b, activation, padding="same", stride=(1, 1)): that performs forward propagation over a convolutional layer of a neural network:

A_prev is a numpy.ndarray of shape (m, h_prev, w_prev, c_prev) containing the output of the previous layer
m is the number of examples
h_prev is the height of the previous layer
w_prev is the width of the previous layer
c_prev is the number of channels in the previous layer
W is a numpy.ndarray of shape (kh, kw, c_prev, c_new) containing the kernels for the convolution
kh is the filter height
kw is the filter width
c_prev is the number of channels in the previous layer
c_new is the number of channels in the output
b is a numpy.ndarray of shape (1, 1, 1, c_new) containing the biases applied to the convolution
activation is an activation function applied to the convolution
padding is a string that is either same or valid, indicating the type of padding used
stride is a tuple of (sh, sw) containing the strides for the convolution
sh is the stride for the height
sw is the stride for the width
you may import numpy as np
Returns: the output of the convolutional layer






Repo:

GitHub repository: holbertonschool-machine_learning
Directory: supervised_learning/0x07-cnn
File: 0-conv_forward.py
Done?!
Help
×
Students who are done with "1. Pooling Forward Prop"
1. Pooling Forward Prop mandatory
Write a function def pool_forward(A_prev, kernel_shape, stride=(1, 1), mode='max'): that performs forward propagation over a pooling layer of a neural network:

A_prev is a numpy.ndarray of shape (m, h_prev, w_prev, c_prev) containing the output of the previous layer
m is the number of examples
h_prev is the height of the previous layer
w_prev is the width of the previous layer
c_prev is the number of channels in the previous layer
kernel_shape is a tuple of (kh, kw) containing the size of the kernel for the pooling
kh is the kernel height
kw is the kernel width
stride is a tuple of (sh, sw) containing the strides for the pooling
sh is the stride for the height
sw is the stride for the width
mode is a string containing either max or avg, indicating whether to perform maximum or average pooling, respectively
you may import numpy as np
Returns: the output of the pooling layer


Repo:

GitHub repository: holbertonschool-machine_learning
Directory: supervised_learning/0x07-cnn
File: 1-pool_forward.py

Students who are done with "2. Convolutional Back Prop"
2. Convolutional Back Prop mandatory
Write a function def conv_backward(dZ, A_prev, W, b, padding="same", stride=(1, 1)): that performs back propagation over a convolutional layer of a neural network:

dZ is a numpy.ndarray of shape (m, h_new, w_new, c_new) containing the partial derivatives with respect to the unactivated output of the convolutional layer
m is the number of examples
h_new is the height of the output
w_new is the width of the output
c_new is the number of channels in the output
A_prev is a numpy.ndarray of shape (m, h_prev, w_prev, c_prev) containing the output of the previous layer
h_prev is the height of the previous layer
w_prev is the width of the previous layer
c_prev is the number of channels in the previous layer
W is a numpy.ndarray of shape (kh, kw, c_prev, c_new) containing the kernels for the convolution
kh is the filter height
kw is the filter width
b is a numpy.ndarray of shape (1, 1, 1, c_new) containing the biases applied to the convolution
padding is a string that is either same or valid, indicating the type of padding used
stride is a tuple of (sh, sw) containing the strides for the convolution
sh is the stride for the height
sw is the stride for the width
you may import numpy as np
Returns: the partial derivatives with respect to the previous layer (dA_prev), the kernels (dW), and the biases (db), respectively
Repo:

GitHub repository: holbertonschool-machine_learning
Directory: supervised_learning/0x07-cnn
File: 2-conv_backward.py

Students who are done with "3. Pooling Back Prop"
3. Pooling Back Prop mandatory
Write a function def pool_backward(dA, A_prev, kernel_shape, stride=(1, 1), mode='max'): that performs back propagation over a pooling layer of a neural network:

dA is a numpy.ndarray of shape (m, h_new, w_new, c_new) containing the partial derivatives with respect to the output of the pooling layer
m is the number of examples
h_new is the height of the output
w_new is the width of the output
c is the number of channels
A_prev is a numpy.ndarray of shape (m, h_prev, w_prev, c) containing the output of the previous layer
h_prev is the height of the previous layer
w_prev is the width of the previous layer
kernel_shape is a tuple of (kh, kw) containing the size of the kernel for the pooling
kh is the kernel height
kw is the kernel width
stride is a tuple of (sh, sw) containing the strides for the pooling
sh is the stride for the height
sw is the stride for the width
mode is a string containing either max or avg, indicating whether to perform maximum or average pooling, respectively
you may import numpy as np
Returns: the partial derivatives with respect to the previous layer (dA_prev)
Repo:

GitHub repository: holbertonschool-machine_learning
Directory: supervised_learning/0x07-cnn
File: 3-pool_backward.py

Students who are done with "4. LeNet-5 (Tensorflow)"
4. LeNet-5 (Tensorflow) mandatory


Write a function def lenet5(x, y): that builds a modified version of the LeNet-5 architecture using tensorflow:

x is a tf.placeholder of shape (m, 28, 28, 1) containing the input images for the network
m is the number of images
y is a tf.placeholder of shape (m, 10) containing the one-hot labels for the network
The model should consist of the following layers in order:
Convolutional layer with 6 kernels of shape 5x5 with same padding
Max pooling layer with kernels of shape 2x2 with 2x2 strides
Convolutional layer with 16 kernels of shape 5x5 with valid padding
Max pooling layer with kernels of shape 2x2 with 2x2 strides
Fully connected layer with 120 nodes
Fully connected layer with 84 nodes
Fully connected softmax output layer with 10 nodes
All layers requiring initialization should initialize their kernels with the he_normal initialization method: tf.contrib.layers.variance_scaling_initializer()
All hidden layers requiring activation should use the relu activation function
you may import tensorflow as tf
you may NOT use tf.keras
Returns:
a tensor for the softmax activated output
a training operation that utilizes Adam optimization (with default hyperparameters)
a tensor for the loss of the netowrk
a tensor for the accuracy of the network


Repo:

GitHub repository: holbertonschool-machine_learning
Directory: supervised_learning/0x07-cnn
File: 4-lenet5.py

Students who are done with "5. LeNet-5 (Keras)"
5. LeNet-5 (Keras) mandatory
Write a function def lenet5(X): that builds a modified version of the LeNet-5 architecture using keras:

X is a K.Input of shape (m, 28, 28, 1) containing the input images for the network
m is the number of images
The model should consist of the following layers in order:
Convolutional layer with 6 kernels of shape 5x5 with same padding
Max pooling layer with kernels of shape 2x2 with 2x2 strides
Convolutional layer with 16 kernels of shape 5x5 with valid padding
Max pooling layer with kernels of shape 2x2 with 2x2 strides
Fully connected layer with 120 nodes
Fully connected layer with 84 nodes
Fully connected softmax output layer with 10 nodes
All layers requiring initialization should initialize their kernels with the he_normal initialization method
All hidden layers requiring activation should use the relu activation function
you may import tensorflow.keras as K
Returns: a K.Model compiled to use Adam optimization (with default hyperparameters) and accuracy metrics


Repo:

GitHub repository: holbertonschool-machine_learning
Directory: supervised_learning/0x07-cnn
File: 5-lenet5.py

6. Summarize Like a Pro mandatory
A common practice in the machine learning industry is to read and review journal articles on a weekly basis. Read and write a summary of Krizhevsky et. al.‘s 2012 paper ImageNet Classification with Deep Convolutional Neural Networks. Your summary should include:

Introduction: Give the necessary background to the study and state its purpose.
Procedures: Describe the specifics of what this study involved.
Results: In your own words, discuss the major findings and results.
Conclusion: In your own words, summarize the researchers’ conclusions.
Personal Notes: Give your reaction to the study.
Your posts should have examples and at least one picture, at the top. Publish your blog post on Medium or LinkedIn, and share it at least on Twitter and LinkedIn.

When done, please add all URLs below (blog post, tweet, etc.)

Please, remember that these blogs must be written in English to further your technical ability in a variety of settings.

Add URLs here:
