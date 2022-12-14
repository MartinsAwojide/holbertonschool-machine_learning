# 0x09\. Transfer Learning

## Learning Objectives

At the end of this project, you are expected to be able to [explain to anyone], **without the help of Google**:

### General

*   What is a transfer learning?
*   When to use transfer learning
*   How to use transfer learning with Keras

## Requirements

### General

*   Allowed editors: `vi`, `vim`, `emacs`
*   All your files will be interpreted/compiled on Ubuntu 16.04 LTS using `python3` (version 3.5)
*   Your files will be executed with `numpy` (version 1.15) and `tensorflow` (version 1.12)
*   All your files should end with a new line
*   The first line of all your files should be exactly `#!/usr/bin/env python3`
*   A `README.md` file, at the root of the folder of the project, is mandatory
*   Your code should use the `pycodestyle` style (version 2.4)
*   All your modules should have documentation (`python3 -c 'print(__import__("my_module").__doc__)'`)
*   All your classes should have documentation (`python3 -c 'print(__import__("my_module").MyClass.__doc__)'`)
*   All your functions (inside and outside a class) should have documentation (`python3 -c 'print(__import__("my_module").my_function.__doc__)'` and `python3 -c 'print(__import__("my_module").MyClass.my_function.__doc__)'`)
*   Unless otherwise noted, you are not allowed to import any module except `import tensorflow.keras as K`
*   All your files must be executable
*   The length of your files will be tested using `wc`


* * *

## Tasks


#### 0\. Transfer Knowledge <span class="alert alert-warning mandatory-optional">mandatory</span>

Write a python script that trains a convolutional neural network to classify the CIFAR 10 dataset:

*   You must use one of the applications listed in [Keras Applications]
*   Your script must save your trained model in the current working directory as `cifar10.h5`
*   Your saved model should be compiled
*   Your saved model should have a validation accuracy of 88% or higher
*   Your script should not run when the file is imported
*   _Hint: The training may take a while, start early!_

In the same file, write a function `def preprocess_data(X, Y):` that pre-processes the data for your model:

*   `X` is a `numpy.ndarray` of shape `(m, 32, 32, 3)` containing the CIFAR 10 data, where m is the number of data points
*   `Y` is a `numpy.ndarray` of shape `(m,)` containing the CIFAR 10 labels for `X`
*   Returns: `X_p, Y_p`
    *   `X_p` is a `numpy.ndarray` containing the preprocessed `X`
    *   `Y_p` is a `numpy.ndarray` containing the preprocessed `Y`
```
    alexa@ubuntu-xenial:0x09-transfer_learning$ cat 0-main.py
    #!/usr/bin/env python3

    import tensorflow.keras as K
    preprocess_data = __import__('0-transfer').preprocess_data

    # fix issue with saving keras applications
    K.learning_phase = K.backend.learning_phase 

    _, (X, Y) = K.datasets.cifar10.load_data()
    X_p, Y_p = preprocess_data(X, Y)
    model = K.models.load_model('cifar10.h5')
    model.evaluate(X_p, Y_p, batch_size=128, verbose=1)
    alexa@ubuntu-xenial:0x09-transfer_learning$ ./0-main.py
    10000/10000 [==============================] - 159s 16ms/sample - loss: 0.3329 - acc: 0.8864
```
**Repo:**

*   GitHub repository: `holbertonschool-machine_learning`
*   Directory: `0x09-transfer_learning`
*   File: `0-transfer.py`



#### 1\. "Research is what I'm doing when I don't know what I'm doing." - Wernher von Braun <span class="alert alert-warning mandatory-optional">mandatory</span>

Write a blog post explaining your experimental process in completing the task above written as a journal-style scientific paper.
