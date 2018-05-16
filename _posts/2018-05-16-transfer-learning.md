---
layout: post
title: Transfer Learning
---

Transfer Learning for the Lazy

#### Overview

Transfer learning, simply, is using a model developed for one task and applying what it's learned to a new task.  A frequent example, and one that I've used frequently, is for image recognition.  Below is a summary of some of the more common convolutional neural network architectures and their accraucy on [ImageNet](http://image-net.org/challenges/LSVRC/2016/index).  

![ImageNet Accuracies](https://cdn-images-1.medium.com/max/1600/1*ZqkLRkMU2ObOQWIHLBg8sw.png)

ImageNet is a massive dataset with images belonging to 1000 different classes, and the models pictured above are trained for hundreds of hours on multiple GPUs.  Obviously, you could train your `Inception` if you wanted to, but this would take weeks, as warned on the [TensorFlow GitHub](https://github.com/tensorflow/models/tree/master/research/inception).  Instead it makes more sense to just use the learned weights from ImageNet and apply it to your own dataset.  The general idea for training the model is "freezing" the bottom layers of the network (ie. don't train / update these layers), and then apply a few more layers that are trainable on top of these frozen layers.   

At [IHME](http://www.healthdata.org/), this is the strategy I took for the anomaly detection process I created.  Below are a few snippets of code that you can apply to your data as well!

#### Code

```python
import numpy as np

import keras.backend as K
from keras.models import Model
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D
from keras.applications.inception_v3 import InceptionV3
```

These imports will help get the Inception v3 weights, add a few more layers on top of Inception, and retrain on your new dataset.

```python
K.clear_session()

incept = InceptionV3(weights='imagenet', include_top=False, 
                         input_tensor=Input(shape=(299, 299, 3)))
transfer = incept.output
transfer = Convolution2D(32, 3, 3))(transfer)
transfer = AveragePooling2D(pool_size=(2, 2))(transfer)
transfer = Dropout(.2)(transfer)
transfer = Flatten()(transfer)
predictions = Dense(2, init='glorot_uniform', W_regularizer=l2(.0005), 
                    activation='softmax')(transfer)

model = Model(input=incept.input, output=predictions)
opt = SGD(lr=.01, momentum=.9)
model.compile(optimizer=opt, loss='categorical_crossentropy', 
              metrics=['accuracy', 'mae'])
```

I have a more fleshed out example on the [IHME GitHub](https://github.com/ihmeuw/ihme_dl/blob/master/ihme_deeplearning.ipynb), but that's all you need to do!  

#### Summary

As you can see, this is a very easy way of getting a very high performing model with very little effort.  We're basically taking the hardwork, and many GPU hours of training, that other people have done and then applying it to our data and getting great accuracy.