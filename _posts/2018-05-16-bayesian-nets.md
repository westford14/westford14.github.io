---
layout: post
title: Bayesian Neural Networks in Survival Analysis
--- 

Bayesian Neural Networks and You

#### Overview

![Image of Bayes Theorem](https://www.kdnuggets.com/wp-content/uploads/bayesian-neon.jpeg)

Bayesian neural networks definitely sound like a strange combination of two different data science buzzwords, but they can have some very interesting applications.  A typical feed-forward neural network will have, somtimes, many layers of neurons connected to each other.  These connections have weights and biases associated with them, and the constant adjustment of these weights is how the network learns.  The predictive power of neural networks cannot be argued with, but there is an ineteresting issue with them.  Let's say for a binary classification problem (ie. a presented image is either a dog or not a dog) you want to get the uncertainty associated with this prediction.  A typical neural network will only give you one prediction, which doesn't help you address this issue.  This is where Bayesian neural networks can help.

In Bayesian neural networks there is a prior distribution over the weights.  By utilizing this probabilistic approach to training, we are now able to consider the entire distribution for our predictions.  Now, when we want to know the uncertainity of our binary classifier, we can now get a thousand guesses from the neural network and, thus, create confidence intervals of its predictive power.  A more indepth explanation of Bayesian neural networks can be found [here](https://arxiv.org/abs/1801.07710) and [here](https://www.cs.toronto.edu/~radford/bnn.book.html). 


#### Use Case

In epidemiology we love working with cohort data.  Cohort data being repeated measures of the same population over a long stretch of time.  We like working with this data because it gives us an insight into how people's health changes over time and what lifestyle choices they made that impacted their health.  One thing that is particularly interesting that I've been working on is trying to predict when someone might have a cardiac event.  Instead of purely trying to predict whether or not someone will have an event, there is more power in being able to say that someone will have an event in 5 years given their current characteristics.  For example, let's say you have the [NHANES Epidemiologic Followup Study](https://wwwn.cdc.gov/nchs/nhanes/nhefs/default.aspx/), and you want to be able to make this type of prediction; I will show you how this can be done!

#### Code 

This code is all in Python and uses the [Pyro](http://pyro.ai/) and [PyTorch](https://pytorch.org/) libraries. Let's import the necessary libraries:

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

import pyro
from pyro.distributions import Normal
from pyro.infer import SVI
from pyro.optim import Adam

from tqdm import tqdm_notebook
```

I'm using Pyro version 0.1.2 and PyTorch version 0.3.1.post2.

I'm making assumptions about the dataset that we're using, but let's say we have an ideal, clean dataset that has a bunch of columns with baseline characteristics.  These might include the person's age, sex, average consumption of different types of foods, whether or not this person is a smoker, etc.  I'm also assuming that there is a column that has the number of months until the person had a cardiac event.  This ranges from 0 (no event) to 60 for example.  The number 60 says that particular person had a cardiac event 5 years into the study.  We will then want to create our dataset where we one-hot encode the time variable.  This changes the problem from a binary classification into a categorical classification problem.  I've also found I get better performance when I scale the other values in the dataset rather than leave them at their original value.

```python
data = pd.read_csv('example.csv')
y = np.around(data['y'].values)
enc = OneHotEncoder()
y_hot = enc.fit_transform(y.reshape(-1, 1)).toarray()

temp = data.drop('y', axis=1)
X = temp[temp.columns.tolist()[1:]].fillna(0)
scale = StandardScaler()
X = scale.fit_transform(X)
```

Let's now create our models.  This is a really simple feedforward network: 

```python
class BNN(nn.Module):
    def __init__(self, columns, outputs=1):
        super(BNN, self).__init__()
        hidden = round(columns * .75)
        self.hidden = nn.Linear(columns, hidden)
        self.predict = nn.Linear(hidden, outputs)
        
    def forward(self, x):
        x = F.selu(self.hidden(x))
        x = F.selu(self.predict(x))
        return x
```

We then instantiate the `model` and `guide` objects. The model function will define priors over for the weights and biases for each layer of our neural network.  The priors will be pulled from a Normal distribution between 0 and 1.  I haven't tested changing distributions and changing scales yet, but that is definitely a next step to explore.

```python
def model(X, y):
    x_data = X
    y_data = y
    
    # First Layer
    mu = Variable(torch.zeros(second_layer, first_layer))
    sigma = Variable(torch.ones(second_layer, first_layer))
    bias_mu = Variable(torch.zeros(second_layer))
    bias_sigma = Variable(torch.ones(second_layer))
    w_prior, b_prior = Normal(mu, sigma), Normal(bias_mu, bias_sigma)
    
    # Second Layer
    mu2 = Variable(torch.zeros(1, second_layer))
    sigma2 = Variable(torch.ones(1, second_layer))
    bias_mu2 = Variable(torch.zeros(1))
    bias_sigma2 = Variable(torch.ones(1))
    w_prior2, b_prior2 = Normal(mu2, sigma2), Normal(bias_mu2, bias_sigma2)
    
    priors = {"hidden.weight": w_prior,
             "hidden.bias": b_prior,
             "predict.weight": w_prior2,
             "predict.bias": b_prior2}
    
    lifted_module = pyro.random_module("module", bnn, priors)
    lifted_reg_model = lifted_module()
    
    prediction_mean = lifted_reg_model(x_data).squeeze()
    pyro.sample("obs",
               Normal(prediction_mean, Variable(torch.ones(x_data.size(0)))),
               obs=y_data.squeeze())
``` 

But, in order to learn, we must "guide" the model.  This can be thought of as a parameterized family of distributions over the weights and biases.  We also will make these parameters trainable so that the network can learn. 

```python
def guide(X, y):
    #First Layer
    w_mu = Variable(torch.randn(second_layer, first_layer), requires_grad=True)
    w_log_sig = Variable(0.1 * torch.ones(second_layer, first_layer), requires_grad=True)
    b_mu = Variable(torch.randn(second_layer), requires_grad=True)
    b_log_sig = Variable(0.1 * torch.ones(second_layer), requires_grad=True)
    
    mw_param = pyro.param('guide_mean_weight', w_mu)
    sw_param = softplus(pyro.param("guide_log_sigma_weight", w_log_sig))
    mb_param = pyro.param('guide_mean_bias', b_mu)
    sb_param = softplus(pyro.param("guide_log_sigma_bias", b_log_sig))
    
    w_dist = Normal(mw_param, sw_param)
    b_dist = Normal(mb_param, sb_param)
    
    #Second Layer
    w_mu2 = Variable(torch.randn(1, second_layer), requires_grad=True)
    w_log_sig2 = Variable(0.1 * torch.randn(1, second_layer), requires_grad=True)
    b_mu2 = Variable(torch.randn(1), requires_grad=True)
    b_log_sig2 = Variable(0.1 * torch.ones(1), requires_grad=True)
    
    mw_param2 = pyro.param("guide_mean_weight2", w_mu2)
    sw_param2 = softplus(pyro.param("guide_log_sigma_weight2", w_log_sig2))
    mb_param2 = pyro.param("guide_mean_bias2", b_mu2)
    sb_param2 = softplus(pyro.param("guide_log_sigma_bias2", b_log_sig2))
    
    w_dist2 = Normal(mw_param2, sw_param2)
    b_dist2 = Normal(mb_param2, sb_param2)
    
    dists = {
        "hidden.weight": w_dist,
        "hidden.bias": b_dist,
        "predict.weight": w_dist2,
        "predict.bias": b_dist2
    }
    
    lifted_module = pyro.random_module("module", bnn, dists)
    return lifted_module()
```

Whew.  Now that we have set everything up we can then declare our network and uses [stochastic variational inference](http://www.columbia.edu/~jwp2128/Papers/HoffmanBleiWangPaisley2013.pdf) to train our model.  I'm also using the [Adam](https://arxiv.org/abs/1412.6980) optimizer to train the model.  I'm also storing the loss after every thousandth epoch, so I can subsequently plot the losses to ensure that the model is actually learning over time.

```python
softplus = nn.Softplus()

first_layer = X_train.shape[1]
second_layer = round(first_layer * .75)
bnn = BNN(columns=first_layer, outputs=y_hot.shape[1])

optim = Adam({"lr": 0.001})
svi = SVI(model, guide, optim, loss="ELBO")

pyro.clear_param_store()
loss = []
for j in tqdm_notebook(range(20000)):
    ep_loss = 0.0
    
    perm = np.random.permutation(len(X_train))
    x_epoch = X_train[perm]
    y_epoch = np.argmax(y_train[perm], axis=1)
    
    ep_loss += svi.step(Variable(torch.Tensor(x_epoch)),
                        Variable(torch.Tensor(y_epoch)))
    loss.append(ep_loss / len(X_train))
    
    if j % 1000 == 0:
        print("Epoch {0}, average loss: {1:.4f}".format(j, ep_loss / len(X_train)))
```

After the 20,000 iterations, we can see our loss has drastically decreased.  

![Graph of Loss](https://raw.githubusercontent.com/westford14/westford14.github.io/master/images/loss_convergence.png)

We can also now predict how long into the future someone might have a cardiac event.  But, most importantly, we get uncertainty around this.  So if we make a thousand predictions based of one person's characteristics, we can now describe the distribution of time intervals that an event may happen in!  Pretty exciting! 

```python 
preds = []
for i in tqdm_notebook(range(1000)):
    sampled_reg_model = guide(Variable(torch.Tensor(X_test)), Variable(torch.Tensor(y_test)))
    pred = sampled_reg_model(Variable(torch.Tensor(X_test)))
    preds.append(pred)

preds = [softplus(x).data.numpy().flatten() for x in preds]
mean = np.mean(preds, axis=0)
low = np.percentile(preds, 2.5, axis=0) 
high = np.percentile(preds, 97.5, axis=0) 
```

If we're happy with the performance we can save the model and trained parameters for later use.

```python
torch.save(bnn, 'best_model.p')
pyro.get_param_store().save('best_params.save')
```

#### Summary

In this post, I've given an overview of why we might want to use Bayesian neural networks, how to implement them in Pyro, and how to predict off the trained network.  The combination of probabilitistic programming and neural networks is super cool, and I'm looking forward to see what other uses across different fields people will have for Bayesian neural networks.