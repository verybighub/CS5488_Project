# CS5488 Project - Price Prediction on Cryptocurrencies

## Objective
In this project, we attempt to forecast the prices of cryptocurrencies using multiple historical features, by training a deep neural network in a distributed manner across several nodes. 

Distributed training is a collection of techniques for using multiple processors located on different machines for training machine learning models. It is an increasingly important deep learning technique, since it enables the training of wider neural networks which is too cumbersome to manage on one to machine only. 

## Preliminary Plan
The [**Horovod** library](https://github.com/horovod/horovod) ([paper](https://towardsdatascience.com/paper-summary-horovod-fast-and-easy-distributed-deep-learning-in-tensorflow-5be535c748d1)) will be used and the training machines will be grouped by an [**Apache Spark** cluster](https://horovod.readthedocs.io/en/stable/spark_include.html) (which will be covered in the last few weeks of the lecture). Horovod distributes training batches to machines for training, averages the gradients of gradient descents, and aggregates the validation metrics returned by each machine. It supports common deep learning frameworks like **Keras**, **TensorFlow** and **PyTorch**.

We will compare the convergence rates with and without distributed training using Tensorboard.

## Requirements
To install the dependencies, enter the following in the command line:
```
pip3 install -r requirements.txt
```
