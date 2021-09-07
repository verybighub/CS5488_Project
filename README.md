# CS5488 Project

## Objective
In this project, we attempt to predict the exchange rates of cryptocurrencies using historical prices of other cryptocurrencies, by training a deep neural network distributedly across several machines.

## Preliminary Plan
The [**Horovod** library](https://github.com/horovod/horovod) will be used and the GPU-equipped machines will be grouped by an [**Apache Spark** cluster](https://horovod.readthedocs.io/en/stable/spark_include.html) (which will be covered in the last few weeks of the lecture). Horovod distributes training batches to machines for training, averages the gradients of gradient descents, and aggregates the validation metrics returned by each machine. It supports common deep learning frameworks like **Keras**, **TensorFlow** and **PyTorch**.

We will compare the convergence rates with and without distributed training.

## Requirements
To install the dependencies, enter the following in the command line:
```
pip3 install -r requirements.txt
```
