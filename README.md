# Dog Breed Identification
Determine the breed of a dog in an image

[![Machine Learning](https://img.shields.io/badge/%F0%9F%A4%96-Machine_Learning-black)](https://en.wikipedia.org/wiki/Machine_learning)
[![Deep Learning](https://img.shields.io/badge/%F0%9F%A4%96-Deep_Learning-orange)](https://en.wikipedia.org/wiki/Deep_learning)
[![Neural Networks](https://img.shields.io/badge/%F0%9F%A7%A0-Neural_Networks-pink)](https://en.wikipedia.org/wiki/Artificial_neural_network)
[![Transfer Learning](https://img.shields.io/badge/%E2%9A%97%EF%B8%8F-Transfer_Learning-green)](https://en.wikipedia.org/wiki/Transfer_learning)
[![Udemy](https://img.shields.io/badge/%F0%9F%8E%93-Udemy-a435f0)](https://www.udemy.com/)
[![Kaggle](https://img.shields.io/badge/%F0%9F%92%BB-Kaggle-20beff)](https://www.kaggle.com/)
[![Dog Breeds](https://img.shields.io/badge/%F0%9F%90%BE-Dog_Breeds-lightgrey)](https://en.wikipedia.org/wiki/List_of_dog_breeds)

![Python](https://img.shields.io/badge/Python-informational?style=flat&logo=python&logoColor=f7db5d&color=326998)
![Tensorflow](https://img.shields.io/badge/Tensorflow-informational?style=flat&logo=tensorflow&color=326998)

[![GPU](https://img.shields.io/badge/%F0%9F%8E%AE-TensorFlow_with_GPU-FF9300)](https://www.tensorflow.org/guide/gpu)

## Course
This project was taken from [Complete A.I. & Machine Learning, Data Science Bootcamp](https://www.udemy.com/course/complete-machine-learning-and-data-science-zero-to-mastery/?couponCode=KEEPLEARNING), specifically the module **Neural Networks: Deep Learning, Transfer Learning and TensorFlow 2**.

In this project we will use a GPU to create a neural network and train it using TensorFlow and Transfer Data with the data downloaded from [Kaggle **Closed** Competition](https://www.kaggle.com/c/dog-breed-identification/overview) to the folders `/data/train` and `/data/test`, we will follow the requirements of the competition to analyze the data and create the models necessary to identify the dog breed listed in the `/data/labels.csv`.

## Competition Overview
### Description
> Who's a good dog? Who likes ear scratches? Well, it seems those fancy deep neural networks don't have all the answers. However, maybe they can answer that ubiquitous question we all ask when meeting a four-legged stranger: what kind of good pup is that?
> 
> In this playground competition, you are provided a strictly canine subset of ImageNet in order to practice fine-grained image categorization. How well you can tell your Norfolk Terriers from your Norwich Terriers? With 120 breeds of dogs and a limited number training images per class, you might find the problem more, err, ruff than you anticipated.

![Dogos](./assets/border_collies.png)

### Evaluation
Submissions are evaluated on [Multi Class Log Loss](https://www.kaggle.com/wiki/MultiClassLogLoss) between the predicted probability and the observed target.

## Features
Some information about the data
* We're dealing with images(unstructured data)
* There are 120 breed of dogs (Meaning there are 120 different classes)
* There are around 10k+ images in the training set(This set has labels)
* There are around 10k+ images in the test set(This set doesn't have labels)

## Notes
For Google Colab working environment you can check the following resources:
* [Welcome To Colaboratory](https://colab.research.google.com/notebooks/intro.ipynb)
* [External data: Local Files, Drive, Sheets, and Cloud Storage](https://colab.research.google.com/notebooks/io.ipynb)
* [Google Colab FAQ](https://research.google.com/colaboratory/faq.html)
* [TensorFlow with GPU](https://colab.research.google.com/notebooks/gpu.ipynb)

## Warnings
* In case you choose to work using VS Code, Anaconda or any other local environment you might use this repository and check information about how to use a GPU with Tensorflow(Link in the badges) and information on GPU Capability from [NVIDIA](https://developer.nvidia.com/cuda-gpus)

* Since the data from the `dog-breed-identification.zip` is to much if I upload it unzipped I get up to 10k+ changes to commit, therefore I added `*.jpg` to the `.gitignore` file, you can go to the following link to the [Kaggle Competition Data tab](https://www.kaggle.com/c/dog-breed-identification/data) and either choose the files you want to download or click on the `Download All` button to get the `train/*.jpg`, `test/*.jpg`, `labels.csv` and `sample_submission.csv`
