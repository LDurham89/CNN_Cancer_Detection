# This is a title

## Introduction
The Capstone project due as part of Udacity's Data Science Nanodegree allows students to design their own project and produce a notebook and blog post to go with it.

I chose to do a project that involves building a model to effectively predict if diagnositic imaging samples contain cancerous tissue. As you will no doubt be aware, cancer is one of the leading causes of mortality in the developed world. For example in 2020, cancers accounted for 24% of all deaths in the UK (See here for more detail: https://www.cancerresearchuk.org/health-professional/cancer-statistics/mortality). At the same time, many western countries are facing ageing populations and a shortage of medical professionals. Against this background there are clear benefits to using automated methods of processing diagnostic images to complement the expertise of medical professionals.

## Data
This project will work with data on Invasive Ductal Carcinoma (IDC) -a form of breast cancer- taken from the kaggle page here: https://www.kaggle.com/datasets/paultimothymooney/breast-histopathology-images. The data consists of images, each showing a 50 x 50 pixel region from a breast scan. I will process the data for analysis and then build a model that can distinguish cancerous sub-samples from non-cancerous ones.

The original data consists of nearly 300,000 images and is around 4Gb in size. Unfortunately this is far too big for my machine to process in an acceptable amount of time, and so I created a much smaller sample data set (around 1200 images) that would allow me to learn and complete this project.

A point to note here is that the data is organised by its author in a way that makes it easier to manage computationally, however the way this has been done means that each folder ID is not unique to one patient (see this post https://www.kaggle.com/datasets/paultimothymooney/breast-histopathology-images/discussion/137446) . As a result, I can't draw any inferences about the patients sampled in this data or pick out any demographic trends.

- Show images and the chart of value counts


## Method
In this project I use a deep learning model that utilises Convolution Neural Networks (CNN's). A major advantage of working with CNN's is that they are the industry standard for computer vision and thus there are many tools predeicated on this method, with helpful documentation. Furthermore, they are designed specifically for image analysis. However, there are some alternative methods that I decided not to use.

-One option is to use Recurrent Neural Networks, however these are more appropriate for sequences of information (i.e. there is a temporal dimension). This is why they are frequently used for tasks such as translation - where the order of the information is crucial to its meaning - and analysing videos whereas my data consists of non-sequential photos. Recurrent Neural Networks are also slower than CNN's, which could be an issue given that my final model will use a lot of data.
-Restricted Boltzman models are another option that didn't seem appropriate for this task. This approach appears to be used more for modelling systems using unsupervised learning, although I understand that they can be used for classification tasks. With the data set used in this project labels are available, allowing us to train the model with supervised learning methods, which tend to be more accurate (if interested you can see the discussion here: https://www.ibm.com/cloud/blog/supervised-vs-unsupervised-learning#:~:text=While%20supervised%20learning%20models%20tend,weather%20conditions%20and%20so%20on.)

I will then run several models to find the best parameters and model architecture. I will explain my thinking to justify modifications to model hyperparameters as I go through the iterations.¶

## Metrics

Why accuracy, what else could I have used?

## Model performance


## Conclusions
