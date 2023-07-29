# This is a title

## Introduction
The Capstone project due as part of Udacity's Data Science Nanodegree allows students to design their own project and produce a notebook and blog post to go with it.

I chose to do a project that involves building a model to effectively predict if diagnositic imaging samples contain cancerous tissue. As you will no doubt be aware, cancer is one of the leading causes of mortality in the developed world. For example in 2020, cancers accounted for 24% of all deaths in the UK (See here for more detail: https://www.cancerresearchuk.org/health-professional/cancer-statistics/mortality). At the same time, many western countries are facing ageing populations and a shortage of medical professionals. Against this background there are clear benefits to using automated methods of processing diagnostic images to complement the expertise of medical professionals.

## Data
This project will work with data on Invasive Ductal Carcinoma (IDC) -a form of breast cancer- taken from the kaggle page here: https://www.kaggle.com/datasets/paultimothymooney/breast-histopathology-images. The data consists of images, each showing a 50 x 50 pixel region from a breast scan. Images are classed as 0 for non-cancerous samples and 1 for cancerous samples. I will process the data for analysis and then build a model that can distinguish cancerous sub-samples from non-cancerous ones.

The original data consists of nearly 300,000 images and is around 4Gb in size. Unfortunately this is far too big for my machine to process in an acceptable amount of time, and so I created a much smaller sample data set (around 1200 images) that would allow me to learn and complete this project.

A point to note here is that the data is organised by its author in a way that makes it easier to manage computationally, however the way this has been done means that each folder ID is not unique to one patient (see this post https://www.kaggle.com/datasets/paultimothymooney/breast-histopathology-images/discussion/137446) . As a result, this project doesn't try to draw any inferences about the patients sampled in this data, or pick out any demographic trends.

To give you a better idea of what images are being used here, the figure below presents a few samples from each class:

<img src="/assets/class0_take2.jpg" alt="non-cancerous images" width="750" height="200">

<img src="/assets/class1.jpg" alt="cancerous images" width="750" height="200">


- Show images and the chart of value counts


## Method
In this project I use a deep learning model that utilises Convolution Neural Networks (CNN's). A major advantage of working with CNN's is that they are the industry standard for computer vision and thus there are many tools predeicated on this method, with helpful documentation. Furthermore, they are designed specifically for image analysis. However, there are some alternative methods that I decided not to use.

-One option is to use Recurrent Neural Networks, however these are more appropriate for sequences of information (i.e. there is a temporal dimension). This is why they are frequently used for tasks such as translation - where the order of the information is crucial to its meaning - and analysing videos whereas my data consists of non-sequential photos. Recurrent Neural Networks are also slower than CNN's, which could be an issue given that my final model will use a lot of data.
-Restricted Boltzman models are another option that didn't seem appropriate for this task. This approach appears to be used more for modelling systems using unsupervised learning, although I understand that they can be used for classification tasks. With the data set used in this project labels are available, allowing us to train the model with supervised learning methods, which tend to be more accurate (if interested you can see the discussion here: https://www.ibm.com/cloud/blog/supervised-vs-unsupervised-learning#:~:text=While%20supervised%20learning%20models%20tend,weather%20conditions%20and%20so%20on.)

I will then run several models to find the best parameters and model architecture. I will explain my thinking to justify modifications to model hyperparameters as I go through the iterations.¶

## Metrics

Why accuracy, what else could I have used?

## Model performance
To evaluate these it would be tempting to present a chart showing the accuracy scores of each model, however this would miss the relationship between the training and validation data across across the model. While the accuracy scores have improved most (but not all) times that we've added complexity to the model the behaviour of the validation data has not been great in many of the models. In many models tha validation loss has plateaued very early on, while the loss on the training data has continued to fall, which suggests that the model is likely to be overfitting. In other models the validation loss and accuracy have been very volatile between epochs, or even increased after a certain point - again suggesting that the model is not to predict new data very well (check this).
Personally I am quite happy with model 9. It has an accuracy score of 0.85, meaning that it was able to predict 85% of labels in the test data correctly. Furthermore the loss and accuracy performance looks good, with both the validation and training loss decreasing across epochs and the validation loss plateauing much later than in other models - albeit the model was still probably trained for too many epochs. Conversely both training and validation accuracy scores increase relatively consistently - with the same caveat about the number of epochs.
If we compare the performance of model 9 to some of other decent models, we can see that the validation loss / accuracy is nowhere near as volatile as for model 8, while it plateaus maybe 10 epochs later than model 7.
In terms of application of the model, a quick googling brought up this article evaluating the performance of existing AI tools used to identify certain cancers.
https://www.frontiersin.org/articles/10.3389/fmed.2022.1018937/full#:~:text=The%20sensitivity%20and%20specificity%20of,99.1)%20(Supplementary%20Figures).

Although they use the language of sensitivity and specificity rather than accuracy, the typical percentage scores they find are somewhere between the high 80's and low 90's. The result for model 9 is comparable to this, suggesting that if I were to apply it to the full data set the result might be clinically useful.

- Look at this again and make sure you can draw parrellel between accuracy and sensitivity / specificity

## Conclusions
