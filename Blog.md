# Using neural networks for medical diagnosis

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

The figure below shows the number of samples in each class. The data isn't perfectly balanced, although I believe there are enough examples of class 1 for the model to learn what a canverous smple looks like. Furthermore, the data is balanced enough that if the model were to predict the same label for every sample in the test data, this would be reflected in a poor performance metric.
<img src="/assets/value_counts.jpg" alt="cancerous images" width="550" height="300">

## Methodology
In this project I use a deep learning model that utilises Convolution Neural Networks (CNN's). A major advantage of working with CNN's is that they are the industry standard for computer vision and thus there are many tools predeicated on this method, with helpful documentation. Furthermore, they are designed specifically for image analysis. However, there are some alternative methods that I decided not to use.

-One option is to use Recurrent Neural Networks, however these are more appropriate for sequences of information (i.e. there is a temporal dimension). This is why they are frequently used for tasks such as translation - where the order of the information is crucial to its meaning - and analysing videos whereas my data consists of non-sequential photos. Recurrent Neural Networks are also slower than CNN's, which could be an issue given that my final model will use a lot of data.
-Restricted Boltzman models are another option that didn't seem appropriate for this task. This approach appears to be used more for modelling systems using unsupervised learning, although I understand that they can be used for classification tasks. With the data set used in this project labels are available, allowing us to train the model with supervised learning methods, which tend to be more accurate (if interested you can see the discussion here: https://www.ibm.com/cloud/blog/supervised-vs-unsupervised-learning#:~:text=While%20supervised%20learning%20models%20tend,weather%20conditions%20and%20so%20on.)

To build a model I decided to start off with a simple base architecture and to then experiemtn with adding layers and tuning hyperparameters to find the best performing version of the model. Below is the base architecture

- show base architecture and then describe what changes

After running the first iteration of my model I then decided to try and improive model performance by doing data augmentation. This is a process when you apply random alterations to the data - for example, you could rotate images, flip them or zoom in / out. The aim of this is to increase the amount of variation in the data, enabling the modle to see more examples of each class and focus less on irrelevant variables. In a way, this is a bit like how we humans learn - think of how diverse dogs are, yet we still manage to identify them correctly. Most people ave seen enough breeds of sog to not think that a dog is defined by a certain size or colour for example. For this project I decided to apply random flips, so the image could be flipped vertically, horizontally, or both ways.

- Examples

After augmenting the data I experimented with adding convolutional layers and max pooling layers, introducing a batch size, changing the number of filters in the convolutional layer and removing the drop out layer.

## Metrics

When looking at the performance of the models I used accuracy and loss as the main metrics. The accuracy score is the percentage of all predictions that are correct. For this metric I was seeking a score above 80% as a model being used for diagnosis purposes has to perform much better than a coin flip. Given this, I think that the accuracy metric is suitable for this data, given that the data is balance enough that a high accuracy score cannot be achieved by providing the same label for all predictions.

I looked at the accuracy score that came from model.evaluate method, which uses the model to make predictions based on your X_test data and compare this to your Y_test data. This is informative as it tells us how well the model can generalise to data that is hasn't previously seen. However, as well as this I looked at the behaviour of the accuracy and loss metrics over the epochs of training. The behaviour across epochs can be very informative as it tells us how well the model is learning and also shows us if the model is overfitting.

Once I has found the version of the model that had the highest accuracy I calculated an f1 score as an additional check. The f1 scores is a mean of the models precision and recall. The precision tells us the number of accurate predictions, while the recall tells us what proportion of true positives awere correctly identified by the model. This is a nice metric as it means that the model is punished for failing to identify relevant elements in each class. 

## Model performance
To evaluate the performance of the models it would be tempting to present a chart showing the accuracy scores of each model, however this would miss the relationship between the training and validation data for each model. While the accuracy scores have improved most (but not all) times that I've added complexity to the model the behaviour of the validation data has not been great in many of the models. In many models tha validation loss has plateaued very early on, while the loss on the training data has continued to fall, which suggests that the model is likely to be overfitting. 
- Here is an example from an early itertion of the model
  
In other models the validation loss and accuracy have been very volatile between epochs, or even increased after a certain point - again suggesting that the model is not to predict new data very well (check this).

- example
- 
Personally I am quite happy with model 9. It has an accuracy score of 0.85, meaning that it was able to predict 85% of labels in the test data correctly. Furthermore the loss and accuracy performance looks good, with both the validation and training loss decreasing across epochs and the validation loss plateauing much later than in other models - albeit the model was still probably trained for too many epochs. Conversely both training and validation accuracy scores increase relatively consistently - with the same caveat about the number of epochs.
- Show below

If we compare the performance of model 9 to some of other decent models, we can see that the validation loss / accuracy is nowhere near as volatile as for model 8, while it plateaus maybe 10 epochs later than model 7.

In terms of application of the model, a quick googling brought up this article evaluating the performance of existing AI tools used to identify certain cancers.
https://www.frontiersin.org/articles/10.3389/fmed.2022.1018937/full#:~:text=The%20sensitivity%20and%20specificity%20of,99.1)%20(Supplementary%20Figures).

Although they use the language of sensitivity and specificity rather than accuracy, the typical percentage scores they find are somewhere between the high 80's and low 90's. The result for model 9 is comparable to this, suggesting that if I were to apply it to the full data set the result might be clinically useful.

- Look at this again and make sure you can draw parrellel between accuracy and sensitivity / specificity

## Conclusions
