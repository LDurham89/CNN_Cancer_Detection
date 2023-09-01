# Using neural networks for medical diagnosis

## Section 1: Project Definition

### Project Overview

The Capstone project due as part of Udacity's Data Science Nanodegree allows students to design their own project and produce a notebook and blog post to go with it.
I chose to do a project that involves building a model to effectively predict if diagnositic imaging samples contain cancerous tissue. 


### Problem statement
As you will no doubt be aware, cancer is one of the leading causes of mortality in the developed world. For example in 2020, cancers accounted for 24% of all deaths in the UK (See here for more detail: https://www.cancerresearchuk.org/health-professional/cancer-statistics/mortality). At the same time, many western countries are facing ageing populations and a shortage of medical professionals. Against this background there are clear benefits to using automated methods of processing diagnostic images to complement the expertise of medical professionals.
With that in mind this project aims to build a model that can predict if images show cancerous tissue samples are not. This project will work with data on Invasive Ductal Carcinoma (IDC) -a form of breast cancer- taken from the kaggle page here: https://www.kaggle.com/datasets/paultimothymooney/breast-histopathology-images. The data consists of images, each showing a 50 x 50 pixel region from a breast scan. I will process the data for analysis and then build a model that can distinguish cancerous sub-samples from non-cancerous ones.
In this project I use a deep learning model that utilises Convolution Neural Networks (CNN's).

Tasks involved in this project:
1) Explore data and QA data
2) Carrying out preprocessing.
3) Identify an architecture that returns the most accurate results.
4) Tune hyperparameters
5) Test against unseen data
6) Evaluate results


### Metrics
When looking at the performance of the models I used accuracy and loss as the main metrics. The accuracy score is the percentage of all predictions that are correct. For this metric I was seeking a score above 80% as a model being used for diagnosis purposes has to perform much better than a coin flip. Given this, I think that the accuracy metric is suitable for this data, given that the data is balance enough that a high accuracy score cannot be achieved by providing the same label for all predictions.

I looked at the accuracy score that came from model.evaluate method, which uses the model to make predictions based on your X_test data and compare this to your Y_test data. This is informative as it tells us how well the model can generalise to data that is hasn't previously seen. However, as well as this I looked at the behaviour of the accuracy and loss metrics over the epochs of training. The behaviour across epochs can be very informative as it tells us how well the model is learning and also shows us if the model is overfitting.

Once I has found the version of the model that had the highest accuracy I calculated an f1 score as an additional check. The f1 scores is a mean of the models precision and recall. The precision tells us the number of accurate predictions, while the recall tells us what proportion of true positives awere correctly identified by the model. This is a nice metric as it means that the model is punished for failing to identify relevant elements in each class.

## Section 2: Analysis

### Data Exploration
This project will work with data on Invasive Ductal Carcinoma (IDC) -a form of breast cancer- taken from the kaggle page here: https://www.kaggle.com/datasets/paultimothymooney/breast-histopathology-images. The data consists of images, each showing a 50 x 50 pixel region from a breast scan. Images are classed as 0 for non-cancerous samples and 1 for cancerous samples. I will process the data for analysis and then build a model that can distinguish cancerous sub-samples from non-cancerous ones.

The original data consists of nearly 300,000 images and is around 4Gb in size. Unfortunately this is far too big for my machine to process in an acceptable amount of time, and so I created a much smaller sample data set (around 1200 images) that would allow me to learn and complete this project.

A point to note here is that the data is organised by its author in a way that makes it easier to manage computationally, however the way this has been done means that each folder ID is not unique to one patient (see this post https://www.kaggle.com/datasets/paultimothymooney/breast-histopathology-images/discussion/137446) . As a result, this project doesn't try to draw any inferences about the patients sampled in this data, or pick out any demographic trends.

As this data consists of images and correspondimg labels there is a limit to the amount of exploratory analysis hat can be done. One thing though that is useful to check is the distribution of the labels. The figure below shows the number of samples in each class. The data isn't perfectly balanced, although I believe there are enough examples of class 1 for the model to learn what a cancerous smple looks like. Furthermore, the data is balanced enough that if the model were to predict the same label for every sample in the test data, this would be reflected in a poor performance metric.

<img src="/assets/value_counts.jpg" alt="cancerous images" width="550" height="300">

### Data Visualisation
To give you a better idea of what images are being used here, the figure below presents a few samples from each class:

<img src="/assets/class0_take2.jpg" alt="non-cancerous images" width="750" height="200">

<img src="/assets/class1.jpg" alt="cancerous images" width="750" height="200">


## Section 3: Methodology

### Data preprocessing 
After reading in the data the first task was to retrieve the labels. These were contained within the file names, so to do this I had to loop through these and extract the relevant digit (0 or 1) for each sample.

The next step was to QA to data. In this context of image data this meant checking that each image is actually a valid image file and nthatvthe dimensions are at least approximately correct. Most of te files are 50 x 50, so I searched for images that deviated significantly from this either in terms of height or width. Only one file appeared not to be a valid image as it was only lone pixel tall. Such images can be treated as data generation erros and reoved from th data set.

After running the first iteration of my model I then decided to try and improve model performance by doing data augmentation. This is a process when you apply random alterations to the data - for example, you could rotate images, flip them or zoom in / out. The aim of this is to increase the amount of variation in the data, enabling the modle to see more examples of each class and focus less on irrelevant variables. In a way, this is a bit like how we humans learn - think of how diverse dogs are, yet we still manage to identify them correctly. Most people have seen enough breeds of dog to not think that a dog is defined by a certain size or colour for example. For this project I decided to apply random flips, so the image could be flipped vertically, horizontally, or both ways. Below is an example of an image from the original data (top) and the same image after augmentation (below):

<img src="/assets/original_sample.jpg" width="200" height="200">

<img src="/assets/augmented_sample.jpg" width="200" height="200">

You can see that the same patterns are present in each image, but the bottom image in upside-down. After augmenting the data I experimented with adding additional convolutional layers and max pooling layers to the model architecure, introducing a batch size, changing the number of filters in the convolutional layer and removing the drop-out layer.

Before building my model, I split the data into different sets to allow for better model results and incresed ease of evaluation. I split the observations in my data as follows:

- Training data ( 56% of observations): This is the main bulk of the data, which is used in the process of fitting model hyperparameters and calculating weights.
- Validation data ( 14% of observations): The model will aim to minimise a loss function with respect to the training data. In order to prevent overfitting, we can create a validation set against which the hyperparameters created by fitting on the training data are evaluated.
- Test data (approx 30% of observations): Once we have a fitted model can evaluate its performance by seeing how well it can predicted values for data that it hasn't seen before. This unseen data is our test set. In this project I will use the model to predict values for the labels in the test data (Y_test) based on the images in the test set (X_test). We can then compare the predicted values of Y_test against the real (observed) values and evaluate the model using metrics such as accuracy or the f1 score.

### Implementation
In this project I use a deep learning model that utilises Convolution Neural Networks (CNN's). A major advantage of working with CNN's is that they are the industry standard for computer vision and thus there are many tools predeicated on this method, with helpful documentation. Furthermore, they are designed specifically for image analysis. However, there are some alternative methods that I decided not to use.

-One option is to use Recurrent Neural Networks, however these are more appropriate for sequences of information (i.e. there is a temporal dimension). This is why they are frequently used for tasks such as translation - where the order of the information is crucial to its meaning - and analysing videos whereas my data consists of non-sequential photos. Recurrent Neural Networks are also slower than CNN's, which could be an issue given that my final model will use a lot of data.
-Restricted Boltzman models are another option that didn't seem appropriate for this task. This approach appears to be used more for modelling systems using unsupervised learning, although I understand that they can be used for classification tasks. With the data set used in this project labels are available, allowing us to train the model with supervised learning methods, which tend to be more accurate (if interested you can see the discussion here: https://www.ibm.com/cloud/blog/supervised-vs-unsupervised-learning#:~:text=While%20supervised%20learning%20models%20tend,weather%20conditions%20and%20so%20on.)

Let's briefly consider what a CNN does and how we can use it as the basis of a predictive model. Below is a (pretty cute) illustration of the architecture used in this project (credit for the images goes to the author of this article: https://www.analyticsvidhya.com/blog/2022/01/convolutional-neural-network-an-overview). The main difference in my project is that I am solving a binary classification problem rather than a multiclass classification one:

<img src="/assets/Tweety_cnn.jpg" alt="Tweety_cnn" width ="550" height="250">

As shown above we can consider our model to be made up of several parts:

- An input (in this case an image)
- A feature extraction stage
- A learning stage
- An output

In the feature extraction stage we can use convolutional layers to apply filters to the image. These filters are effectively small (relative to the image) matrices containing weights that spatially arranged to search for a given pattern. For example, in the image below the filter on the left searches for vertical lines, while the filter on the right searches for horizontal lines (image credit goes to the author of this article [LINK](https://www.analyticsvidhya.com/blog/2018/12/guide-convolutional-neural-network-cnn/). Other filters may search for more complex or abstract patterns. 

<img src="/assets/filter_horiz_vert.png" alt="cnn filters" width="550" height="220">

The neural network will initially use random filters, then selects the filters that best fit the patterns in the image during a process called backpropagation. Depending on the complexity of the image we may need a lot of feature maps, which lead to high dimensionality and a risk of overfitting in the training stage. To reduce the dimensionality, we can use Max Pooling layers. These take the feature maps as input and again apply a filter to the image. These filters simply take the largest value within each region of the image and use these to create a smaller representation of the information in the feature maps. In our model we use a dropout layer to reduce the risk of overfitting. This is effectively a version of the dense layer where a random selection of nodes in the neural network are not activated, treating them as though they weren't in the network at all.

Once the features are extracted we can flatten the features maps into a vectors that can be read by a dense layer. Now, we're at the learning stage, where we carry out our classification task. The first dense layer takes the flattened image vectors and labels from the training data as inputs and applies an optimization algorithm (in this case the 'Adam' optimizer) to these in order to calculate weights. The weights can then be used to predict labels in new images. These weights are repeatedly predicted, evaluated against a loss function and updated in processes called Feedforward and Backpropagation. The second dense layer then uses a sigmoid activation function to return probabilities between 0 and 1, which can be rounded to provide the predicted label.

To build a model I decided to start off with a simple base architecture and to then experiment with adding layers and tuning hyperparameters to find the best performing version of the model. Below is the base architecture:

<img src="/assets/model1_architecture_snippet.jpg" alt="model architecure" width ="400" height="300">

I then incrementally added convolutional and maxpooling layers until I felt that the resulting accuracy scores were good. I then tuned the hyperparameters using sci-kit learn's GridSearch CV...


### Refinement

A few approaches were taken to refining the model used here. The first was to start off with a base architecture, to which I then added more layers to evaluate performance. While not strictly speaking a method of refinement, I also carried out data augmentation as discussed earlier in order to increase the amount of information within the data sets.

The main tool I used for refinement was a cross-validation method called GridSearchCV. This is a very nice feature provided by Sci-Kit Learn, which allows us to find the best combination of hyperparameters to use on our model. Using GridSearchCV provides a convenient and objective measure for the how well different sets of hyperparameters performs without requiring us to extensively test different model combinations based on intuition (although good intuition is needed to get the process off the ground).

I used GridSearchCV to find optimal values for:

- The number and size of filters in the convolutional layers
- The batch size used when fitting the model.

I gave chosen not to pass the padding hyperparameter to GridSearchCV for two reasons. The first is that as you pass more hyperparameters to GridSearchCV it becomes exponentially more expensive to run computationally. The second reason is that I think there is good reason to allow the filters to spill over the edge of the image. From looking at the sample of images there does not appear to be a specific focus as such. As a result it appears that all regions of the image are informationally valuable. This is in contrast to an image with a defined region of interest or even a typical picture like a portrait, where the centre of the image is considerably more important than the edges.

One potential issue with this form of cross-validation is that may return hyperparameters that lead to overfitting as it uses the same set of training and validation data for all combinations of hyperparameters. The prevent this I will also run the tuned model with regularisation on the first dense layer. Regularisation is a technique that imposes a penalty in the loss function for larger weights. This pushes the values of weights on less significant parameters towards (maybe actually reaching) zero - meaning that the resulting model will be more generalisable to new data. The technique I chose is called l1 (or lasso) regulation, which adds a penalty to the absolute (rather than squared values as is the case in l2 regularisation) value of the model weights. 

In the next section you will see the optimal parameters and how the tuned model performed in comparison to other versions of the model.

## Section 4 - Results

### Model evaluation & validation
Below is the architecture of the final model:

<img src="/assets/tuned_model__architecture_snippet.jpg" alt="final model architecure" width ="400" height="300">

Key features of this model:
- It consists of two convolutional layers,
- Two Max Pooling layers
- A flatten layer
- Two dense layers - the first of these layers where the learning process is carried out, weights are generated and a signmoid activation function return values between 0 and 1
- A drop out layer - a tool to prevent overfitting by having a layer where a set fraction of nodes in the network are effectively switched of at random.

As mentioned, I took the model through several stages of refinement to find the best solution. Firstly, I took an iterative approach to including layers in the model, added augmented data and then used cross-validation methods to tune the hyperparameters of my model. I feel that the resulting model is intuitive and performs well, as is discussed in more detail below.

### Justification

Let's look at the performance of the different iterations of the model that I have run.

<img src="/assets/cnn_rsults_table.jpg" alt="table of accuracy scores" width ="400" height="300">

As we can see, the performance of the model has improved with each iteration of the model. By augmenting our data we were able to increase the amount of information contained in the dat set. Then by adding more convolutional and max pooling layers we were able to extract more of this information from the images. By using cross-validation and regularisation we were able to obtain the best hyperparameters for this model, but without overfitting to te training data, meaning that in the final iteration of the model it was possible to achieve an accuracy rate of 83% when attempting to predict the values in the test data.

We can also evaluate the models by looking at the relationship between the training and validation data for each model. Ideally the accuracy scores for the training and validation data should increase over the course of training and be reasonably close together. The picture should be similar for the loss scores, except these should decline with throughout the learning process. In early iterations we do not see this, with the validation accuracy plateauing early on, which the accuracy score (for the training data) keeps increasing. You can see the accuracy across epochs of model 1 in the figure below. This suggests that model 1 performed relatively poorly on the test data as it was overfitting to the training data.

FIGURE

As I added more complexity to the model the performance of the accuracy and loss scores gradually improved. For contrast with the performance in model one, I have included the accuracy history of model 6 below. You can see that is is much closer to the ideal described, although the validation accuracy is volatile in the earlier, both lines follow a clear upward trend, don't significantly diverge from each other and reach their peak in the final epoch of training.

<img src="/assets/tuned_model_accuracy.jpg" alt="final model accuracy and validation accuracy" width ="400" height="300">
  
Personally I am quite happy with model 6. It has an accuracy score of 0.83, meaning that it was able to predict 83% of labels in the test data correctly.
In terms of application of the model, a quick googling brought up this article evaluating the performance of existing AI tools used to identify certain cancers.
https://www.frontiersin.org/articles/10.3389/fmed.2022.1018937/full#:~:text=The%20sensitivity%20and%20specificity%20of,99.1)%20(Supplementary%20Figures).

Although they look at sensitivity and specificity (and thus the ability to avoid false negative and false positives) rather than accuracy, the typical percentage scores they find are somewhere between the high 80's and low 90's. The result for model 6 is not far off of this, suggesting that if I were to apply it to the full data set the result might be clinically useful.

## Section 5:Conclusion
### Reflection
Convolutional neural networks are powerful tools for understanding image data and can provide vital tools for various tasks that require images to be interpreted. We have seen here that even a fairly simple model can provide a good starting point for developing tools that could be used in the real world.

### Improvement
Ideally this project would have been run with a larger dataset, giving the neural network the opportuity to see more examples of each class in the training process.
