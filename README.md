# CNN_Cancer_Detection

__Project description:__ The fourth project due as part of Udacity's Data Science Nanodegree gives students some freedom to design a project themselves and produce a notebook or app.

The task I chose was to build a model that can predict if images show cancerous tissue samples are not. In this project I use a deep learning model that utilises Convolution Neural Networks (CNN's). A major advantage of working with CNN's is that they are the industry standard for computer vision and thus there are many tools predeicated on this method, with helpful documentation. Furthermore, they are designed specifically for image analysis. However, there are some alternative methods that I decided not to use:
- One option is to use Recurrent Neural Networks, however these are more appropriate for sequences of information (i.e. there is a temporal dimension). This is why they are frequently used for tasks such as translation - where the order of the information is crucial to its meaning - and analysing videos whereas my data consists of non-sequential photos. Recurrent Neural Networks are also slower than CNN's, which could be an issue given that my final model will use a lot of data.
- Restricted Boltzman models are another option that didn't seem appropriate for this task. This approach appears to be used more for modelling systems using unsupervised learning, although I understand that they can be used for classification tasks. With the data set used in this project labels are available, allowing us to train the model with supervised learning methods, which tend to be more accurate (if interested you can see the discussion here: https://www.ibm.com/cloud/blog/supervised-vs-unsupervised-learning#:~:text=While%20supervised%20learning%20models%20tend,weather%20conditions%20and%20so%20on.)
  
In this notebook I will explore the data, prepare it for analysis and build a model. Building CNN's and tuning parameters is a process that requires logic but also some trial and error to find the model that's most appropriate to the data.

There are 3 general stages to this project:

- Read in and explore the data
- Prepare it for analysis
- Test several models to find one that can reliably be generalised to unseen images

The end product is a jupyter notebook showing these steps and the logic behind the process.

__File descriptions:___ 

All of my python code is presented in the 'CNN_Breast_Cancer.ipybn' file. This file is a notebook which also shows relevant visualisations and commentary.

The data used is taken from the kaggle page here: https://www.kaggle.com/datasets/paultimothymooney/breast-histopathology-images. The data consists of images, each showing a 50 x 50 pixel region from a breast scan. There are two options for accessing it:

- I have created a folder called 'Sample_data'. This contains two subfolders (one for each pseudopatient), each of which also contains two subfolders(one for cancerous images and one for non-cancerous images), in which you can find the .png image files.
- The other option is to download the 'Data' zip folder and extract the files onto your machine.

__Usage instructions:__

If wish to run the notebook yourself this should be fairly simple. You will need to download the dataset to your own machine. You will then need to change the root_directory given in the notebook to the location where you save the data. All that's left is to hit 'run all' in the jupyter notebook. On my machine the whole notebook takes maybe two or three minutes to run.

__Packages used:__ There are quite a few packages used in this project.

This project uses packages for a wide variety of tasks.

First are some of the most common general data processing packages, plus re which is useful for editting data:

os
pandas 
seaborn 
matplotlib.pyplot
numpy
PIL.Image
re

sklearn.model_selection.train_test_split

tensorflow
tensorflow.keras.layers
tensorflow.keras.models.Sequential
tensorflow.keras.layers: Dense, Dropout, Conv2D, MaxPool2D & Flatten
tensorflow.keras.preprocessing.image
keras.preprocessing.image.ImageDataGenerator 

__Model performance and difficulties faced:__



__Contact information:__ The maintainer of this project is me - Laurence Durham - contactable at laurence.durham89@gmail.com

__Necessary acknowledgments:__ This project presented several interesting challenges and made we question several times what was happening 'behind the scenes' in my code. As a result I drew on a few online resources to help get my code right and to interpret my model and its performance better.
