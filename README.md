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

All of my python code is presented in the 'CNN project - final version.ipynb' file (available here: https://github.com/LDurham89/CNN_Cancer_Detection/blob/main/CNN%20project%20-%20final%20version.ipynb). This file is a notebook which also shows relevant visualisations and commentary.

One fyi - and an apology - there are some commands that produce long lists which on my machine were confined to a small window whihc was easy to scroll past. However in the notebook presented here these lists are produced in full. As such you will need to do a lot of scrolling. If this is too tedious, then you might want to download the notebook so you can skip through. 

The data used is taken from the kaggle page here: https://www.kaggle.com/datasets/paultimothymooney/breast-histopathology-images. The data consists of images, each showing a 50 x 50 pixel region from a breast scan. The original data consists of nearly 300,000 images and is around 4Gb in size. This is far too big for my machine to process in an acceptable amount of time, and so I created a much smaller smaple data set that would allow me to learn and compelte this project. This sample data is saved in this repository in a folder called 'Sample_data'. This contains two subfolders (one for each pseudopatient), each of which also contains two subfolders(one for cancerous images and one for non-cancerous images), in which you can find the .png image files. Probably the best way to access the data is navigate to the root of the repository, then click on the green 'code' button, and download all the file files in the repo to your machine.


__Usage instructions:__

If wish to run the notebook yourself this should be fairly simple. You will need to download the dataset to your own machine. You will then need to change the root_directory given in the notebook to the location where you save the data. All that's left is to hit 'run all' in the jupyter notebook. On my machine the whole notebook takes maybe two or three minutes to run.

Alternatively, if you just want to see my code and commentary you can just read through 'CNN project - final version.ipynb'.

One thing to note is that if you chose to run the code you are unlikely to get exactly the same results as me. This is due to some randm steps in the deep learning process, as well as randomness introduced by the data augmentation process. I have used random_state arguments when defining training, test and validation data nin an effort to minimise variation - but this doesn't eliminate all randomness.

__Packages used:__ There are quite a few packages used in this project.

This project uses packages for a wide variety of tasks.

First are some of the most common general data processing packages:

- pandas 
- seaborn 
- matplotlib.pyplot
- numpy

Then some packages are used for more specific data analysis and peprocessing tasks:

- os: I use the .walk() method to navigate through the data folders when reading in the images
- PIL.Image: this is crucial for many of the manipulations done on the image data
- re: regular expressions, used for extracting data from the list of file paths

The last group of packages were used for the preparing the model and creating the data to pass to it:
- sklearn.model_selection.train_test_split: This is used to create the training, test and validation data
- tensorflow.keras.layers: this contains the augmentations that I apply to the data
- tensorflow.keras.models.Sequential: this creates the model object and provides a method for adding the various layers
- tensorflow.keras.layers: Dense, Dropout, Conv2D, MaxPool2D & Flatten - these are the types of layers used in the CNN model to extract information from the data, put it in a format that the neural network can read and then analyse
- tensorflow.keras.preprocessing.image - this contains the method used for converting arrays back into images after doing data augmentation.

__Model performance and difficulties faced:__

In the notebook I run 9 iterations of the model, starting with a very simple model and ending with something more complex. My final model achieves an accuracy score of around 85% when predicting the labels in the test data. In the notebook I link to an article showing that the accuracy of models actually in clinical use for diagnosing some other types of cancer varies from around 87% - 93%, so I think my results are quite good though more work would be needed to make the model reliable in a real world context.

I'm also happy with the hehaviour of the loss and validation loss during the process of training the final model. In early versions of my model the loss would fall sharply in early epochs and then plateau, while in some models the validation loss would increase after several epochs. In the final version of the model both loss and validation loss fall quite consistently across epochs, which suggests that the neural network is actually learning. 

While this project did go smoother than anticipated, there were some real challenges. The first of these was the data augmentation process and the fact that I wanted it to create data that could both be plotted visually and could be passed to the model for training. It took a littlw while for me to realise that due to the output that data augmentation gives I would need separate object for plotting the augmented images vidually and for passing to the model. 

Beyond that the main challenge was trying to get an idea of how the choice of layers and hyperparameters in my model related to my data. As this is my first time working with image data I still needed to build intuition around things like: how complex is this image? How many filters are needed to pick out the key patterns? Do I need to feed in the data in batches? It was also interesting to consider how the evaluation metrics related to the aim of model and how givent that my objective was to create something that can classify new data, I'd evaluate it very differently to a modelt hat is meant to explain features of the current data.

__Contact information:__ The maintainer of this project is me - Laurence Durham - contactable at laurence.durham89@gmail.com

__Necessary acknowledgments:__ This project presented several interesting challenges and made we question several times what was happening 'behind the scenes' in my code. As a result I drew on a few online resources to help get my code right and to interpret my model and its performance better.

Firstly, this stackoverflow post was very helpful understanding the os.walk() method, as I hadn't previously worked with data spread across various folders:
https://stackoverflow.com/questions/25868109/read-all-files-in-directory-and-subdirectories-in-python

When checking the dimensions of my images I drew a total blank on list comprehension, so this article was useful for that:
https://stackoverflow.com/questions/51761784/how-to-delete-list-elements-based-on-condition-in-python

When it came to data augmentation, this documentation helped with understanding what options were available:
https://www.tensorflow.org/api_docs/python/tf/keras/layers

These and Udacity GPT were the maim sources for code, however in terms of understanding methods and how the neural network functions I found useful sources that I have referred to in the notebook.


