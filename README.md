# Machine Learning Course in Octave / MatLab

Based on [**Machine Learning course**](https://www.coursera.org/learn/machine-learning/home/welcome) by Andrew Ng, Stanford University.

> For Python/Jupyter version of Machine Learning algorithms please check [Machine Learning using Jupiter Notebook and Google Colab](https://github.com/ElizaLo/ML-using-Jupiter-Notebook-and-Google-Colab) repository.

This repository contains Octave/MatLab examples of popular machine learning algorithms with code examples and mathematics behind them being explained.

## Machine Learning Map

![machine-learning-map](https://github.com/ElizaLo/Machine-Learning-Course-Octave/blob/master/images/machine-learning-map.png)

## Supervised Learning

In supervised learning, we are given a data set and already know what our correct output should look like, having the idea that there is a relationship between the input and the output.

In supervised learning we have a set of training data as an input and a set of labels or "correct answers" for each training set as an output. Then we're training our model (machine learning algorithm parameters) to map the input to the output correctly (to do correct prediction). The ultimate purpose is to find such model parameters that will successfully continue correct input→output mapping (predictions) even for new input examples.

Supervised learning problems are categorized into **"regression"** and **"classification"** problems. 
- [ ] In a **regression problem**, we are trying to predict results within a continuous output, meaning that we are trying to map input variables to some continuous function. 
- [ ] In a **classification problem**, we are instead trying to predict results in a discrete output. In other words, we are trying to map input variables into discrete categories.

_**Example 1:**_

- Given data about the size of houses on the real estate market, try to predict their price. Price as a function of size is a continuous output, so this is a regression problem.

We could turn this example into a classification problem by instead making our output about whether the house "sells for more or less than the asking price." Here we are classifying the houses based on price into two discrete categories.

_**Example 2:**_

- **Regression** - Given a picture of a person, we have to predict their age on the basis of the given picture

- **Classification** - Given a patient with a tumor, we have to predict whether the tumor is malignant or benign.

### ▶️ Regression

In regression problems we do real value predictions. Basically we try to draw a line/plane/n-dimensional plane along the training examples. 

💻 _**Usage examples:** stock price forecast, sales analysis, dependency of any number, etc._

  🔵 **Linear Regression**
  
  - 📘 [Math](https://github.com/ElizaLo/Machine-Learning-Course-Octave/tree/master/Linear%20Regression#linear-regression) - theory and [useful links](https://github.com/ElizaLo/Machine-Learning-Course-Octave/tree/master/Linear%20Regression#-references)
  
### ▶️ Classification

In classification problems we split input examples by certain characteristic.

💻 _**Usage examples:** spam-filters, language detection, finding similar documents, handwritten letters recognition, etc._

## Unsupervised Learning

Unsupervised learning allows us to approach problems with little or no idea what our results should look like. We can derive structure from data where we don't necessarily know the effect of the variables.

- [ ] We can derive this structure by clustering the data based on relationships among the variables in the data.

- [ ] With unsupervised learning there is no feedback based on the prediction results.

Unsupervised learning is a branch of machine learning that learns from test data that has not been labeled, classified or categorized. Instead of responding to feedback, unsupervised learning identifies commonalities in the data and reacts based on the presence or absence of such commonalities in each new piece of data.

_**Example:**_

- **Clustering:** Take a collection of 1,000,000 different genes, and find a way to automatically group these genes into groups that are somehow similar or related by different variables, such as lifespan, location, roles, and so on.

- **Non-clustering:** The "Cocktail Party Algorithm", allows you to find structure in a chaotic environment. (i.e. identifying individual voices and music from a mesh of sounds at a [cocktail party](https://en.wikipedia.org/wiki/Cocktail_party_effect)).

### ▶️ Clustering

In clustering problems we split the training examples by unknown characteristics. The algorithm itself decides what characteristic to use for splitting.

💻 _**Usage examples:** market segmentation, social networks analysis, organize computing clusters, astronomical data analysis, image compression, etc._

## What's is the difference between _train, validation and test set_, in neural networks?

**Training Set:** this data set is used to adjust the weights on the neural network.

**Validation Set:** this data set is used to minimize overfitting. You're not adjusting the weights of the network with this data set, you're just verifying that any increase in accuracy over the training data set actually yields an increase in accuracy over a data set that has not been shown to the network before, or at least the network hasn't trained on it (i.e. validation data set). If the accuracy over the training data set increases, but the accuracy over the validation data set stays the same or decreases, then you're overfitting your neural network and you should stop training.

> The **validation data set** is a set of data for the function you want to learn, which you are not directly using to train the network. You are training the network with a set of data which you call the training data set. If you are using gradient based algorithm to train the network then the error surface and the gradient at some point will completely depend on the training data set thus the training data set is being directly used to adjust the weights. To make sure you don't overfit the network you need to input the validation dataset to the network and check if the error is within some range. Because the validation set is not being using directly to adjust the weights of the network, therefore a good error for the validation and also the test set indicates that the network predicts well for the train set examples, also it is expected to perform well when new example are presented to the network which was not used in the training process.

**Testing Set:** this data set is used only for testing the final solution in order to confirm the actual predictive power of the network.

**Also**, in the case you do not have enough data for a validation set, you can use **cross-validation** to tune the parameters as well as estimate the test error.

**Cross-validation set** is used for model selection, for example, select the polynomial model with the least amount of errors for a given parameter set. The test set is then used to report the generalization error on the selected model. 

**[Early stopping](https://en.wikipedia.org/wiki/Early_stopping)** is a way to stop training. There are different variations available, the main outline is, both the train and the validation set errors are monitored, the train error decreases at each iteration ([backpropagation](https://en.wikipedia.org/wiki/Backpropagation) and brothers) and at first the validation error decreases. The training is stopped at the moment the validation error starts to rise. The weight configuration at this point indicates a model, which predicts the training data well, as well as the data which is not seen by the network . But because the validation data actually affects the weight configuration indirectly to select the weight configuration. This is where the Test set comes in. This set of data is never used in the training process. Once a model is selected based on the validation set, the test set data is applied on the network model and the error for this set is found. This error is a representative of the error which we can expect from absolutely new data for the same problem.
 
