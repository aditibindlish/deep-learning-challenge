# deep-learning-challenge
Venture Funding

## Overview of the analysis: 
The purpose of this analysis to be able predict which ventures should be funded by Alphabet Soup by developing a model that can predict the outcome of the venture fudning, i.e. whether the venture will be successful or not. 

Data Preprocessing


For this we have been provided with existing venture funding data along with the target variable of 'Is Successful'. 
Other than that, we have various facts and information of each venture which will be treated as features of the model. 
Some of these will be redundant to analysis like name and id and will be dropped.

The data set provided by AlphabetSoup is described below. Some of the features have too many unique values, such that they become rare after a point. These values are clubbed together using a method called Binning as others to reduce the number of distnct values in each feature. 

EIN and NAME—Identification columns - To be Dropped
APPLICATION_TYPE—Alphabet Soup application type - To be Dropped
AFFILIATION—Affiliated sector of industry - To be binned
CLASSIFICATION—Government organization classification - To be binned
USE_CASE—Use case for funding
ORGANIZATION—Organization type
STATUS—Active status
INCOME_AMT—Income classification
SPECIAL_CONSIDERATIONS—Special considerations for application
ASK_AMT—Funding amount requested
IS_SUCCESSFUL—Was the money used effectively
 
The entire data set is preprocessed to remove some columns and create bins for some features with the name 'Others' as explained above. 

After this, categorical data is encoded using get_dummies.

The dataset is now scaled using standardscaler and then split into training and testing data. 

This splitting is necessary so that we can use the existing data to train a model and then test and compare using actual values with the predicted valued, this will tell aboput the accuracy and variance of the model. If the accuracy is >75%, the model can be considered to be well trained to use on new data points for upcoming venture funding, which will be of assistance in decision making as to approve or reject the funding request. 

## Compiling, Training, and Evaluating the Model
We are using the Keras neural model for the analyis. 

Looking at the processed data, we have 43 features which become the input dimensionf for the model

Neurons: In general, we begin with number of neurons in first layer as roughly two times the number features, so we choose 80n neurons in first layer

Layers: As a rule of thumb, we begin with a simple model with 1 input layer, 2 hidden layers with reducing number of neurons and one output layer with one neuron since we only trying to predict a binary outcome, i.e. successful or not.

Activation functions: We have used Relu function in initial layers and sigmoid in output layer given the complex data for the features and binary outcome making sigmoid suitable.  

## Result: 

Target model perfromance i.e. accuarcy >75% was not achieved with accuracy stuck in the 71-74% range on training data. 

The results are saved in the file: 'AlphabetSoupCharity.h5'

This led to a conclusion that the model is either underfitting or overfitting. We reject the overfitting hypothesis, as during the fitting of the model, teh accuracy at each epoch did not fluctuate much, it was range bound between 71%-74%;. If the model showed some shocks or steep change in accuarcy at any epoch level, it would be easier to conclude that the training data is overfitting the model.

So we analyse the underfitting scenario, that is the model doesnt have enough information or the input info is not appropriate and suitable to the model requirements or the model parameters are not being chosen correctly. Possibly the training data has too many outliers or confusing data points which are restricting its learning ability, so we decided to implement a number of strategies as below to work around the underfitting scenario.  

In order to acheive target performance, we used various methods as described below:

1) Tried to alter the number of neurons, epochs, added hidden layers and tried different activation functions, however, desired accuracy was not achieved. Only one iteration was saved in the model for brevity. Accuracy was 72.5%.

2) Tried to drop some columns which did not represent very useful info, starting with 'Is Successful_N' as this is redundant. Multiple iterations were by dropping columns which logically looked like something that wouldnt affect a venyure's success, however even after dropping a number of columns, desired performance was not recieved.Only one iterastion was saved in the model for brevity. Accuracy dropped to around 70.4%. 

3) Also used the best hyperparameters method to get best number of layer, neurons and hidden layers, using keras_tuner library, however, even this method did not yield optimal results. Accuracy was 72.9%.

Optimization results are saved in the file 'AlphabetSoupCharity_Optimization.h5'

4) Finally, we also tried another method, we allowed more unique values in the Classification column by retaining top 10 by value counts and binning others. The resulting model still did not achieve desirerd output. 

    The corresponding code is saved in file "Code_Change in Data.ipynb" and the corresponding results are saved in the file 'AlphabetSoupCharity_model with change in data.h5'

