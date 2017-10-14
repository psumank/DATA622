__author__  = 'Suman'
# When this is called using python train_model.py in the command line, this will take in the training dataset csv, perform
# the necessary data cleaning and imputation, and fit a classification model to the dependent Y. There must be data check
# steps and clear commenting for each step inside the .py file. The output for running this file is the random forest model
#  saved as a .pkl file in the local directory. Remember that the thought process and decision for why you chose the final
# model must be clearly documented in this section

#Here's a description of some of the more cryptic variable names:
# **Variable** | **Description**
#------------- | ----------------
# Embarked | Port of embarkation (S = Southhampton, C = Chersbourg, Q = Queenstown)
# Parch | Number of parents/children of passenger on board
# Pclass | Passenger's class (1st, 2nd, or 3rd)
# SibSp | Number of siblings or spouses on board

#import required packages
import pandas as pd
import numpy as np
from sklearn.preprocessing import Imputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
import pickle

def imputedata(df):
    """
    Apply the imputations for the given data frame.
    From our research we noticed null values for 'fare', so we would apply the median price of that 'class'
    and for 'Age' , we would apply the median age.
    Also, we are going to drop the 'Ticket', 'Name', and 'Cabin' as these are non-numeric/non-categorical.
    :param df:
    :return:
    """
    df.loc[df['Fare'] == 0, 'Fare'] = np.nan
    df["Fare"] = df.groupby("Pclass")["Fare"].transform(lambda x: x.fillna(x.median()))
    imp = Imputer(missing_values='NaN', strategy='median', axis=0)
    imp.fit(df[["Age"]])
    df[["Age"]] = imp.transform(df[["Age"]])
    # drop Ticket, Name, Cabin
    df = df.drop(['Name', 'Ticket', 'Cabin'], 1);
    return df

def getdummies(df):
    """
    Encode categorical features numerically.
    :param df:
    :return:
    """
    cols_to_transform = ['Sex', 'Embarked']
    df = pd.get_dummies(df, columns=cols_to_transform, drop_first=True)
    return df

def validate(df):
    return not(df.isnull().any().any())

def train_classifier(model, df):
    """
    Lets take the given data frame, and split it into training and test sets.
    And fit the classifier with the given model, on the train data set.
    return the classifier for prediction use.
    :param model:
    :param df:
    :return:
    """
    # lets first define the target variable.
    y = df.Survived
    features = df.columns.values[1:]
    X_train, X_test, y_train, y_test = train_test_split(df[features], y, test_size=0.20, stratify=y)
    classifier = model()
    classifier.fit(X_train, y_train)
    print('model ' , model, 'score is:', classifier.score(X_test, y_test))
    return classifier

def save_model(model, filename):
    """ Dump the trained classifier with Pickle """
    # Open the file to save as pkl file
    model_pkl = open(filename, 'wb')
    pickle.dump(model, model_pkl)
    # Close the pickle instances
    model_pkl.close()

def read_and_clean_training_data():
    # read the downloaded csv file.
    df = pd.read_csv('train.csv', index_col=0)
    # impute the data
    df = imputedata(df)
    # get dummies
    df = getdummies(df)
    return df

if __name__ == '__main__':
    #init vars, to save our model.
    kNN_pkl_filename = 'kNN_classifier.pkl'
    rf_pkl_filename = 'rf_classifier.pkl'

    df = read_and_clean_training_data()

    if (validate(df)):
        print("Ok, we got proper dataset to operate on!, lets build classifiers...")
        kNN = train_classifier(KNeighborsClassifier, df)
        save_model(kNN, kNN_pkl_filename)
        forest = train_classifier(RandomForestClassifier, df)
        save_model(forest, rf_pkl_filename)
    else:
        print("Please review the data  !, Tidy the data set !!")

#From the above, the random forest model provides clearly better score than the kNN.