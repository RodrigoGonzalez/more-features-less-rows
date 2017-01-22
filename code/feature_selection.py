import pandas as pd
import numpy as np
from collections import OrderedDict
import string

from scipy.stats import skew
from scipy.stats.stats import pearsonr
import cPickle as pickle
import matplotlib.pyplot as plt

print "\nLoad Necessary Packages"
from sklearn.preprocessing import StandardScaler
from sklearn.cross_validation import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.decomposition import PCA
from sklearn.grid_search import GridSearchCV
from sklearn.linear_model import LogisticRegression


def load_data():
    '''
    The following data has been anonymized and munged, and contains 300 features and 250 rows which are used to train the machine learning algorithms in a classification task. The data is available to download in the data sub folder of this project.
    '''

    print "\nPreprocessing"
    df = pd.read_csv("../anonymized_data.csv")
    df_train = df.iloc[:250]
    df_test = df.iloc[250:]
    df_train.drop("train", axis=1, inplace=True)
    df_test.drop("train", axis=1, inplace=True)
    df_test.drop("target_eval", axis=1, inplace=True)
    df_train_id = df_train.pop("id")
    df_test_id = df_test.pop("id")
    y = df_train.pop("target_eval").get_values()
    X = df_train.get_values()

    return X, y

def pipeline(X, y):
    '''
    SKLearn comes with excellent libraries that make it easy to fit and process data in one easy step.
    Additionally, the results of each transformation are available in the event that they are needed.

    We will start first with standardizing the data, which transforms the data so that the mean (mu) is zero and the standard deviation (sigma) is one, since machine learning algorithms tend to really like standardized data.
    This is especially necessary for algorithms that calculate distance metrics.

    We will be using a logistic regression to determine the features to feed into our models. L1 regularization, uses a penalty term which encourages the sum of the absolute values of the parameters to be small. It has frequently been observed that L1 regularization in many models causes many parameters to equal zero, so that the parameter vector is sparse. This makes it a natural candidate in feature selection settings, where we believe that many features should be ignored. The best parameters are found using a grid search cross-validation with the mean squared error as the scoring method.

    '''

    pipe_lr = Pipeline([('scl', StandardScaler()),
                        ('clf', LogisticRegression(penalty='l1'))])

    Cs = np.logspace(-4, 4, 100)

    param_grid = [{'clf__C': Cs}]

    gs = GridSearchCV(estimator=pipe_lr,
                      param_grid=param_grid,
                      scoring='mean_squared_error',
                      cv=10,
                      verbose=1,
                      n_jobs=-1)

    return gs.fit(X, y)

def principal_components():

    return

def fitting(X, y):
    '''
    
    '''

    print "\nFit models"

    gs = pipeline(X, y)

    print '\nBest Score: {}'.format(gs.best_score_)
    print '\n\nBest Parameters: {}'.format(gs.best_params_)



    return

if __name__ == '__main__':
    X, y = load_data()

    fitting()
