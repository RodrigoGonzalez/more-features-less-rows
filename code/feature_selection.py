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
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression

class data_transporter(object):
    '''
    This class makes it easier to move all of the data, including the raw data,
    into the various python machine learning algorithms without haveing to load
    them (etc.) all over again.
    '''
    def __init__(self, filename):
        self.package = self.load_data(filename)
        self.unpack(self.package)
        self.model = 0

    def load_data(self, filename):
        '''
        The following data has been anonymized and munged, and contains 300 features
        and 250 rows which are used to train the machine learning algorithms in a
        classification task. The data is available to download in the data sub folder
        of this project.
        '''

        print "\nPreprocessing"
        df = pd.read_csv(filename)
        df_train = df.loc[:249, 'var_1':'var_300']
        df_test = df.loc[250:, 'var_1':'var_300']

        df_train_id = df['id'][:249]
        df_test_id = df['id'][250:]
        y = df['target_eval'][:250].get_values()
        X = df_train.get_values()
        X_pred = df_test.get_values() # data to predict on

        package = [df, df_train, df_test, df_train_id, df_test_id, y, X, X_pred]

        return package

    def unpack(self, package):
        '''
        INPUT: Package of raw data
        OUPUT: No output returned simply defining instance variables
        '''
        print "\nLoading Data"
        self.df = package[0]
        self.df_train = package[1]
        self.df_test = package[2]
        self.df_train_id = package[3]
        self.df_test_id = package[4]
        self.y = package[5]
        self.X = package[6]
        self.X_pred = package[7]
        self.allfeatures = self.df_train.columns.unique()

    def update_data(self):
        '''
        When run, this method will select the columns with the important features
        as determined by runnning the logisitic regression, and update the data frames
        to only retain the most important features.
        '''

        self.df_train_full = self.df_train
        self.df_test_full = self.df_test
        self.X_pred_full = self.X_pred
        self.df_train = self.df_train[self.features]
        self.df_test = self.df_test[self.features]
        self.X_pred = self.X_pred.get_values()

    def feature_model(self, model):
        '''
        INPUT: Pipeline
        No ouput. The loaded logitic regression and coefficients resulting from the non-beta
        coefficients from the lr model are loaded on the the data transporter.
        '''

        coeffs = model.coef_.nonzero()[1]
        self.features = self.allfeatures[coeffs]
        self. update_data()

def grid_search(dt):
    '''
    INPUT: data transporter class
    OUTPUT: result of logistic regression gridsearch

    SKLearn comes with excellent libraries that make it easy to fit and process data in one easy step.
    Additionally, the results of each transformation are available in the event that they are needed.

    We will start first with standardizing the data, which transforms the data so that the mean (mu)
    is zero and the standard deviation (sigma) is one, since machine learning algorithms tend to
    really like standardized data. This is especially necessary for algorithms that calculate distance metrics.

    We will be using a logistic regression to determine the features to feed into our models. L1 regularization,
    uses a penalty term which encourages the sum of the absolute values of the parameters to be small.
    It has frequently been observed that L1 regularization in many models causes many parameters to equal zero,
    so that the parameter vector is sparse. Forcing beta coefficients to zero makes it a natural candidate in
    feature selection settings, where we believe that many features should be ignored. The best parameters are
    found using a grid search cross-validation with the mean squared error as the scoring method.

    All of the training data can be used since we are merely selecting the best features to use. As a side note,
    the use PCA and Random Forrests was explored as well, but the variances were very close to each other and
    no natural cutoff point existed in the PCA (also 250 features were somehow determined to be important),
    the same issue arose when comparing the random forrest feature importances.
    '''

    pipe_lr = Pipeline([('scl', StandardScaler()),
                        ('clf', LogisticRegression(penalty='l1'))])

    Cs = np.logspace(-1, 1, 1000) # Can start with a logspace(-4, 4, 1000)
    # The vast majority of regularization parameters are within this logspace
    # Additionally, fitting this model is very inexpensive computationally

    param_grid = [{'clf__C': Cs}] # LogisticRegression takes only one parameter

    gs = GridSearchCV(estimator=pipe_lr,
                      param_grid=param_grid,
                      scoring='neg_mean_squared_error',
                      cv=10,
                      verbose=1,
                      n_jobs=-1) # Use all the available cores

    print "\nFit models"

    gs.fit(dt.X, dt.y)

    print '\nBest Score: {}'.format(gs.best_score_)
    print '\n\nBest Parameter (C): {}'.format(gs.best_params_)

    return gs

def find_features(grid_search):
    '''
    The resulting regularization parameter of the grid search can then be used to fit
    a LogisticRegression model. This model calculates the beta coefficients that solve
    the data system. As in the aforementioned function, the model forces a number of beta
    coefficients to zero, and therefore we use the non-zero beta coefficients to select
    features for feeding into the machine learning algorithms
    '''

    gs = grid_search

    lr = LogisticRegression(penalty='l1',
                            C=gs.best_params_['clf__C'])

    lr.fit(dt.X, dt.y)

    print "\n Best Score: {}".format(lr.score(dt.X, dt.y))
    print "\n Number of non-zero beta coeffs: {}".format(len(lr.coef_.nonzero()[1]))

    return lr


if __name__ == '__main__':
    data_location = "../data/anonymised_data.csv"
    dt = data_transporter(data_location)

    # Run grid search
    gs = grid_search(dt)

    # Run a logistic regression
    lr = find_features(gs)

    # Run the feature_model method to load data with non important features removed
    dt.feature_model(lr)

    # Save
    file_Name = 'data_transporter.pkl'
    fileObject = open(file_Name,'wb')
    pickle.dump(dt,fileObject)
    fileObject.close()
