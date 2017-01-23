import time
# time.ctime()
import pandas as pd
import numpy as np
print "Set Variables"
CV = 5
RANDOM_STATE = 0
SCORING = 'roc_auc'
version = 'final_'
modeltype = 'bagging_' + str(RANDOM_STATE) + '_' + SCORING

print "\nSet hyperparameters"
n_components = [95, 100]
n_estimators = [475, 500, 525, 550, 575, 600, 625]
max_samples = [0.80, 0.85]
max_features = [0.80]

print "\nLoad Necessary Packages"
import cPickle
from sklearn.preprocessing import StandardScaler
from sklearn.cross_validation import train_test_split, cross_val_score, StratifiedShuffleSplit, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.decomposition import PCA
from sklearn.grid_search import GridSearchCV

print "\nProcess Data & Fit Model"

df_train = pd.read_csv("../data/train_data.csv")
df_test = pd.read_csv("../data/test_data.csv")
y_entire = df_train.pop("target_eval").get_values()
X_entire = df_train.get_values()

print "\nSplit Data"

X, X_test, y, y_test = train_test_split(X_entire, y_entire, test_size=0.1, random_state=RANDOM_STATE, stratify=y_entire)

###############################################################
print "Load model"
from sklearn.ensemble import BaggingClassifier

###############################################################
print "\nGrid Search"

pipe = Pipeline([('scl', StandardScaler()),
                 ('pca', PCA()),
                 ('clf', BaggingClassifier(bootstrap=True,
                                           bootstrap_features=True,
                                           oob_score=True,
                                           random_state=RANDOM_STATE+1))])


param_grid = [{'pca__n_components': n_components,
               'clf__n_estimators': n_estimators,
               'clf__max_samples': max_samples,
               'clf__max_features': max_features}]

grid_search = GridSearchCV(estimator=pipe,
                  param_grid=param_grid,
                  scoring=SCORING,
                  cv=CV,
                  verbose=1,
                  n_jobs=-1)

###############################################################
print '\nStart nested loops'

best = []

cv_outer = StratifiedShuffleSplit(y, n_iter=5, test_size=0.2, random_state=RANDOM_STATE)

for training_set_indices_i, testing_set_indices_i in cv_outer:
    training_set_i = X[training_set_indices_i], y[training_set_indices_i]
    testing_set_i = X[testing_set_indices_i], y[testing_set_indices_i]
    grid_search.fit(*training_set_i)
    print grid_search.best_params_, '\t\t', grid_search.score(*testing_set_i)
    params = np.array(grid_search.best_params_.items())
    score = ['score', grid_search.score(*testing_set_i)]
    best.append(np.vstack((params, score)))

for i, model in enumerate(best):
    print "Model " + str(i)
    print model


###############################################################
print "\nFinal Model"

gs = GridSearchCV(estimator=pipe,
                  param_grid=param_grid,
                  scoring=SCORING,
                  cv=CV,
                  verbose=1,
                  n_jobs=-1)

print "\nFit models"
gs = gs.fit(X, y)

print "\nBest Score"
print (gs.best_score_)
print "\nBest Parameters"
print(gs.best_params_)

print "\nFinal Scores"
y_pred = gs.predict(X)
acc, roc_auc = accuracy_score(y, y_pred), roc_auc_score(y, y_pred)
print "\nAccuracy Score: " + str(acc)
print "ROC_AUC: " + str(roc_auc)


###############################################################

print "\nSaving predictions and probabilities to csv files"

y_prob = gs.predict_proba(X)
probabilities_df = pd.DataFrame(y_prob)
predictions_df = pd.DataFrame(y_pred, columns=['predictions'])

###############################################################

print "\nTraining data"
scl_pca_df = pd.concat([probabilities_df, predictions_df], axis=1)
scl_pca_df.to_csv('../1st_gen_predictions_train/train' + version + modeltype + '.csv', mode='w', head=True, index=False)

###############################################################

bp = pd.DataFrame(gs.best_params_.items(), columns=['parameter', 'value'])
name = gs.estimator.named_steps['clf'].__module__
bp.to_csv("../parameters/parameter_" + modeltype + version + ".csv", mode='w', index=False)

###############################################################
X_submit = df_test.get_values()
y_prob_train = gs.predict_proba(X_submit)
y_pred_train = gs.predict(X_submit)
probabilities_train_df = pd.DataFrame(y_prob_train)
predictions_train_df =  pd.DataFrame(y_pred_train, columns=['predictions'])
df_train = pd.concat([probabilities_train_df, predictions_train_df], axis=1)
df_train.to_csv('../1st_gen_predictions_submission/submission' + version + modeltype + ".csv", mode='w', head=True, index=False)

###############################################################
y_prob_train = gs.predict_proba(X_test)
y_pred_train = gs.predict(X_test)
acc, roc_auc = accuracy_score(y_test, y_pred_train), roc_auc_score(y_test, y_pred_train)
print "\nAccuracy Score: " + str(acc)
print "ROC_AUC: " + str(roc_auc)
probabilities_train_df = pd.DataFrame(y_prob_train)
predictions_train_df =  pd.DataFrame(y_pred_train, columns=['predictions'])
df_train = pd.concat([probabilities_train_df, predictions_train_df], axis=1)
df_train.to_csv('../1st_gen_predictions_test/test' + version + modeltype + ".csv", mode='w', head=True, index=False)

# mean_absolute_error 0 {'clf__max_features': 0.9, 'pca__n_components': 100, 'clf__n_estimators': 500, 'clf__max_samples': 0.85}
# f1 0 {'clf__max_features': 0.85, 'pca__n_components': 100, 'clf__n_estimators': 525, 'clf__max_samples': 0.85}
# log_loss 0 {'clf__max_features': 0.85, 'pca__n_components': 100, 'clf__n_estimators': 500, 'clf__max_samples': 0.9}
# roc_auc 0 {'clf__max_features': 0.8, 'pca__n_components': 100, 'clf__n_estimators': 500, 'clf__max_samples': 0.8}
