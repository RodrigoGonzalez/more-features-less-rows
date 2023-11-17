import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

print("Set Variables")
CV = 5
RANDOM_STATE = 0
SCORING = "f1"
version = "final_"
model_type = f"nuscv_{RANDOM_STATE}_{SCORING}"

print("\nSet hyperparameters")
print(SCORING)
n_components = [100]
nu = [0.01]
kernel = ["poly"]
gamma = np.logspace(-5, -4, 100)
coef0 = [1.0]
shrinking = [True]


print("\nProcess Data & Fit Model")

df_train = pd.read_csv("../data/train_data.csv")
df_test = pd.read_csv("../data/test_data.csv")
y_entire = df_train.pop("target_eval").get_values()
X_entire = df_train.get_values()

print("\nSplit Data")

X, X_test, y, y_test = train_test_split(
    X_entire, y_entire, test_size=0.1, random_state=RANDOM_STATE, stratify=y_entire
)

###############################################################
print("Load model")
from sklearn.svm import NuSVC

###############################################################
print("\nGrid Search")

pipe = Pipeline(
    [
        ("scl", StandardScaler()),
        ("pca", PCA()),
        ("clf", NuSVC(degree=3, probability=True, random_state=RANDOM_STATE)),
    ]
)


param_grid = [
    {
        "pca__n_components": n_components,
        "clf__kernel": kernel,
        "clf__nu": nu,
        "clf__gamma": gamma,
        "clf__coef0": coef0,
        "clf__shrinking": shrinking,
    }
]

grid_search = GridSearchCV(
    estimator=pipe, param_grid=param_grid, scoring=SCORING, cv=CV, verbose=1, n_jobs=-1
)
###############################################################
print("Start nested loops")

best = []

cv_outer = StratifiedShuffleSplit(y, n_iter=3, test_size=0.2, random_state=RANDOM_STATE)

for training_set_indices_i, testing_set_indices_i in cv_outer:
    training_set_i = X[training_set_indices_i], y[training_set_indices_i]
    testing_set_i = X[testing_set_indices_i], y[testing_set_indices_i]
    grid_search.fit(*training_set_i)
    print(grid_search.best_params_, "\t\t", grid_search.score(*testing_set_i))
    params = np.array(grid_search.best_params_.items())
    score = ["score", grid_search.score(*testing_set_i)]
    best.append(np.vstack((params, score)))


for i, model in enumerate(best):
    print(f"Model {i}")
    print(model)

###############################################################
print("\nFinal Model")

gs = GridSearchCV(
    estimator=pipe, param_grid=param_grid, scoring=SCORING, cv=CV, verbose=1, n_jobs=-1
)

print("\nFit models")
gs = gs.fit(X, y)

print("\nBest Score")
print(gs.best_score_)
print("\nBest Parameters")
print(gs.best_params_)

print("\nFinal Scores")
y_pred = gs.predict(X)
acc, roc_auc = accuracy_score(y, y_pred), roc_auc_score(y, y_pred)
print(f"\nAccuracy Score: {acc}")
print(f"\nAccuracy Score: {roc_auc}")


###############################################################

print("\nSaving predictions and probabilities to csv files")

y_prob = gs.predict_proba(X)
probabilities_df = pd.DataFrame(y_prob)
predictions_df = pd.DataFrame(y_pred, columns=["predictions"])

###############################################################

print("\nTraining data")
scl_pca_df = pd.concat([probabilities_df, predictions_df], axis=1)
scl_pca_df.to_csv(
    f"../1st_gen_predictions_train/train{version}" + model_type + ".csv",
    mode="w",
    head=True,
    index=False,
)

###############################################################

bp = pd.DataFrame(gs.best_params_.items(), columns=["parameter", "value"])
name = gs.estimator.named_steps["clf"].__module__
bp.to_csv("../parameters/parameter_" + model_type + version + ".csv", mode="w", index=False)

###############################################################
X_submit = df_test.get_values()
y_prob_train = gs.predict_proba(X_submit)
y_pred_train = gs.predict(X_submit)
probabilities_train_df = pd.DataFrame(y_prob_train)
predictions_train_df = pd.DataFrame(y_pred_train, columns=["predictions"])
df_train = pd.concat([probabilities_train_df, predictions_train_df], axis=1)
df_train.to_csv(
    f"../1st_gen_predictions_submission/submission{version}"
    + model_type
    + ".csv",
    mode="w",
    head=True,
    index=False,
)

###############################################################
y_prob_train = gs.predict_proba(X_test)
y_pred_train = gs.predict(X_test)
acc, roc_auc = accuracy_score(y_test, y_pred_train), roc_auc_score(y_test, y_pred_train)
print(f"\nAccuracy Score: {acc}")
print(f"\nAccuracy Score: {roc_auc}")
probabilities_train_df = pd.DataFrame(y_prob_train)
predictions_train_df = pd.DataFrame(y_pred_train, columns=["predictions"])
df_train = pd.concat([probabilities_train_df, predictions_train_df], axis=1)
df_train.to_csv(
    f"../1st_gen_predictions_test/test{version}" + model_type + ".csv",
    mode="w",
    head=True,
    index=False,
)

# 'mean_absolute_error' 0 {'clf__gamma': 0.00035938136638046257, 'clf__coef0': 5.0, 'clf__shrinking': True, 'clf__nu': 0.001, 'pca__n_components': 100, 'clf__kernel': 'poly'}
# f1 0 {'clf__gamma': 0.0031622776601683794, 'clf__coef0': 1.0, 'clf__shrinking': True, 'clf__nu': 0.001, 'pca__n_components': 100, 'clf__kernel': 'poly'}
# roc_auc 0 {'clf__gamma': 0.0023713737056616554, 'clf__coef0': 1.0, 'clf__shrinking': True, 'clf__nu': 0.001, 'pca__n_components': 100, 'clf__kernel': 'poly'}
# log_loss 0 {'clf__gamma': 0.0031622776601683794, 'clf__coef0': 1.0, 'clf__shrinking': True, 'clf__nu': 0.001, 'pca__n_components': 100, 'clf__kernel': 'poly'}
