# Features \>\>\> Observations

In many disciplines such as business and genomics, analysis often requires making inferences based on limited amounts of data where there are more variables than observations (e.g. [genetic research](https://deepblue.lib.umich.edu/bitstream/handle/2027.42/46947/10994_2005_Article_422926.pdf?sequence=1&isAllowed=y), [image processing](https://en.wikipedia.org/wiki/Image_processing), [text analysis](https://en.wikipedia.org/wiki/Information_retrieval), [recommender systems](http://s3.amazonaws.com/academia.edu.documents/32978074/Recommender_systems_handbook.pdf?AWSAccessKeyId=AKIAIWOWYYGZ2Y53UL3A&Expires=1485158590&Signature=6C7gZaKoaEjmh2Ag0fJRQcv2X5o%3D&response-content-disposition=inline%3B%20filename%3DEditors.pdf): i.e. amazon buyers, netflix viewers).

In this project, 300 variables are used to predict binary target variable (250 observations). The anonimised data is provided in a coma separated value file [`anonymized_data.csv`](data/anonymized_data.csv) in the data folder. The goal is to develop a system for engineering features that retain much of the predictice power while avoiding over-fitting, which is very likely when there are more features than observations. The data is prepared and the process by which this was done as well as the best feature selection model are included in [`feature_selection.py`](src/more_features_less_rows/feature_selection.py) in the code folder.

## Feature Selection
Feature selection is going to be an essential part of the project given that the training data set has 250 rows and 300 features. In order to get the features down to a more appropriate size, a combination of [principal component analysis](https://en.wikipedia.org/wiki/Principal_component_analysis), random forests to calculate [feature importances](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC4387916/), and [logistic regression model coefficients](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC2633005/) were employed . For the principal component analysis, the features were all standardized due to the nature of orthogonal transformations into new coordinate systems, and for the other two models either standardized or un-standardized features were used.

In the case of PCA, .90 of the variation in the data was explained by approximately 150 features that are a composition of the orthogonal component vectors of all of the features. The number of features used was also included as a hyper parameter in specific models, which will be discussed further below. The random forest used to calculate feature importances yielded values that were too small and close together from which to extract relevant information. The logistic regression, however, contained 83 features that yielded non-zero coefficients and these features were ultimately used to fit the models. Un-standardized features were not very useful in any of the feature selection models used.

The input data provided was transformed using the aforementioned feature selection algorithms, and saved in separate csv files, one for the training and one for the test data set. The classes in the training data set were pretty balanced so no corrections for class imbalance were incorporated.

## Tools Used
1.	[Python](https://www.python.org/): the coding language for this project.
2.	[sklearn](http://scikit-learn.org/): Scikit-Learn, machine learning modules.
3.	[AWS](https://aws.amazon.com/): Amazon Web Services, spun up large instance for model fitting.
4.	[xgboost](https://github.com/dmlc/xgboost): Gradient boosting library.

## Model Exploration
Numerous classification algorithms were explored to fit the training data as part of the model selection process. Given that the labels were generated from the features using an unknown process, many kinds of models were explored. Extreme gradient boosting was considered, however, given the rows in the training set being so small, there was concern that the method would overfit the data.

A selection of models explored in the analysis along with their parameters and hyper-parameters is given below:

| Classifier                                                                                                                                                                                                                                                                                                                                                                                                                | Tuning Parameters                                                                                                                                                                                                                                                                                         |
|---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| [Logistic Regression](https://en.wikipedia.org/wiki/Logistic_regression)                                                                                                                                                                                                                                                                                                                                                  | The penalty, C, and what kind of solver to use were investigated.                                                                                                                                                                                                                                         |
| [Various ensemble methods](https://en.wikipedia.org/wiki/Ensemble_learning) ([random forest](https://en.wikipedia.org/wiki/Random_forest), [extremely randomized](http://link.springer.com/article/10.1007/s10994-006-6226-1), [bagging](https://en.wikipedia.org/wiki/Bootstrap_aggregating), [adaboost](https://en.wikipedia.org/wiki/AdaBoost), and [gradient boost](https://en.wikipedia.org/wiki/Gradient_boosting)) | Number of trees/estimators, max depth, max features, learning rate, functions to measure quality of split, whether to use bootstrapping, whether to use out-of-bag samples to estimate the generalization error, whether to use stochastic gradient boosting, etc. depending on the ensemble method used. |
| [Passive aggressive algorithms](http://jmlr.csail.mit.edu/papers/volume7/crammer06a/crammer06a.pdf)                                                                                                                                                                                                                                                                                                                       | With regularization as the hyper-parameter (Explored but not used because cannot calculate predictive probabilities to be calculated).                                                                                                                                                                    |
| [Gaussian Naïve Bayes](https://en.wikipedia.org/wiki/Naive_Bayes_classifier) classifier                                                                                                                                                                                                                                                                                                                                   | Only number of principal components.                                                                                                                                                                                                                                                                      |
| [Support Vector Machines](https://en.wikipedia.org/wiki/Support_vector_machine)                                                                                                                                                                                                                                                                                                                                           | Type of kernel used, gamma, whether to use a shrinking heuristic, nu, and gamma kernel coefficients.                                                                                                                                                                                                      |
| [K Nearest Neighbors](https://en.wikipedia.org/wiki/K-means_clustering)                                                                                                                                                                                                                                                                                                                                                   | Number of neighbors, distance metrics (with corresponding hyper parameters), and although used with all aforementioned models, PCA was of particular importance and was used to lower the number of features used to calculate the distance metrics (interestingly only about 5-10 were optimal choices). |

## Model Selection
Given the number of models used and the number of hyper-parameters explored, a very specific process was developed in order to efficiently select the best models. A pipeline algorithm that incorporated all of the transformations was used for efficiency with a parameter grid that could easily be updated depending on the parameters and hyper parameters (whether or not to use PCA for example) the models employ using a grid search. In the relevant models (i.e. contain decisions trees) both the [gini impurity and entropy](https://en.m.wikipedia.org/wiki/Decision_tree_learning#Gini_impurity) were explored when fitting the models (although ultimately gini gave the best performing models).

### Nested Cross-Validation
The algorithm for fitting the models incorporated nested cross-validation with stratified KFolds to ensure balanced folds with the parameters nested cross-validation . The nested cross-validation was used to avoid biased performance estimates and hold out sets of the inner and outer CV’s were 20% of the training data with 5 KFolds. If the resulting parameters determined by the nested cross validation converged and were stable, then the model minimizes both [variance](https://en.wikipedia.org/wiki/Variance) and [bias](https://en.wikipedia.org/wiki/Bias_of_an_estimator), which is extremely useful given the normal [bias–variance tradeoff](https://en.wikipedia.org/wiki/Bias%E2%80%93variance_tradeoff), which is normally encountered in statistical and machine learning. The following snippet, provides the python script used for the nested cross validation. [Cawley and Talbot, 2010](http://jmlr.csail.mit.edu/papers/volume11/cawley10a/cawley10a.pdf) provide an excellent explanation on how nested cross-validation is used to avoid over-fitting in model selection and subsequent selection bias in performance evaluation.

```python
best = []

cv_outer = StratifiedShuffleSplit(y, test_size=0.2, random_state=RANDOM_STATE)

for training_set_indices_i, testing_set_indices_i in cv_outer:

    training_set_i = X[training_set_indices_i], y[training_set_indices_i]
    testing_set_i = X[testing_set_indices_i], y[testing_set_indices_i]

    grid_search.fit(*training_set_i)

    print(grid_search.best_params_, '\t\t', grid_search.score(*testing_set_i))

    params = np.array(grid_search.best_params_.items())

    score = ['score', grid_search.score(*testing_set_i)]
```

### Scoring Methods
Scoring methods explored for the both the inner and outer CV’s used were accuracy, ROC AUC, f1, and log-loss. I also explored using mean absolute error out of curiosity on the inner CV and evaluating the outer CV scores using ROC AUC and Accuracy on the hold out training set and the results were pretty awful as expected. The various models were saved as different script (e.g. `bagging.py`, `nusvc.py`) in order to parallelize the process over multiple cores, however, this process took a few days to complete.

| Scoring Method                                                                            | Description                                                                                                                  | Equation                                                                                                                                   |
|-------------------------------------------------------------------------------------------|------------------------------------------------------------------------------------------------------------------------------|--------------------------------------------------------------------------------------------------------------------------------------------|
| [accuracy](https://en.wikipedia.org/wiki/Accuracy_and_precision#In_binary_classification) | Accuracy is the proportion of true results (both true positives and true negatives) among the total number of cases examined | Accuracy = (True Positive + True Negative) / (Condition Positives + Negatives)                                                             |
| [f1](https://en.wikipedia.org/wiki/F1_score)                                              | The F1 score can be interpreted as a weighted average of the precision and recall                                            | f<sub>1</sub> = 2 * (precision * recall) / (precision + recall)                                                                            |
| [log-loss]()                                                                              | The negative log-likelihood of the true labels given a probabilistic classifier’s predictions                                | -log P(y<sub>t</sub> given y<sub>p</sub>) = -(y<sub>t</sub> log(y<sub>p</sub>) + (1 - y<sub>t</sub>) log(1 - y<sub>p</sub>))  <sup>1</sup> |
| [ROC AUC](https://en.wikipedia.org/wiki/Receiver_operating_characteristic)                | The ROC is created by plotting the fraction of True Positives vs the fraction of False Positives                             | TP / (TP+FN) vs TP / (TP+FN)                                                                                                               |

<sub><sup>1. The log loss is only defined for two or more labels. The log loss given is for a single sample with true label yt in {0,1} and estimated probability yp that yt = 1 (definition from sklearn)</sup></sub>

The ultimate strategy is to stack or blend a few different models in order to both decrease the variance and increases the predictive abilities of the final algorithm, which will decrease the likelihood of overfitting.

The best models were selected using both the accuracy and ROC AUC of the hold out set. Additionally, predictions for the test dataset (19,750 rows) were calculated as were the Pearson correlation coefficients on these predictions between all of the models, in order to assess how different all of the prediction were from each other. The motivation for this was to identify high scoring models that had predictions that were also uncorrelated, which would bring more information to the final model and helps reduce over fitting.
Ultimately, I decided to just take the averages of the predicted probabilities of the best models, rather than stack an additional model, since this method gave excellent results on the holdout set, and I did not want to overfit on the training data set.

## Solution
The final solution was calculated using the entire training data set to train the models using the optimized parameters found during the grid searches. The Logistic Regression was the only method that was optimized with 83 features, all the others were fit using 100 principal components. The predicted probabilities of all of these methods were then averaged to get the resulting predicted probabilities that are reported. In order to determine which features are most important, PCA should be avoided since the features and transformed to a new feature space.



### The Best Models

#### Bagging Classifier

- **Tuning Parameters**: Approximately 500 trees, and max features & max samples of around 0.85

#### Extremely Randomized Trees Classifier

- **Tuning Parameters**: Between 475-525 trees, at a max depth of 7, max features calculated using natural log using a Gini coefficient, with bootstrapping.

#### Logistic Regression

- **Tuning Parameters**: An L1 penalty was found to be the best, using a liblinear solver, with 95 principal components, and various C parameters depending on the scoring method.

#### Nu-Support Vector Classification

- **Tuning Parameters**: A nu of 0.001 worked very well, with a third polynomial kernel type, gammas in the thousandths, a coefficient term of 1, with shrinking.

#### Random Forest

- **Tuning Parameters**: Out-of-bag samples were used to estimate the generalization error, around 500 trees with a max depth of 6.

#### C-Support Vector Classification

- **Tuning Parameters**: Gammas in the thousands range, coefficients of 9.0, with third-degree polynomial kernels and shrinking.


## Conclusion
The objective of this challenge was to build a model based on a training set of 250 rows with 300 features, in order to predict probabilities for 19,750 rows. Feature selection was very important in order to train a robust model, and a logistic regression identified features of interest. If a description of the feature were would have been provided, this would have been an excellent way to identify data that is important to collect for predicting a target value.

Although principal component analysis was used on the features that were identified, this feature extraction technique did not add much information to the final models.

The final models’ predicted probabilities were then averaged, to get a final predicted probabilities that resulted in an AUC of 0.97. Averaging works quite well for a wide range of problems.

## Acknowledgements
- scikit learn had excellent machine learning algorithms that were employed
- [Cawley, G.C.; Talbot, N.L.C. On over-fitting in model selection and subsequent selection bias in performance evaluation. J. Mach. Learn. Res 2010,11, 2079-2107.]()

## Keywords
[binary-classification] [classification] [python] [random-forest] [supervised-learning][svc][crossvalidation]
