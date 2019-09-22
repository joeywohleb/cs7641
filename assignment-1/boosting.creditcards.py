import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from matplotlib.legend_handler import HandlerLine2D
from sklearn import tree, preprocessing
from sklearn.model_selection import train_test_split, GridSearchCV, KFold, learning_curve
import csv
from sklearn.tree.export import export_text
from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import make_scorer, accuracy_score
import datetime

# https://scikit-learn.org/stable/auto_examples/model_selection/plot_learning_curve.html


def plot_learning_curve(estimator, title, X, y, ylim=None, cv=None,
                        n_jobs=None, train_sizes=np.linspace(.1, 1.0, 5)):
    """
    Generate a simple plot of the test and training learning curve.

    Parameters
    ----------
    estimator : object type that implements the "fit" and "predict" methods
        An object of that type which is cloned for each validation.

    title : string
        Title for the chart.

    X : array-like, shape (n_samples, n_features)
        Training vector, where n_samples is the number of samples and
        n_features is the number of features.

    y : array-like, shape (n_samples) or (n_samples, n_features), optional
        Target relative to X for classification or regression;
        None for unsupervised learning.

    ylim : tuple, shape (ymin, ymax), optional
        Defines minimum and maximum yvalues plotted.

    cv : int, cross-validation generator or an iterable, optional
        Determines the cross-validation splitting strategy.
        Possible inputs for cv are:
          - None, to use the default 3-fold cross-validation,
          - integer, to specify the number of folds.
          - :term:`CV splitter`,
          - An iterable yielding (train, test) splits as arrays of indices.

        For integer/None inputs, if ``y`` is binary or multiclass,
        :class:`StratifiedKFold` used. If the estimator is not a classifier
        or if ``y`` is neither binary nor multiclass, :class:`KFold` is used.

        Refer :ref:`User Guide <cross_validation>` for the various
        cross-validators that can be used here.

    n_jobs : int or None, optional (default=None)
        Number of jobs to run in parallel.
        ``None`` means 1 unless in a :obj:`joblib.parallel_backend` context.
        ``-1`` means using all processors. See :term:`Glossary <n_jobs>`
        for more details.

    train_sizes : array-like, shape (n_ticks,), dtype float or int
        Relative or absolute numbers of training examples that will be used to
        generate the learning curve. If the dtype is float, it is regarded as a
        fraction of the maximum size of the training set (that is determined
        by the selected validation method), i.e. it has to be within (0, 1].
        Otherwise it is interpreted as absolute sizes of the training sets.
        Note that for classification the number of samples usually have to
        be big enough to contain at least one sample from each class.
        (default: np.linspace(0.1, 1.0, 5))
    """
    plt.figure()
    plt.title(title)
    if ylim is not None:
        plt.ylim(*ylim)
    plt.xlabel("Training examples")
    plt.ylabel("Accuracy")
    train_sizes, train_scores, test_scores = learning_curve(
        estimator, X, y, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes)
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    plt.grid()

    plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=0.1,
                     color="r")
    plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std, alpha=0.1, color="g")
    plt.plot(train_sizes, train_scores_mean, 'o-', color="r",
             label="Training")
    plt.plot(train_sizes, test_scores_mean, 'o-', color="g",
             label="Cross-validation")

    plt.legend(loc="best")
    return plt


def run_kfold(clf, x, y, folds):
    kf = KFold(folds)
    outcomes = []
    fold = 0

    for train_index, test_index in kf.split(x):
        fold += 1
        X_train, X_test = x.values[train_index], x.values[test_index]
        y_train, y_test = y.values[train_index], y.values[test_index]
        clf.fit(X_train, y_train)
        predictions = clf.predict(X_test)
        accuracy = accuracy_score(y_test, predictions)
        outcomes.append(accuracy)
        print("Fold {0} accuracy: {1}".format(fold, accuracy))
    mean_outcome = np.mean(outcomes)
    print("Mean Accuracy: {0}".format(mean_outcome))
    return mean_outcome


cols = ['LIMIT_BAL', 'SEX', 'EDUCATION', 'MARRIAGE', 'AGE', 'PAY_0', 'PAY_2', 'PAY_3', 'PAY_4', 'PAY_5', 'PAY_6', 'BILL_AMT1',
        'BILL_AMT2', 'BILL_AMT3', 'BILL_AMT4', 'BILL_AMT5', 'BILL_AMT6', 'PAY_AMT1', 'PAY_AMT2', 'PAY_AMT3', 'PAY_AMT4', 'PAY_AMT5',
        'PAY_AMT6']

# https://www.kaggle.com/rajatshah/scikit-learn-ml-from-start-to-finish
data_train = pd.read_csv('creditcards.train.csv', quoting=csv.QUOTE_NONE)
data_test = pd.read_csv('creditcards.test.csv', quoting=csv.QUOTE_NONE)

y_train = data_train['DEFAULT_NEXT_MONTH']
y_test = data_test['DEFAULT_NEXT_MONTH']

removedCols = ['DEFAULT_NEXT_MONTH', 'ID']

X_train = data_train.drop(removedCols, axis=1)
X_test = data_test.drop(removedCols, axis=1)

"""
https://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeClassifier.html#sklearn.tree.DecisionTreeClassifier
decision trees
criterion : string, optional (default=”gini”)
The function to measure the quality of a split. Supported criteria are “gini” for the Gini impurity and “entropy”
for the information gain.

splitter : string, optional (default=”best”)
The strategy used to choose the split at each node. Supported strategies are “best” to choose the best split and
“random” to choose the best random split.

max_depth : int or None, optional (default=None)
The maximum depth of the tree. If None, then nodes are expanded until all leaves are pure or until all leaves
contain less than min_samples_split samples.

min_samples_split : int, float, optional (default=2)
The minimum number of samples required to split an internal node:
If int, then consider min_samples_split as the minimum number.
If float, then min_samples_split is a fraction and ceil(min_samples_split * n_samples) are the minimum number of
samples for each split.

min_samples_leaf : int, float, optional (default=1)
The minimum number of samples required to be at a leaf node. A split point at any depth will only be considered if
it leaves at least min_samples_leaf training samples in each of the left and right branches. This may have the
effect of smoothing the model, especially in regression.
If int, then consider min_samples_leaf as the minimum number.
If float, then min_samples_leaf is a fraction and ceil(min_samples_leaf * n_samples) are the minimum number of
samples for each node.

min_weight_fraction_leaf : float, optional (default=0.)
The minimum weighted fraction of the sum total of weights (of all the input samples) required to be at a leaf node.
Samples have equal weight when sample_weight is not provided.

max_features : int, float, string or None, optional (default=None)
The number of features to consider when looking for the best split:
If int, then consider max_features features at each split.
If float, then max_features is a fraction and int(max_features * n_features) features are considered at each split.
If “auto”, then max_features=sqrt(n_features).
If “sqrt”, then max_features=sqrt(n_features).
If “log2”, then max_features=log2(n_features).
If None, then max_features=n_features.
Note: the search for a split does not stop until at least one valid partition of the node samples is found, even if
it requires to effectively inspect more than max_features features.

random_state : int, RandomState instance or None, optional (default=None)
If int, random_state is the seed used by the random number generator; If RandomState instance, random_state is the
random number generator; If None, the random number generator is the RandomState instance used by np.random.

max_leaf_nodes : int or None, optional (default=None)
Grow a tree with max_leaf_nodes in best-first fashion. Best nodes are defined as relative reduction in impurity.
If None then unlimited number of leaf nodes.

min_impurity_decrease : float, optional (default=0.)
A node will be split if this split induces a decrease of the impurity greater than or equal to this value.
The weighted impurity decrease equation is the following:
N_t / N * (impurity - N_t_R / N_t * right_impurity
                    - N_t_L / N_t * left_impurity)
where N is the total number of samples, N_t is the number of samples at the current node, N_t_L is the number of
samples in the left child, and N_t_R is the number of samples in the right child.
N, N_t, N_t_R and N_t_L all refer to the weighted sum, if sample_weight is passed.

presort : bool, optional (default=False)
Whether to presort the data to speed up the finding of best splits in fitting. For the default settings of a
decision tree on large datasets, setting this to true may slow down the training process. When using either a
smaller dataset or a restricted depth, this may speed up the training.

##############################################
AdaBoostClassifier
https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.AdaBoostClassifier.html

base_estimator : object, optional (default=None)
The base estimator from which the boosted ensemble is built. Support for sample weighting is required, as well as
proper classes_ and n_classes_ attributes. If None, then the base estimator is DecisionTreeClassifier(max_depth=1)

n_estimators : integer, optional (default=50)
The maximum number of estimators at which boosting is terminated. In case of perfect fit, the learning procedure is
stopped early.

learning_rate : float, optional (default=1.)
Learning rate shrinks the contribution of each classifier by learning_rate. There is a trade-off between learning_rate
and n_estimators.

algorithm : {‘SAMME’, ‘SAMME.R’}, optional (default=’SAMME.R’)
If ‘SAMME.R’ then use the SAMME.R real boosting algorithm. base_estimator must support calculation of class
probabilities. If ‘SAMME’ then use the SAMME discrete boosting algorithm. The SAMME.R algorithm typically converges
faster than SAMME, achieving a lower test error with fewer boosting iterations.

random_state : int, RandomState instance or None, optional (default=None)
If int, random_state is the seed used by the random number generator; If RandomState instance, random_state is the
random number generator; If None, the random number generator is the RandomState instance used by np.random.
"""

train_results = []
test_results = []

params = [1, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60]
for p in params:

    clf = AdaBoostClassifier(
        tree.DecisionTreeClassifier(
            random_state=0,
            criterion='entropy',
            splitter='random',
            max_depth=7,
            min_samples_split=30,
        ),
        random_state=0,
        n_estimators=p,
    )

    now = datetime.datetime.now()
    print("{0} tree n_estimators = '{1}'".format(
        now.strftime("%m/%d %I:%M %p"), p))

    print(clf)

    cv_mean = run_kfold(clf, X_train, y_train, 3)
    print("Mean Improvement: {0}".format(cv_mean - 0.8199166666666665))
    train_results.append(cv_mean)

    clf = clf.fit(X_train, y_train)
    predictions = clf.predict(X_test)
    test_accuracy = accuracy_score(y_test, predictions)
    print("Test Accuracy: {0}".format(test_accuracy))
    print("Test Correct: {0}".format(
        accuracy_score(y_test, predictions, normalize=False)))
    test_results.append(test_accuracy)

    print("Test Improvement: {0}".format(
        test_accuracy - 0.8203333333333334))

line1, = plt.plot(params, train_results, 'b', label="Train Avg")
line2, = plt.plot(params, test_results, 'r', label="Test")
plt.legend(handler_map={line1: HandlerLine2D(numpoints=2)})
plt.title('Credit Cards Boosting Hyperparameter "n_estimators"')
plt.ylabel('Accuracy')
plt.xlabel('n_estimators')
plt.savefig('creditcards.decisiontreesboost.n_estimators.png')
# https://medium.com/@mohtedibf/indepth-parameter-tuning-for-decision-tree-6753118a03c3


train_results = []
test_results = []

params = [None, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10,
          11, 12, 13, 14, 15, 16, 17, 18, 19, 20,
          21, 22, 23, 24, 25, 26, 27, 28, 29, 30]
for p in params:

    clf = AdaBoostClassifier(
        tree.DecisionTreeClassifier(
            random_state=0,
            criterion='entropy',
            splitter='random',
            max_depth=p,
            min_samples_split=30,
        ),
        random_state=0,
        n_estimators=1,
    )

    now = datetime.datetime.now()
    print("{0} tree max_depth = '{1}'".format(
        now.strftime("%m/%d %I:%M %p"), p))

    print(clf)

    cv_mean = run_kfold(clf, X_train, y_train, 3)
    print("Mean Improvement: {0}".format(cv_mean - 0.8199166666666665))
    train_results.append(cv_mean)

    clf = clf.fit(X_train, y_train)
    predictions = clf.predict(X_test)
    test_accuracy = accuracy_score(y_test, predictions)
    print("Test Accuracy: {0}".format(test_accuracy))
    print("Test Correct: {0}".format(
        accuracy_score(y_test, predictions, normalize=False)))
    test_results.append(test_accuracy)

    print("Test Improvement: {0}".format(
        test_accuracy - 0.8203333333333334))

plt.figure()
line1, = plt.plot(params, train_results, 'b', label="Train Avg")
line2, = plt.plot(params, test_results, 'r', label="Test")
plt.legend(handler_map={line1: HandlerLine2D(numpoints=2)})
plt.title('Credit Cards Boosting Hyperparameter "max_depth"')
plt.ylabel('Accuracy')
plt.xlabel('max_depth')
plt.savefig('creditcards.decisiontreesboost.max_depth.png')
# https://medium.com/@mohtedibf/indepth-parameter-tuning-for-decision-tree-6753118a03c3


clf = AdaBoostClassifier(
    tree.DecisionTreeClassifier(
        random_state=0,
        criterion='entropy',
        splitter='random',
        max_depth=7,
        min_samples_split=30,
    ),
    random_state=0,
    n_estimators=1,
)

plt = plot_learning_curve(
    clf, 'Credit Cards Boosting Learning Curve', X_train, y_train, None, 10)
plt.savefig('creditcards.decisiontreesboost.learningcurve.png')
