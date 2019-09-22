import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from matplotlib.legend_handler import HandlerLine2D
import seaborn as sns
from sklearn import preprocessing
from sklearn.model_selection import train_test_split, GridSearchCV, KFold, learning_curve
import csv
from sklearn.tree.export import export_text
from sklearn.svm import SVC
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

# https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html
# C : float, optional (default=1.0)
# kernel : {'linear', 'poly', 'rbf', 'sigmoid', 'precomputed'}, default 'rbf'
# degree : int, optional (default=3)
# gamma : float, optional (default='auto'), Kernel coefficient for 'rbf', 'poly' and 'sigmoid'.
# coef0 : float, optional (default=0.0), only significant in 'poly' and 'sigmoid'
# shrinking : boolean, optional (default=True)
# probability : boolean, optional (default=False)
# tol : float, optional (default=1e-3)
# cache_size : float, optional
# class_weight : {dict, 'balanced'}, optional
# verbose : bool, default: False
# max_iter : int, optional (default=-1),  or -1 for no limit.
# decision_function_shape : 'ovo', 'ovr', default='ovr'
# random_state : int, RandomState instance or None, optional (default=None)

train_results = []
test_results = []


params = [.1, .2, .3, .4, .5, .6, .7, .8, .9, 1]
for p in params:
    clf = SVC(
        random_state=0,
        kernel='rbf',
        gamma=p,
        tol=.04,
    )

    now = datetime.datetime.now()
    print("{0} gamma = '{1}'".format(
        now.strftime("%m/%d %I:%M %p"), p))

    print(clf)

    cv_mean = run_kfold(clf, X_train, y_train, 3)
    train_results.append(cv_mean)

    clf = clf.fit(X_train, y_train)
    predictions = clf.predict(X_test)
    test_accuracy = accuracy_score(y_test, predictions)
    print("Test Accuracy: {0}".format(test_accuracy))
    print("Test Correct: {0}".format(
        accuracy_score(y_test, predictions, normalize=False)))
    test_results.append(test_accuracy)

plt.figure()
line1, = plt.plot(params, train_results, 'b', label="Train Avg")
line2, = plt.plot(params, test_results, 'r', label="Test")
plt.legend(handler_map={line1: HandlerLine2D(numpoints=2)})
plt.title('Credit Cards SVM Hyperparameter "gamma"')
plt.ylabel('Accuracy')
plt.xlabel('gamma')
plt.savefig('creditcards.svm.gamma.png')
# https://medium.com/@mohtedibf/indepth-parameter-tuning-for-decision-tree-6753118a03c3

train_results = []
test_results = []


params = [.01, .02, .03, .04, .05, .06, .07, .08, .09, .1]
for p in params:

    clf = SVC(
        random_state=0,
        gamma=.1,
        tol=p,
        kernel='rbf',
    )

    now = datetime.datetime.now()
    print("{0} tol = '{1}'".format(
        now.strftime("%m/%d %I:%M %p"), p))

    print(clf)

    cv_mean = run_kfold(clf, X_train, y_train, 3)
    train_results.append(cv_mean)

    clf = clf.fit(X_train, y_train)
    predictions = clf.predict(X_test)

    test_accuracy = accuracy_score(y_test, predictions)
    print("Test Accuracy: {0}".format(test_accuracy))
    print("Test Correct: {0}".format(
        accuracy_score(y_test, predictions, normalize=False)))
    test_results.append(test_accuracy)

plt.figure()
line1, = plt.plot(params, train_results, 'b', label="Train Avg")
line2, = plt.plot(params, test_results, 'r', label="Test")
plt.legend(handler_map={line1: HandlerLine2D(numpoints=2)})
plt.title('Credit Cards SVM Hyperparameter "tol"')
plt.ylabel('Accuracy')
plt.xlabel('tol')
plt.savefig('creditcards.svm.tol.png')
# https://medium.com/@mohtedibf/indepth-parameter-tuning-for-decision-tree-6753118a03c3

clf = SVC(
    random_state=0,
    kernel='rbf',
    gamma=.1,
    tol=.04,
)

plt = plot_learning_curve(
    clf, 'Credit Cards SVM Learning Curve', X_train, y_train, None, 10)
plt.savefig('creditcards.svm.learningcurve.png')
