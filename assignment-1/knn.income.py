import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from matplotlib.legend_handler import HandlerLine2D
import seaborn as sns
from sklearn import preprocessing
from sklearn.model_selection import train_test_split, GridSearchCV, KFold, learning_curve
import csv
from sklearn.tree.export import export_text
from sklearn.neighbors import KNeighborsClassifier
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


def encode_features(df_train, df_test):
    features = ['WORK_CLASS', 'MARITAL_STATUS', 'OCCUPATION',
                'RELATIONSHIP', 'RACE', 'SEX', 'NATIVE_COUNTRY', 'CLASS_LABEL']
    df_combined = pd.concat([df_train[features], df_test[features]])

    for feature in features:
        le = preprocessing.LabelEncoder()
        le = le.fit(df_combined[feature])
        df_train[feature] = le.transform(df_train[feature])
        df_test[feature] = le.transform(df_test[feature])
    return df_train, df_test


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


cols = ['WORK_CLASS', 'EDUCATION_NUM',
        'MARITAL_STATUS', 'OCCUPATION', 'RELATIONSHIP', 'RACE', 'SEX', 'CAPITAL_GAIN', 'CAPITAL_LOSS', 'HOURS_WORKED_PER_WEEK', 'NATIVE_COUNTRY']

# https://www.kaggle.com/rajatshah/scikit-learn-ml-from-start-to-finish
data_train = pd.read_csv('income.train.csv', quoting=csv.QUOTE_NONE)
data_test = pd.read_csv('income.test.csv', quoting=csv.QUOTE_NONE)

data_train, data_test = encode_features(data_train, data_test)

y_train = data_train['CLASS_LABEL']
y_test = data_test['CLASS_LABEL']

removedCols = ['CLASS_LABEL', 'FINAL_WEIGHT', 'EDUCATION']

X_train = data_train.drop(removedCols, axis=1)
X_test = data_test.drop(removedCols, axis=1)

# https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsClassifier.html
# n_neighbors : int, optional (default = 5)
# weights : {'uniform', 'distance'} optional (default = 'uniform')
# algorithm : {'auto', 'ball_tree', 'kd_tree', 'brute'}, optional
# leaf_size : int, optional (default = 30), Leaf size passed to BallTree or KDTree
# p : integer, optional (default = 2), p = 1 for manhattan_distance, euclidean_distance for p = 2
# metric : string or callable, default 'minkowski'
# n_jobs : int or None, optional (default=None), None means 1, -1 means using all processors

train_results = []
test_results = []

params = [
    1, 2, 3, 4, 5, 6, 7, 8, 9, 10,
    11, 12, 13, 14, 15, 16, 17, 18, 19, 20,
    21, 22, 23, 24, 25, 26, 27, 28, 29, 30,
    31, 32, 33, 34, 35, 36, 37, 38, 39, 40,
    41, 42, 43, 44, 45, 46, 47, 48, 49, 50,
    51, 52, 53, 54, 55, 56, 57, 58, 59, 60,
    61, 62, 63, 64, 65, 66, 67, 68, 69, 70,
    71, 72, 73, 74, 75, 76, 77, 78, 79, 80,
    81, 82, 83, 84, 85, 86, 87, 88, 89, 90,
    91, 92, 93, 94, 95, 96, 97, 98, 99, 100
]

for p in params:

    clf = KNeighborsClassifier(
        n_neighbors=p,
        algorithm='brute',
        p=1,
    )

    now = datetime.datetime.now()
    print("{0} n_neighbors = '{1}'".format(
        now.strftime("%m/%d %I:%M %p"), p))

    print(clf)

    cv_mean = run_kfold(clf, X_train, y_train, 10)
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
plt.title('Income k-NN Hyperparameter "n_neighbors"')
plt.ylabel('Accuracy')
plt.xlabel('n_neighbors')
plt.savefig('income.knn.n_neighbors.png')
# https://medium.com/@mohtedibf/indepth-parameter-tuning-for-decision-tree-6753118a03c3

train_results = []
test_results = []


params = ['chebyshev', 'euclidean', 'manhattan',
          'minkowski', 'seuclidean', 'mahalanobis']
for p in params:

    clf = KNeighborsClassifier(
        n_neighbors=14,
        algorithm='brute',
        p=1,
        metric=p,
    )

    now = datetime.datetime.now()
    print("{0} metric = '{1}'".format(
        now.strftime("%m/%d %I:%M %p"), p))

    print(clf)

    cv_mean = run_kfold(clf, X_train, y_train, 10)
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
plt.title('Income k-NN Hyperparameter "metric"')
plt.ylabel('Accuracy')
plt.xlabel('metric')
plt.savefig('income.knn.metric.png')
# https://medium.com/@mohtedibf/indepth-parameter-tuning-for-decision-tree-6753118a03c3

clf = KNeighborsClassifier(
    n_neighbors=14,
    algorithm='brute',
    p=1,
)

plt = plot_learning_curve(
    clf, 'Income k-NN Learning Curve', X_train, y_train, None, 10)
plt.savefig('income.knn.learningcurve.png')
