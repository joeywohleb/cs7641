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
from sklearn.cluster import KMeans
from yellowbrick.cluster import KElbowVisualizer, InterclusterDistance, SilhouetteVisualizer
import seaborn as sns

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

removedCols = ['DEFAULT_NEXT_MONTH', 'ID', 'SEX', 'EDUCATION', 'MARRIAGE']

X_train = data_train.drop(removedCols, axis=1)
X_test = data_test.drop(removedCols, axis=1)

"""
https://scikit-learn.org/stable/modules/generated/sklearn.cluster.KMeans.html
n_clusters : int, optional, default: 8
The number of clusters to form as well as the number of centroids to generate.

init : {‘k-means++’, ‘random’ or an ndarray}
Method for initialization, defaults to ‘k-means++’:

‘k-means++’ : selects initial cluster centers for k-mean clustering in a smart way to speed up convergence. See section
Notes in k_init for more details.

‘random’: choose k observations (rows) at random from data for the initial centroids.

If an ndarray is passed, it should be of shape (n_clusters, n_features) and gives the initial centers.

n_init : int, default: 10
Number of time the k-means algorithm will be run with different centroid seeds. The final results will be the best
output of n_init consecutive runs in terms of inertia.

max_iter : int, default: 300
Maximum number of iterations of the k-means algorithm for a single run.

tol : float, default: 1e-4
Relative tolerance with regards to inertia to declare convergence

precompute_distances : {‘auto’, True, False}
Precompute distances (faster but takes more memory).

‘auto’ : do not precompute distances if n_samples * n_clusters > 12 million. This corresponds to about 100MB overhead
per job using double precision.

True : always precompute distances

False : never precompute distances

verbose : int, default 0
Verbosity mode.

random_state : int, RandomState instance or None (default)
Determines random number generation for centroid initialization. Use an int to make the randomness deterministic. See
Glossary.

copy_x : boolean, optional
When pre-computing distances it is more numerically accurate to center the data first. If copy_x is True (default), then
the original data is not modified, ensuring X is C-contiguous. If False, the original data is modified, and put back
before the function returns, but small numerical differences may be introduced by subtracting and then adding the data
mean, in this case it will also not ensure that data is C-contiguous which may cause a significant slowdown.

n_jobs : int or None, optional (default=None)
The number of jobs to use for the computation. This works by computing each of the n_init runs in parallel.

None means 1 unless in a joblib.parallel_backend context. -1 means using all processors. See Glossary for more details.

algorithm : “auto”, “full” or “elkan”, default=”auto”
K-means algorithm to use. The classical EM-style algorithm is “full”. The “elkan” variation is more efficient by using 
the triangle inequality, but currently doesn’t support sparse data. “auto” chooses “elkan” for dense data and “full” for
sparse data.
"""


model = KMeans(
    random_state=0,
    n_jobs=-1,
)

# https://www.scikit-yb.org/en/latest/api/cluster/elbow.html
visualizer = KElbowVisualizer(model, k=(1, 20))

visualizer.fit(X_train)        # Fit the data to the visualizer
# Finalize and render the figure
visualizer.show(outpath="charts/creditcards.k-means.KElbowVisualizer.png")
visualizer.poof()

model = KMeans(
    n_clusters=5,
    random_state=0,
    n_jobs=-1,
)
visualizer = InterclusterDistance(model)

visualizer.fit(X_train)        # Fit the data to the visualizer
# Finalize and render the figure
visualizer.show(outpath="charts/creditcards.k-means.InterclusterDistance.png")
visualizer.poof()

# https://www.scikit-yb.org/en/latest/api/cluster/silhouette.html
model = KMeans(
    n_clusters=5,
    random_state=0
)
visualizer = SilhouetteVisualizer(model)

visualizer.fit(X_train)       # Fit the data to the visualizer
# Finalize and render the figure
visualizer.show(outpath="charts/creditcards.k-means.SilhouetteVisualizer.png")
