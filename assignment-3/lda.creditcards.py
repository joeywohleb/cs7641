import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from matplotlib.legend_handler import HandlerLine2D
from sklearn import tree, preprocessing
from sklearn.preprocessing import minmax_scale
from sklearn.model_selection import train_test_split, GridSearchCV, KFold, learning_curve
import csv
from sklearn.tree.export import export_text
from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import make_scorer, accuracy_score
import datetime
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.cluster import KMeans
from yellowbrick.cluster import KElbowVisualizer, InterclusterDistance, SilhouetteVisualizer

from sklearn.mixture import GaussianMixture

import itertools
from scipy import linalg
import matplotlib as mpl


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
	
n_components : int, float, None or string
Number of components to keep. if n_components is not set all components are kept:

n_components == min(n_samples, n_features)
If n_components == 'mle' and svd_solver == 'full', Minka’s MLE is used to guess the dimension. Use of
n_components == 'mle' will interpret svd_solver == 'auto' as svd_solver == 'full'.

If 0 < n_components < 1 and svd_solver == 'full', select the number of components such that the amount of variance that
needs to be explained is greater than the percentage specified by n_components.

If svd_solver == 'arpack', the number of components must be strictly less than the minimum of n_features and n_samples.

Hence, the None case results in:

n_components == min(n_samples, n_features) - 1
copy : bool (default True)
If False, data passed to fit are overwritten and running fit(X).transform(X) will not yield the expected results, use
fit_transform(X) instead.

whiten : bool, optional (default False)
When True (False by default) the components_ vectors are multiplied by the square root of n_samples and then divided by
the singular values to ensure uncorrelated outputs with unit component-wise variances.

Whitening will remove some information from the transformed signal (the relative variance scales of the components) but
can sometime improve the predictive accuracy of the downstream estimators by making their data respect some hard-wired
assumptions.

svd_solver : string {‘auto’, ‘full’, ‘arpack’, ‘randomized’}
auto :
the solver is selected by a default policy based on X.shape and n_components: if the input data is larger than 500x500
and the number of components to extract is lower than 80% of the smallest dimension of the data, then the more efficient
‘randomized’ method is enabled. Otherwise the exact full SVD is computed and optionally truncated afterwards.

full :
run exact full SVD calling the standard LAPACK solver via scipy.linalg.svd and select the components by postprocessing

arpack :
run SVD truncated to n_components calling ARPACK solver via scipy.sparse.linalg.svds. It requires strictly
0 < n_components < min(X.shape)

randomized :
run randomized SVD by the method of Halko et al.

New in version 0.18.0.

tol : float >= 0, optional (default .0)
Tolerance for singular values computed by svd_solver == ‘arpack’.

New in version 0.18.0.

iterated_power : int >= 0, or ‘auto’, (default ‘auto’)
Number of iterations for the power method computed by svd_solver == ‘randomized’.

New in version 0.18.0.

random_state : int, RandomState instance or None, optional (default None)
If int, random_state is the seed used by the random number generator; If RandomState instance, random_state is the
random number generator; If None, the random number generator is the RandomState instance used by np.random. Used when
svd_solver == ‘arpack’ or ‘randomized’.
"""

train_results = []
test_results = []


clf = LatentDirichletAllocation(
    random_state=0,
)

print(clf)

results = clf.fit_transform(minmax_scale(X_train, feature_range=(0, 1)))

model = KMeans(
    random_state=0,
    n_jobs=-1,
)

# https://www.scikit-yb.org/en/latest/api/cluster/elbow.html
visualizer = KElbowVisualizer(model, k=(1, 20))

visualizer.fit(results)        # Fit the data to the visualizer
# Finalize and render the figure
visualizer.show(outpath="charts/creditcards.k-means.LDA.KElbowVisualizer.png")
visualizer.poof()

model = KMeans(
    n_clusters=6,
    random_state=0,
    n_jobs=-1,
)
visualizer = InterclusterDistance(model)

visualizer.fit(results)        # Fit the data to the visualizer
# Finalize and render the figure
visualizer.show(
    outpath="charts/creditcards.k-means.LDA.InterclusterDistance.png")
visualizer.poof()

model = KMeans(
    n_clusters=6,
    random_state=0
)
visualizer = SilhouetteVisualizer(model)

visualizer.fit(results)       # Fit the data to the visualizer
# Finalize and render the figure
visualizer.show(
    outpath="charts/creditcards.k-means.LDA.SilhouetteVisualizer.png")


lowest_bic = np.infty
bic = []
n_components_range = range(1, 10)
cv_types = ['spherical', 'tied', 'diag', 'full']
for cv_type in cv_types:
    for n_components in n_components_range:
        # Fit a Gaussian mixture with EM
        gmm = GaussianMixture(n_components=n_components,
                              covariance_type=cv_type)
        gmm.fit(results)
        bic.append(gmm.bic(results))
        if bic[-1] < lowest_bic:
            lowest_bic = bic[-1]
            best_gmm = gmm

bic = np.array(bic)
color_iter = itertools.cycle(['navy', 'turquoise', 'cornflowerblue',
                              'darkorange'])
clf = best_gmm
bars = []

# Plot the BIC scores
plt.figure(figsize=(8, 6))
spl = plt.subplot(2, 1, 1)
for i, (cv_type, color) in enumerate(zip(cv_types, color_iter)):
    xpos = np.array(n_components_range) + .2 * (i - 2)
    bars.append(plt.bar(xpos, bic[i * len(n_components_range):
                                  (i + 1) * len(n_components_range)],
                        width=.2, color=color))
plt.xticks(n_components_range)
plt.ylim([bic.min() * 1.01 - .01 * bic.max(), bic.max()])
plt.title('BIC score per model')
xpos = np.mod(bic.argmin(), len(n_components_range)) + .65 +\
    .2 * np.floor(bic.argmin() / len(n_components_range))
plt.text(xpos, bic.min() * 0.97 + .03 * bic.max(), '*', fontsize=14)
spl.set_xlabel('Number of components')
spl.legend([b[0] for b in bars], cv_types)
plt.savefig('charts/expectation-max.lda.creditcards.png')

results = clf.predict(X_train)

uniqueValues, occurCount = np.unique(results, return_counts=True)

print("Clusters : ", uniqueValues)
print("Cluster memberships : ", occurCount)
