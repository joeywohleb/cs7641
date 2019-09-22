import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from matplotlib.legend_handler import HandlerLine2D
import seaborn as sns
from sklearn import preprocessing
from sklearn.model_selection import train_test_split, GridSearchCV, KFold, learning_curve
import csv
from sklearn.tree.export import export_text
from sklearn.neural_network import MLPClassifier
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

# https://scikit-learn.org/stable/modules/generated/sklearn.neural_network.MLPClassifier.html
# hidden_layer_sizes : tuple, length = n_layers - 2, default (100,)
# activation : {'identity', 'logistic', 'tanh', 'relu'}, default 'relu'
# solver : {'lbfgs', 'sgd', 'adam'}, default 'adam'
# beta_1 : float, optional, default 0.9; Only used when solver='adam'
# beta_2 : float, optional, default 0.999; Only used when solver='adam'
# epsilon : float, optional, default 1e-8; Only used when solver='adam'
# n_iter_no_change : int, optional, default 10; Only used when solver='adam' or 'sgd'
# momentum : float, default 0.9; only for 'sgd' solver
# nesterovs_momentum : boolean, default True; only for 'sgd' solver
# alpha : float, optional, default 0.0001
# batch_size : int, optional, default 'auto'
# learning_rate : {'constant', 'invscaling', 'adaptive'}, default 'constant'
# learning_rate_init : double, optional, default 0.001
# power_t : double, optional, default 0.5
# max_iter : int, optional, default 200
# shuffle : bool, optional, default True
# random_state : int, RandomState instance or None, optional, default None
# tol : float, optional, default 1e-4
# early_stopping : bool, default False
# validation_fraction : float, optional, default 0.1; only used if early_stopping True

train_results = []
test_results = []


params = [.001, .1, .2, .3, .4, .5, .6, .7, .8, .9, .999]
for p in params:

    clf = MLPClassifier(
        random_state=0,
        hidden_layer_sizes=(100, 100),
        solver='adam',
        activation='relu',
        beta_1=p,
        beta_2=.9,
        n_iter_no_change=10,
        alpha=.0001,
        batch_size='auto',
        learning_rate='constant',
        learning_rate_init=.001,
        power_t=.5,
        max_iter=200,
        early_stopping=False,
        warm_start=True,
    )

    now = datetime.datetime.now()
    print("{0} beta_1 = '{1}'".format(
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
plt.title('Income Neural Network Hyperparameter "beta_1"')
plt.ylabel('Accuracy')
plt.xlabel('beta_1')
plt.savefig('income.neuralnetwork.beta_1.png')
# https://medium.com/@mohtedibf/indepth-parameter-tuning-for-decision-tree-6753118a03c3

train_results = []
test_results = []

params = [.001, .1, .2, .3, .4, .5, .6, .7, .8, .9, .999]
for p in params:

    clf = MLPClassifier(
        random_state=0,
        hidden_layer_sizes=(100, 100),
        solver='adam',
        activation='relu',
        beta_1=.7,
        beta_2=p,
        n_iter_no_change=10,
        alpha=.0001,
        batch_size='auto',
        learning_rate='constant',
        learning_rate_init=.001,
        power_t=.5,
        max_iter=200,
        early_stopping=False,
        warm_start=True,
    )

    now = datetime.datetime.now()
    print("{0} beta_2 = '{1}'".format(
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
plt.title('Income Neural Network Hyperparameter "beta_2"')
plt.ylabel('Accuracy')
plt.xlabel('beta_2')
plt.savefig('income.neuralnetwork.beta_2.png')
# https://medium.com/@mohtedibf/indepth-parameter-tuning-for-decision-tree-6753118a03c3

clf = MLPClassifier(
    random_state=0,
    hidden_layer_sizes=(100, 100),
    solver='adam',
    activation='relu',
    beta_1=.7,
    beta_2=.9,
    n_iter_no_change=10,
    alpha=.0001,
    batch_size='auto',
    learning_rate='constant',
    learning_rate_init=.001,
    power_t=.5,
    max_iter=200,
    early_stopping=False,
    warm_start=True,
)

plt = plot_learning_curve(
    clf, 'Income Neural Network Learning Curve', X_train, y_train, None, 10)
plt.savefig('income.neuralnetwork.learningcurve.png')

clf = MLPClassifier(
    random_state=0,
    hidden_layer_sizes=(100, 100),
    solver='adam',
    activation='relu',
    beta_1=.7,
    beta_2=.9,
    n_iter_no_change=10,
    alpha=.0001,
    batch_size='auto',
    learning_rate='constant',
    learning_rate_init=.001,
    power_t=.5,
    early_stopping=False,
    warm_start=True,
    max_iter=10,
)

np.random.seed(1)

X_train = X_train.as_matrix()
X_test = X_test.as_matrix()
y_train = y_train.as_matrix()
y_test = y_test.as_matrix()

# https://stackoverflow.com/questions/46912557/is-it-possible-to-get-test-scores-for-each-iteration-of-mlpclassifier
N_TRAIN_SAMPLES = X_train.shape[0]
N_EPOCHS = 100
N_BATCH = 128
N_CLASSES = np.unique(y_train)

scores_train = []
scores_test = []

# EPOCH
epoch = 0
while epoch < N_EPOCHS:
    print('epoch: ', epoch)
    # SHUFFLING
    random_perm = np.random.permutation(X_train.shape[0])
    mini_batch_index = 0
    while True:
        # MINI-BATCH
        indices = random_perm[mini_batch_index:mini_batch_index + N_BATCH]
        clf.partial_fit(X_train[indices], y_train[indices], classes=N_CLASSES)
        mini_batch_index += N_BATCH

        if mini_batch_index >= N_TRAIN_SAMPLES:
            break

    # SCORE TRAIN
    scores_train.append(clf.score(X_train, y_train))

    # SCORE TEST
    scores_test.append(clf.score(X_test, y_test))

    epoch += 1

plt.figure()
plt.plot(scores_train, color='b', alpha=0.8, label='Train')
plt.plot(scores_test, color='r', alpha=0.8, label='Test')
plt.title("Income Neural Network Accuracy over epochs", fontsize=14)
plt.xlabel('Epochs')
plt.ylabel("Accuracy")
plt.legend(loc='upper left')
plt.savefig('income.neuralnetwork.accuracy.png')
