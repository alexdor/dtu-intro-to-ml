from os import path

import matplotlib.pyplot as plt
import numpy as np

import pandas as pd

import sklearn.linear_model as lm
import torch
from matplotlib.pylab import (
    figure,
    grid,
    legend,
    loglog,
    semilogx,
    show,
    subplot,
    title,
    xlabel,
    ylabel,
)
from scipy import stats
from sklearn import linear_model as lm
from sklearn import model_selection

from toolbox_02450 import draw_neural_net, rlr_validate, train_neural_net


class AutoVivification(dict):
    """Implementation of perl's autovivification feature."""

    def __getitem__(self, item):
        try:
            return dict.__getitem__(self, item)
        except KeyError:
            value = self[item] = type(self)()
            return value


n_replicates = 15  # number of networks trained in each k-fold
max_iter = 10000

script_dir = path.dirname(__file__)  # <-- absolute dir the script is in
rel_path = "./heart.csv"
data_file = path.join(script_dir, rel_path)


df = pd.read_csv(data_file)
target = "chol"

y = np.array([df[target]])
y = (y - y.mean()) / y.std()
y = y.transpose()
X = df.loc[:, df.columns != target]
norm_vars = df[["age", "trestbps", "thalach", "oldpeak", "ca", "slope"]]
norm_vars = (norm_vars - norm_vars.mean()) / norm_vars.std()
new = pd.DataFrame()

discrete = ["sex", "cp", "fbs", "restecg", "exang", "thal", "target"]
for col in discrete:
    temp = pd.get_dummies(df[col], prefix=col)
    new = pd.concat([new, temp], axis=1)

d = pd.DataFrame(np.ones((new.shape[0], 1)))
X = pd.concat([d, new], axis=1)
attribute_names = list(X)
X = np.array(X)

N, M = X.shape
C = 2


## Crossvalidation
# Create crossvalidation partition for evaluation
K = 10
internal_cross_validation = K
CV = model_selection.KFold(K, shuffle=False)

# Values of lambda
lambdas = list(np.power(10.0, range(-3, 7)))
hidden_layers = list(np.arange(start=1, stop=19))


# Initialize variables
Error_train = np.empty((K, 1))
Error_test = np.empty((K, 1))
Error_train_rlr = np.empty((K, 1))
Error_test_rlr = np.empty((K, 1))
Error_train_nofeatures = np.empty((K, 1))
Error_test_nofeatures = np.empty((K, 1))
w_rlr = np.empty((M, K))
mu = np.empty((K, M - 1))
sigma = np.empty((K, M - 1))

k = 1

results = AutoVivification()
final_results = AutoVivification()
for train_index, test_index in CV.split(X, y):

    # extract training and test set for current CV fold
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]

    inner_scores = []
    # split the training data

    CV_nest = model_selection.KFold(internal_cross_validation, shuffle=True)
    M_nest = X.shape[1]

    w = np.empty((M, internal_cross_validation, len(lambdas)))
    f = 0
    y = y.squeeze()
    opt_lambda = 0

    results = {"linear": np.array([])}
    linera_test_error = np.empty((K, len(lambdas)))
    ann_test_error = np.empty((K, len(hidden_layers)))
    for train_nest_index, test_nest_index in CV_nest.split(X_test, y_test):

        X_nest_train, X_nest_test = (
            X_test[train_nest_index],
            X_test[test_nest_index],
        )
        y_nest_train, y_nest_test = (
            y_test[train_nest_index],
            y_test[test_nest_index],
        )

        train_error = np.empty(len(lambdas))
        test_error = np.empty(len(lambdas))
        # Standardize the training and set set based on training set moments
        mu = np.mean(X_nest_train[:, 1:], 0)
        sigma = np.std(X_nest_train[:, 1:], 0)

        # precompute terms
        Xty = X_nest_train.T @ y_nest_train
        XtX = X_nest_train.T @ X_nest_train

        for l in range(0, len(lambdas)):
            # Compute parameters for current value of lambda and current CV fold
            # note: "linalg.lstsq(a,b)" is substitue for Matlab's left division operator "\"
            lambdaI = lambdas[l] * np.eye(M)
            lambdaI[0, 0] = 0  # remove bias regularization
            w[:, f, l] = np.linalg.solve(XtX + lambdaI, Xty).squeeze()
            # Evaluate training and test performance
            linera_test_error[f, l] = np.power(
                y_nest_test - X_nest_test @ w[:, f, l].T, 2
            ).mean()

        ann_err = []
        for hidden in hidden_layers:
            # Define the model
            model = lambda: torch.nn.Sequential(
                torch.nn.Linear(M_nest, hidden),  # M features to n_hidden_units
                torch.nn.Tanh(),  # 1st transfer function,
                torch.nn.Linear(hidden, 1),  # n_hidden_units to 1 output neuron
                # no final tranfer function, i.e. "linear output"
                torch.nn.Sigmoid(),
            )
            loss_fn = (
                torch.nn.MSELoss()
            )  # notice how this is now a mean-squared-error loss

            # print("Training model of type:\n\n{}\n".format(str(model())))
            errors = (
                []
            )  # make a list for storing generalizaition error in each loop

            # Extract training and test set for current CV fold, convert to tensors
            X_train_nest_ann = torch.tensor(X_nest_train, dtype=torch.float)
            y_train_nest_ann = torch.tensor(y_nest_train, dtype=torch.float)
            X_test_nest_ann = torch.tensor(X_nest_test, dtype=torch.float)
            y_test_nest_ann = torch.tensor(y_nest_test, dtype=torch.float)

            # Train the net on training data
            net, final_loss, learning_curve = train_neural_net(
                model,
                loss_fn,
                X=X_train_nest_ann,
                y=y_train_nest_ann,
                n_replicates=n_replicates,
                max_iter=max_iter,
            )

            # Determine estimated class labels for test set
            y_test_est = net(X_test_nest_ann)

            # Determine errors and errors
            se = (
                y_test_est.float() - y_test_nest_ann.float()
            ) ** 2  # squared error
            mse = np.sum(se.tolist()) / len(y_test_nest_ann)  # mean
            ann_test_error[
                f, hidden - 1
            ] = mse  # store error rate for current CV fold

        f += 1

    opt_val_err = np.min(np.mean(test_error))
    opt_lambda = lambdas[np.argmin(np.mean(linera_test_error, axis=0))]

    opt_hidden = hidden_layers[np.argmin(np.mean(ann_err, axis=0))]

    Xty = X_train.T @ y_train
    XtX = X_train.T @ X_train

    # Estimate weights for the optimal value of lambda, on entire training set
    lambdaI = opt_lambda * np.eye(M)
    lambdaI[0, 0] = 0  # Do no regularize the bias term
    w_rlr[:, k - 1] = np.linalg.solve(XtX + lambdaI, Xty).squeeze()

    m = lm.LinearRegression().fit(X_train, y_train)

    model = lambda: torch.nn.Sequential(
        torch.nn.Linear(M, opt_hidden),  # M features to n_hidden_units
        torch.nn.Tanh(),  # 1st transfer function,
        torch.nn.Linear(opt_hidden, 1),  # n_hidden_units to 1 output neuron
        # no final tranfer function, i.e. "linear output"
        torch.nn.Sigmoid(),
    )
    loss_fn = (
        torch.nn.MSELoss()
    )  # notice how this is now a mean-squared-error loss

    # print("Training model of type:\n\n{}\n".format(str(model())))
    errors = []  # make a list for storing generalizaition error in each loop

    # Extract training and test set for current CV fold, convert to tensors
    X_train_ann = torch.tensor(X_train, dtype=torch.float)
    y_train_ann = torch.tensor(y_train, dtype=torch.float)
    X_test_ann = torch.tensor(X_test, dtype=torch.float)
    y_test_ann = torch.tensor(y_test, dtype=torch.float)

    # Train the net on training data
    net, final_loss, learning_curve = train_neural_net(
        model,
        loss_fn,
        X=X_train_ann,
        y=y_train_ann,
        n_replicates=n_replicates,
        max_iter=max_iter,
    )

    # Determine estimated class labels for test set
    y_test_est = net(X_test_ann)

    # Determine errors and errors
    se = (y_test_est.float() - y_test_ann.float()) ** 2  # squared error
    mse = np.sum(se.tolist()) / len(y_test_ann)

    final_results[k - 1] = {
        "baseline": float(
            np.square(y_test - m.predict(X_test)).sum() / y_test.shape[0]
        ),
        "ann": {"layer": float(opt_hidden), "err": float(mse)},
        "linear": {
            "lambda": opt_lambda,
            "opt_val_err": opt_val_err,
            "err": float(
                np.sum(
                    np.square(y_test - X_test @ w_rlr[:, k - 1]).sum(axis=0)
                    / y_test.shape[0]
                )
            ),
        },
    }

    k += 1

import json

with open(path.join(script_dir, "reg_b.json"), "w") as file:
    file.write(json.dumps(final_results))
