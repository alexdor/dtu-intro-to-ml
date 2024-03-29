import json
from os import path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from scipy import stats
from scipy.io import loadmat
from sklearn import model_selection

from tools import draw_neural_net, train_neural_net

script_dir = path.dirname(__file__)  # <-- absolute dir the script is in
rel_path = "./heart.csv"
data_file = path.join(script_dir, rel_path)

plot = False

df = pd.read_csv(data_file)
target = "chol"
y = np.array([df[target]])
y = (y - y.mean()) / y.std()
y = y.transpose()
X = df.loc[:, df.columns != target]


results = {}

for do_pca_preprocessing in [True, False]:
    results[f"pca: {do_pca_preprocessing}"] = {}
    for K in [5, 10]:
        results[f"pca: {do_pca_preprocessing}"][f"k: {K}"] = {}
        results[f"pca: {do_pca_preprocessing}"][f"k: {K}"]["losses"] = {}
        # Normalize data
        # do_pca_preprocessing = False
        if do_pca_preprocessing:
            Y = stats.zscore(X, 0)
            U, S, V = np.linalg.svd(Y, full_matrices=False)
            V = V.T
            # Components to be included as features
            k_pca = 2
            X = X @ V[:, 0:k_pca]
            N, M = X.shape

        else:
            norm_vars = df[
                ["age", "trestbps", "thalach", "oldpeak", "ca", "slope"]
            ]
            norm_vars = (norm_vars - norm_vars.mean()) / norm_vars.std()
            new = pd.DataFrame()

            discrete = [
                "sex",
                "cp",
                "fbs",
                "restecg",
                "exang",
                "thal",
                "target",
            ]
            for col in discrete:
                temp = pd.get_dummies(df[col], prefix=col)
                new = pd.concat([new, temp], axis=1)

            d = pd.DataFrame(np.ones((new.shape[0], 1)))
            X = pd.concat([d, new], axis=1)
        attribute_names = list(X)
        X = np.array(X)

        N, M = X.shape
        C = 2

        # Parameters for neural network classifier
        n_hidden_units = 1  # number of hidden units
        n_replicates = 15  # number of networks trained in each k-fold
        max_iter = 10000  #

        # K-fold crossvalidation
        # K = 5  # only three folds to speed up this example
        CV = model_selection.KFold(K, shuffle=True)

        # Setup figure for display of learning curves and error rates in fold
        summaries, summaries_axes = plt.subplots(1, 2, figsize=(10, 5))

        # Define the model
        model = lambda: torch.nn.Sequential(
            torch.nn.Linear(M, n_hidden_units),  # M features to n_hidden_units
            torch.nn.Tanh(),  # 1st transfer function,
            torch.nn.Linear(
                n_hidden_units, 1
            )  # n_hidden_units to 1 output neuron
            # no final tranfer function, i.e. "linear output"
        )
        loss_fn = (
            torch.nn.MSELoss()
        )  # notice how this is now a mean-squared-error loss

        # print("Training model of type:\n\n{}\n".format(str(model())))
        errors = (
            []
        )  # make a list for storing generalizaition error in each loop
        for (k, (train_index, test_index)) in enumerate(CV.split(X, y)):
            # print("\nCrossvalidation fold: {0}/{1}".format(k + 1, K))

            # Extract training and test set for current CV fold, convert to tensors
            X_train = torch.tensor(X[train_index, :], dtype=torch.float)
            y_train = torch.tensor(y[train_index], dtype=torch.float)
            X_test = torch.tensor(X[test_index, :], dtype=torch.float)
            y_test = torch.tensor(y[test_index], dtype=torch.uint8)

            # Train the net on training data
            net, final_loss, learning_curve = train_neural_net(
                model,
                loss_fn,
                X=X_train,
                y=y_train,
                n_replicates=n_replicates,
                max_iter=max_iter,
            )

            print(f"\n\tBest loss: {final_loss}\n")
            results[f"pca: {do_pca_preprocessing}"][f"k: {K}"]["losses"][
                f"fold: {k + 1}"
            ] = final_loss.tolist()

            # Determine estimated class labels for test set
            y_test_est = net(X_test)

            # Determine errors and errors
            se = (y_test_est.float() - y_test.float()) ** 2  # squared error
            mse = (sum(se).type(torch.float) / len(y_test)).data.numpy()  # mean
            errors.append(mse)  # store error rate for current CV fold

            # Make a list for storing assigned color of learning curve for up to K=10
            color_list = [
                "tab:orange",
                "tab:green",
                "tab:purple",
                "tab:brown",
                "tab:pink",
                "tab:gray",
                "tab:olive",
                "tab:cyan",
                "tab:red",
                "tab:blue",
            ]

            if plot:
                # Display the learning curve for the best net in the current fold
                h, = summaries_axes[0].plot(learning_curve, color=color_list[k])
                h.set_label("CV fold {0}".format(k + 1))
                summaries_axes[0].set_xlabel("Iterations")
                summaries_axes[0].set_xlim((0, max_iter))
                summaries_axes[0].set_ylabel("Loss")
                summaries_axes[0].set_title("Learning curves")

        if plot:
            # Display the MSE across folds
            summaries_axes[1].bar(
                np.arange(1, K + 1), np.squeeze(errors), color=color_list
            )
            summaries_axes[1].set_xlabel("Fold")
            summaries_axes[1].set_xticks(np.arange(1, K + 1))
            summaries_axes[1].set_ylabel("MSE")
            summaries_axes[1].set_title("Test mean-squared-error")

            print("Diagram of best neural net in last fold:")
            weights = [net[i].weight.data.numpy().T for i in [0, 2]]
            biases = [net[i].bias.data.numpy() for i in [0, 2]]
            tf = [str(net[i]) for i in [1, 2]]

            draw_neural_net(
                weights, biases, tf, attribute_names=attribute_names
            )

        # Print the average classification error rate
        print(
            "\nEstimated generalization error, RMSE: {0}".format(
                round(np.sqrt(np.mean(errors)), 4)
            )
        )
        results[f"pca: {do_pca_preprocessing}"][f"k: {K}"]["rmse"] = float(
            round(np.sqrt(np.mean(errors)), 4)
        )

        if plot:
            # When dealing with regression outputs, a simple way of looking at the quality
            # of predictions visually is by plotting the estimated value as a function of
            # the true/known value - these values should all be along a straight line "y=x",
            # and if the points are above the line, the model overestimates, whereas if the
            # points are below the y=x line, then the model underestimates the value
            plt.figure(figsize=(10, 10))
            y_est = y_test_est.data.numpy()
            y_true = y_test.data.numpy()
            axis_range = [
                np.min([y_est, y_true]) - 1,
                np.max([y_est, y_true]) + 1,
            ]
            plt.plot(axis_range, axis_range, "k--")
            plt.plot(y_true, y_est, "ob", alpha=0.25)
            plt.legend(["Perfect estimation", "Model estimations"])
            plt.title("Chol: estimated versus true value (for last CV-fold)")
            plt.ylim(axis_range)
            plt.xlim(axis_range)
            plt.xlabel("True value")
            plt.ylabel("Estimated value")
            plt.grid()

            plt.show()

with open(path.join(script_dir, "results.json"), "w") as file:
    file.write(json.dumps(results))
