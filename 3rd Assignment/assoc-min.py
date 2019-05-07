# exercise 11.4.1
import numpy as np
from matplotlib.pyplot import (
    figure,
    imshow,
    bar,
    title,
    xticks,
    yticks,
    cm,
    subplot,
    show,
)
import pandas as pd
from scipy.io import loadmat
from toolbox_02450 import gausKernelDensity
from sklearn.neighbors import NearestNeighbors

from os import path
from apyori import apriori

script_dir = path.dirname(__file__)  # <-- absolute dir the script is in
rel_path = "../heart.csv"
data_file = path.join(script_dir, rel_path)

df = pd.read_csv(data_file)

target = "chol"
var = "trestbps"

y = np.array([df[target]])
y = (y - y.mean()) / y.std()
X = df.loc[:, df.columns != target]
# X = np.array([df[]])

norm_vars = df[["age", "trestbps", "thalach", "oldpeak", "ca", "slope"]]
norm_vars = (norm_vars - norm_vars.mean()) / norm_vars.std()
new = pd.DataFrame()

# discrete = ["sex", "cp", "fbs", "restecg", "exang", "thal", "target"]
# for col in discrete:
#     temp = pd.get_dummies(df[col], prefix=col)
#     new = pd.concat([new, temp], axis=1)

# d = pd.DataFrame(np.ones((new.shape[0], 1)))
# X = pd.concat([d, new], axis=1)
attribute_names = list(X)
X = np.array(X)

N, M = np.shape(X)


def df_to_transactions(X, labels=[]):
    T = []
    for i in range(X.shape[0]):
        l = np.nonzero(X[i, :])[0].tolist()
        if labels:
            l = [labels[i] for i in l]
        T.append(l)
    return T


# We will now transform the original dataset into a binary format. Notice the changed attribute names:
from similarity import binarize2

Xbin, attributeNamesBin = binarize2(X, attribute_names)
print("X, i.e. the original dataset, has now been transformed into:")
print(Xbin)
print(attributeNamesBin)


T = df_to_transactions(Xbin, labels=attributeNamesBin)
rules = apriori(T, min_support=0.75, min_confidence=0.9)
# This function print the found rules and also returns a list of rules in the format:
# [(x,y), ...]
# where x -> y
def print_apriori_rules(rules):
    frules = []
    with open("f", "a+") as f:
        for r in rules:
            for o in r.ordered_statistics:
                conf = o.confidence
                supp = r.support
                x = ", ".join(list(o.items_base))
                y = ", ".join(list(o.items_add))
                print(
                    "{%s} -> {%s}  (supp: %.3f, conf: %.3f)"
                    % (x, y, supp, conf)
                )
                f.write(
                    "{%s} -> {%s}  (supp: %.3f, conf: %.3f)\n"
                    % (x, y, supp, conf)
                )
                frules.append((x, y))
    return frules


print_apriori_rules(rules)
