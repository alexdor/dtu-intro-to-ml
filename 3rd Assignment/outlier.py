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
from tmp_tools import show_outlier

from os import path


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


### Gausian Kernel density estimator
# cross-validate kernel width by leave-one-out-cross-validation
# (efficient implementation in gausKernelDensity function)
# evaluate for range of kernel widths
widths = X.var(axis=0).max() * (2.0 ** np.arange(-10, 3))
logP = np.zeros(np.size(widths))
for i, w in enumerate(widths):
    print("Fold {:2d}, w={:f}".format(i, w))
    density, log_density = gausKernelDensity(X, w)
    logP[i] = log_density.sum()

val = logP.max()
ind = logP.argmax()

width = widths[ind]
print("Optimal estimated width is: {0}".format(width))

# evaluate density for estimated width
density, log_density = gausKernelDensity(X, width)

# Sort the densities
i = (density.argsort(axis=0)).ravel()
density = density[i].reshape(-1)

# Plot density estimate of outlier score
figure(1)
bar(range(20), density[:20])
title("Density estimate")

# Show possible outlierS
show_outlier(X, i)

print("-----------------------------------------------")

### K-neighbors density estimator
# Neighbor to use:
K = 5

# Find the k nearest neighbors
knn = NearestNeighbors(n_neighbors=K).fit(X)
D, i = knn.kneighbors(X)

density = 1.0 / (D.sum(axis=1) / K)

# Sort the scores
i = density.argsort()
density = density[i]

# Plot k-neighbor estimate of outlier score (distances)
figure(3)
bar(range(20), density[:20])
title("KNN density: Outlier score")
# Plot possible outliers
print("KNN density: Possible outliers")

show_outlier(X, i)

print("-----------------------------------------------")


### K-nearest neigbor average relative density
# Compute the average relative density

knn = NearestNeighbors(n_neighbors=K).fit(X)
D, i = knn.kneighbors(X)
density = 1.0 / (D.sum(axis=1) / K)
avg_rel_density = density / (density[i[:, 1:]].sum(axis=1) / K)

# Sort the avg.rel.densities
i_avg_rel = avg_rel_density.argsort()
avg_rel_density = avg_rel_density[i_avg_rel]

# Plot k-neighbor estimate of outlier score (distances)
figure(5)
bar(range(20), avg_rel_density[:20])
title("KNN average relative density: Outlier score")
# Plot possible outliers
print("KNN average relative density")
show_outlier(X, i_avg_rel)

print("-----------------------------------------------")

### Distance to 5'th nearest neighbor outlier score
K = 5

# Find the k nearest neighbors
knn = NearestNeighbors(n_neighbors=K).fit(X)
D, i = knn.kneighbors(X)

# Outlier score
score = D[:, K - 1]
# Sort the scores
i = score.argsort()
score = score[i[::-1]]

# Plot k-neighbor estimate of outlier score (distances)
# figure(7)
# bar(range(20), score[:20])
# title("5th neighbor distance: Outlier score")
# # Plot possible outliers
# print("5th neighbor distance")
# show_outlier(X, i)
# print("-----------------------------------------------")

# Plot random observations (the first 20 in the data set), for comparison
print("Random obs")
for k in range(1, 21):
    print(X[[k]])
print("-----------------------------------------------")
print()
show()
