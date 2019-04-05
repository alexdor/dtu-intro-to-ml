import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.pyplot import (
    boxplot,
    figure,
    hist,
    show,
    subplot,
    title,
    xlabel,
    xticks,
    ylim,
    yticks,
)
from sklearn.decomposition import PCA
from sklearn.model_selection import GridSearchCV, cross_val_score, train_test_split

import io
import seaborn as sns

data = pd.read_csv("../input/" + os.listdir("../input")[0])
df = data.copy()


sumStats = []

for col in df.columns:
    appe = str(col) + "\n" + str(df[col].describe().transpose())
    sumStats.append(appe.split("\n"))


summary = df.describe()
summary = summary.transpose()
summary.head()

fig, ax = plt.subplots(figsize=(10, 8), nrows=4, ncols=4)
count = 0
for i in range(4):
    for j in range(4):
        try:
            ax[i][j].hist(df[str(df.columns[count])])
            ax[i][j].set_title(str(df.columns[count]))
        except:
            pass
        count += 1
fig.delaxes(ax[3][2])
fig.delaxes(ax[3][3])
plt.tight_layout()
fig.savefig("feature_hists.png")
plt.show()
print("dg")


dataCorrMat = df.corr()
plt.figure(figsize=(14, 8))
sns.heatmap(dataCorrMat, annot=True, cmap="coolwarm", linewidths=0.1)
plt.savefig("correlation_matrix.png")
plt.show()


fig, axes = plt.subplots(1, 1, figsize=(14, 10))
new_data = data[["age", "trestbps", "chol", "fbs", "thalach", "oldpeak", "ca", "thal"]]
axes.get_yaxis().set_visible(False)
new_data.boxplot(grid=False)
show()


# import numpy as np
# from scipy.stats import norm
# # import matplotlib.pyplot as plt

# fig,axes =plt.subplots(14,1, figsize=(12, 9))
# for v, i in enumerate(data):

#     # Fit a normal distribution to the data:
#     mu, std = norm.fit(data[i])


#     plt.hist(data[i], bins=25, density=True, alpha=0.6, color='b', ax=axes[v])
#     # Plot the PDF.
#     xmin, xmax = plt.xlim()
#     x = np.linspace(xmin, xmax, 100)
#     p = norm.pdf(x, mu, std)
#     plt.plot(x, p, 'k', linewidth=2)
#     title = "Fit results: mu = %.2f,  std = %.2f" % (mu, std)
#     plt.title(title)

# plt.tight_layout()
# show()


X = data.drop(["target"], axis=1)
X.corrwith(data["target"]).plot.bar(figsize=(12, 8), fontsize=20, rot=90, grid=False)
plt.gca().yaxis.grid(True)


#%%
dataX = X
dataY = data["target"]

# Normalize

X = (dataX - np.min(dataX) / (np.max(dataX) - np.min(dataX))).values


pca = PCA().fit(X)


cumulative = np.cumsum(pca.explained_variance_ratio_)
plt.figure(figsize=(12, 8))
plt.step([i for i in range(len(cumulative))], cumulative)
plt.show()


pca2 = PCA(n_components=2)
pca2.fit(X)
X1 = pca2.fit_transform(X)


with plt.style.context("seaborn-whitegrid"):
    plt.figure(figsize=(12, 8))
    for lab, col in zip((0, 1), ("blue", "red")):
        plt.scatter(X1[Y == lab, 0], X1[Y == lab, 1], label=lab, c=col)
    plt.xlabel("Principal Component 1")
    plt.ylabel("Principal Component 2")
    plt.legend(loc="lower center")
    plt.tight_layout()
    plt.show()


#%%
def pca_results(good_data, pca):
    """
	Create a DataFrame of the PCA results
	Includes dimension feature weights and explained variance
	Visualizes the PCA results
	"""

    # Dimension indexing
    dimensions = dimensions = [
        "Dimension {}".format(i) for i in range(1, len(pca.components_) + 1)
    ]

    # PCA components
    components = pd.DataFrame(
        np.round(pca.components_, 4), columns=list(good_data.keys())
    )
    components.index = dimensions

    # PCA explained variance
    ratios = pca.explained_variance_ratio_.reshape(len(pca.components_), 1)
    variance_ratios = pd.DataFrame(np.round(ratios, 4), columns=["Explained Variance"])
    variance_ratios.index = dimensions

    # Create a bar plot visualization
    fig, ax = plt.subplots(figsize=(14, 8))

    # Plot the feature weights as a function of the components
    components.plot(ax=ax, kind="bar")
    ax.set_ylabel("Feature Weights")
    ax.set_xticklabels(dimensions, rotation=0)

    # Display the explained variance ratios
    for i, ev in enumerate(pca.explained_variance_ratio_):
        ax.text(
            i - 0.40,
            ax.get_ylim()[1] + 0.05,
            "                       Explained Variance\n                               %.4f"
            % (ev),
        )

        # Return a concatenated DataFrame
    return pd.concat([variance_ratios, components], axis=1)


pca_results = pca_results(dataX, pca2)
