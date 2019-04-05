import io

import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt

df = pd.read_csv("heart.csv", sep=",", header=0)

print("done")


df.columns
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
