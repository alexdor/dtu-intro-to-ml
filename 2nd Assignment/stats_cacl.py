from matplotlib.pyplot import figure, boxplot, xlabel, ylabel, show
import numpy as np
from scipy.io import loadmat
import sklearn.linear_model as lm
from sklearn import model_selection, tree
from scipy import stats

K = 10

ann = np.array(
    [
        0.13639037137299082,
        0.13233222476086697,
        0.115019788440846,
        0.13244031930808,
        0.14862508250711,
        0.1436970368959001,
        0.1300457999396699,
        0.1731616296776399,
        0.23721282529610999,
        0.011312308845422003,
    ]
)
baseline = np.array(
    [
        0.12334662761935544,
        0.26305555055270184,
        0.38284438972733814,
        0.3010273103355740165,
        0.2154357044204659,
        0.2146704360656115,
        0.24566613038948415,
        0.10699845643663863,
        0.23696009127551576,
        0.2055035867807385,
    ]
)
linear = np.array(
    [
        0.25320323509992404,
        0.07271313851102268,
        0.19052318427388455,
        0.1454824754938552,
        0.23376257242237617,
        0.26408103693558524,
        0.2772270536048823,
        0.17762882093958501,
        0.18027375589014139,
        0.2540316339105092,
    ]
)


def get_stats(first, second, name):
    z = first - second
    zb = z.mean()
    nu = K - 1
    sig = (z - zb).std() / np.sqrt(K - 1)
    alpha = 0.05

    zL = zb + sig * stats.t.ppf(alpha / 2, nu)
    zH = zb + sig * stats.t.ppf(1 - alpha / 2, nu)

    print(f"\n Stats for {name}")

    if zL <= 0 and zH >= 0:
        print("Classifiers are not significantly different")
    else:
        print("Classifiers are significantly different.")
    print(f"zL: {zL}")
    print(f"zH: {zH}")


get_stats(ann, baseline, "ANN / Baseline")

get_stats(linear, baseline, "Linear / Baseline")

get_stats(ann, linear, "ANN / Linear")
