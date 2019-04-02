import numpy as np 
import pandas as pd 
import io
import pandas as pd
from matplotlib import pyplot as plt
df=pd.read_csv('heart.csv', sep=',',header=0)

print(df.columns)
y=df["chol"]
continuous_vars=df[["age","trestbps","thalach","oldpeak","ca","slope"]]
continuous_norm=(continuous_vars - continuous_vars.mean())/(continuous_vars.std())
discrete=["sex",'cp','fbs','restecg','exang','thal']
one_of_k_discrete_vars=pd.DataFrame()

for col in discrete:
    dummy=pd.get_dummies(df[col])
    dummy = dummy.add_suffix(col)
    one_of_k_discrete_vars=pd.concat([one_of_k_discrete_vars,dummy])

all_norm_features=pd.concat([one_of_k_discrete_vars,continuous_norm])
print("shit")


