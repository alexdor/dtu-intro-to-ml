import numpy as np 
import pandas as pd 
import io
import pandas as pd
from matplotlib import pyplot as plt
from matplotlib.pylab import (figure, semilogx, loglog, xlabel, ylabel, legend, 
                           title, subplot, show, grid)
from scipy.io import loadmat
import sklearn.linear_model as lm
from tools import rlr_validate
from sklearn.preprocessing import normalize
from sklearn.model_selection import KFold,cross_val_score,train_test_split
from sklearn.linear_model import LogisticRegression
from tools import rocplot, confmatplot
df = pd.read_csv('heart.csv')

y=pd.DataFrame(df["target"]).to_numpy()

norm_vars=df[["trestbps","thalach","oldpeak","ca","slope","chol"]]
norm_vars=(norm_vars-norm_vars.mean())/norm_vars.std()

discrete=["sex",'cp','fbs','restecg','exang','thal']
for col in discrete:
    temp=pd.get_dummies(df[col],prefix=col)
    norm_vars=pd.concat([norm_vars,temp],axis=1)


d = pd.DataFrame(np.ones((norm_vars.shape[0], 1)))
X=pd.concat([d,norm_vars],axis=1)
# Add offset attribute

## Crossvalidation
# Create crossvalidation partition for evaluation
K=10
#added some parameters
kf = KFold(n_splits = K, shuffle = True, random_state = 2)
partition=kf.split(X,y)

lambda_interval = np.power(10.,range(-5,9))
names=np.asarray(X.columns.to_list())
names[0]='age'

X=X.to_numpy()
# Initialize variables
train_error_rate = np.zeros(len(lambda_interval))
test_error_rate = np.zeros(len(lambda_interval))
coefficient_norm = np.zeros(len(lambda_interval))

#split data into train and test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.1, stratify=y)

#baseline calculations
baseline_model=LogisticRegression(penalty='l2' )
baseline_model.fit(X_train,y_train)
base_y_train_est = baseline_model.predict(X_train).T
base_y_train_est=np.asarray(base_y_train_est).reshape(len(base_y_train_est),1)

base_y_test_est = baseline_model.predict(X_test)
base_y_test_est=np.asarray(base_y_test_est).reshape(len(base_y_test_est),1)

base_train_error_rate = np.sum(base_y_train_est != y_train) / len(y_train)
base_test_error_rate = np.sum(base_y_test_est != y_test) / len(y_test)


#for each lambda, train the model and track the train/test error rates

for k in range(0, len(lambda_interval)):
    mdl = LogisticRegression(penalty='l2', C=lambda_interval[k] )
    
    mdl.fit(X_train, y_train)

    y_train_est = mdl.predict(X_train).T
    y_train_est=np.asarray(y_train_est).reshape(len(y_train_est),1)

    y_test_est = mdl.predict(X_test)
    y_test_est=np.asarray(y_test_est).reshape(len(y_test_est),1)
    
    train_error_rate[k] = np.sum(y_train_est != y_train) / len(y_train)
    test_error_rate[k] = np.sum(y_test_est != y_test) / len(y_test)

    w_est = mdl.coef_[0] 
    coefficient_norm[k] = np.sqrt(np.sum(w_est**2))


#find best lambda
min_error = np.min(test_error_rate)
opt_lambda_idx = np.where(test_error_rate==min(test_error_rate))
train_test_min_indexes=train_error_rate[opt_lambda_idx]
min_both_error=np.min(train_test_min_indexes)
min_both_err_index=np.where(train_error_rate==min_both_error)
opt_lambda = lambda_interval[min_both_err_index][0]


#logistic model with best lambda
mdl = LogisticRegression(penalty='l2', C=opt_lambda )
    
mdl.fit(X_train, y_train)

y_train_est = mdl.predict(X_train).T
y_train_est=np.asarray(y_train_est).reshape(len(y_train_est),1)

y_test_est = mdl.predict(X_test)
y_test_est=np.asarray(y_test_est).reshape(len(y_test_est),1)


best_train_error_rate = (y_train_est != y_train).sum() / len(y_train)
best_test_error_rate = (y_test_est != y_test).sum() / len(y_test)



plt.figure(figsize=(8,8))
plt.plot(lambda_interval, train_error_rate,'b')
plt.plot(lambda_interval, test_error_rate,'r')
plt.plot([opt_lambda], [min_error], 'go')
#plt.yscale('log')
plt.xscale('log')
# plt.semilogx(lambda_interval, train_error_rate*100)
# plt.semilogx(lambda_interval, test_error_rate*100)
# plt.semilogx(opt_lambda, min_error*100, 'o')
plt.text(1e-8, 3, "Minimum test error: " + str(np.round(min_error*100,2)) + ' % at 1e' + str(np.round(np.log10(opt_lambda),2)))
plt.xlabel('Regularization strength, $\log_{10}(\lambda)$')
plt.ylabel('Error rate (%)')
plt.title('Classification error')
plt.legend(['Training error','Test error','Test minimum'],loc='upper right')
#plt.ylim([0,1])
plt.grid()
plt.show()    


plt.figure(figsize=(8,8))
plt.semilogx(lambda_interval, coefficient_norm,'k')
plt.ylabel('L2 Norm')
plt.xlabel('Regularization strength, $\log_{10}(\lambda)$')
plt.title('Parameter vector L2 norm')
plt.grid()
plt.show()    

