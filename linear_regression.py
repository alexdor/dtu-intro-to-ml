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

df = pd.read_csv('heart.csv')

y=pd.DataFrame(df["chol"]).to_numpy()

norm_vars=df[["trestbps","thalach","oldpeak","ca","slope"]]
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

lambdas = np.power(10.,range(-5,9))
names=X.columns.to_list()
names[0]='age'

X=X.to_numpy()
# Initialize variables
N=X.shape[0]
M=X.shape[1]

Error_train = np.empty((K,1))
Error_test = np.empty((K,1))
Error_train_rlr = np.empty((K,1))
Error_test_rlr = np.empty((K,1))
Error_train_nofeatures = np.empty((K,1))
Error_test_nofeatures = np.empty((K,1))

w_rlr = np.empty((M,K))
mu = np.empty((K, M-1))
sigma = np.empty((K, M-1))
w_noreg = np.empty((M,K))

k=0
for train_index, test_index in partition:
    X_train = X[train_index]
    Y_train = y[train_index]
    X_test = X[test_index]
    Y_test = y[test_index]

    # extract training and test set for current CV fold
    internal_cross_validation = 10    
    
    opt_val_err, opt_lambda, mean_w_vs_lambda, train_err_vs_lambda, test_err_vs_lambda = rlr_validate(X_train, Y_train, lambdas, internal_cross_validation)

    # Standardize outer fold based on training set, and save the mean and standard
    # deviations since they're part of the model (they would be needed for
    # making new predictions) - for brevity we won't always store these in the scripts
    
    
    Xty = X_train.T @ Y_train
    XtX = X_train.T @ X_train
    
    # Compute mean squared error without using the input data at all
    Error_train_nofeatures[k] = np.square(Y_train-Y_train.mean()).sum(axis=0)/Y_train.shape[0]
    Error_test_nofeatures[k] = np.square(Y_test-Y_test.mean()).sum(axis=0)/Y_test.shape[0]

    # Estimate weights for the optimal value of lambda, on entire training set
    lambdaI = opt_lambda * np.eye(M)
    lambdaI[0,0] = 0 # Do no regularize the bias term
    w_rlr[:,k] = np.linalg.solve(XtX+lambdaI,Xty).squeeze()
    # Compute mean squared error with regularization with optimal lambda
    temp=X_train @ w_rlr[:,k]
    temp=temp.reshape((temp.shape[0],1))
    Error_train_rlr[k] = np.square(Y_train-temp).sum(axis=0)/Y_train.shape[0]
    temp=X_test @ w_rlr[:,k]
    temp=temp.reshape((temp.shape[0],1))
    Error_test_rlr[k] = np.square(Y_test-temp).sum(axis=0)/Y_test.shape[0]

    # Estimate weights for unregularized linear regression, on entire training set
   # w_noreg[:,k] = np.linalg.solve(XtX,Xty).squeeze()
    # Compute mean squared error without regularization
    
    m = lm.LinearRegression().fit(X_train, Y_train)
    Error_train[k] = np.square(Y_train-m.predict(X_train)).sum()/Y_train.shape[0]
    Error_test[k] = np.square(Y_test-m.predict(X_test)).sum()/Y_test.shape[0]

    # Display the results for the last cross-validation fold
    if k == K-1:
        figure(k, figsize=(12,8))
        subplot(1,2,1)
        semilogx(lambdas,mean_w_vs_lambda.T[:,1:],'.-') # Don't plot the bias term
        xlabel('Regularization factor')
        ylabel('Mean Coefficient Values')
        grid()
        # You can choose to display the legend, but it's omitted for a cleaner 
        # plot, since there are many attributes
        #legend(attributeNames[1:], loc='best')
        
        subplot(1,2,2)
        title('Optimal lambda: 1e{0}'.format(np.log10(opt_lambda)))
        loglog(lambdas,train_err_vs_lambda.T,'b.-',lambdas,test_err_vs_lambda.T,'r.-')
        xlabel('Regularization factor')
        ylabel('Squared error (crossvalidation)')
        legend(['Train error','Validation error'])
        grid()
    
    # To inspect the used indices, use these print statements
    #print('Cross validation fold {0}/{1}:'.format(k+1,K))
    #print('Train indices: {0}'.format(train_index))
    #print('Test indices: {0}\n'.format(test_index))

    k+=1

show()
# Display results
print('Linear regression without feature selection:')
print('- Training error: {0}'.format(Error_train.mean()))
print('- Test error:     {0}'.format(Error_test.mean()))
print('- R^2 train:     {0}'.format((Error_train_nofeatures.sum()-Error_train.sum())/Error_train_nofeatures.sum()))
print('- R^2 test:     {0}\n'.format((Error_test_nofeatures.sum()-Error_test.sum())/Error_test_nofeatures.sum()))
print('Regularized linear regression:')
print('- Training error: {0}'.format(Error_train_rlr.mean()))
print('- Test error:     {0}'.format(Error_test_rlr.mean()))
print('- R^2 train:     {0}'.format((Error_train_nofeatures.sum()-Error_train_rlr.sum())/Error_train_nofeatures.sum()))
print('- R^2 test:     {0}\n'.format((Error_test_nofeatures.sum()-Error_test_rlr.sum())/Error_test_nofeatures.sum()))

print('Weights in last fold:')
for m in range(M):
    print('{:>15} {:>15}'.format(names[m], np.round(w_rlr[m,-1],2)))

print('Ran Exercise 8.1.1')