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
from tools import rocplot, confmatplot,draw_neural_net, train_neural_net

import torch


df = pd.read_csv('heart.csv')

y=pd.DataFrame(df["target"]).to_numpy()

norm_vars=df[["age","trestbps","thalach","oldpeak","ca","slope","chol"]]
norm_vars=(norm_vars-norm_vars.mean())/norm_vars.std()

discrete=["sex",'cp','fbs','restecg','exang','thal']
for col in discrete:
    temp=pd.get_dummies(df[col],prefix=col)
    norm_vars=pd.concat([norm_vars,temp],axis=1)


d = pd.DataFrame(np.ones((norm_vars.shape[0], 1)))
X=pd.concat([d,norm_vars],axis=1)
# Add offset attribute
N, M = X.shape

## Crossvalidation
# Create crossvalidation partition for evaluation
K=5
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

test_error_rate_by_fold=[]
train_error_rate_by_fold=[]


base_train_error_rate = []
base_test_error_rate = []

n_replicates = 5  # number of networks trained in each k-fold
max_iter = 10000  #
n_hidden_units=range(1,4)

ann_errors = [] 
binary_min_errors=[]
part_count=1
for train_index, test_index in partition:
    
    print("On k-fold number: ",part_count)
    X_train = X[train_index]
    Y_train = y[train_index]
    X_test = X[test_index]
    Y_test=y[test_index]
#split data into train and test

#baseline calculations
    kf2 = KFold(n_splits = K, shuffle = True, random_state = 2)
    partition2=kf2.split(X_train,Y_train)

    (values,counts) = np.unique(Y_test,return_counts=True)
    ind=np.argmax(counts)
    base_test_predictions=np.ones((len(Y_test),1))*values[ind]

    (values,counts) = np.unique(Y_train,return_counts=True)
    ind=np.argmax(counts)
    base_train_predictions=np.ones((len(Y_train),1))*values[ind] 
    


    base_train_error_rate.append( np.sum(base_train_predictions != Y_train) / len(Y_train))
    base_test_error_rate.append([np.sum(base_test_predictions != Y_test) / len(Y_test),values[ind],counts[ind]])


    inner_test_err=[]
    inner_test_lambda=[]
    

    inner_ann_err=np.zeros((5,len(n_hidden_units)+1))
    count=0
    binary_error=np.zeros((5,len(n_hidden_units)+1))
    for  train_index,test_index in partition2:
        X_train2 = X[train_index]
        Y_train2 = y[train_index]
        X_test2 = X[test_index]
        Y_test2=y[test_index]
        temp_test_error=[]
        for k in range(0, len(lambda_interval)):
            mdl = LogisticRegression(penalty='l2', C=lambda_interval[k] )
            
            mdl.fit(X_train2, Y_train2)

            Y_train_est2 = mdl.predict(X_train2).T
            Y_train_est2=np.asarray(Y_train_est2).reshape(len(Y_train_est2),1)

            y_test_est2 = mdl.predict(X_test2)
            y_test_est2=np.asarray(y_test_est2).reshape(len(y_test_est2),1)
            
            temp_test_error.append(np.sum(y_test_est2 != Y_test2) / len(Y_test2))

           



        #find best lambda
        min_error = np.min(temp_test_error)
        
        opt_lambda_idx = np.where(temp_test_error==min(temp_test_error))
        train_test_min_indexes=temp_test_error[opt_lambda_idx[0][0]]
        min_both_error=np.min(train_test_min_indexes)
        min_both_err_index=np.where(temp_test_error==min_both_error)
        opt_lambda = lambda_interval[min_both_err_index][0]
        inner_test_lambda.append(opt_lambda)
        inner_test_err.append(min_error)
# #ANN stuff
       
        
        for i in n_hidden_units:
            print("Hidden unit: ",i)
            model = lambda: torch.nn.Sequential(
                                torch.nn.Linear(M, i), #M features to H hiden units
                                torch.nn.Tanh(),   # 1st transfer function,
                                torch.nn.Linear(i, 1), # H hidden units to 1 output neuron
                                torch.nn.Sigmoid() # final tranfer function
                                )
            loss_fn = torch.nn.MSELoss() 

            X_train2 = torch.tensor(X_train2, dtype=torch.float)
            y_train2 = torch.tensor(Y_train2, dtype=torch.float)
            X_test2 = torch.tensor(X_test2, dtype=torch.float)
            y_test2 = torch.tensor(Y_test2, dtype=torch.uint8)
            # Train the net on training data
            net, final_loss, learning_curve = train_neural_net(
                model,
                loss_fn,
                X=X_train2,
                y=y_train2,
                n_replicates=n_replicates,
                max_iter=max_iter,
            )
            print("\n\tBest loss: {}\n".format(final_loss))
            # Determine estimated class labels for test set
            y_test_est = net(X_test2)

            binary_pred=[]
            actual_y_test=[]
            for j in range(len(y_test_est)):
                binary_pred.append(round(y_test_est[j].tolist()[0]))
                actual_y_test.append(y_test2[j].tolist()[0])
            err=np.square(np.asarray(binary_pred)-np.asarray(actual_y_test))
            binary_error[count,i]=np.sum(err)/len(err)
         
        
    mean_hidden_test_error=np.mean(binary_error,axis=1)
    best_ann_hidden_cell_number=np.argmin(mean_hidden_test_error)

 

    model = lambda: torch.nn.Sequential(
                    torch.nn.Linear(M, best_ann_hidden_cell_number), #M features to H hiden units
                    torch.nn.Tanh(),   # 1st transfer function,
                    torch.nn.Linear(best_ann_hidden_cell_number, 1), # H hidden units to 1 output neuron
                    torch.nn.Sigmoid() # final tranfer function
                    )
    loss_fn = torch.nn.MSELoss() 

    X_train = torch.tensor(X_train, dtype=torch.float)
    y_train = torch.tensor(Y_train, dtype=torch.float)
    X_test = torch.tensor(X_test, dtype=torch.float)
    y_test = torch.tensor(Y_test, dtype=torch.uint8)
    # Train the net on training data
    net, final_loss, learning_curve = train_neural_net(
        model,
        loss_fn,
        X=X_train,
        y=y_train,
        n_replicates=n_replicates,
        max_iter=max_iter,
    )
    print("\n\tBest loss: {}\n".format(final_loss))
    # Determine estimated class labels for test set
    y_test_est = net(X_test)

    binary_pred=[]
    actual_y_test=[]
    for j in range(len(y_test_est)):
        binary_pred.append(round(y_test_est[j].tolist()[0]))
        actual_y_test.append(y_test[j].tolist()[0])
    err=np.square(np.asarray(binary_pred)-np.asarray(actual_y_test))
    ann_errors.append([np.sum(err)/len(err),best_ann_hidden_cell_number])





    #fit logistic regression model based on best lambda from inner old
    best_from_inner_test_err_index=np.where(inner_test_err==min(inner_test_err))
    outer_opt_lambda=inner_test_lambda[np.min(best_from_inner_test_err_index)]


        #logistic model with best lambda
    mdl = LogisticRegression(penalty='l2', C=outer_opt_lambda )
        
    mdl.fit(X_train, Y_train)

    Y_train_est = mdl.predict(X_train).T
    Y_train_est=np.asarray(Y_train_est).reshape(len(Y_train_est),1)

    y_test_est = mdl.predict(X_test)
    y_test_est=np.asarray(y_test_est).reshape(len(y_test_est),1)


    best_train_error_rate = (Y_train_est != Y_train).sum() / len(Y_train)
    best_test_error_rate = (y_test_est != Y_test).sum() / len(Y_test)

    test_error_rate_by_fold.append([best_test_error_rate,opt_lambda])
    train_error_rate_by_fold.append(best_train_error_rate)

        
    part_count+=1

print("done")

