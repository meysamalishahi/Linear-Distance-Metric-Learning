#!/usr/bin/env python
# coding: utf-8

# In[1]:
import random
import numpy as np
from scipy.stats import multivariate_normal
from scipy.stats import norm
from scipy.stats import multinomial
from scipy.stats import logistic
from scipy.stats import expon, uniform, laplace, hypsecant
from scipy import linalg as LA

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
from Main_functions import *

# def truncated_error(X_T, Y_T, D_T, D_true_T, 
#                     X_test, Y_test, D_test, D_true_test,
#                     model, M_true, tau_true, k):
#     d = X_T.shape[1]
#     U_B_hat, S_B_hat, V_B_hat = LA.svd(model.B.detach().numpy(), full_matrices = False)
#     S_B_hat_truncated = np.zeros(S_B_hat.shape)
#     for i in range(k):
#         S_B_hat_truncated[i] = S_B_hat[i]
#     B_hat_truncated = U_B_hat @ np.diag(S_B_hat_truncated) @ V_B_hat
#     Pred_T = pred(X_T, Y_T, B_hat_truncated, model.Tau)
    
#     print("train accuracy using M_hat_{} = ".format(k), (Pred_T == D_T).sum().item()/D_T.shape[0])


#     Pred_test = pred(X_test, Y_test, B_hat_truncated, model.Tau)
#     print("test accuracy using M_hat_{} = ".format(k), (Pred_test == D_test).sum().item()/D_test.shape[0])
    
#     model_B = model.B.detach().numpy()
#     model_tau = model.Tau[0].item()
#     M_hat_normal = model_B @ model_B.T / model_tau
    
#     M_star_normal = M_true@M_true.T/tau_true
    
#     r_s_norm = d_metric(M_star_normal, 0, M_hat_normal, 0)/d_metric(M_star_normal, 0)
#     print('Realative spectral norm for k = {} is {}'.format(d, r_s_norm))
#     r_f_norm = F_R(M_star_normal, M_hat_normal)
#     print('Realative Frobenius norm for k = {} is {}'.format(d, r_f_norm))
    
    
#     M_hat_normal_k = B_hat_truncated @ B_hat_truncated.T / model_tau
#     r_s_norm_k = d_metric(M_star_normal, 0, M_hat_normal_k, 0)/d_metric(M_star_normal, 0)
#     print('Realative spectral norm for k = {} is {}'.format(k, r_s_norm_k))
#     r_f_norm_k = F_R(M_star_normal, M_hat_normal_k)
#     print('Realative Frobenius norm for k = {} is {}'.format(k, r_f_norm_k))


# In[2]:


def d_metric(M_1, t_1, M_2 = None, t_2 = None):
    if M_2 is None:
        _, E, _ = LA.svd(M_1, full_matrices=False)
        return E[0]+t_1
    else:
        _, E, _ = LA.svd(M_1 - M_2, full_matrices=False)
        return E[0] + np.abs(t_1-t_2)


# In[3]:


F_R = lambda M_1, M_2: LA.norm(M_1 - M_2)/LA.norm(M_1)


# In[ ]:
def gen_train_test_data(X, X_projected, n_trn, tau_true, scale = 0):
    n, d = X.shape
    n_tst = n - n_trn
    Range = [i for i in range(n)]
    random.shuffle(Range)
    I = [Range[i] for i in range(0, n, +2)]
    J = [Range[i] for i in range(1, n, +2)]
    X_projected_I = X_projected[I]
    Y_projected_J = X_projected[J]
    dists = ((X_projected_I - Y_projected_J) ** 2).sum(axis = 1)
    print('Average squared distance is {}'.format(dists.mean()))
    if scale > 0:
        D = (dists + np.random.normal(loc = 0, scale = scale, size = dists.shape[0]) >= tau_true) + 0
        D_true = (dists >= tau_true) + 0
    else: 
        D_true = (dists >= tau_true) + 0
        D = D_true
        
    print('Far labeled pairs {}%'.format(100*D_true.sum()/D_true.shape[0]))
    D = torch.tensor(D, dtype = torch.torch.int64)
    D_true = torch.tensor(D_true, dtype = torch.torch.int64)
    D_T = D[:n_trn]
    D_true_T = D_true[:n_trn]
    D_test = D[n_trn:]
    D_true_test = D_true[n_trn:]
    X_I = X[I]
    X_J = X[J]
    X = torch.tensor(X_I, dtype = torch.float64)
    Y = torch.tensor(X_J, dtype = torch.float64)
    
    X_T = X[:n_trn,:]
    Y_T = Y[:n_trn,:]

    X_test = X[n_trn:,:]
    Y_test = Y[n_trn:,:]
    return X_T, Y_T, D_T, D_true_T, X_test, Y_test, D_test, D_true_test


def Multi_run(n_runs, 
              d, k, 
              X, X_projected, n_trn, tau_true, scale = 0, 
              n_labels = 2,
              learning_rate = 5e-1, 
               n_iters = 20001, 
                decay = .9,
              show_log = False):   
    moled_list = []
    for i in range(n_runs):
        print( 'trial number {} begins.'.format(i))
        X_T, Y_T, D_T, D_true_T, X_test, Y_test, D_test, D_true_test = gen_train_test_data(X,
                                                                                           X_projected,
                                                                                           n_trn = n_trn,
                                                                                           tau_true= tau_true,
                                                                                           scale = scale)
        model = ML(d, k, n_labels, 
                 X_T, Y_T, D_T, D_true_T, 
                 X_test, Y_test, D_test, D_true_test)
        model.train(learning_rate = learning_rate, 
                    n_iters = n_iters, 
                    decay = decay,
                    show_log = show_log)

        moled_list.append(model)
    return moled_list


def truncated_error(model, M_true, tau_true, k = None, show_log = True):
    M_star_normal = M_true@M_true.T/tau_true    
    if k is not None:
        U_B_hat, S_B_hat, V_B_hat = LA.svd(model.B.detach().numpy(), full_matrices = False)
        S_B_hat_truncated = np.zeros(S_B_hat.shape)
        for i in range(k):
            S_B_hat_truncated[i] = S_B_hat[i]
        B_hat_truncated = U_B_hat @ np.diag(S_B_hat_truncated) @ V_B_hat
    else: 
        B_hat_truncated = model.B.detach().numpy()
        
    model_tau = model.Tau[0].item()
    if model_tau == 0.0:
        model_tau += 1e-20
    M_hat_normal_k = B_hat_truncated @ B_hat_truncated.T / model_tau
        
    Pred_T = pred(model.X_T, model.Y_T, B_hat_truncated, model.Tau)
    train_accuracy = (Pred_T == model.D_T).sum().item()/model.D_T.shape[0]
    
    Pred_test = pred(model.X_test, model.Y_test, B_hat_truncated, model.Tau)
    test_accuracy = (Pred_test == model.D_test).sum().item()/model.D_test.shape[0]
     
    r_s_norm_k = d_metric(M_star_normal, 0, M_hat_normal_k, 0)/d_metric(M_star_normal, 0)
    r_f_norm_k = F_R(M_star_normal, M_hat_normal_k)
    
    if show_log:
        print("train accuracy using M_hat_{} = ".format(k), train_accuracy)
        print("test accuracy using M_hat_{} = ".format(k), test_accuracy)
        print('Realative spectral norm for k = {} is {}'.format(k, r_s_norm_k))
        print('Realative Frobenius norm for k = {} is {}'.format(k, r_f_norm_k))
    return train_accuracy, test_accuracy, r_s_norm_k, r_f_norm_k


def create_report(model_list, n_runs, tau_true, M_true, k = None, show_log = False):
    train_accuracy_list = []
    test_accuracy_lis = []
    r_s_norm_k_list = []
    r_f_norm_k_list = []
    for i in range(n_runs):
        train_accuracy, test_accuracy, r_s_norm_k, r_f_norm_k = truncated_error(model_list[i], 
                                                                                M_true = M_true, 
                                                                                tau_true = tau_true, 
                                                                                k = k, 
                                                                                show_log = show_log)
        
        train_accuracy_list.append(train_accuracy)
        test_accuracy_lis.append(test_accuracy)
        r_s_norm_k_list.append(r_s_norm_k)
        r_f_norm_k_list.append(r_f_norm_k)
    return train_accuracy_list, test_accuracy_lis, r_s_norm_k_list, r_f_norm_k_list