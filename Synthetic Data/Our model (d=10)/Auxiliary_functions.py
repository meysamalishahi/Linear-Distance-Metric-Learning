#!/usr/bin/env python
# coding: utf-8

# In[1]:
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

def truncated_error(X_T, Y_T, D_T, D_true_T, 
                    X_test, Y_test, D_test, D_true_test,
                    model, M_true, tau_true, k):
    d = X_T.shape[1]
    U_B_hat, S_B_hat, V_B_hat = LA.svd(model.B.detach().numpy(), full_matrices = False)
    S_B_hat_truncated = np.zeros(S_B_hat.shape)
    for i in range(k):
        S_B_hat_truncated[i] = S_B_hat[i]
    B_hat_truncated = U_B_hat @ np.diag(S_B_hat_truncated) @ V_B_hat
    Pred_T = pred(X_T, Y_T, B_hat_truncated, model.Tau)
    
    print("train accuracy using M_hat_{} = ".format(k), (Pred_T == D_T).sum().item()/D_T.shape[0])


    Pred_test = pred(X_test, Y_test, B_hat_truncated, model.Tau)
    print("test accuracy using M_hat_{} = ".format(k), (Pred_test == D_test).sum().item()/D_test.shape[0])
    
    model_B = model.B.detach().numpy()
    model_tau = model.Tau[0].item()
    M_hat_normal = model_B @ model_B.T / model_tau
    
    M_star_normal = M_true@M_true.T/tau_true
    
    r_s_norm = d_metric(M_star_normal, 0, M_hat_normal, 0)/d_metric(M_star_normal, 0)
    print('Realative spectral norm for k = {} is {}'.format(d, r_s_norm))
    r_f_norm = F_R(M_star_normal, M_hat_normal)
    print('Realative Frobenius norm for k = {} is {}'.format(d, r_f_norm))
    
    
    M_hat_normal_k = B_hat_truncated @ B_hat_truncated.T / model_tau
    r_s_norm_k = d_metric(M_star_normal, 0, M_hat_normal_k, 0)/d_metric(M_star_normal, 0)
    print('Realative spectral norm for k = {} is {}'.format(k, r_s_norm_k))
    r_f_norm_k = F_R(M_star_normal, M_hat_normal_k)
    print('Realative Frobenius norm for k = {} is {}'.format(k, r_f_norm_k))


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




