#!/usr/bin/env python
# coding: utf-8

# In[ ]:
import numpy as np
from scipy.stats import multivariate_normal
from scipy.stats import norm
from scipy.stats import multinomial
from scipy.stats import logistic
from scipy import linalg as LA

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision

ReLu = nn.ReLU(inplace=True)
m = nn.Softmax(dim = 1)

def proj(x, V):
    pr = np.zeros(x.shape)
    for i in range(V.shape[1]):
        pr += np.dot(x, V[:,i]) * V[:,i]
    return 1 - np.dot(pr,x)

def M_true(D, seed):
    d = D.shape[0]
    np.random.seed(42)
    A_ = np.random.randn(d, d)
    U, _, _ = LA.svd(A_, full_matrices=False)
    return U @ np.diag(D) @ U.T, U @ np.diag(np.sqrt(D)), U

def random_covariance(D, seed):
    d = D.shape[0]
    np.random.seed(seed)
    A_ = np.random.randn(d, d)
    U, _, _ = LA.svd(A_, full_matrices=False)
    return U @ np.diag(D) @ U.T, U

def sample(f, N):
    X = f.rvs(N)
    Y = f.rvs(N)
    return X, Y

def label(d, tau):
    l = tau.shape[0]+1
    T = np.zeros(l)
    T[0] = d
    for i in range(1,l):
        T[i] = T[i-1] + d - tau[i-1]
    return np.argmax(T)

def data_generator(X, Y, B, tau, noise_type = None  , noise_par = None):
    n , d = X.shape
    l = tau.shape[0]+1
    D_noisy = np.zeros(n, dtype = int) 
    D_no_noisy = np.zeros(n, dtype = int)
    
    dist = (((X - Y) @ B)**2).sum(axis = 1) 
    
    for i in range(n):
        d = dist[i]  
        D_no_noisy[i] = label(d, tau)
        
        if noise_type == 'Gaussian':
            noise = norm(0, noise_par).rvs()
            D_noisy[i] = label(d + noise, tau)
            
        elif noise_type == 'Noisy_labeling':
            
            if np.random.uniform() > noise_par:
                D_noisy[i] = label(d, tau)
            else: 
                D_noisy[i] = np.argmax(multinomial(1, np.ones(l)/l).rvs())
                
        elif noise_type == 'logistic':
            noise = logistic(loc = 0, scale = noise_par).rvs()
            D_noisy[i] = label(d + noise, tau)
                
        elif noise_type is None:
            D_noisy[i] = D_no_noisy[i]
            
        else:
            print("there is no such noise_type!!!!")
            return 
    return D_noisy, D_no_noisy

def train_test_split(X, Y, D, D_no_noise, n_train):
    n, k = X.shape
    
    X_T, Y_T, D_T = X[:n_train], Y[:n_train], D[:n_train]
    X_test, Y_test, D_test = X[n_train:], Y[n_train:], D[n_train:]
    D_no_noise_test = D_no_noise[n_train:]
    D_no_noise_T = D_no_noise[:n_train]
    
    return X, Y, D, X_T, Y_T, D_T, D_no_noise_T, X_test, Y_test, D_test, D_no_noise_test


log = lambda x: torch.log(1e-20 + x)

def Accumulative_Sum(tau):
    l = tau.shape[0] + 1
    accum_tau = torch.zeros(l, dtype = torch.float64)
    for i in range(1, l):
        accum_tau[i] = accum_tau[i-1] + tau[i-1]
    return accum_tau

def pred(x, y, B, tau):
    dis = ((x - y) @ B).square().sum(axis = 1)
    l = tau.shape[0]+1
    accum_tau = Accumulative_Sum(tau)
    W = dis.reshape(-1,1) * torch.tensor([i for i in range(1, l+1)], 
                                         dtype = torch.float64).reshape(1,-1) - accum_tau.reshape(1,-1)
    return torch.argmax(W, dim=1)

def prob(dist, tau):
    l = tau.shape[0]+1
    accum_tau = Accumulative_Sum(tau)
    W = dist.reshape(-1,1) * torch.tensor([i for i in range(1,l+1)], 
                                          dtype = torch.float64).reshape(1,-1) - accum_tau.reshape(1,-1)
    return m(W)

def loss(Z, L, B, tau):
    dist = (Z @ B).square().sum(axis = 1)
    P = prob(dist, tau)
    Scores = -torch.log(1e-30 + P[range(P.shape[0]), L])    
    return Scores.mean()

def error(epoch, loss, B_grad, X_T, Y_T, D_T, D_no_noise_T, X_test, Y_test, D_test, D_no_noise_test, B, Tau):
    if epoch>=0:
        print(f'epoch {epoch+1}:\n norm of B.grad = {B_grad.square().sum().item()},\n B.grad.max = {B_grad.max().item()},\n loss = {loss}')
        print(Tau)
    else:
        print(f'epoch {0}: loss = {loss}')
        
                
    Pred_T = pred(X_T, Y_T, B, Tau)
    print("train accuracy with noise", (Pred_T == D_T).sum().item()/D_T.shape[0])
    print("train accuracy without noise", (Pred_T == D_no_noise_T).sum().item()/D_no_noise_T.shape[0])


    Pred_test = pred(X_test, Y_test, B, Tau)
    print("test accuracy with noise", (Pred_test == D_test).sum().item()/D_test.shape[0])
    print("test accuracy without noise", (Pred_test == D_no_noise_test).sum().item()/D_no_noise_test.shape[0])
    
    
def L_1_f_norm(f, B, tau, B_hat, tau_hat, N = 10000):
    B_star_normal = B / np.sqrt(tau[0])
    B_hat_normal = B_hat /np.sqrt(1e-20+tau_hat[0])
    X, Y = sample(f, N)
    Z = X - Y 
    temp = ((Z @ B_star_normal)**2).sum(axis = 1) - ((Z @ B_hat_normal)**2).sum(axis = 1)
    return (np.abs(temp)).mean()    
    
def make_history(X_T, Y_T, D_T, D_no_noise_T, X_test, Y_test, D_test, D_no_noise_test, B, Tau, 
                 f, B_star, tau_star, N, 
                 train_accuracy_with_noise,
                 train_accuracy_without_noise,
                 test_accuracy_with_noise,
                 test_accuracy_without_noise,
                 loss_history,
                 L_1_f_norm_history):
    Pred_T = pred(X_T, Y_T, B, Tau)
    train_accuracy_with_noise.append((Pred_T == D_T).sum().item()/D_T.shape[0])
    train_accuracy_without_noise.append((Pred_T == D_no_noise_T).sum().item()/D_no_noise_T.shape[0])


    Pred_test = pred(X_test, Y_test, B, Tau)
    test_accuracy_with_noise.append((Pred_test == D_test).sum().item()/D_test.shape[0])
    test_accuracy_without_noise.append((Pred_test == D_no_noise_test).sum().item()/D_no_noise_test.shape[0])
    loss_history.append(loss(X_T - Y_T, D_T, B, Tau))
    if f is not None:
        L_1_f_norm_history.append(L_1_f_norm(f, B_star, tau_star, 
                                             B.detach().numpy(), 
                                             Tau.detach().numpy(), 
                                             N))
    

class ML():
    def __init__(self, 
                 d, k, n_labels, 
                 X_T, Y_T, D_T, D_no_noise_T, 
                 X_test, Y_test, D_test, D_no_noise_test):
        
        self.X_T = X_T.requires_grad_(False) 
        self.Y_T = Y_T.requires_grad_(False) 
        self.D_T = D_T.requires_grad_(False)
        
        self.d = d
        self.k = k
        self.n_labels = n_labels
        
        self.D_no_noise_T = D_no_noise_T
        self.X_test = X_test 
        self.Y_test = Y_test 
        self.D_test = D_test
        self.D_no_noise_test = D_no_noise_test
        self.train_accuracy_with_noise = []
        self.train_accuracy_without_noise = []
        self.test_accuracy_with_noise = []
        self.test_accuracy_without_noise = []
        self.epoch_history = []
        self.loss_history = []
        self.L_1_f_norm_history = []
        
        self.B = (1e-8 * torch.randn((self.d,self.k), 
                                     dtype = torch.float64)).clone().detach().requires_grad_(True) #1e-6
        
        tau = [np.random.uniform()]
        for i in range(1, self.n_labels - 1):
            tau.append(tau[i-1] + np.random.uniform())
        self.Tau  = (1 * torch.tensor(tau, dtype = torch.float64)).clone().detach().requires_grad_(True)        
        
    def train(self, learning_rate = 5e-1, n_iters = 30000, decay = 1, show_log = True,
             f = None, B_star = None, tau_star = None, N = None):
        
        if show_log: print("Starting Tau: ", self.Tau)
        
        with torch.no_grad(): 
            
            self.epoch_history.append(1)
            make_history(self.X_T, self.Y_T, self.D_T, self.D_no_noise_T, 
                         self.X_test, self.Y_test, self.D_test, 
                         self.D_no_noise_test, self.B, self.Tau, 
                         f, B_star, tau_star, N, 
                         self.train_accuracy_with_noise,
                         self.train_accuracy_without_noise,
                         self.test_accuracy_with_noise,
                         self.test_accuracy_without_noise,
                         self.loss_history, 
                         self.L_1_f_norm_history) 
            

            if show_log:
                error(-1, loss(self.X_T - self.Y_T, self.D_T, self.B, self.Tau).item(), 
                      None, self.X_T, self.Y_T, self.D_T, self.D_no_noise_T, 
                      self.X_test, self.Y_test, self.D_test, self.D_no_noise_test, 
                      self.B, self.Tau)
                

        for epoch in range(n_iters):
            l = loss(self.X_T - self.Y_T, self.D_T, self.B, self.Tau)
            # calculate gradients = backward pass
            l.backward()

            # update weights
            with torch.no_grad():
                self.B -= learning_rate * self.B.grad
                self.Tau -= learning_rate * self.Tau.grad
                
                if self.Tau[0] < 0:
                    ReLu(self.Tau)
                
                

            if (epoch+1) % 500 == 0:
                with torch.no_grad(): 
                    self.epoch_history.append(epoch+1)
                    make_history(self.X_T, self.Y_T, self.D_T, self.D_no_noise_T, 
                         self.X_test, self.Y_test, self.D_test, 
                         self.D_no_noise_test, self.B, self.Tau, 
                         f, B_star, tau_star, N, 
                         self.train_accuracy_with_noise,
                         self.train_accuracy_without_noise,
                         self.test_accuracy_with_noise,
                         self.test_accuracy_without_noise,
                         self.loss_history, 
                         self.L_1_f_norm_history)

            if (epoch) % 5000 == 0:
                with torch.no_grad():
                    learning_rate *= decay
                    if show_log:
                        error(epoch, l.item(), self.B.grad, 
                              self.X_T, self.Y_T, self.D_T, self.D_no_noise_T, 
                              self.X_test, self.Y_test, self.D_test, 
                              self.D_no_noise_test, self.B, self.Tau)

            self.B.grad.zero_()
            self.Tau.grad.zero_()


    def accuracy(self, X, Y, L):
        Pred = pred(X, Y, self.B, self.Tau)
        return (Pred == L).sum().item()/L.shape[0]    
    
class DG:
    def __init__(self, M_diag, seed):
        self.M_diag = M_diag
        self.M_t, self.B_t, self.U = M_true(M_diag, seed)
        print("Ground truth M_t and B_t are generated (M_t = B_t @ B_t.T)")
        print("To access to M_t and B_t, use .M_t and .B_t")
        
    def generate(self, f, N, tau, noise_type = None, noise_par = None):
        self.tau_t = tau
        X, Y =  sample(f, N)
        D_noisy, D_no_noisy = data_generator(X, Y, 
                                             self.B_t, 
                                             tau = tau, 
                                             noise_type = noise_type, 
                                             noise_par = noise_par)
        return X, Y, D_noisy, D_no_noisy    
    