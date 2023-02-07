# -*- coding: utf-8 -*-
"""
Created on Sat Aug  6 20:01:33 2022

@author: 61995
"""

import torch
import torch.nn as nn
import numpy as np
from copy import deepcopy
import matplotlib.pyplot as plt
from metrics.visualization_metrics import visualization
import torch.optim as optim


class GRU(nn.Module):
    def __init__(self, num_series, hidden):

        super(GRU, self).__init__()
        self.p = num_series
        self.hidden = hidden

        # Set up network.
        self.gru = nn.GRU(num_series, hidden, batch_first=True)
        self.gru.flatten_parameters()
        self.linear = nn.Linear(hidden, 1)
        self.sigmoid = nn.Sigmoid()

    def init_hidden(self, batch):
        #Initialize hidden states
        device = self.gru.weight_ih_l0.device
        return torch.zeros(1, batch, self.hidden, device=device)
               

    def forward(self, X, z, connection, mode = 'train'):
        
        X=X[:,:,np.where(connection!=0)[0]]
        device = self.gru.weight_ih_l0.device
        tau = 0
        if mode == 'train':
          X_right, hidden_out = self.gru(torch.cat((X[:,0:1,:],X[:,11:-1,:]),1), z)
          X_right = self.linear(X_right)

          return X_right, hidden_out
          
class VRAE4E(nn.Module):
    def __init__(self, num_series, hidden):
        '''
        Error VAE
        '''
        super(VRAE4E, self).__init__()
        self.device = torch.device('cuda')
        self.p = num_series
        self.hidden = hidden
        
        self.gru_left = nn.GRU(num_series, hidden, batch_first=True)
        self.gru_left.flatten_parameters()
        
        self.fc_mu = nn.Linear(hidden, hidden)#nn.Linear(hidden, 1)
        self.fc_std = nn.Linear(hidden, hidden)
        
        self.linear_hidden = nn.Linear(hidden, hidden)
        self.tanh = nn.Tanh()
        
        
        self.gru = nn.GRU(num_series, hidden, batch_first=True)
        self.gru.flatten_parameters()
        self.linear = nn.Linear(hidden, num_series)
        

        


    def init_hidden(self, batch):
        '''Initialize hidden states for GRU cell.'''
        device = self.gru.weight_ih_l0.device
        return torch.zeros(1, batch, self.hidden, device=device)
               

    def forward(self, X, mode = 'train'):
        
        X = torch.cat((torch.zeros(X.shape,device = self.device)[:,0:1,:],X),1)
        if mode == 'train':
            

            hidden_0 = torch.zeros(1, X.shape[0], self.hidden, device=self.device)
            out, h_t = self.gru_left(X[:,1:,:], hidden_0.detach())
            
            mu = self.fc_mu(h_t)
            log_var = self.fc_std(h_t)
            
            sigma = torch.exp(0.5*log_var)
            z = torch.randn(size = mu.size())
            z = z.type_as(mu) 
            z = mu + sigma*z
            z = self.tanh(self.linear_hidden(z))

            
            X_right, hidden_out = self.gru(X[:,:-1,:], z)

            pred = self.linear(X_right)
            
            

            return pred, log_var, mu
        if mode == 'test':

            X_seq = torch.zeros(X[:,:1,:].shape).to(self.device)
            h_t = torch.randn(size = (1, X_seq[:,-2:-1,:].size(0),self.hidden)).to(self.device)
            for i in range(int(20/1)+1):
                out, h_t = self.gru(X_seq[:,-1:,:], h_t)
                out = self.linear(out)
                #out = self.sigmoid(out)
                X_seq = torch.cat([X_seq,out],dim = 1)
            return X_seq
        

          


            
                
          


class CRVAE(nn.Module):
    def __init__(self, num_series, connection, hidden):
        '''
        connection: pruned networks
        '''
        super(CRVAE, self).__init__()
        
        self.device = torch.device('cuda')
        self.p = num_series
        self.hidden = hidden
        
        self.gru_left = nn.GRU(num_series, hidden, batch_first=True)
        self.gru_left.flatten_parameters()
        
        self.fc_mu = nn.Linear(hidden, hidden)
        self.fc_std = nn.Linear(hidden, hidden)
        self.connection = connection

        # Set up networks.
        self.networks = nn.ModuleList([
            GRU(int(connection[:,i].sum()), hidden) for i in range(num_series)])

    def forward(self, X, noise = None, mode = 'train', phase = 0):

        if phase == 0:
            X = torch.cat((torch.zeros(X.shape,device = self.device)[:,0:1,:],X),1)
            if mode == 'train':
                
    
                hidden_0 = torch.zeros(1, X.shape[0], self.hidden, device=self.device)
                out, h_t = self.gru_left(X[:,1:11,:], hidden_0.detach())
                
                mu = self.fc_mu(h_t)
                log_var = self.fc_std(h_t)
                
                sigma = torch.exp(0.5*log_var)
                z = torch.randn(size = mu.size())
                z = z.type_as(mu)
                z = mu + sigma*z
    
                pred = [self.networks[i](X, z, self.connection[:,i])[0]
                      for i in range(self.p)]
    
                return pred, log_var, mu
            if mode == 'test':
                X_seq = torch.zeros(X[:,:1,:].shape).to(self.device)
                h_0 = torch.randn(size = (1, X_seq[:,-2:-1,:].size(0),self.hidden)).to(self.device)
                ht_last =[]
                for i in range(self.p):
                    ht_last.append(h_0)
                for i in range(int(20/1)+1):#int(20/2)+1
                    
                    ht_new = []
                    for j in range(self.p):
                        # out, h_t = self.gru_out[j](X_seq[:,-1:,:], ht_last[j])
                        # out = self.fc[j](out)
                        out, h_t = self.networks[j](X_seq[:,-1:,:], ht_last[j], self.connection[:,j])
                        if j == 0:
                            X_t = out
                        else:
                            X_t = torch.cat((X_t,out),-1)
                        ht_new.append(h_t)
                    ht_last = ht_new
                    if i ==0:
                        X_seq = X_t
                    else:
                        X_seq = torch.cat([X_seq,X_t],dim = 1)
                        
                    #out = self.sigmoid(out)
                    
                return X_seq
            
        
        if phase == 1:
            X = torch.cat((torch.zeros(X.shape,device = self.device)[:,0:1,:],X),1)
            if mode == 'train':
                
    
                hidden_0 = torch.zeros(1, X.shape[0], self.hidden, device=self.device)
                out, h_t = self.gru_left(X[:,1:11,:], hidden_0.detach())
                
                mu = self.fc_mu(h_t)
                log_var = self.fc_std(h_t)
                
                sigma = torch.exp(0.5*log_var)
                z = torch.randn(size = mu.size())
                z = z.type_as(mu) # Setting z to be .cuda when using GPU training 
                z = mu + sigma*z
    
                pred = [self.networks[i](X, z, self.connection[:,i])[0]
                      for i in range(self.p)]
                
                
    
                return pred, log_var, mu
            if mode == 'test':
                X_seq = torch.zeros(X[:,:1,:].shape).to(self.device)
                h_0 = torch.randn(size = (1, X_seq[:,-2:-1,:].size(0),self.hidden)).to(self.device)
                ht_last =[]
                for i in range(self.p):
                    ht_last.append(h_0)
                for i in range(int(20/1)+1):#int(20/2)+1
                    
                    ht_new = []
                    for j in range(self.p):
                        # out, h_t = self.gru_out[j](X_seq[:,-1:,:], ht_last[j])
                        # out = self.fc[j](out)
                        out, h_t = self.networks[j](X_seq[:,-1:,:], ht_last[j], self.connection[:,j])
                        if j == 0:
                            X_t = out
                        else:
                            X_t = torch.cat((X_t,out),-1)
                        ht_new.append(h_t)
                    ht_last = ht_new
                    if i ==0:
                        X_seq = X_t + 0.1*noise[:,i:i+1,:] 
                    else:
                        X_seq = torch.cat([X_seq,X_t+0.1*noise[:,i:i+1,:]],dim = 1)
                        
                    #out = self.sigmoid(out)
                    
                return X_seq
        

    def GC(self, threshold=True):
        '''
        Extract learned Granger causality.

        Args:
          threshold: return norm of weights, or whether norm is nonzero.

        Returns:
          GC: (p x p) matrix. Entry (i, j) indicates whether variable j is
            Granger causal of variable i.
        '''
        GC = [torch.norm(net.gru.weight_ih_l0, dim=0)
              for net in self.networks]
        GC = torch.stack(GC)
        #print(GC)
        if threshold:
            return (torch.abs(GC) > 0).int()
        else:
            return GC





def prox_update(network, lam, lr):
    '''Perform in place proximal update on first layer weight matrix.'''
    W = network.gru.weight_ih_l0
    norm = torch.norm(W, dim=0, keepdim=True)
    W.data = ((W / torch.clamp(norm, min=(lam * lr)))
              * torch.clamp(norm - (lr * lam), min=0.0))
    network.gru.flatten_parameters()





def regularize(network, lam):
    '''Calculate regularization term for first layer weight matrix.'''
    W = network.gru.weight_ih_l0
    return lam * torch.sum(torch.norm(W, dim=0))


def ridge_regularize(network, lam):
    '''Apply ridge penalty at linear layer and hidden-hidden weights.'''
    return lam * (
        torch.sum(network.linear.weight ** 2) +
        torch.sum(network.gru.weight_hh_l0 ** 2))# + 
        #torch.sum(network.fc_std.weight ** 2) + 
        #torch.sum(network.fc_mu.weight ** 2) + 
        #torch.sum(network.fc_std.weight ** 2))



def restore_parameters(model, best_model):
    '''Move parameter values from best_model to model.'''
    for params, best_params in zip(model.parameters(), best_model.parameters()):
        params.data = best_params


def arrange_input(data, context):
    '''
    Arrange a single time series into overlapping short sequences.

    Args:
      data: time series of shape (T, dim).
      context: length of short sequences.
    '''
    assert context >= 1 and isinstance(context, int)
    input = torch.zeros(len(data) - context, context, data.shape[1],
                        dtype=torch.float32, device=data.device)
    target = torch.zeros(len(data) - context, context, data.shape[1],
                         dtype=torch.float32, device=data.device)
    for i in range(context):
        start = i
        end = len(data) - context + i
        input[:, i, :] = data[start:end]
        target[:, i, :] = data[start+1:end+1]
    return input.detach(), target.detach()


def MinMaxScaler(data):
  """Min-Max Normalizer.
  
  Args:
    - data: raw data
    
  Returns:
    - norm_data: normalized data
    - min_val: minimum values (for renormalization)
    - max_val: maximum values (for renormalization)
  """    
  min_val = np.min(np.min(data, axis = 0), axis = 0)
  data = data - min_val
    
  max_val = np.max(np.max(data, axis = 0), axis = 0)
  norm_data = data / (max_val + 1e-7)
    
  return norm_data


def train_phase2(crvae, vrae, X, context, lr, max_iter, lam=0, lam_ridge=0,
                     lookback=5, check_every=50, verbose=1,sparsity = 100, batch_size = 1024):
    '''Train model with Adam.'''
    optimizer = optim.Adam(vrae.parameters(), lr=1e-3)
    p = X.shape[-1]
    device = crvae.networks[0].gru.weight_ih_l0.device
    loss_fn = nn.MSELoss()
    train_loss_list = []
    batch_size = batch_size
    # Set up data.
    X, Y = zip(*[arrange_input(x, context) for x in X])
    X_all = torch.cat(X, dim=0)
    Y_all = torch.cat(Y, dim=0)
    
    
    idx = np.random.randint(len(X_all), size=(batch_size,))
    
    X = X_all[idx]
    
    Y = Y_all[idx]
    X_v = X_all[batch_size:]
    start_point = 0#context-10-1
    beta = 1#0.001
    beta_e = 1
    # For early stopping.
    best_it = None
    best_loss = np.inf
    best_model = None
    
    # Calculate smooth error.
    pred,mu,log_var = crvae(X)#


    
    loss = sum([loss_fn(pred[i][:, :, 0], X[:, 10:, i]) for i in range(p)])

    
    
    mmd = (-0.5*(1+log_var - mu**2- torch.exp(log_var)).sum(dim = -1).sum(dim = 0)).mean(dim =0)
    #mmd =  sum([MMD(torch.randn(200, Y[:, :, 0].shape[-1], requires_grad = False).to(device), latent[i][:,:,0]) for i in range(p)])
    ridge = sum([ridge_regularize(net, lam_ridge) for net in crvae.networks])
    smooth = loss + ridge + beta*mmd
    
    
    error = (-torch.stack(pred)[:, :, :, 0].permute(1,2,0) + X[:, 10:, :]).detach()
    pred_e,mu_e,log_var_e = vrae(error)
    loss_e = loss_fn(pred_e, error)
    mmd_e = (-0.5*(1+log_var_e - mu_e**2- torch.exp(log_var_e)).sum(dim = -1).sum(dim = 0)).mean(dim =0)
    smooth_e = loss_e + beta_e*mmd_e

    best_mmd = np.inf       
            
########################################################################   
    #lr = 1e-3        
    for it in range(max_iter):
        # Take gradient step.
        smooth_e.backward()
        if lam == 0:
            optimizer.step()
            optimizer.zero_grad()  
        
        smooth.backward()
        for param in crvae.parameters():
            param.data -= lr * param.grad

        # Take prox step.
        if lam > 0:
            for net in crvae.networks:
                prox_update(net, lam, lr)

        

        crvae.zero_grad()

        # Calculate loss for next iteration.
        idx = np.random.randint(len(X_all), size=(batch_size,))
    
        #X = X_all[idx]
       
        #Y = Y_all[idx]
        
        pred,mu,log_var = crvae(X)#
        loss = sum([loss_fn(pred[i][:, :, 0], X[:, 10:, i]) for i in range(p)])

        
        mmd = (-0.5*(1+log_var - mu**2- torch.exp(log_var)).sum(dim = -1).sum(dim = 0)).mean(dim =0)
        
        ridge = sum([ridge_regularize(net, lam_ridge)
                     for net in crvae.networks])
        smooth = loss + ridge + beta*mmd
        
        
        error = (-torch.stack(pred)[:, :, :, 0].permute(1,2,0) + X[:, 10:, :]).detach()
        pred_e,mu_e,log_var_e = vrae(error)
        loss_e = loss_fn(pred_e, error)
        mmd_e = (-0.5*(1+log_var_e - mu_e**2- torch.exp(log_var_e)).sum(dim = -1).sum(dim = 0)).mean(dim =0)
        smooth_e = loss_e + beta_e*mmd_e
        


        # Check progress.
        if (it) % check_every == 0:

            
            
            X_t = X
            pred_t,mu_t ,log_var_t= crvae(X_t)
            
        
            loss_t = sum([loss_fn(pred_t[i][:, :, 0], X_t[:, 10:, i]) for i in range(p)])
        

            
            mmd_t = (-0.5*(1+log_var_t - mu_t**2- torch.exp(log_var_t)).sum(dim = -1).sum(dim = 0)).mean(dim =0) 
        
            ridge_t = sum([ridge_regularize(net, lam_ridge)
                     for net in crvae.networks])
            smooth_t = loss_t + ridge_t# + beta*mmd_t
            
            nonsmooth = sum([regularize(net, lam) for net in crvae.networks])
            mean_loss = (smooth_t) / p

            if verbose > 0:
                print(('-' * 10 + 'Iter = %d' + '-' * 10) % (it ))
                print('Loss = %f' % mean_loss)
                print('KL = %f' % mmd)
                

                print('Loss_e = %f' % smooth_e)
                print('KL_e = %f' % mmd_e)
                
                if lam>0:
                  print('Variable usage = %.2f%%'
                        % (100 * torch.mean(crvae.GC().float())))


            
            if mean_loss < best_loss:
                best_loss = mean_loss
                best_it = it
                best_model = deepcopy(crvae)


                
            start_point = 0
            predicted_error = vrae(error, mode = 'test').detach()
            
            predicted_data = crvae(X_t, predicted_error, mode = 'test', phase = 1)
            syn = predicted_data[:,:-1,:].cpu().detach().numpy()
            ori= X_t[:,start_point:,:].cpu().detach().numpy()
            

            if it % 1000 ==0:
                plt.plot(ori[0,:,1])
                plt.plot(syn[0,:,1])
                plt.show()

                visualization(ori, syn, 'pca')
                visualization(ori, syn, 'tsne')
                np.save('ori_henon.npy',ori)
                np.save('syn_henon.npy',syn)


    # Restore best model.
    restore_parameters(crvae, best_model)

    return train_loss_list


def train_phase1(crvae, X, context, lr, max_iter, lam=0, lam_ridge=0,
                     lookback=5, check_every=50, verbose=1,sparsity = 100, batch_size = 2048):
    '''Train model with Adam.'''
    p = X.shape[-1]
    device = crvae.networks[0].gru.weight_ih_l0.device
    loss_fn = nn.MSELoss()
    train_loss_list = []
    batch_size = batch_size
    # Set up data.
    X, Y = zip(*[arrange_input(x, context) for x in X])
    X_all = torch.cat(X, dim=0)
    Y_all = torch.cat(Y, dim=0)
    
    

    idx = np.random.randint(len(X_all), size=(batch_size,))
    
    X = X_all[idx]
    
    Y = Y_all[idx]
    start_point = 0
    beta = 0.1
    # For early stopping.
    best_it = None
    best_loss = np.inf
    best_model = None
    
    # Calculate crvae error.
    pred,mu,log_var = crvae(X)

    loss = sum([loss_fn(pred[i][:, :, 0], X[:, 10:, i]) for i in range(p)])

    
    mmd = (-0.5*(1+log_var - mu**2- torch.exp(log_var)).sum(dim = -1).sum(dim = 0)).mean(dim =0)
    #mmd =  sum([MMD(torch.randn(200, Y[:, :, 0].shape[-1], requires_grad = False).to(device), latent[i][:,:,0]) for i in range(p)])
    ridge = sum([ridge_regularize(net, lam_ridge) for net in crvae.networks])
    smooth = loss + ridge + beta*mmd
    

    best_mmd = np.inf       
            
########################################################################   
    #lr = 1e-3        
    for it in range(max_iter):
        # Take gradient step.
        smooth.backward()
        for param in crvae.parameters():
            param.data -= lr * param.grad

        # Take prox step.
        if lam > 0:
            for net in crvae.networks:
                prox_update(net, lam, lr)
        

        

        crvae.zero_grad()


        pred,mu,log_var = crvae(X)
        loss = sum([loss_fn(pred[i][:, :, 0], X[:, 10:, i]) for i in range(p)])

        
        mmd = (-0.5*(1+log_var - mu**2- torch.exp(log_var)).sum(dim = -1).sum(dim = 0)).mean(dim =0)
        
        ridge = sum([ridge_regularize(net, lam_ridge)
                     for net in crvae.networks])
        smooth = loss + ridge + beta*mmd
        

        # Check progress.
        if (it) % check_every == 0:     
            X_t = X
            Y_t = Y
            
            pred_t,mu_t ,log_var_t= crvae(X_t)
            
        
            loss_t = sum([loss_fn(pred_t[i][:, :, 0], X_t[:, 10:, i]) for i in range(p)])
        

            
            mmd_t = (-0.5*(1+log_var_t - mu_t**2- torch.exp(log_var_t)).sum(dim = -1).sum(dim = 0)).mean(dim =0) 
        
            ridge_t = sum([ridge_regularize(net, lam_ridge)
                     for net in crvae.networks])
            smooth_t = loss_t + ridge_t# + beta*mmd_t
            
            nonsmooth = sum([regularize(net, lam) for net in crvae.networks])
            mean_loss = (smooth_t) / p

            if verbose > 0:
                print(('-' * 10 + 'Iter = %d' + '-' * 10) % (it ))
                print('Loss = %f' % mean_loss)
                print('KL = %f' % mmd)
                
                if lam>0:
                  print('Variable usage = %.2f%%'
                        % (100 * torch.mean(crvae.GC().float())))



            if mean_loss < best_loss:
                best_loss = mean_loss
                best_it = it
                best_model = deepcopy(crvae)

                
            start_point = 0
            predicted_data = crvae(X_t,mode = 'test')
            syn = predicted_data[:,:-1,:].cpu().detach().numpy()
            ori= X_t[:,start_point:,:].cpu().detach().numpy()
            
            syn = MinMaxScaler(syn)
            ori = MinMaxScaler(ori)

    # Restore best model.
    restore_parameters(crvae, best_model)

    return train_loss_list