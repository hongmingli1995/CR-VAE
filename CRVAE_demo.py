# -*- coding: utf-8 -*-
"""
Created on Sat Aug  6 20:00:04 2022

@author: 61995
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from models.cgru_error import CRVAE, VRAE4E, train_phase1, train_phase2
import scipy.io





device = torch.device('cuda')
X_np = np.load('henon.npy').T
dim = X_np.shape[-1]
GC = np.zeros([dim,dim])
for i in range(dim):
    GC[i,i] = 1
    if i!=0:
        GC[i,i-1] = 1
X = torch.tensor(X_np[np.newaxis], dtype=torch.float32, device=device)


full_connect = np.ones(GC.shape)
cgru = CRVAE(X.shape[-1], full_connect, hidden=64).cuda(device=device)
vrae = VRAE4E(X.shape[-1], hidden=64).cuda(device=device)


#%%


train_loss_list = train_phase1(
    cgru, X, context=20, lam=0.1, lam_ridge=0, lr=5e-2, max_iter=1000,
    check_every=50)#0.1

#%%


GC_est = cgru.GC().cpu().data.numpy()

print('True variable usage = %.2f%%' % (100 * np.mean(GC)))
print('Estimated variable usage = %.2f%%' % (100 * np.mean(GC_est)))
print('Accuracy = %.2f%%' % (100 * np.mean(GC == GC_est)))

# Make figures
fig, axarr = plt.subplots(1, 2, figsize=(10, 5))
axarr[0].imshow(GC, cmap='Blues')
axarr[0].set_title('Causal-effect matrix')
axarr[0].set_ylabel('Effect series')
axarr[0].set_xlabel('Causal series')
axarr[0].set_xticks([])
axarr[0].set_yticks([])

axarr[1].imshow(GC_est, cmap='Blues', vmin=0, vmax=1, extent=(0, len(GC_est), len(GC_est), 0))
axarr[1].set_ylabel('Effect series')
axarr[1].set_xlabel('Causal series')
axarr[1].set_xticks([])
axarr[1].set_yticks([])

# Mark disagreements
for i in range(len(GC_est)):
    for j in range(len(GC_est)):
        if GC[i, j] != GC_est[i, j]:
            rect = plt.Rectangle((j, i-0.05), 1, 1, facecolor='none', edgecolor='red', linewidth=1)
            axarr[1].add_patch(rect)

plt.show()

#np.save('GC_henon.npy', GC_est)
full_connect = np.load('GC_henon.npy')


#%%
cgru = CRVAE(X.shape[-1], full_connect, hidden=64).cuda(device=device)
vrae = VRAE4E(X.shape[-1], hidden=64).cuda(device=device)


train_loss_list = train_phase2(
    cgru, vrae, X, context=20, lam=0., lam_ridge=0, lr=5e-2, max_iter=10000,
    check_every=50)

