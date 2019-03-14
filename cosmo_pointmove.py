#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

Created on Wed Jan 30 11:33:42 2019
@author: antoinebrochard
"""

import numpy as np
import torch
import matplotlib.pyplot as plt
from torch.autograd import grad
import scipy.optimize as opt
#import torch.nn.functional as F
import scipy.io
import sys
sys.path.append('/users/trec/brochard/kymatio_wph_pp')


'''
We approximate the given image by a sum of Gaussians, at different locations
and with different variance, but with the same total mass. The algorithm
optimizes the locations and variances of the Gaussians to match the wph moments
of the original image. The number of Gaussians for the approximation is a
parameter to choose.
'''

# load image

gpu = True
pi = torch.tensor(np.pi)
if gpu:
    pi = pi.cuda()
res = 256


im = scipy.io.loadmat('./data/cosmodata0_c5_N256.mat')['imgs'][:,:,0]

#plt.imshow(im); plt.show();

im = torch.from_numpy(im).type(torch.float).unsqueeze(0).unsqueeze(0).cuda()

m = torch.mean(im)  # We need the total mass to determine the mass of the Gaussians


nb_points = int(5*1e3)  # We choose the number of points for approximation


# define point model

def weight(index, pos, res, gpu, sigma):
    '''
    Returns a tensor w with size (nb_points, res).
    w[i,:] is of the form C*e^(-x^2 /(2*sigma^2)), where C is adapted
    to keep the mass constant, and x is the horizontal or vertical distance
    from point i to the image pixels.
    index: vector of pixel indices
    pos: vector of Gaussians positions (1d)
    res: resolution of the image (to periodize) (=N)
    sigma: vetor of Gaussians std.
    '''
    sigma_ = sigma.unsqueeze(1).expand(-1, res)
    # print(sigma_.shape)
    if gpu:
        index = torch.tensor(index).cuda()
        pos = pos.cuda()
        sigma_ = sigma_.cuda()
        res=torch.tensor(res).type(torch.float).cuda()
    v = torch.min(torch.min(torch.abs(pos-index), torch.abs(pos+res-index)),
                  torch.abs(pos-res-index))
    w = torch.exp(-torch.mul(v,v)/(2*(sigma_**2)))/(sigma_*torch.sqrt(2*pi))*torch.sqrt((m/nb_points))
    return w



def pos_to_im3(x, res, gpu, sigma):
    '''
    Uses the weight function to convert a tensor of positions and std to an
    image. It multiplies the horizontal and vertical weights to greate a 2d
    tensor. Then returns a tensor with shape (1,1,N,N).
    x: tensor of positions, with shape (nb_points, 2)
    sigma: tensor of std, with shape (nb_points, 1)
    '''
    Mx = torch.arange(0, res).type(torch.float)
    My = torch.arange(0, res).type(torch.float)
    if gpu:
        Mx = Mx.cuda(); My = My.cuda()
    im_x = weight(Mx.unsqueeze(0), x[:, 0].unsqueeze(1), res, gpu, sigma).unsqueeze(2)
    im_y = weight(My.unsqueeze(0), x[:, 1].unsqueeze(1), res, gpu, sigma).unsqueeze(1)
    #print(im_x.shape, im_y.shape)
    M = torch.matmul(im_x, im_y).sum(0)
    #print(M.shape)
    return M.unsqueeze(0).unsqueeze(0)



# Parameters for transform

J = 8
L = 8
M, N = im.shape[-2], im.shape[-1]
delta_j = 4
delta_l = L/2
delta_k = 1
nb_chunks = 20
nb_restarts = 0


# kymatio scattering
from kymatio.phaseharmonics2d.phase_harmonics_k_bump_chunkid_simplephase \
    import PhaseHarmonics2d

Sims = []
factr = 1e5
wph_ops = dict()
nCov = 0
for chunk_id in range(nb_chunks+1):
    wph_op = PhaseHarmonics2d(M, N, J, L, delta_j, delta_l, delta_k, nb_chunks, chunk_id)
    wph_op = wph_op.cuda()
    wph_ops[chunk_id] = wph_op
    Sim_ = wph_op(im)*factr # (nb,nc,nb_channels,1,1,2)
    nCov += Sim_.shape[2]
    Sims.append(Sim_)
print(1)

# ---- Reconstruct marks. At initiation, every point has the average value of the marks.----#
#---- Trying scipy L-BFGS ----#
def obj_fun(x,chunk_id):
    if x.grad is not None:
        x.grad.data.zero_()
    global wph_ops
    wph_op = wph_ops[chunk_id]
    im_ = res*res*pos_to_im3(x[:,:2], res, gpu, x[:,2])
#    print(im_.mean())
    p = wph_op(im_)*factr
    diff = p-Sims[chunk_id]
    loss = torch.mul(diff,diff).sum()/nCov
    return loss

grad_err = torch.zeros(nb_points, 3)

def grad_obj_fun(x_gpu):
    loss = 0
    global grad_err
    grad_err[:] = 0
    #global wph_ops
    for chunk_id in range(nb_chunks+1):
        x_t = x_gpu.clone().requires_grad_(True)
        loss_t = obj_fun(x_t,chunk_id)
        grad_err_t, = grad([loss_t],[x_t], retain_graph=False)
        loss = loss + loss_t
        grad_err = grad_err + grad_err_t
    return loss, grad_err

count = 0
from time import time
time0 = time()
def fun_and_grad_conv(x):
    x_float = torch.reshape(torch.tensor(x, requires_grad=True,dtype=torch.float),
                            (x.shape[0]//3, 3))
    loss, grad_err = grad_obj_fun(x_float)
    global count
    global time0
    count += 1
    if count%50 == 0:
        print(loss)
        #im = pos_to_im3(x_float, size, False).detach()
        #plt.imshow(im.squeeze().numpy())
        #plt.show(); plt.pause(.01)
#        print(count, loss, 'using time (sec):' , time()-time0)
#        time0 = time()
    return  loss.cpu().item(), np.asarray(grad_err.reshape(3*x_float.size(0)).cpu().numpy(), dtype=np.float64)

#float(loss)
def callback_print(x):
    return
'''
We initialize with iid uniform positions, and random std.
'''
x_pos = torch.torch.Tensor(nb_points, 2).uniform_(0, res)
x_sigmas = 1 + torch.Tensor(nb_points, 1).normal_(std=.1)
x = torch.cat((x_pos, x_sigmas), dim=-1)
x0 = x.reshape(3*x.size(0)).numpy()
x0 = np.asarray(x0, dtype=np.float64)

for start in range(nb_restarts + 1):
    if start == 0:
        x_opt = x0
    result = opt.minimize(fun_and_grad_conv, x_opt, method='L-BFGS-B', jac=True, tol=None,
                       callback=callback_print,
                       options={'maxiter': 1000, 'gtol': 1e-14, 'ftol': 1e-14, 'maxcor': 100
                                })
    final_loss, x_opt, niter, msg = result['fun'], result['x'], result['nit'], result['message']
    print('OPT fini avec:', final_loss,niter,msg)

x_fin = (torch.reshape(torch.tensor(x_opt,dtype=torch.float),
                       (x_opt.shape[0]//3, 3)))
#x_fin[:,:2] = (x_fin[:,:2]%res)/res

torch.save(x_fin, './results/cosmobis4.pt')
pi = pi.cpu()
m = m.cpu()
im_fin = pos_to_im3(res*x_fin[:,:2], res, False, sigma=x_fin[:,2])
plt.imshow(im_fin.squeeze().numpy())
plt.show()
