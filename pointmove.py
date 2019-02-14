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
import torch.nn.functional as F

import sys
sys.path.append('/users/trec/brochard/kymatio_wph')



def weight(index, pos, res, gpu, sigmasq):
    if gpu:
        index = torch.tensor(index).cuda()
        pos = pos.cuda()
        res=torch.tensor(res).type(torch.float).cuda()
    #w = F.relu(1 - torch.abs(-index + pos))
    v = torch.min(torch.min(torch.abs(pos-index), torch.abs(pos+res-index)),
                  torch.abs(pos-res-index))
    w = torch.exp(-torch.mul(v,v)/sigmasq)
    return w


def pos_to_im3(x, res, gpu, sigmasq):
    Mx = torch.arange(0, res).type(torch.float)
    My = torch.arange(0, res).type(torch.float)
    if gpu:
        Mx = Mx.cuda(); My = My.cuda()
    im_x = weight(Mx.unsqueeze(0), x[:, 0].unsqueeze(1), res, gpu, sigmasq).unsqueeze(2)
    im_y = weight(My.unsqueeze(0), x[:, 1].unsqueeze(1), res, gpu, sigmasq).unsqueeze(1)
    #print(im_x.shape, im_y.shape)
    M = torch.matmul(im_x, im_y).sum(0)
    #print(M.shape)
    return M.unsqueeze(0).unsqueeze(0)


# load image

size = 128
res=128
gpu = True
sigmasq=2

filename = '/users/trec/brochard/kymatio_wph/data/poissoncircle_70_25_35.txt'
#filename = '/users/trec/brochard/kymatio_wph/poissonline3.txt'
pos = size*np.loadtxt(fname=filename, delimiter=',')
#pos = 128*np.loadtxt(fname=filename, delimiter=',', skiprows=1, usecols=(1,2))
nb_points = pos.shape[0]
pos = torch.from_numpy(pos).type(torch.float)
im = pos_to_im3(pos, res, gpu, sigmasq)
#plt.imshow(im.cpu().squeeze().numpy())
#plt.show()

#size = 64
#gpu = True
#im = torch.load('dot_circle_64.pt').cuda()

#x = torch.tensor([[15+20*np.cos(2*np.pi/12*theta),15+20*np.sin(2*np.pi/12*theta)] for theta in range(12)]).type(torch.float)
#im = pos_to_im3(x, size, gpu)
#plt.imshow(im.cpu().squeeze().numpy())
#plt.show()
#nb_points=12

# Parameters for transforms

J = 7
L = 8
M, N = im.shape[-2], im.shape[-1]
delta_j = 4
delta_l = L/2
delta_k = 1
nb_chunks = 2
nb_restarts = 0


# kymatio scattering
from kymatio.phaseharmonics2d.phase_harmonics_k_bump_chunkid_simplephase \
    import PhaseHarmonics2d

Sims = []
factr = 1e-3
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
    im = pos_to_im3(x, size, gpu, sigmasq)
    p = wph_op(im)*factr
    diff = p-Sims[chunk_id]
    loss = torch.mul(diff,diff).sum()/nCov
    return loss

grad_err = torch.zeros(nb_points, 2)

def grad_obj_fun(x_gpu):
    loss = 0
    global grad_err
    grad_err[:] = 0
    #global wph_ops
    for chunk_id in range(nb_chunks+1):
        x_t = x_gpu.clone().requires_grad_(True)
        #print('chunk_id in grad', chunk_id)
        #if chunk_id not in wph_ops.keys():
        #    wph_op = PhaseHarmonics2d(M, N, J, L, delta_j, delta_l, delta_k, nb_chunks, chunk_id)
        #    wph_op = wph_op.cuda()
        #    wph_ops[chunk_id] = wph_op
        loss_t = obj_fun(x_t,chunk_id)
        grad_err_t, = grad([loss_t],[x_t], retain_graph=False)
        loss = loss + loss_t
        grad_err = grad_err + grad_err_t
        #x_t.detach()
        #del x_t
        #del grad_err_
        #del wph_ops[chunk_id]
        #gc.collect()

    return loss, grad_err

count = 0
from time import time
time0 = time()
def fun_and_grad_conv(x):
    x_float = torch.reshape(torch.tensor(x, requires_grad=True,dtype=torch.float),
                            (x.shape[0]//2,2))
    loss, grad_err = grad_obj_fun(x_float)
    #x_float = torch.reshape(torch.tensor(x,dtype=torch.float),(1,1,size,size))
    #x_gpu = x_float.cuda()#.requires_grad_(True)
    #loss, grad_err = grad_obj_fun(x_t)
    #del x_gpu
    #gc.collect()
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
    return  loss.cpu().item(), np.asarray(grad_err.reshape(2*x_float.size(0)).cpu().numpy(), dtype=np.float64)

#float(loss)
def callback_print(x):
    return

x = torch.torch.Tensor(nb_points ,2).uniform_(0,size)
#plt.imshow(pos_to_im3(x, size, False).squeeze().numpy())
#plt.show()
x0 = x.reshape(2*x.size(0)).numpy()
x0 = np.asarray(x0, dtype=np.float64)

for start in range(nb_restarts + 1):
    if start == 0:
        x_opt = x0
    res = opt.minimize(fun_and_grad_conv, x0, method='L-BFGS-B', jac=True, tol=None,
                       callback=callback_print,
                       options={'maxiter': 2500, 'gtol': 1e-14, 'ftol': 1e-14, 'maxcor': 20
                                })
    final_loss, x_opt, niter, msg = res['fun'], res['x'], res['nit'], res['message']
    print('OPT fini avec:', final_loss,niter,msg)


x_fin = torch.reshape(torch.tensor(x_opt,dtype=torch.float),
                      (x_opt.shape[0]//2,2))
#torch.save(x_fin, 'poissonline3_pos.pt')
torch.save(x_fin, 'poissoncircle_70_25_35_pos_s128_J7.pt')
#im = pos_to_im3(x_fin, size, False)
#plt.imshow(im.squeeze().numpy())
#plt.show()
