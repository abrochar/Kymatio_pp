import numpy as np
import matplotlib.pyplot as plt
import torch

#filename = './poissonline3_pos.pt'
filename = './poissoncircle_70_25_35_pos_s128_J7.pt'
res = 256

def weight(index, pos, res):
    print(index.shape, pos.shape)
    #w = F.relu(1 - torch.abs(-index + pos))
    v = torch.min(torch.min(torch.abs(pos-index), torch.abs(pos+res-index)),
                  torch.abs(pos-res-index))
    w = 1e3*torch.exp(-torch.mul(v,v)/2)
    return w


def pos_to_im3(x, res):
    Mx = torch.arange(0, res).type(torch.float)
    My = torch.arange(0, res).type(torch.float)
    im_x = weight(Mx.unsqueeze(0), x[:, 0].unsqueeze(1), res).unsqueeze(2)
    im_y = weight(My.unsqueeze(0), x[:, 1].unsqueeze(1), res).unsqueeze(1)
    #print(im_x.shape, im_y.shape)
    M = torch.matmul(im_x, im_y).sum(0)
    #print(M.shape)
    return M.unsqueeze(0).unsqueeze(0)



#pos0 = np.loadtxt('./poissonline3.txt', delimiter=',', usecols=(1,2), skiprows=1)
pos0 = np.loadtxt('./data/poissoncircle_70_25_35.txt', delimiter=',')
pos0 = res*torch.from_numpy(pos0).type(torch.float)
im0 = pos_to_im3(pos0, res)
center = np.random.randint(low=0, high=255-64, size=2)
#plt.imshow(im0.squeeze().numpy()[center[0]:center[0]+64, center[1]:center[1]+64])
plt.imshow(im0.squeeze().numpy())
plt.show()



pos = 2*torch.load(filename)


im = pos_to_im3(pos, res).squeeze().numpy()

plt.imshow(im)
plt.show()


def pos_to_pp(pos, res):
    M = torch.zeros(res, res)
    for i in range(pos.shape[0]):
        a = int(pos[i,0]%res)
        b = int(pos[i,1]%res)
        M[a, b] = 1
    return M


#im = pos_to_pp(pos, res)

#plt.imshow(im.numpy())
#plt.show()

