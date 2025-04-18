from matplotlib.image import imread
import matplotlib.pyplot as plt
import numpy as np
import os
from math import log10
import math
import random
import Block_SVD as svd

def svd_bpso(path, p_n, n_iterations):
    X = imread(os.path.join(path))
    U, S, VT = svd.Block_SVD(X)
    img = plt.imshow(X)
    img.set_cmap('gray')
    plt.axis('off')
    plt.show()
    
    c1 = 2
    c2 = 2
    r1 = random.uniform(0,1)
    r2 = random.uniform(0,1)
    k_end = int(max(X.shape[0],X.shape[1])/(int(min(X.shape[0],X.shape[1])**0.5)))
    
    def compressed_img(Xapr,k):
        U_k = U[:, :k]
        sig_k = sig[:k, :k]
        VT_k = VT[:k, :]
        cmpd_img = np.dot(U_k , np.dot(sig_k , VT_k))
        MSE = ((X-cmpd_img)**2).mean(axis=None)
        PSNR = 10*log10(((255)**2)/MSE)
        if PSNR>0:
            return PSNR
        return -PSNR
    
    O = list(range(k_end,k_end+100))
    gbest = random.choice(O)
    k= gbest
    
    def fitness(X,k):
        k = k
        Xapr = X
        PSNR=compressed_img(Xapr,k)
        compression_ratio = k*(1+X.shape[0] + X.shape[1])/(X.shape[0]*X.shape[1])
        fitness = ((PSNR) +2/compression_ratio)*10
        return fitness
    
    class particles:
        def _init_(self,bounds):
            bounds = (k_end,k_end+100)
            self.position =random.randint(bounds[0],bounds[1])
            self.velocity = 0
            self.best_position = self.position
    
    bounds = (k_end,k_end+100)
    swarm = [particles(bounds) for _ in range(p_n)]
    
    for t in range(n_iterations):
        for i in swarm:
            sum_pbest=0
            for j in swarm:
                sum_pbest = sum_pbest+j.best_position
            current_fitness = fitness(X, i.position)
            pbest_fitness = fitness(X, i.best_position)
    
            if current_fitness < pbest_fitness:
                i.best_position = i.position
    
            if current_fitness < fitness(X,gbest):
                gbest = i.position
            r1 = random.uniform(0,1)
            r2 = random.uniform(0,1)
            P_t = gbest/sum_pbest
            i.velocity = (1-t/n_iterations)i.velocity + (1-P_t)(math.exp(-1+t/n_iterations))r1*c1(i.best_position-i.position) + P_t*r2*c2*(gbest-i.position)
            i.position = i.position + int(math.floor(i.velocity+0.5))
            if i.position < bounds[0]:
                i.position = bounds[0]
            elif i.position > bounds[1]:
                i.position = bounds[1]
    Xcmp = np.dot(U[:,:gbest], np.dot(sig[:gbest, :gbest], VT[:gbest, :]))
    img = plt.imshow(Xcmp)
    img.set_cmap('gray')
    plt.axis('off')
    plt.show()
    final = input('Enter the name for the compressed image whcih you wish to save')
    return plt.imsave(final,Xcmp,cmap='gray')