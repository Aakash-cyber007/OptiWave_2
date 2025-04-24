from matplotlib.image import imread
import matplotlib.pyplot as plt
import numpy as np
import os
from math import log10
import math
import random
import Block_SVD as svd

def svd_bpso(path, p_n, n_iterations):
    # Load the image from the specified path
    X = imread(os.path.join(path))
    # Apply SVD decomposition to the image using the custom Block_SVD function
    U, S, VT = svd.Block_SVD(X)
    # Display the original image in grayscale
    img = plt.imshow(X)
    img.set_cmap('gray')
    plt.axis('off')
    plt.show()
    
    # Initialize PSO algorithm parameters
    c1 = 2 # Cognitive learning factor
    c2 = 2 # Social learning factor
    r1 = random.uniform(0,1) # Random factor for cognitive component
    r2 = random.uniform(0,1) # Random factor for social component
    # Calculate the lower bound for k (number of singular valuesnto retain)
    # This is a heuristic based on the image dimensions
    k_end = int(max(X.shape[0],X.shape[1])/(int(min(X.shape[0],X.shape[1])**0.5)))
    
    def compressed_img(Xapr,k):
        # Function to create a compressed image using k singular values and calculate PSNR
        U_k = U[:, :k] # Take only the first k columns of U
        sig_k = sig[:k, :k] # Take only the first k singular values (should be S instead of sig)
        VT_k = VT[:k, :] # Take only the first k rows of VT
        
        # Reconstruct the image using matrix multiplication
        cmpd_img = np.dot(U_k , np.dot(sig_k , VT_k))
        # Calculate Mean Squared Error between original and compressed images
        MSE = ((X-cmpd_img)**2).mean(axis=None)
        # Calculate Peak Signal-to-Noise Ratio (PSNR) in decibels
        PSNR = 10*log10(((255)**2)/MSE)
        # Return positive PSNR value
        if PSNR>0:
            return PSNR
        return -PSNR
    # Define the range of possible k values to explore
    O = list(range(k_end,k_end+100))
    # Initialize global best position randomly
    gbest = random.choice(O)
    k= gbest
    
    def fitness(X,k):
        # Function to calculate fitness for a given k value
        # Higher fitness indicates better compression performance
        k = k
        Xapr = X
        # Get PSNR for the compressed image
        PSNR=compressed_img(Xapr,k)
        # Calculate compression ratio
        compression_ratio = k*(1+X.shape[0] + X.shape[1])/(X.shape[0]*X.shape[1])
        # Calculate fitness as a weighted combination of PSNR and inverse compression ratio
        fitness = ((PSNR) +2/compression_ratio)*10
        return fitness
    
    class particles:
        def _init_(self,bounds):
            # Initialize particle with random position within bounds
            # Note: Constructor should be __init__ with double underscores
            bounds = (k_end,k_end+100)  # This overwrites the input parameter
            self.position =random.randint(bounds[0],bounds[1])
            self.velocity = 0
            self.best_position = self.position
    # Set bounds for particle positions
    bounds = (k_end,k_end+100)
    # Initialize swarm of particles
    swarm = [particles(bounds) for _ in range(p_n)]
    
    # Main PSO optimization loop
    for t in range(n_iterations):
        for i in swarm:
            # Calculate sum of all particles' best positions
            sum_pbest=0
            for j in swarm:
                sum_pbest = sum_pbest+j.best_position
            # Evaluate current and personal best fitness
            current_fitness = fitness(X, i.position)
            pbest_fitness = fitness(X, i.best_position)
            # Update personal best if current position is better
            if current_fitness < pbest_fitness:
                i.best_position = i.position
            # Update global best if current position is better
            if current_fitness < fitness(X,gbest):
                gbest = i.position
            # Generate new random factors
            r1 = random.uniform(0,1)
            r2 = random.uniform(0,1)
            # Calculate particle probability
            P_t = gbest/sum_pbest
            # Update particle velocity
            # Note: Syntax error - missing multiplication operators (*)
            i.velocity = (1-t/n_iterations)i.velocity + (1-P_t)(math.exp(-1+t/n_iterations))r1*c1(i.best_position-i.position) + P_t*r2*c2*(gbest-i.position)
            # Update particle position
            i.position = i.position + int(math.floor(i.velocity+0.5))
            # Apply boundary constraints
            if i.position < bounds[0]:
                i.position = bounds[0]
            elif i.position > bounds[1]:
                i.position = bounds[1]
     # Generate the final compressed image using the optimal number of singular values
    
    Xcmp = np.dot(U[:,:gbest], np.dot(sig[:gbest, :gbest], VT[:gbest, :]))
    # Should use S instead of sig

    # Display the compressed image
    img = plt.imshow(Xcmp)
    img.set_cmap('gray')
    plt.axis('off')
    plt.show()
    # Get filename from user and save the compressed image
    final = input('Enter the name for the compressed image whcih you wish to save')
    return plt.imsave(final,Xcmp,cmap='gray')
