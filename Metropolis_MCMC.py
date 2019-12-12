# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

# Prior belief
prior_mean = 7
prior_std = 3

# Likelihood belief
data_mean = 1
data_std = 1

#MCMC hyperparameters
N = 1000 #Number of iterations
N_chains = 10
std_proposal = 5

def get_candidate(loc):
    return np.random.normal(loc=loc, scale=std_proposal)

def prior(x):
    return norm.pdf(x,loc=prior_mean,scale=prior_std)

def likelihood(x):
    return norm.pdf(x,loc=data_mean,scale=data_std) + norm.pdf(x,loc=prior_mean,scale=prior_std)

def posterior(x):
    return likelihood(x)*prior(x)

def metropolis_chain(posterior, N):
    
    samples = []
    rejected = 0
    # Initialise the algorithm
    x_current = np.random.normal(loc=np.random.normal(),scale=np.abs(np.random.normal()))
    
    for i in range(N):
        samples.append(x_current)
        x_candidate = get_candidate(x_current)
        p_acceptance = (posterior(x_candidate))/(posterior(x_current))
        if p_acceptance > np.random.uniform():
            x_current = x_candidate
        else:
            rejected += 1
    return samples, rejected/N
        
def main():
    
    x_min = -5
    x_max = 15
    #Ploting the analytical solution
    fig, (ax1, ax2) = plt.subplots(nrows=2, ncols=1) # two axes on figure
    x = np.linspace(x_min,x_max,100)
    y = list(map(posterior,x))
    ax1.plot(x,y)
    ax1.set(xlim=(x_min,x_max))
    
    for n in range(N_chains):
        samples, rejection_ratio = metropolis_chain(posterior, N)
        ax2.hist(samples, density=True)
        print("Rejection ratio is ", rejection_ratio)
        ax2.set(xlim=(x_min,x_max))
    
main()
            
    