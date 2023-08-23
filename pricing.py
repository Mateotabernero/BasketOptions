import numpy as np 
import math 
import main 


def GBM(number_of_assets, initial_values, num_steps, T, means, std_devs, cov_matrix): 
    delta_t = T/num_steps 
    S = np.ones((number_of_assets, num_steps +1)) 
    S[:,0] = initial_values
    
    W = np.random.multivariate_normal(np.zeros(number_of_assets), cov_matrix, num_steps) 
    for i in range(num_steps): 
        S[:,i+1] = S[:,i] + means * S[:, i] * delta_t + std_devs * S[:,i]*math.sqrt(delta_t)*W[i,:]
    return S 
    

def price (number_of_assets, weights, initial_values, num_steps, T, means, std_devs, cov_matrix, payOff, num_simulations = 10000): 
    vPayOff = np.vectorize(payOff) 

    values = np.zeros(num_simulations) 
    
    for i in range(num_simulations):
        S = main.GBM(number_of_assets, initial_values, num_steps, T, means, std_devs, cov_matrix) 
        values[i] = np.dot(vPayOff(S[:, num_steps]), weights) 
     
    V = np.mean(values)

    
