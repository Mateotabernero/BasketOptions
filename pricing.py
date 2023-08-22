import numpy as np 
import math 
import main 


# This prices basket options given the weights of assets 
def price (number_of_assets, weights, initial_values, num_steps, T, means, std_devs, cov_matrix, payOff, num_simulations = 10000): 
    vPayOff = np.vectorize(payOff) 

    values = np.zeros(num_simulations) 
    
    for i in range(num_simulations):
        S = main.GBM(number_of_assets, initial_values, num_steps, T, means, std_devs, cov_matrix) 
        values[i] = np.dot(vPayOff(S[:, num_steps]), weights) 
     
    V = np.mean(values)

    
