import numpy as np 
import math 
import main 

"""
Problemas y cosas que revolver. 

Uno de los problemas es el de la función, habrá que cogerlo por tomar como uno de los argumentos los strike price y que tipo son y luego definir la función externa de payOff

Mirar más variance reduction methods (control variates pinta bien pero parece complicado) 
"""

def euler_GBM(number_of_assets, initial_values, num_steps, T, means, std_devs, cov_matrix, ant_variates = False): 
    delta_t = T/num_steps 
    S = np.ones((number_of_assets, num_steps +1)) 
    S[:,0] = initial_values
    if ant_variates: 
        ant_S = np.ones((number_of_assets, num_steps+1))
        ant_S[:,0] = initial_values
        
    W = math.sqrt(delta_t)* np.random.multivariate_normal(np.zeros(number_of_assets), cov_matrix, num_steps) 
    for i in range(num_steps): 
        S[:,i+1] = S[:,i] + means * S[:, i] * delta_t + std_devs * S[:,i]*W[i,:]

    if ant_variates: 
        for i in range(num_steps): 
            ant_S[:,i+1] = ant_S[:,i] + means*ant_S[:,i] + delta_t + std_devs*ant_S[:,i]*(-W[i,:]) 
        return (S, ant_S) 
    return S 
    

def milstein_GBM(number_of_assets, initial_values, num_steps, T, meansm std_devs, cov_matrix): 
    delta_t = T/num_steps 
    S = np.ones((number_of_assets, num_steps +1)) 
    S[:,0] = initial_values 
    W = math.sqrt(delta_t) *np.random.multivariate_normal(np.zeros(number_of_assets), cov_matrix, num_steps) 
    for i in range(num_steps): 
        S[:,i+1] = S[:,i] + means*S[:,i]*delta_t + std_devs*S[:,i]*W[i,:] + 1/2 * std_devs**2 * S[:,i]*(W[i,:]**2 - delta_t) 
    
    return S


def price (number_of_assets, weights, initial_values, num_steps, T, means, std_devs, cov_matrix, payOff, num_simulations = 10000, ant_variates = False, integration_method = 'E'): 

    values = np.zeros(num_simulations) 
    if ant_variates: 
        ant_values = np.zeros(num_simulations) 
    for _ in range(num_simulations):
        
        if not ant_variates:
            S = integration_method(number_of_assets, initial_values, num_steps, T, means, std_devs, cov_matrix) 
            values[i] = np.dot(vPayOff(S[:, num_steps]), weights) 
        else: 
            S, ant_S =  integration_method(number_of_assets, initial_values, num_steps, T, means, std_devs, cov_matrix) 
            values[i] = np.dot(np.vectorize(lambda f, v: f(v))(function_array, S[:, -1]), weights)
            ant_values[i] = np.dot(np.vectorize(lambda f, v: f(v))(function_array, ant_S[:, -1]), weights)

    if not ant_variates: 
        V = np.mean(values)   
    else:
        V = 1/2*(np.mean(values) + np.mean(ant_values)) 
    
