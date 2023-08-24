import numpy as np 
import math 
import main 
import paths

"""
Problemas y cosas que revolver. 

Uno de los problemas es el de la función, habrá que cogerlo por tomar como uno de los argumentos los strike price y que tipo son y luego definir la función externa de payOff

Mirar más variance reduction methods (control variates pinta bien pero parece complicado) 
"""


def price (number_of_assets, weights, initial_values, num_steps, T, means, std_devs, cov_matrix, payOffs, num_simulations = 10000, ant_variates = False, integration_method = 'E'): 

    # We first get the function corresponding to the integration_method 

    if integration_method == 'E': 
        integration_method =paths.euler_GBM
    
    elif integration_method == 'M':
        integration_method = paths.milstein_GBM
    
    elif integration_method == 'RK': 
        integration_method = paths.rungeKutta_GBM 
    
    else:
        raise ValueError("Please choose an appropiate integration method (Euler ('E'), Milstein ('M') or Runge-Kutta('RK')")


    values = np.zeros(num_simulations) 
    if ant_variates: 
        ant_values = np.zeros(num_simulations) 
    for i in range(num_simulations):
        if i%1000 == 0:
            print(i)
        if not ant_variates:
            S = integration_method(number_of_assets, initial_values, num_steps, T, means, std_devs, cov_matrix) 
            values[i] = np.dot(np.vectorize(lambda f, v: f(v))(payOffs, S[:,-1]), weights) 
        else: 
            S, ant_S =  integration_method(number_of_assets, initial_values, num_steps, T, means, std_devs, cov_matrix) 
            
            values[i] = np.dot(np.vectorize(lambda f, v: f(v))(payOffs, S[:,-1]), weights)
            ant_values[i] = np.dot(np.vectorize(lambda f, v: f(v))(payOffs, ant_S[:,-1]), weights)

    if not ant_variates: 
        V = np.mean(values)   
    else:
        V = 1/2*(np.mean(values) + np.mean(ant_values)) 
    
    return V
