 

ef euler_GBM(number_of_assets, initial_values, num_steps, T, means, std_devs, cov_matrix, ant_variates = False): 
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



def milstein_GBM(number_of_assets, initial_values, num_steps, T, means, std_devs, cov_matrix): 
    delta_t = T/num_steps 
    S = np.ones((number_of_assets, num_steps +1)) 
    S[:,0] = initial_values 

    if ant_variates: 
        ant_S = np.ones((number_of_assets, num_steps+1))
        ant_S[:,0] = initial_values
        
    W = math.sqrt(delta_t) *np.random.multivariate_normal(np.zeros(number_of_assets), cov_matrix, num_steps) 
    for i in range(num_steps): 
        S[:,i+1] = S[:,i] + means*S[:,i]*delta_t + std_devs*S[:,i]*W[i,:] + 1/2 * std_devs**2 * S[:,i]*(W[i,:]**2 - delta_t) 

    if ant_variates: 
        for i in range(num_steps): 
            ant_S[:,i+1] = ant_S[:,i] + means*ant_S[:,i] + delta_t + std_devs*ant_S[:,i]*(-W[i,:]) 
        return (S, ant_S) 
        
    return S
