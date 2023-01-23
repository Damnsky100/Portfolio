import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import optimize
import scipy.optimize as sco



## Without risk-free asset :

# Determination of the maximum return portfolio (undercertain constraintes) :

def Max_return_ptf(Initial_weights,bounds_max_return,constraints_max_ret,Industrie_selected,E_return):
    
# Define objective fonction to minimize :
    def Max_return_obj (weigth,E_return = E_return.values):
        return - np.dot(weigth.T,E_return)
# Compute the maximum return portfolio by optimisation under constraints: 

    Max_ret_PTF = sco.minimize(

      fun = Max_return_obj, 
      x0 = Initial_weights, 
      method = 'SLSQP',
      bounds = bounds_max_return, 
      constraints = constraints_max_ret
    
    )

    Max_ret_weigth = pd.DataFrame(Max_ret_PTF['x'],index=Industrie_selected,columns= ['Weigth'])

    Max_ret_return = pd.DataFrame([np.dot(Max_ret_PTF['x'].T,E_return.values)],columns= ['return'])
    
    Max_return = dict([('Max_ret_return',Max_ret_return),
                       ('Max_ret_weigth',Max_ret_weigth.T)])
    return(Max_return)

##### Determination of the minimum-variance  portfolio  :

def Min_var_ptf(Initial_weights,bounds,constraints_min_var,Industrie_selected,E_cov,E_return):
    
    
    # Define objective fonction to minimize :

    def Min_Variance_obj (weigth,Cov_matrix = E_cov) : 
        return (0.5*np.dot(np.dot(weigth.T,Cov_matrix),weigth))
# Compute the minimum variance portfolio by optimisation under constraints: 

    Min_var_PTF = sco.minimize(

      fun = Min_Variance_obj, 
      x0 = Initial_weights, 
      method = 'SLSQP',
      constraints = constraints_min_var,
      bounds = bounds
    
    )

    Min_var_weigth = pd.DataFrame(Min_var_PTF['x'],index=Industrie_selected,columns= ['Weigth'])

    Min_var_return = pd.DataFrame([np.dot(Min_var_PTF['x'].T,E_return.values)],columns= ['return'])
    
    Min_var = dict([('Min_var_return',Min_var_return),
                       ('Min_var_weigth',Min_var_weigth.T)])
    return(Min_var)

##### Determination of the optimal  portfolio  for a given return  :

def Ptf_target_optimization(Target_return,Initial_weights,bounds,Nbr_PTF,Var_level,Weigth_level,Weigth_names,E_cov,E_return):
    
    # Define objective fonction to minimize :

    def Min_Variance_obj (weigth,Cov_matrix = E_cov) : 
        return (0.5*np.dot(np.dot(weigth.T,Cov_matrix),weigth))
# Optimization for each return target :

    constraints = (
        {'type': 'eq', 'fun': lambda x: np.sum(x*E_return.values) - target}, # Optimization for a target return 
        {'type': 'eq', 'fun': lambda x: np.sum(x) - 1}
    )

    for i in range(0,Nbr_PTF):
        target = Target_return[i]
        Efficient_frontier_result =  sco.minimize(
            fun = Min_Variance_obj, 
            x0 = Initial_weights, 
            method = 'SLSQP', 
            constraints = constraints,
            bounds = bounds
        ) 
        
        Weigth_level[:,i] = Efficient_frontier_result ['x']
        Var_level[0,i] = np.sqrt(2*Efficient_frontier_result['fun'])
    
    Return_effectiv= np.dot(E_return.values,Weigth_level)
    Efficient_frontiere = np.concatenate((np.array([Target_return]),Var_level))
    
    # Data formatting 
    
    Efficient_frontiere_df = pd.DataFrame(Efficient_frontiere.T,columns=['Returns','Standard_deviation'])
    Efficient_frontiere_df.set_index('Standard_deviation', inplace=True)
    
    Efficient_frontiere_weigth = pd.DataFrame(Weigth_level.T,columns=Weigth_names)
    
    Efficient_frontiere_result = dict([('Efficient_frontiere',Efficient_frontiere_df),
                                           ('Efficient_frontiere_weigth',Efficient_frontiere_weigth)])
    
    return(Efficient_frontiere_result)



## With risk-free asset :

def Ptf_target_optimization_W_Rf(Target_return,Initial_weights,Nbr_PTF,Var_level_W_O_S,Weigth_level_W_O_S,E_cov,E_return,Expected_Risk_free,Weigth_names,bounds):
    
    # Define objective fonction to minimize :

    def Min_Variance_obj (weigth,Cov_matrix = E_cov) : 
        return (0.5*np.dot(np.dot(weigth.T,Cov_matrix),weigth))
# Constraint :

    constraints_W_RF = (
          {'type': 'eq', 'fun': lambda x: np.sum(x*(E_return.values-Expected_Risk_free)) - (target-Expected_Risk_free)}) # Optimization for a target return 

    for i in range(0,Nbr_PTF):
        target = Target_return[i]
        Efficient_frontier_r =  sco.minimize(
            fun = Min_Variance_obj, 
            x0 = Initial_weights, 
            method = 'SLSQP', 
            constraints = constraints_W_RF,
            bounds = bounds
        ) 
        Var_level_W_O_S[0,i] = np.sqrt(2*Efficient_frontier_r['fun'])
        Weigth_level_W_O_S[1:6,i] = Efficient_frontier_r['x']
        Weigth_level_W_O_S[0,i] = 1-np.sum(Efficient_frontier_r['x'])
    
    Expected_return_W_RF= np.concatenate((np.array([Expected_Risk_free]),E_return))
    Return_effectiv= np.dot(Expected_return_W_RF,Weigth_level_W_O_S)
    
    Efficient_frontiere = np.concatenate((np.array([Return_effectiv]),Var_level_W_O_S))
    
    # Data formatting 
    
    Efficient_frontiere_df = pd.DataFrame(Efficient_frontiere.T,columns=['Returns','Standard_deviation'])
    Efficient_frontiere_df.set_index('Standard_deviation', inplace=True)
    Efficient_frontiere_weigth = pd.DataFrame(Weigth_level_W_O_S.T,columns=Weigth_names)
    
    Efficient_frontiere_result = dict([('Efficient_frontiere',Efficient_frontiere_df),
                                           ('Efficient_frontiere_weigth',Efficient_frontiere_weigth)])
    
    return(Efficient_frontiere_result)
