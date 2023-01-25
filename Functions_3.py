import importlib
import numpy as np
import Functions as f
importlib.reload(f)

import numpy as np
import pandas as pd
from scipy import optimize
import scipy.optimize as sco

def msr_3(risk_free_rate, er, cov, bounds):
    """
    Returns the weights of the portfolio that gives you the maximum sharpe ratio given 
    the riskfree rate and expected returns and a covariance matrix
    
    """
    n = er.shape[0] #determine the number of assets
    init_guess = np.repeat(1/n, n) #Initial weight vector is equally distributed
    
    n_sharp = 10
    
    weights_sum_to_1 = {
        "type":"eq",         #{constraints} eq = equalize to 0
        "fun": lambda weights: np.sum(weights) - 1
    }
    def neg_sharpe_ratio(weights, risk_free_rate, er, cov):
        """
        Returns the negative of the sharpe ratio, given weights
        
        """
        r = f.portfolio_return(weights, er)
        vol = f.portfolio_vol(weights, cov)
        return  -(r-risk_free_rate) / vol
        
    for i in range(len(bounds)): 
        results = sco.minimize(neg_sharpe_ratio, init_guess,
                        args = (risk_free_rate, er, cov,), method="SLSQP", 
                        options= {"disp": False},
                        constraints=(weights_sum_to_1),
                        bounds = bounds[i]
                        )
        w = results.x
        if neg_sharpe_ratio(w, risk_free_rate, er, cov) <  n_sharp:
            w_max = w 
        
    return w_max



def Max_return_ptf_3(E_return, bounds):
      
    #Initialize variables
    
    Initial_weights = np.array([1 / len(E_return)] * len(E_return))
    ret_m = 0
    Industrie_selected = E_return.index
    constraints_max_ret = ( {'type': 'eq', 'fun': lambda x: np.sum(x) - 1}) # Sum of the weigth need to sum to 1
    
# Define objective fonction to minimize :
    def Max_return_obj (weigth, E_return = E_return.values):
        return - np.dot(weigth.T, E_return)
    
# Compute the maximum return portfolio by optimisation under constraints: 
    for i in range(len(bounds)):
        Max_ret_PTF = sco.minimize(

        fun = Max_return_obj, 
        x0 = Initial_weights, 
        method = 'SLSQP',
        bounds = bounds[i], 
        constraints = constraints_max_ret
        
        )
        w = Max_ret_PTF['x']
        ret = f.portfolio_return(w, E_return)
        if ret > ret_m:
            ret_m = ret
            w_max = w

    Max_ret_weigth = pd.DataFrame(w, index=Industrie_selected,columns = ['Weigth'])

    Max_ret_return = pd.DataFrame([ret_m], columns = ['return'])
    
    Max_return = {'Max_ret_return' : Max_ret_return,
                       'Max_ret_weigth' : Max_ret_weigth.T}
    return(Max_return)



#Vieille fonction pour trouver à chaque target ret le weight qui maximise le sharp ratio


def Ptf_target_optimization_3(E_return, E_cov, Nbr_PTF, bounds):
    
    #Define Variables
    
  
    
    Initial_weights = np.array([1 / len(E_return)] * len(E_return))
    n = E_return.shape[0]
    
    w = np.zeros((1, n))
    w_max = np.zeros((1, n))
    
    
    Min_var_result = f.Min_var_ptf(E_return, E_cov, bounds[0][0]) #Declare the right initial bound (0,1) or short allowed (-2,2)
    Max_return_result = Max_return_ptf_3(E_return, bounds)
    Target_return = np.linspace (Min_var_result['Min_var_return'].iloc[0]['return'], Max_return_result['Max_ret_return'].iloc[0]['return'], Nbr_PTF)
    
    Weigth_names = E_return.index
    Weigth_level = np.zeros((len(E_return), Nbr_PTF)) #Initialize the array
    Var_level = np.zeros((1, Nbr_PTF))
    
    
    
    constraints = (
        {'type': 'eq', 'fun': lambda x: np.sum(x*E_return.values) - target}, # Optimization for a target return 
        {'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
    
    
    # Define objective fonction to minimize :

    def Min_Variance_obj (weigth, Cov_matrix = E_cov) : 
        return (0.5*np.dot(np.dot(weigth.T, Cov_matrix), weigth))
    
    # Optimization for each return target :

    

    for i in range(0,Nbr_PTF):
        m_vol = 100
        m_vol2 = 100
        m_sum = 0
        for j in range(len(bounds)):
            
            target = Target_return[i]
            Efficient_frontier_result =  sco.minimize(
                fun = Min_Variance_obj, 
                x0 = Initial_weights, 
                method = 'SLSQP', 
                constraints = constraints,
                bounds = bounds[j]
            ) 
            
            
            w = np.round(Efficient_frontier_result ['x'], 2)
            sum = np.sum(w)
            
            vol = np.sqrt(2*Efficient_frontier_result['fun'])
            
            if  (vol < m_vol) and (np.sum(w) == 1) : #2 Conditions : 1) to see if the combinations of the 3 assets is better than the previous ones, 2) We wants the weights to sum to one
                w_max = w
                m_vol = vol
            elif m_vol == 100 : #Donc la somme n'égale pas 1 des weights
                if(sum > m_sum): #On sélectionne celui qui a la plus grand sum
                    w_max_tmp = w
                    m_vol2 = vol
                
        if m_vol == 100: #Si aucun np.sum(w) == 1, on selectionne celui qui minimise la variance
            w_max = w_max_tmp
            m_vol = m_vol2
                
            
        Weigth_level[:,i] = w_max
        Var_level[0,i] = m_vol
    
    Return_effectiv = np.dot(E_return.values, Weigth_level)
    Efficient_frontiere = np.concatenate((np.array([Target_return]), Var_level))
    
    # Data formatting 
    
    Efficient_frontiere_df = pd.DataFrame(Efficient_frontiere.T, columns=['Returns', 'Standard_deviation'])
    Efficient_frontiere_df.set_index('Standard_deviation', inplace=True)
    
    Efficient_frontiere_weigth = pd.DataFrame(Weigth_level.T, columns=Weigth_names)
    
    Efficient_frontiere_result = {'Efficient_frontiere': Efficient_frontiere_df,
                                           'Efficient_frontiere_weigth': Efficient_frontiere_weigth}
    
    return(Efficient_frontiere_result)






def Ptf_target_optimization_W_Rf_3(E_return, E_cov, Expected_Risk_free, Nbr_PTF, bounds):
    
     #Define Variables
    
    Max_return_result = Max_return_ptf_3(E_return, bounds)
    
    Initial_weights = np.array([1 / len(E_return)] * len(E_return))
    
    w_max = np.zeros((1, E_return.shape[0]))
    
    Target_return = np.linspace(Expected_Risk_free, Max_return_result['Max_ret_return'].iloc[0]['return'], Nbr_PTF)
    Weigth_names =  np.concatenate((np.array(['Risk_free']) , E_return.index.values))
    
    Var_level_W_O_S = np.zeros((1, Nbr_PTF))
    Weigth_level_W_O_S = np.zeros((len(E_return) + 1 , Nbr_PTF)) #Initialize the array
    
    
    
    # Define objective fonction to minimize :

    def Min_Variance_obj (weigth, Cov_matrix = E_cov) : 
        return (0.5*np.dot(np.dot(weigth.T, Cov_matrix), weigth))
    # Constraint :

    constraints_W_RF = (
          {'type': 'eq', 'fun': lambda x: np.sum(x*(E_return.values-Expected_Risk_free)) - (target-Expected_Risk_free)}) # Optimization for a target return 


    for i in range(0,Nbr_PTF):
        m_vol = 100
        for j in range(len(bounds)):
            target = Target_return[i]
            Efficient_frontier_r =  sco.minimize(
                fun = Min_Variance_obj, 
                x0 = Initial_weights, 
                method = 'SLSQP', 
                constraints = constraints_W_RF,
                bounds = bounds[j]
            ) 
            
            w = np.round(Efficient_frontier_r ['x'], 2)
            sum = np.sum(w)
            
            vol = np.sqrt(2*Efficient_frontier_r['fun'])
            
            if  (vol < m_vol) : #2 Conditions : 1) to see if the combinations of the 3 assets is better than the previous ones
                w_max = np.round(w, 4)
                m_vol = np.round(vol, 4)
           
            
            
        Var_level_W_O_S[0,i] = m_vol
        Weigth_level_W_O_S[1:(len(E_return) + 1), i] = w_max
        Weigth_level_W_O_S[0,i] = 1-np.sum(w_max)
    
    Expected_return_W_RF= np.concatenate((np.array([Expected_Risk_free]),E_return))
    Return_effectiv= np.round(np.dot(Expected_return_W_RF,Weigth_level_W_O_S), 4)
    
    Efficient_frontiere = np.concatenate((np.array([Return_effectiv]),Var_level_W_O_S))
    
    # Data formatting 
    
    Efficient_frontiere_df = pd.DataFrame(Efficient_frontiere.T,columns=['Returns','Standard_deviation'])
    Efficient_frontiere_df.set_index('Standard_deviation', inplace=True)
    Efficient_frontiere_weigth = pd.DataFrame(Weigth_level_W_O_S.T,columns=Weigth_names)
    
    Efficient_frontiere_result = dict([('Efficient_frontiere',Efficient_frontiere_df),
                                           ('Efficient_frontiere_weigth',Efficient_frontiere_weigth)])
    
    return(Efficient_frontiere_result)




"""
def find_tangent_3(w, E_return,  E_cov,  Expected_Risk_free):
    n = len(w)
    
    
    sharp_m = 0
    
    for i in range(n):
        w_tmp = np.array(w.iloc[i,1:])
        ret = f.portfolio_return(w_tmp, np.array(E_return)) 
        vol = f.portfolio_vol(w_tmp, E_cov)
        sharp = (ret - Expected_Risk_free)/vol
        if sharp > sharp_m:
            sharp_m = sharp
        
            ret_max = ret
            vol_max = vol
    
    return ret_max, vol_max"""









def plot_ef_3(E_return, E_cov, Expected_Risk_free, Nbr_PTF, bounds,  show_cml=False, show_gmv=False):
    """Function to plot Efficient Frontier

    Args:
        E_return (Array: Nx1): Annualized return
        E_cov (NxN): variance-covariance matrix
        Expected_Risk_free (int): Annualized Risk-Free rate
        Nbr_PTF (int): Number of points for your graph
        bounds (tuple, optional): _description
    """
    n = E_return.shape[0]
    Efficient_frontiere_result = Ptf_target_optimization_3(E_return, E_cov, Nbr_PTF,
                                                         bounds) #Without risk-free asset
    
    #Start by plotting result of efficient frontier without rf
    plt = Efficient_frontiere_result['Efficient_frontiere']['Returns'].plot(kind='line',figsize=(15,11), 
                                                                        xlim = [0, max(Efficient_frontiere_result['Efficient_frontiere'].index)+0.5], 
                                                                        ylim = [0, max(Efficient_frontiere_result['Efficient_frontiere']['Returns'])+0.2]) 
    if show_cml :
        
        
        
        Efficient_frontiere_result_W_RF = Ptf_target_optimization_W_Rf_3(E_return, E_cov,
                                                                   Expected_Risk_free, Nbr_PTF,
                                                                   bounds)
        
        
        
        Efficient_frontiere_result_W_RF['Efficient_frontiere']['Returns'].plot(kind='line')
        #With risk-free Assets
        
        
        #Plot Portfolio Tangente
        w = msr_3(Expected_Risk_free, E_return, E_cov, bounds)
        y = f.portfolio_return(w, E_return)
        x = f.portfolio_vol(w, E_cov)
        plt.scatter(x ,y, marker="o")
        plt.annotate("Pf_t",(x,  y))
        plt.legend(['Wihtout RF', 'With RF'])
        
    for i in E_return.index :
            plt.scatter([np.sqrt(E_cov[i][i])],[E_return[i]], marker='o')
            plt.annotate(i,(np.sqrt(E_cov[i][i]),E_return[i]))      
          
    #Plot GMV
    if show_gmv:
        w = msr_3(0, np.repeat(1, n), E_cov, bounds)
        y = f.portfolio_return(w, E_return)
        x =  f.portfolio_vol(w, E_cov)   
        plt.scatter(x,y, marker="o")
        plt.annotate("GMV",(x,  y))
    
    #Set Titles 
    plt.set_title("Efficient Frontier")
    plt.set_xlabel("Volatility (Std)")
    plt.set_ylabel("Return Annualized")
    
    return plt