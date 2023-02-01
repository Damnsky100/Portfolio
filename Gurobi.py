import gurobipy as gp
from gurobipy import GRB, abs_
import numpy as np
import pandas as pd


##################### Function to help #####################

def annualize_rets(r, periods_per_year):
    """
    Annualizes a set of returns
    """
    compounded_growth = (1+r).prod()
    n_periods = r.shape[0]
    return compounded_growth**(periods_per_year/n_periods)-1



def annualize_vol(r, periods_per_year):
    """
    Annualizes the vol of a set of returns
    """
    return r.std()*(periods_per_year**0.5)



## Max return Function
def max_return_Gurobi(E_return, K, bounds = (-2,2)):
  """Fonction qui donne le return maximum selon la composition
  du portefeuille

  Args:
      E_return (_type_): Pd.Series --> Return des actifs Nx1
      K (_type_): Contrainte sur le nombre d'actifs
      bounds (tuple, optional): Contrainte sur les poids pour chaque actif.

  Returns:
      _type_: Dictionnary return + weight
  """
    #E_return must be a pandas.core.series.Series
  sampled_tickers = np.array(E_return.index)
    
  m_const = gp.Model("asset_constrained_model")
  
  #Déclare la variable weight
  w = pd.Series(m_const.addVars(sampled_tickers,
                  lb = bounds[0],
                  ub = bounds[1],          
                vtype = gp.GRB.CONTINUOUS),
                index =  sampled_tickers)
  
  #Variable pour le nombre d'actif (Valeur 1 ou 0)
  z = pd.Series(m_const.addVars(sampled_tickers, 
                vtype = gp.GRB.BINARY),
                index =  sampled_tickers)
    
  # Maximum number of assets constraint
  m_const.addConstr(sum(z[i_ticker] for i_ticker in sampled_tickers) <= K, "max_assets")
    
  #sum(w_i) == 1 (Long only)
  m_const.addConstr(w.sum() == 1, "port_budget")
  
  #Enlever les prints de recherche de fonction
  m_const.Params.LogToConsole = 0
  
  #Contrainte pour que abs(w) < z qui prend une valeur binaire
  for i_ticker in sampled_tickers:
      m_const.addConstr(w[i_ticker] <= z[i_ticker] * 2, f"w_{i_ticker}_positive")
      m_const.addConstr(-w[i_ticker] <= z[i_ticker] * 2, f"w_{i_ticker}_negative")
      
  # Set the objective function as the portfolio risk
  portfolio_risk = -E_return @ w
    
  m_const.setObjective(portfolio_risk, GRB.MINIMIZE)
  
  m_const.optimize()
    
  weigth = [np.round(w[i].x, 4) for i in range(len(sampled_tickers))]
    
    
  res = {"Return": -m_const.ObjVal, 
           "Weight": weigth}
    
  return res


## Min variance PF

def min_variance_Gurobi(E_return, E_cov, K, bounds = (-2,2)):
      
      
      
    #E_return must be a pandas.core.series.Series
    sampled_tickers = np.array(E_return.index)
    
  
    
    
    #Create the model
    m_const = gp.Model("asset_constrained_model")
    w = pd.Series(m_const.addVars(sampled_tickers,
                  lb = bounds[0],
                  ub = bounds[1],          
                vtype = gp.GRB.CONTINUOUS),
                index =  sampled_tickers)
    
    z = pd.Series(m_const.addVars(sampled_tickers, 
                vtype = gp.GRB.BINARY),
                index =  sampled_tickers)
    
    #Constraints
    m_const.addConstr(w.sum() == 1, "port_budget")
    m_const.Params.LogToConsole = 0
    m_const.addConstr(sum(z[i_ticker] for i_ticker in sampled_tickers) <= K, "max_assets")
    
    for i_ticker in sampled_tickers:
      m_const.addConstr(w[i_ticker] <= z[i_ticker] * 2, f"w_{i_ticker}_positive")
      m_const.addConstr(-w[i_ticker] <= z[i_ticker] * 2, f"w_{i_ticker}_negative")
      
      # Set the objective function as the portfolio risk
    portfolio_risk = w @ E_cov @ w
    m_const.setObjective(portfolio_risk, GRB.MINIMIZE)
    m_const.optimize()
    weigth = [np.round(w[i].x, 4) for i in range(len(sampled_tickers))]
    vol = (m_const.ObjVal)**0.5
    ret = E_return @ weigth
    
    res = {"Return": ret, 
           "Weight": weigth, 
           "Vol": vol}
    
    return res

 
 
 
##### Determination of the optimal  portfolio  for a given return  #####
def Ptf_target_optimization_Gurobi(E_return, E_cov, K, Nbr_PTF, bounds = (-2, 2)):
      
  
  sampled_tickers = np.array(E_return.index)
  n = E_return.shape[0]
      
  
  Weigth = np.zeros((Nbr_PTF, n)) #Initialize the array
  Var = np.zeros((1, Nbr_PTF))
  
  r = np.linspace(min_variance_Gurobi(E_return, E_cov, K, bounds)["Return"], max_return_Gurobi(E_return, K, bounds)["Return"], Nbr_PTF)
  

  for i in range(Nbr_PTF):
          
    m_const = gp.Model("asset_constrained_model")

    #w_i : i_th stock gets a weight w_i
    w = pd.Series(m_const.addVars(sampled_tickers,
                        lb = bounds[0],
                        ub = bounds[1],          
                      vtype = gp.GRB.CONTINUOUS),
                      index =  sampled_tickers)
    
    z = pd.Series(m_const.addVars(sampled_tickers, 
                      vtype = gp.GRB.BINARY),
                      index =  sampled_tickers)

    #sum(w_i) == 1 (Long only)
    m_const.addConstr(w.sum() == 1, "port_budget")
    
    m_const.Params.LogToConsole = 0


    # Maximum number of assets constraint
    m_const.addConstr(sum(z[i_ticker] for i_ticker in sampled_tickers) <= K, "max_assets")

      #Return Constrain
    portfolio_ret = E_return @ w
    portfolio_targ = r[i]
    m_const.addConstr(portfolio_ret == portfolio_targ, "return")

    for i_ticker in sampled_tickers:
              
        m_const.addConstr(w[i_ticker] <= z[i_ticker] * bounds[1], f"w_{i_ticker}_positive")
        m_const.addConstr(-w[i_ticker] <= z[i_ticker] * bounds[1], f"w_{i_ticker}_negative")


        # Set the objective function as the portfolio risk
    portfolio_risk = w @ E_cov @ w
    m_const.setObjective(portfolio_risk, GRB.MINIMIZE) 
        
    m_const.optimize()
        
    Weigth[i, :] = np.round((m_const.X)[:n] ,4)
    Var[0,i] = np.sqrt(m_const.ObjVal)
        

  Efficient_frontiere = np.concatenate((np.array([r]), Var))
  # Data formatting 
      
  Efficient_frontiere_df = pd.DataFrame(Efficient_frontiere.T, columns=['Returns', 'Standard_deviation'])
  Efficient_frontiere_df.set_index('Standard_deviation', inplace=True)
      
  Efficient_frontiere_weigth = pd.DataFrame(Weigth, columns=sampled_tickers)
      
  Efficient_frontiere_result = {'Efficient_frontiere': Efficient_frontiere_df,
                                            'Efficient_frontiere_weigth': Efficient_frontiere_weigth}
  return Efficient_frontiere_result



#################### With risk-free asset ####################


def Ptf_target_optimization_W_Rf_Gurobi(E_return, E_cov, Expected_Risk_free, K, Nbr_PTF, bounds = (-2, 2)):
      
  sampled_tickers = np.array(E_return.index)
  
  Weigth_names = np.concatenate((np.array(['Risk_free']), np.array(E_return.index)))
  
  n = E_return.shape[0]
      
  
  Weigth = np.zeros((Nbr_PTF, n + 1)) #Initialize the array
  Var = np.zeros((1, Nbr_PTF))
  
  r = np.linspace(Expected_Risk_free, max_return_Gurobi(E_return, K, bounds)["Return"], Nbr_PTF)
  

  for i in range(Nbr_PTF):
          
    m_const = gp.Model("asset_constrained_model")

    #w_i : i_th stock gets a weight w_i
    w = pd.Series(m_const.addVars(sampled_tickers,
                        lb = bounds[0],
                        ub = bounds[1],          
                      vtype = gp.GRB.CONTINUOUS),
                      index =  sampled_tickers)
    
    z = pd.Series(m_const.addVars(sampled_tickers, 
                      vtype = gp.GRB.BINARY),
                      index =  sampled_tickers)

    
    # Maximum number of assets constraint
    m_const.addConstr(sum(z[i_ticker] for i_ticker in sampled_tickers) <= K, "max_assets")
    m_const.Params.LogToConsole = 0

     #Return Constrain
    portfolio_ret = (E_return - Expected_Risk_free) @ w
    portfolio_targ = (r[i] - Expected_Risk_free)
    m_const.addConstr(portfolio_ret == portfolio_targ, "return")

    for i_ticker in sampled_tickers:
              
        m_const.addConstr(w[i_ticker] <= z[i_ticker] * bounds[1], f"w_{i_ticker}_positive")
        m_const.addConstr(-w[i_ticker] <= z[i_ticker] * bounds[1], f"w_{i_ticker}_negative")


        # Set the objective function as the portfolio risk
    portfolio_risk = w @ E_cov @ w
    m_const.setObjective(portfolio_risk, GRB.MINIMIZE) 
        
    m_const.optimize()
        
    Weigth[i, 1:(n + 1)] = np.round((m_const.X)[:n], 4)
    Weigth[i, 0] = np.round(1 - np.sum(Weigth[i, 1:(len(E_return) + 1)]), 4)
    Var[0,i] = np.round(np.sqrt(m_const.ObjVal), 4)
        

  Expected_return_W_RF= np.concatenate((np.array([Expected_Risk_free]), E_return))
  Return_effectiv= Expected_return_W_RF @ Weigth.T
    
  Efficient_frontiere = np.concatenate((np.array([Return_effectiv]), Var))
    
    # Data formatting 
    
  Efficient_frontiere_df = pd.DataFrame(Efficient_frontiere.T,columns=['Returns','Standard_deviation'])
  Efficient_frontiere_df.set_index('Standard_deviation', inplace=True)
  Efficient_frontiere_weigth = pd.DataFrame(Weigth, columns=Weigth_names)
    
  Efficient_frontiere_result = dict([('Efficient_frontiere',Efficient_frontiere_df),
                                           ('Efficient_frontiere_weigth',Efficient_frontiere_weigth)])
    
  return(Efficient_frontiere_result)




## Find Tangent Portfolio
def tangent_Gurobi(E_return, E_cov, Expected_Risk_free, K, Nbr_PTF, bounds = (-2, 2)):
    
    tmp1 = Ptf_target_optimization_Gurobi(E_return, E_cov, K, Nbr_PTF, bounds)["Efficient_frontiere_weigth"]
    tmp = Ptf_target_optimization_Gurobi(E_return, E_cov, K, Nbr_PTF, bounds)["Efficient_frontiere"]
    
    vol = np.array(tmp.index)
    ret = np.array(tmp["Returns"])
    

    
    frontier = pd.DataFrame(tmp1)
    frontier["Return"] = ret
    frontier["Volatility"] = vol

    frontier['Sharpe'] = (frontier['Return'] - Expected_Risk_free) / frontier['Volatility']
    idx = frontier['Sharpe'].max()
    sharpeMax = frontier.loc[frontier['Sharpe'] == idx]
    return sharpeMax




#################### Plotting ####################
def plot_ef_Gurobi(E_return, E_cov, Expected_Risk_free, K, Nbr_PTF, bounds, show_cml=False, show_gmv=False):
    """Function to plot Efficient Frontier

    Args:
        E_return (Array: Nx1): Annualized return
        E_cov (NxN): variance-covariance matrix
        Expected_Risk_free (int): Annualized Risk-Free rate
        Nbr_PTF (int): Number of points for your graph
        bounds (tuple, optional): _description
    """
    
    n = E_return.shape[0]
    
    Efficient_frontiere_result = Ptf_target_optimization_Gurobi(E_return, E_cov, K, Nbr_PTF, bounds) #Without risk-free asset
    
    
    max_vol = max(np.array(E_cov).diagonal().max()**0.5, max(Efficient_frontiere_result['Efficient_frontiere'].index)) 
    #Start by plotting result of efficient frontier without rf
    plt = Efficient_frontiere_result['Efficient_frontiere']['Returns'].plot(kind='line',figsize=(15,11), 
                                                                        xlim = [0, max_vol + 1], 
                                                                        ylim = [0, max(Efficient_frontiere_result['Efficient_frontiere']['Returns'])+0.05]) 
    if show_cml :
        Efficient_frontiere_result_W_RF =Ptf_target_optimization_W_Rf_Gurobi(E_return, E_cov, Expected_Risk_free,
                                                                             K, Nbr_PTF, bounds)['Efficient_frontiere']['Returns'] #With risk-free Assets
        Efficient_frontiere_result_W_RF.plot(kind='line')
        
        #Plot Portfolio Tangente
        res = tangent_Gurobi(E_return, E_cov, Expected_Risk_free, K, Nbr_PTF, bounds)
        y = res.iloc[0][n]
        x = res.iloc[0][n+1]
        plt.scatter(x,y, marker="o")
        plt.annotate("Pf_t",(x,  y))
        plt.legend(['Wihtout RF', 'With RF'])
        
    for i in E_return.index :
            plt.scatter([np.sqrt(E_cov[i][i])],[E_return[i]], marker='o')
            plt.annotate(i,(np.sqrt(E_cov[i][i]),E_return[i]))      
          
    #Plot GMV
    if show_gmv:
        res = min_variance_Gurobi(E_return, E_cov, K, bounds)
        y = res["Return"]
        x = res["Vol"] 
        plt.scatter(x,y, marker="o")
        plt.annotate("GMV",(x,  y))
    
    #Set Titles 
    plt.set_title("Efficient Frontier")
    plt.set_xlabel("Volatility (Std)")
    plt.set_ylabel("Return Annualized")
    
    return plt