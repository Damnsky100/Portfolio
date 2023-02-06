import gurobipy as gp
from gurobipy import GRB, abs_
import numpy as np
import pandas as pd
from arch.bootstrap import StationaryBootstrap
import Functions as f
import seaborn as sns
import matplotlib
import matplotlib.pyplot as plt
##################### Function to help #####################

#def annualize_rets(r, periods_per_year):
#    """
#    Annualizes a set of returns
#    """
#    compounded_growth = (1+r).prod()
#    n_periods = r.shape[0]
#    return compounded_growth**(periods_per_year/n_periods)-1



#def annualize_vol(r, periods_per_year):
#    """
#    Annualizes the vol of a set of returns
#    """
#    return r.std()*(periods_per_year**0.5)



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
def tangent_Gurobi(E_return, E_cov, Expected_Risk_free, K, Nbr_PTF, bounds):
    
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
                                                                        ylim = [min(Efficient_frontiere_result['Efficient_frontiere']['Returns'])-0.01, max(Efficient_frontiere_result['Efficient_frontiere']['Returns'])+0.005]) 
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
    plt.set_ylabel("Return")
    
    return plt


#Resample

def boostrap(Data,Nsim):
    np.random.seed(123)
    block_size=len(Data)**(1/3)
    bs=StationaryBootstrap(block_size,Data)
    Resample=[block_size]
    dates=pd.date_range(start="2000-01-01",periods=len(Data),freq="MS")
    for data in bs.bootstrap(Nsim):
        Resample.append(data[0][0].set_index(dates))
    del Resample[0]
    return  Resample
 
def resample_no_risk(E_return, E_cov,Resample,K, Nbr_PTF,path_regression,bounds = (-2, 2)): #With Stationnary block resampling
    RE_weights=np.zeros((Nbr_PTF,len(E_return)))
    REE_return=np.zeros(Nbr_PTF)
    REE_Cov=np.zeros(Nbr_PTF)
    sims=len(Resample)
    for j in range(0,sims):
        Data=Resample[j]
        #Estimation Return
        RE_return=f.return_est(Data,path_regression)
        RE_cov=f.Ledoit_wolf(Data)
        result=Ptf_target_optimization_Gurobi(RE_return, RE_cov,K,Nbr_PTF, bounds)
        RE_weights=RE_weights+ result['Efficient_frontiere_weigth'].to_numpy()
        REE_return=REE_return+ result['Efficient_frontiere'].to_numpy().reshape((Nbr_PTF))
    
    RE_weights=RE_weights/sims
    #REE_return=REE_return/sims
    
    for i in range(0,Nbr_PTF):
        REE_return[i]=np.dot( RE_weights[i,:].T,E_return)
        REE_Cov[i]=(np.dot( RE_weights[i,:],np.dot(E_cov.to_numpy(),RE_weights[i,:].T)))**0.5
    #Generate multivariate_normal with return distribution
    
    # Data formatting 
    
    REfficient_frontiere_df = pd.DataFrame({"Returns":REE_return,"Standard_deviation":REE_Cov})
    REfficient_frontiere_df.set_index('Standard_deviation', inplace=True)
    REfficient_frontiere_weigth = pd.DataFrame(RE_weights,columns=E_return.index)
    
    REfficient_frontiere_result = dict([('REfficient_frontiere',REfficient_frontiere_df),
                                           ('REfficient_frontiere_weigth',REfficient_frontiere_weigth)])
    
    return(REfficient_frontiere_result)

def resample_with_risk(E_return, E_cov,Resample,Expected_Risk_free, K, Nbr_PTF,path_regression, bounds = (-2, 2)): #With Stationnary block resampling
    Weigth_names = np.concatenate((np.array(['Risk_free']), np.array(E_return.index)))
    RE_weights=np.zeros((Nbr_PTF,len(E_return)+1))
    REE_return=np.zeros(Nbr_PTF)
    REE_Cov=np.zeros(Nbr_PTF)
    sims=len(Resample)
    for j in range(0,sims):
        Data=Resample[j]
        #Estimation Return
        RE_return=f.return_est(Data,path_regression)
        RE_cov=f.Ledoit_wolf(Data)
        result=Ptf_target_optimization_W_Rf_Gurobi(RE_return, RE_cov,Expected_Risk_free, K, Nbr_PTF, bounds)
        RE_weights=RE_weights+ result['Efficient_frontiere_weigth'].to_numpy()
        REE_return=REE_return+ result['Efficient_frontiere'].to_numpy().reshape((Nbr_PTF))
    
    RE_weights=RE_weights/sims
    #REE_return=REE_return/sims
    Expected_return_W_RF= np.concatenate((np.array([Expected_Risk_free]), E_return))

    for i in range(0,Nbr_PTF):
        REE_return[i]=np.dot(RE_weights[i,:].T,Expected_return_W_RF)
        REE_Cov[i]=(np.dot(RE_weights[i,1:],np.dot(E_cov.to_numpy(),RE_weights[i,1:].T)))**0.5
    #Generate multivariate_normal with return distribution
    
    # Data formatting 
    
    REfficient_frontiere_df = pd.DataFrame({"Returns":REE_return,"Standard_deviation":REE_Cov})
    REfficient_frontiere_df.set_index('Standard_deviation', inplace=True)
    REfficient_frontiere_weigth = pd.DataFrame(RE_weights,columns=Weigth_names)
    
    REfficient_frontiere_result = dict([('REfficient_frontiere',REfficient_frontiere_df),
                                           ('REfficient_frontiere_weigth',REfficient_frontiere_weigth)])
    
    return(REfficient_frontiere_result)
    

def tangent_resample(result,Expected_Risk_free):
    
    tmp1 = result["REfficient_frontiere_weigth"]
    
    tmp = result["REfficient_frontiere"]
    
    vol = np.array(tmp.index)
    ret = np.array(tmp["Returns"])
    

    
    frontier = pd.DataFrame(tmp1)
    frontier["Return"] = ret
    frontier["Volatility"] = vol

    frontier['Sharpe'] = (frontier['Return'] - Expected_Risk_free) / frontier['Volatility']
    idx = frontier['Sharpe'].max()
    sharpeMax = frontier.loc[frontier['Sharpe'] == idx]
    return sharpeMax
    
  #Plot Resampling

def plot_Resample_ef_Gurobi(E_return,E_cov,result1,result2,Expected_Risk_free,show_cml=False, show_gmv=False):
    """Function to plot Efficient Frontier

    Args:
        E_return (Array: Nx1): Annualized return
        E_cov (NxN): variance-covariance matrix
        Expected_Risk_free (int): Annualized Risk-Free rate
        Nbr_PTF (int): Number of points for your graph
        bounds (tuple, optional): _description
    """
    
    n = E_return.shape[0]
    
    Efficient_frontiere_result = result1 #Without risk-free asset
    
    
    max_vol = max(np.array(E_cov).diagonal().max()**0.5, max(Efficient_frontiere_result['REfficient_frontiere'].index)) 
    #Start by plotting result of efficient frontier without rf
    plt = Efficient_frontiere_result['REfficient_frontiere']['Returns'].plot(kind='line',figsize=(15,11), 
                                                                        xlim = [0, max_vol + 1], 
                                                                        ylim = [min(Efficient_frontiere_result['REfficient_frontiere']['Returns'])-0.01, max(Efficient_frontiere_result['REfficient_frontiere']['Returns'])+0.005]) 
    if show_cml :
        REEfficient_frontiere_result_W_RF=result2
        Efficient_frontiere_result_W_RF=REEfficient_frontiere_result_W_RF['REfficient_frontiere']['Returns'] #With risk-free Assets
        
        Efficient_frontiere_result_W_RF.plot(kind='line')
        
        #Plot Portfolio Tangente
        res =tangent_resample(Efficient_frontiere_result,Expected_Risk_free)

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
        res= Efficient_frontiere_result['REfficient_frontiere'].reset_index()
        y =res['Returns'].iloc[0]  
        x =res['Standard_deviation'].iloc[0]
        plt.scatter(x,y, marker="o")
        plt.annotate("GMV",(x,  y))
    
    #Set Titles 
    plt.set_title("Efficient Frontier")
    plt.set_xlabel("Volatility (Std)")
    plt.set_ylabel("Return")
    
    return plt

#Frontier Portfolio Composition Maps


def transition_map(E_return, E_cov, Expected_Risk_free, K, Nbr_PTF,bounds,no_risk=True):
   
    colors=['red','orange','green','blue','purple','gray','violet','black','navy','cyan']
    
    if no_risk:
        
        Efficient_frontiere_result = Ptf_target_optimization_Gurobi(E_return, E_cov, K, Nbr_PTF, bounds) #Without risk-free asset
    
        labels=Efficient_frontiere_result['Efficient_frontiere_weigth'].columns
        ret_df=Efficient_frontiere_result['Efficient_frontiere_weigth']
        frontier=Efficient_frontiere_result['Efficient_frontiere'].reset_index()
    
        x = frontier['Standard_deviation']
        pal = colors[:len(labels)]
        # absolute values so shorts don't create chaos
        y_list = [abs(ret_df[l]) for l in labels]
        
        fig = plt.figure(figsize=(8, 4.5))
        ax1 = fig.add_subplot(111)

        ax1.stackplot(x, y_list, labels=labels, colors=pal)
        ax1.set_ylim((0,bounds[1]))
        ax1.set_xlim((frontier['Standard_deviation'].iloc[0], frontier['Standard_deviation'].iloc[-1]))
        ax1.set_xlabel('Portfolio Vol')
        ax1.set_ylabel("Portfolio Weight")
        ax1.legend(loc='lower right')
        ax2 = ax1.twiny()
        ax2.set_xlim((frontier['Returns'].iloc[0], frontier['Returns'].iloc[-1]))
        ax2.set_xlabel("Portfolio Real Return")
    
    
        plt.title('MV Frontier Portfolio Composition Maps', y=1.16);

    
    else:
        
        Efficient_frontiere_result = Ptf_target_optimization_W_Rf_Gurobi(E_return, E_cov,Expected_Risk_free,K, Nbr_PTF, bounds) #Without risk-free asset
    
        labels=Efficient_frontiere_result['Efficient_frontiere_weigth'].columns
        ret_df=Efficient_frontiere_result['Efficient_frontiere_weigth']
    
        frontier=Efficient_frontiere_result['Efficient_frontiere'].reset_index()
    
        x = frontier['Standard_deviation']
        pal =  colors[:len(labels)]
        # absolute values so shorts don't create chaos
        y_list = [abs(ret_df[l]) for l in labels]
        
        
        fig = plt.figure(figsize=(8, 4.5))
        ax1 = fig.add_subplot(111)

        ax1.stackplot(x, y_list, labels=labels, colors=pal)
        ax1.set_ylim((0,bounds[1]))
        ax1.set_xlim((frontier['Standard_deviation'].iloc[0], frontier['Standard_deviation'].iloc[-1]))
        ax1.set_xlabel('Portfolio Vol')
        ax1.set_ylabel("Portfolio Weight")
        ax1.legend(loc='lower right')
        ax2 = ax1.twiny()
        ax2.set_xlim((frontier['Returns'].iloc[0], frontier['Returns'].iloc[-1]))
        ax2.set_xlabel("Portfolio Real Return")
    
    
        plt.title('MV Frontier Portfolio Composition Maps', y=1.16);


    
def Resample_transition_map(result,bounds):
    
    colors=['red','orange','green','blue','purple','gray','violet','black','navy','cyan']
    
   
        
    Efficient_frontiere_result = result #Without risk-free asset
    
    labels=Efficient_frontiere_result['REfficient_frontiere_weigth'].columns
    ret_df=Efficient_frontiere_result['REfficient_frontiere_weigth']
    
    frontier=Efficient_frontiere_result['REfficient_frontiere'].reset_index()
    
    x = frontier['Standard_deviation']
    pal =  colors[:len(labels)]
    # absolute values so shorts don't create chaos
    y_list = [abs(ret_df[l]) for l in labels]
        
    fig = plt.figure(figsize=(8, 4.5))
    ax1 = fig.add_subplot(111)

    ax1.stackplot(x, y_list, labels=labels, colors=pal)
    ax1.set_ylim((0,bounds[1]))
    ax1.set_xlim((frontier['Standard_deviation'].iloc[0], frontier['Standard_deviation'].iloc[-1]))
    ax1.set_xlabel('Portfolio Vol')
    ax1.set_ylabel("Portfolio Weight")
    ax1.legend(loc='lower right')
    ax2 = ax1.twiny()
    ax2.set_xlim((frontier['Returns'].iloc[0], frontier['Returns'].iloc[-1]))
    ax2.set_xlabel("Portfolio Real Return")
    
    
    plt.title('Resample Frontier Portfolio Composition Maps', y=1.16);
