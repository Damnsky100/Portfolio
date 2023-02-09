##Importer les librairies
import importlib

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import optimize
import scipy.optimize as sco

from sklearn.model_selection import TimeSeriesSplit
from sklearn.linear_model import Lasso,LassoCV
from sklearn.metrics import mean_squared_error,r2_score,mean_absolute_percentage_error
from sklearn.preprocessing import StandardScaler
from sklearn import covariance



import gurobipy as gp
from gurobipy import GRB, abs_
#################### Importer les données ####################


# Loader Industry data (Veuillez ajouter votre path)
def Load_ind_data(path):
    
    dateparse = lambda x: pd.datetime.strptime(x,'%Y%m') 

    Data = pd.read_csv(path ,skiprows=11,nrows=1157,index_col=0,parse_dates=True,date_parser=dateparse)
    Data.columns = Data.columns.str.strip() #Enlever l'espace dans le titre des colonnes

    ind_names = list(Data.columns) #List of 48 industry available
    
    return {"Data": Data, "Industry_name": ind_names}



#Loader RF data (À voir si on change la méthodologie pour la sélection)
def Load_rf(path, date = '2022-08-01'):
    
    Data_SOFR = pd.read_excel(path)

    Data_SOFR['observation_date'] = pd.to_datetime(Data_SOFR['observation_date'], format='%Y-%m-%d')
    Data_SOFR.set_index('observation_date', inplace=True)

    Data_SOFR=Data_SOFR.dropna()



    #Expected_Risk_free = Data_SOFR[Data_SOFR.index >= date].mean() # do again 
    
    # Keep the last value : 
    
    Expected_Risk_free = Data_SOFR.iloc[-1]
    
    # convert the annual riskfree rate to per period
    rf_per_period = ((1+Expected_Risk_free)**(1/12))-1
    
    return  (rf_per_period.iloc[0])

# Load factor and macroeconomics  data for the prediction of returns

def Load_lasso_variable(path_regression):
    Variable_regression = pd.read_excel(path_regression ,header=0,skiprows=3)
    Variable_regression['observation_date'] = pd.to_datetime(Variable_regression['observation_date'], format='%Y-%m-%d')
    Variable_regression.set_index('observation_date', inplace=True)
    return(Variable_regression)



#################### Without risk-free asset : ####################

##### Determination of the maximum return portfolio (under certain constraints) #####






##################### Gurobi Function to help #####################

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
                                                                        xlim = [0, max_vol + 0.01], 
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





def areaplot(E_return, E_cov, K, Nbr_PTF, bounds):
    tmp1 = Ptf_target_optimization_Gurobi(E_return, E_cov, K, Nbr_PTF, bounds)["Efficient_frontiere"]
    tmp = Ptf_target_optimization_Gurobi(E_return, E_cov, K, Nbr_PTF, bounds)["Efficient_frontiere_weigth"]
    
    tmp.index = tmp1.index
    return tmp.plot.area(stacked = False)


def areaplot_wrf(E_return, E_cov, K, Expected_Risk_free, Nbr_PTF, bounds):
    tmp1 = Ptf_target_optimization_W_Rf_Gurobi(E_return, E_cov, Expected_Risk_free, K, Nbr_PTF, bounds)["Efficient_frontiere"]
    tmp = Ptf_target_optimization_W_Rf_Gurobi(E_return, E_cov, Expected_Risk_free, K, Nbr_PTF, bounds)["Efficient_frontiere_weigth"]
    
    tmp.index = tmp1.index
    return tmp.plot.area(stacked = False)
#################### Determination of the expected return  : ####################

##### Lasso regression  #####

def Split_dataset(Data_return,Variable_regression,Date_variable,Date_train,Date_test,Last_date):
    
    
    # Gestion Date :  
    Data_return=Data_return.append(Data_return.iloc[-1])
    Date = Data_return.index.values
    Date[-1]='2022-12-01T00:00:00.000000000'
    Data_return.index=Date

# Split the dataset of variable : 

    V_R_train = Variable_regression[Variable_regression.index  <= Date_train]
    V_R_test = Variable_regression[Variable_regression.index >= Date_test]
    V_R_test = V_R_test[V_R_test.index <= Last_date]
    
    # Construction of the lagged return :
    
    Return_L1 = Data_return.shift(periods=1).iloc[1:,:]
    
    Return_l1_train = Return_L1[Return_L1.index >= Date_variable]
    Return_l1_train = Return_l1_train[Return_l1_train.index <= Date_train]
    
    Return_l1_test =  Return_L1[Return_L1.index >= Date_test]
    Return_l1_test =  Return_l1_test[Return_l1_test.index <= Last_date]
    
    #
    
    Return_L2 = Data_return.shift(periods=2).iloc[2:,:]
    
    Return_l2_train = Return_L2[Return_L2.index >= Date_variable]
    Return_l2_train = Return_l2_train[Return_l2_train.index <= Date_train]
    
    Return_l2_test =  Return_L2[Return_L2.index >= Date_test]
    Return_l2_test =  Return_l2_test[Return_l2_test.index <= Last_date]
    
    #
    
    Return_L3 = Data_return.shift(periods=3).iloc[3:,:]
    
    Return_l3_train = Return_L3[Return_L3.index >= Date_variable]
    Return_l3_train = Return_l3_train[Return_l3_train.index <= Date_train]
    
    Return_l3_test =  Return_L3[Return_L3.index >= Date_test]
    Return_l3_test =  Return_l3_test[Return_l3_test.index <= Last_date]
    
    Return_L3 = Data_return.shift(periods=3).iloc[3:,:]
# Traitement of the dataset : 

    V_R_train=V_R_train.reset_index()
    V_R_train=V_R_train.drop(columns=["observation_date"])

    V_R_test=V_R_test.reset_index()
    V_R_test=V_R_test.drop(columns=["observation_date"])
    
 
    Return_l1_train=Return_l1_train.reset_index()
    Return_l1_train=(Return_l1_train.drop(columns=["index"]))
    
    
    Return_l1_test=Return_l1_test.reset_index()
    Return_l1_test=(Return_l1_test.drop(columns=["index"]))
    
    Return_l2_train=Return_l2_train.reset_index()
    Return_l2_train=(Return_l2_train.drop(columns=["index"]))
    
    
    Return_l2_test=Return_l2_test.reset_index()
    Return_l2_test=(Return_l2_test.drop(columns=["index"]))
    
    
    Return_l3_train=Return_l3_train.reset_index()
    Return_l3_train=(Return_l3_train.drop(columns=["index"]))
    
    
    Return_l3_test=Return_l3_test.reset_index()
    Return_l3_test=(Return_l3_test.drop(columns=["index"]))
    
# Split the dataset of output :  

    Y= Data_return[Data_return.index >= Date_variable]
    Y_train = Y[Y.index <= Date_train]
    Y_test = Y[Y.index >= Date_test]
    Y_test = Y_test[Y_test.index < Last_date]
    
# Traitement of the dataset : 

    Y_train=Y_train.reset_index()
    Y_train=(Y_train.drop(columns=["index"]))/100

    Y_test=Y_test.reset_index()
    Y_test=(Y_test.drop(columns=["index"]))/100
    

    Data_regression_lasso = dict([
        ('V_R_train',V_R_train),
        ('V_R_test',V_R_test),
        ('Y_train',Y_train),
        ('Y_test',Y_test),
        ('Return_l1_train',Return_l1_train),
        ('Return_l1_test',Return_l1_test),
        ('Return_l2_train',Return_l2_train),
        ('Return_l2_test',Return_l2_test),
        ('Return_l3_train',Return_l3_train),
        ('Return_l3_test',Return_l3_test)
    ])
    
    return(Data_regression_lasso)


def Lasso_regression(Data_regression_lasso):
    
# Input lasso regression : 

    
    Y_train = Data_regression_lasso ['Y_train']
    Y_test = Data_regression_lasso ['Y_test']
    
   
    result_lasso = np.zeros((6,len(Y_train.columns)))
    


# Regression for each industry : 

    for i in range (0,len(Y_train.columns)):
        
        
        V_R_train = Data_regression_lasso ['V_R_train']
        V_R_test = Data_regression_lasso ['V_R_test']
        
        industrie = Y_train.columns[i]
        
        
        V_R_train["Return_l1"] = Data_regression_lasso ['Return_l1_train'][industrie]
        V_R_test["Return_l1"]=  Data_regression_lasso ['Return_l1_test'][industrie]
        
        V_R_train["Return_l2"] = Data_regression_lasso ['Return_l2_train'][industrie]
        V_R_test["Return_l2"]=  Data_regression_lasso ['Return_l2_test'][industrie]
        
        V_R_train["Return_l3"] = Data_regression_lasso ['Return_l3_train'][industrie]
        V_R_test["Return_l3"]=  Data_regression_lasso ['Return_l3_test'][industrie]
        
        # Standardisation : 

        scaler = StandardScaler().fit(V_R_train) 
        V_R_train = scaler.transform(V_R_train)
        V_R_test = scaler.transform(V_R_test)
    
        
    
    # Model parametrization :
        tscv = TimeSeriesSplit(n_splits=3,test_size=9) # Cross validatation 
        model = LassoCV(cv=tscv) # try to find the optimal alpha (penalization)
        model.fit(V_R_train, Y_train[industrie])
    
    # Fit the model to the return of the specific industry 
    
        lasso_best = Lasso(alpha=model.alpha_)
        lasso_best.fit(V_R_train, Y_train[industrie])
    
    # Prediction : 
    
        pred = lasso_best.predict(V_R_test)
    
    # keep only the last prediction for next month 
        Last_pred = pred[-1]
    
        R_squared_train = lasso_best.score(V_R_train, Y_train[industrie])*100
        R_squared_test =lasso_best.score(V_R_test[0:-1], Y_test[industrie])*100
        
        #  Performance measurement out of sample:
        
        MSE_test_set = mean_squared_error(Y_test[industrie], pred[0:(len(pred)-1)])
        
        
        mean_historique = np.array([Y_train[industrie].mean()] * len(Y_test[industrie]))
        
        MSE_test_set_mean = mean_squared_error(Y_test[industrie], mean_historique)
        
        ratio_mse = MSE_test_set/MSE_test_set_mean
        
        
        
        
        result_lasso [:,i] = [Last_pred,R_squared_train,MSE_test_set,R_squared_test,MSE_test_set_mean,ratio_mse]
        
    Resultat_prediction = pd.DataFrame(result_lasso, columns=Y_train.columns, index = ['E_R','R_squared_train','MSE_test_set','R_squared_test','MSE_test_set_mean','ratio_mse'])
    
    return(Resultat_prediction)

def select_pred(Resultat_prediction,Data) :
    """ Select only the prediction where the lasso approch perform better on the train set base on MSE ratio and R square
    """
    R_T = Resultat_prediction.T
    
    # Selection criterion : MSE_ratio < 1 and/or R_squared > 0 on train set :
    
    R_T['E_R'] = np.where((R_T['R_squared_test'] <= 0) | (R_T['ratio_mse'] >= 1), np.NaN, R_T['E_R'])
    R_T = R_T.dropna()
    
    # For each industrie where the prediction base on lasso are not keep we use the historical mean :
    
    E_return_mean =Data.mean()/100
    E_return_mean=E_return_mean.drop(index=R_T.index)
    E_return_mean.name = 'E_R'
    
    E_return_select = R_T['E_R'].append(E_return_mean)
    
    return(E_return_select)

#################### Plotting Functions ####################


def plot_ef(E_return, E_cov, Expected_Risk_free, Nbr_PTF, bounds,  show_cml=False, show_gmv=False):
    """Function to plot Efficient Frontier

    Args:
        E_return (Array: Nx1): Annualized return
        E_cov (NxN): variance-covariance matrix
        Expected_Risk_free (int): Annualized Risk-Free rate
        Nbr_PTF (int): Number of points for your graph
        bounds (tuple, optional): _description
    """
    
    Efficient_frontiere_result = Ptf_target_optimization(E_return, E_cov, Nbr_PTF,
                                                         bounds) #Without risk-free asset
    
    #Start by plotting result of efficient frontier without rf
    plt = Efficient_frontiere_result['Efficient_frontiere']['Returns'].plot(kind='line',figsize=(15,11), 
                                                                        xlim = [0, max(Efficient_frontiere_result['Efficient_frontiere'].index)+0.5], 
                                                                        ylim = [min(Efficient_frontiere_result['Efficient_frontiere']['Returns'])-0.3, max(Efficient_frontiere_result['Efficient_frontiere']['Returns'])+0.2]) 
    if show_cml :
        Efficient_frontiere_result_W_RF = Ptf_target_optimization_W_Rf(E_return, E_cov,
                                                                   Expected_Risk_free, Nbr_PTF,
                                                                   bounds)['Efficient_frontiere']['Returns'] #With risk-free Assets
        Efficient_frontiere_result_W_RF.plot(kind='line')
        
        #Plot Portfolio Tangente
        w = msr(Expected_Risk_free, E_return, E_cov, bounds)
        y = portfolio_return(w, E_return)
        x = portfolio_vol(w, E_cov)
        plt.scatter(x,y, marker="o")
        plt.annotate("Pf_t",(x,  y))
        plt.legend(['Wihtout RF', 'With RF'])
        
    for i in E_return.index :
            plt.scatter([np.sqrt(E_cov[i][i])],[E_return[i]], marker='o')
            plt.annotate(i,(np.sqrt(E_cov[i][i]),E_return[i]))      
          
    #Plot GMV
    if show_gmv:
        w = gmv(E_cov, bounds)
        y = portfolio_return(w, E_return)
        x =  portfolio_vol(w, E_cov)   
        plt.scatter(x,y, marker="o")
        plt.annotate("GMV",(x,  y))
    
    #Set Titles 
    plt.set_title("Efficient Frontier")
    plt.set_xlabel("Volatility (Std)")
    plt.set_ylabel("Return Annualized")
    
    return plt

#################### Covariance Matrix ####################


def Ledoit_wolf(returns):
    Est_cov= pd.DataFrame(data=covariance.ledoit_wolf(returns)[0],index=returns.columns,columns=returns.columns)
    return Est_cov
