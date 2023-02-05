##Importer les librairies

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

def Max_return_ptf(E_return, bounds = (0, 1)):
      
    #Initialize variables
    
    Initial_weights = np.array([1 / len(E_return)] * len(E_return))
    bounds = tuple(bounds for w in Initial_weights)
    Industrie_selected = E_return.index
    constraints_max_ret = ( {'type': 'eq', 'fun': lambda x: np.sum(x) - 1}) # Sum of the weigth need to sum to 1
    
# Define objective fonction to minimize :
    def Max_return_obj (weigth, E_return = E_return.values):
        return - np.dot(weigth.T, E_return)
    
# Compute the maximum return portfolio by optimisation under constraints: 
    Max_ret_PTF = sco.minimize(

      fun = Max_return_obj, 
      x0 = Initial_weights, 
      method = 'SLSQP',
      bounds = bounds, 
      constraints = constraints_max_ret
    
    )

    Max_ret_weigth = pd.DataFrame(Max_ret_PTF['x'], index=Industrie_selected,columns = ['Weigth'])

    Max_ret_return = pd.DataFrame([np.dot(Max_ret_PTF['x'].T, E_return.values)], columns = ['return'])
    
    Max_return = {'Max_ret_return' : Max_ret_return,
                       'Max_ret_weigth' : Max_ret_weigth.T}
    return(Max_return)





##### Determination of the minimum-variance  portfolio  #####

def Min_var_ptf(E_return, E_cov, bounds = (-2,2)):
    
    #Initialize variables
    
    Initial_weights = np.array([1 / len(E_return)] * len(E_return))
    bounds = tuple(bounds for w in Initial_weights)
    Industrie_selected = E_return.index
    constraints_min_var = ( {'type': 'eq', 'fun': lambda x: np.sum(x) - 1}) # Sum of the weigth need to sum to 1
   
    
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

    Min_var_weigth = pd.DataFrame(Min_var_PTF['x'], index=Industrie_selected, columns= ['Weigth'])

    Min_var_return = pd.DataFrame([np.dot(Min_var_PTF['x'].T, E_return.values)], columns= ['return'])
    
    Min_var = {'Min_var_return': Min_var_return,
                     'Min_var_weigth': Min_var_weigth.T}
  
    return(Min_var)




##### Determination of the optimal  portfolio  for a given return  #####

def Ptf_target_optimization(E_return, E_cov, Nbr_PTF, bounds = (-2, 2)):
    
    #Define Variables
    
    Min_var_result = Min_var_ptf(E_return, E_cov, bounds)
    Max_return_result = Max_return_ptf(E_return, bounds)
    
    Initial_weights = np.array([1 / len(E_return)] * len(E_return))
    bounds = tuple(bounds for w in Initial_weights)
    
    
    
    Target_return = np.linspace (Min_var_result['Min_var_return'].iloc[0]['return'], Max_return_result['Max_ret_return'].iloc[0]['return'], Nbr_PTF)
    Weigth_names = E_return.index
    Weigth_level = np.zeros((len(E_return), Nbr_PTF)) #Initialize the array
    Var_level = np.zeros((1, Nbr_PTF))
    
    
    # Define objective fonction to minimize :

    def Min_Variance_obj (weigth, Cov_matrix = E_cov) : 
        return (0.5*np.dot(np.dot(weigth.T, Cov_matrix), weigth))
    
    # Optimization for each return target :

    constraints = (
        {'type': 'eq', 'fun': lambda x: np.sum(x*E_return.values) - target}, # Optimization for a target return 
        {'type': 'eq', 'fun': lambda x: np.sum(x) - 1})

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
    
    Return_effectiv = np.dot(E_return.values, Weigth_level)
    Efficient_frontiere = np.concatenate((np.array([Target_return]), Var_level))
    
    # Data formatting 
    
    Efficient_frontiere_df = pd.DataFrame(Efficient_frontiere.T, columns=['Returns', 'Standard_deviation'])
    Efficient_frontiere_df.set_index('Standard_deviation', inplace=True)
    
    Efficient_frontiere_weigth = pd.DataFrame(Weigth_level.T, columns=Weigth_names)
    
    Efficient_frontiere_result = {'Efficient_frontiere': Efficient_frontiere_df,
                                           'Efficient_frontiere_weigth': Efficient_frontiere_weigth}
    
    return(Efficient_frontiere_result)






#################### With risk-free asset ####################


def Ptf_target_optimization_W_Rf(E_return, E_cov, Expected_Risk_free, Nbr_PTF, bounds = (-2,2)):
    
     #Define Variables
    
    Max_return_result = Max_return_ptf(E_return, bounds)
    
    Initial_weights = np.array([1 / len(E_return)] * len(E_return))
    bounds = tuple(bounds for w in Initial_weights)
    
    
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
        target = Target_return[i]
        Efficient_frontier_r =  sco.minimize(
            fun = Min_Variance_obj, 
            x0 = Initial_weights, 
            method = 'SLSQP', 
            constraints = constraints_W_RF,
            bounds = bounds
        ) 
        
        Var_level_W_O_S[0,i] = np.sqrt(2*Efficient_frontier_r['fun'])
        Weigth_level_W_O_S[1:(len(E_return) + 1), i] = Efficient_frontier_r['x']
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





#################### Porfolio Tangent & GMV Functions ####################


#def annualize_rets(r, periods_per_year):
#    """
#    Annualizes a set of returns
#    """
#    compounded_growth = (1+r).prod()
#    n_periods = r.shape[0]
#    return compounded_growth**(periods_per_year/n_periods)-1

#def annualize_rets_prediction(r, periods_per_year):
#    """
#    Annualizes a set of returns
#    """
#    R_A = (((1+r)**periods_per_year)-1)
#    return R_A


#def annualize_vol(r, periods_per_year):
#    """
#    Annualizes the vol of a set of returns
#    """
#    return r.std()*(periods_per_year**0.5)


def sharpe_ratio(r, riskfree_rate, periods_per_year):
    """
    Computes the annualized sharpe ratio of a set of returns
    """
    # convert the annual riskfree rate to per period
    #rf_per_period = (1+riskfree_rate)**(1/periods_per_year)-1
    excess_ret = r - rf_per_period
    
    #ann_ex_ret = annualize_rets(excess_ret, periods_per_year)
    #ann_vol = annualize_vol(r, periods_per_year)
    vol =r.std()
    return ex_ret.mean()/vol





def portfolio_return(weights, returns):
    """
    Computes the return on a portfolio from constituent returns and weights
    weights are a numpy array or Nx1 matrix and returns are a numpy array or Nx1 matrix
    """
    return weights.T @ returns


def portfolio_vol(weights, covmat):
    """
    Computes the vol of a portfolio from a covariance matrix and constituent weights
    weights are a numpy array or N x 1 maxtrix and covmat is an N x N matrix
    """
    return (weights.T @ covmat @ weights)**0.5


# Give the Global Minimum Vol Portfolio
def gmv(cov, bounds):
    """
    Returns the weight of the Global Minimum Vol Portfolio given the cov matrix
    """
    n = cov.shape[0]
    return msr(0, np.repeat(1, n), cov, bounds)





def msr(risk_free_rate, er, cov, bounds):
    """
    Returns the weights of the portfolio that gives you the maximum sharpe ratio given 
    the riskfree rate and expected returns and a covariance matrix
    
    """
    n = er.shape[0] #determine the number of assets
    init_guess = np.repeat(1/n, n) #Initial weight vector is equally distributed
    bounds = (bounds,) * n #I don't want to be able to short, multiply a tuple make some copy of it
    
    
    weights_sum_to_1 = {
        "type":"eq",         #{constraints} eq = equalize to 0
        "fun": lambda weights: np.sum(weights) - 1
    }
    def neg_sharpe_ratio(weights, risk_free_rate, er, cov):
        """
        Returns the negative of the sharpe ratio, given weights
        
        """
        r = portfolio_return(weights, er)
        vol = portfolio_vol(weights, cov)
        return  -(r-risk_free_rate) / vol
        
        
    results = sco.minimize(neg_sharpe_ratio, init_guess,
                       args = (risk_free_rate, er, cov,), method="SLSQP", 
                       options= {"disp": False},
                       constraints=(weights_sum_to_1),
                       bounds = bounds
                       )
    return results.x

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