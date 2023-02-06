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
        tscv = TimeSeriesSplit(n_splits=2,test_size=8) # Cross validatation 
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


def return_est(Data,path_regression):
    Variable_regression = Load_lasso_variable(path_regression)
    # Data use for the split : 

    First_factor_obs = '2013-09-01'
    Date_train = '2020-01-01' # Train set end
    Date_test = '2021-06-01' # Test set beginning 
    Last_date = '2022-12-01'

    Variable_regression = Variable_regression[Variable_regression.index >= First_factor_obs]

    Data_regression_lasso = Split_dataset(Data,Variable_regression,First_factor_obs,Date_train,Date_test,Last_date)
    
    Resultat_prediction  = Lasso_regression(Data_regression_lasso)
    
    E_return_select = select_pred(Resultat_prediction,Data)
    
    return E_return_select



#################### Covariance Matrix ####################


def Ledoit_wolf(returns):
    Est_cov= pd.DataFrame(data=covariance.ledoit_wolf(returns)[0],index=returns.columns,columns=returns.columns)
    return Est_cov