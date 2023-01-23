import pandas as pd
import numpy as np


def annualize_rets(r, periods_per_year):
    """
    Annualizes a set of returns
    We should infer the periods per year
    but that is currently left as an exercise
    to the reader :-)
    """
    compounded_growth = (1+r).prod()
    n_periods = r.shape[0]
    return compounded_growth**(periods_per_year/n_periods)-1


def annualize_vol(r, periods_per_year):
    """
    Annualizes the vol of a set of returns
    We should infer the periods per year
    but that is currently left as an exercise
    to the reader :-)
    """
    return r.std()*(periods_per_year**0.5)


def sharpe_ratio(r, riskfree_rate, periods_per_year):
    """
    Computes the annualized sharpe ratio of a set of returns
    """
    # convert the annual riskfree rate to per period
    rf_per_period = (1+riskfree_rate)**(1/periods_per_year)-1
    excess_ret = r - rf_per_period
    ann_ex_ret = annualize_rets(excess_ret, periods_per_year)
    ann_vol = annualize_vol(r, periods_per_year)
    return ann_ex_ret/ann_vol





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




from scipy.optimize import minimize
def minimize_vol(target_return, er, cov):
    """
    target_ret --> Weight vector
    
    """
    n = er.shape[0] #determine the number of assets
    init_guess = np.repeat(1/n, n) #Initial weight vector is equally distributed
    bounds = ((-2, 2),) * n #I don't want to be able to short, multiply a tuple make some copy of it
    
    return_is_target = {
        "type": "eq",
        "args":(er,),
        "fun": lambda weights, er: target_return - portfolio_return(weights, er)# We can create a lambda function
            
    }
    weights_sum_to_1 = {
        "type":"eq",         #{constraints} eq = equalize to 0
        "fun": lambda weights: np.sum(weights) - 1
    }
    results = minimize(portfolio_vol, init_guess,
                       args = (cov,), method="SLSQP", 
                       options= {"disp": False},
                       constraints=(return_is_target, weights_sum_to_1),
                       bounds = bounds
                       )
    return results.x


def optimal_weights(n_points, er, cov):
    """
    --> generate a list of weights to run the optimizer on to minimize the vol
    """
    target_rs = np.linspace(er.min(), er.max(), n_points)
    weights = [minimize_vol(target_return, er, cov) for target_return in target_rs]
    return weights


def gmv(cov):
    """
    Returns the weight of the Global Minimum Vol Portfolio given the cov matrix
    """
    n = cov.shape[0]
    return msr(0, np.repeat(1, n), cov)
    

def plot_ef(n_points, er, cov, style=".-", show_cml=False, riskfree_rate=0, show_ew = False, show_gmv= False):
    """
    Plots the N-asset efficient frontier
    """
   
    weights = optimal_weights(n_points, er ,cov)
    rets = [portfolio_return(w, er) for w in weights]
    vols = [portfolio_vol(w, cov) for w in weights]
    ef = pd.DataFrame({
        "Returns": rets, 
        "Volatility": vols
    })
    ax = ef.plot.line(x="Volatility", y="Returns", style=style)
    
    
    if show_gmv:
        w_gmv = gmv(cov)

        r_gmv = portfolio_return(w_gmv, er)
        vol_gmv = portfolio_vol(w_gmv, cov)
        #display GMV
        ax.plot([vol_gmv],[r_gmv], color = "midnightblue", markersize = 10, marker = "o")
    if show_ew:
        n = er.shape[0]
        w_ew = np.repeat(1/n, n)
        r_ew = portfolio_return(w_ew, er)
        vol_ew = portfolio_vol(w_ew, cov)
        #display EW
        ax.plot([vol_ew],[r_ew], color = "goldenrod", markersize = 10, marker = "o")
    if show_cml:
        ax.set_xlim(left = 0)
        rf = 0.1
        w_msr = msr(riskfree_rate, er, cov)
        r_msr = portfolio_return(w_msr, er)
        vol_msr = portfolio_vol(w_msr, cov)
        #Add capital market line
        cml_x = [0, vol_msr]
        cml_y = [riskfree_rate, r_msr]
        ax.plot(cml_x, cml_y, color="green", marker="o", linestyle="dashed", markersize = 5, linewidth= 2)
        return ax
    

def msr(risk_free_rate, er, cov):
    """
    Returns the weights of the portfolio that gives you the maximum sharpe ratio given 
    the riskfree rate and expected returns and a covariance matrix
    
    """
    n = er.shape[0] #determine the number of assets
    init_guess = np.repeat(1/n, n) #Initial weight vector is equally distributed
    bounds = ((-2, 2),) * n #I don't want to be able to short, multiply a tuple make some copy of it
    
    
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
        
        
    results = minimize(neg_sharpe_ratio, init_guess,
                       args = (risk_free_rate, er, cov,), method="SLSQP", 
                       options= {"disp": False},
                       constraints=(weights_sum_to_1),
                       bounds = bounds
                       )
    return results.x