import numpy as np
import pandas as pd
import timeit
import numba
import math

import matplotlib.pyplot as plt
from scipy.stats import norm
import seaborn as sns # data visualization Library 

from scipy import stats
from scipy.stats import moment, skew, kurtosis

#%matplotlib inline

# toggle seed
np.random.seed(1)

##============================================================================##
##================== P R I C E R S  A N D  S I M U L A T I O N ===============##
##============================================================================##

# Black-Scholes European Pricer
def BS(S,K,r,q,sigma,T,flavor='c',style='euro',display='no',t=0):
    '''This function takes input parameters for vanilla european options.
    Flavor for p/c, style, display. Returns closed-form price and greeks.'''
    dt = T-t
    d1 = ((np.log(S/K) + (r - q)*(dt)) / sigma*np.sqrt(dt)) + 0.5*sigma*np.sqrt(dt)
    d2 = ((np.log(S/K) + (r - q)*(dt)) / sigma*np.sqrt(dt)) - 0.5*sigma*np.sqrt(dt)
    d1_ = ((np.log(S/K) - (r - q)*(dt)) / sigma*np.sqrt(dt)) - 0.5*sigma*np.sqrt(dt)
    d2_ = ((np.log(S/K) - (r - q)*(dt)) / sigma*np.sqrt(dt)) + 0.5*sigma*np.sqrt(dt)
    if flavor == 'c':
        price = S*np.exp(-q*dt)*norm.cdf(d1) - K*np.exp(-r*dt)*norm.cdf(d2)
        delta = np.exp(-q*dt)*norm.cdf(d1)
        gamma = np.exp(-q*dt)*(1/(S*sigma*np.sqrt(dt)))*norm.pdf(d1)
        vega = S*np.exp(-q*dt)*np.sqrt(dt)*norm.pdf(d1)
        theta = -S*sigma*np.exp(-q*dt)*(1/(2*np.sqrt(dt)))*norm.pdf(d1) +q*S*np.exp(-q*dt)*norm.cdf(d1) - r*K*np.exp(-r*dt)*norm.cdf(d2)
        rho_q = -dt*S*np.exp(-q*T)*norm.cdf(d1)
    if flavor == 'p':
        price = K*np.exp(-r*dt)*norm.cdf(d2_) - S*np.exp(-q*dt)*norm.cdf(d1_)
        delta = -np.exp(-q*dt)*norm.cdf(d1_)
        gamma = np.exp(-q*dt)*(1/(S*sigma*np.sqrt(dt)))*norm.pdf(d1)
        vega = S*np.exp(-q*dt)*np.sqrt(dt)*norm.pdf(d1)
        theta = -S*sigma*np.exp(-q*dt)*(1/(2*np.sqrt(dt)))*norm.pdf(d1) -q*S*np.exp(-q*dt)*norm.cdf(d1_) + r*K*np.exp(-r*dt)*norm.cdf(d2_)
    if display == "yes":
        print("Price: {:.4f}".format(price))
        print("Delta: {:.4f}".format(delta))
        print("Gamma: {:.4f}".format(gamma))
        print("Vega: {:.4f}".format(vega))
        print("Theta: {:.4f}".format(theta))
    return price, delta, gamma, vega, theta, rho_q


## ====================== Monte Carlo Simulator ===============================##

def MC(batches,St,K,r,q,sigma,T,flavor='Call',alpha=0,style='euro',seed=2,t=0):
    '''This function is a bare bones Monte Carlo simulator.  It will generate a 
    Geometric Brownian Motion using inputs for parameters, style, seed.  

    Alpha is used for power and root payout structures.'''
    
    dt = T-t
    mu = (r - q - sigma*sigma*0.5)*dt
    
    d1 = ((np.log(St/K) + (r - q)*(dt)) / sigma*np.sqrt(dt)) + 0.5*sigma*np.sqrt(dt)
    d2 = ((np.log(St/K) + (r - q)*(dt)) / sigma*np.sqrt(dt)) - 0.5*sigma*np.sqrt(dt)
    #np.random.seed(seed)
    price = []
    stdev = []
    
    for n in batches:
        draws = np.random.normal(size=n)
        stoch = sigma*np.sqrt(dt)*draws
        ST = St*np.exp(mu + stoch)
        ST_ = St*np.exp(mu - stoch)

        
        if flavor == 'Put':
            payoff = np.maximum((K - ST),0)
        elif flavor == 'Call':
            payoff = np.maximum((ST - K),0)
        elif flavor == 'Call Antithetic':
            payoff = np.mean([np.maximum((ST - K),0),np.maximum((ST_ - K),0)],axis=0)
        elif flavor == 'AC':
            payoff = np.maximum((K - K*K/ST),0)
        elif flavor == 'AC Antithetic':
            payoff = np.mean([np.maximum((K - K*K/ST),0),np.maximum((K - K*K/ST_),0)],axis=0)
        elif flavor == 'Control Variate':
            payoff = (K/ST - alpha)* np.maximum((ST - K),0)
        elif flavor == 'Control Variate2':
            payoff = (K/ST - alpha)* np.maximum((ST - K),0) + St*np.exp(-q*dt)*norm.cdf(d1) - K*np.exp(-r*dt)*norm.cdf(d2)
        elif flavor == 'Power Call':
            payoff = np.maximum((ST*ST - K),0)
        else:
            payoff = ST
       
        PVs = np.exp(-r*dt)*payoff 
        price.append(np.mean(PVs))
        stdev.append(np.std(PVs)/np.sqrt(n))
        
    return price, stdev


##=================== Lookback Option Monte Carlo Function =====================##

def lookback_MC(batches, freqs, S,K,r,q,sigma,T,antithetic='no',flavor='c',style='float',display='no',t=0):
    '''Bare-bones Monte Carlo sim for a fixed strike lookback option.
    
    This function generate batches (a list) of n normally distributed 
    random variables, which are used to generate a geometric brownian motion 
    along n sample paths.
    
    Alternatively, you can run a list of frequencies for a given batch size.
    
    Then we calculate the maximum values, for each batch, along
    each n paths, for the specified frequency and number of steps.  

    We end up with a row vector of n maximums, 
    which are then used to calculate the discounted payoff, then we take the 
    mean of each of these.
    
    Ouputs price and standard error.'''
    
    num_steps = 252
    dt = T/num_steps
    
    
    
    
    Smax = []
    Smin = []
    
    
    
    
    payout = np.zeros(num_steps)
    payoutLookback = []
    lookbackMean = []
    lookbackVariance = []
    PV = []
    if antithetic == 'yes':
        ST = np.zeros((num_steps+1,batches))
        ST[0,:] = S
        
        
        for i in range(0,num_steps):
        
            mu = (r - q - sigma*sigma*0.5)*dt
            draws = np.random.normal(size=batches//2)
            stoch = sigma*np.sqrt(dt)*draws
            stoch = np.concatenate((stoch, -stoch),axis=0) ## antithetic variate##
            
       
            ST[i+1,:] = ST[i,:]*np.exp(mu + stoch)
            
            payout[i] = np.average(np.maximum((ST[i,:] - K),0))
    else:
        ST = np.zeros((num_steps+1,batches))
        ST[0,:] = S
        
        for i in range(0,num_steps):   
            mu = (r - q - sigma*sigma*0.5)*dt
            draws = np.random.normal(size=batches)
            stoch = sigma*np.sqrt(dt)*draws
            
            ST[i+1,:] = ST[i,:]*np.exp(mu + stoch)
            
            
            payout[i] = np.average(np.maximum((ST[i,:] - K),0))
    
    for m in freqs:
        Smax.append(np.max(ST[int(num_steps/m)-1:num_steps:int(num_steps/m)],0))
        Smin.append(np.min(ST[int(num_steps/m)-1:num_steps:int(num_steps/m)],0))

    for i in range(0,num_steps):
        PV.append(np.exp(-r*T)*payout[i])
    for i in range(0,len(freqs)):
        
        if antithetic=='yes':
            payouts = np.maximum((Smax[i]-K),0)
            avg_payouts = np.average([payouts[0:batches//2],payouts[batches//2:]],axis=0)
            payoutLookback.append(np.exp(-r*T)*avg_payouts)
        else:
            payoutLookback.append(np.exp(-r*T)*np.maximum((Smax[i] - K),0))
        lookbackVariance.append(np.std(payoutLookback[i]))
        lookbackMean.append(np.mean(payoutLookback[i]))
   
    return lookbackMean,lookbackVariance/np.sqrt(batches)



##================== Lookback Option Monte Carlo Function =====================##

def lookback_MC_CV_european(batches, freqs, S,K,r,q,sigma,T,CV_val,flavor='c',style='float',display='no',t=0):
    '''Monte Carlo sim for a fixed strike lookback option using a european 
    call as a control variate for variance reduction.
    
    This function generates batches of n normally distributed random variables, 
    which are used to generate a geometric brownian motion along n sample paths.  
    
    Then we calculate the maximum values, for each batch, alongeach n paths, 
    using the specified sampling frequency and number of steps.  
    
    Alternatively it can run the frequencies in batches for a specified batch size. 
    
    We end up with a row vector of n maximums, which are then used to calculate 
    the discounted payoffs.  
    
    Here we are also doing the same process as above for the simulated vanilla 
    european option.
    
    We take the mean of the diiference of these values, then add back the value 
    of the known control variate to get the true estimator. 

    This should significantly reduce variance, with marginal computational expense.'''
    
    
    num_steps = 252
    dt = T/num_steps
    
    ST = np.zeros((num_steps+1,batches))
    
    
    Smax = []
    Smin = []
    ST[0,:] = S
    
    
    
    payout = np.zeros(num_steps)
    payoutLookback = []
    lookbackMean = []
    lookbackVariance = []
    PV = []
    
    for i in range(0,num_steps):   
        mu = (r - q - sigma*sigma*0.5)*dt
        draws = np.random.normal(size=batches)
        stoch = sigma*np.sqrt(dt)*draws
    
        ST[i+1,:] = ST[i,:]*np.exp(mu + stoch)

    
    for m in freqs:
        Smax.append(np.max(ST[int(num_steps/m)-1:num_steps:int(num_steps/m)],axis=0))


    for i in range(0,len(freqs)):
        payoutLookback = np.exp(-r*T)*np.maximum((Smax[i] - K),0)
        payout_euro = np.exp(-r*T)*(np.maximum(ST[-1,:]-K,0))
        diff = payoutLookback - payout_euro
        CV_adj = np.mean(diff)
        lookbackVariance.append(np.std(diff))
        lookbackMean.append((CV_val + CV_adj))
   
    return lookbackMean,lookbackVariance/np.sqrt(batches)



##================== Lookback Option Monte Carlo Function =====================##

def lookback_MC_CV_bridge(batches, freqs, S,K,r,q,sigma,T,CV_val,flavor='c',style='float',display='no',t=0):
    '''Monte Carlo sim for a fixed-strike lookback option using a 
    Brownian Bridge technique as a control variate for variance reduction.
    
    This function generates batches of n normally distributed random variables, 
    which are used to generate a geometric brownian motion along n sample paths.  
    Then we calculate the maximum values, for each batch, along
    each n paths, for the specified frequency and number of steps.  
    
    Alternatively it can run the frequencies in batches for a specified batch size.  
    
    We end up with a row vector of n maximums, which are then used to calculate 
    each discounted payoff.  
    
    We use the Box-Muller method to generate normal distributions drawn from a 
    uniform[0,1] to fill in the gaps between sampling periods for variance reduction.  
    Same process as above, then subtract, and take the mean of the diff.
    
    Then we add back the known control variate value to get the true estimator.'''
    
    
    num_steps = 252
    dt = T/num_steps
    
    ST = np.zeros((num_steps,batches))
    
    
    Smax = []
    Smin = []
    ST[0,:] = S
    
    
    
    payout = np.zeros(num_steps)
    payoutLookback = []
    lookbackMean = []
    lookbackVariance = []
    PV = []
    
    for i in range(1,num_steps):   
        mu = (r - q - sigma*sigma*0.5)*dt
        draws = np.random.normal(size=batches)
        stoch = sigma*np.sqrt(dt)*draws
    
        ST[i,:] = ST[i-1,:]*np.exp(mu + stoch)

    
    for m in freqs:
        Smax.append(np.max(ST[int(num_steps/m)-1:num_steps:int(num_steps/m)],axis=0))


    for i in range(0,len(freqs)):
        m= freqs[i]
        if freqs[i]==1:
            x_t = (ST[-1,:] - ST[0,:])/ST[0,:]  # Normalize for starting at 0
            u = np.random.uniform(size=(batches)) # generate uniform rv on [0,1]
            # using inverse CDF method (Box-Muller) generate normal rv for the bridge
            b_max = (((x_t + np.sqrt(x_t**2-2*sigma**2 * T*np.log(u)))/2)*ST[0,:]) + ST[0,:] # Reverse normalization from above
        
        else:
            temp_ref = list(range(0,252))
            temp_index = temp_ref[int(num_steps/m)-1:num_steps:int(num_steps/m)]
            b_max = np.zeros((len(temp_index),batches))
            for j in range(len(temp_index)):
                x_t = (ST[temp_index[j],:]-ST[temp_index[j-1],:]) / (ST[temp_index[j-1],:])
                u = np.random.uniform(size=(batches)) # generate uniform rv on [0,1]
                # using inverse CDF method (Box-Muller) generate normal rv for the bridge
                b_max[j,:] = (((x_t + np.sqrt(x_t**2-2*sigma**2 * (T/m)*np.log(u)))/2)*ST[temp_index[j-1],:]) + ST[temp_index[j-1],:] # Reverse normalization from above
            
            b_max = np.max(b_max,axis=0)
        payoutLookback = np.exp(-r*T)*np.maximum((Smax[i] - K),0)
        payout_cts = np.exp(-r*T)*(np.maximum(b_max-K,0))
        diff = payout_cts - payoutLookback
        CV_adj = np.mean(diff)
        lookbackVariance.append(np.std(diff))
        lookbackMean.append((CV_val - CV_adj))
   
    return lookbackMean,lookbackVariance/np.sqrt(batches)


## ================ Closed-Form Lookback Function =============================##

def lookback_Analytic(S,K,r,q,sigma,T,flavor = 'fixed strike call',t=0):
    '''This function computes the continuous time analytical price for lookback 
    options.
    
    Input to select parameters, flavor = fixed/float and put/call.
    Outputs final price. '''
    
    dt = T-t
    d1 = ((np.log(S/K) + (r - q)*(dt)) / sigma*np.sqrt(dt)) + 0.5*sigma*np.sqrt(dt)
    d2 = ((np.log(S/K) + (r - q)*(dt)) / sigma*np.sqrt(dt)) - 0.5*sigma*np.sqrt(dt)
    d1_ = ((np.log(S/K) - (r - q)*(dt)) / sigma*np.sqrt(dt)) - 0.5*sigma*np.sqrt(dt)
    d2_ = ((np.log(S/K) - (r - q)*(dt)) / sigma*np.sqrt(dt)) + 0.5*sigma*np.sqrt(dt)
    
    euro_call = S*np.exp(-q*dt)*norm.cdf(d1) - K*np.exp(-r*dt)*norm.cdf(d2)
    euro_put = K*np.exp(-r*dt)*norm.cdf(d2_) - S*np.exp(-q*dt)*norm.cdf(d1_)
    
    lookback1 = np.exp(-r*dt)*(sigma*sigma/(2*r))*S
    lookback_c = (np.exp(r*dt)*norm.cdf(d1) - (np.power((S/K),(-(2*r)/(sigma*sigma))))*norm.cdf(d1-(((2*r)/(sigma))*np.sqrt(dt))))
    lookback_p = (np.power((S/K),(-(2*r)/(sigma*sigma))))*norm.cdf(-d1+(((2*r)/(sigma))*np.sqrt(dt)))-(np.exp(r*dt)*norm.cdf(-d1))
    
    floating_c = S*(2 + (sigma*sigma*dt*0.5))*norm.cdf((np.sqrt((sigma*sigma*dt))*0.5)) + S*np.sqrt((sigma*sigma*dt)/(2*math.pi))*np.exp((-sigma*sigma*dt*(1/8))) - S
    float_perc = (floating_c) / S
    
    if flavor == 'fixed strike call':
        price =  euro_call + lookback1*lookback_c
    if flavor == 'fixed strike put':
        price = euro_put + lookback1*(lookback_p)
    if flavor == 'floating strike call':
        price =  float_perc
    
    
    return price

##=============================================================================##
##============ C H A R T  A N D  T A B L E  G E N E R A T I O N ===============##
##=============================================================================##

def getLiquidity(sigList, TList, TNames,type=None):
    '''This program takes a list of volatilities and time horizons as inputs.
    Outputs dataframe summary of illiquidity premium sorted by time horizon (columns)
    and volatility (rows). 
    
    Set type to intraday (bps) or None (%)'''


    S = K = 1
    r = .023
    q = .0198
    
    
    
    Ts = np.zeros(len(TList))
    df = pd.DataFrame()
    
    for sig in sigList: 
        for i in range(len(TList)):
            Ts[i] = lookback_Analytic(S,K,r,q,sig,TList[i]/252,flavor = 'floating strike call',t=0)
        df[sig] = Ts
        
    if type == 'Intraday':
        df=df*10000
        df.name = 'Maximum Estimated Intraday Illiquidity Premium (Basis Points)'
        df.index = TNames
        df.index.name = 'Execution Time'
        
    else:
        df = df*100
        df.name = 'Maximum Estimated Long-Term Illiquidity Premium (Percentage)'
        df.index = TNames
        df.index.name = 'Lockout Period'
    df.columns.name = '$$\sigma =$$'
 
    print(df.name)
    return df.round(2)


##======= ILLUSTRATION OF ILLIQUIDITY PREMIUM "TIME DURATION" (INVERSE OF FI IR DURATION)"==========##

def plotLiquidity(sigList, TList, TNames, type=None):
    '''This program takes a list of volatilities and time horizons as inputs.
    Outputs plots of illiquidity premium (y-axis) vs time horizon (x-axis).
    
    Use with getCharts() function. 
    
    Set type to intraday (bps) or None (%)'''



    S = K = 1
    r = .023
    q = .0198
    
   
    

    
    Ts = np.zeros(len(TList))
    df = pd.DataFrame()
    
    for sig in sigList:
        for i in range(len(TList)):
            Ts[i] = lookback_Analytic(S,K,r,q,sig,TList[i]/252,flavor = 'floating strike call',t=0)
        df[sig] = Ts
    
    
    if type == 'Intraday':
        df=df*10000
        df.name = 'Maximum Estimated Intraday Illiquidity Premium (In Percentage Basis Points)'
            
        # create plots
        plt.plot(TList,df.loc[:,(.3,.2,.1,.05)])
        plt.ylabel('Intraday Illiquidity Premium (Bps)')
        plt.xlabel('Hours')
        plt.legend(['sigma = 0.30','sigma = 0.20','sigma = 0.10','sigma = 0.05'])
        plt.title('Upper Bound Illiquidity Premium vs Execution Time')
        plt.show()
    
    else:
        df = df*100
        df.name = 'Maximum Estimated Illiquidity Premium (In Percentage Points)'
   
        # create plots
        plt.plot(TList,df.loc[:,(.3,.2,.1,.05)])
        plt.ylabel('Illiquidity Premium (%)')
        plt.xlabel('Days')
        plt.legend(['sigma = 0.30','sigma = 0.20','sigma = 0.10','sigma = 0.05'])
        plt.title('Upper Bound Illiquidity Premium vs Lockout Period')
        plt.show()
    
    df.columns.name = '$$\sigma =$$'
    
    return df



def getCharts(sigList,TList1,TNames1,TList2,TNames2,type1=None, type2=None):
    '''This is a program that executes plotLiquidity() function specifically
    desined for a time series on the x-axis.'''
    # define x axis
    
    
    # define liquidity premium output
    dfplotI = plotLiquidity(sigList,TList1,TNames1,'Intraday')
    dfplot = plotLiquidity(sigList,TList2,TNames2)
    
    return dfplotI, dfplot

##============= SHOW CONVERGENCE PROPERTIES OF FIXED STRIKE CASE =============##


def generateMC_BS_chart():
    '''This program is predefined to run a MC sim for a lookback option,
    calulates the closed-form analytic solution for the same option,
    prices a vanilla european call with the same parameters using 
    Black-Sholes.  Generates a chart showing how increasing the
    sampling frequency of the discretely sampled lookback MC sim causes
    the values to shift from the Black-Scholes value (for m=1) towards 
    the value of the continuous lookback (as m grows large).'''
    
    # Define parameters and run simulations for each strike
    n = 100000
    S = 2885
    K = [S, S*1.1, S*1.2, S*1.3]
    r = .023
    q = .0198
    
    sigma = 0.157
    T = 1
    dt = T/252
    results = []
    resultsCF = []
    resultsBS = []

    for k in K:
        results.append(lookback_MC(n,[252,52,12,4,2,1],S,k,r,q,sigma,T))
        resultsCF.append(lookback_Analytic(S,k,r,q,sigma,T,'fixed strike call'))
        resultsBS.append(BS(S,k,r,q,sigma,T)[0])
        
        
    # pull values
    K_0c = lookback_MC(n,[252,52,12,4,2,1],S,K[0],r,q,sigma,T)
    K_1c = lookback_MC(n,[252,52,12,4,2,1],S,K[1],r,q,sigma,T)
    K_2c = lookback_MC(n,[252,52,12,4,2,1],S,K[2],r,q,sigma,T)
    K_3c = lookback_MC(n,[252,52,12,4,2,1],S,K[3],r,q,sigma,T)
    
    # create a list for charting
    k1 = list([K_0c[0],K_1c[0],K_2c[0],K_3c[0]])
    
    # generate plot
    plt.plot(K, resultsCF,'--d',K,k1, '', K, resultsBS,'g^')
    plt.title('Values of MC Simulated Fixed Strike Lookback Call with Increasing Frequecy (m)')
    plt.ylabel('Price')
    plt.xlabel('Strike')
    plt.legend(['Lookback','m=252', 'm=52', 'm=12', 'm=4', 'm=2', 'm=1', 'Black-Scholes'])
    plt.show()

##=== VISUAL REPRESENTATION OF VARIANCE REDUCTION FOR DIFFERENT STRIKES ======##

def generateVarChart():
    '''Variance reduction using antithetic variates.  This program is 
    predefined to run a MC sim for a lookback option, and generates a 
    chart which shows the effects of using antithetic variates 
    with various sampling frequencies.'''

    # Define parameters and run simulations for each strike
    n = 100000
    S = 2885
    K = [S, S*1.1, S*1.2, S*1.3]
    r = .023
    q = .0198
    sigma = 0.157
    T = 1
    dt = T/252
    results_a = []
    resultsCF = []
    resultsBS = []

    for k in K:
        results_a.append(lookback_MC(n,[252,52,12,4,2,1],S,k,r,q,sigma,T,antithetic='yes'))
        resultsCF.append(lookback_Analytic(S,k,r,q,sigma,T,'fixed strike call'))
        resultsBS.append(BS(S,k,r,q,sigma,T)[0])
        
    # pull values
    K_0 = lookback_MC(n,[252,52,12,4,2,1],S,K[0],r,q,sigma,T)
    K_1 = lookback_MC(n,[252,52,12,4,2,1],S,K[1],r,q,sigma,T)
    K_2 = lookback_MC(n,[252,52,12,4,2,1],S,K[2],r,q,sigma,T)
    K_3 = lookback_MC(n,[252,52,12,4,2,1],S,K[3],r,q,sigma,T)

    # pull values antithetic
    K_0a = lookback_MC(n,[252,52,12,4,2,1],S,K[0],r,q,sigma,T,antithetic='yes')
    K_1a = lookback_MC(n,[252,52,12,4,2,1],S,K[1],r,q,sigma,T,antithetic='yes')
    K_2a = lookback_MC(n,[252,52,12,4,2,1],S,K[2],r,q,sigma,T,antithetic='yes')
    K_3a = lookback_MC(n,[252,52,12,4,2,1],S,K[3],r,q,sigma,T,antithetic='yes')

    # create a list for charting
    k1 = list([K_0[0],K_1[0],K_2[0],K_3[0]])
    k1se = list([K_0[1],K_1[1],K_2[1],K_3[1]])
    k1A = list([K_0a[0],K_1a[0],K_2a[0],K_3a[0]])
    k1seA = list([K_0a[1],K_1a[1],K_2a[1],K_3a[1]])
    
    # generate plot
    plt.plot(K,k1se, '', K, k1seA,'g^--')
    plt.title('Standard Errors of MC Simulated Fixed Strike Lookback Call with Increasing Frequecy (m)\nPlotted against the Antithetic SEs')
    plt.ylabel('Standard Error')
    plt.xlabel('Strike')
    plt.legend(['m=252', 'm=52', 'm=12', 'm=4', 'm=2', 'm=1', 'Antithetic'])
    plt.show()


##============= MONTE CARLO SIM TO EVAULUATE ILLIQUIDITY PREMIUM =============##

def runLiquidity_MC():
    '''This program is predefined to set up the necessary control variates,
    run a MC sim for a fixed strike lookback option and compute the
    BS price, continuous lookback price, and discretely sampled sim
    prices.  The goal is to compute an upper bound for the expected
    illiquidity premium (in percentage terms for various windows of time*)
    by comparing the simulated discretely sampled lookback prices 
    to those of the continuous analytical price.
    
    Generates a table with summary statistics.
    
    *For intraday windows: this can be viewed as the max premium to pay
    to compensate for the time lag it would take to execute an illiquid position.
    
    *For +1 day time horizon: this can be viewed as a restricted stock or
    investment with a lockout period.
    
    As sampling freq (m) approaches zero: the sim price moves closer to the BS
    value (illiquid) and further from the continuous analytical value (liquid)
    
    As sampling freq (m) approaches days-to-maturity, the sim price moves closer to
    the continuous value (liquid) and further from the BS value (illiquid)'''
    
    # Define parameters and run simulations for each strike
    # Define parameters and run simulations for each strike
    n = 100000
    S = 2885
    K = [S, S*1.1, S*1.2, S*1.3]
    r = .023
    q = .0198
    
    sigma = 0.157
    T = 1
    dt = T/252
    results_a = []
    resultsCF = []
    resultsBS = []

    
    K_0a = []

    for k in K:
        results_a.append(lookback_MC(n,[252,52,12,4,2,1],S,k,r,q,sigma,T,antithetic='yes'))
        resultsCF.append(lookback_Analytic(S,k,r,q,sigma,T,'fixed strike call'))
        resultsBS.append(BS(S,k,r,q,sigma,T)[0])
    
    # control variates
    cv_valBS = BS(2885,2885,r,q,sigma,T,flavor='c',style='euro',display='no',t=0)[0]
    cv_valLook = lookback_Analytic(2885,2885,r,q,sigma,T,flavor = 'fixed strike call',t=0)
        
    n = 100000
    K_0 = lookback_MC(n,[252,52,12,4,2,1],S,K[0],r,q,sigma,T) # no variance reduction
    K_0a = lookback_MC(n,[252,52,12,4,2,1],S,K[0],r,q,sigma,T,antithetic='yes') # antithetic variates
    K_0e = lookback_MC_CV_european(n,[252,52,12,4,2,1],S,K[0],r,q,sigma,T,cv_valBS) # euro control variate
    K_0bb = lookback_MC_CV_bridge(n,[252,52,12,4,2,1],S,K[0],r,q,sigma,T,cv_valLook) # brownian bridge control variate
    
    
    n = 1000000
    K_0_ = lookback_MC(n,[252,52,12,4,2,1],S,K[0],r,q,sigma,T) # no variance reduction
    K_0a_ = lookback_MC(n,[252,52,12,4,2,1],S,K[0],r,q,sigma,T,antithetic='yes') # antithetic variates
    K_0e_ = lookback_MC_CV_european(n,[252,52,12,4,2,1],S,K[0],r,q,sigma,T,cv_valBS) # euro control variate
    K_0bb_ = lookback_MC_CV_bridge(n,[252,52,12,4,2,1],S,K[0],r,q,sigma,T,cv_valLook) # brownian bridge control variate
    
    
    percI = (cv_valLook / np.array(K_0[0])) - 1 #[0]
    perc = (cv_valLook / np.array(K_0_[0])) - 1
    
    # Display Results
    df = pd.DataFrame({'MC Price:':K_0[0], 'Illiquidity Premium (%)': perc,'Std Error':K_0[1] },index = ['Daily', 'Weekly', 'Monthly', 'Quarterly','Semi-Annual','Maturity'])
    df.index.name = 'Sampling Freq'

    print('\nBS Value for Vanilla Euro Call:',cv_valBS.round(2)) #[0]
    print('\nClosed-Form Value for Fixed Strike Lookback Call:',cv_valLook.round(2)) #[0]

    print('\nImplied Upper Bound Illiquidity Premium (Percentage Points) and Standard Errors Using Variance Reduction Techniques:')
    print()
    
    # n = 100k
    df1 = pd.DataFrame({'MC Price (n=100k):':K_0[0],'Illiquidity Premium': percI,'Standard Error':K_0[1],'Std Error (Antithetic)':K_0a[1],'Std Error (Euro CV)':K_0e[1],'Std Error (Brownian Bridge)':K_0bb[1]},index = ['Daily', 'Weekly', 'Monthly', 'Quarterly','Semi-Annual','Maturity'])
    df1.index.name = 'Sampling Freq'
    
    # n = 1mm
    df2 = pd.DataFrame({'MC Price (n=1mm):':K_0_[0],'Illiquidity Premium': perc,'Standard Error':K_0_[1],'Std Error (Antithetic)':K_0a_[1],'Std Error (Euro CV)':K_0e_[1],'Std Error (Brownian Bridge)':K_0bb_[1]},index = ['Daily', 'Weekly', 'Monthly', 'Quarterly','Semi-Annual','Maturity'])
    df2.index.name = 'Sampling Freq'

    return df1.round(2), df2.round(2)
    





