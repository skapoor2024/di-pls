# -*- coding: utf-8 -*-
"""
Bottleneck Analytics GmbH
info@bottleneck-analytics.com
@author: Dr. Ramin Nikzad-Langerodi
"""

# Modules
import numpy as np
import matplotlib.pyplot as plt
import di_pls_func_2 as algo
import scipy.stats


class model:
    def __init__(self, x, y, xs, xt, A):
        self.x = x                         # Labeled X-Data (usually x = xs)
        self.n = np.shape(x)[0]            # Number of X samples
        self.ns = np.shape(xs)[0]          # Number of Xs samples
        self.nt = np.shape(xt)[0]          # Number of Xs samples
        self.k = np.shape(x)[1]            # Number of X variables
        self.y = y                         # Response variable corresponding to X data
        self.xs = xs                       # Source domain X data
        self.xt = xt                       # Target domain X data
        self.mu = np.mean(x, 0)            # Column means of x
        self.mu_s = np.mean(xs, 0)         # Column means of xs
        self.mu_t = np.mean(xt, 0)         # Column means of xt 
        self.std = np.std(x,axis = 0,ddof = 1) # column std of x
        self.std_s = np.std(xs,axis = 0,ddof = 1) # column std of xs
        self.std_t = np.std(xt,axis = 0,ddof = 1)# #column std of xt

        self.T = []                        # Projections (scores)
        self.Ts = []                       # Source domain scores
        self.Tt = []                       # Target domain scores
        self.P = []                        # Loadings
        self.Ps = []                       # Source domain loadings
        self.Pt = []                       # Target domain loadings
        self.W = []                        # Weights
        self.A = A                         # Number of LVs in the model
        self.opt_l = []                    # Optimal set of regularization parameters 
        self.b0 = np.mean(y,0)           # Offset
        self.bstd = np.std(y,axis = 0,ddof =1)
        self.bstd[self.bstd == 0.0] = 1.0  # std for the y
        self.b = []                        # Regression coefficients
        self.yhat= []                      # Predicted response values
        self.rmsec = []                    # Root Mean Squared Error of Calibration
        self.C = []                        # Inner relationship coefficients such that y = c*T


    def fit(self, l=0, centering=True):
        """
        Fit di-PLS model.
        
        
        Parameters
        ----------
        l: float or numpy array (1 x A)
            Regularization parameter. Either a single or different l's for each
            can be passed
            
        centering: bool
            If True Source and Target Domain Data are Mean Centered (default)
            
        heuristic: bool
            If True the regularization parameter is set to a heuristic value
                        
        """
           
        # Mean Centering
        #b0 = np.mean(self.y)
        y = self.y - self.b0
        y = y/self.bstd

        if centering is True:

            x = self.x[..., :] - self.mu
            x = x/self.std
            xs = self.xs[..., :] - self.mu_s
            xs = xs/self.std_s
            xt = self.xt[..., :] - self.mu_t
            xt = xt / self.std_t


        else:

            x = self.x 
            xs = self.xs
            xt = self.xt

    
        # Fit model and store matrices
        A = self.A
        (b, T, Ts, Tt, W, P, Ps, Pt, E, Es, Et, Ey, C, opt_l, discrepancy) = algo.dipals(x, y, xs, xt, A, l)
        self.b = b*self.bstd
        #self.b0 = b0
        self.T = T
        self.Ts = Ts
        self.Tt = Tt
        self.W = W
        self.P = P
        self.Ps = Ps
        self.Pt = Pt
        self.E = E
        self.Es = Es
        self.Et = Et
        self.Ey = Ey
        self.C = C
        self.discrepancy = discrepancy
        
        
        self.opt_l = opt_l

            
    def predict(self, x_test, y_test=[], rescale='Target'):
        """
        Predict function for di-PLS models
        
        Parameters
        ----------
        
        x_test: numpy array (N x K)
            X data
            
        y_test: numpy array (N x 1)
            Y data (optional)
            
        rescale: str or numpy.ndarray
            Determines Rescaling of the Test Data (Default is Rescaling to Target Domain Training Set)
            If Array is passed, than Test Data will be Rescaled to mean of the provided Array
        Returns
        -------
    
        yhat: numpy array (N x 1)
            Predicted Y
            
        
        RMSE: int
            Root mean squared error             
        """
        
        # Rescale Test data
        if(type(rescale) is str):

            if(rescale == 'Target'):

                Xtest = x_test[...,:] - self.mu_t
                Xtest = Xtest/self.std_t

            elif(rescale == 'Source'):

                Xtest = x_test[...,:] - self.mu_s
                Xtest = Xtest/self.std_s

            elif(rescale == 'none'):

                Xtest = x_test

        elif(type(rescale) is np.ndarray):

             Xtest = x_test[...,:] - np.mean(rescale,0)
             Xtest = Xtest/np.std(Xtest,axis =0,ddof = 1)

        else: 

            raise Exception('rescale must either be Source, Target or a Dataset')
            
        
        yhat = Xtest@self.b + self.b0

        if y_test is np.ndarray:

            error = algo.rmse(yhat,y_test)

        else:

            error = np.nan
        

        return yhat,error