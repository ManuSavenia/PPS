import numpy as np
import time

from matplotlib import pylab as plt
from IPython import display

from grafica import *

class NeuronaLineal(object):
    """
    Parameters
    ------------
    alpha : float
        Learning rate (between 0.0 and 1.0)
    n_iter : int
        Passes over the training dataset.
    cotaE : float
        Minimum error threshold for convergence.
    random_state : int
        Random number generator seed for random weight initialization.
    draw : int
        1 si dibuja -  0 si no
    title : list con 2 elementos
        titulos de los ejes - sólo 2D
    w_init : array-like, optional
        Pesos iniciales custom (longitud = n_features)
    b_init : float, optional
        Bias inicial custom
        
    Attributes
    -----------
    w_ : 1d-array
        Weights after fitting.
    b_ : float
        Bias after fitting.
    errors_ : list
        Error cuadrático en cada época.
    """
    def __init__(self, alpha=0.01, n_iter=50, cotaE=1e-7, random_state=None, draw=0, title=['X1','X2'], w_init=None, b_init=None):
        self.alpha = alpha
        self.n_iter = n_iter
        self.cotaE = cotaE
        self.random_state = random_state
        self.draw = draw
        self.title = title
        self.w_init = w_init
        self.b_init = b_init

        # si me pasan pesos y bias -> los uso directamente
        if w_init is not None:
            self.w_ = np.array(w_init, dtype=float)
        else:
            self.w_ = None
        if b_init is not None:
            self.b_ = float(b_init)
        else:
            self.b_ = None


    def fit(self, X, y):
        """Fit training data"""
        if (self.draw):
            ycol = y.reshape(-1,1)
            puntos = np.concatenate((X,ycol), axis=1)
            T = np.zeros(X.shape[0])
            
        rgen = np.random.RandomState(self.random_state)

        # inicialización de pesos y bias
        if self.w_init is not None:
            self.w_ = np.array(self.w_init, dtype=float)
        else:
            self.w_ = rgen.uniform(-0.5, 0.5, size=X.shape[1]) 
        
        if self.b_init is not None:
            self.b_ = float(self.b_init)
        else:
            self.b_ = rgen.uniform(-0.5, 0.5)

        self.errors_ = []
        ph = 0  
        ErrorAnt, ErrorAct = 0, 1
        
        i = 0
        while (i < self.n_iter) and (np.abs(ErrorAnt - ErrorAct) > self.cotaE):
            ErrorAnt = ErrorAct
            ErrorAct = 0
            for xi, target in zip(X, y):
                errorXi = (target - self.predict(xi))
                update = self.alpha * errorXi
                self.w_ += update * xi
                self.b_ += update
                ErrorAct += errorXi**2
                
            self.errors_.append(ErrorAct)
            
            if (self.draw):
                ph = dibuPtosRecta(puntos, T, np.array([self.w_, -1],dtype=object), self.b_, self.title, ph)
            
            i += 1
        return self

    def net_input(self, X):
        """Calculate net input"""
        return np.dot(X, self.w_) + self.b_

    def predict(self, X):
        """Return continuous output"""
        return self.net_input(X)
