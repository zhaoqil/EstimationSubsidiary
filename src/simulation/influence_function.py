import numpy as np
from sklearn.neighbors import KernelDensity
from sklearn.linear_model import LogisticRegression
import xgboost as xgb


class influenceFunction():

    def __init__(self, x, a, y=None, p=None):
        """
        p: use simulation to generate y.
        y: use real data y (or z). 
        """
        self.x = x
        self.a = a
        self.p = p
        self.y = y

    def estimate_pa_x_xgb(self):
        # use XGBoost to estimate pa_x - not performing as well as kde!
        pa_x = xgb.XGBClassifier(learning_rate=0.02, max_depth=3, n_estimators=40)
        pa_x.fit(self.x, self.a)
        return pa_x

    def estimate_pa_x(self, bandwidth=None):
        if bandwidth==None:
            bandwidth = 0.2 # tuned best for 1D setting
        kde_x = KernelDensity(bandwidth=bandwidth, kernel="gaussian")
        kde_ax = KernelDensity(bandwidth=bandwidth, kernel="gaussian")
        if np.ndim(self.x) == 1:
            x = self.x[:, None].copy()
        else:
            x = self.x.copy()
            
        kde_x.fit(x)
        kde_ax.fit(x[self.a==1])
        log_prob = kde_ax.score_samples(x) + np.log(np.mean(self.a)) - kde_x.score_samples(x)
        p1_x = np.exp(log_prob)
        return p1_x

    def estimate_p(self):
        # different best tuning parameters
        if np.ndim(self.x) == 1:
            learning_rate = 0.05
            max_depth = 2
        else:
            learning_rate = 0.02
            max_depth = 3
        py_1x = xgb.XGBRegressor(n_estimators=40, learning_rate=learning_rate, max_depth=max_depth)
        py_0x = xgb.XGBRegressor(n_estimators=40, learning_rate=learning_rate, max_depth=max_depth)
        #self.py_1x = xgb.XGBRegressor()
        #self.py_0x = xgb.XGBRegressor()
        X_1 = self.x[self.a==1].copy()
        y_1 = self.y[self.a==1].copy()
        X_0 = self.x[self.a==0].copy()
        y_0 = self.y[self.a==0].copy()
        if np.ndim(X_1) == 1:
            py_1x.fit(X_1.reshape(-1, 1), y_1)
            py_0x.fit(X_0.reshape(-1, 1), y_0)
        else:
            py_1x.fit(X_1, y_1)
            py_0x.fit(X_0, y_0)
        return py_1x, py_0x
        
    def p_hat(self, a, x, py_1x, py_0x):
        res = np.zeros(len(a))
        ind1 = np.where(a==1)[0]
        ind0 = np.where(a==0)[0]
        X_1 = x[a==1]
        X_0 = x[a==0]
        if np.ndim(X_1) == 1:
            res[ind1] = py_1x.predict(X_1.reshape(-1, 1))
            res[ind0] = py_0x.predict(X_0.reshape(-1, 1))
        else:
            res[ind1] = py_1x.predict(X_1)
            res[ind0] = py_0x.predict(X_0)
        return res
        

    def compute_influence_fun(self, d, x, a, y, p1_x, py_1x, py_0x):
        """
        p: could be p_Y or p_Z. 
        d: some rule. 
        y: here could be Y or Z. 
        """
        if self.p is None:
            self.p = self.p_hat

        pa_x = p1_x * (a==1) + (1-p1_x) * (a==0)
        return (a == d(x))/pa_x * (y-self.p(a, x, py_1x, py_0x))+self.p(d(x), x, py_1x, py_0x) - np.mean(self.p(d(x), x, py_1x, py_0x))

    def compute_phi_hat(self, d, x, a, y, p1_x, py_1x, py_0x):
        if self.p is None:
            self.p = self.p_hat
        #pa_x = self.pa_x.predict(x)
        pa_x = p1_x * (a==1) + (1-p1_x) * (a==0)
        return (a == d(x))/pa_x * (y-self.p(a, x, py_1x, py_0x))+self.p(d(x), x, py_1x, py_0x)