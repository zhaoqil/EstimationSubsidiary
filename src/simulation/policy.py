import numpy as np

class ruleGenerator():

    def __init__(self, form, m_lst):
        self.form = form
        if self.form == "Indicator":
            self.it = iter(m_lst)
            self.finite = True
            self.end = False
            self.m_lst = m_lst
        elif self.form == "Tree":
            self.m_lst = m_lst
            self.s = [True, True]

    def generate(self, i):
        if self.form == "Indicator":
            m = self.m_lst[i]
            d = lambda x: x>=m
        elif self.form == "Tree":
            m = self.m_lst[i]
            tree_predictor = tree(m, self.s)
            d = lambda x: tree_predictor.predict(x)

            #val = next(self.it, None)
            #if val is None:
            #    self.end = True
            #    d = None
            #else:
            #    d = lambda x: x>=val
        return d

class tree():
    
    def __init__(self, t, s):
        """
        t: set of cut-off thresholds, d-dimensional vector [t1, t2, ...]
        s: set of boolean signs determining > or < [> ,<, >, ...]
        """
        self.t = t
        self.s = s
        
    def predict(self, X):
        """
        X: vector, each row should be d-dimensional
        """
        if X.ndim == 1:
            raise ValueError("X should be 2-dimensional array!")
        res = np.logical_xor(X < self.t, self.s) # an array containing trues and falses
        return np.prod(res, axis=1)
