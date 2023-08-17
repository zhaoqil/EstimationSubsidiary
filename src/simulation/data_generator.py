import numpy as np

class dataGenerator():
    
    def __init__(self, dim, n, p_Y, p_Z):
        self.dim = dim
        self.n = n
        self.p_Y = p_Y
        self.p_Z = p_Z
    
    def generate(self):
        if self.dim > 1:
            X_s = np.random.multivariate_normal(mean=np.zeros(self.dim), cov=np.eye(self.dim), size=self.n)
            A_s = np.zeros(self.n)
            for i in range(self.n):
                A_s[i] = np.random.binomial(1, 1/(1+np.exp(-X_s[i,0])+np.exp(-2*X_s[i,1])+np.exp(-X_s[i,2])))  
        else:
            X_s = np.random.uniform(-1, 1, self.n)
            A_s = np.random.binomial(1, 0.5+0.1*X_s)
        Y_s = self.p_Y(A_s, X_s)
        Z_s = self.p_Z(A_s, X_s)
        return X_s, A_s, Y_s, Z_s
    