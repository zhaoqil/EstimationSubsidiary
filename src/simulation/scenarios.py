import numpy as np

class Scenario():
    
    def __init__(self, typ, grid_length=501):
        self.typ = typ
        self.m_lst = np.linspace(-1, 1, grid_length)
        if (typ == "nonunique"):
            self.p_Y_out = lambda A, X: self.p_Y(A, X, self.mean_func_nonuni)
            self.p_Z_out = self.p_Z_nonuni
            self.mean_func_out = self.mean_func_nonuni
            self.mean_func_Z_out = self.mean_func_Z_nonuni
        elif (typ == "unique_corr"):
            self.p_Y_out = lambda A, X: self.p_Y(A, X, self.mean_func_uni)
            self.p_Z_out = self.p_Z_uni
            self.mean_func_out = self.mean_func_uni
            self.mean_func_Z_out = self.mean_func_Z_uni
        elif (typ == "unique_noncorr"):
            self.p_Y_out = lambda A, X: self.p_Y(A, X, self.mean_func_uni)
            self.p_Z_out = self.p_Z_uni_2
            self.mean_func_out = self.mean_func_uni
            self.mean_func_Z_out = self.mean_func_Z_uni_2
        elif (typ == "high_dim"):
            self.p_Y_out = lambda A, X: self.p_Y_high(A, X, self.mean_func_Y_high)
            self.p_Z_out = lambda A, X: self.p_Z_high(A, X, self.mean_func_Z_high)
            self.mean_func_out = self.mean_func_Y_high
            self.mean_func_Z_out = self.mean_func_Z_high
        elif (typ == "high_dim_nonmargin"):
            self.p_Y_out = lambda A, X: self.p_Y_high(A, X, self.mean_func_Y_high_2)
            self.p_Z_out = lambda A, X: self.p_Z_high(A, X, self.mean_func_Z_high_2)
            self.mean_func_out = self.mean_func_Y_high_2
            self.mean_func_Z_out = self.mean_func_Z_high_2
        else:
            raise NotImplementedError("Method not implemented!")
        
    def p_Y(self, A, X, mean_func):
        # conditional distribution of Y|A,X
        return np.random.normal(loc=mean_func(A, X), scale=0.25)

    def d_anal(self, x, p):
        # analytical form of the optimal rule
        # p: mean function that takes in A and X
        return (p(1, x) - p(0, x) >= 0)

    # Scenario 1: non-unique policy
    def p_Z_nonuni(self, A, X):
        # conditional distribution of Z|A,X
        # E[Z|A,X]=(X-0.01)**(1/3)*A so that s_b(x)=(X-0.01)**(1/3)*A
        # note that s is not bounded by q in this case
        return np.random.normal(loc=(4/(1+np.exp(-100*(X-0.3)))-2)*A, scale=0.25)

    def mean_func_nonuni(self, A, X):
        return ((X+0.4)*(X<=-0.4)+X*(X>=0))*A
    
    def mean_func_Z_nonuni(self, A, X):
        return (4/(1+np.exp(-100*(X-0.3)))-2)*A

    # Scenario 2: unique policy, Y and Z correlated
    def p_Z_uni(self, A, X):
        # conditional distribution of Z|A,X
        # E[Z|A,X]=(X-0.01)**(1/3)*A so that s_b(x)=(X-0.01)**(1/3)*A
        # note that s is not bounded by q in this case
        # return np.random.normal(loc=np.exp(-2*np.abs(X))*A, scale=0.25)
        # return np.random.normal(loc=np.exp(-2*np.abs(X))*A, scale=0.25)
        return np.random.normal(loc=(1/(1+np.exp(-100*(X)))-1/2)*A, scale=0.25)
    
    def p_Y_uni(self, A, X):
        # conditional distribution of Y|A,X
        return np.random.normal(loc=X*A, scale=0.16)
    
    def mean_func_uni(self, A, X):
        return X*A
    
    def mean_func_Z_uni(self, A, X):
        return (1/(1+np.exp(-100*(X)))-1/2)*A

    # Scenario 3: unique policy, Y and Z not correlated
    def p_Z_uni_2(self, A, X):
        # conditional distribution of Z|A,X
        return np.random.normal(loc=((X+0.2)*(X<=-0.2)+(X-0.2)*(X>=0.2))*2*A, scale=0.25)
    
    def mean_func_Z_uni_2(self, A, X):
        return ((X+0.2)*(X<=-0.2)+(X-0.2)*(X>=0.2))*2*A
    
    def p_A(self, X):
        # conditional distribution of A|X
        return 0.5+0.1*X
    
    # Scenario 4: high dimensional data
    def mean_func_Y_high(self, A, X):
        # conditional distribution of Y|A,X
        t1 = A * X[:,0]
        t2 = A * X[:,1]
        t3 = A * X[:,2]
        return 0.2 * np.sign(t1) + 0.1 * np.sign(t2) - 0.05 * np.sign(t3) + 0.36

    def mean_func_Z_high(self, A, X):
        t1 = A * X[:,0]
        t2 = A * X[:,1]
        t3 = A * X[:,2]
        return 0.12 * np.sign(t1) + 0.1 * np.sign(t2) + 0.18 * np.sign(t3) + 0.42

    def p_Y_high(self, A, X, mean_func):
        # conditional distribution of Y|A,X
        return np.random.binomial(1, mean_func(A, X))
    
    def p_Z_high(self, A, X, mean_func):
        # conditional distribution of Z|A,X
        return np.random.binomial(1, mean_func(A, X))
    
    def p_A_high(self, X):
        # conditional distribution of A|X
        n = X.shape[0]
        A = np.zeros(n)
        for i in range(n):
            A[i] = np.random.binomial(1, 1/(1+np.exp(-X[i,0])+np.exp(-2*X[i,1])+np.exp(-X[i,2])))
        return A

    
    # Scenario 5: high dimensional data without margin condition
    def mean_func_Y_high_2(self, A, X):
        # conditional distribution of Y|A,X
        t1 = A * X[:,0]
        t2 = A * X[:,1]
        t3 = A * X[:,2]
        return 0.003 * ((t1>=0).astype(int)**2 + (t2>=0).astype(int)**2 + (t3>=0).astype(int)**2 - 0.9*np.all([t1==0, t2==0, t3==0],axis=0)) + 0.36

    def mean_func_Z_high_2(self, A, X):
        t1 = A * X[:,0]
        t2 = A * X[:,1]
        t3 = A * X[:,2]
        return 0.1 * (np.sign(t1+0.2)+1) + 0.1 * (np.sign(t2+0.2)+1) + 0.15 * (np.sign(t3+0.2)+1) + 0.22