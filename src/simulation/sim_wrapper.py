import numpy as np
from sim import simulation
from scipy.stats import norm
import multiplier_bootstrap as mb
from scenarios import Scenario
from influence_function import influenceFunction
import pandas as pd
import nlopt
from functools import partial
from policy import ruleGenerator, tree
import scipy.stats as stats

class simWrapper():
    
    def __init__(self, scene_typ, method_typ, arggs, grid_length=501, sample_size=2000, bootstrap_sample_size=1000, n_iters=1000):
        scene = Scenario(scene_typ, grid_length)
        self.scene_typ = scene_typ
        self.mean_func = scene.mean_func_out
        self.mean_func_Z = scene.mean_func_Z_out
        self.p_Y = scene.p_Y_out
        self.p_Z = scene.p_Z_out
        self.m_lst = scene.m_lst
        
        if scene_typ == "high_dim" or scene_typ == "high_dim_nonmargin":
            self.dim = 3 # some fixed dimension
            self.p_A = scene.p_A_high
        else:
            self.dim = 1
            self.p_A = scene.p_A
        
        if scene_typ == "high_dim" or scene_typ == "high_dim_nonmargin":
            nn = 100000
            X = np.random.multivariate_normal(mean=np.zeros(self.dim), cov=np.eye(self.dim), size=nn)
            A = np.zeros(nn)
            for i in range(nn):
                A[i] = np.random.binomial(1, 1/(1+np.exp(-X[i,0])+np.exp(-2*X[i,1])+np.exp(-X[i,2])))
            
        else:
            X = np.random.uniform(-1, 1, 1000000)
            A = np.random.binomial(1, self.p_A(X))
        
        self.X = X

        # THE USE OF d_anal AND d_anal_2 ASSUMES MONOTONICITY! - SCENARIO-SPECIFIC, NOT GENERAL!
        Y = self.p_Y(self.d_anal(X, self.mean_func), X)
        self.EY = np.mean(Y)
        Z = self.p_Z(self.d_anal(X, self.mean_func), X)
        self.EZ = np.mean(Z)
        Z_l = self.p_Z(self.d_anal_2(X, self.mean_func), X)
        self.EZ_l = np.mean(Z_l)
        
        self.method_typ = method_typ
        self.K = grid_length
        self.n = sample_size
        self.B = bootstrap_sample_size
        self.n_iters = n_iters
        self.arggs = arggs
                
    def run(self, alpha, typ="Indicator"):
        self.sim_lst = list()
        
        beta = alpha / 5

        sim = simulation(scene_typ=self.scene_typ, EZ=self.EZ, n=self.n, n_iters=self.n_iters, m_lst=self.m_lst, K=self.K, B=self.B, typ=typ, method_typ=self.method_typ, dim=self.dim)
        sim.run(alpha)
        self.sim_lst.append(sim)
        low = min(self.EZ_l, self.EZ)
        high = max(self.EZ_l, self.EZ)
        if self.scene_typ == "high_dim" or self.scene_typ == "high_dim_nonmargin":
            if self.method_typ == "os":
                self.compute_oracle_high_dim(alpha, self.n_iters)
                self.os_coverage = np.mean((self.EZ >= sim.lcb_z) * (self.EZ <= sim.ucb_z))
                self.os_width = np.mean(sim.ucb_z - sim.lcb_z)
                df = pd.DataFrame({'alpha': alpha, 'os coverage': self.os_coverage, 'os width': self.os_width, 'oracle coverage': self.oracle_coverage, 'oracle width': self.oracle_width}, index=[0])
                df.to_csv("{}/os_oracle_alpha={}_scene_{}.csv".format(self.arggs, alpha, self.scene_typ))
            else:
                self.band_coverage = np.mean((low >= sim.lcb_z) * (high <= sim.ucb_z))
                self.band_width = np.mean(sim.ucb_z - sim.lcb_z)
                df = pd.DataFrame({'alpha': alpha, 'coverage': self.band_coverage, 'width': self.band_width}, index=[0])
                df.to_csv("{}/alpha={}_method_{}_scene_{}.csv".format(self.arggs, alpha, self.method_typ, self.scene_typ))
        else:
            if self.scene_typ == "nonunique":
                self.band_coverage = np.mean((low >= sim.lcb_z) * (high <= sim.ucb_z))
                self.band_coverage_unif = np.mean((low >= sim.lcb_unif) * (high <= sim.ucb_unif))
            else:
                self.band_coverage = np.mean((self.EZ >= sim.lcb_z) * (self.EZ <= sim.ucb_z))
                self.band_coverage_unif = np.mean((self.EZ >= sim.lcb_unif) * (self.EZ <= sim.ucb_unif))
            self.band_width = np.mean(sim.ucb_z - sim.lcb_z)
            self.band_width_unif = np.mean(sim.ucb_unif - sim.lcb_unif)
            self.len_pi_hat = np.mean(sim.len_pi_hat)

            if self.method_typ == "os":
                self.compute_oracle(alpha, self.n_iters)
                if self.scene_typ == "nonunique":
                    self.os_coverage = np.mean((low >= sim.lcb_z) * (high <= sim.ucb_z))
                else:
                    self.os_coverage = np.mean((self.EZ >= sim.lcb_z) * (self.EZ <= sim.ucb_z))
                self.os_width = np.mean(sim.ucb_z - sim.lcb_z)
                df = pd.DataFrame({'alpha': alpha, 'os coverage': self.os_coverage, 'os width': self.os_width, 'oracle coverage': self.oracle_coverage, 'oracle width': self.oracle_width}, index=[0])
                df.to_csv("{}/os_oracle_alpha={}_scene_{}.csv".format(self.arggs, alpha, self.scene_typ))
            elif self.method_typ == "os_split":
                if self.scene_typ == "nonunique":
                    self.os_sp_coverage = np.mean((low >= sim.lcb_z) * (high <= sim.ucb_z))
                else:
                    self.os_sp_coverage = np.mean((self.EZ >= sim.lcb_z) * (self.EZ <= sim.ucb_z))
                self.os_sp_width = np.mean(sim.ucb_z - sim.lcb_z)
                df = pd.DataFrame({'alpha': alpha, 'os sp coverage': self.os_sp_coverage, 'os sp width': self.os_sp_width}, index=[0])
                df.to_csv("{}/os_split_alpha={}_scene_{}.csv".format(self.arggs, alpha, self.scene_typ))
            else:
                self.policy_cov = self.calc_policy_coverage(self.mean_func, self.sim_lst)
                df = pd.DataFrame({'alpha': alpha, 'coverage': self.band_coverage, 'width': self.band_width, 'policy coverage': self.policy_cov, 'coverage_unif': self.band_coverage_unif, 'width_unif': self.band_width_unif, 'len_pi_hat': self.len_pi_hat})
                df.to_csv("{}/alpha={}_method_{}_scene_{}.csv".format(self.arggs, alpha, self.method_typ, self.scene_typ))

    def d_anal(self, x, p):
        # analytical form of the optimal rule
        # p: mean function that takes in A and X
        return (p(1, x) - p(0, x) >= 0)
    
    def d_anal_2(self, x, p):
        # analytical form of the optimal rule
        # p: mean function that takes in A and X
        return (p(1, x) - p(0, x) > 0)
    
    def find_all_max(self, lst):
        """ Find all occurrences of maximum in a list."""
        m = np.amax(lst)
        inds = [i for i, x in enumerate(lst) if x == m]
        return inds

    def find_pi_star(self, mean_func, d_anal, d_anal_2):
        Y_1 = d_anal(self.m_lst, mean_func)
        Y_2 = d_anal_2(self.m_lst, mean_func)
        ind1 = np.where(Y_1==1)[0][0]
        ind2 = np.where(Y_2==1)[0][0]
        m_opt_lst = self.m_lst[ind1:ind2]
        return m_opt_lst

    def calc_policy_coverage(self, mean_func, sim_lst):
        policy_cov = list()
        d_anal_1 = lambda x, p: (p(1, x) - p(0, x) >= 0)
        d_anal_2 = lambda x, p: (p(1, x) - p(0, x) > 0)
        m_opt_lst = self.find_pi_star(mean_func, d_anal_1, d_anal_2)
        for sim in sim_lst:
            sets = sim.policy_sets
            ct = 0
            for pi_hat in sets:
                flag = (m_opt_lst[0] >= np.amin(pi_hat)) and (m_opt_lst[-1] <= np.amax(pi_hat))
                ct += flag
            policy_cov.append(ct / len(sets))
        return policy_cov
    
    def compute_oracle(self, alpha, n_iters):
        # TODO: move this inside run - save some time!
        m_lst = np.linspace(-1, 1, self.K)
        #alpha = 0.05
        lcb_oracle = np.zeros(n_iters)
        ucb_oracle = np.zeros(n_iters)
        for i in range(n_iters):
            X = np.random.uniform(-1, 1, self.n)
            A = np.random.binomial(1, 0.5+0.1*X)
            Y = self.p_Y(A, X)
            Z = self.p_Z(A, X)
            
            pi_star = self.find_pi_star(self.mean_func, self.d_anal, self.d_anal_2)
            dl = lambda x: x>=pi_star[0]
            dm = lambda x: x>=pi_star[-1]
            pa_x = (0.5+0.1*X) * (A==1) + (0.5-0.1*X) * (A==0)
            psi_oracle_1 = (A == dl(X))/pa_x * (Z-self.mean_func_Z(A, X))+self.mean_func_Z(dl(X), X)
            psi_oracle_2 = (A == dm(X))/pa_x * (Z-self.mean_func_Z(A, X))+self.mean_func_Z(dm(X), X)
            ll_orac_1 = np.mean(psi_oracle_1)-np.std(psi_oracle_1)*norm.ppf(1-alpha/2)/np.sqrt(self.n)
            ll_orac_2 = np.mean(psi_oracle_2)-np.std(psi_oracle_2)*norm.ppf(1-alpha/2)/np.sqrt(self.n)
            uu_orac_1 = np.mean(psi_oracle_1)+np.std(psi_oracle_1)*norm.ppf(1-alpha/2)/np.sqrt(self.n)
            uu_orac_2 = np.mean(psi_oracle_2)+np.std(psi_oracle_2)*norm.ppf(1-alpha/2)/np.sqrt(self.n)
            
            lcb_oracle[i] = min(ll_orac_1, ll_orac_2)
            ucb_oracle[i] = max(uu_orac_1, uu_orac_2)
        
        if self.scene_typ == "nonunique":
            low = min(self.EZ_l, self.EZ)
            high = max(self.EZ_l, self.EZ)
            self.oracle_coverage = np.mean((low >= lcb_oracle) * (high <= ucb_oracle))
        else:
            self.oracle_coverage = np.mean((self.EZ >= lcb_oracle) * (self.EZ <= ucb_oracle))
        self.oracle_width = np.mean(ucb_oracle - lcb_oracle)
        #print("os and oracle for alpha={} completed.".format(alpha))

    def compute_oracle_high_dim(self, alpha, n_iters):
        no_error = True
        dim = 3
        def p_A_true(X):
            # conditional distribution of A|X
            n = X.shape[0]
            p = np.zeros(n)
            for i in range(n):
                p[i] = 1/(1+np.exp(-X[i,0])+np.exp(-2*X[i,1])+np.exp(-X[i,2]))
            return p
        
        def phi_oracle(t, _grad, X, A, Y):
            ndim = len(t)
            s = [True] * ndim # some fixed s
            tr = tree(t, s)
            d = lambda m: tr.predict(m)
            pa_x = p_A_true(X) * (A == 1) + (1-p_A_true(X)) * (A == 0)
            phi = (A == d(X))/pa_x * (Y-self.mean_func(A, X))+self.mean_func(d(X), X)
            phi_mean = np.mean(phi)
            return phi_mean

        def find_max_phi_oracle(X, A, Y):
            ndim = X.shape[1]
            opt = nlopt.opt(nlopt.LN_BOBYQA, ndim)
            opt.set_lower_bounds(-1*np.ones(ndim))
            opt.set_upper_bounds(np.ones(ndim))
            new_f = partial(phi_oracle, X=X, A=A, Y=Y)
            opt.set_max_objective(new_f)
            opt.set_xtol_rel(1e-4)  # use a smaller tolerance
            # use a random initial guess
            x_init = np.random.uniform(low=-1, high=1, size=ndim)
            try:
                x = opt.optimize(x_init)
            except nlopt.RoundoffLimited:
                no_error = False
                x = x_init
            return x
        
        lcb_oracle = []
        ucb_oracle = []
        for i in range(n_iters):
            X = np.random.multivariate_normal(mean=np.zeros(dim), cov=np.eye(dim), size=self.n)
            A = self.p_A(X)
            Y = self.p_Y(A, X)
            Z = self.p_Z(A, X)
            
            pi_star = find_max_phi_oracle(X, A, Y)
            # print("pi star for oracle: ", pi_star)
            if no_error:
                ndim = len(pi_star)
                s = [True] * ndim # some fixed s
                tr = tree(pi_star, s)
                d_opt = lambda m: tr.predict(m)

                pa_x = p_A_true(X) * (A == 1) + (1-p_A_true(X)) * (A == 0)
                psi_oracle = (A == d_opt(X))/pa_x * (Z-self.mean_func_Z(A, X))+self.mean_func_Z(d_opt(X), X)
                lcb_oracle.append(np.mean(psi_oracle)-np.std(psi_oracle)*np.sqrt(stats.chi2.ppf(1-alpha,dim))/np.sqrt(self.n))
                ucb_oracle.append(np.mean(psi_oracle)+np.std(psi_oracle)*np.sqrt(stats.chi2.ppf(1-alpha,dim))/np.sqrt(self.n))
        
        lcb_oracle = np.array(lcb_oracle)
        ucb_oracle = np.array(ucb_oracle)
        self.oracle_coverage = np.mean((self.EZ >= lcb_oracle) * (self.EZ <= ucb_oracle))
        self.oracle_width = np.mean(ucb_oracle - lcb_oracle)