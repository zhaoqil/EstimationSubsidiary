import numpy as np
from bands import uniformBands
import multiplier_bootstrap as mb
from scipy.stats import norm
import scipy.stats as stats
from influence_function import influenceFunction
import pdb
import multiprocessing as mp
import os
import time
from multiplier_bootstrap import rradem
import ray
import nlopt
from functools import partial
from policy import ruleGenerator, tree
import pdb

def compute_phi_f(t, _grad, iF_Y, X, A, Y, p1_x, py_1x, py_0x):
    d = lambda m: m>=t
    phi = iF_Y.compute_phi_hat(d, X, A, Y, p1_x, py_1x, py_0x)
    phi_hat = np.mean(phi)
    return phi_hat
    
def compute_aipw_CI(alpha, X, A, Y, iF_Y, Z, iF_Z, p1_x, py_1x, py_0x, pz_1x, pz_0x):
    n = X.shape[0]
    opt = nlopt.opt(nlopt.LN_COBYLA, 1) # dim=1
    opt.set_lower_bounds(np.amin(X))
    opt.set_upper_bounds(np.amax(X))
    new_f = partial(compute_phi_f, iF_Y=iF_Y, X=X, A=A, Y=Y, p1_x=p1_x, py_1x=py_1x, py_0x=py_0x)
    # new_f = partial(compute_phi_f, iF_Y, X, A, Y, p1_x, py_1x, py_0x)
    opt.set_max_objective(new_f)
    opt.set_ftol_rel(5e-3)  # use a smaller tolerance
    # use a random initial guess
    x_init = 0.1
 
    try:
        m_opt = opt.optimize([x_init])
        t_val = opt.last_optimum_value()
    except nlopt.RoundoffLimited:
        print("RoundoffLimited error occurred in aipw!") # this happens majority of times!

    d_opt = lambda x: x>=m_opt
    psi_os = iF_Z.compute_phi_hat(d_opt, X, A, Z, p1_x, pz_1x, pz_0x)
    lcb_os = np.mean(psi_os)-np.std(psi_os)*norm.ppf(1-alpha/2)/np.sqrt(n)
    ucb_os = np.mean(psi_os)+np.std(psi_os)*norm.ppf(1-alpha/2)/np.sqrt(n)
    return lcb_os, ucb_os

@ray.remote
def worker(alpha, X, A, Y, Z, method_typ, B, r):
    np.random.seed(r)
    start = time.time() 
    iF_Y = influenceFunction(X, A, Y)
    iF_Z = influenceFunction(X, A, Z)

    p1_x = iF_Y.estimate_pa_x()
    py_1x, py_0x = iF_Y.estimate_p() # no significant speedup since xgboost already uses multiple cores
    pz_1x, pz_0x = iF_Z.estimate_p()
    
    time_1 = time.time()
    if method_typ == "naive_unique":
        beta = alpha * 0.2
        t_lst, seed_lst = mb.mult_boot_one_high_dim(X, A, Y, iF_Y, p1_x, py_1x, py_0x, B=B)
        _, _, t = mb.find_thresholds(t_lst, target=1-beta/2)
        bands = uniformBands(X, A, Y, Z, p1_x, py_1x, py_0x, pz_1x, pz_0x, t=t, u=norm.ppf(1-(alpha-beta)/2), iF_Y=iF_Y, iF_Z=iF_Z)
        lcb, ucb = bands.compute_uniform_conf_bands()
        
    elif method_typ == "os":
        lcb, ucb = compute_aipw_CI(alpha, X, A, Y, iF_Y, Z, iF_Z, p1_x, py_1x, py_0x, pz_1x, pz_0x)
        
    else: 
        num = 50 # grid size for (t, u)
        t_nonzero, u_nonzero = mb.mult_boot_joint_high_dim(num, X, A, Y, Z, iF_Y, iF_Z, p1_x, py_1x, pz_1x, py_0x, pz_0x, B=B)
        min_sum_idx = np.argmin(t_nonzero + u_nonzero)
        # uncomment below if still undercoverage
        min_u_idx = np.argmin(u_nonzero)
        t1 = t_nonzero[min_sum_idx]
        u1 = u_nonzero[min_sum_idx]
        t2 = t_nonzero[min_u_idx]
        u2 = u_nonzero[min_u_idx]
        
        bands1 = uniformBands(X, A, Y, Z, p1_x, py_1x, py_0x, pz_1x, pz_0x, t=t1, u=u1, iF_Y=iF_Y, iF_Z=iF_Z)
        lcb1, ucb1 = bands1.compute_uniform_conf_bands()
        
        bands2 = uniformBands(X, A, Y, Z, p1_x, py_1x, py_0x, pz_1x, pz_0x, t=t2, u=u2, iF_Y=iF_Y, iF_Z=iF_Z)
        lcb2, ucb2 = bands2.compute_uniform_conf_bands()
        
        if ucb1 - lcb1 <= ucb2 - lcb2:
            bands = bands1
            lcb = lcb1
            ucb = ucb1
        else:
            bands = bands2
            lcb = lcb2
            ucb = ucb2

            
    print("running one iteration for task {}: {}".format(r, time.time() - time_1))
    return lcb, ucb

class simulation():

    def __init__(self, X, A, Y, Z, method_typ, n_iters):
        """
        X: context. 
        A: action. 
        Y: short-term outcome. 
        Z: long-term outcome. 
        n: sample size
        n_iters: number of iterations - currently set to 1. 
        typ: type of policy class. 
        t, q: quantiles of supremum Gaussian process. 
        K: size of policy class. 
        alpha: confidence level. 
        beta: first-stage confidence level. 
        """
        self.n_iters = n_iters
        self.B = 1000 # not change it for now
        self.X = X
        self.A = A
        self.Y = Y
        self.Z = Z
        self.method_typ = method_typ
        self.avg_CI_width = list()
        self.CI_coverage = list()
        self.band_coverage = list()
        self.conf_bands = list()
        
    def run(self, alpha):
        self.lcb_z = np.zeros(self.n_iters)
        self.ucb_z = np.zeros(self.n_iters)

        ray.shutdown()
        ray.init()
        
        start_time = time.time()
        all_results = ray.get(
            [
                worker.remote(alpha, self.X, self.A, self.Y, self.Z, self.method_typ, self.B, r) for r in range(self.n_iters)
            ]
        ).copy()
        ray.shutdown()
        duration = time.time() - start_time

        self.runtime = time.time() - start_time

        self.lcb_z = np.array([r[0] for r in all_results])
        self.ucb_z = np.array([r[1] for r in all_results])

        print(f"\rRunning method {self.method_typ}.", end="")