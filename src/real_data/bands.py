import numpy as np
from influence_function import influenceFunction
from policy import ruleGenerator, tree
import nlopt

class uniformBands():

    def __init__(self, x, a, y, z, p1_x, py_1x, py_0x, pz_1x, pz_0x, t, u, iF_Y=None, iF_Z=None):
        """
        d: function that said the optimal rule. - function
        n: sample size
        B: bootstrap sample size
        dim: dimension of policy class
        """
        self.X = x
        self.A = a
        self.Y = y
        self.Z = z
        self.p1_x = p1_x
        self.py_1x = py_1x
        self.py_0x = py_0x
        self.pz_1x = pz_1x
        self.pz_0x = pz_0x
        self.t = t
        self.u = u
        self.iF_Y = iF_Y
        self.iF_Z = iF_Z
        self.no_error = True
        
    def compute_lcb_ucb_f(self, t, _grad):
        d = lambda m: m>=t
        phi = self.iF_Y.compute_phi_hat(d, self.X, self.A, self.Y, self.p1_x, self.py_1x, self.py_0x)
            
        phi_hat = np.mean(phi)
        sigma_hat = np.std(phi)
        lcb = phi_hat - sigma_hat * self.t / np.sqrt(self.A.shape[0])
        ucb = phi_hat + sigma_hat * self.t / np.sqrt(self.A.shape[0])
        return lcb, ucb
    
    def lcb_f(self, t, _grad):
        lcb, ucb = self.compute_lcb_ucb_f(t, _grad)
        return lcb
    
    def ucb_f(self, t, _grad):
        lcb, ucb = self.compute_lcb_ucb_f(t, _grad)
        return ucb
    
    def find_max_lcb_f(self):
        opt = nlopt.opt(nlopt.LN_COBYLA, 1)
        opt.set_lower_bounds(np.amin(self.X))
        opt.set_upper_bounds(np.amax(self.X))
        opt.set_max_objective(self.lcb_f)
        opt.set_xtol_rel(5e-3)
        x_init = 0.1
        try:
            x = opt.optimize([x_init])
            mmax = opt.last_optimum_value()
        except nlopt.RoundoffLimited:
            print("RoundoffLimited in lcb_f!")
            mmax = None
            self.no_error = False
        return mmax
    
    def compute_lcb_ucb_f_tilde(self, t, _grad):
        d = lambda m: m>=t
        psi = self.iF_Z.compute_phi_hat(d, self.X, self.A, self.Z, self.p1_x, self.pz_1x, self.pz_0x)
        psi_hat = np.mean(psi)
        sigma_hat = np.std(psi)
        lcb = psi_hat - sigma_hat * self.u / np.sqrt(self.A.shape[0])
        ucb = psi_hat + sigma_hat * self.u / np.sqrt(self.A.shape[0])
        return lcb, ucb
    
    def lcb_f_tilde(self, t, _grad):
        lcb, ucb = self.compute_lcb_ucb_f_tilde(t, _grad)
        return lcb
    
    def ucb_f_tilde(self, t, _grad):
        lcb, ucb = self.compute_lcb_ucb_f_tilde(t, _grad)
        return ucb
            
    def find_ub_z_unif(self, max_lcb_f):
        opt = nlopt.opt(nlopt.LN_COBYLA, 1)
        opt.set_lower_bounds(np.amin(self.X))
        opt.set_upper_bounds(np.amax(self.X))
        opt.set_max_objective(self.ucb_f_tilde)
        hh = lambda x, _grad: max_lcb_f - self.ucb_f(x, _grad) # first-stage requirement
        opt.add_inequality_constraint(hh)
        opt.set_xtol_rel(5e-3)
        x_init = 0.1
        try:
            x = opt.optimize([x_init])
            mmax = opt.last_optimum_value()
        except nlopt.RoundoffLimited:
            print("RoundoffLimited in ub_z_unif!")
            mmax = None
            self.no_error = False
        return mmax
    
    def find_lb_z_unif(self, max_lcb_f):
        opt = nlopt.opt(nlopt.LN_COBYLA, 1)
        opt.set_lower_bounds(np.amin(self.X))
        opt.set_upper_bounds(np.amax(self.X))
        opt.set_min_objective(self.lcb_f_tilde)
        hh = lambda x, _grad: max_lcb_f - self.ucb_f(x, _grad) # first-stage requirement
        opt.add_inequality_constraint(hh)
        opt.set_xtol_rel(5e-3)
        x_init = 0.1
        try:
            x = opt.optimize([x_init])
            mmax = opt.last_optimum_value()
        except nlopt.RoundoffLimited:
            print("RoundoffLimited in lb_z_unif!")
            mmax = None
            self.no_error = False
        return mmax
    
    def compute_uniform_conf_bands(self):
        max_lcb_f = self.find_max_lcb_f()
        if self.no_error:
            lb_z = self.find_lb_z_unif(max_lcb_f)
        if self.no_error:
            ub_z = self.find_ub_z_unif(max_lcb_f)
        if not self.no_error:
            lb_z = None
            ub_z = None
        return lb_z, ub_z