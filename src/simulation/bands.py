import numpy as np
from influence_function import influenceFunction
from policy import ruleGenerator
import pdb
        
class uniformBands():

    def __init__(self, x, a, y, z, n, typ, p1_x, py_1x, py_0x, pz_1x, pz_0x, t, q, pa_x=None, iF_Y=None, iF_Z=None):
        """
        x: context. 
        a: action. 
        y: short-term outcome. 
        z: long-term outcome. 
        n: sample size
        typ: type of policy class. 
        t, q: quantiles of supremum Gaussian process. 
        """
        self.x = x
        self.a = a
        self.y = y
        self.z = z
        
        self.n = n
        self.typ = typ
        self.t = t
        self.q = q
        self.iF_Y = iF_Y
        self.iF_Z = iF_Z
        self.p1_x = p1_x
        self.py_1x = py_1x
        self.py_0x = py_0x
        self.pz_1x = pz_1x
        self.pz_0x = pz_0x
        # self.gen_yz = dataGenerator(1, n, p_Y, p_Z)

#         if iF_Z is None:
#             self.iF_Y = influenceFunction(x, a, y=y)
#             self.iF_Z = influenceFunction(x, a, y=z)
#             self.iF_Y.estimate_pa_x()
#             self.iF_Z.estimate_pa_x()
#             self.iF_Y.estimate_p()
#             self.iF_Z.estimate_p()
#         else:
#             self.iF_Y = iF_Y
#             self.iF_Z = iF_Z

    def gen_lcb_ucb(self, lst, X_s, A_s, Y_s, iFun, t, typ, p1_x, py_1x, py_0x):
        """
        lst: list of rules. 
        """
        K = lst.shape[0]
        phi_hat = np.zeros(K)
        sigma_hat = np.zeros(K)
        gen = ruleGenerator(typ, lst)
        for i in range(K):
            d = gen.generate(i)
            phi = iFun.compute_phi_hat(d, X_s, A_s, Y_s, p1_x, py_1x, py_0x)
            # phi = (A_s == d(X_s))/pa_x * (Y_s-p_Y(A_s, X_s))+p_Y(d(X_s), X_s)
            phi_hat[i] = np.mean(phi)
            sigma_hat[i] = np.std(phi)

        lcb = phi_hat - sigma_hat * t / np.sqrt(self.n)
        ucb = phi_hat + sigma_hat * t / np.sqrt(self.n)
        return lcb, ucb, phi_hat, sigma_hat

    def generate_m_lst(self, typ, K):
        if typ == "Indicator":
            m_lst = np.linspace(-1, 1, K)
        elif typ == "Tree":
            m_lst = np.linspace(-1, 1, K)
            xx,yy = np.meshgrid(m_lst, m_lst)
            m_lst = np.array((xx.ravel(), yy.ravel())).T
        return m_lst
    
    def compute_uniform_conf_bands(self, K):
        # generate 1-dimensional policy class
        m_lst = self.generate_m_lst(self.typ, K)
        
        # first-stage pi_hat
        K = m_lst.shape[0]
        # X_s, A_s, Y_s, Z_s, pa_x = self.gen_yz.generate()
        
        #pdb.set_trace()
        lcb, ucb, phi_hat, sigma_hat = self.gen_lcb_ucb(m_lst, self.x, self.a, self.y, self.iF_Y, self.t, self.typ, self.p1_x, self.py_1x, self.py_0x)
        self.phi_hat = phi_hat
        self.first_stage_lcb = lcb
        self.first_stage_ucb = ucb
        lcb_max = np.amax(lcb)
        pi_hat = m_lst[ucb > lcb_max]
        self.pi_hat = pi_hat
        
        if (len(pi_hat) == 0):
            print("no pi_hat!")
            return 0, 0
        
        # second-stage lcb/ucb
        lcb_2, ucb_2, psi_hat, sigma_hat_z = self.gen_lcb_ucb(pi_hat, self.x, self.a, self.z, self.iF_Z, self.q, self.typ, self.p1_x, self.pz_1x, self.pz_0x)
        self.second_stage_lcb = lcb_2
        self.second_stage_ucb = ucb_2
        lcb_max_2 = np.amax(lcb_2)
        lb = np.amin(lcb_2[ucb_2 > lcb_max_2])
        ub = np.amax(ucb_2[ucb_2 > lcb_max_2])
        return lb, ub


