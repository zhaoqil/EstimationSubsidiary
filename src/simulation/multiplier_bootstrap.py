import numpy as np
from policy import ruleGenerator, tree
import nlopt
from functools import partial
import time
import ray
from scipy.optimize import brentq
import pdb

BEST_PROB = 0

def rradem(n):
    return (np.random.binomial(1, 0.5, n)-0.5)*2

def rradem_sd(n, sd):
    np.random.seed(sd)
    return (np.random.binomial(1, 0.5, n)-0.5)*2

def f(t, _grad, iF_Y, rdm, X, A, Y, p1_x, py_1x, py_0x):
    ndim = len(t)
    s = [True] * ndim # some fixed s
    tr = tree(t, s)
    d = lambda m: tr.predict(m)
    phi = iF_Y.compute_phi_hat(d, X, A, Y, p1_x, py_1x, py_0x)
    phi_center = (phi - np.mean(phi)) / np.std(phi)
    res = np.sqrt(len(rdm)) * np.mean(phi_center * rdm)
    return res

@ray.remote
def find_max(X, A, Y, iF_Y, p1_x, py_1x, py_0x, B, r):
    ndim = X.shape[1]
    np.random.seed(r) # for reproducibility
    rdm = rradem_sd(X.shape[0], r)
    opt = nlopt.opt(nlopt.LN_BOBYQA, ndim)
    opt.set_lower_bounds(-1*np.ones(ndim))
    opt.set_upper_bounds(np.ones(ndim))
    new_f = partial(f, iF_Y=iF_Y, rdm=rdm, X=X, A=A, Y=Y, p1_x=p1_x, py_1x=py_1x, py_0x=py_0x)
    opt.set_max_objective(new_f)
    opt.set_ftol_rel(5e-3)  # use a smaller tolerance
    # use a random initial guess
    x_init = np.random.uniform(low=-1, high=1, size=ndim)
 
    try:
        x = opt.optimize(x_init)
        t_val = opt.last_optimum_value()
    except nlopt.RoundoffLimited:
        # print("RoundoffLimited error occurred!") # this happens majority of times!
        t_val = None
        r = None
    return t_val, r

def f_joint(x, seed_lst, X, A, Z, iF_Z, p1_x, pz_1x, pz_0x, B):
    # computes the value of \tilde{f}_\pi for a some policies using multiplier bootstrap
    n, ndim = X.shape
    s = [True] * ndim # some fixed s
    tr = tree(x, s)
    d = lambda m: tr.predict(m) 
    D_Z = iF_Z.compute_phi_hat(d, X, A, Z, p1_x, pz_1x, pz_0x)
    DZ_center = (D_Z - np.mean(D_Z)) / np.std(D_Z)
    lst_Z = np.zeros(len(seed_lst))
    for j, sd in enumerate(seed_lst):
        rademacher_rv = rradem_sd(n, sd)
        #pdb.set_trace()
        lst_Z[j] = np.sqrt(n) * np.mean(rademacher_rv * DZ_center)
    return lst_Z  # corresponding Z's

def find_prob(x, _grad, t, u, t_lst, seed_lst, X, A, Z, iF_Z, p1_x, pz_1x, pz_0x, B):
    # returns Pr(sup f_\pi<t, \tilde{f}_\pi<u) for some pi, t and u
    global BEST_PROB
    psi_pi_lst = f_joint(x, seed_lst, X, A, Z, iF_Z, p1_x, pz_1x, pz_0x, B)
    #print("psi_pi_lst shape: ", psi_pi_lst.shape)
    #print("t lst shape: ", t_lst.shape)
    prob = np.mean((t_lst <= t) * (psi_pi_lst <= u))
    if prob < BEST_PROB:
        BEST_PROB = prob
    return prob

def g_u(u, t, t_lst, seed_lst, X, A, Z, iF_Z, p1_x, pz_1x, pz_0x, B, alpha=0.05):
    # returns inf_{\pi\in\Pi} Pr(sup f_\pi<t, \tilde{f}_\pi<u)-alpha for some t and u
    ndim = X.shape[1]
    opt = nlopt.opt(nlopt.LN_BOBYQA, ndim)
    opt.set_lower_bounds(-1*np.ones(ndim))
    opt.set_upper_bounds(np.ones(ndim))
    new_f = partial(find_prob,t=t,u=u,t_lst=t_lst,seed_lst=seed_lst,X=X,A=A,Z=Z,iF_Z=iF_Z,p1_x=p1_x,pz_1x=pz_1x,pz_0x=pz_0x,B=B)
    opt.set_min_objective(new_f)
    opt.set_ftol_rel(5e-3)  # use a smaller tolerance
    # use a random initial guess
    x_init = np.random.uniform(low=-1, high=1, size=ndim)
    try:
        x = opt.optimize(x_init)
        opt_prob = opt.last_optimum_value()
    except nlopt.RoundoffLimited:
        opt_prob = BEST_PROB
    return opt_prob - (1-alpha)

# @ray.remote
def find_opt_joint(t, u_min, u_max, t_lst, seed_lst, X, A, Z, iF_Z, p1_x, pz_1x, pz_0x, B):
    new_ff = partial(g_u,t=t,t_lst=t_lst,seed_lst=seed_lst,X=X,A=A,Z=Z,iF_Z=iF_Z,p1_x=p1_x,pz_1x=pz_1x,pz_0x=pz_0x,B=B)
    if new_ff(u_max) < 0:
        u_opt = u_max
    else:
        u_opt = brentq(new_ff, u_min, u_max)
    return u_opt, t
    
def mult_boot_joint_high_dim(num, X, A, Y, Z, iF_Y, iF_Z, p1_x, py_1x, pz_1x, py_0x, pz_0x, B, alpha=0.05):
    t_lst, seed_lst = mult_boot_one_high_dim(X, A, Y, iF_Y, p1_x, py_1x, py_0x, B=B)
    t_min = np.quantile(t_lst, 1-alpha)
    t_max = np.amax(t_lst)
    u_min, u_max = 0, 5 # some arbitrary u range
    t_range = np.linspace(t_min, t_max, num=num)
    # uncomment below if ray hasn't initialized yet
#     ray.shutdown()
#     ray.init()
#     start_time = time.time()
#     all_results = ray.get(
#         [
#             find_opt_joint.remote(t, u_min, u_max, t_lst, seed_lst, X, A, Z, iF_Z, p1_x, pz_1x, pz_0x, B) for t in t_range
#         ]
#     ).copy()
#     ray.shutdown()
#     u_vals = np.array([r[0] for r in all_results])
#     t_vals = np.array([r[1] for r in all_results])
    
    u_vals = np.zeros(num)
    for i, t in enumerate(t_range):
        new_ff = partial(g_u,t=t,t_lst=t_lst,seed_lst=seed_lst,X=X,A=A,Z=Z,iF_Z=iF_Z,p1_x=p1_x,pz_1x=pz_1x,pz_0x=pz_0x,B=B)
        if new_ff(u_max) < 0:
            u_opt = u_max
        else:
            u_opt = brentq(new_ff, u_min, u_max)
        #print("u_opt: ", u_opt)
        u_vals[i] = u_opt
        t_vals = t_range.copy()
    return t_vals, u_vals

def mult_boot_one_high_dim(X, A, Y, iF_Y, p1_x, py_1x, py_0x, B=1000):
    ndim = X.shape[1]
#     t_lst = np.zeros(B)
#     ray.shutdown()
#     ray.init()
    start_time = time.time()
    all_results = ray.get(
        [
            find_max.remote(X, A, Y, iF_Y, p1_x, py_1x, py_0x, B, r) for r in range(B)
        ]
    ).copy()
#     ray.shutdown()
    duration = time.time() - start_time
    print("ray running time: ", duration)
    
    t_lst = np.array([r[0] for r in all_results])
    seed_lst = np.array([r[1] for r in all_results])
    t_lst = np.array([r for r in t_lst if r is not None])
    seed_lst = np.array([r for r in seed_lst if r is not None])
#     for j in range(B):
#         rdm = rradem(X.shape[0])
#         opt = nlopt.opt(nlopt.LN_COBYLA, ndim)
#         opt.set_lower_bounds(-1*np.ones(ndim))
#         opt.set_upper_bounds(np.ones(ndim))
#         new_f = partial(f, iF_Y=iF_Y, rdm=rdm, X=X, A=A, Y=Y, p1_x=p1_x, py_1x=py_1x, py_0x=py_0x)
#         opt.set_max_objective(new_f)
#         opt.set_xtol_rel(1e-4)  # use a smaller tolerance
#         start = time.time()
#         # use a random initial guess
#         x_init = np.random.uniform(low=-1, high=1, size=ndim)
#         x = opt.optimize(x_init)
#         print("optimization complete. Runtime: ", time.time() - start)
#         t_lst[j] = opt.last_optimum_value()
#         print("iterations {} completed.".format(j))
    return t_lst, seed_lst
    
def mult_boot_one(m_lst, X, A, Y, iF_Y, p1_x, py_1x, py_0x, B=1000):
    # estimate quantile using multiplier bootstrap
    # here p_Y is mean_func!
    n = X.shape[0]
    K = m_lst.shape[0]
    gen = ruleGenerator("Indicator", m_lst)
    pa_x = (0.5+0.1*X) * (A==1) + (0.5-0.1*X) * (A==0) # some fixed pa_x
    DY_center = np.zeros([n, K])
    for i in range(K):
        d = gen.generate(i)
        D_Y = iF_Y.compute_phi_hat(d, X, A, Y, p1_x, py_1x, py_0x)
        DY_center[:,i] = (D_Y - np.mean(D_Y)) / np.std(D_Y) # D_pi/sigma_pi
    
    t_lst = np.zeros(B)
    for j in range(B):
        rademacher_rv = rradem(n)
        lst_Y = np.sqrt(n) * np.mean(rademacher_rv[:,None] * DY_center, axis=0) # size K
        t_lst[j] = np.amax(lst_Y)
    return t_lst

def mult_boot_joint(m_lst, X, A, Y, iF_Y, Z, iF_Z, p1_x, py_1x, py_0x, pz_1x, pz_0x, B=1000):
    # estimate quantile using multiplier bootstrap
    n = X.shape[0]
    K = m_lst.shape[0]
    gen = ruleGenerator("Indicator", m_lst)
    sup_t_lst = np.zeros(B)
    DY_center = np.zeros([n, K])
    DZ_center = np.zeros([n, K])
    for i in range(K):
        d = gen.generate(i)
        D_Y = iF_Y.compute_phi_hat(d, X, A, Y, p1_x, py_1x, py_0x)
        DY_center[:,i] = (D_Y - np.mean(D_Y)) / np.std(D_Y) # D_pi/sigma_pi
        D_Z = iF_Z.compute_phi_hat(d, X, A, Z, p1_x, pz_1x, pz_0x)
        DZ_center[:,i] = (D_Z - np.mean(D_Z)) / np.std(D_Z) # centered variable for a fixed policy
    
    t_lst = np.zeros(B)
    u_lst = np.zeros([B, K])
    for j in range(B):
#         rademacher_rv = rradem([n, K])
#         lst_Y = np.sqrt(n) * np.mean(rademacher_rv * DY_center, axis=0) # size K
#         lst_Z = np.sqrt(n) * np.mean(rademacher_rv * DZ_center, axis=0)

        rademacher_rv = rradem(n)
        lst_Y = np.sqrt(n) * np.mean(rademacher_rv[:,None] * DY_center, axis=0) # size K
        lst_Z = np.sqrt(n) * np.mean(rademacher_rv[:,None] * DZ_center, axis=0)
        
        t_lst[j] = np.amax(lst_Y)
        u_lst[j] = lst_Z
    return t_lst, u_lst
    
def find_thresholds(T, U=None, target=0.95):
    if U is None:
        # in the case of union bounding method
        t_union = np.percentile(T, target*100) # convert it to percentage
        t_nonzero = -1
        u_nonzero = -1
    else:
        # in the case of joint method
        B, K = U.shape
        if len(T) != B:
            raise ValueError("T and U must have the same number of rows.")

        # For each column in U, pair it with T and sort the pairs
        pair_sorted = [sorted(zip(T, U[:, k])) for k in range(K)]

        # Initialize thresholds
        t = np.zeros(K)
        u = np.zeros(K)

        # Iterate to find the desired pair (t, u) for each column in U
        for k in range(K):
            for i, (t_val, u_val) in enumerate(pair_sorted[k]):
                # Vectorized computation for mean values
                mean_vals = np.mean((T[:, None] <= t_val) * (U <= u_val), axis=0)
                min_mean = np.min(mean_vals)
                if min_mean >= target:
                    t[k] = t_val
                    u[k] = u_val
                    break

        # Create a boolean mask for non-zero entries
        mask = t != 0
        t_nonzero = t[mask]
        u_nonzero = u[mask]
        t_union = -1

#         # Find the index of the pair with the minimum sum
#         min_sum_idx = np.argmin(t_nonzero + u_nonzero)
#         t_min_sum = t_nonzero[min_sum_idx]
#         u_min_sum = u_nonzero[min_sum_idx]

        # Find the index of the pair with the minimum u
#         min_sum_idx = np.argmin(u_nonzero)
#         t_min_sum = t_nonzero[min_sum_idx]
#         u_min_sum = u_nonzero[min_sum_idx]

    return t_nonzero, u_nonzero, t_union

# def calc_t_u(t_lst, u_lst, t, u):
#     probs = np.mean((t_lst <= t)[:,None] * (u_lst <= u), axis=0)
#     prob_min = np.amin(probs)
#     return prob_min

# def calc_dic_t_u(alpha_vals, probm, t_range, u_range):
#     dic_t = {}
#     dic_u = {}
#     for alpha in alpha_vals:
#         #arr1 = np.where(probm == 1-alpha)
#         arr1 = np.where(np.abs(probm - (1-alpha))<0.001)
#         arr1_sum = np.sum(arr1, axis=0)
#         opt_ind = np.argmin(arr1_sum)
#         t_ind = arr1[0][opt_ind]
#         u_ind = arr1[1][opt_ind]
#         t = t_range[t_ind]
#         u = u_range[u_ind]
#         dic_t[alpha] = t
#         dic_u[alpha] = u
#     return dic_t, dic_u