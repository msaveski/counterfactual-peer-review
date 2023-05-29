
import time
import utils
import numpy as np
import gurobipy as gp

from constants import *


class LipMonAnalysis():
    '''
    Class to handle saving, loading, and computing Lipschitz/Monotonicity imputations for a single off-policy + outcome.
    '''
    def __init__(self, rc, Y_var, Y, coeff_flat, violations, obs_idxs, y_min, y_max, verbose=False, lam=None):
        self.rc = rc
        self.Y_var = Y_var
        self.verbose = verbose

        self.violations = violations
        self.obs_idxs = obs_idxs
        self.N_obs = np.sum(self.obs_idxs)
        self.N_vio = np.sum(self.violations)

        self.y_min = y_min
        self.y_max = y_max
        self.Y = Y
        self.coeff_flat = coeff_flat
        self.covariates_loaded = False

        self.lam = lam if lam is not None else 1e9

    def load_covariates(self):
        if self.covariates_loaded:
            return
       
        # Y observed
        self.Y_obs = self.Y.flatten()[self.obs_idxs]
        
        # bound coefficient for all violations
        self.Fb_vio = self.coeff_flat[self.violations]

        if self.verbose:
            print('Loading covariates')
        # load covariates only if needed

        similarity_info = np.load(SIMILARITY_FILE[self.rc.stage])
        B = utils.get_bids(self.rc.stage, similarity_info).flatten()
        TEXT = utils.get_text_sim(self.rc.stage, similarity_info).flatten()
        K = utils.get_keyword_sim(self.rc.stage, similarity_info).flatten()
           
        X_all = np.column_stack((B, TEXT, K))
        # normalize covariates to [0, 1]
        utils.norm_zero_one(X_all)
        self.X_vio = X_all[self.violations, :]
        self.X_obs = X_all[self.obs_idxs, :]
    
        self.D_obs_obs = utils.cdist_lip(self.X_obs, self.X_obs)
        self.D_vio_obs = utils.cdist_lip(self.X_vio, self.X_obs)
        self.D_vio_vio = utils.cdist_lip(self.X_vio, self.X_vio)
        self.covariates_loaded = True


    def get_fname(self, L):
        policy_name = self.rc.get_fname()
        if L is None: # Monotone
            fname = ALT_FOLDER[self.rc.stage] + f'{policy_name}_MONO_Y{self.Y_var}_results.npz'
        else:
            fname = ALT_FOLDER[self.rc.stage] + f'{policy_name}_L{L}_Y{self.Y_var}_results.npz' # T values depend on policy, Y, L
        return fname

    def load_T(self, L):
        fname = self.get_fname(L)
        data = np.load(fname, allow_pickle=True) # allow loading of None matrices
        print(f'Loaded results for L={L}, fname={fname}')
        T_min = data['T_min']
        T_max = data['T_max']
        assert np.all(T_min == None) or T_min.size == self.N_obs + self.N_vio
        assert np.all(T_max == None) or T_max.size == self.N_obs + self.N_vio
        self.lam = data['info'].item().get('lam', self.lam)
        return T_min, T_max

    def compute_T(self, L, recompute=False):
        if not recompute:
            try:
                return self.load_T(L)
            except FileNotFoundError as e:
                pass

        self.load_covariates()
        fname = self.get_fname(L)
        print(f'Computing results for L={L}, fname={fname}')
        if L is None:
            T_min, T_max, info = self.solve_monotone_LP()
        else:
            T_min, T_max, info = self.solve_lipschitz_LP(L)
        print(f'Saving results to {fname}...') 
        np.savez(fname, T_min=T_min, T_max=T_max, info=info)
        return T_min, T_max

    def solve_lipschitz_LP(self, L):
        lam = self.lam

        stime = time.time()
        stime0 = stime
        N_vio = self.D_vio_vio.shape[0]
        N_obs = self.D_vio_obs.shape[1]
        assert N_vio == self.N_vio and N_obs == self.N_obs
        D = np.zeros((N_obs + N_vio, N_obs + N_vio))
        D[:N_obs, :N_obs] = self.D_obs_obs
        D[N_obs:, N_obs:] = self.D_vio_vio
        D[N_obs:, :N_obs] = self.D_vio_obs
        D[:N_obs, N_obs:] = self.D_vio_obs.T
        
        m = gp.Model()
        m.setParam('OutputFlag', 0)
        m.setParam('Method', 1)
        
        obj_T = np.zeros(N_obs+N_vio)
        obj_T[N_obs:] = self.Fb_vio
        T = m.addMVar(N_obs + N_vio, lb=self.y_min, ub=self.y_max, obj=obj_T)
        delta_obs = m.addMVar(N_obs, lb=0, ub=self.y_max-self.y_min, obj=lam)
        m.modelSense = gp.GRB.MINIMIZE
    
        if self.verbose:
            print('Created variables:', time.time() - stime) 
            print(f'N_obs = {N_obs}, N_vio = {N_vio}')
        stime = time.time()
    
        m.addConstr(T[:N_obs] - self.Y_obs <= delta_obs)
        m.addConstr(self.Y_obs - T[:N_obs] <= delta_obs)
        mask = np.zeros((N_obs + N_vio, N_obs + N_vio))
        I = np.eye(N_obs + N_vio)
        xtime = time.time()
        for i in range(N_obs + N_vio):
            mask[:, i] = 1
            m.addConstr((mask - I) @ T <= L * D[:, i])
            mask[:, i] = 0
            
        if self.verbose:
            print('Created constraints:', time.time() - stime)
        stime = time.time()
        
        m.optimize()
        
        if self.verbose:
            print('Solved minimization:', time.time() - stime)
        stime = time.time()
            
        info = {'lam' : lam}
        T_min = None
        if m.status == gp.GRB.OPTIMAL:
            dist = np.abs(delta_obs.x - np.abs(T.x[:N_obs] - self.Y_obs))
            if self.verbose:
                print('Max delta dist:', np.max(dist), '; Total dist:', np.sum(dist))
    
            T_min = T.x
            info['min_obj'] = m.objVal
            info['min_T_obj'] = np.sum(self.Fb_vio * T.x[N_obs:])
            info['min_delta_obs'] = delta_obs.x
        else:
            info['min_status'] = m.status
    
        if self.verbose:
            print('Output solution:', time.time() - stime)
        stime = time.time()
        
        T.obj = -obj_T
        m.optimize()
        
        if self.verbose:
            print('Solved maximization:', time.time() - stime)
        stime = time.time()
        
        T_max = None
        if m.status == gp.GRB.OPTIMAL:
            dist = np.abs(delta_obs.x - np.abs(T.x[:N_obs] - self.Y_obs))
            if self.verbose:
                print('Max delta dist:', np.max(dist), '; Total dist:', np.sum(dist))
    
            T_max = T.x
            info['max_obj'] = m.objVal
            info['max_T_obj'] = np.sum(self.Fb_vio * T.x[N_obs:])
            info['max_delta_obs'] = delta_obs.x
        else:
            info['max_status'] = m.status
     
        if self.verbose:
            print('Output solution:', time.time() - stime)
            print('Total time:', time.time() - stime0)
        return T_min, T_max, info

    def solve_monotone_LP(self):
        lam = self.lam

        stime = time.time()
        stime0 = stime
        N_vio = self.X_vio.shape[0]
        N_obs = self.X_obs.shape[0]
        assert N_vio == self.N_vio and N_obs == self.N_obs

        # Compute dominance
        def cdom(X1, X2):
            weak = np.all(X1[:, np.newaxis, :] >= X2[np.newaxis, :, :], axis=-1)
            assert weak.shape == (X1.shape[0], X2.shape[0])
            strong = np.any(X1[:, np.newaxis, :] > X2[np.newaxis, :, :], axis=-1)
            assert strong.shape == (X1.shape[0], X2.shape[0])
            dom = (weak & strong)
            return dom # dom_ij == 1 <-> i dom j
        X_obs_vio = np.concatenate((self.X_obs, self.X_vio))
        dom = cdom(X_obs_vio, X_obs_vio)
    
        m = gp.Model()
        m.setParam('OutputFlag', 0)
        m.setParam('Method', 1)
    
        obj_T = np.zeros(N_obs+N_vio)
        obj_T[N_obs:] = self.Fb_vio
        T = m.addMVar(N_obs + N_vio, lb=self.y_min, ub=self.y_max, obj=obj_T)
        delta_obs = m.addMVar(N_obs, lb=0, ub=self.y_max-self.y_min, obj=lam)
        m.modelSense = gp.GRB.MINIMIZE
    
        if self.verbose:
            print('Created variables:', time.time() - stime)
        stime = time.time()
    
        m.addConstr(T[:N_obs] - self.Y_obs <= delta_obs)
        m.addConstr(self.Y_obs - T[:N_obs] <= delta_obs)
        mask = np.zeros((N_obs + N_vio, N_obs + N_vio))
        I = np.eye(N_obs + N_vio)
        LB = np.full(dom.shape, self.y_min - self.y_max) # no bound if not dom, else 0
        LB[dom] = 0
        for i in range(N_obs + N_vio):
            mask[:, i] = 1
            # dom[i, :] is "does i dominate :"
            m.addConstr(((mask - I) @ T) >= LB[i, :]) # T_i - T_j for all j * dom_ij
            mask[:, i] = 0
    
        if self.verbose:
            print('Created constraints:', time.time() - stime)
        stime = time.time()
    
        m.optimize()
    
        if self.verbose:
            print('Solved minimization:', time.time() - stime)
        stime = time.time()
    
        info = {'lam' : lam}
        T_min = None
        if m.status == gp.GRB.OPTIMAL:
            dist = np.abs(delta_obs.x - np.abs(T.x[:N_obs] - self.Y_obs))
            if self.verbose:
                print('Max delta dist:', np.max(dist), '; Total dist:', np.sum(dist))
    
            T_min = T.x
            info['min_obj'] = m.objVal
            info['min_T_obj'] = np.sum(self.Fb_vio * T.x[N_obs:])
            info['min_delta_obs'] = delta_obs.x
        else:
            info['min_status'] = m.status
 
        if self.verbose:
            print('Output solution:', time.time() - stime)
        stime = time.time()
    
        T.obj = -obj_T
        m.optimize()
    
        if self.verbose:
            print('Solved maximization:', time.time() - stime)
        stime = time.time()
    
        T_max = None
        if m.status == gp.GRB.OPTIMAL:
            dist = np.abs(delta_obs.x - np.abs(T.x[:N_obs] - self.Y_obs))
            if self.verbose:
                print('Max delta dist:', np.max(dist), '; Total dist:', np.sum(dist))
    
            T_max = T.x
            info['max_obj'] = m.objVal
            info['max_T_obj'] = np.sum(self.Fb_vio * T.x[N_obs:])
            info['max_delta_obs'] = delta_obs.x
        else:
            info['max_status'] = m.status
     
        if self.verbose:
            print('Output solution:', time.time() - stime)
            print('Total time:', time.time() - stime0)
        return T_min, T_max, info
