
import numpy as np
import pandas as pd
from scipy.stats import norm
from scipy.optimize import minimize
from joblib import Parallel, delayed

import utils 
from constants import *
from core import RunConfig
from lip_mon_analysis import LipMonAnalysis


'''
Main script to run the analysis of an alternate policy.
'''

class DataLoader():
    '''
    Holds common data to prevent re-loading.
    '''
    def __init__(self, stage, Y_var, load_cov=True):
        self.stage = stage
        self.Y_var = Y_var

        onpolicy_data = np.load(ON_POLICY_FILE[stage])
        self.X = onpolicy_data['A']
        self.Fa = onpolicy_data['F']

        self.Y, self.y_min, self.y_max = utils.get_y_min_max(stage, Y_var)

        self.Cov = None
        if load_cov:
            self.Cov = np.load(COVAR_FILE[stage])['Cov']

        self.Y_preds = {}
        for model_name in model_names:
            Y_pred = np.load(f'{MODEL_DIR}/preds/{Y_var}_s{stage}_{model_name}.npz')['Y_preds']
            self.Y_preds[model_name] = Y_pred


def analyses_HT(rc, Y_var, lipschitz_Ls=[], verbose=False, data_loader=None, compute_lip_only=False):
    if verbose:
        print(f'Analyzing: Y={Y_var}, params={rc.params}')

    Fb = rc.load()['F']

    if data_loader is None:
        assert rc.stage != -1
        data_loader = DataLoader(rc.stage, Y_var, load_cov=not compute_lip_only)
    else:
        assert rc.stage == data_loader.stage
        assert Y_var == data_loader.Y_var

    X = data_loader.X
    Fa = data_loader.Fa
    Y = data_loader.Y.copy() # gets modified
    y_min = data_loader.y_min
    y_max = data_loader.y_max
    Cov = data_loader.Cov

    X_att = (np.isnan(Y) & (X == 1)).astype(int)
    X_new = (~np.isnan(Y) & (X == 0)).astype(int)
    Obs = (X == 1) & (X_att != 1)
    absent_reviewers = (Obs.sum(axis=0) == 0)
    Y[X_new == 1] = np.nan

    assert np.all(Fa[Fa < 1e-13] == 0)
    assert np.all(Fb[Fb < 1e-13] == 0)

    # sanity checks
    assert np.all(Obs == ~np.isnan(Y))
    # all assignments, except for attrition, have an outcome
    assert np.sum(np.isnan(Y[(X == 1) & (X_att != 1)])) == 0
    # all attrition cases in the assignment
    assert np.sum(X[X_att == 1]) == np.sum(X_att)
    # Y present and assignment 0 => no cases
    assert np.sum((np.isnan(Y) == False) & (X == 0)) == 0
    # proposed assignment has positive probability
    assert np.sum(Fa[X == 1] == 0) == 0

    P_minus = (Fa == 0) & (Fb > 0)
    P_plus = ~P_minus 
    assert np.all(P_plus + P_minus == 1)
    P_abs = P_plus & (X_att == 1) & absent_reviewers[np.newaxis, :]
    P_att = P_plus & (X_att == 1) & ~absent_reviewers[np.newaxis, :]
    P_pres = P_plus & (~P_att) & (~P_abs) 
    assert np.all(P_att + P_pres + P_abs == P_plus)

    N = np.sum(X)
    Y[np.isnan(Y)] = -10000 # clean up nans, should not be used in computation
    Pb_pos = np.sum(Fb[P_plus]) / N
    W = np.divide(Fb, Fa, out=np.zeros_like(Fa), where=(Fa > 0))
    XW = X * W
    Coeff = np.zeros_like(Fa) # the coefficient for each term in the weighted avg
    Coeff[P_plus] = XW[P_plus]
    Coeff[P_minus] = Fb[P_minus]

    if np.sum(XW[P_pres]) == 0:
        print('Error: sum(X*W) = 0')
        mu_b_plus = np.nan
    else:
        mu_b_plus = np.sum(Y[P_pres] * XW[P_pres])/ np.sum(XW[P_pres]) # the Y estimate we will impute

    def compute_bounds(Y_to_impute, side=None):
        if type(Y_to_impute) is np.ndarray:
            assert Y_to_impute.size == np.sum(P_att | P_minus | P_abs)
        Y_ = Y.copy()
        Y_[P_att | P_minus | P_abs] = Y_to_impute # for lip/mon, use all computed values
        if not np.isnan(mu_b_plus):
            Y_[P_abs] = mu_b_plus # overwrite: impute absent with mu_b_plus in all cases
        YC_ = Y_ * Coeff
        mu_b = np.sum(YC_) / N
        var_b = None
        mu_b_95_ci = None
        if side is not None:
            var_b = (YC_[Fa > 0] @ Cov @ YC_[Fa > 0]) / N**2
            if side == 'lb':
                mu_b_95_ci = mu_b - (1.96 * np.sqrt(var_b / N))
            elif side == 'ub':
                mu_b_95_ci = mu_b + (1.96 * np.sqrt(var_b / N))
        return mu_b, var_b, mu_b_95_ci

    def compute_adaptive_bounds(mu_b_min, mu_b_max, var_b_min, var_b_max):
        Cn_obj = lambda Cn, SE: (0.95 - (norm.cdf(Cn + SE) - norm.cdf(-Cn)))**2

        SE_b = np.sqrt(N) * (mu_b_max - mu_b_min) \
             / max(np.sqrt(var_b_min), np.sqrt(var_b_max))
        
        opt_res = minimize(
            Cn_obj,
            x0=1.96,
            args=(SE_b),
            method='nelder-mead',
            options={'maxiter':1e2, 'xatol': 1e-16, 'disp': False}
        )
        
        Cn_b = opt_res.x[0]
        Cn_b_alpha = norm.cdf(Cn_b + SE_b) - norm.cdf(-Cn_b)

        mu_b_cn_ci_lb = mu_b_min - (Cn_b * np.sqrt(var_b_min / N))
        mu_b_cn_ci_ub = mu_b_max + (Cn_b * np.sqrt(var_b_max / N))
        return mu_b_cn_ci_lb, mu_b_cn_ci_ub

    res = {}
    if not compute_lip_only:
        # Impute mean
        mu_b_impute, _, _ = compute_bounds(mu_b_plus, side=None)
    
        # Manski
        mu_b_min, var_b_min, mu_b_95_ci_lb = compute_bounds(y_min, side='lb')
        mu_b_max, var_b_max, mu_b_95_ci_ub = compute_bounds(y_max, side='ub')
        mu_b_cn_ci_lb, mu_b_cn_ci_ub = compute_adaptive_bounds(mu_b_min, mu_b_max, var_b_min, var_b_max)
    
        if verbose:
            print(f'Imputed mean: {mu_b_impute}')
            print(f'Width for Manski: {mu_b_max - mu_b_min}')
            print(f'Width for Manski 95-CI: {mu_b_95_ci_ub - mu_b_95_ci_lb}')
            print(f'Width for Manski Adaptive-CI: {mu_b_cn_ci_ub - mu_b_cn_ci_lb}')

        total_variation = 0.5 * np.sum(np.abs(Fa - Fb))
    
        res = {
            # meta
            "outcome": Y_var,
            "name_b": rc.get_fname(),
            # B
            "Pb_pos": Pb_pos,
            "mu_b_impute": mu_b_impute,
            "mu_b_min": mu_b_min,
            "mu_b_max": mu_b_max,
            "mu_b_95_ci_lb" : mu_b_95_ci_lb,
            "mu_b_95_ci_ub" : mu_b_95_ci_ub,
            "mu_b_cn_ci_lb" : mu_b_cn_ci_lb,
            "mu_b_cn_ci_ub" : mu_b_cn_ci_ub,
            # AB
            "total_variation": total_variation
        }

        # Model imputation
        for model_name, Y_pred in data_loader.Y_preds.items():
            mu_b_impute_model, _, _ = compute_bounds(Y_pred[P_minus | P_att | P_abs], side=None)
            res[f'mu_b_impute_{model_name}'] = mu_b_impute_model
 

    # Lipschitz handling
    if len(lipschitz_Ls) > 0:
        obs_idxs = Obs.flatten()
        violations = (P_minus | P_att | P_abs).flatten() 
        coeff_flat = Coeff.flatten()

        if np.sum(violations) > 1e5:
            print(f'Skipping lipschitz due to too many violations : {np.sum(violations)}')
            return  res


        analysis = LipMonAnalysis(rc=rc, Y_var=Y_var, Y=Y, coeff_flat=coeff_flat, violations=violations, obs_idxs=obs_idxs, y_min=y_min, y_max=y_max, verbose=verbose)
        for L in lipschitz_Ls:
            if compute_lip_only:
                T_min, T_max = analysis.compute_T(L)
            else:
                try:
                    T_min, T_max = analysis.load_T(L)
                except FileNotFoundError as e:
                    T_min, T_max = None, None
                    print(f'No T file found for L={L}')
    
                missing = [np.all(T_min == None), np.all(T_max == None)] 
                if any(missing):
                    print('Missing T_min, T_max = {missing}')
                    continue

                mu_b_min_L, var_b_min_L, mu_b_95_ci_lb_L = compute_bounds(T_min[analysis.N_obs:], side='lb')
                mu_b_max_L, var_b_max_L, mu_b_95_ci_ub_L = compute_bounds(T_max[analysis.N_obs:], side='ub')
                mu_b_cn_ci_lb_L, mu_b_cn_ci_ub_L = compute_adaptive_bounds(mu_b_min_L, mu_b_max_L, var_b_min_L, var_b_max_L)
    
                if verbose:
                    print(f'Width for L={L}: {mu_b_max_L - mu_b_min_L}')
                    print(f'Width for L={L} 95-CI: {mu_b_95_ci_ub_L - mu_b_95_ci_lb_L}')
                    print(f'Width for L={L} Adaptive-CI: {mu_b_cn_ci_ub_L - mu_b_cn_ci_lb_L}')
    
                if L is None:
                    lstring = 'mon'
                else:
                    lstring = f'lip{L}'
                L_results = {f'mu_b_min_{lstring}' : mu_b_min_L, f'mu_b_max_{lstring}' : mu_b_max_L, 
                        f'mu_b_95_ci_lb_{lstring}' : mu_b_95_ci_lb_L, f'mu_b_95_ci_ub_{lstring}' : mu_b_95_ci_ub_L,
                        f'mu_b_cn_ci_lb_{lstring}' : mu_b_cn_ci_lb_L, f'mu_b_cn_ci_ub_{lstring}' : mu_b_cn_ci_ub_L,
                        f'lambda_{lstring}' : analysis.lam}
                res.update(L_results)

    return res

def sample(n_jobs, configs):
    from tqdm import tqdm
    parallel = Parallel(n_jobs=n_jobs, verbose=10)
    parallel(
        delayed(analyses_HT)(**kwargs, verbose=False, data_loader=None, compute_lip_only=True) \
            for kwargs in tqdm(configs)
        )


def main():
    args = utils.parse_arguments()
    Y_var = args.outcome
    stage = args.stage
    n_jobs = args.n_jobs
    is_mono = args.use_monotonicity
    param_names = args.xparam

    Ls = []
    if args.use_lipschitz:
        Ls = utils.get_lipschitz_Ls(stage=stage, Y_var=Y_var, use_monotonicity=is_mono) 
    elif args.use_monotonicity:
        Ls = [None]
    print(Y_var, stage, n_jobs, Ls, param_names)

    configs = []
    for x_param in param_names + [None]:
        for param in utils.get_experiment_params(x_param, stage):
            rc = RunConfig(stage, param)
            kwargs = {'rc':rc, 'Y_var' : Y_var, 'lipschitz_Ls' : Ls}
            configs.append(kwargs)

    if n_jobs > 1:
	# only compute Lipschitz/monotonicity bounds
        sample(n_jobs, configs)

    if n_jobs == 1:
        # save data
        param_string = '-'.join(param_names)
        exp_name = f'X{param_string}_Y{Y_var}_S{stage}'

        data_loader = DataLoader(stage, Y_var)
        res_list = []
        for kwargs in configs:
            res = analyses_HT(**kwargs, verbose=True, data_loader=data_loader, compute_lip_only=False)
            res.update(kwargs['rc'].params)
            res_list.append(res)
        df = pd.DataFrame(res_list)
        df.to_csv(f'{exp_name}_results.csv')

if __name__ == "__main__":
    main()
