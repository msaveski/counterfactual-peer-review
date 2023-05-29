
import csv
import click
import numpy as np

from pathlib import Path
from itertools import product
from scipy.stats import norm
from scipy.optimize import minimize
from ortools.linear_solver import pywraplp

from joblib import Parallel, delayed

from core import ROOT, RunConfig, DataLoader



def solve_lipschitz_LPs(L, lambda_, Y_obs, W_vio, D_obs_vio_pairs, y_min, y_max, tol=1e-13):
    
    N_obs = Y_obs.shape[0]
    N_vio = W_vio.shape[0]
    
    #
    # Set up the LP
    #
    solver = pywraplp.Solver.CreateSolver('GLOP')

    # create variables 
    theta = [solver.NumVar(0, solver.infinity(), f"theta_{i}") for i in range(N_obs)]
    T_obs = [solver.NumVar(y_min, y_max, f"T_obs_{i}") for i in range(N_obs)]
    T_vio = [solver.NumVar(y_min, y_max, f"T_vio_{i}") for i in range(N_vio)]
    T = T_obs + T_vio

    # set the contraints
    for i in range(N_obs + N_vio):
        for j in range(i + 1, N_obs + N_vio):
            solver.Add((T[i] - T[j]) <= L * D_obs_vio_pairs[i, j])
            solver.Add((T[j] - T[i]) <= L * D_obs_vio_pairs[i, j])

    for i in range(N_obs):
        solver.Add((T_obs[i] - Y_obs[i]) <= theta[i])
        solver.Add((Y_obs[i] - T_obs[i]) <= theta[i])

    # set the objective
    for i in range(N_obs):
        solver.Objective().SetCoefficient(T_obs[i], 0)

    for i in range(N_vio):
        solver.Objective().SetCoefficient(T_vio[i], W_vio[i])
    
    #
    # Solve minimization
    #
    for i in range(N_obs):
        solver.Objective().SetCoefficient(theta[i], lambda_)

    solver.Objective().SetMinimization()

    status = solver.Solve()

    T_obs_min = None
    T_vio_min = None

    if status == pywraplp.Solver.OPTIMAL:
        T_obs_min = [i.solution_value() for i in T_obs]
        T_vio_min = [i.solution_value() for i in T_vio]
        theta_ = [i.solution_value() for i in theta]
        T_min = T_obs_min + T_vio_min

        # verify the solution
        for i in range(N_obs + N_vio):
            for j in range(i + 1, N_obs + N_vio):
                assert abs(T_min[i] - T_min[j]) <= (L * D_obs_vio_pairs[i, j] + tol)

        for i in range(N_obs):
            assert abs(Y_obs[i] - T_obs_min[i]) < (theta_[i] + tol)
    
    #
    # Solve maximization
    #
    for i in range(N_obs):
        solver.Objective().SetCoefficient(theta[i], -lambda_)

    solver.Objective().SetMaximization()

    status = solver.Solve()

    T_obs_max = None
    T_vio_max = None

    if status == pywraplp.Solver.OPTIMAL:
        T_obs_max = [i.solution_value() for i in T_obs]
        T_vio_max = [i.solution_value() for i in T_vio]
        theta_ = [i.solution_value() for i in theta]
        T_max = T_obs_max + T_vio_max

        # verify the solution
        for i in range(N_obs + N_vio):
            for j in range(i + 1, N_obs + N_vio):
                assert abs(T_max[i] - T_max[j]) <= (L * D_obs_vio_pairs[i, j] + tol)

        for i in range(N_obs):
            assert abs(Y_obs[i] - T_obs_max[i]) < (theta_[i] + tol)
    
    return T_vio_min, T_vio_max


def norm_zero_one(X_all):
    assert X_all.shape[1] == 2
    for dim in range(X_all.shape[1]):
        Xi = X_all[:, dim]
        mask = ~np.isnan(Xi)
        a = np.min(Xi[mask])
        b = np.max(Xi[mask])
        X_all[:, dim] = (Xi - a) / (b - a)
    return X_all


# l1 distance
def cdist_lip(X1, X2):
    D = np.zeros((X1.shape[0], X2.shape[0]))
    for i in range(X1.shape[1]):
        Di = np.abs(np.subtract.outer(X1[:, i], X2[:, i]))
        Di[np.isnan(Di)] = 1
        D += Di
    D /= X1.shape[1]
    return D


def analyses_HT_lipschitz(config_b, Y_var, L_vals, lambda_):

    EXPS_DIR = Path(f"{ROOT}/exps/")

    #
    # load common data
    #
    dl = DataLoader()

    X = dl.get_proposed_assignment_matrix()

    X_att, X_new = dl.get_manual_reassignments()

    # NB: default bid-value map
    Bids = dl.get_bids_matrix(RunConfig())

    Aff = dl.get_affinity_matrix()

    Y = dl.get_outcome_matrix(Y_var)

    y_min, y_max = dl.get_outcome_bounds(Y_var)

    # to avoid errors remove Ys observed due to manual reassignment
    Y[X_new == 1] = np.nan

    #
    # load A data
    #
    config_a = RunConfig()

    assert config_a.get_name() == "_onpolicy"

    A_path = EXPS_DIR / config_a.get_name()

    with (A_path / "F.npy").open("rb") as fin:
        Fa = np.load(fin)

    with (A_path / "covariance.npy").open("rb") as fin:
        Cov = np.load(fin)

    #
    # load B data
    #
    B_path = EXPS_DIR / config_b.get_name()

    with (B_path / "F.npy").open("rb") as fin:
        Fb = np.load(fin)

    # new: values ~0 => 0 (numpy can be a bit inconsistent)
    sum_Fa = np.sum(Fa)
    sum_Fb = np.sum(Fb)

    Fa[Fa < 1e-13] = 0
    Fb[Fb < 1e-13] = 0

    assert np.isclose(np.sum(Fa), sum_Fa)
    assert np.isclose(np.sum(Fb), sum_Fb)

    #
    # flatten
    #
    X = X.flatten()
    X_att = X_att.flatten()

    Bids = Bids.flatten()
    Aff = Aff.flatten()

    Y = Y.flatten()

    Fa = Fa.flatten()
    Fb = Fb.flatten()

    #
    # sanity checks
    #

    # all assignments, except for attrition, have an outcome
    assert np.sum(np.isnan(Y[(X == 1) & (X_att != 1)])) == 0

    # all attrition cases in the assignment
    assert np.sum(X[X_att == 1]) == np.sum(X_att)

    # all assignment, expect the manual reassignments, have Fa = 0.5
    assert np.sum(Fa[(X == 1) & (X_att != 1)] != 0.5) == 0

    # Y present and assignment 0 => no cases
    assert np.sum((np.isnan(Y) == False) & (X == 0)) == 0

    # proposed assignment has positive probability
    assert np.sum(Fa[X == 1] != 0.5) == 0

    #
    # Common variables
    #
    pairs_obs = (X == 1) & (X_att != 1)

    Y_obs = Y[pairs_obs]

    Cov_obs = Cov[pairs_obs, :][:, pairs_obs]

    N_obs = np.sum(pairs_obs)

    N = np.sum(X)

    assert N_obs == 282
    assert N == 285

    Cn_obj = lambda Cn, SE: (0.95 - (norm.cdf(Cn + SE) - norm.cdf(-Cn)))**2

    #
    # Prepare the data for the Lipschitz LPs
    #

    # violations
    pairs_vio = ((Fa == 0) & (Fb > 0)) | (X_att == 1)

    N_vio = np.sum(pairs_vio)

    # covariates
    X_all = np.column_stack((Bids, Aff))
    X_all = norm_zero_one(X_all)
    X_vio = X_all[pairs_vio, :]
    X_obs = X_all[pairs_obs, :]
    X_obs_vio = np.row_stack((X_obs, X_vio))

    # distances
    D_obs_vio_pairs = cdist_lip(X_obs_vio, X_obs_vio)

    # set the weights of the violations
    pairs_att = (Fa > 0)  & (X_att == 1)
    pairs_neg = (Fa == 0) & (Fb > 0)

    with np.errstate(divide='ignore', invalid='ignore'):
        W = Fb / Fa

    W_vio_full = np.full_like(W, fill_value=np.nan)
    W_vio_full[pairs_att] = W[pairs_att]
    W_vio_full[pairs_neg] = Fb[pairs_neg]

    W_vio = W_vio_full[pairs_vio]
    
    #
    # Loop
    #
    all_results = []

    for L in L_vals:
        # print("L =", L)

        #
        # Solve LPs
        #
        T_min, T_max = solve_lipschitz_LPs(
            L, lambda_, 
            Y_obs, W_vio, 
            D_obs_vio_pairs, 
            y_min, y_max, 
            tol=1e-13
        )

        if T_min is None or T_max is None:
            continue

        # T to arrays
        T_min_arr = np.full_like(Y, fill_value=np.nan)
        T_min_arr[pairs_vio] = T_min

        T_max_arr = np.full_like(Y, fill_value=np.nan)
        T_max_arr[pairs_vio] = T_max


        #
        # Estimation (off-policy)
        #
        pairs_pos     = (Fa > 0)
        pairs_pos_asg = (Fa > 0) & (X == 1)

        pairs_pres     = (Fa > 0) & (X_att != 1)
        pairs_pres_obs = (X == 1) & (X_att != 1) # = obs_idx

        pairs_att = (Fa > 0)  & (X_att == 1)

        pairs_neg = (Fa == 0) & (Fb > 0)    

        # sanity checks
        assert np.sum(pairs_pos) == np.sum(pairs_pres | pairs_att)
        assert np.sum(pairs_pos | pairs_neg) == np.sum((Fa > 0) | (Fb > 0))
        assert np.sum(pairs_pres | pairs_att | pairs_neg) == np.sum((Fa > 0) | (Fb > 0))
        assert np.sum(Fa[pairs_pres]) + np.sum(Fa[pairs_att]) == np.sum(Fa)
        assert np.isclose(np.sum(Fb[pairs_pos | pairs_neg]), np.sum(Fb))
        assert np.isclose(np.sum(Fb[pairs_pres | pairs_att | pairs_neg]), np.sum(Fb))
        assert np.sum(pairs_pos_asg) == N
        assert np.sum(pairs_pres_obs) == N_obs
        assert np.sum(pairs_pres_obs) + np.sum(pairs_att) == N

        # weights
        with np.errstate(divide='ignore', invalid='ignore'):
            W = Fb / Fa

        ess_b = np.square(np.sum(W[pairs_pres_obs])) \
              / np.sum(np.square(W[pairs_pres_obs]))

        mu_b_plus = np.sum(Y[pairs_pres_obs] * W[pairs_pres_obs]) / np.sum(W[pairs_pres_obs])

        mu_b_impute = (np.sum(Y[pairs_pres_obs] * W[pairs_pres_obs]) \
                    + np.sum(mu_b_plus * W[pairs_att]) \
                    + np.sum(mu_b_plus * Fb[pairs_neg])) \
                    / N                

        # variance
        assert np.all(pairs_pres_obs == pairs_obs)
        YW_obs = Y[pairs_pres_obs] * W[pairs_pres_obs]
        var_b = (YW_obs @ Cov_obs @ YW_obs) / N_obs**2

        # mu_b_impute: 95% CIs
        mu_b_impute_95_ci = 1.96 * np.sqrt(var_b / ess_b)
        mu_b_impute_95_lb = mu_b_impute - mu_b_impute_95_ci
        mu_b_impute_95_ub = mu_b_impute + mu_b_impute_95_ci

        # mu_b: Manski bounds
        pairs_min_max = (pairs_pres_obs | pairs_att | pairs_neg)

        Y_b_min = np.copy(Y)
        Y_b_min[pairs_att] = y_min
        Y_b_min[pairs_neg] = y_min

        Y_b_max = np.copy(Y)
        Y_b_max[pairs_att] = y_max
        Y_b_max[pairs_neg] = y_max

        W_min_max = np.copy(W)
        W_min_max[pairs_neg] = Fb[pairs_neg]

        mu_b_min = np.sum(Y_b_min[pairs_min_max] * W_min_max[pairs_min_max]) / N
        mu_b_max = np.sum(Y_b_max[pairs_min_max] * W_min_max[pairs_min_max]) / N

        # mu_b: Manski variances
        Cov_min_max = Cov[pairs_min_max, :][:, pairs_min_max]

        YW_b_min = Y_b_min[pairs_min_max] * W_min_max[pairs_min_max]
        YW_b_max = Y_b_max[pairs_min_max] * W_min_max[pairs_min_max]

        var_b_min = (YW_b_min @ Cov_min_max @ YW_b_min) / N**2
        var_b_max = (YW_b_max @ Cov_min_max @ YW_b_max) / N**2

        # mu_b: Manski bounds + 95% CIs
        mu_b_95_ci_lb = mu_b_min - (1.96 * np.sqrt(var_b_min / N))
        mu_b_95_ci_ub = mu_b_max + (1.96 * np.sqrt(var_b_max / N))

        # mu_b: Manski bounds + adapative CIs per Imbens & Manski
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

        #
        # Lipschitz
        #

        # mu_b: Lipschitz Manski bounds
        pairs_min_max = (pairs_pres_obs | pairs_att | pairs_neg)

        Y_b_min_lip = np.copy(Y)
        Y_b_min_lip[pairs_att] = T_min_arr[pairs_att]
        Y_b_min_lip[pairs_neg] = T_min_arr[pairs_neg]

        Y_b_max_lip = np.copy(Y)
        Y_b_max_lip[pairs_att] = T_max_arr[pairs_att]
        Y_b_max_lip[pairs_neg] = T_max_arr[pairs_neg]

        W_min_max = np.copy(W)
        W_min_max[pairs_neg] = Fb[pairs_neg]

        mu_b_lip_min = np.sum(Y_b_min_lip[pairs_min_max] * W_min_max[pairs_min_max]) / N
        mu_b_lip_max = np.sum(Y_b_max_lip[pairs_min_max] * W_min_max[pairs_min_max]) / N

        # mu_b: Lipschitz Manski variances
        Cov_min_max = Cov[pairs_min_max, :][:, pairs_min_max]

        YW_b_lip_min = Y_b_min_lip[pairs_min_max] * W_min_max[pairs_min_max]
        YW_b_lip_max = Y_b_max_lip[pairs_min_max] * W_min_max[pairs_min_max]

        var_b_lip_min = (YW_b_lip_min @ Cov_min_max @ YW_b_lip_min) / N**2
        var_b_lip_max = (YW_b_lip_max @ Cov_min_max @ YW_b_lip_max) / N**2

        # mu_b: Lipschitz Manski bounds + 95% CIs
        mu_b_lip_95_ci_lb = mu_b_lip_min - (1.96 * np.sqrt(var_b_lip_min / N))
        mu_b_lip_95_ci_ub = mu_b_lip_max + (1.96 * np.sqrt(var_b_lip_max / N))

        # mu_b: Lipschitz Manski bounds + adapative CIs per Imbens & Manski
        SE_b_lip = np.sqrt(N) * (mu_b_lip_max - mu_b_lip_min) \
                 / max(np.sqrt(var_b_lip_min), np.sqrt(var_b_lip_max))

        opt_res = minimize(
            Cn_obj, 
            x0=1.96, 
            args=(SE_b_lip),
            method='nelder-mead', 
            options={'maxiter':1e2, 'xatol': 1e-16, 'disp': False}
        )

        Cn_b_lip = opt_res.x[0]

        Cn_b_lip_alpha = norm.cdf(Cn_b_lip + SE_b_lip) - norm.cdf(-Cn_b_lip)

        mu_b_lip_cn_ci_lb = mu_b_lip_min - (Cn_b_lip * np.sqrt(var_b_lip_min / N))
        mu_b_lip_cn_ci_ub = mu_b_lip_max + (Cn_b_lip * np.sqrt(var_b_lip_max / N))

        bound_prc_reduction = (mu_b_lip_max - mu_b_lip_min) / (mu_b_max - mu_b_min) - 1
        bound_CIs_prc_reduction = (mu_b_lip_cn_ci_ub - mu_b_lip_cn_ci_lb) \
                                / (mu_b_cn_ci_ub - mu_b_cn_ci_lb) - 1

        all_results.append({
            # meta
            "outcome": Y_var,    
            "name_b": config_b.get_name(),
            **config_b.get_params(),
            # Lipschitz settings
            "lip_L": L,
            "lip_lambda": lambda_,
            # Mean imputation
            "mu_b_impute": mu_b_impute,
            "mu_b_impute_95_lb": mu_b_impute_95_lb,
            "mu_b_impute_95_ub": mu_b_impute_95_ub,
            # Manski bounds
            "mu_b_min": mu_b_min,
            "mu_b_max": mu_b_max,
            "mu_b_95_ci_lb": mu_b_95_ci_lb,
            "mu_b_95_ci_ub": mu_b_95_ci_ub,
            "mu_b_cn_ci_lb": mu_b_cn_ci_lb,
            "mu_b_cn_ci_ub": mu_b_cn_ci_ub,
            # Lipschitz bounds
            "mu_b_lip_min": mu_b_lip_min,
            "mu_b_lip_max": mu_b_lip_max,
            "mu_b_lip_95_ci_lb": mu_b_lip_95_ci_lb,
            "mu_b_lip_95_ci_ub": mu_b_lip_95_ci_ub,
            "mu_b_lip_cn_ci_lb": mu_b_lip_cn_ci_lb,
            "mu_b_lip_cn_ci_ub": mu_b_lip_cn_ci_ub,
            # 
            "lip_bound_prc_reduction": bound_prc_reduction,
            "lip_bound_CIs_prc_reduction": bound_CIs_prc_reduction
        })

    return all_results


def test():    
    L_vals = [5, 6, 8, 20, 30, 50, 60, 300, 400]    
    lambda_ = 1e9
    Y_var = "expertise"
    
    configs = RunConfig.get_all("H2")
    configs_map = {cfg.get_name(): cfg for cfg in configs}
    config_b = configs_map["Q_1_0000__aff_w_1_0000__bid_w_0_0000"]
    
    config_b_res = analyses_HT_lipschitz(config_b, Y_var, L_vals, lambda_)
    
    print(">>", len(config_b_res))
    
    for i in config_b_res:
        print(
            i["lip_L"], 
            i["lip_lambda"], 
            i["lip_bound_prc_reduction"], 
            abs(i["mu_b_max"] - i["mu_b_min"]),
            abs(i["mu_b_lip_max"] - i["mu_b_lip_min"])
        )


@click.command()
@click.option('--h_num',type=click.Choice(["H1", "H2"]), required=True)
@click.option('--n_jobs', required=True, type=int)
def main(h_num, n_jobs):
    
    L_vals = [5, 6, 8, 20, 30, 50, 60, 300, 400]

    lambda_ = 1e9

    Y_vars = ["confidence", "expertise"]
    
    configs = RunConfig.get_all(h_num)
    
    all_res = []
    
    if n_jobs == 1:
        for Y_var, config in product(Y_vars, configs):
            print(f">> {config.get_name()} | {Y_var}")
            config_res = analyses_HT_lipschitz(config, Y_var, L_vals, lambda_)
            all_res += config_res
            
    else:
        parallel = Parallel(n_jobs=n_jobs, verbose=10)
        runs_res = parallel(
            delayed(analyses_HT_lipschitz)(config, Y_var, L_vals, lambda_) \
                for Y_var, config in product(Y_vars, configs)
            )
        
        # flatten the list of lists
        for run_res in runs_res:
            for res in run_res:
                all_res.append(res)
    
    # output to csv
    csv_fpath = f"{ROOT}/results/{h_num}_lipschitz.csv"

    with open(csv_fpath, "w") as fout:
        fieldnames = list(all_res[0].keys())
        writer = csv.DictWriter(fout, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_res)

    print("Done!")
    
        
if __name__ == "__main__":
    main()
    # test()

# END