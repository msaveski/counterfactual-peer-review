
import csv
import click
import numpy as np

from pathlib import Path
from itertools import product
from scipy.stats import norm
from scipy.optimize import minimize

from joblib import Parallel, delayed

from core import ROOT, RunConfig, DataLoader


def analyses_HT(config_b, Y_var, verbose=True):

    if verbose:
        print(f">> {config_b.get_name()} | {Y_var}")

    EXPS_DIR = Path(f"{ROOT}/exps/")

    #
    # load common data
    #
    dl = DataLoader()

    X = dl.get_proposed_assignment_matrix()

    X_att, X_new = dl.get_manual_reassignments()

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

    with (B_path / "F_fraction_of_optimal.txt").open("rb") as fin:
        Fb_fraction_optimal = float(fin.read())

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

    Y = Y.flatten()

    Fa = Fa.flatten()
    Fb = Fb.flatten()

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
    # A (on-policy)
    #

    # mu_a_impute: mean imputation
    mu_a_impute = np.sum(Y_obs) / N_obs

    # variance
    var_a = (Y_obs @ Cov_obs @ Y_obs) / N_obs**2

    # mu_a_impute: 95% CIs
    mu_a_impute_95_ci = 1.96 * np.sqrt(var_a / N_obs)
    mu_a_impute_95_lb = mu_a_impute - mu_a_impute_95_ci
    mu_a_impute_95_ub = mu_a_impute + mu_a_impute_95_ci

    # mu_a: Manski bounds
    pairs_att = (X_att == 1)

    Y_a_min = np.copy(Y)
    Y_a_min[pairs_att] = y_min

    Y_a_max = np.copy(Y)
    Y_a_max[pairs_att] = y_max

    mu_a_min = np.sum(Y_a_min[pairs_obs|pairs_att] * X[pairs_obs|pairs_att]) / N
    mu_a_max = np.sum(Y_a_max[pairs_obs|pairs_att] * X[pairs_obs|pairs_att]) / N

    # mu_a: Manski bounds sanity check
    mu_a_min_tmp = (np.sum(Y_obs) + (N - N_obs) * y_min) / N
    mu_a_max_tmp = (np.sum(Y_obs) + (N - N_obs) * y_max) / N

    assert np.isclose(mu_a_min_tmp, mu_a_min)
    assert np.isclose(mu_a_max_tmp, mu_a_max)

    # mu_a: Manski variances
    Cov_obs_att = Cov[pairs_obs|pairs_att, :][:, pairs_obs|pairs_att]

    var_a_min = (Y_a_min[pairs_obs|pairs_att] @ Cov_obs_att @ Y_a_min[pairs_obs|pairs_att]) / N**2

    var_a_max = (Y_a_max[pairs_obs|pairs_att] @ Cov_obs_att @ Y_a_max[pairs_obs|pairs_att]) / N**2

    # mu_a: Manski bounds + 95% CIs
    mu_a_95_ci_lb = mu_a_min - (1.96 * np.sqrt(var_a_min / N))
    mu_a_95_ci_ub = mu_a_max + (1.96 * np.sqrt(var_a_max / N))

    # mu_a: Manski bounds + adapative CIs per Imbens & Manski
    SE_a = np.sqrt(N) * (mu_a_max - mu_a_min) \
         / max(np.sqrt(var_a_min), np.sqrt(var_a_max))

    opt_res = minimize(
        Cn_obj, 
        x0=1.96, 
        args=(SE_a),
        method='nelder-mead', 
        options={'maxiter':1e2, 'xatol': 1e-16, 'disp': False}
    )

    Cn_a = opt_res.x[0]
    
    Cn_a_alpha = norm.cdf(Cn_a + SE_a) - norm.cdf(-Cn_a)

    mu_a_cn_ci_lb = mu_a_min - (Cn_a * np.sqrt(var_a_min / N))
    mu_a_cn_ci_ub = mu_a_max + (Cn_a * np.sqrt(var_a_max / N))

    # sanity check the order of the estimates
    assert mu_a_95_ci_lb < mu_a_cn_ci_lb < mu_a_min < mu_a_impute
    assert mu_a_impute < mu_a_max < mu_a_cn_ci_ub < mu_a_95_ci_ub


    #
    # B (off-policy)
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

    # unused
    Pb_pres = np.sum(Fb[pairs_pres]) / N

    with np.errstate(divide='ignore', invalid='ignore'):
        W = Fb / Fa

    # unused
    ess_b = np.square(np.sum(W[pairs_pres_obs])) \
          / np.sum(np.square(W[pairs_pres_obs]))

    # variance
    assert np.all(pairs_pres_obs == pairs_obs)

    YW_obs = Y[pairs_pres_obs] * W[pairs_pres_obs]

    var_b = (YW_obs @ Cov_obs @ YW_obs) / N_obs**2

    mu_b_plus = np.sum(Y[pairs_pres_obs] * W[pairs_pres_obs]) / np.sum(W[pairs_pres_obs])

    mu_b_impute = (np.sum(Y[pairs_pres_obs] * W[pairs_pres_obs]) \
                + np.sum(mu_b_plus * W[pairs_att]) \
                + np.sum(mu_b_plus * Fb[pairs_neg])) \
                / N                

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

    # mu_b: Manski bounds, sanity check
    mu_b_min_tmp = (np.sum(Y[pairs_pres_obs] * W[pairs_pres_obs]) \
                 + y_min * np.sum(W[pairs_att]) \
                 + y_min * np.sum(Fb[pairs_neg])) / N

    mu_b_max_tmp = (np.sum(Y[pairs_pres_obs] * W[pairs_pres_obs]) \
                 + y_max * np.sum(W[pairs_att]) \
                 + y_max * np.sum(Fb[pairs_neg])) / N

    assert np.isclose(mu_b_min, mu_b_min_tmp)
    assert np.isclose(mu_b_max, mu_b_max_tmp)

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

    # sanity check the order of the estimates
    assert mu_b_95_ci_lb < mu_b_cn_ci_lb < mu_b_min < mu_b_impute
    assert mu_b_impute < mu_b_max < mu_b_cn_ci_ub < mu_b_95_ci_ub

    #
    # A / B: total variation
    #
    total_variation = 0.5 * np.sum(np.abs(Fa - Fb))

    #
    # output dict
    #
    res = {
        # meta
        "outcome": Y_var,    
        "name_b": config_b.get_name(),
        **config_b.get_params(),
        # A
        "mu_a_impute": mu_a_impute,
        "mu_a_min": mu_a_min,
        "mu_a_max": mu_a_max,
        "var_a": var_a,
        "mu_a_impute_95_lb": mu_a_impute_95_lb,
        "mu_a_impute_95_ub": mu_a_impute_95_ub,
        "mu_a_95_ci_lb": mu_a_95_ci_lb,
        "mu_a_95_ci_ub": mu_a_95_ci_ub,
        "Cn_a": Cn_a,
        "Cn_a_alpha": Cn_a_alpha,
        "mu_a_cn_ci_lb": mu_a_cn_ci_lb,
        "mu_a_cn_ci_ub": mu_a_cn_ci_ub,
        # B
        "ess_b": ess_b,
        "mu_b_impute": mu_b_impute,
        "mu_b_min": mu_b_min,
        "mu_b_max": mu_b_max,
        "var_b": var_b,    
        "mu_b_impute_95_lb": mu_b_impute_95_lb,
        "mu_b_impute_95_ub": mu_b_impute_95_ub,
        "mu_b_95_ci_lb": mu_b_95_ci_lb,
        "mu_b_95_ci_ub": mu_b_95_ci_ub,
        "Cn_b": Cn_b,
        "Cn_b_alpha": Cn_b_alpha,
        # "mu_b_cn_ci": mu_b_cn_ci,    
        "mu_b_cn_ci_lb": mu_b_cn_ci_lb,
        "mu_b_cn_ci_ub": mu_b_cn_ci_ub,
        "fraction_optimal_b": Fb_fraction_optimal,
        # AB
        "total_variation": total_variation
    }

    return res


@click.command()
@click.option('--h_num',type=click.Choice(["H1", "H2"]), required=True)
@click.option('--n_jobs', required=True, type=int)
def main(h_num, n_jobs):
    
    Y_vars = ["confidence", "expertise"]

    configs = RunConfig.get_all(h_num)

    if n_jobs == 1:
        all_res = [analyses_HT(config, Y_var, verbose=True) \
            for Y_var, config in product(Y_vars, configs)]

    else:
        parallel = Parallel(n_jobs=n_jobs, verbose=10)
        all_res = parallel(
            delayed(analyses_HT)(config, Y_var, verbose=False) \
                for Y_var, config in product(Y_vars, configs)
            )

    csv_fpath = f"{ROOT}/results/{h_num}_manski.csv"

    with open(csv_fpath, "w") as fout:
        fieldnames = list(all_res[0].keys())
        writer = csv.DictWriter(fout, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_res)

    print("Done!")


if __name__ == "__main__":
    main()

# END