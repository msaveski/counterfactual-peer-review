
import csv
import click
import numpy as np

from pathlib import Path
from itertools import product
from scipy.stats import norm
from scipy.optimize import minimize
from joblib import Parallel, delayed

from core import ROOT, RunConfig, DataLoader



def read_predictions(preds_fpath):
    
    with open(preds_fpath) as fin:
        rows = []
        for row in csv.DictReader(fin):
            rows.append({
                "paper_idx": int(row["paper_idx"]),
                "reviewer_idx": int(row["reviewer_idx"]),
                "pred": float(row["pred"])
            })

    n_papers = max([row["paper_idx"] for row in rows]) + 1
    n_reviewers = max([row["reviewer_idx"] for row in rows]) + 1
    
    Y_preds = np.zeros((n_papers, n_reviewers))

    for row in rows:
        Y_preds[row["paper_idx"], row["reviewer_idx"]] = row["pred"]

    assert np.all(Y_preds != 0)
    
    return Y_preds


def analyses(config_b, Y_var, model_name, verbose=True):

    if verbose:
        print(f">> {config_b.get_name()} | {Y_var} | {model_name}")

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
    # load predictions
    #
    preds_fpath = f"{ROOT}/model_imputation/preds/{Y_var}_{model_name}.csv"
    Y_preds = read_predictions(preds_fpath)

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

    #
    # flatten
    #
    X = X.flatten()
    X_att = X_att.flatten()
    X_new = X_new.flatten()

    Y = Y.flatten()

    Y_preds = Y_preds.flatten()

    Fa = Fa.flatten()
    Fb = Fb.flatten()


    #
    # On-policy
    #
    N = np.sum(X == 1)
    pairs_asg = (X == 1)
    pairs_obs = ((X == 1) & (X_att != 1))
    pairs_att = ((X == 1) & (X_att == 1))

    Cov_asg = Cov[pairs_asg, :][:, pairs_asg]

    # impute only the attrition pairs
    Y_a = np.full_like(Y, np.nan)
    Y_a[pairs_obs] = Y[pairs_obs]
    Y_a[pairs_att] = Y_preds[pairs_att]

    mu_a_imp = np.sum(Y_a[pairs_asg]) / N

    var_a_imp = (Y_a[pairs_asg] @ Cov_asg @ Y_a[pairs_asg]) / N**2

    mu_a_imp_95_ci = 1.96 * np.sqrt(var_a_imp / N)
    mu_a_imp_95_lb = mu_a_imp - mu_a_imp_95_ci
    mu_a_imp_95_ub = mu_a_imp + mu_a_imp_95_ci

    # sanity check
    assert mu_a_imp_95_lb < mu_a_imp < mu_a_imp_95_ub


    #
    # B (off-policy)
    #

    # >> pairs hierarchy
    # - pos (& pos_asg = pres_obs + att)
    #     - pres (pres_obs = X_i = 1)
    #     - att 
    # - neg

    N = np.sum(X == 1)

    pairs_pos     = (Fa > 0)
    pairs_pos_asg = (Fa > 0) & (X == 1)

    pairs_pres     = (Fa > 0) & (X_att != 1)
    pairs_pres_obs = (X == 1) & (X_att != 1) # = obs_idx

    pairs_att = (Fa > 0)  & (X_att == 1)

    pairs_neg = (Fa == 0) & (Fb > 0)

    with np.errstate(divide='ignore', invalid='ignore'):
        W = Fb / Fa
    
    #
    # mu_b => Manski bounds
    #
    mu_b_min = (
        np.sum(Y[pairs_pres_obs] * W[pairs_pres_obs]) + 
        np.sum(y_min * W[pairs_att]) + 
        np.sum(y_min * Fb[pairs_neg])) / N

    mu_b_max = (
        np.sum(Y[pairs_pres_obs] * W[pairs_pres_obs]) + 
        np.sum(y_max * W[pairs_att]) + 
        np.sum(y_max * Fb[pairs_neg])) / N
    

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

    # mu_b: Manski bounds + adapative CIs per Imbens & Manski
    SE_b = np.sqrt(N) * (mu_b_max - mu_b_min) \
            / max(np.sqrt(var_b_min), np.sqrt(var_b_max))

    Cn_obj = lambda Cn, SE: (0.95 - (norm.cdf(Cn + SE) - norm.cdf(-Cn)))**2

    opt_res = minimize(
        Cn_obj, 
        x0=1.96, 
        args=(SE_b),
        method='nelder-mead', 
        options={'maxiter':1e2, 'xatol': 1e-16, 'disp': False}
    )

    Cn_b = opt_res.x[0]

    mu_b_cn_ci_lb = mu_b_min - (Cn_b * np.sqrt(var_b_min / N))
    mu_b_cn_ci_ub = mu_b_max + (Cn_b * np.sqrt(var_b_max / N))

    #
    # mu_b_imp => impute only attition & PV pairs
    #
    b_imp_pairs = pairs_pres_obs | pairs_att | pairs_neg

    W_b_imp = np.full_like(W, np.nan)
    W_b_imp[pairs_pres_obs] = W[pairs_pres_obs]
    W_b_imp[pairs_att] = W[pairs_att]
    W_b_imp[pairs_neg] = Fb[pairs_neg]

    Y_b_imp = np.full_like(Y, np.nan)
    Y_b_imp[pairs_pres_obs] = Y[pairs_pres_obs]
    Y_b_imp[pairs_att] = Y_preds[pairs_att]
    Y_b_imp[pairs_neg] = Y_preds[pairs_neg]

    mu_b_imp = np.sum(Y_b_imp[b_imp_pairs] * W_b_imp[b_imp_pairs]) / N

    # sanity check
    mu_b_imp_tmp = (
        np.sum(Y[pairs_pres_obs] * W[pairs_pres_obs]) + 
        np.sum(Y_preds[pairs_att] * W[pairs_att]) + 
        np.sum(Y_preds[pairs_neg] * Fb[pairs_neg])) / N

    assert np.isclose(mu_b_imp, mu_b_imp_tmp)

    YW_b_imp = Y_b_imp[b_imp_pairs] * W_b_imp[b_imp_pairs]
    Cov_b_imp = Cov[b_imp_pairs, :][:, b_imp_pairs]

    var_b_imp = (YW_b_imp @ Cov_b_imp @ YW_b_imp) / N**2

    mu_b_imp_95_ci = 1.96 * np.sqrt(var_b_imp / N)
    mu_b_imp_95_lb = mu_b_imp - mu_b_imp_95_ci
    mu_b_imp_95_ub = mu_b_imp + mu_b_imp_95_ci

    # sanity check
    assert mu_b_min < mu_b_imp < mu_b_max
    assert mu_b_imp_95_lb < mu_b_imp < mu_b_imp_95_ub

    res = {
        # meta
        "outcome": Y_var,
        "model": model_name,
        "name_b": config_b.get_name(),
        **config_b.get_params(),
        # A
        "mu_a_imp": mu_a_imp,
        "mu_a_imp_95_lb": mu_a_imp_95_lb,
        "mu_a_imp_95_ub": mu_a_imp_95_ub,
        # B
        "mu_b_min": mu_b_min,
        "mu_b_max": mu_b_max,
        "mu_b_cn_ci_lb": mu_b_cn_ci_lb,
        "mu_b_cn_ci_ub": mu_b_cn_ci_ub,
        "mu_b_imp": mu_b_imp,
        "mu_b_imp_95_lb": mu_b_imp_95_lb,
        "mu_b_imp_95_ub": mu_b_imp_95_ub
    }

    return res


@click.command()
@click.option('--n_jobs', required=True, type=int)
def main(n_jobs):

    Y_vars = ["expertise", "confidence"]
    
    model_names = [
        "logistic-reg",
        "logistic-reg-stded",
        "knn-clf",
        "rigdge-clf",
        "ord-logit",
        "ord-probit",
        "ord-logistic-at",
        "ord-logistic-it",
        "rec-svdpp",
        "rec-knn-with-means",
        "rec-knn-with-z-score"    
    ]

    #
    # H1
    #
    print("--- H1 ---")
    configs = RunConfig.get_all("H1")
    
    if n_jobs == 1:
        all_res = [analyses(cfg, Y_var, m_name, verbose=True) \
            for Y_var, cfg, m_name in product(Y_vars, configs, model_names)]
        
    else:
        parallel = Parallel(n_jobs=n_jobs, verbose=10)
        all_res = parallel(
            delayed(analyses)(cfg, Y_var, m_name, verbose=False) \
                for Y_var, cfg, m_name in product(Y_vars, configs, model_names)
            )
        
    csv_fpath = f"{ROOT}/results/H1_model_imputation.csv"

    with open(csv_fpath, "w") as fout:
        fieldnames = list(all_res[0].keys())
        writer = csv.DictWriter(fout, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_res)

    #
    # H2
    #
    print("--- H2 ---")
    configs = RunConfig.get_all("H2")

    if n_jobs == 1:
        all_res = [analyses(cfg, Y_var, m_name, verbose=True) \
            for Y_var, cfg, m_name in product(Y_vars, configs, model_names)]
        
    else:
        parallel = Parallel(n_jobs=n_jobs, verbose=10)
        all_res = parallel(
            delayed(analyses)(cfg, Y_var, m_name, verbose=False) \
                for Y_var, cfg, m_name in product(Y_vars, configs, model_names)
            )
    
    csv_fpath = f"{ROOT}/results/H2_model_imputation.csv"

    with open(csv_fpath, "w") as fout:
        fieldnames = list(all_res[0].keys())
        writer = csv.DictWriter(fout, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_res)

    print("Done!")


if __name__ == "__main__":
    main()

# END
