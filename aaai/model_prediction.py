
import csv
import click
import numpy as np
import pandas as pd

from IPython.utils import io
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.feature_selection import VarianceThreshold
from sklearn.linear_model import LogisticRegression, RidgeClassifierCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import RepeatedStratifiedKFold, GridSearchCV
from sklearn.metrics import mean_absolute_error
from statsmodels.miscmodels.ordinal_model import OrderedModel
from mord import LogisticAT, LogisticIT
from surprise import Dataset, Reader
from surprise.model_selection import GridSearchCV as SpGridSearchCV
from surprise.model_selection.split import RepeatedKFold as SpRepeatedKFold
from surprise import SVDpp, KNNWithMeans, KNNWithZScore

import utils
from constants import SIMILARITY_FILE, MODEL_DIR

'''
Save model predictions for use in imputation.
'''

class OrderedModelWrapper():
    def __init__(self, distr="probit"):
        self.distr = distr
        self.trim_X = False        

    def fit(self, X, Y):
        X_ = X.copy()
        
        # hacky way to avoid implicit intercept when using 1-hot encoding
        if X_.shape[1] > 2 and len(np.unique(X_[:, 0])) <= 2:
            self.trim_X = True
            X_ = X_[:, 1:]
        
        self.cat_vals = np.sort(np.unique(Y))
        self.model_ = OrderedModel(Y, X_, distr=self.distr)
        self.fit_ = self.model_.fit(method='bfgs', disp=False)
        return self
    
    def predict(self, X):
        X_ = X.copy()
        
        if self.trim_X:
            X_ = X_[:, 1:]
            
        pred_probs = self.fit_.model.predict(self.fit_.params, exog=X_)
        pred_probs_argmax = np.argmax(pred_probs, 1)
        preds = self.cat_vals[pred_probs_argmax]
        return preds
    

clf_ord_configs = [
    # -- clf
    {
        "model_name": "logistic-reg",
        "model": LogisticRegression(),
        "bids_one_hot": False,
        "standardize": False
    },
    {
        "model_name": "logistic-reg-stded",
        "model": LogisticRegression(),
        "bids_one_hot": False,
        "standardize": True
    },
    {
        "model_name": "knn-clf",
        "model": GridSearchCV(
            estimator=KNeighborsClassifier(), 
            param_grid={"n_neighbors": [1, 3, 5, 7, 10, 15, 20, 25, 30, 40, 50]}, 
            cv=RepeatedStratifiedKFold(n_splits=10, n_repeats=10, random_state=0),
            scoring="neg_mean_absolute_error"),
        "bids_one_hot": False,
        "standardize": True
    },
    {
        "model_name": "rigdge-clf",
        "model": RidgeClassifierCV(
            alphas=np.arange(0.1, 10, 0.25), 
            store_cv_values=True, 
            scoring="neg_mean_absolute_error"),
        "bids_one_hot": True,
        "standardize": True
    },
    # -- ord
    {
        "model_name": "ord-logit",
        "model": OrderedModelWrapper(distr="logit"),
        "bids_one_hot": True,
        "standardize": True
    },
    {
        "model_name": "ord-probit",
        "model": OrderedModelWrapper(distr="probit"),
        "bids_one_hot": True,
        "standardize": True
    },
    {
        "model_name": "ord-logistic-at",
        "model": LogisticAT(),
        "bids_one_hot": False,
        "standardize": True
    },
    {
        "model_name": "ord-logistic-it",
        "model": LogisticIT(),
        "bids_one_hot": False,
        "standardize": True
    }
]

rec_configs = [
    {
        "model_name": "rec-svdpp",
        "model": SpGridSearchCV(
            SVDpp, 
            {"n_factors": [2, 3, 4, 5, 7, 10, 15, 20], "random_state": [0]}, 
            cv=SpRepeatedKFold(n_splits=10, n_repeats=10, random_state=0), 
            measures=["mae"], 
            refit=True
        )
    },
    {
        "model_name": "rec-knn-with-means",
        "model": SpGridSearchCV(
            KNNWithMeans, 
            {"k": [2, 3, 4, 5, 7]}, 
            cv=SpRepeatedKFold(n_splits=10, n_repeats=10, random_state=0), 
            measures=["mae"], 
            refit=True,    
        )
    },
    {
        "model_name": "rec-knn-with-z-score",
        "model": SpGridSearchCV(
            KNNWithZScore, 
            {"k": [2, 3, 4, 5, 7]}, 
            cv=SpRepeatedKFold(n_splits=10, n_repeats=10, random_state=0), 
            measures=["mae"], 
            refit=True
        )
    }
]



def clf_ord_fit_predict(stage, Y_var, model_name, model, bids_one_hot, standardize):
    #
    # Load data
    #
    Y, Y_min, Y_max = utils.get_y_min_max(stage, Y_var)
    A = (~np.isnan(Y)).astype(int) # includes new reviewers, do before casting Y to int
    Y = Y.astype(int)
    similarity_info = np.load(SIMILARITY_FILE[stage])
    Bids = utils.get_bids(stage, similarity_info)
    Aff = utils.get_text_sim(stage, similarity_info)
    K = utils.get_keyword_sim(stage, similarity_info)

    Aff[np.isnan(Aff)] = 0
    K[np.isnan(K)] = 0


    idxs_2d = np.array([[f"{i}_{j}" \
         for j in range(Y.shape[1])] for i in range(Y.shape[0])])

    # flatten
    A = A.flatten()
    Bids = Bids.flatten()
    Aff = Aff.flatten()
    K = K.flatten() #
    Y = Y.flatten()
    idxs_2d = idxs_2d.flatten()

    # fetch observed
    Bids_obs = Bids[A == 1]
    Aff_obs = Aff[A == 1]
    K_obs = K[A == 1]
    idxs_2d_obs = idxs_2d[A == 1]

    # bids: 1-hot encoding
    if bids_one_hot:
        enc_1hot = OneHotEncoder(sparse=False)
        enc_1hot.fit(Bids.reshape(-1, 1))
        Bids_obs = enc_1hot.transform(Bids_obs.reshape(-1, 1))
        Bids = enc_1hot.transform(Bids.reshape(-1, 1))

    X = np.column_stack((Bids, Aff, K))
    X_obs = np.column_stack((Bids_obs, Aff_obs, K_obs))
    Y_obs = Y[A == 1]

    # standardize
    if standardize:
        enc_scaler = StandardScaler()
        enc_scaler.fit(X_obs)
        X_obs = enc_scaler.transform(X_obs)
        X = enc_scaler.transform(X)

    if bids_one_hot:
        # remove constant features: happens when using 1-hot encoding
        enc_var = VarianceThreshold(threshold=0)
        enc_var.fit(X_obs)
        X_obs = enc_var.transform(X_obs)
        X = enc_var.transform(X)

    #
    # fit & predict
    #
    model.fit(X_obs, Y_obs)

    Y_obs_hat = model.predict(X_obs)
    Y_hat = model.predict(X)

    best_params = model.best_params_ if isinstance(model, GridSearchCV) else None
    mae_obs = mean_absolute_error(Y_obs, Y_obs_hat)

    print(f">> {Y_var}\t {model_name}\t mae_obs: {mae_obs:.4f}\t best_params: {best_params}")

    #
    # output
    #
    df_y_hat_rows = []

    for ij_str, y_hat_ij in zip(idxs_2d, Y_hat):
        i, j = ij_str.split("_")
        i, j = int(i), int(j)    
        df_y_hat_rows.append({
            "paper_idx": i, 
            "reviewer_idx": j, 
            "pred": y_hat_ij
        })

    df_y_hat = pd.DataFrame(df_y_hat_rows)

    df_y_hat.to_csv(
        f"{MODEL_DIR}/preds/{Y_var}_s{stage}_{model_name}.csv",
        index=False
    )



def rec_fit_predict(stage, Y_var, model_name, model):
    #
    # Load data
    #
    Y, Y_min, Y_max = utils.get_y_min_max(stage, Y_var)
    A = (~np.isnan(Y)).astype(int) # includes new reviewers, do before casting Y to int
    Y = Y.astype(int)
    idxs_2d = np.array([[f"{i}_{j}" \
         for j in range(Y.shape[1])] for i in range(Y.shape[0])])

    # flatten
    A = A.flatten()
    Y = Y.flatten()
    idxs_2d = idxs_2d.flatten()

    # fetch observed
    idxs_2d_obs = idxs_2d[A == 1]
    Y_obs = Y[A == 1]

    # originaly: (paper, reviewer) 
    # here: (reviewer, paper) 
    idxs_2d_obs_pairs = [(i.split("_")[1], i.split("_")[0]) for i in idxs_2d_obs]
    idxs_2d_pairs = [(i.split("_")[1], i.split("_")[0]) for i in idxs_2d]

    #
    # Make dataset
    #
    df_obs_rows = []

    for ij, Y_ij in zip(idxs_2d_obs_pairs, Y_obs):
        df_obs_rows.append({
            "reviewer": ij[0],
            "paper": ij[1],
            "Y": int(Y_ij)
        })

    df_obs = pd.DataFrame(df_obs_rows)

    reader = Reader(rating_scale=(Y_min, Y_max))

    ds_obs = Dataset.load_from_df(df_obs[["reviewer", "paper", "Y"]], reader)

    #
    # fit & predict
    #
    with io.capture_output():
        if isinstance(model, SpGridSearchCV):
            model.fit(ds_obs)
        else:
            model.fit(ds_obs.build_full_trainset())

    Y_obs_hat = np.array([model.predict(*i).est for i in idxs_2d_obs_pairs])

    best_params = model.best_params['mae'] if isinstance(model, SpGridSearchCV) else None

    mae_obs = mean_absolute_error(Y_obs, Y_obs_hat)

    print(f">> {Y_var}\t {model_name}\t mae_obs: {mae_obs:.4f}\t best_params: {best_params}")

    #
    # Output predictions
    #
    df_y_hat_rows = []

    for i, j in idxs_2d_pairs:
        y_hat_ij = model.predict(i, j).est
        # NB: idxs were switched
        df_y_hat_rows.append({
            "paper_idx": j, 
            "reviewer_idx": i, 
            "pred": y_hat_ij
        })

    df_y_hat = pd.DataFrame(df_y_hat_rows)

    df_y_hat.to_csv(
        f"{MODEL_DIR}/preds/{Y_var}_s{stage}_{model_name}.csv",
        index=False
    )


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
                                                        
    Y_preds = np.full((n_papers, n_reviewers), np.nan)

    for row in rows:
        Y_preds[row["paper_idx"], row["reviewer_idx"]] = row["pred"]

    assert np.all(~np.isnan(Y_preds))
    return Y_preds



@click.command()
@click.option('--stage', required=True, type=int)
@click.option('--outcome', required=True, type=str)
def main(stage, outcome):
    Y_vars = [outcome]
    print(stage, Y_vars)
    assert stage in (0, 1)
    assert outcome in ("expertise", "confidence")
    
    for cfg in clf_ord_configs:
        for Y_var in Y_vars:
            clf_ord_fit_predict(stage,
                Y_var, 
                cfg["model_name"],
                cfg["model"], 
                cfg["bids_one_hot"], 
                cfg["standardize"]
            )
    
    for cfg in rec_configs:
        for Y_var in Y_vars:
            rec_fit_predict(stage,
                Y_var, 
                cfg["model_name"],
                cfg["model"], 
            )
    
    
    model_names = [i["model_name"] for i in clf_ord_configs + rec_configs]
    
    for model_name in model_names:
        for Y_var in Y_vars:
            # read true Ys
            Y, Y_min, Y_max = utils.get_y_min_max(stage, Y_var)
            X_obs = (~np.isnan(Y)) # includes new reviewers, do before casting Y to int
            Y = Y.astype(int)
    
            # read pred Ys 
            preds_fpath = f"{MODEL_DIR}/preds/{Y_var}_s{stage}_{model_name}"
            Y_preds = read_predictions(preds_fpath + '.csv')
            assert Y_preds.shape == Y.shape
    
            # compute MAE
            mae = mean_absolute_error(Y[X_obs], Y_preds[X_obs])
            np.savez(preds_fpath + '.npz', Y_preds=Y_preds, mae=mae)
            print(f">> s{stage} {Y_var}\t {model_name}\t mae: {mae:.4f}")

if __name__ == "__main__":
    main()
