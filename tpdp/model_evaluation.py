
import csv
import click
import numpy as np
import pandas as pd

from core import ROOT, RunConfig, DataLoader

from IPython.utils import io

from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.feature_selection import VarianceThreshold
from sklearn.dummy import DummyRegressor
from sklearn.linear_model import LinearRegression, RidgeCV, ARDRegression
from sklearn.linear_model import LogisticRegression, RidgeClassifierCV
from sklearn.svm import SVR, SVC
from sklearn.neighbors import KNeighborsRegressor, KNeighborsClassifier
from sklearn.ensemble import AdaBoostRegressor, GradientBoostingRegressor, GradientBoostingClassifier
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.model_selection import train_test_split, RepeatedStratifiedKFold, GridSearchCV
from sklearn.metrics import mean_absolute_error

from statsmodels.miscmodels.ordinal_model import OrderedModel
from mord import LogisticAT, LogisticIT

from surprise import Dataset, Reader
from surprise.model_selection import GridSearchCV as SpGridSearchCV
from surprise.model_selection.split import RepeatedKFold as SpRepeatedKFold
from surprise import BaselineOnly, SlopeOne, CoClustering
from surprise import KNNBasic, KNNBaseline, KNNWithMeans, KNNWithZScore
from surprise import SVD, SVDpp, NMF



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
    
    
def make_recs_datasets(idxs_2d_train_pairs, Y_train, idxs_2d_test_pairs, Y_test,
        Y_min, Y_max):
    
    # make train and test dataframes
    df_train_rows = []

    for ij, Y_ij in zip(idxs_2d_train_pairs, Y_train):
        df_train_rows.append({
            "reviewer": ij[0],
            "paper": ij[1],
            "Y": int(Y_ij)
        })

    df_test_rows = []

    for ij, Y_ij in zip(idxs_2d_test_pairs, Y_test):
        df_test_rows.append({
            "reviewer": ij[0],
            "paper": ij[1],
            "Y": int(Y_ij)
        })

    df_train = pd.DataFrame(df_train_rows)
    df_test = pd.DataFrame(df_test_rows)
    df_all = pd.DataFrame(df_train_rows + df_test_rows)

    # make a dataset
    reader = Reader(rating_scale=(Y_min, Y_max))

    # NB: must be ordered as: user id, item id and ratings
    ds_train = Dataset.load_from_df(df_train[["reviewer", "paper", "Y"]], reader)
    ds_test = Dataset.load_from_df(df_test[["reviewer", "paper", "Y"]], reader)
    ds_all = Dataset.load_from_df(df_all[["reviewer", "paper", "Y"]], reader)
    
    return ds_train, ds_test, ds_all


def make_models():
    
    reg_models = [
        ("dummy-mean", DummyRegressor(strategy="mean")),
        ("linear-reg", LinearRegression()),
        ("rigde-reg", RidgeCV(
            alphas=np.arange(0.1, 10, 0.25), 
            scoring="neg_mean_absolute_error")),
        ("SVR", GridSearchCV(
            estimator=SVR(), 
            param_grid={
                "kernel": ["poly", "rbf", "sigmoid"], 
                "C": np.arange(0.01, 2, 0.05)}, 
            cv=RepeatedStratifiedKFold(n_splits=10, n_repeats=10, random_state=0),
            scoring="neg_mean_absolute_error")),    
        ("gaussian-process-reg", GaussianProcessRegressor()),
        ("knn-reg", GridSearchCV(
            estimator=KNeighborsRegressor(), 
            param_grid={"n_neighbors": [1, 3, 5, 7, 10, 15, 20, 25, 30]}, 
            cv=RepeatedStratifiedKFold(n_splits=10, n_repeats=10, random_state=0),
            scoring="neg_mean_absolute_error")),
        ("adaboost-reg", GridSearchCV(
            estimator=AdaBoostRegressor(random_state=0), 
            param_grid={"n_estimators": [3, 5, 10, 15, 20, 25, 30, 50, 75, 100]}, 
            cv=RepeatedStratifiedKFold(n_splits=10, n_repeats=10, random_state=0),
            scoring="neg_mean_absolute_error")),
        ("gradient-boosting-reg", GridSearchCV(
            estimator=GradientBoostingRegressor(random_state=0), 
            param_grid={"n_estimators": [3, 5, 10, 15, 20, 25, 30, 50, 75, 100]}, 
            cv=RepeatedStratifiedKFold(n_splits=10, n_repeats=10, random_state=0),
            scoring="neg_mean_absolute_error")),
        ("ard-reg", ARDRegression())
    ]

    clf_models = [
        ("dummy-median", DummyRegressor(strategy="median")),
        ("logistic-reg", LogisticRegression()),
        ("rigdge-clf", RidgeClassifierCV(
            alphas=np.arange(0.1, 10, 0.25), 
            store_cv_values=True, 
            scoring="neg_mean_absolute_error")),
        ("svc", GridSearchCV(
            estimator=SVC(), 
            param_grid={
                "kernel": ["poly", "rbf", "sigmoid"], 
                "C": np.arange(0.01, 2, 0.05)}, 
            cv=RepeatedStratifiedKFold(n_splits=10, n_repeats=10, random_state=0),
            scoring="neg_mean_absolute_error")),
        ("knn-clf", GridSearchCV(
            estimator=KNeighborsClassifier(), 
            param_grid={"n_neighbors": [1, 3, 5, 7, 10, 15, 20, 25, 30, 40, 50]}, 
            cv=RepeatedStratifiedKFold(n_splits=10, n_repeats=10, random_state=0),
            scoring="neg_mean_absolute_error")),
        ("gradient-boosting-clf", GridSearchCV(
            estimator=GradientBoostingClassifier(random_state=0), 
            param_grid={"n_estimators": [3, 5, 10, 15, 20, 25, 30, 50, 75, 100]}, 
            cv=RepeatedStratifiedKFold(n_splits=10, n_repeats=10, random_state=0),
            scoring="neg_mean_absolute_error"))
    ]

    ord_models = [
        ("ord-logit", OrderedModelWrapper(distr="logit")),
        ("ord-probit", OrderedModelWrapper(distr="probit")),
        ("ord-logistic-at", LogisticAT()),
        ("ord-logistic-it", LogisticIT())
    ]

    reg_models = [("reg", name, model) for name, model in reg_models]
    clf_models = [("clf", name, model) for name, model in clf_models]
    ord_models = [("ord", name, model) for name, model in ord_models]

    reg_clf_ord_models = reg_models + clf_models + ord_models
    
    
    rec_models = [
        ("baseline-only", BaselineOnly()),
        ("rec-knn-basic", SpGridSearchCV(
            KNNBasic, 
            {"k": [3, 4, 5, 7]}, 
            cv=SpRepeatedKFold(n_splits=10, n_repeats=10, random_state=0), 
            measures=["mae"], 
            refit=True)),
        ("rec-knn-baseline", SpGridSearchCV(
            KNNBaseline, 
            {"k": [2, 3, 4, 5, 7]}, 
            cv=SpRepeatedKFold(n_splits=10, n_repeats=10, random_state=0), 
            measures=["mae"], 
            refit=True)),
        ("rec-knn-with-means", SpGridSearchCV(
            KNNWithMeans, 
            {"k": [2, 3, 4, 5, 7]}, 
            cv=SpRepeatedKFold(n_splits=10, n_repeats=10, random_state=0), 
            measures=["mae"], 
            refit=True)),
        ("rec-knn-with-z-score", SpGridSearchCV(
            KNNWithZScore, 
            {"k": [2, 3, 4, 5, 7]}, 
            cv=SpRepeatedKFold(n_splits=10, n_repeats=10, random_state=0), 
            measures=["mae"], 
            refit=True)),
        ("rec-svd", SpGridSearchCV(
            SVD, 
            {"n_factors": [2, 3, 4, 5, 7, 10, 15, 20]}, 
            cv=SpRepeatedKFold(n_splits=10, n_repeats=10, random_state=0), 
            measures=["mae"], 
            refit=True)),
        ("rec-svd++", SpGridSearchCV(
            SVDpp, 
            {"n_factors": [2, 3, 4, 5, 7, 10, 15, 20]}, 
            cv=SpRepeatedKFold(n_splits=10, n_repeats=10, random_state=0), 
            measures=["mae"], 
            refit=True)),
        ("rec-nmf", SpGridSearchCV(
            NMF, 
            {"n_factors": [2, 3, 4, 5, 7, 10, 15, 20, 30, 40, 50, 75, 100]}, 
            cv=SpRepeatedKFold(n_splits=10, n_repeats=10, random_state=0), 
            measures=["mae"], 
            refit=True)),
        ("rec-slope-one", SlopeOne()),
        ("rec-co-clustering", SpGridSearchCV(
            CoClustering, 
            {"n_cltr_u": [2, 3, 4, 5], "n_cltr_i": [2, 3, 4, 5]}, 
            cv=SpRepeatedKFold(n_splits=10, n_repeats=10, random_state=0), 
            measures=["mae"], 
            refit=True))
    ]

    rec_models = [("rec", name, model) for name, model in rec_models]
    
    return reg_clf_ord_models, rec_models



@click.command()
@click.option('--outcome', required=True, type=str)
@click.option('--standardize', '-std', is_flag=True)
@click.option('--bids_one_hot', '-1hot', is_flag=True)
def main(outcome, standardize, bids_one_hot):
    
    print(outcome, standardize, bids_one_hot)
    
    assert outcome in ("expertise", "confidence")
    Y_var = outcome
        
    n_runs = 10

    #
    # Load data
    #
    dl = DataLoader()

    A = dl.get_assignment_matrix()
    Bids = dl.get_bids_matrix(RunConfig()) # default bid values
    Aff = dl.get_affinity_matrix()
    Y = dl.get_outcome_matrix(Y_var)
    Y = Y.astype(int) # required by mord, ok with all other
    Y_min, Y_max = dl.get_outcome_bounds(Y_var)
    idxs_2d = np.array([[f"{i}_{j}" \
         for j in range(Y.shape[1])] for i in range(Y.shape[0])])

    # flatten
    A = A.flatten()
    Bids = Bids.flatten()
    Aff = Aff.flatten()
    Y = Y.flatten()
    idxs_2d = idxs_2d.flatten()

    # split into observed vs unobsered 
    Bids_unobs = Bids[A == 0]
    Bids_obs = Bids[A == 1]

    Aff_unobs = Aff[A == 0]
    Aff_obs = Aff[A == 1]

    idxs_2d_unobs = idxs_2d[A == 0]
    idxs_2d_obs = idxs_2d[A == 1]

    # bids: 1-hot encoding
    if bids_one_hot:
        enc = OneHotEncoder(sparse=False)
        enc.fit(Bids.reshape(-1, 1))
        Bids_obs = enc.transform(Bids_obs.reshape(-1, 1))
        Bids_unobs = enc.transform(Bids_unobs.reshape(-1, 1))

    X = np.column_stack((Bids_obs, Aff_obs))
    Y = Y[A == 1]

    # main loop
    df_res_rows = []

    for run_num in range(n_runs):

        # train / test split
        X_train, X_test, Y_train, Y_test, idxs_2d_train, idxs_2d_test = \
            train_test_split(X.copy(), Y.copy(), idxs_2d_obs.copy(), 
                             test_size=0.25, random_state=run_num, stratify=Y)

        # rec models prep: have to be ordered as: user, item, rating
        idxs_2d_train_pairs = [(i.split("_")[1], i.split("_")[0]) for i in idxs_2d_train]
        idxs_2d_test_pairs = [(i.split("_")[1], i.split("_")[0]) for i in idxs_2d_test]

        ds_train, ds_test, ds_all = make_recs_datasets(idxs_2d_train_pairs, Y_train, 
            idxs_2d_test_pairs, Y_test, Y_min, Y_max)

        # standardize
        if standardize:
            scaler = StandardScaler()
            scaler.fit(X_train)
            X_train = scaler.transform(X_train)
            X_test = scaler.transform(X_test)

        # remove constant features: happens when using 1-hot encoding
        enc = VarianceThreshold(threshold=0)
        enc.fit(X_train)
        X_train = enc.transform(X_train)
        X_test = enc.transform(X_test)    

        #
        # Make models
        #
        reg_clf_ord_models, rec_models = make_models()

        #
        # Train ALL
        #

        # train reg/clf/ord models
        for model_type, model_name, model in reg_clf_ord_models:
            print(f">> {Y_var} __ {run_num} __ {model_type} __ {model_name}")

            model.fit(X_train.copy(), Y_train.copy())

            Y_train_hat = model.predict(X_train.copy())
            Y_test_hat = model.predict(X_test.copy())

            df_res_rows.append({
                "outcome": Y_var,
                "run_num": run_num,
                "type": model_type,
                "name": model_name,
                "train_mae": mean_absolute_error(Y_train, Y_train_hat),
                "test_mae": mean_absolute_error(Y_test, Y_test_hat),
                "params": model.best_params_ if isinstance(model, GridSearchCV) else None
            })

        # train rec models
        for model_type, model_name, model in rec_models:
            print(f">> {Y_var} __ {run_num} __ {model_type} __ {model_name}")

            with io.capture_output():
                if isinstance(model, SpGridSearchCV):
                    model.fit(ds_train)
                else:
                    model.fit(ds_train.build_full_trainset())

            Y_train_hat = np.array([model.predict(*i).est for i in idxs_2d_train_pairs])
            Y_test_hat = np.array([model.predict(*i).est for i in idxs_2d_test_pairs])

            df_res_rows.append({
                "outcome": Y_var,
                "run_num": run_num,
                "type": model_type,
                "name": model_name,
                "train_mae": mean_absolute_error(Y_train, Y_train_hat),
                "test_mae": mean_absolute_error(Y_test, Y_test_hat),
                "params": model.best_params['mae'] if isinstance(model, SpGridSearchCV) else None
            })
    
    # make a dataframe and output to file
    df = pd.DataFrame(df_res_rows)
    
    out_fname = Y_var
    
    if bids_one_hot: 
        out_fname += "_1hot"
    if standardize: 
        out_fname += "_stded"

    out_fpath = f"{ROOT}/model_imputation/cv/{out_fname}.csv"    

    df.to_csv(out_fpath, index=False, quoting=csv.QUOTE_NONNUMERIC)
    
    print("Done!")
    

if __name__ == "__main__":
    main()

# END