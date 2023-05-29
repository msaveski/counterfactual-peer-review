
import csv
import click
import itertools
import numpy as np
import pandas as pd
from IPython.utils import io
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.feature_selection import VarianceThreshold
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_absolute_error
from surprise import Dataset, Reader
from surprise.model_selection import GridSearchCV as SpGridSearchCV

import utils
from constants import SIMILARITY_FILE, MODEL_DIR
from model_prediction import clf_ord_configs, rec_configs

'''
Evaluate model performance via train-test splits.
'''

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


def run(stage, Y_var, standardize, bids_one_hot):
    n_runs = 10

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
    K = K.flatten()
    Y = Y.flatten()
    idxs_2d = idxs_2d.flatten()

    assert np.all((A == 0) | (A == 1))
    print(f'n_obs = {np.sum(A == 1)}, n_unobs = {np.sum(A == 0)}')

    # split into observed vs unobsered 
    Bids_unobs = Bids[A == 0]
    Bids_obs = Bids[A == 1]

    Aff_unobs = Aff[A == 0]
    Aff_obs = Aff[A == 1]

    K_unobs = K[A == 0]
    K_obs = K[A == 1]

    idxs_2d_unobs = idxs_2d[A == 0]
    idxs_2d_obs = idxs_2d[A == 1]

    # bids: 1-hot encoding
    if bids_one_hot:
        enc = OneHotEncoder(sparse=False)
        enc.fit(Bids.reshape(-1, 1))
        Bids_obs = enc.transform(Bids_obs.reshape(-1, 1))
        Bids_unobs = enc.transform(Bids_unobs.reshape(-1, 1))

    X = np.column_stack((Bids_obs, Aff_obs, K_obs))
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

        # train reg/clf/ord models
        for config in clf_ord_configs:
            model_name, model, is_1hot, is_std = config['model_name'], config['model'], config['bids_one_hot'], config['standardize']
            if is_1hot != bids_one_hot or is_std != standardize:
                continue

            print(f">> {Y_var} __ {run_num} __ {model_name}")

            model.fit(X_train.copy(), Y_train.copy())

            Y_train_hat = model.predict(X_train.copy())
            Y_test_hat = model.predict(X_test.copy())

            df_res_rows.append({
                "outcome": Y_var,
                "run_num": run_num,
                "name": model_name,
                "train_mae": mean_absolute_error(Y_train, Y_train_hat),
                "test_mae": mean_absolute_error(Y_test, Y_test_hat),
                "params": model.best_params_ if isinstance(model, GridSearchCV) else None,
                "bids_one_hot" : bids_one_hot,
                "standardize" : standardize
            })

        # train rec models
        if not standardize and not bids_one_hot:
            for config in rec_configs:
                model_name, model = config['model_name'], config['model']
                print(f">> {Y_var} __ {run_num} __ {model_name}")
    
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
                    "name": model_name,
                    "train_mae": mean_absolute_error(Y_train, Y_train_hat),
                    "test_mae": mean_absolute_error(Y_test, Y_test_hat),
                    "params": model.best_params['mae'] if isinstance(model, SpGridSearchCV) else None
                })
    
    # make a dataframe and output to file
    df = pd.DataFrame(df_res_rows)
    return df
    

@click.command()
@click.option('--stage', required=True, type=int, multiple=True)
@click.option('--outcome', required=True, type=str, multiple=True)
def main(stage, outcome):

    print(stage, outcome)
    
    for s, Y_var in itertools.product(stage, outcome):
        assert s in (0, 1)
        assert Y_var in ("expertise", "confidence")
        print(s, Y_var)
            
        out_fname = Y_var + str(s)
        out_fpath = f"{MODEL_DIR}/eval/{out_fname}.csv"    
    
        dfs = []
        for standardize in [False, True]:
            for bids_one_hot in [False, True]:
                df = run(s, Y_var, standardize, bids_one_hot)
                dfs.append(df)
        df = pd.concat(dfs)
        df.to_csv(out_fpath, index=False, quoting=csv.QUOTE_NONNUMERIC)
        print('Done')


if __name__ == "__main__":
    main()
