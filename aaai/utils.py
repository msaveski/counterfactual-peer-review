import argparse
import numpy as np
from constants import *

'''
Utilities used in multiple modules.
'''

def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('-y', '--outcome', required=True, type=str)
    parser.add_argument('-s', '--stage', default=0, type=int)
    parser.add_argument('-t', '--n_jobs', default=1, type=int)
    parser.add_argument('-m', '--use_monotonicity', action='store_true')
    parser.add_argument('-l', '--use_lipschitz', action='store_true')
    parser.add_argument('-x', '--xparam', action='append', required=True)

    args = parser.parse_args()
    assert args.stage == 0 or args.stage == 1 
    if len(args.xparam) == 1 and args.xparam[0] == 'all':
        args.xparam = ['text_weight', 'bid_lambda', 'Q']
    return args

def get_param_values(x_param):
    ws_map = {
        'text_weight' : [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 1],
        'bid_lambda' :  np.linspace(0, 3, num=13).tolist(), 
        'Q' : [0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]
    }
    return ws_map[x_param]

def get_experiment_params(x_param, stage=None):
    if x_param is None:
        return [{}] # onpolicy
    elif x_param == 'bad':
        assert stage in (0, 1)
        return [{'Q' : 1, 'conflicts' : 'unsupported', 'formula' : 'negated'},
                {'Q' : 1, 'conflicts' : 'unsupported', 'formula' : 'original'}
                ]

    param_list = []
    ws = get_param_values(x_param)
    for w in ws:
        if x_param == 'Q':
            param = {'formula' : 'original', 'Q' : w}  
        else:
            param = {'Q':1, x_param:w}
        param_list.append(param)
    return param_list

def get_lipschitz_Ls(stage, Y_var, use_monotonicity):
    Ls = []
    assert stage == 0 or stage == 1
    if Y_var == "confidence":
        Ls += [20, 70, 30]
    elif Y_var == "expertise":
        Ls += [30, 100, 40]
    if use_monotonicity:
        Ls += [None]
    return Ls

def get_y_min_max(stage, Y_var, outcome_data=None):
    if outcome_data is None:
        outcome_data = np.load(OUTCOME_MATRIX_FILE[stage])
    y_min = 0
    if Y_var == 'confidence':
        Y = outcome_data['Y_conf']
        y_max = 4
    elif Y_var =='expertise':
        Y = outcome_data['Y_exp']
        y_max = 5
    else:
        assert False
    return Y, y_min, y_max

def get_bids(stage, similarity_info=None):
    if similarity_info is None:
        similarity_info = np.load(SIMILARITY_FILE[stage])

    assert stage == 0 or stage == 1
    B_orig = similarity_info['B']
   
    # Fix bid values
    B = np.zeros_like(B_orig)
    count = 0
    for old, new in zip([0.05, 1, 2, 4, 6], [0, 1, 2, 3, 4]):
        mask = B_orig == old
        B[mask] = new
        count += np.sum(mask)
    assert count == B.size
    return B
 
def get_text_sim(stage, similarity_info=None):
    if similarity_info is None:
        similarity_info = np.load(SIMILARITY_FILE[stage])

    assert stage == 0 or stage == 1
    TPMS = similarity_info['tpms']
    TEXT = TPMS.copy()
    if stage == 0:
        NACL = similarity_info['nacl']
        TEXT[np.isnan(TPMS)] = NACL[np.isnan(TPMS)]
    return TEXT
 
def get_keyword_sim(stage, similarity_info=None):
    if similarity_info is None:
        similarity_info = np.load(SIMILARITY_FILE[stage])

    assert stage == 0 or stage == 1
    K = similarity_info['k']
    return K

# Lipschitz distance
def cdist_lip(X1, X2):
    D = np.zeros((X1.shape[0], X2.shape[0]))
    for i in range(X1.shape[1]):
        Di = np.abs(np.subtract.outer(X1[:, i], X2[:, i]))
        Di[np.isnan(Di)] = 1
        D += Di
    D /= X1.shape[1]
    return D

# norm X for lipschitz
def norm_zero_one(X_all):
    assert X_all.shape[1] <= 3
    for dim in range(X_all.shape[1]):
        Xi = X_all[:, dim]
        mask = ~np.isnan(Xi)
        a = np.min(Xi[mask])
        b = np.max(Xi[mask])
        X_all[:, dim] = (Xi - a) / (b - a)

