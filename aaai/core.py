
import os
import time
import pickle
import numpy as np
import gurobipy as gp

from constants import *


'''
Module containing parameters for alternate policies and functions for matching and computing similarities.
'''

def construct_pv_penalty(epsilon, F_onpolicy):
    return epsilon * (F_onpolicy == 0)

def fast_match(S, M, Q, rev_loads, pap_loads, verbose=False):
    if verbose:
        stime = time.time()
        print('Constructing LP')

    m = gp.Model()
    m.setParam('OutputFlag', 0)
    m.setParam('Method', 1)
    F = m.addMVar(S.shape, lb=0, ub=(1-M)*Q, obj=S)
    m.modelSense = gp.GRB.MAXIMIZE
    m.addConstrs(F[p, :] @ np.ones(S.shape[1]) == pap_loads[p] for p in range(S.shape[0]))
    m.addConstrs(np.ones(S.shape[0]) @ F[:, r] <= rev_loads[r] for r in range(S.shape[1]))

    if verbose:
        print('Done constructing', time.time() - stime)
        stime = time.time()
        print('Solving LP')

    m.optimize()
    if m.status != gp.GRB.OPTIMAL:
        print("Model not solved")
        raise RuntimeError('unsolved')

    if verbose:
        print('Done solving', time.time() - stime)
        stime = time.time()
        print('Outputting LP')   

    F_ = F.x

    if verbose:
        print('Done outputting', time.time() - stime)
    return F_

def match(S, Q, constraints):
    M = constraints['M']
    rev_loads = constraints['rev_loads']
    pap_loads = constraints['pap_loads']
    return fast_match(S, M, Q, rev_loads, pap_loads)

def translate_bids(B, vec):
    exponents = np.zeros_like(B)
    bid_values = [0.05, 1, 2, 4, 6] # values in saved bid matrix
    for b, e in zip(bid_values, vec):
        exponents[B == b] = e
    return exponents

def construct_lambda_bids(lam, default_bids):
    new_bids = [((x-1)*lam) + 1 for x in default_bids]
    new_bids[0] = 0.05
    assert new_bids[1] == 1
    return new_bids

def calc_aaai_simplified(k, tpms, nacl, B, text_weight, bid_weights, acl_weight):
    # simplified  version of aaai formula
    
    text = ((1-acl_weight) * tpms) + (acl_weight * nacl)
    text[np.isnan(tpms)] = nacl[np.isnan(tpms)]
    text[np.isnan(nacl)] = tpms[np.isnan(nacl)]

    base = (text_weight * text) + ((1 - text_weight) * k)
    base[np.isnan(text)] = k[np.isnan(text)]
    base[np.isnan(k)] = text[np.isnan(k)]
    base[np.isnan(base)] = 0

    E = translate_bids(B, bid_weights)
    S = base ** (1 / E)
    return S

def calc_aaai_phase1(k, tpms, nacl, B, text_weight, bid_weights, acl_weight, rev_info):
    text = ((1-acl_weight) * tpms) + (acl_weight * nacl)
    text[np.isnan(tpms)] = nacl[np.isnan(tpms)]
    text[np.isnan(nacl)] = tpms[np.isnan(nacl)]

    base = (text_weight * text) + ((1 - text_weight) * k)
    base[np.isnan(text)] = k[np.isnan(text)]
    base[np.isnan(k)] = text[np.isnan(k)]
    base[np.isnan(base)] = 0

    condn = (B >= 4) & (k < 0.0000001)
    base[condn] = text[condn]

    E = translate_bids(B, bid_weights)
    S = base ** (1.0 / E)

    lower_thr = 0.15
    cond_low = (S <= lower_thr) & (~np.isnan(k))
    S[cond_low] = np.clip(k[cond_low] ** (1.0/E[cond_low]), a_min=0, a_max=lower_thr)

    for rid, info in enumerate(rev_info):
        if info['missing_DBLP']:
            S[:, rid] *= 0.9
    return S


def calc_aaai_phase2(k, tpms, B, text_weight, bid_weights, rev_info):
    text = tpms
    base = (text_weight * text) + ((1 - text_weight) * k)
    base[np.isnan(text)] = (0.9 * k[np.isnan(text)])
    base[np.isnan(k)] = text[np.isnan(k)]
    base[np.isnan(base)] = 0

    E = translate_bids(B, bid_weights)
    S = base ** (1.0 / E)

    for rid, info in enumerate(rev_info):
        if info['missing_DBLP']:
            S[:, rid] *= 0.9
    return S


class RunConfig:
    '''
    formula actual: actual phase1 formula
    formula simplified: use weights and simplified aaai formula
    formula original: use on-policy similarity matrix
    '''
    def __init__(self, stage, init_params={}, penalty_epsilons=[1e-9, 1e-6, 1e-3]):
        self.stage = stage # 0 or 1
        self.defaults = [{'text_weight' : 0.75, 
                'bid_weights' : [0.05, 1, 1.5, 2.5, 4],
                'acl_weight' : 0, 
                'Q' : 0.52,
                'formula' : 'actual'}, 
            {'text_weight' : 0.6,
                'bid_weights' : [0.05, 1, 1.5, 4, 6],
                'Q' : 0.52,
                'formula' : 'actual'}
            ][stage]
        self.penalty_epsilons = sorted(penalty_epsilons) # small to large

        self.params = {k : (init_params[k] if k in init_params else v) for k, v in self.defaults.items()}
        if 'bid_lambda' in init_params:
            self.params['bid_lambda'] = init_params['bid_lambda']
            self.params['bid_weights'] = construct_lambda_bids(init_params['bid_lambda'], self.defaults['bid_weights'])

        if 'conflicts' in init_params:
            assert init_params['conflicts'] in ('zero', 'unsupported')
            self.params['conflicts'] = init_params['conflicts']

    def load_data(self):
        with open(REVIEWER_PAPER_INFO[self.stage], 'rb') as f:
            self.rev_info, self.pap_info = pickle.load(f)
        self.data = np.load(SIMILARITY_FILE[self.stage])
        self.F_onpolicy = np.load(ON_POLICY_FILE[self.stage])['F']
        constraints_file = np.load(CONSTRAINTS_FILE[self.stage])
        self.constraints = {k : constraints_file[k] for k in ['M', 'rev_loads', 'pap_loads']}

        self.COI = self.constraints['M']
        if 'conflicts' in self.params:
            if self.params['conflicts'] == 'zero':
                self.COI = np.zeros_like(self.COI)
            elif self.params['conflicts'] == 'unsupported':
                self.COI = (self.F_onpolicy == 0).astype(self.COI.dtype)
            self.constraints['M'] = self.COI

        self.Q = self.params['Q']
        if self.params['formula'] == 'simplified' and self.stage == 0:
            self.S = calc_aaai_simplified(self.data['k'], self.data['tpms'], self.data['nacl'], self.data['B'], 
                    self.params['text_weight'], self.params['bid_weights'], self.params['acl_weight'])
        elif self.params['formula'] == 'actual':
            if self.stage == 0:
                self.S = calc_aaai_phase1(self.data['k'], self.data['tpms'], self.data['nacl'], self.data['B'], 
                        self.params['text_weight'], self.params['bid_weights'], self.params['acl_weight'], self.rev_info)
            elif self.stage == 1:
                self.S = calc_aaai_phase2(self.data['k'], self.data['tpms'], self.data['B'], 
                        self.params['text_weight'], self.params['bid_weights'], self.rev_info)
            if self.params == self.defaults:
                S_orig = np.load(SIMILARITY_FILE[self.stage])['S_orig']
                assert np.sum(self.S - S_orig >= 1e-5) == 0
        elif self.params['formula'] == 'original':
            self.S = np.load(SIMILARITY_FILE[self.stage])['S_orig']
        elif self.params['formula'] == 'negated':
            self.S = -1 * np.load(SIMILARITY_FILE[self.stage])['S_orig']
        else:
            assert False

    def save(self):
        fname = self.get_fname()
        assert fname != 'onpolicy', "Don't overwrite the original onpolicy file"
        np.savez_compressed(ALT_FOLDER[self.stage]+fname+'.npz', F=self.F, params=self.params, penalty_epsilons=self.penalty_epsilons, best_epsilon=self.best_epsilon, F_baseline=self.F_baseline)

    def load(self):
        fname = self.get_fname()
        if fname == 'onpolicy':
            data = np.load(ON_POLICY_FILE[self.stage])
        else:
            data = np.load(ALT_FOLDER[self.stage]+fname+'.npz')
        self.F = data['F']
        if 'F_baseline' in data:
            self.penalty_epsilons = data['penalty_epsilons']
            self.best_epsilon = data['best_epsilon']
            self.F_baseline = data['F_baseline']
        return data

    def match(self):
        def pv_mass(F_):
            return np.sum(F_[self.F_onpolicy == 0])

        self.F_baseline = match(self.S, self.Q, self.constraints)
        self.F = self.F_baseline
        self.best_epsilon = 0
        target_obj = np.sum(self.S * self.F_baseline)

        results = {}
        for i, epsilon in enumerate(self.penalty_epsilons):
            if epsilon <= 0:
                continue
            penalty = construct_pv_penalty(epsilon, self.F_onpolicy)
            F_ = match(self.S - penalty, self.Q, self.constraints)
            delta = target_obj - np.sum(self.S * F_)
            if delta > 1e-5:
                break
            else:
                self.best_epsilon = epsilon
                self.F = F_
        return self.F

    def get_fname(self):
        if ((self.params['formula'] == 'original' and self.params['Q'] == self.defaults['Q'])
            or self.params == self.defaults):
            return 'onpolicy'

        param_strs = []
        for param, val in self.params.items():
            if param in self.defaults and val == self.defaults[param]:
                continue
            if param in ("Q", "text_weight", "acl_weight", "bid_lambda"):
                val = "%.4f" % val
            elif param == 'bid_weights':
                if "bid_lambda" not in self.params:
                    val = '-'.join(["%.4f" % bw for bw in val])
                else:
                    continue # bid lambda is sufficient
            param_str = param + "_" + str(val).replace(".", "_")
            param_strs.append(param_str)
        name = "__".join(sorted(param_strs))
        if len(name) == 0:
            name = 'default'
        return name

    def file_exists(self):
        fname = self.get_fname()
        return (fname == 'onpolicy') or os.path.isfile(ALT_FOLDER[self.stage]+fname+'.npz')
