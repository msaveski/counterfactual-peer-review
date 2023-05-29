
import time
import numpy as np
from cffi import FFI
from joblib import Parallel, delayed

from constants import *
from bvn_extension import run_bvn

'''
Save samples used to compute the covariance of on-policy F.
'''

def run_samples(F, n_samples):
    (npap, nrev) = F.shape
    ffi = FFI()
    one = 100000
    intF = np.round(F * one).flatten().astype(np.int32).tolist()
    Sbuf = ffi.new("int[]", nrev * [1])
    total = npap*nrev

    As = np.zeros(shape=(n_samples, total), dtype=np.int8)
    for t in range(n_samples):
        Fbuf = ffi.new("int[]", intF)
        run_bvn(Fbuf, Sbuf, npap, nrev, one)
        As[t, :] = np.frombuffer(ffi.buffer(Fbuf, total*4), dtype=np.int32).astype(np.int8)
    return As

def sample_and_save(F, n_samples, suffix, stage):
    print(f'Run {suffix}')
    stime = time.time()
    As_full = run_samples(F, n_samples)
    supp = F.flatten() > 0
    As_supp = As_full[:, supp]
    if stage == 0:
        np.savez_compressed(f'{COVAR_DIR}samples/samples_{suffix}.npz', As_supp)
    elif stage == 1:
        np.savez_compressed(f'{COVAR_DIR}samples_s2/samples_{suffix}.npz', As_supp)
    else:
        assert False
    runtime = (time.time() - stime) / 60
    print(f'Run {suffix}: time {runtime}')
    return

def sample_parallel(F, n_assignments_per_run, n_runs, n_jobs, stage):
    if n_jobs > 1:
        parallel = Parallel(n_jobs=n_jobs, verbose=10, backend="threading")
        parallel(
            delayed(sample_and_save)(F, n_assignments_per_run, suffix, stage) \
                for suffix in range(n_runs)
               )
    else:
        for _ in range(n_runs):
            sample_and_save(F, n_assignments_per_run, time.time_ns(), stage)
    return

if __name__ == '__main__':
    stage = 0
    print(f'Stage {stage}')
    F = np.load(ON_POLICY_FILE[stage])['F']
    n_assignments_per_run = 1000
    n_jobs = 1
    n_runs = 100

    sample_parallel(F, n_assignments_per_run, n_runs, n_jobs, stage)
