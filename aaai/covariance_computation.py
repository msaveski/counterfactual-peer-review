
import time
import pathlib
import numpy as np
from tqdm import tqdm

from constants import *
from constants import COVAR_DIR

'''
Compute covariance of on-policy F from saved samples.
'''

def compute_covariances(F, samples_dir, target_n_samples):
    p = pathlib.Path(samples_dir)
    As_list = []
    file_count = 0
    stime = time.time()
    for samples_file in tqdm(p.iterdir()):
        As = np.load(str(samples_file))['arr_0']
        assert np.all(np.isclose(np.sum(As, axis=1), F.sum()))
        As_list.append(As)
        file_count += 1
        if target_n_samples is not None and file_count == int(target_n_samples / 1000):
            break
    rtime = time.time() - stime
    print(f"{file_count} files loaded, time {rtime}")

    stime = time.time()
    A_flat_all = np.concatenate(As_list, axis=0)
    del As_list
    rtime = time.time() - stime
    print(f"A_flat_all.shape={A_flat_all.shape}, concat time {rtime}")

    n_samples, _ = A_flat_all.shape 
    assert n_samples == target_n_samples

    # rowvar=False => each col represents a variable
    stime = time.time()
    A_cov = np.cov(A_flat_all, rowvar=False, bias=True)        
    rtime = time.time() - stime
    print(f"|A_cov| = {A_cov.shape}, covar time {rtime}")

    return A_cov, n_samples


if __name__ == '__main__':
    stage = 0
    print(f'Stage {stage}')
    F = np.load(ON_POLICY_FILE[stage])['F']
    target_n_samples = 1000000
    if stage == 0:
        samples_dir = COVAR_DIR + "samples/"
    elif stage == 1:
        samples_dir = COVAR_DIR + "samples_s2/"
    A_cov, n_samples = compute_covariances(F, samples_dir, target_n_samples)
    if stage == 0:
        out_file = COVAR_DIR + f"cov{n_samples}.npz"
    elif stage == 1:
        out_file = COVAR_DIR + f"cov{n_samples}_s2.npz"
    np.savez_compressed(out_file, Cov=A_cov, n_samples=n_samples)
