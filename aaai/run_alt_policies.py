from core import RunConfig
import numpy as np
from constants import ON_POLICY_FILE
from joblib import Parallel, delayed
import time
import utils

'''
Run the off-policies for analysis.
'''

def sample_run(rc, force=False):
    if not rc.file_exists() or force:
        rc.load_data()
        rc.match()
        rc.save()
    return

def sample(n_jobs, configs, force=False):
    if n_jobs > 1:
        parallel = Parallel(n_jobs=n_jobs, verbose=10)
        parallel(
            delayed(sample_run)(rc, force) \
                for rc in configs
            )

    else:
        for rc in configs:
            sample_run(rc, force)

if __name__ == '__main__':
    args = utils.parse_arguments()
    stage = args.stage
    param_names = args.xparam
    n_jobs = args.n_jobs

    print(stage, n_jobs, param_names)

    configs = []
    for x_param in param_names + [None]:
        for param in utils.get_experiment_params(x_param, stage):
            rc = RunConfig(stage, param)
            configs.append(rc)

    sample(n_jobs, configs)
