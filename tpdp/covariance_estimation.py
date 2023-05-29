
import gzip
import json
import pickle
import numpy as np 

from hashlib import md5
from pathlib import Path
from joblib import Parallel, delayed

from core import ROOT, RunConfig, DataLoader, Solver

EXPS_DIR_NAME = "exps"



def setup():
    # make exps dir
    exps_dir = Path(f"{ROOT}/{EXPS_DIR_NAME}")
    exps_dir.mkdir(exist_ok=True)

    # # output bid maps (for reference)
    # with (exps_dir / "_bid_maps.json").open("w") as fout:
    #     bid_maps = RunConfig.generate_bid_maps()
    #     bid_maps = list(enumerate(bid_maps))
    #     json.dump(bid_maps, fout, indent=2)

    # create run dirs, output config params and F
    run_configs = RunConfig.get_all()

    for config in run_configs:
        print(f">> {config.get_name()}")

        # make run dir
        run_dir = exps_dir / config.get_name()
        run_dir.mkdir(exist_ok=True)

        # output config params
        with (run_dir / "params.json").open("w") as fout:
            json.dump(config.params, fout, indent=2)
        
        # run solver
        dl = DataLoader()
        dl.compute_scores(config)
        solver = Solver(dl, Q=config.params["Q"])
        F = solver.solve()
        F_hash = md5(F.tobytes()).hexdigest()
        F_fraction_optimal = solver.solver.get_fraction_of_opt()

        # output F
        with (run_dir / "F.npy").open("wb") as fout:
            np.save(fout, F)
        
        # output F's hash
        with (run_dir / "F_hash.txt").open("w") as fout:
            fout.write(F_hash + "\n")
        
        # output F's fraction of the optimal solution
        with (run_dir / "F_fraction_of_optimal.txt").open("w") as fout:
            fout.write(str(F_fraction_optimal) + "\n")
        
        # # test outputs
        # with (run_dir / "F.npy").open("rb") as fin:
        #     F_ = np.load(fin)
        #     assert np.all(F == F_)

        # with (run_dir / "F_hash.txt").open("r") as fin:
        #     F_hash_ = fin.read()
        #     assert F_hash == F_hash_.strip()

        # with (run_dir / "F_fraction_of_optimal.txt").open("r") as fin:
        #     F_fraction_optimal_ = float(fin.read())
        #     assert F_fraction_optimal == F_fraction_optimal_


def sample_run(config, n_assignments):
    print(f">> {config.get_name()} | starting ...")

    run_dir = Path(f"{ROOT}/{EXPS_DIR_NAME}/{config.get_name()}")

    # load data
    dl = DataLoader()
    dl.compute_scores(config)

    # solve LP
    solver = Solver(dl, Q=config.params["Q"])
    solver.solve()
    
    # verify solution
    F = solver.solver.fractional_assignment_matrix
    F_hash = md5(F.tobytes()).hexdigest()

    with (run_dir / "F.npy").open("rb") as fout:
        F_ = np.load(fout)
        assert np.all(F == F_)

    with (run_dir / "F_hash.txt").open("r") as fout:
        F_hash_ = fout.read()
        assert F_hash == F_hash_.strip()

    # sample assignments
    As = []
    A_hash_counts = {}

    for i in range(n_assignments):
        if i % 10000 == 0:
            print(f">> {config.get_name()} | sample # {i}")

        solver.solver.sample_assignment()
        A = solver.solver.flow_matrix
        A_hash = md5(A.tobytes()).hexdigest()

        As.append(A)
        A_hash_counts[A_hash] = A_hash_counts.get(A_hash, 0) + 1

    # output assignments
    print(f">> {config.get_name()} | outputting assignments ...")
    with gzip.open(run_dir / "assignments.pkl.gz", "wb") as fout:
        pickle.dump(As, fout)                    

    # output hash counts
    A_hash_counts = list(A_hash_counts.items())
    A_hash_counts.sort(key=lambda i: i[1], reverse=True)

    with (run_dir / "assignments_hash_counts.txt").open("w") as fout:
        for A_hash, count in A_hash_counts:
            fout.write(f"{A_hash} {count}\n")

    print(f">> {config.get_name()} ... done!")


def sample(n_jobs, n_assignments):

    configs = RunConfig.get_all()

    if n_jobs > 1:
        # multiprocess
        parallel = Parallel(n_jobs=n_jobs, verbose=10)
        parallel(
            delayed(sample_run)(config, n_assignments) \
                for config in configs
            )

    else:
        # single process    
        for config in configs:
            sample_run(config, n_assignments)


def compute_covariances():

    configs = RunConfig.get_all()

    for config in configs:

        print(f">> {config.get_name()}")

        run_dir = Path(f"{ROOT}/{EXPS_DIR_NAME}/{config.get_name()}")

        # load F
        with (run_dir / "F.npy").open("rb") as fin:
            F = np.load(fin)

        F_flat_pos = F.flatten() > 0

        print(f"|F| = {F.shape} | sum(F_pos) = {np.sum(F_flat_pos)}")

        # load assignments
        with gzip.open(run_dir / "assignments.pkl.gz", "rb") as fin:
            As = pickle.load(fin)

        # stack assignments [two versions: 1) all (2) positive support only]
        A_flat_all = []
        A_flat_sub_all = []

        for A in As:
            A_flat = A.flatten()

            # sanity check: make sure that A_ij = 1 iff F_ij > 0
            assert np.sum(A_flat) == np.sum(A_flat[F_flat_pos])

            A_flat_all.append(A_flat)

            A_flat_sub = A_flat[F_flat_pos]
            A_flat_sub_all.append(A_flat_sub)

        # lists to numpy arrays
        A_flat_all = np.array(A_flat_all)
        A_flat_sub_all = np.array(A_flat_sub_all)

        print("|A_flat_all| = ", A_flat_all.shape, end=" | ")
        print("|A_flat_sub_all| = ", A_flat_sub_all.shape)

        # compute covariance (rowvar=False => each col represents a variable)
        A_cov = np.cov(A_flat_all, rowvar=False, bias=True)        
        A_cov_sub = np.cov(A_flat_sub_all, rowvar=False, bias=True)
        
        print(f"|A_cov| = {A_cov.shape} | |A_cov_sub| = {A_cov_sub.shape}")

        # save covariance
        print("Saving ...")
        with (run_dir / "covariance.npy").open("wb") as fout:
            np.save(fout, A_cov)

        with (run_dir / "covariance_pos.npy").open("wb") as fout:
            np.save(fout, A_cov_sub)

    print("Done!")


def main():

    # 1) setup: make dirs, save F, and the hash of F
    setup()

    # 2) sample assignments
    # sample(
    #     n_jobs=10, 
    #     n_assignments=1000000 # 1M
    # )
    # elapsed: 2619.8min finished

    # 3) compute covariances
    # compute_covariances()

    

if __name__ == "__main__":
    main()

# END