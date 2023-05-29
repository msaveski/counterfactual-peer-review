
import csv
import numpy as np

# open-review matcher module
from matcher.encoder import Encoder
from matcher.solvers import RandomizedSolver

ROOT = "../data/tpdp"

#
# Config
#
class RunConfig:

    defaults = {
        "Q": 0.5,
        "aff_w": 0.5,
        "aff_type": "openreview",
        "bid_w": 0.5,
        "bid_map_idx": 0,
        "agg": "sum"
    }

    def __init__(self,
            Q=defaults["Q"], 
            aff_w=defaults["aff_w"], aff_type=defaults["aff_type"], 
            bid_w=defaults["bid_w"], bid_map_idx=defaults["bid_map_idx"], 
            agg=defaults["agg"]):
        
        bid_maps = RunConfig.generate_bid_maps()

        self.params = {
            "Q": Q,
            "aff_w": aff_w,
            "aff_type": aff_type,
            "bid_w": bid_w,
            "bid_map": bid_maps[bid_map_idx],
            "bid_map_idx": bid_map_idx,
            "agg": agg
        }

    def get_name(self):
        param_strs = []

        for param, val in self.params.items():
            # skip the bid map (dict)
            if param == "bid_map":
                continue

            # exclude defaults
            if val == self.defaults[param]:
                continue

            # 0-fill bid_map_idx
            if param == "bid_map_idx":
                val = str(val).zfill(3)
            elif param in ("Q", "aff_w", "bid_w"):
                val = "%.4f" % val

            param_str = param + "_" + str(val).replace(".", "_")
            param_strs.append(param_str)

        name = "__".join(param_strs)

        if len(name) == 0:
            name = "_onpolicy"

        return name

    def get_params(self):
        params_cp = self.params.copy()

        # flatten the bid map (e.g., "bid_map_very_low": -1)
        bid_map_ = {}

        for bid_name, bid_val in params_cp["bid_map"].items():
            bid_name = "bid_map_" + bid_name.lower().replace(" ", "_")
            bid_map_[bid_name] = bid_val

        # delete bid map dict
        del params_cp["bid_map"]

        params_cp.update(bid_map_)

        return params_cp

    @staticmethod
    def generate_bid_maps():
        bid_maps_all = []

        # generate the default one
        bid_maps_all.append({
            "Very Low": -1.0,
            "Low": -0.5,
            "Neutral": 0.0,
            "High": 0.5,
            "Very High": 1.0,
            "default": 0.0
        })

        return bid_maps_all

    @staticmethod
    def get_all(subset=None):
        cfgs = []
                
        #
        # H1: varying Q
        #
        Qs = np.concatenate((
            np.arange(0.2, 1.025, 0.025),
            1 / np.array(np.linspace(1.9, 1, num=10, endpoint=False))
        ))
        Qs = sorted(list(Qs))

        if subset is None or subset == "H1" or "H1" in subset:
            cfgs += [RunConfig(Q=round(Q, 4)) for Q in Qs]
        
        #
        # H2: varying w_aff agg + mul (no w_aff variation)
        #
        aff_Qs = [0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
        aff_ws = np.concatenate((
            np.arange(0.0, 0.025, 0.0025), 
            np.arange(0.025, 1.025, 0.025)
        ))

        if subset is None or subset == "H2" or "H2" in subset:
            # aggs = sum
            for Q in aff_Qs:
                for aff_w in aff_ws:
                    Q = round(Q, 4)
                    aff_w = round(aff_w, 4)
                    bid_w = round(1.0 - aff_w, 4)
                    cfg = RunConfig(Q=Q, aff_w=aff_w, bid_w=bid_w, agg="sum")
                    cfgs.append(cfg)
            
            # aggs = mul
            for Q in aff_Qs:
                cfgs.append(RunConfig(Q=round(Q, 4), agg="mul"))

        # remove duplicates
        cfgs_deduped = []
        cfg_names = set()
        for cfg in cfgs:
            if cfg.get_name() not in cfg_names:
                cfgs_deduped.append(cfg)
                cfg_names.add(cfg.get_name())

        cfgs = cfgs_deduped

        # sanity check
        all_names = set([cfg.get_name() for cfg in cfgs])
        assert len(all_names) == len(set(all_names))

        return cfgs_deduped


#
# Data loader
#
BIDS_MAP = {
    "Very Low": -1.0,
    "Low": -0.5,
    "Neutral": 0.0,
    "High": 0.5,
    "Very High": 1.0,
    "default": 0.0
}

RATING_MAP = {
    "1: Trivial or wrong": 1,
    "2: Strong rejection": 2,
    "3: Clear rejection": 3,
    "4: Ok but not good enough - rejection": 4,
    "5: Marginally below acceptance threshold": 5,
    "6: Marginally above acceptance threshold": 6,
    "7: Good paper, accept": 7,
    "8: Top 50% of accepted papers, clear accept": 8,
    "9: Top 15% of accepted papers, strong accept": 9,
    "10: Top 5% of accepted papers, seminal paper": 10,
}

CONFIDENCE_MAP = {
    "1: The reviewer's evaluation is an educated guess": 1,
    "2: The reviewer is willing to defend the evaluation, but it is quite likely that the reviewer did not understand central parts of the paper": 2,
    "3: The reviewer is fairly confident that the evaluation is correct": 3,
    "4: The reviewer is confident but not absolutely certain that the evaluation is correct": 4,
    "5: The reviewer is absolutely certain that the evaluation is correct and very familiar with the relevant literature": 5
}

EXPERTISE_MAP = {
    "Irrelevant: This paper has little or no connection to my work.": 1,
    "Slightly Relevant: This paper has little connection to my work and overlaps only marginally with my area of expertise.": 2,
    "Relevant: This paper significantly overlaps with my work.": 3,
    "Very Relevant: This paper is within my current core research focus. I'm actively working in all areas of the paper.": 4
}


class DataLoader:

    raw_data_dir = f"{ROOT}/_raw/csv"

    outcome_names = [
        "confidence",
        "expertise",
    ]

    def __init__(self):

        # affinity
        self.affinity_triples = []

        with open(f"{self.raw_data_dir}/affinity.csv") as fin:
            for row in csv.DictReader(fin):
                self.affinity_triples.append((
                    row["paper_id_anon"],
                    row["reviewer_id_anon"],
                    float(row["weight"])
                ))
        
        # bids
        self.bid_triples = []

        with open(f"{self.raw_data_dir}/bids.csv") as fin:
            for row in csv.DictReader(fin):
                self.bid_triples.append((
                    row["paper_id_anon"],
                    row["reviewer_id_anon"],
                    row["label"]
                ))
            
        # conflicts
        self.conflict_triples = []

        with open(f"{self.raw_data_dir}/conflict.csv") as fin:
            for row in csv.DictReader(fin):
                # all conflicts must have a weight of -1
                assert row["weight"] == "-1"

                self.conflict_triples.append((
                    row["paper_id_anon"],
                    row["reviewer_id_anon"],
                    -1
                ))
        
        # assignment
        self.assignment_pairs = []

        with open(f"{self.raw_data_dir}/assignment.csv") as fin:
            for row in csv.DictReader(fin):
                self.assignment_pairs.append((
                    row["paper_id_anon"],
                    row["reviewer_id_anon"]
                ))

        # manual reassignments, ignoring the new assignments
        self.manual_reassignments = []

        with open(f"{self.raw_data_dir}/manual_reassignments.csv") as fin:
            for row in csv.DictReader(fin):
                self.manual_reassignments.append({
                    "paper_id": row["paper_id_anon"],
                    "from_reviewer_id": row["from_reviewer_id_anon"],
                    "to_reviewer_id": row["to_reviewer_id_anon"]
                })

        # reviews
        self.reviews = []

        with open(f"{self.raw_data_dir}/reviews.csv") as fin:
            for row in csv.DictReader(fin):
                self.reviews.append({
                    "paper_id": row["paper_id_anon"],
                    "reviewer_id": row["reviewer_id_anon"],
                    "rating": RATING_MAP[row["rating"]],
                    "confidence": CONFIDENCE_MAP[row["confidence"]],
                    "expertise": EXPERTISE_MAP[row["expertise"]]
                })

        # compile paper/reviewer ids
        # NOTE: assumes all ids are contained in the affinity matrix
        self.paper_ids = sorted(list(set([p_id for p_id, _, _ in self.affinity_triples])))
        self.reviewer_ids = sorted(list(set([r_id for _, r_id, _ in self.affinity_triples])))
        
        # build paper/reviewer indices
        self.index_papers = {p_id: i for i, p_id in enumerate(self.paper_ids)}
        self.index_reviewers = {r_id: i for i, r_id in enumerate(self.reviewer_ids)}

        self.n_papers = len(self.index_papers)
        self.n_reviewers = len(self.index_reviewers)

    def stats(self):
        print("Data Stats:")
        
        atts = {
            "paper_ids": self.paper_ids,
            "reviewer_ids": self.reviewer_ids,
            "affinity_triples": self.affinity_triples,
            "bid_triples": self.bid_triples,
            "conflict_triples": self.conflict_triples,
            "assignment_pairs": self.assignment_pairs,
            "manual_reassignments": self.manual_reassignments,
            "reviews": self.reviews   
        }
        
        for att, obj in atts.items():
            print(f"|{att}| = {len(obj)}")

    def compute_scores(self, config):
        # extract parameters from config
        w_affinity = config.params["aff_w"]
        w_bids = config.params["bid_w"]
        bids_map = config.params["bid_map"]
        agg_fun = config.params["agg"]

        # affinity
        affinity_pairs_map = {}

        for p_id, r_id, affinity in self.affinity_triples:
            affinity_pairs_map[(p_id, r_id)] = affinity

        # bids
        # bids_map = BIDS_MAP if bids_map is None else bids_map
        bids_pairs_map = {}

        for p_id, r_id, bid_str in self.bid_triples:
            bids_pairs_map[(p_id, r_id)] = bids_map[bid_str]

        # compute scores
        self.score_triples = []

        for p_id, r_id in affinity_pairs_map.keys():
            affinity = affinity_pairs_map[(p_id, r_id)]
            bid = bids_pairs_map.get((p_id, r_id), bids_map["default"])

            if agg_fun == "sum":
                score = w_affinity * affinity + w_bids * bid
            elif agg_fun == "mul":
                score = pow(2.0, bid) * affinity
            else:
                raise Exception(f"DataLoader: invalid agg_fun '{agg_fun}'")

            self.score_triples.append((p_id, r_id, score))

    def get_bids_matrix(self, config):
        Bids = np.zeros((self.n_papers, self.n_reviewers))        
        bids_map = config.params["bid_map"]
        
        for p_id, r_id, bid_str in self.bid_triples:
            p_idx = self.index_papers[p_id]
            r_idx = self.index_reviewers[r_id]
            bid_val = bids_map[bid_str]    
            Bids[p_idx, r_idx] = bid_val

        return Bids

    def get_bids_str_matrix(self):
        Bids = np.full((self.n_papers, self.n_reviewers), "none", dtype=object)
        
        for p_id, r_id, bid_str in self.bid_triples:
            p_idx = self.index_papers[p_id]
            r_idx = self.index_reviewers[r_id]
            Bids[p_idx, r_idx] = bid_str.replace(" ", "-").lower()

        return Bids
    
    def get_affinity_matrix(self):
        Aff = np.zeros((self.n_papers, self.n_reviewers))        

        for p_id, r_id, affinity in self.affinity_triples:
            p_idx = self.index_papers[p_id]
            r_idx = self.index_reviewers[r_id]
            Aff[p_idx, r_idx] = affinity

        return Aff

    def get_assignment_matrix(self):
        A = np.zeros((self.n_papers, self.n_reviewers))

        for p_id, r_id in self.assignment_pairs:
            p_idx = self.index_papers[p_id]
            r_idx = self.index_reviewers[r_id]
            A[p_idx, r_idx] = 1

        return A

    def get_proposed_assignment_matrix(self):
        # recreate the list of proposed assignment pairs
        pairs_realized = set(self.assignment_pairs)
        pairs_removed = set()
        pairs_added = set()

        for manual_reassignment in self.manual_reassignments:
            p_id = manual_reassignment["paper_id"]
            r_rmd_id = manual_reassignment["from_reviewer_id"]
            r_add_id = manual_reassignment["to_reviewer_id"]

            pairs_removed.add((p_id, r_rmd_id))
            pairs_added.add((p_id, r_add_id))

        pairs_proposed = pairs_realized - pairs_added | pairs_removed

        # construct the matrix
        A_proposed = np.zeros((self.n_papers, self.n_reviewers))

        for p_id, r_id in pairs_proposed:
            p_idx = self.index_papers[p_id]
            r_idx = self.index_reviewers[r_id]
            A_proposed[p_idx, r_idx] = 1

        return A_proposed

    def get_manual_reassignments(self):
        # assignments removed 
        R_rmd = np.zeros((self.n_papers, self.n_reviewers))
        # assignments added
        R_add = np.zeros((self.n_papers, self.n_reviewers))
        
        for manual_reassignment in self.manual_reassignments:
            p_id = manual_reassignment["paper_id"]
            r_rmd_id = manual_reassignment["from_reviewer_id"]
            r_add_id = manual_reassignment["to_reviewer_id"]

            p_idx = self.index_papers[p_id]
            r_rmd_idx = self.index_reviewers[r_rmd_id]
            r_add_idx = self.index_reviewers[r_add_id]
            
            R_rmd[p_idx, r_rmd_idx] = 1
            R_add[p_idx, r_add_idx] = 1
        
        return R_rmd, R_add

    def get_outcome_matrix(self, outcome_name):
        assert outcome_name in self.outcome_names

        # Y = np.zeros((self.n_papers, self.n_reviewers))
        Y = np.empty((self.n_papers, self.n_reviewers))
        Y[:] = np.nan

        for review in self.reviews:
            p_id = review["paper_id"]
            r_id = review["reviewer_id"]

            p_idx = self.index_papers[p_id]
            r_idx = self.index_reviewers[r_id]

            Y[p_idx, r_idx] = review[outcome_name]

        return Y

    def get_outcome_bounds(self, outcome_name):
        assert outcome_name in self.outcome_names

        outcome_maps = {
            "rating": RATING_MAP,
            "confidence": CONFIDENCE_MAP,
            "expertise": EXPERTISE_MAP
        }
        
        if outcome_name in outcome_maps:
            vals = outcome_maps[outcome_name].values()

        else:
            vals = [review[outcome_name]  for review in self.reviews]

        return min(vals), max(vals)
        
    def get_conflicts_matrix(self):
        Conflicts = np.zeros((self.n_papers, self.n_reviewers))        

        for p_id, r_id, _ in self.conflict_triples:
            p_idx = self.index_papers[p_id]
            r_idx = self.index_reviewers[r_id]
            Conflicts[p_idx, r_idx] = 1

        return Conflicts

#
# Solver
#
class Solver:

    # conference matching setup
    MIN_PAPERS = 8
    MAX_PAPERS = 9
    NUM_REVIEWERS = 3

    def __init__(self, data_loader, Q):
        self.reviewer_ids = data_loader.reviewer_ids
        self.paper_ids = data_loader.paper_ids
        self.conflict_triples = data_loader.conflict_triples
        self.score_triples = data_loader.score_triples
        self.Q = Q

    def solve(self):
        # setup encoder
        self.encoder = Encoder(
            reviewers=self.reviewer_ids,
            papers=self.paper_ids,
            constraints=self.conflict_triples,
            scores_by_type={"scores": {"edges": self.score_triples}},
            weight_by_type={"scores": 1.0},
            probability_limits=self.Q
        )

        # setup solver
        self.solver = RandomizedSolver(
            minimums=[self.MIN_PAPERS] * len(self.reviewer_ids),
            maximums=[self.MAX_PAPERS] * len(self.reviewer_ids),
            demands=[self.NUM_REVIEWERS] * len(self.paper_ids),
            encoder=self.encoder,
            allow_zero_score_assignments=True
        )

        # run solver
        self.solver.solve()
    
        # make sure it converged
        assert self.solver.solved

        return self.solver.fractional_assignment_matrix.copy()


def test():
    #
    # RunConfig
    #
    import json
    
    rc = RunConfig()
    
    print(json.dumps(rc.get_params(), indent=2))

    #
    # Data Loader
    #
    dl = DataLoader()

    A = dl.get_assignment_matrix()
    print("Assignment:", A.shape, np.sum(A))

    A_prop = dl.get_proposed_assignment_matrix()
    print("Proposed assignment", A_prop.shape, np.sum(A_prop))

    R_rmd, R_add = dl.get_manual_reassignments()
    print("R_rmd", R_rmd.shape, np.sum(R_rmd))
    print("R_add", R_add.shape, np.sum(R_add))

    Y = dl.get_outcome_matrix("expertise")
    print("Y", Y.shape, np.sum(Y[np.isnan(Y) == False]))

    y_min, y_max = dl.get_outcome_bounds("expertise")
    print("y_min, y_max", y_min, y_max)


if __name__ == "__main__":
    test()

# END