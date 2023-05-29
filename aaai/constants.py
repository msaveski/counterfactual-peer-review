
root = './data/'
COVAR_DIR = './data/cov/'
MODEL_DIR = './data/models/'

p1_prefix = root + 'phase1/'
p2_prefix = root + 'phase2/'
prefixes = [p1_prefix, p2_prefix]

REVIEWER_PAPER_INFO = [p + 'rp_info.pkl' for p in prefixes]
SIMILARITY_FILE = [p + 'similarities.npz' for p in prefixes]
OUTCOME_MATRIX_FILE = [p + 'outcomes.npz' for p in prefixes]
ON_POLICY_FILE = [p + 'onpolicy.npz' for p in prefixes]
CONSTRAINTS_FILE = [p + 'constraints.npz' for p in prefixes]

COVAR_FILE = [f'{COVAR_DIR}cov1000000{p}.npz' for p in ['', '_s2']]

ALT_FOLDER = [root + 'alts/phase1/', root + 'alts/phase2/']

model_names = ['logistic-reg', 'logistic-reg-stded', 'knn-clf', 'rigdge-clf', 'ord-logit', 'ord-probit', 
        'ord-logistic-at', 'ord-logistic-it', 'rec-svdpp', 'rec-knn-with-means', 'rec-knn-with-z-score']
