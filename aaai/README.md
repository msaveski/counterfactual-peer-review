# AAAI Analysis
This subdirectory contains scripts for running large-scale analysis on the AAAI data. 

The primary scripts are:
- `analyze_policy.py` : runs the main policy analysis
- `run_alt_policies.py` : computes and saves off-policy assignments for analysis
- `covariance_estimation.py/covariance_computation.py` : computes the covariance estimate for use in analysis
- `model_evaluation.py/model_prediction.py` : runs the model predictions for use in analysis

Note: the analysis in the paper is for "Stage 1" of the AAAI assignment, which corresponds to setting stage=0 in all scripts.
