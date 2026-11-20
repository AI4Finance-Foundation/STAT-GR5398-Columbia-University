"""Stock selection (course-provided).

Quarterly fundamentals and price momentum feed six regressors plus a stacked
ensemble, retrained every month; each week the portfolio holds the top 10
growth_tech names, weighted by score between 5% and 15%.

    selection/selector.py      the model: features, training, monthly inference
    selection/strategy.py      monthly predictions -> ranked picks -> target_weights()
    selection/allocation.py    score-based weights, 5% to 15% each
    selection/ranking.py       picking within a bucket, incumbent preference
    selection/config.py        builds everything from selection/configs/*.json
    selection/scores.py        replays the precomputed monthly scores (fast)
    selection/build_scores.py  regenerates those scores by retraining every month (slow)
    selection/run_picks.py     prints the target portfolio for one date
"""
from .scores import MonthlyScores
from .strategy import BUCKETS, ModelOptions, MonthlyPredictions, Parameters, Prediction, target_weights

BUCKET = "growth_tech"

__all__ = ["BUCKET", "BUCKETS", "ModelOptions", "MonthlyPredictions", "MonthlyScores", "Parameters",
           "Prediction", "target_weights"]
