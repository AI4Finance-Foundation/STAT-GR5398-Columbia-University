"""Missing-feature handling that is fitted on the training rows of ONE model fit.

Use `TrainOnlyImputer` as the first step of the model pipeline. A pipeline is
re-fitted by the estimator on exactly the rows it is given, so inside a
cross-validation each fold's imputer sees only that fold's training rows, and
the outer model's imputer sees only the outer training rows. Validation and
inference rows are only ever transformed.

`fit_logged` is the way to fit outside a pipeline: it records the rows in the
execution log under a TRAINING role first; any other role is refused.

`diagnostic_policy_as_of` (medians over all earlier decisions) is a description
of the data only. It is not the policy a model uses.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from fit_boundary_log import role_kind  # noqa: E402
from build_features import FEATURES  # noqa: E402

try:
    from sklearn.base import BaseEstimator, TransformerMixin
except ImportError:  # the logic below does not depend on scikit-learn
    BaseEstimator = TransformerMixin = object


class TrainOnlyImputer(BaseEstimator, TransformerMixin):
    """Median fill plus a missing indicator per feature; statistics come only from the rows passed to fit."""

    def __init__(self, features=tuple(FEATURES)):
        self.features = features

    def fit(self, X, y=None):
        frame = pd.DataFrame(X)
        self.medians_ = {}
        self.observed_ = {}
        for feature in self.features:
            values = pd.to_numeric(frame[feature], errors='coerce')
            self.observed_[feature] = int(values.notna().sum())
            self.medians_[feature] = None if values.notna().sum() == 0 else float(values.median())
        self.fitted_rows_ = int(len(frame))
        return self

    def transform(self, X):
        frame = pd.DataFrame(X).copy()
        for feature in self.features:
            values = pd.to_numeric(frame[feature], errors='coerce')
            frame[feature + '_missing'] = values.isna().astype(int)
            median = self.medians_[feature]
            # never observed in training: keep a neutral constant and rely on the indicator
            frame[feature] = values.fillna(0.0 if median is None else median)
        assert len(frame) == len(pd.DataFrame(X))
        return frame[[c for f in self.features for c in (f, f + '_missing')]]


def fit_logged(train_rows, log, window, role, cutoff, features=tuple(FEATURES)):
    """Fit on rows that the execution log accepts as training rows for this window; refuse any other role."""
    if role_kind(role) not in ('outer_train', 'inner_train'):
        raise ValueError(f'imputer may only be fitted on training rows, not on {role}')
    entry = log.record(window, role, train_rows, cutoff, context=dict(purpose='fit missing-value imputer'))
    imputer = TrainOnlyImputer(features).fit(train_rows)
    imputer.fitted_on_ = dict(window=window, role=role, key_sha256=entry['key_sha256'], rows=entry['rows'])
    return imputer


def diagnostic_policy_as_of(panel, cutoff, features=tuple(FEATURES)):
    """Descriptive medians over every row decided strictly before the cutoff. NOT a model's policy."""
    train = panel.loc[pd.to_datetime(panel['decision_date']) < pd.Timestamp(cutoff)]
    fitted = TrainOnlyImputer(features).fit(train)
    return dict(cutoff=str(pd.Timestamp(cutoff).date()), rows=int(len(train)),
                last_decision=str(pd.to_datetime(train['decision_date']).max().date()) if len(train) else None,
                features={f: dict(median=fitted.medians_[f], observed=fitted.observed_[f], missing=int(len(train)-fitted.observed_[f]))
                          for f in features})
