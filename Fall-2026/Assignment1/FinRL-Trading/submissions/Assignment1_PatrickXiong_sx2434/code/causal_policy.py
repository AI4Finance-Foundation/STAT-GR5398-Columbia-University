"""Eligibility from contemporaneous entry quotes and dated public facts only.

No realized return, target missingness, future delisting, endpoint coverage, or
current-survivor membership is allowed to decide the supervised universe.
"""
import pandas as pd

RIGHTS_AVAILABLE={75107:pd.Timestamp('2022-11-04'),19502:pd.Timestamp('2025-03-11')}

def decision_eligibility(frame, hypothetical_rights_stress=False):
    restricted=pd.Series(False,index=frame.index)
    if not hypothetical_rights_stress:
        for perm,public_date in RIGHTS_AVAILABLE.items():
            restricted |= frame.permno.eq(perm)&frame.decision_date.gt(public_date)
    return frame.entry_observed.astype(bool)&~restricted

def label_observation_clock(frame,events,repairs):
    clock=pd.to_datetime(frame.label_end).copy()
    for row in events.itertuples(index=False):
        hit=frame.permno.eq(row.permno)&frame.decision_date.lt(row.crsp_return_date)&frame.label_end.ge(row.crsp_return_date)
        clock.loc[hit]=clock.loc[hit].map(lambda date:max(date,row.recognition_date))
    for row in repairs.itertuples(index=False):
        hit=frame.permno.eq(row.permno)&frame.decision_date.lt(row.gap_day)&frame.label_end.ge(row.gap_day)
        clock.loc[hit]=clock.loc[hit].map(lambda date:max(date,row.resume_day))
    return clock

def available_labels(frame):
    return pd.to_datetime(frame.label_available_date) if 'label_available_date' in frame else pd.to_datetime(frame.label_end)

def verify_known_horizon_outcomes(frame,last_observation,hypothetical_rights_stress=False):
    eligible=decision_eligibility(frame,hypothetical_rights_stress)
    required=eligible&frame.label_end.le(pd.Timestamp(last_observation))&available_labels(frame).le(pd.Timestamp(last_observation))
    missing=frame.loc[required&frame.target.isna(),['security_id','decision_date','label_end']]
    if len(missing):
        raise ValueError('Eligible realized-horizon outcomes missing; do not drop securities: '+str(missing.to_dict('records')))
    return eligible

def causal_partition(frame,train_cutoff,forecast_day,hypothetical_rights_stress=False):
    eligible=decision_eligibility(frame,hypothetical_rights_stress)
    tr=frame.loc[eligible&frame.decision_date.lt(train_cutoff)&frame.label_end.lt(train_cutoff)&available_labels(frame).lt(train_cutoff)]
    va=frame.loc[eligible&frame.decision_date.ge(train_cutoff)&frame.decision_date.lt(forecast_day)&frame.label_end.lt(forecast_day)&available_labels(frame).lt(forecast_day)]
    inf=frame.loc[frame.decision_date.eq(forecast_day)] # No eligibility or outcome filter.
    if tr.target.isna().any() or va.target.isna().any():
        raise ValueError('Missing eligible train/validation outcome; refusing silent row deletion.')
    return tr,va,inf
