"""Meaningful boundary checks; no course data or network required."""
import numpy as np
import pandas as pd
from historical_ev_ebitda import historical_ev_ebitda_value


class FakeView:
    def __init__(self, first='2019-01-01', margin=.2, revise=False):
        self.as_of = pd.Timestamp('2020-01-31')
        self.dates = pd.bdate_range(first, self.as_of)
        self.margin, self.revise = margin, revise

    def close(self, symbols):
        return pd.DataFrame({s:100. for s in symbols}, index=self.dates).loc[:self.as_of]

    def last_close(self, symbols):
        return {s:100. for s in symbols}

    def fundamentals(self, tickers):
        # A later revision changes only the final snapshot, not the historic
        # windows. This catches filtering only the latest visible version.
        rev = 20. if self.revise and self.as_of >= pd.Timestamp('2020-01-01') else 10.
        return pd.DataFrame([dict(ticker=t, datadate=d,
            accepted_date=d+pd.Timedelta(days=20), filing_date=d+pd.Timedelta(days=20),
            revenue_per_share=rev, ebitda_margin=self.margin,
            interest_debt_per_share=2., cash_per_share=1.)
            for t in tickers for d in pd.date_range('2017-12-31','2018-09-30',freq='QE')])


def run_boundary_tests():
    messages=[]
    eleven=historical_ev_ebitda_value('2020-01-31',['X'],FakeView(first='2019-02-01')).iloc[0]
    assert eleven.valid_months==11 and eleven.status=='unavailable' and pd.isna(eleven.fair_value)
    messages.append('PASS 11 valid months are unavailable; failure row retained')
    twelve,h=historical_ev_ebitda_value('2020-01-31',['X'],FakeView(),return_history=True)
    assert twelve.iloc[0].valid_months==12 and twelve.iloc[0].status=='ok'
    assert (pd.to_datetime(h.month_end)<pd.Timestamp('2020-01-31')).all()
    messages.append('PASS 12 valid months are accepted; valuation day excluded')
    neg=historical_ev_ebitda_value('2020-01-31',['X'],FakeView(margin=-.2)).iloc[0]
    assert neg.status=='unavailable' and 'EBITDA' in neg.reason and pd.isna(neg.fair_value)
    messages.append('PASS nonpositive EBITDA is unavailable with a stated reason')
    revised,h=historical_ev_ebitda_value('2020-01-31',['X'],FakeView(revise=True),return_history=True)
    assert np.allclose(h.multiple,101/8) and abs(revised.iloc[0].fair_value-201)<1e-10
    messages.append('PASS earlier views re-query filings instead of backfilling latest revised values')
    empty=historical_ev_ebitda_value('2020-01-31',[],FakeView())
    assert empty.empty and 'fair_value' in empty
    try:
        historical_ev_ebitda_value('2020-02-01',['X'],FakeView())
    except ValueError:
        pass
    else:
        raise AssertionError('future cutoff accepted')
    messages.append('PASS empty ticker list and future-cutoff rejection')
    return messages
