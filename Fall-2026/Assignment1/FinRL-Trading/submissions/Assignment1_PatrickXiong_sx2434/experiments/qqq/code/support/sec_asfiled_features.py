"""As-filed fundamental features from official SEC data, selected by accession.

For an issuer (CIK) and a decision date this module
  1. picks the periodic filing with the latest period of report that was
     available before the decision (no calendar-quarter bucketing), and
  2. builds a small set of ratios ONLY from facts carried by that accession.

Strict mode (default) accepts a quarterly flow only when the filing itself
reports a three-month fact. A 10-K normally reports the fiscal year only, so in
strict mode its flow ratios stay missing. With strict=False the fourth quarter
is computed as fiscal year (10-K) minus nine months (earlier 10-Q). That is a
difference ACROSS two filings: the annual figure can embed revisions of the
first nine months, so the result is labelled `cross_version_difference` and
must not be treated as a same-version quarter.

Later filings can restate or split-adjust the same period; they are never read
for an earlier decision. Nothing here is taken from the vendor tables.

Reduced feature set (seven ratios). Anything that cannot be traced is kept as a
row with a reason; rows are never dropped.

Not covered: identity of the issuer behind a historical ticker, prices,
historical sector labels, and the other vendor features.
"""
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

NEW_YORK = ZoneInfo('America/New_York')
ORIGINAL_FORMS = ('10-Q', '10-K', '10-QT', '10-KT')
ANNUAL_FORMS = ('10-K', '10-KT')

INSTANT_TAGS = {
    'assets_current': ['AssetsCurrent'],
    'liabilities_current': ['LiabilitiesCurrent'],
    'assets': ['Assets'],
    'liabilities': ['Liabilities'],
    'equity': ['StockholdersEquity', 'StockholdersEquityIncludingPortionAttributableToNoncontrollingInterest'],
    'cash': ['CashAndCashEquivalentsAtCarryingValue', 'CashCashEquivalentsRestrictedCashAndRestrictedCashEquivalents'],
}
FLOW_TAGS = {
    'revenue': ['RevenueFromContractWithCustomerExcludingAssessedTax', 'Revenues', 'SalesRevenueNet',
                'RevenueFromContractWithCustomerIncludingAssessedTax', 'SalesRevenueGoodsNet'],
    'net_income': ['NetIncomeLoss', 'ProfitLoss'],
}
FEATURES = ['cur_ratio', 'cash_ratio', 'liabilities_to_assets', 'equity_to_assets',
            'net_margin_q', 'roe_q', 'asset_turnover_q']


class Unresolved(Exception):
    pass


def available_at(filing):
    """Two full days after the end (New York) of the later recorded date.

    Same conservative policy as bias_remediation sec_asof.available_at. The API
    acceptance clock is not treated as a certified instant; only its date is used.
    """
    recorded_day = max(filing['filingDate'], filing['acceptanceDateTime'][:10])
    boundary = datetime.fromisoformat(recorded_day + 'T23:59:59').replace(tzinfo=NEW_YORK)
    return boundary + timedelta(days=2)


def decision_instant(decision_date):
    return datetime.fromisoformat(decision_date + 'T16:00:00').replace(tzinfo=NEW_YORK)


def merge_filings(*submission_blocks):
    """Merge `filings.recent`-style columnar blocks into one de-duplicated list."""
    seen = {}
    for block in submission_blocks:
        keys = list(block)
        for i, accession in enumerate(block['accessionNumber']):
            seen.setdefault(accession, {k: block[k][i] for k in keys})
    return list(seen.values())


def choose_filing(filings, decision_date):
    """Latest period of report among original periodic filings available before the decision."""
    cutoff = decision_instant(decision_date)
    usable = [f for f in filings if f['form'] in ORIGINAL_FORMS and f.get('reportDate')
              and f['reportDate'] <= decision_date and available_at(f) <= cutoff]
    if not usable:
        return None
    latest = max(f['reportDate'] for f in usable)
    return sorted((f for f in usable if f['reportDate'] == latest),
                  key=lambda f: (f['filingDate'], f['acceptanceDateTime']))[0]


def _days(start, end):
    return (datetime.fromisoformat(end) - datetime.fromisoformat(start)).days


def _accession_facts(companyfacts, accession, tag):
    node = companyfacts.get('facts', {}).get('us-gaap', {}).get(tag)
    if not node:
        return []
    return [dict(f, tag=tag) for f in node.get('units', {}).get('USD', []) if f.get('accn') == accession]


def _single(candidates, what):
    values = {c['val'] for c in candidates}
    if not candidates:
        return None
    if len(values) != 1:
        raise Unresolved(f'ambiguous {what}: {len(values)} different values in one accession')
    return candidates[0]


def instant_fact(companyfacts, accession, tags, end):
    for tag in tags:
        found = _single([f for f in _accession_facts(companyfacts, accession, tag)
                         if f['end'] == end and 'start' not in f], tag)
        if found:
            return found
    return None


def duration_fact(companyfacts, accession, tag, end, low, high, start=None):
    candidates = [f for f in _accession_facts(companyfacts, accession, tag)
                  if f['end'] == end and 'start' in f and low <= _days(f['start'], f['end']) <= high
                  and (start is None or abs(_days(start, f['start'])) <= 3)]
    return _single(candidates, tag)


def quarter_flow(companyfacts, filings, chosen, name, decision_date, strict=True):
    """Quarterly flow for the chosen filing, with every source fact listed.

    A three-month fact in the chosen accession is always preferred (some 10-Ks
    carry one). Otherwise, strict mode stops. With strict=False a 10-K fiscal-year
    fact minus the nine-month fact of the preceding 10-Q is returned and labelled
    as a cross-version difference.
    """
    end = chosen['reportDate']
    accession = chosen['accessionNumber']
    for tag in FLOW_TAGS[name]:
        fact = duration_fact(companyfacts, accession, tag, end, 80, 100)
        if fact:
            return fact['val'], [fact], 'reported_quarter'
    if chosen['form'] not in ANNUAL_FORMS:
        return None, [], ''
    if strict:
        raise Unresolved(f'{name}: annual filing reports no three-month fact; strict mode does not difference across filings')
    for tag in FLOW_TAGS[name]:
        year = duration_fact(companyfacts, accession, tag, end, 350, 380)
        if not year:
            continue
        cutoff = decision_instant(decision_date)
        prior = [f for f in filings if f['form'] in ('10-Q', '10-QT') and f.get('reportDate')
                 and 80 <= _days(f['reportDate'], end) <= 100 and available_at(f) <= cutoff]
        if len({f['reportDate'] for f in prior}) != 1:
            raise Unresolved(f'{name}: fourth quarter needs exactly one preceding 10-Q period, found {len(prior)}')
        third = sorted(prior, key=lambda f: (f['filingDate'], f['acceptanceDateTime']))[0]
        to_date = duration_fact(companyfacts, third['accessionNumber'], tag, third['reportDate'], 250, 290, start=year['start'])
        if not to_date:
            raise Unresolved(f'{name}: nine-month fact with tag {tag} not in preceding 10-Q {third["accessionNumber"]}')
        return year['val'] - to_date['val'], [year, to_date], 'cross_version_difference'
    return None, [], ''


def build_row(companyfacts, filings, decision_date, strict=True):
    """One output row; never raises for missing data. Returns (row, lineage)."""
    row = dict(decision_date=decision_date, status='', reason='', form='', period_of_report='', accession='',
               filing_date='', acceptance_api_raw='', available_under_policy='', flow_basis='', strict_mode=strict,
               **{f'{n}_tag': '' for n in list(INSTANT_TAGS) + list(FLOW_TAGS)},
               **{f: None for f in FEATURES})
    lineage = []
    chosen = choose_filing(filings, decision_date)
    if chosen is None:
        row.update(status='unresolved', reason='no periodic filing available before the decision in the official list')
        return row, lineage
    row.update(form=chosen['form'], period_of_report=chosen['reportDate'], accession=chosen['accessionNumber'],
               filing_date=chosen['filingDate'], acceptance_api_raw=chosen['acceptanceDateTime'],
               available_under_policy=available_at(chosen).isoformat())
    accession, end = chosen['accessionNumber'], chosen['reportDate']
    values, problems = {}, []
    for name, tags in INSTANT_TAGS.items():
        try:
            fact = instant_fact(companyfacts, accession, tags, end)
        except Unresolved as error:
            problems.append(str(error))
            continue
        if fact is None:
            problems.append(f'{name}: no fact for the period of report in this accession')
        else:
            values[name] = fact['val']
            row[f'{name}_tag'] = fact['tag']
            lineage.append(dict(item=name, tag=fact['tag'], value=fact['val'], start='', end=fact['end'],
                                accession=fact['accn'], filed=fact['filed'], form=fact.get('form', '')))
    bases = set()
    for name in FLOW_TAGS:
        try:
            value, facts, basis = quarter_flow(companyfacts, filings, chosen, name, decision_date, strict)
        except Unresolved as error:
            problems.append(str(error))
            continue
        if value is None:
            problems.append(f'{name}: no quarterly or fiscal-year fact in this accession')
            continue
        values[name] = value
        row[f'{name}_tag'] = facts[0]['tag']
        bases.add(basis)
        for fact in facts:
            lineage.append(dict(item=name, tag=fact['tag'], value=fact['val'], start=fact['start'], end=fact['end'],
                                accession=fact['accn'], filed=fact['filed'], form=fact.get('form', '')))
    row['flow_basis'] = ';'.join(sorted(bases))

    def ratio(numerator, denominator):
        if numerator in values and denominator in values:
            if values[denominator] == 0:
                note = f'{denominator} is zero in this filing; ratios on it are undefined'
                if note not in problems:
                    problems.append(note)
                return None
            return values[numerator] / values[denominator]
        return None
    row.update(cur_ratio=ratio('assets_current', 'liabilities_current'), cash_ratio=ratio('cash', 'liabilities_current'),
               liabilities_to_assets=ratio('liabilities', 'assets'), equity_to_assets=ratio('equity', 'assets'),
               net_margin_q=ratio('net_income', 'revenue'), roe_q=ratio('net_income', 'equity'),
               asset_turnover_q=ratio('revenue', 'assets'))
    built = sum(row[f] is not None for f in FEATURES)
    if not lineage:
        row.update(status='unresolved', reason='accession has no usable us-gaap USD facts in companyfacts: ' + ' | '.join(problems))
    elif built == len(FEATURES):
        row.update(status='complete', reason='')
    else:
        row.update(status='partial', reason=' | '.join(problems))
    # No lineage row may come from a filing that was not available, whatever the caller passed in.
    cutoff = decision_instant(decision_date)
    by_accession = {f['accessionNumber']: f for f in filings}
    for item in lineage:
        assert available_at(by_accession[item['accession']]) <= cutoff, 'unavailable accession leaked into lineage'
    return row, lineage
