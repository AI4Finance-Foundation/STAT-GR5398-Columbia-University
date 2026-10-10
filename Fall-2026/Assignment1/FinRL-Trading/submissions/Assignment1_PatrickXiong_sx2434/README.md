# FinRL Assignment 1 - Patrick Xiong (sx2434)

**Status: draft; historical data repair pending WRDS approval.**

Medium: [https://medium.com/@sx2434/a-finrl-backtest-after-repairing-prices-identities-and-financial-dates-1771f8ed587d](https://medium.com/@sx2434/a-finrl-backtest-after-repairing-prices-identities-and-financial-dates-1771f8ed587d)

Report: [research PDF](FinRL_Assignment1_Research_Draft.pdf).

Unzip `research_outputs.zip` into this folder before running the commands below. The archive contains derived forecasts, diagnostics and comparison outputs; raw course datasets and recovered quotes are excluded.

## Research snapshot - 8 October 2026

This package preserves the 7 October work separately and evaluates a fixed
**top-10% development candidate**, with top 25% as the course baseline. Top 10%
was selected using the earlier development results before this repaired run;
this is not an untouched holdout and no new threshold search was conducted.
The report states whether this candidate beats SPY in the observed full window.
The incomplete historical universe prevents a survivorship-free claim.

## What changed

- Price recovery has a per-symbol decision ledger, current SEC issuer checks,
  official rename evidence, complete in-life adjusted prices and recorded
  overlap checks. Forty-seven recovered ticker series were accepted. Ambiguous
  failed-bank, split/share-class and merger histories remain excluded.
- Official aliases are mapped to one security identity. Each security uses one
  complete price source for both training labels and execution. Financial rows
  are selected once per security/quarter by the documented timestamp/canonical
  symbol rule. Source disagreements and duplicate decisions are preserved.
- Raw statements are joined by the aggregate income filing and actual fiscal
  period. Fiscal-quarter labels alone previously matched the wrong records.
- Daily holdings drift; purchases earn returns only after the entry close;
  fees are 10 bps of each bought/sold dollar, funded from assets. Missing held
  prices stop the simulation rather than indefinitely carrying a stale mark.
- Full available trading history and the prior 2020-09-01 common window are
  both reported. Stock/sector gross dollar-PnL contributions minus the separate
  fee contribution equal the change in NAV.
- `sample_flow.csv` is produced during training; the verifier and report accept
  alternate run directories, closing the previous clean-report gap.

## Reproduce

Use Python 3.9–3.11. The training environment and pinned dependencies are in
`environment_training.txt` and `requirements.txt`. macOS may need `libomp` for
XGBoost/LightGBM. Report creation requires ReportLab. Run from this directory:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python code/reproduce.py --cache data_cache --prefix reproduced
```

On Windows activate `.venv\Scripts\activate`. A 7z-compatible extractor (7-Zip
or bsdtar) is needed. This command verifies/downloads the three pinned course
sources, prepares data, rebuilds accepted recovery quotes, runs the financial
spot checks, trains, compares both windows, verifies arithmetic/identities,
and produces the PDF and unpublished blog. It writes new `reproduced_run`,
`reproduced_full` and `reproduced_common` folders.

The exact course preparation command invoked by the bootstrap is:

```bash
python code/prepare_data.py --db data_cache/finrl_trading.db --fundamentals data_cache/fundamental_data_full.csv --constituents data_cache/sp500_historical_constituents.csv --output-dir data_cache/prepared --start 2015-01-01 --end 2025-09-30 --trade-start 2020-09-30
```

To rebuild downstream outputs from saved fitted predictions (no model refit):

```bash
python code/reproduce.py --cache data_cache --prefix checked --reuse-saved-training
```

This second command requires the frozen accepted recovery CSVs under
`evidence/price_recovery/accepted`. Restore the private input snapshot and frozen SPY/QQQ input CSVs first. Benchmark inputs are downloaded for a fresh run and excluded from the public archive.
It checks all 47 input hashes and the alias mapping against the fitted run,
and checks the pinned course source hashes. Changed inputs stop replay.
A fresh network recovery belongs to a new training run, in a separate copy
of this working directory; it must not silently replace the saved experiment.

Individual steps, useful for rerunning after a failed comparison:

```bash
python code/compare_backtests.py --cache data_cache --baseline reference/original --prior reference/overnight --audited audited_run --output comparison_full --window full
python code/compare_backtests.py --cache data_cache --baseline reference/original --prior reference/overnight --audited audited_run --output comparison_common --window common
python code/verify_outputs.py --root . --run audited_run --comparison comparison_full
python code/verify_outputs.py --root . --run audited_run --comparison comparison_common
python code/build_report.py --root . --run audited_run --comparison comparison_full --common comparison_common
```

If report libraries live in a separate interpreter, pass
`--report-python /absolute/path/to/python` to `reproduce.py`. The main saved PDF
uses `audited_run`, `comparison_full` and `comparison_common`.

## Exact private input replay versus a fresh download

For an exact replay of this local run, restore the separately supplied private
recovery-input snapshot into `evidence/price_recovery/accepted/` and verify its
CSV hashes against `audited_run/audit.json` (`recovery_price_sha256`). Keep that input snapshot outside
the course submission. The pinned course data can be recreated by the bootstrap.
A clean-directory replay with frozen inputs validates file dependencies and
arithmetic; it does not demonstrate that mutable network prices are unchanged.
`--reuse-saved-training` also reuses predictions and therefore does not verify a
fresh model refit. Any delivered reproduction record states which mode was run.

A clean-directory downstream replay can be run with:

```bash
python code/clean_replay.py --destination /absolute/new/empty/staging/path --cache data_cache --report-python /absolute/path/to/python
```

It copies only code, frozen predictions and declared inputs, generates both
comparisons and the PDF from scratch, and checks numerical agreement. The
`evidence/clean_replay_verification.json` record explicitly says that the replay
reuses the fitted predictions and does not make a fresh network download.

## Evidence and outputs

- `candidate_plan.json`: development choice and fixed comparison rules.
- `evidence/price_recovery/recovery_decisions.csv`: accepted/rejected records,
  provider symbol, SEC CIK/title, overlap errors, official alias source.
- `evidence/price_recovery/sec_company_tickers_provenance.json`: official SEC
  target subset used when the direct endpoint rejected automated requests.
- `evidence/root_corporate_action_sources.json`: official rename/share-class
  and spin-off sources. Unresolved symbols are not silently remapped.
- `evidence/financial/`: deterministic 48-row statement spot check and mapping
  evidence. The supplied PE equals quarter-date price / quarterly EPS; this
  differs from the provider's PE definition without establishing leakage.
- `audited_run/`: labels, availability, canonical identity decisions, price
  source choices, coverage, predictions, validation scores and importances.
- `comparison_full/`, `comparison_common/`: metrics, daily NAV, holdings,
  cost/delay tests, disclosure buffer, stock/sector contributions and checks.
- `FinRL_Assignment1_Research_Draft.pdf`, `Medium_draft.md`:
  evidence-backed drafts. The Medium article is published at the link above; the PDF remains a research draft.

The original/first-rerun predictions in `reference/` are frozen comparison
inputs; their older models are not refitted. The revised models are refitted
from the public inputs. All strategies use 52 financial features, fold-local
median imputation, chronological folds and completed-label purges. The original
starter notebook remains unchanged in the course checkout.

## Reproducibility limits

The course data are pinned to FinRL-Trading commit
`dd1fdcf5db69b4cf42e90a185211defa8d5ae273` and verified by SHA-256. Public price
recovery is a dated provider snapshot, not an immutable market-data release.
Later dividends, corrections, delistings or API changes can alter downloads.
Recovery metadata and hashes record exactly what was used locally. A fresh
network reproduction must check the decision ledger and report any changed
inputs; it must not pretend changed quotes are bitwise identical.

A current SEC issuer match plus adjusted-price history does not independently
certify every historical corporate action. Missing historical members include
failed/acquired companies; known absence is disclosed. Original financial
vintages and historical sector assignments remain unverified. The 2024 onward
slice was already observed during development. Bootstrap intervals are
conditional diagnostics, not a strategy-selection correction or an extra
course requirement. Costs are assumptions, not measured live trading costs.

Do not commit `data_cache`, `public_price_probe/*.csv` other than the small
`recovery_inventory.csv`, raw Yahoo JSON, accepted quote CSVs, intermediate
training runs, or temporary rendered pages. They are input data, not final
results. Retain source URLs, hashes, audit decisions and derived results.

## Submission status

Author and UNI are confirmed. Medium is published. The course PR is submitted as a draft, pending historical price/terminal-event and original financial-vintage repairs. The graded bias-free sample requirement remains unresolved; see `submission_status.md`.

## Independent checks completed

Claude reviewed the engine read-only. No result-changing error was found for
the full-window top-10 portfolio; its independent checks covered timing,
turnover, costs, rankings and contribution reconciliation. See
`evidence/claude_review_followup.md` for findings and their disposition.

A separate checker then rebuilt top-10 and top-25 returns in both windows from
source adjusted closes and saved rebalance holdings, checked forecast rankings,
drift, cash and financed costs, and verified SPY/QQQ against saved price ratios:

```bash
python code/verify_market_reconstruction.py --root . --cache data_cache
```

This checks market accounting without invoking the backtest engine. It shares
the documented source loader and historical-membership mapping, so it does not
independently certify vendor data. `missing_held=0` means the strict missing-price
abort was not triggered in this completed sample; it is not a universe-coverage
certificate. Do not liquidate a delisted stock at its last quote without
verified cash or security proceeds. Calendar-date decisions before the start
of a trading window are an untested edge case; the delivered audited inference
dates are actual trading dates.

## Packaging and new-run checks (9 October)

The new-run entry point now generates its missing public-price probe inventory and benchmark input files before recovery. Its dispatch order was smoke-tested without downloading data or retraining. The previously verified clean replay used frozen inputs and saved predictions. No complete fresh-network refit of this updated public package has been certified. Reproduction relies on provider availability; network failures must stop the run. The PDF describes the 8 October experiment.

Raw benchmark CSVs, recovered input prices, prepared financial inputs, caches and credentials are excluded from the public archive. The data-repair request lists and acceptance conditions are in `data_access/`.

The compact archive omits daily stock-contribution ledgers; the backtest regenerates them. Aggregate attribution tables and checks are retained.
