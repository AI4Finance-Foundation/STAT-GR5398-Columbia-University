# Starter code

The starter code for this assignment is **not in this repo** — it is the `finrobot_equity` module
in the FinRobot repository, which is actively maintained:

https://github.com/AI4Finance-Foundation/FinRobot/tree/master/finrobot_equity

```bash
git clone https://github.com/AI4Finance-Foundation/FinRobot.git
cd FinRobot
pip install -r requirements-equity.txt
cp finrobot_equity/core/config/config.ini.example finrobot_equity/core/config/config.ini
# edit config.ini with your FMP and OpenAI keys — never commit it
```

Work against a **branch of your own fork**, so your Part 3 improvement is a clean diff you can
submit (and, if it is good, open as a pull request).

## What is here

| File | Purpose |
| --- | --- |
| `run_pipeline.sh` | Wraps the two-step CLI so one ticker is one command; loop it over your coverage universe |
| `audit_templates/data_audit.csv` | Part 2.1 — pipeline value vs. filing value |
| `audit_templates/valuation_audit.csv` | Part 2.2 — assumptions and your own reconciliation |
| `audit_templates/grounding_audit.csv` | Part 2.3 — per-claim supported / unsupported / contradicted |
| `audit_templates/stability_test.csv` | Part 2.3 — same ticker, three runs |

The templates are column headers, not a straitjacket: add columns if your analysis needs them.

## Optional warm-up

If you have never built an agent before, the older AutoGen tutorial
([`tutorials_advanced/agent_annual_report.ipynb`](https://github.com/AI4Finance-Foundation/FinRobot/blob/master/tutorials_advanced/agent_annual_report.ipynb))
is a 30-minute introduction to the Expert / Shadow / UserProxy pattern. It is optional and
ungraded — it was last year's assignment, and this year's starts where it stops.
