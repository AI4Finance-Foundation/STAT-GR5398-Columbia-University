# Reproduction

For the analyses below, copy the submitted module to `valuation/student_valuation.py`. Overlay the packaged `analysis/`, `tests/`, `output/`, and `run_portfolio.py` in the working copy. The packaged runner provides the filter-rule switch and weekly audit.

## Data audit and valuation checks

```bash
python -m unittest discover -s tests -v
python -m analysis.data_audit
python -m analysis.data_diagnostics
python -m analysis.model_smoke
python -m analysis.tax_sensitivity
python -m analysis.model_hardening
python -m analysis.beta_convention_audit
python -m analysis.engine_rejection_breakdown
```

## Configuration selection and validation

```bash
python -m analysis.optimization_validation dev
python -m analysis.optimization_validation validation
python run_portfolio.py --period val --ablation --timing full
```

## Figures

```bash
python analysis/report_figures.py
```