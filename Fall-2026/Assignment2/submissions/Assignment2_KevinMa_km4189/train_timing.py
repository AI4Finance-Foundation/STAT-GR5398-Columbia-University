#!/usr/bin/env python3
"""Unified DEV-only GRU and selection-NAV contextual-bandit trainer.
From course source_code/: python train_timing.py --course-root .
Default embeds both new banks in timing/student_timing.py; --verify retrains
and compares without modifying the source. No VAL/TEST performance is run.
The selection-only NAV is generated from official course scores/backtester.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import ast
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

# Torch is only required to RETRAIN; the frozen timing inference uses NumPy.
try:
    import torch
    from torch import nn
except ImportError as exc:
    raise SystemExit("Install PyTorch before training: pip install torch") from exc

# Reference run data SHA256: {'daily_adjusted.parquet': '5e5972854acdae9e3c57afc08fe056b519f44f0185f2a3c89831498baed0e56f', 'market_series.parquet': 'e38d2d4106162670e79464ec9d4beec476e118ecd9f992a784c1ec2605851abc'}
# Reference trained bank SHA256: b7a26de5c4f151e5278927359aba23acff5236c2f615033c2d6ccbed1accf1aa
DEV_START = pd.Timestamp("2017-01-03")
DEV_END = pd.Timestamp("2021-12-31")
YEARS = tuple(range(2017, 2022))
SEQUENCE_DAYS = 20
HORIZON_DAYS = 5
INPUT_DIM = 6
HIDDEN_DIM = 8
OUTPUT_DIM = 3
TARGET_SCALE = 0.08
WEIGHT_KEYS = ("weight_ih_l0", "weight_hh_l0", "bias_ih_l0", "bias_hh_l0",
               "head_weight", "head_bias")


def arguments() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--course-root", type=Path, default=Path.cwd(),
                   help="Course source_code directory containing data/prices/ (default: cwd)")
    p.add_argument("--price-file", type=Path, default=None, help="Override adjusted-prices parquet path")
    p.add_argument("--market-file", type=Path, default=None, help="Override market-series parquet path")
    p.add_argument("--out", type=Path, default=None,
                   help="Checkpoint output directory (default: source_code/output/gru_retraining_dev)")
    p.add_argument("--seed", type=int, default=5398)
    p.add_argument("--epochs-initial", type=int, default=75)
    p.add_argument("--epochs-update", type=int, default=35)
    p.add_argument("--lr-initial", type=float, default=0.003)
    p.add_argument("--lr-update", type=float, default=0.0015)
    p.add_argument("--weight-decay", type=float, default=0.01)
    p.add_argument("--clip-grad", type=float, default=1.0)
    p.add_argument("--first-year", type=int, default=2017, choices=YEARS)
    p.add_argument("--last-year", type=int, default=2021, choices=YEARS)
    p.add_argument("--verify-timing", type=Path, default=None,
                   help="Compare all generated tensors to a timing source file")
    p.add_argument("--embed-timing", type=Path, default=None,
                   help="Explicitly replace bank in an existing timing source COPY")
    p.add_argument("--verify", action="store_true", help="Verify both banks without modifying timing")
    return p.parse_args()


def read_data_table(path: Path, columns: list[str]) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(f"Course data not found: {path}")
    try:
        return pd.read_parquet(path, columns=columns)
    except ImportError as exc:
        raise RuntimeError("Reading the course parquet bundle requires pyarrow: pip install pyarrow") from exc


def load_market_data(root: Path, price_file: Path | None, market_file: Path | None):
    # Restrict to DEV before feature or label construction. The underlying
    # parquet scan may read later rows, but they cannot enter training arrays.
    # Earlier 2016 history is input lookback only.
    price_path = price_file or root / "data/prices/daily_adjusted.parquet"
    market_path = market_file or root / "data/prices/market_series.parquet"
    price = read_data_table(price_path, ["date", "symbol", "close"])
    price = price[price["symbol"].isin(("QQQ", "SPY"))].copy()
    price["date"] = pd.to_datetime(price["date"]).dt.normalize()
    price = price[price["date"] <= DEV_END]
    price = price.drop_duplicates(["date", "symbol"], keep="last")
    panel = price.pivot(index="date", columns="symbol", values="close").sort_index()
    if not {"QQQ", "SPY"}.issubset(panel.columns):
        raise ValueError("Course price data must contain QQQ and SPY")

    qqq = pd.to_numeric(panel["QQQ"], errors="coerce")
    qqq = qqq.replace([np.inf, -np.inf], np.nan).dropna()
    qqq = qqq[qqq > 0].astype(float)
    spy = pd.to_numeric(panel["SPY"], errors="coerce").reindex(qqq.index).ffill()

    market = read_data_table(market_path, ["date", "symbol", "close"])
    market = market[market["symbol"] == "^VIX"].copy()
    market["date"] = pd.to_datetime(market["date"]).dt.normalize()
    market = market[market["date"] <= DEV_END]
    market = market.drop_duplicates("date", keep="last")
    vix = pd.to_numeric(market.set_index("date")["close"], errors="coerce")
    vix = vix.sort_index().reindex(qqq.index).ffill()

    if (qqq.empty or qqq.index.min() >= DEV_START or
            qqq.index.max() < DEV_END or len(qqq) < 500):
        raise ValueError("Expected 2016 lookback sessions and complete DEV data through 2021-12-31")
    if not (np.isfinite(spy.to_numpy()).all() and np.isfinite(vix.to_numpy()).all()):
        raise ValueError("SPY or VIX has missing past-only values")
    if not (spy.gt(0).all() and vix.gt(0).all()):
        raise ValueError("SPY/VIX contain nonpositive values")
    return qqq, spy.astype(float), vix.astype(float)


def feature_array(qqq: pd.Series, spy: pd.Series, vix: pd.Series) -> np.ndarray:
    """Exact feature ordering/scales of final ``_sequence_inputs``."""
    r1 = qqq.pct_change(fill_method=None).fillna(0.0)
    q5 = (qqq / qqq.shift(5) - 1.0).fillna(0.0)
    s5 = (spy / spy.shift(5) - 1.0).fillna(0.0)
    vol20 = (r1.rolling(20).std() * math.sqrt(252)).fillna(0.0)
    v5 = (vix / vix.shift(5) - 1.0).fillna(0.0)
    x = np.column_stack((r1.to_numpy() / .025,
                         q5.to_numpy() / .075,
                         s5.to_numpy() / .060,
                         vol20.to_numpy() / .35,
                         vix.to_numpy() / 25.0 - 1.0,
                         v5.to_numpy() / .35))
    # Same numerical clipping and missing-value fallback as frozen inference.
    return np.clip(np.nan_to_num(x, nan=0., posinf=3., neginf=-3.), -3., 3.).astype(np.float32)


def make_examples(qqq: pd.Series, features: np.ndarray):
    """One sample at every valid daily close; label_end is its fifth FUTURE session."""
    p = qqq.to_numpy(dtype=np.float64)
    as_of_idx = np.arange(SEQUENCE_DAYS - 1, len(p) - HORIZON_DAYS, dtype=int)
    # Prices dated before 2017 are for sequence warm-up, not training labels.
    as_of_idx = as_of_idx[qqq.index[as_of_idx] >= DEV_START]
    past_idx = as_of_idx[:, None] - np.arange(SEQUENCE_DAYS - 1, -1, -1)[None, :]
    future_idx = as_of_idx[:, None] + np.arange(1, HORIZON_DAYS + 1)[None, :]
    ratio = p[future_idx] / p[as_of_idx, None] - 1.0
    y = np.column_stack((ratio.min(axis=1), ratio.max(axis=1), ratio[:, -1]))
    y = np.clip(y / TARGET_SCALE, -2.0, 2.0)
    valid = np.isfinite(y).all(axis=1) & np.isfinite(features[past_idx]).all(axis=(1, 2))
    x = features[past_idx[valid]]
    y = y[valid].astype(np.float32)
    observed_dates = qqq.index[as_of_idx[valid]]
    completed_dates = qqq.index[future_idx[valid, -1]]
    if len(x) == 0:
        raise ValueError("No valid DEV training samples")
    assert x.shape[1:] == (SEQUENCE_DAYS, INPUT_DIM)
    assert y.shape[1:] == (OUTPUT_DIM,)
    return x, y, observed_dates, completed_dates


class ForecastGRU(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.gru = nn.GRU(INPUT_DIM, HIDDEN_DIM, batch_first=True)
        self.head = nn.Linear(HIDDEN_DIM, OUTPUT_DIM)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _, h_last = self.gru(x)
        return self.head(h_last[-1])


def extract_numpy_weights(model: ForecastGRU) -> tuple[np.ndarray, ...]:
    weights = tuple(np.asarray(t.detach().cpu().numpy(), dtype=np.float32).copy()
                    for t in (model.gru.weight_ih_l0, model.gru.weight_hh_l0,
                              model.gru.bias_ih_l0, model.gru.bias_hh_l0,
                              model.head.weight, model.head.bias))
    shapes = ((24, 6), (24, 8), (24,), (24,), (3, 8), (3,))
    assert tuple(w.shape for w in weights) == shapes
    return weights


def numerical_compatibility(model: ForecastGRU, sample: np.ndarray) -> float:
    """Check that the exported gate ordering agrees with the frozen NumPy GRU."""
    wi, wh, bi, bh, hw, hb = extract_numpy_weights(model)
    h = np.zeros((sample.shape[0], HIDDEN_DIM), dtype=np.float64)
    seq = sample.astype(np.float64)
    for t in range(SEQUENCE_DAYS):
        gx = seq[:, t, :] @ wi.T + bi
        gh = h @ wh.T + bh
        reset = 1.0 / (1.0 + np.exp(-np.clip(gx[:, :8] + gh[:, :8], -40., 40.)))
        update = 1.0 / (1.0 + np.exp(-np.clip(gx[:, 8:16] + gh[:, 8:16], -40., 40.)))
        new = np.tanh(gx[:, 16:24] + reset * gh[:, 16:24])
        h = (1.0 - update) * new + update * h
    pred_numpy = np.clip(h @ hw.T + hb, -2., 2.)
    model.eval()
    with torch.no_grad():
        pred_torch = np.clip(model(torch.from_numpy(sample)).cpu().numpy(), -2., 2.)
    return float(np.max(np.abs(pred_numpy - pred_torch)))


def python_bank_text(bank: dict[int, tuple[np.ndarray, ...]]) -> str:
    """Copy-paste-ready six-tuple layout for ``_GRU_BANK`` in student_timing.py."""
    lines = ["# Newly trained DEV-only checkpoint bank.",
             "# Exact float32 values promoted to float64 for inference.",
             "import numpy as np", "", "_GRU_BANK = {"]
    for year in sorted(bank):
        lines.append(f"    {year}: (")
        for a in bank[year]:
            arr = a.astype(np.float64).tolist()
            # format with numpy array representation in the same sequence as the final module.
            lines.append(f"        np.array({repr(arr)}, dtype=np.float64),")
        lines.append("    ),")
    lines.extend(["}", ""])
    return "\n".join(lines)


def train(args: argparse.Namespace) -> Path:
    if args.first_year > args.last_year:
        raise ValueError("--first-year must be <= --last-year")
    # Training earlier years is necessary for the warm-start reconstruction.
    if args.first_year != 2017:
        raise ValueError("This warm-start trainer must start at 2017 to build a causal weight history")
    root = args.course_root.expanduser().resolve()
    output = (args.out or root / "output/gru_retraining_dev").expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    qqq, spy, vix = load_market_data(root, args.price_file, args.market_file)
    features = feature_array(qqq, spy, vix)
    x, y, dates, completed = make_examples(qqq, features)
    model = ForecastGRU().cpu()
    loss_function = nn.SmoothL1Loss(beta=0.25)
    bank: dict[int, tuple[np.ndarray, ...]] = {}
    provenance = []

    for year in range(2017, args.last_year + 1):
        cutoff = pd.Timestamp(year, 12, 31)
        eligible = (completed <= cutoff)
        n = int(eligible.sum())
        if n < 100:
            raise ValueError(f"Insufficient completed samples for {year}: {n}")
        assert (dates[eligible] < completed[eligible]).all()
        assert completed[eligible].max() <= cutoff
        xx = torch.from_numpy(x[eligible])
        yy = torch.from_numpy(y[eligible])

        epochs = args.epochs_initial if year == 2017 else args.epochs_update
        lr = args.lr_initial if year == 2017 else args.lr_update
        optimizer = torch.optim.Adam(model.parameters(), lr=lr,
                                     weight_decay=args.weight_decay)
        model.train()
        for epoch in range(epochs):
            optimizer.zero_grad(set_to_none=True)
            out = model(xx)
            loss = loss_function(out, yy)
            if not torch.isfinite(loss):
                raise RuntimeError(f"Non-finite training loss at {year} epoch {epoch+1}")
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad)
            optimizer.step()
        model.eval()
        with torch.no_grad():
            final_loss = float(loss_function(model(xx), yy).cpu().item())
        err = numerical_compatibility(model, x[eligible][:min(n, 8)])
        if err > 3e-5:
            raise RuntimeError(f"NumPy inference and PyTorch disagree for {year}: {err:.3g}")
        weights = extract_numpy_weights(model)
        bank[year] = weights
        weight_map = dict(zip(WEIGHT_KEYS, weights, strict=True))
        np.savez_compressed(output / f"gru_checkpoint_{year}.npz", **weight_map)
        record = {"year": year, "last_complete_label": str(completed[eligible].max().date()),
                  "training_examples": n, "epochs_this_year": epochs, "learning_rate": lr,
                  "train_smooth_l1": final_loss, "torch_vs_numpy_max_abs_error": err,
                  "eligible_for_inference_from": year + 1}
        provenance.append(record)
        print(f"{year}: {n} fully observed examples; loss={final_loss:.6f}; "
              f"gate-check={err:.2e}; deploy no earlier than {year+1}", flush=True)

    (output / "gru_bank_generated.py").write_text(python_bank_text(bank), encoding="utf-8")
    manifest = {
        "status": "NEW_DEV_TRAINING_RUN",
        "warning": "This is a new documented training run, not recovery of historical weights. Cross-platform exact equality is not guaranteed.",
        "bank_sha256_float32_le": bank_digest(bank),
        "data_sha256": {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in (args.price_file or root / "data/prices/daily_adjusted.parquet", args.market_file or root / "data/prices/market_series.parquet")},
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "dataset": "Course QQQ/SPY/^VIX adjusted daily prices; DEV only, 2016 lookback to 2021-12-31",
        "labels": "Future five trading sessions, min/max/last adjusted QQQ close return / 0.08",
        "inputs": "20 days x 6 features, fixed normalization copied from frozen student_timing.py",
        "architecture": {"input_size": 6, "hidden_size": 8, "num_layers": 1,
                         "head_size": 3, "gate_order": "reset, update, new"},
        "optimization": {"seed": args.seed, "epochs_initial": args.epochs_initial,
                         "epochs_update": args.epochs_update, "lr_initial": args.lr_initial,
                         "lr_update": args.lr_update, "weight_decay": args.weight_decay,
                         "loss": "SmoothL1Loss(beta=0.25)", "torch_version": torch.__version__},
        "checkpoints": provenance,
    }
    (output / "gru_training_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    print("Output directory:", output)
    if args.embed_timing is not None:
        embed_bank(args.embed_timing, bank)
    else:
        print("Timing source has NOT been modified.")
    if args.verify_timing is not None:
        verify_bank(args.verify_timing, bank)
    return output



def bank_digest(bank):
    return hashlib.sha256(b"".join(np.asarray(a, dtype="<f4").tobytes()
                                  for y in sorted(bank) for a in bank[y])).hexdigest()


def read_embedded_bank(path):
    # Parse literal array arguments, rather than executing the strategy.
    tree = ast.parse(path.read_text(encoding="utf-8"))
    node = next(n.value for n in tree.body if isinstance(n, ast.Assign)
                and any(isinstance(x, ast.Name) and x.id == "_GRU_BANK" for x in n.targets))
    return {ast.literal_eval(k): tuple(np.asarray(ast.literal_eval(a.args[0]), dtype=np.float64)
                                     for a in v.elts) for k, v in zip(node.keys, node.values)}


def verify_bank(path, bank):
    reference = read_embedded_bank(path)
    if set(reference) != set(bank):
        raise ValueError("Checkpoint year sets differ")
    exact = close = True
    for y in sorted(bank):
        if len(reference[y]) != len(bank[y]):
            raise ValueError("Tensor counts differ")
        for name, expected, actual in zip(WEIGHT_KEYS, reference[y], bank[y], strict=True):
            if expected.shape != actual.shape:
                raise ValueError(f"Shape mismatch: {y} {name}")
            diff = float(np.max(np.abs(expected - actual)))
            same = np.array_equal(expected, actual)
            near = np.allclose(expected, actual, atol=1e-6, rtol=0)
            exact &= same
            close &= near
            print(f"{y} {name:16s} exact={same} max_diff={diff:.3e}")
    print("Embedded SHA256:", bank_digest(reference))
    print("Trained  SHA256:", bank_digest(bank))
    print("EXACT PASS" if exact else "TOLERANCE PASS (not bitwise identical)" if close else "FAIL")
    if not close:
        raise ValueError("Retrained bank differs; do not silently replace frozen weights")


def embed_bank(path, bank):
    if set(bank) != set(YEARS):
        raise ValueError("Embedding requires all five checkpoints")
    source = path.read_text(encoding="utf-8")
    new_bank = python_bank_text(bank).split("_GRU_BANK = ", 1)[1].strip()
    source, count = re.subn(r"(?ms)^_GRU_BANK = \{.*?^\}", lambda _: "_GRU_BANK = " + new_bank, source, count=1)
    if count != 1:
        raise ValueError("Could not locate embedded GRU bank")
    source = re.sub(r"(?m)^GRU_BANK_SHA256 = .*$", "GRU_BANK_SHA256 = " + repr(bank_digest(bank)), source)
    ast.parse(source)
    path.write_text(source, encoding="utf-8")
    print("Explicitly updated timing copy:", path)



def q_digest(bank):
    return hashlib.sha256(b''.join(np.asarray(bank[y], dtype='<f8').tobytes() for y in sorted(bank))).hexdigest()


def train_q(root, gru_bank, out):
    """Generate course selection-only NAV internally, never consume stale output."""
    import os
    import importlib
    from datetime import date
    root = root.resolve()
    os.chdir(root)
    sys.path.insert(0, str(root))
    from selection.config import load
    from selection.scores import MonthlyScores
    from portfolio.backtester import BacktestConfig, run_backtest
    from shared.data_view import DataBundle, DataView
    m = importlib.import_module('timing.student_timing')
    m._GRU_BANK = gru_bank
    _, options, params, prices, _, universe = load('selection/configs/growth.json')
    # Official selection baseline, no timing, valuation or stops.
    cfg = BacktestConfig(start=date(2017,1,3), end=date(2021,12,31))
    result = run_backtest(prices, MonthlyScores(), params, universe, cfg,
                          global_unified=options.inference_mode == 'global-unified')
    nav = result.equity.sort_index()
    nav.index = pd.to_datetime(nav.index)
    nav.to_csv(out / 'q_training_selection_nav.csv')
    bundle = DataBundle.load(prices=prices)
    dates, _, rules, regimes, states = m._build_weekly(DataView(bundle, DEV_END))
    rows = []
    # Each feature row is reconstructed on its own decision date. The next
    # completed decision week supplies a realized close-to-close reward proxy.
    for i in range(1, len(dates)-1):
        d, end = dates[i], dates[i+1]
        if d < pd.Timestamp('2018-01-01') or end > DEV_END or d not in nav.index:
            continue
        path = nav.loc[d:end].to_numpy(float)
        if len(path) < 2 or not np.isfinite(path).all() or path[0] <= 0: continue
        r = path[-1]/path[0]-1
        dd = max(0., float(np.max(1-path/np.maximum.accumulate(path))))
        view = DataView(bundle, d)
        f = m._build_rule_features(view)
        if f is None: continue
        b = float(m._apply_shock_cap(m._rule_exposure(f),f))
        shock = m._shock_regime(f)
        mode = int(regimes[i])
        if shock == 'panic': mode = m.PANIC
        elif shock == 'stress': mode = m.STRESS
        elif mode in (m.PANIC,m.STRESS): mode=m.NORMAL
        prev = float(rules[i-1])
        allowed = m._permission_gate(f, shock, states, i, mode)[mode]
        levels = np.unique(np.r_[b,np.clip(b+allowed,0,1)])
        if shock=='panic': levels=np.minimum(levels,m.PANIC_EXPOSURE_CAP)
        elif shock=='stress': levels=np.minimum(levels,m.STRESS_EXPOSURE_CAP)
        for e in np.unique(levels):
            if abs(e-b)<1e-10: continue
            # Same utility units as direct-Q threshold: percentage points.
            reward = 100*((e-b)*(r+0.08*max(r,0)) - 0.22*(e*e-b*b)*dd
                        - 0.0006*(abs(e-prev)-abs(b-prev)) - 0.00065*abs(e-b))
            x=m._direct_q_features(states[i],b,prev,e,mode)
            rows.append((d,end,x,reward))
    bank={}; manifest=[]
    for y in range(2018,2022):
        cutoff=pd.Timestamp(y,12,31)
        eligible=[r for r in rows if r[1]<=cutoff]
        if not eligible: raise ValueError(f'No completed Q observations for {y}')
        X=np.stack([r[2] for r in eligible]); Y=np.array([r[3] for r in eligible])
        beta=np.linalg.solve(X.T@X+m.DIRECT_Q_RIDGE*np.eye(X.shape[1]),X.T@Y)
        bank[y]=beta
        manifest.append(dict(year=y, decisions=len({r[0] for r in eligible}), action_rows=len(Y),
                             last_complete_label=str(max(r[1] for r in eligible).date()),
                             rmse=float(np.sqrt(np.mean((X@beta-Y)**2))), deploy_from=y+1))
        print('Q',manifest[-1],flush=True)
    return bank,manifest


def read_q(path):
    tree=ast.parse(path.read_text(encoding='utf-8'))
    node=next(n.value for n in tree.body if isinstance(n,ast.Assign) and any(isinstance(x,ast.Name) and x.id=='_DIRECT_Q_BANK' for x in n.targets))
    return {ast.literal_eval(k):np.asarray(ast.literal_eval(v.args[0]),float) for k,v in zip(node.keys,node.values)}


def main():
    args=arguments(); root=args.course_root.resolve(); target=root/'timing/student_timing.py'
    if args.last_year!=2021 or args.first_year!=2017: raise ValueError('Unified training requires all years')
    if args.price_file or args.market_file: raise ValueError("Unified training uses only official data paths")
    if args.embed_timing or args.verify_timing: raise ValueError('Use default to embed both banks, or --verify to verify both')
    args.course_root=root; args.out=(args.out or root/'output/timing_training_dev').resolve()
    out=train(args)
    gru={y:tuple(np.load(out/f'gru_checkpoint_{y}.npz')[k] for k in WEIGHT_KEYS) for y in YEARS}
    q,records=train_q(root,gru,out)
    np.savez(out/'direct_q_bank.npz',**{str(y):v for y,v in q.items()})
    if args.verify:
        verify_bank(target,gru)
        old=read_q(target)
        if set(old)!=set(q): raise ValueError('Q year mismatch')
        diff=max(float(np.max(np.abs(old[y]-q[y]))) for y in q)
        print('Q maximum coefficient error:',diff)
        if diff>1e-8: raise ValueError('Q verification failed')
        print('Q TOLERANCE PASS')
    else:
        # Build both banks in a temporary copy before replacing the source.
        tmp=target.with_suffix('.training_tmp.py')
        tmp.write_text(target.read_text(encoding='utf-8'),encoding='utf-8')
        embed_bank(tmp,gru)
        s=tmp.read_text(encoding='utf-8')
        literal='{'+', '.join(f'{y}: np.asarray({v.tolist()!r}, dtype=np.float64)' for y,v in q.items())+'}'
        s,n=re.subn(r'(?m)^_DIRECT_Q_BANK = .*$',lambda _: '_DIRECT_Q_BANK = '+literal,s)
        if n!=1: raise ValueError('Missing Q bank placeholder')
        s=re.sub(r'(?m)^DIRECT_Q_BANK_SHA256 = .*$',lambda _: 'DIRECT_Q_BANK_SHA256 = '+repr(q_digest(q)),s)
        ast.parse(s); tmp.write_text(s,encoding='utf-8'); tmp.replace(target)
    manifest={'gru_sha256':bank_digest(gru),'q_sha256':q_digest(q),'q_ridge':24.,'q_reward':'selection-only NAV close-to-close counterfactual; upside 0.08, downside 0.22, cost 0.0006, residual 0.00065','q_checkpoints':records,'gru_manifest':'gru_training_manifest.json','training_end':'2021-12-31','numpy':np.__version__,'pandas':pd.__version__,'torch':torch.__version__}
    (out/'training_manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
    print('Unified training finished:',out)

if __name__=='__main__':
    main()
