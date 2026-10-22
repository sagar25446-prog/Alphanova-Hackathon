# AlphaNova red-team report

## Scope and controls

This audit used only the supplied local competition package and its official
runner. No hidden evaluation data, external market data, ticker identities,
internet calls, or third-party submission code were used.

The accepted submission is `alphanova_causal_prism.py`. Its only retained
change from the audited baseline is a ridge penalty of `0.01` rather than
`0.10` for the low-capacity state gate.

## Rule and robustness audit

- Module structure contains only a docstring, imports, and one `Predictor`
  subclass. All computation lives in that class.
- The code has no forward shifts, centered windows, backfills, external I/O,
  dynamic execution, or assertions.
- It uses only same-timestamp cross-sectional ranks/statistics, a model fit on
  the training target, and a causal recursive smoother.
- It does not learn asset identities across periods; ticker labels are used
  solely to preserve the supplied current-column order.
- With permuted training rows and with all-zero labels, it trains and predicts
  finite 20-column output. The maximum absolute row mean was below `1.4e-17`.
- Overlapping inference is finite and repeat calls are deterministic.
- The normal official runner passed the cross-sectional de-mean test. Latest
  local validation Sharpe was `+0.0729`; training was about 1 second and
  prediction below 0.1 seconds, comfortably within limits.

## Predeclared walk-forward protocol

Candidates were selected only on periods 001-019 and confirmed once on the
untouched periods 020-031. The selection grid was intentionally small:

1. Ridge penalty: `0.01`, `0.10`, `1.00`.
2. Conditional on the ridge winner, regime lookback: `5,000`, `10,000`,
   `20,000` rows.

The ridge results were:

| Ridge | Early net Sharpe | Late net Sharpe | Late positive folds |
|---:|---:|---:|---:|
| 0.01 | +0.0251 | +0.0184 | 9 / 12 |
| 0.10 | +0.0247 | +0.0182 | 9 / 12 |
| 1.00 | +0.0227 | not advanced | - |

The `10,000`-row lookback won the early selection at `+0.0251`; `5,000` and
`20,000` scored `+0.0203` and `+0.0165`. No larger search was conducted.

## Full official-runner comparison

| Metric | Baseline ridge 0.10 | Accepted ridge 0.01 |
|---|---:|---:|
| Net walk-forward Sharpe | +0.022017 | **+0.022317** |
| Gross Sharpe | +0.032314 | **+0.032616** |
| Mean IC | +0.026819 | **+0.027160** |
| Mean L1 turnover | 0.135957 | **0.135627** |
| Mean 5 bp cost | 0.00006798 | **0.00006781** |
| Positive folds | 25 / 31 | 25 / 31 |
| Fold-Sharpe standard deviation | 0.033970 | **0.033877** |
| Mean train time | 0.8913 s | 0.9346 s |
| Mean predict time | 0.0251 s | 0.0253 s |
| City novelty proxy | 35.7 degrees | **36.6 degrees** |

The difference is modest, so it is evidence of a local incremental improvement,
not a promise of hidden or live performance.

## Rejected alternatives

- Much slower smoothing improved an early block but deteriorated materially on
  the untouched late folds.
- Higher regime strength and a regime-weighted smoothing seed both failed late
  confirmation.
- A higher Feature-6 interaction increased city novelty to about 50 degrees,
  but reduced net Sharpe and stability.
- Formula-only variants near 60-degree city novelty were materially weaker
  (best approximately +0.0141 net Sharpe). They are not replacements for the
  accepted signal.

## Submission recommendation

Submit only `alphanova_causal_prism.py`, after adding any participant metadata
that the contest form explicitly requires. Do not submit near-clone probes: the
rules state that passing submissions enter the legacy pot and can block future
signals. Local city novelty is only a proxy; the organiser alone calculates the
full trajectory-correlation quality criterion and hidden/live score.

## Reproduce the final local checks

From this directory in PowerShell:

```powershell
$env:PYTHONPATH = "$PWD\.thirdparty"
$env:MPLCONFIGDIR = "$PWD\.mpl-config"
$env:PYTHONUTF8 = '1'
$env:COMP6_DISABLE_PIP_INSTALL = '1'
$py = 'C:\Users\Sagar Jain\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe'
& $py runner.py alphanova_causal_prism.py --full --gauge-fix --output final_results.csv
& $py report_candidate.py
```
