# Prism Five Tilt — red-team result

Offline research used only the supplied competition package and the current
public city-coordinate download. The city file was used for comparison only,
never by the predictor.

## Local full walk-forward

| Metric | Previous Prism | Prism Five Tilt |
|---|---:|---:|
| Net Sharpe | +0.022317 | **+0.023431** |
| Gross Sharpe | +0.032616 | **+0.034143** |
| Mean IC | +0.027160 | **+0.029564** |
| Mean L1 turnover | 0.135627 | 0.150371 |
| Positive folds | 25 / 31 | 25 / 31 |
| Fold Sharpe std. | 0.033877 | 0.034857 |
| Mean train time | 0.935 s | **0.730 s** |
| Mean predict time | 0.025 s | **0.020 s** |
| Local current-map city distance | stale-map 36.6° | current-map **34.6°** |

The improvement is modest and turnover rises by about 0.0147. It is not a
guarantee of hidden-test performance.

## Robustness checks

- Latest runner: validation Sharpe +0.0722; cross-sectional de-mean PASS.
- Finite output under normal, shuffled-row/shuffled-label, zero-label, and
  overlapping inference calls.
- Repeat prediction is deterministic.
- Maximum observed row-mean magnitude: below `3.1e-17`.
- No negative shifts, centered windows, backfills, external I/O, or ticker
  identity features.

## Decision

`alphanova_prism_five_tilt.py` is the strongest tested replacement, but it must
be run through the website's current novelty check before onboarding. If the
website reports **Not novel**, discard it; if it reports **Novel**, onboarding
is reasonable. Do not use the obsolete release city database to override the
website result.
