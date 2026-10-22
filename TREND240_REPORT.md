# Trend240 Ridge candidate

## Selection evidence

The candidate was selected from a fixed local screen after purging the final
120 labelled rows of each training prefix. It uses six same-row
cross-sectional rank trends, where each trend is the current rank minus its
causal 240-row exponential average. A ridge fit maps those trends to the
training target, and positions are causally smoothed over 120 rows.

## Local results

| metric | value |
|---|---:|
| full net Sharpe | +0.0098 |
| late net Sharpe (020--031) | +0.0206 |
| mean IC | +0.0061 |
| positive folds | 16 / 31 |
| fold Sharpe std | 0.0505 |
| mean train time | about 1.2 s |
| mean predict time | under 0.1 s |
| city novelty (current map) | 60.06 degrees |

The nonlinear trend expansion reached +0.0103 locally but had lower IC and
more degrees of freedom, so it was not selected.

## Safety checks

- cross-sectional demeaning: PASS
- no negative shifts, centred windows, backfills, or external data
- final 120 target rows excluded from fitting
- overlap prediction is supported
- no module-level computation beyond imports

Local numbers are not a promise of hidden-test performance. Submit one copy,
record its server test score, and compare it with the current best before any
further change.
