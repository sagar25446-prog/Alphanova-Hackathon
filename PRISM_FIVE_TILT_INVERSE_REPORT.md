# Tactical inverse experiment

The submitted `Prism Five Tilt` received server test score **−0.0167266** and
server IC **−0.0027**. Its fixed inverse has identical model complexity and
turnover; only exposure orientation is reversed.

For each timestamp, if the original gross return is `g` and cost is `c`, the
original net is `g-c` while the inverse is `-g-c`. This is not guaranteed to
be profitable, but it is a direct, auditable A/B response to the observed
server result.

Local latest validation (expectedly opposite to the original): −0.1207. This
does not invalidate the server-feedback experiment; it means local and server
regimes disagree, so no hidden-test guarantee is possible.

Run locally:

```powershell
$env:PYTHONPATH = "$PWD\.thirdparty"
$py = 'C:\Users\Sagar Jain\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe'
& $py runner.py alphanova_prism_five_tilt_inverse.py
```

Use the website's novelty check. If it reports Not novel, onboarding anyway can
still improve the leaderboard score, but the signal is unlikely to qualify for
the prize because absolute correlation with its neighbour is unchanged.
