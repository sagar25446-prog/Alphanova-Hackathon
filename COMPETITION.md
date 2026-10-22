# Biweekly Competition: Cross-Sectional Signal Forecasting — Season 1

> **These are the rules for biweekly SEASON 1.** Different files will be issued for
> future seasons; later seasons may differ — the target, the parameters, and the
> prize structure can all change between seasons.

## Format: seasons of biweekly contests

The Biweekly Competition runs as a **season of five biweekly contests**. Each biweekly is its own
contest with its own leaderboard and its own prize pot:

| | |
|---|---|
| **Submission window** | 2 weeks (boundaries land on the 1st and the 15th) |
| **Live scoring window** | 1 month from the moment submissions close |
| **Submissions per contest** | up to **20 per person**, reset every cycle |
| **Settlement** | at the end of the live window — one month after close, as the cycle after next is wrapping up |
| **Season** | 5 consecutive biweeklies under one target, then a 1-week pause |

Because each live window runs a month, **two of them overlap at any time**: while
cycle N is being scored live, cycle N+1 is already accepting submissions.

### The legacy pot — why each cycle needs new alpha

When a cycle closes, **every submission that passed the entry checks moves into the
season's legacy pot**, regardless of how well it scored. Pot signals are no longer
prize-eligible, but they keep occupying their neighbourhood in signal space for the
rest of the season.

The practical consequence: **resubmitting a previous cycle's entry blocks itself**, as
it correlates ~1.0 with its own copy in the pot. Small perturbations of it fare barely
better. Every cycle rewards signals that are new relative to everything already
submitted this season — including your own earlier work.

At the season boundary the pot is cleared, the fund folds a selection of the season's
best signals into its own book, and the next season starts against a correspondingly
harder target.

## Prize

Each biweekly has its own pot, determined by the number of **quality signals** $Q$ that
cycle admits:

$$\text{Prize}(Q) = \begin{cases} 0 & \text{if } Q = 0 \\ 100 + 2400 \cdot q^{0.75} & \text{if } Q > 0 \end{cases}, \qquad q = \min(Q/13,\ 1)$$

The pot ranges from **\$0** (no quality signals) to **\$2,500 USD** per cycle at
saturation (13 quality signals). Payment is by stablecoin or bank transfer.

**Unspent budget rolls forward.** Each cycle has a prize budget of **\$2,500**; the
formula above sets what the cycle earns on its own. Whatever part of the budget is
not paid out in a cycle — because fewer quality signals were admitted, or fewer than
three places existed — **rolls into the pot of the season's next contest**. A cycle
that follows underspent ones can therefore pay out more than its own formula pot.

Note there is **no user-count term** — unlike comp5, the pot depends only on the alpha
actually found, not on how many people registered.

### Prize split within a cycle

The cycle's pot goes to the **top 3 signals in that cycle's quality set**, ranked by
Sharpe — the same split comp5 used:

| Place | Share |
|-------|-------|
| 1st   | 60%   |
| 2nd   | 25%   |
| 3rd   | 15%   |

Winners are drawn from the quality set, so you must be **both high-Sharpe and
sufficiently uncorrelated** with every higher-ranked winner. A signal that correlates
with a better one never enters the quality set and cannot win, however high its Sharpe.

**If a cycle admits fewer than three quality signals, only the stated shares are
paid to the places that exist**: with one quality signal the winner receives 60% of
the pot; with two, the top two receive 60% and 25%. Shares of places that do not
exist are not paid out in that cycle; the unpaid remainder rolls into the pot of
the season's next contest (see the budget note above).

### Quality signal selection

$Q$ is decided by a **greedy selection**:

1. Signals with a positive Sharpe are ranked by Sharpe.
2. Going down the list, a signal joins the quality set only if its time-averaged
   cross-sectional correlation with **every** signal already occupying space does
   not exceed **$K = 0.5$** in absolute value. This is comp5's threshold, carried
   over unchanged.
3. "Already occupying space" means: signals admitted earlier in the same ranking, and
   the season's legacy pot.

So a signal counts toward $Q$ — and grows the pot for everyone — only if it brings
genuinely new information, not a marginal variation on something already submitted.

---

## Objective

At each timestamp $i$ produce a cross-sectional signal $P(i) = (P_1(i), \ldots, P_J(i))$
over $J = 20$ assets, **cross-sectionally de-meaned**:

$$\sum_{j=1}^{J} P_j(i) = 0$$

You are scored on the Sharpe ratio of the portfolio return, **net of transaction cost**:

$$r(i) = \langle P(i-1),\ X(i) \rangle - c \cdot \lVert P(i-1) - P(i-2) \rVert_1, \qquad c = 5\ \text{bp}$$

$$U = \frac{\mathbb{E}[r(i)]}{\sqrt{\mathrm{Var}[r(i)]}}$$

Here $X(i) = (X_1(i), \ldots, X_J(i))$ is the vector of **AlphaNova proprietary
returns** at time $i$ — the shipped `returns` panel described in "What you are
predicting" below.

Official scoring may include additional execution-cost adjustments; official and
local numbers can differ systematically.

The cost term is new in the Biweekly Competition and it is not decorative. On the
shipped example, rebalancing every hour costs about **6% of the strategy's PnL
volatility** — i.e. it drags Sharpe down by ~0.06, which is larger than most edges you
will find on this target. Smoothing positions over time cuts that to ~1% and is usually
worth more than a slightly better forecast. The demo notebook measures both.

## What you are predicting — read this twice

**The `returns` panel shipped with this competition is not raw price returns.** It is
a proprietary return series constructed by us; how it is constructed is deliberately
not disclosed, like the feature identities. This panel is what the backtest multiplies
your signal against, so it is the only "returns" that matter in this competition —
replicating it from outside data will not work and is not the game.

The **target** you fit is an **unknown target**: a forward-looking series built from
the returns panel, cross-sectionally z-scored and clipped at ±5. Its horizon and
exact construction are not disclosed. Treat it as unknown and let the data tell you
what predicts it.

Three things follow, and they are the whole character of this competition:

1. **The obvious first ideas score ~0.** The signals a quant reaches for first carry
   almost no edge on this target — not because they are wrong, but because the target
   is constructed so that they are not rewarded. (The demo notebook shows this
   happening to a textbook signal.)
2. **The bar is real.** Simple, marginal transformations of the shipped features
   carry little edge against this target. Expect to need structure: interactions,
   nonlinearities, cross-sectional relationships.
3. **Early rows of a month's panel can be exactly 0.0.** The construction needs a
   warm-up window of history before its first value. A flat stretch at the start of a
   panel is expected, not a data bug.

---

## Competition Data

| File | What it is |
|---|---|
| `data/train/{NNN}.small.features.parquet` | 6 features × 20 assets, MultiIndex `(feature, ticker)` |
| `data/train/{NNN}.small.returns.parquet` | the hourly return series (see above) — what the backtest multiplies your signal against |
| `data/train/{NNN}.small.target.parquet` | the forward target, xs-z-scored, clipped ±5 (horizon and construction not disclosed) |
| `data/train/gauge/{NNN}.*.npy` | precomputed gauge matrices for `--gauge-fix` |
| `signal_db/signal_cities_ambient_coords.parquet` | city coordinates of existing signals |

The six feature columns are anonymised on purpose: what each one measures, and how
each is computed, is not disclosed — in this package or anywhere else. They are
features, nothing more is stated. Treat them as `Feature.1`–`Feature.6` and let the
data tell you what they do.

Each period file is **cumulative**: period `NNN` contains all history up to that point,
so there is no need to concatenate periods (doing so duplicates early months heavily).

**Ticker identities are not stable across periods.** `ticker.1` in period 001 is
unrelated to `ticker.1` in period 002 — each period is independently rotated. Your model
must learn cross-sectional structure, never per-ticker habits. Scoring is
invariant under the anonymization, so nothing you need for a good signal is lost.

### The data you get ends before the data you are scored on

Your package contains a fixed set of training periods. Internal scoring runs over a
longer history — including a portion you never receive — plus the live window of your
cycle. That hidden band is deliberate: it is the part of your score that cannot be
reproduced, tuned against, or reverse-engineered offline. Expect your local
validation numbers to differ from the leaderboard.

---

## Signal Cities and Novelty

Compressed to unit norm and gauge-fixed, a signal traces a path on a high-dimensional
sphere; the time-average of that path is a single point — the signal's **city**. It is a
compact summary of what your signal does on average.

- Cities **far apart** ⇒ the signals are very likely uncorrelated.
- Cities **close together** ⇒ they *may* be correlated (closeness is necessary but not
  sufficient).

Because the pot rewards the number of mutually uncorrelated signals, a city far from all
others is your best chance of being admitted. **Aim for at least 60° from the nearest
existing city.**

`runner.py your_submission.py --gauge-fix` computes your city and reports the distance
to the nearest one in the shipped database. An **hourly refreshed** map of all season
cities is published on the contest page, so you can steer away from occupied
territory instead of discovering the collision after submitting.

**City coordinates are for offline navigation only.** They describe other signals, not
future returns; feeding them into your predictor will not help and may get your
submission flagged.

---

## Entry checks and the overfitting gate

Every submission runs through automated checks. Failures are reported as:

| Result | What went wrong | How to fix |
|--------|-----------------|------------|
| **PEEK_THE_FUTURE** | Data leakage — a future-looking operation such as `shift(-n)`, `rolling(center=True)`, or `bfill`/`backfill`. Those are examples, not the list: the reject set is broader, is not published, and is extended as new forms appear. Any transform whose output at hour *t* can depend on hours after *t* — centered kernels, zero-phase filters, wrap-arounds, in any library — is a violation, and ambiguous forms are reported as warnings. Passing the scan is not a certification: the no-look-ahead rule applies however a transform is expressed, and forms the scan cannot see are still violations — screening also continues after acceptance. | Remove every forward-looking transform — the examples here and any centered or zero-phase equivalent — from anywhere in the file, including inside `train()` and in comments. Make each feature causal by construction: at hour *t*, use only hours ≤ *t* (backward differences, trailing windows). The check is a line-by-line scan of the whole submission as the runner executes it, not of `predict()` alone — comments included, notebook magic and `!shell` lines excluded, because the runner strips those before your code runs. |
| **CODE_OUTSIDE_CLASS** | A top-level `def`, a non-Predictor class, or any other module-level statement that makes a call — bare or assigned, in any spelling, `with`, `for`, `try` — unless the call instantiates your Predictor subclass or is one of the permitted configuration calls. A statement with no call in it (`SEED = 42`, `X += 1`) only warns and does not fail. Passing this check is not a certification either: the rule is that no work happens at import time and no logic lives outside the class, however it is written, and forms the check cannot see are still violations. | Move helper functions and non-Predictor classes inside your subclass. `data = pd.read_parquet(...)` and `MODEL = build_model()` fail because what they call is not your Predictor subclass; `predictor = MyPredictor()` is fine. The full list of what may stay at module level is under Submission Format below. |
| **NOT_DEMEANED** | `predict()` output is not cross-sectionally de-meaned. Apply `pred.sub(pred.mean(axis=1), axis=0)` before returning. | Apply the de-mean before returning. The default `python runner.py your_submission.py` prints a `Cross-sectional de-mean check: PASS`/`FAIL` line, so you can catch this before submitting — note that neither `--full` nor `--gauge-fix` runs that check. |
| **TRAINING_TOO_SLOW** | One of two triggers. Either `train()` finished but took longer than 4 minutes — the elapsed time is compared with the limit once the call returns — or your model's combined cost per retrain (constructor + `train()` + `predict()`), multiplied across the overfitting gate's repeated retraining, is projected to exceed the gate's time ceiling; in that case the rejection happens before the gate runs, and — like every failed submission — it does not count against your cycle's 20-submission cap (only pending and accepted submissions do). | Cut boosting rounds/epochs, features, or training rows, and avoid recomputation, so `train()` finishes inside 4 minutes on CPU. For the gate trigger, the whole retrain loop must stay affordable: keep `__init__` light (it runs on every retrain) and reduce the combined constructor + train + predict cost — staying well inside the per-call limits keeps the gate comfortable. |
| **PREDICTION_TOO_SLOW** | `predict()` finished, but took longer than 60 seconds for a call — measured the same way as training. | Move heavy work into `train()` and keep `predict()` light — no refitting, no expensive per-call computation. |
| **OVERFITTING_FAILED** | Your signal did not beat its own shuffled-label null (see below). | Two causes look identical here. If you may be memorizing, cut complexity and regularize (shallower trees, stronger L2, fewer rounds) and prefer engineered, generalizable features. If the signal is simply weak, the answer is a stronger signal — see below. |
| **CANT_RUN** | Load failure, exception during train/predict, or another runtime error. A call cut off by the sandbox's absolute timeout lands here too (see Computation limits). | Reproduce with `python runner.py your_submission.py`, confirm your subclass implements `train()`/`predict()`, and declare extra dependencies via [PEP 723](https://peps.python.org/pep-0723/). Running out of memory also lands here — keep peak usage under 8 GB. |

### What the overfitting gate actually does

We retrain your model many times on data whose labels have been shuffled, so that no
predictable structure remains, and compare your real in-sample result against that null
distribution. A model that scores well on shuffled labels is memorizing, and is rejected.

Because the gate retrains your model repeatedly, its runtime is your `train()` +
`predict()` cost multiplied many times over. The gate's time budget scales with your
model's own measured cost up to an operator ceiling — a model that is inside the
per-call limits but too expensive for repeated retraining to finish within that
ceiling is rejected as TRAINING_TOO_SLOW before the gate runs; like every
failed submission, it does not count against your cycle's submission cap. Staying well inside the per-call limits keeps the gate
comfortable.

Because the gate retrains on variants whose time rows are permuted, your code —
`train()` and `predict()` alike — must tolerate data it would never see in a
real run. An `assert` about row order, about any quantity you derive from it
(autocorrelation, rolling statistics, trends, the ranking of computed
features), or about the target's values can fire on a shuffled retrain: the
features are permuted row-wise and the target is recomputed from the permuted
data, so its distribution differs from the real batch. A run in which any
single retrain crashes is refused rather than scored, because a data-dependent
crash could bias the test. What does stay fixed is the column layout: every
batch in a run carries the same columns, in the same order, as the unshuffled
one, so checks on column names are safe. Row counts are not: a batch is the
training slice of one particular month, so any length your local runs happen to
produce says nothing about what arrives here. Handle surprising values
tolerantly instead of asserting them away — for a sort you rely on, add a
stable tiebreaker rather than asserting that ties cannot happen. And note that
the local runner does not shuffle, so a clean local run cannot tell you whether
your code survives the gate.

**Be aware that this gate also rejects honest signals that are simply weak.** On a
target this hard a near-zero-edge model cannot beat its own null. This is a
property of a genuinely hard target, not a fault in your code or in our pipeline. If
you are rejected here, the answer is a stronger signal, not a bug report.

Passing the gate is a floor, not a score. Your ranking comes from Sharpe over the full
scored history plus your cycle's live window.

**The live window is a verification period, not a scoring period.** You are not ranked
on the live month by itself — it enters your Sharpe at its natural weight (about 2% of
the scored hours), and its real job is different: a signal built by de-obfuscating the
data gives itself away when its live behaviour diverges from its historical record.
Honest signals have nothing to do here but wait.

---

## Submission Format

A single **`.py` file** (or a notebook) containing your Predictor:

```python
from predictor import Predictor

class MyPredictor(Predictor):
    def train(self, features, target): ...
    def predict(self, features): ...   # must be cross-sectionally de-meaned
```

- Inherit from `Predictor` in `predictor.py`; do not modify the provided `.py` files.
- **All logic inside the class.** No top-level helper functions — including one
  written as `helper = lambda: ...`, which is a helper function however it is
  spelled — and no module-level statement that makes a call. These may stay at
  module level:
  - imports, a docstring or PEP 723 header, the Predictor subclass
  - `predictor = MyPredictor()` — any *other* call in that statement must itself be
    a permitted configuration call
  - **configuration calls**: `warnings.filterwarnings`/`simplefilter`,
    `logging.basicConfig`/`getLogger`, `pd.set_option`, `np.seterr` — written dotted,
    via `from … import …`, or as up to two chained method calls on the result of one
  - an **import guard**: `try: import x` / `except ImportError:` — raise a builtin
    exception (quoting the error with `str`/`repr`/`format` is fine), bind a
    call-free fallback, or import a fallback module
  - `if __name__ == "__main__":` — either operand order, and **no** `else:` (its
    `else` runs when your file is imported for scoring)
- Extra dependencies via [PEP 723](https://peps.python.org/pep-0723/) at the top of the
  file.

Pre-installed: `numpy`, `pandas`, `scikit-learn`, `xgboost`, `lightgbm`, `pyarrow`.

### Local validation

```bash
python runner.py your_submission.py             # quick: latest period, holdout split
python runner.py your_submission.py --full      # full walk-forward across all periods
python runner.py your_submission.py --gauge-fix # full + city / novelty metrics
```

This catches **NOT_DEMEANED** and **CANT_RUN** before you submit. Leakage and code
structure are checked server-side; the overfitting gate runs after submission.

---

## Computation limits

- **No GPU.** `train()` must finish within **4 minutes** on a standard CPU.
- `predict()` must finish within **60 seconds** per call.
- Memory must stay under 8 GB.
- A method that exceeds its limit is normally allowed to finish; its elapsed time is then compared with the limit and the submission is rejected with the matching timing verdict. The sandbox also puts an absolute budget on each validation phase, only minutes above the per-method limits — a run that overshoots its limit by minutes rather than seconds hits that budget first, is cut off before returning, and reports **CANT_RUN** instead of a timing verdict. There is no partial credit for nearly making it.

---

## Timeline

**Season 1, Cycle 1 opens 15 August 2026, 00:00 UTC** (decided 2026-08-10).

| | |
|---|---|
| Submission windows | **per cycle — the contest page is authoritative** (each window runs boundary-to-boundary on the 1st and 15th, UTC) |
| Live scoring | the month following each cycle's close |
| Prizes settle | shortly after the live window closes |

Subsequent cycles open on the 1st and the 15th of each month; the season runs
for five cycles, then pauses for a week before the next season begins under a
new target.

## Important

- Do not modify any provided `.py` files or base-class signatures.
- All code must live inside your Predictor subclass.
- One account per person. Multi-account operations are detected and disqualified.
- Submissions with implausibly high validation Sharpe may be flagged for data
  reverse-engineering and disqualified.
