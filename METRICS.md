# What we measure

This document describes the per-signal quantities that show up on
your Submissions dashboard and on the public leaderboards. Each
quantity is computed from your model's predictions on the historical
walk-forward window and updated nightly.

Throughout, $i$ indexes timestamps and $j$ indexes the $J$ tickers in
the universe ($J = 20$ for the `small` world). Your prediction at
time $i$ is the cross-sectionally de-meaned vector
$P(i) = (P_1(i), \ldots, P_J(i))$ with $\sum_j P_j(i) = 0$. The
realized one-period return for ticker $j$ at time $i$ is $X_j(i)$
— in the Biweekly Competition this is the **AlphaNova proprietary returns**
series shipped as the `returns` panel, not a raw price return
(see `COMPETITION.md`).

## Sharpe

Sharpe is the headline grade. It's the risk-adjusted profitability
of trading your signal as a portfolio.

The portfolio return at time $i$ is the lagged inner product of your
signal with realized returns,

$$
r(i) = \underbrace{\langle P(i-1),\, X(i) \rangle}_{\text{gross}}
     \; - \; \underbrace{c \sum_{j=1}^{J} \lvert P_j(i-1) - P_j(i-2) \rvert}_{\text{transaction cost}},
\qquad c = 5\ \text{bp},
$$

new in the Biweekly Competition: every rebalance is charged. Rebalancing hourly
costs roughly 6% of a strategy's PnL volatility — a Sharpe drag of
~0.06, larger than most edges available on this target — while a
smoothed version of the same signal pays ~1%. Turnover control is
therefore a first-class lever, often larger than improving the
forecast.

The Sharpe ratio is

$$
U \;=\; \frac{\mathbb{E}_i[r(i)]}{\sqrt{\mathrm{Var}_i[r(i)]}}.
$$

Aggregated over the full walk-forward window. Higher is better;
zero means no edge after risk normalization. Negative means the
signal is anti-predictive — flipping its sign would help.

Official scoring may include additional execution-cost adjustments
beyond the 5 bp charge; official and local numbers can differ
systematically.

## IC (information coefficient)

IC measures how well your signal *correlates* with realized returns
cross-sectionally, before any portfolio construction. It's the
geometric counterpart to Sharpe: Sharpe asks "does the strategy
make money?", IC asks "does the prediction match the relative
ranking of the assets?"

At each time $i$, define the cosine similarity between your de-meaned
prediction and the de-meaned **forward target** — the same vector the
gauge's first axis is aligned to (your IC is measured against the target's
own horizon, not the next single period; the horizon is not disclosed).
Writing $T(i)$ for that target — the `target` column your model trains on,
not the one-period $X(i)$ from the Sharpe formula above:

$$
\rho(i) \;=\; \frac{\langle P(i-1),\, T(i) \rangle}{\|P(i-1)\|\,\|T(i)\|}.
$$

After the gauge transform (which puts the target direction along
the first axis $e_0$ of the lower-dimensional space), $\rho(i)$
shows up as the first component of your signal's gauge-fixed
trajectory $\tilde{s}(i)$:

$$
\tilde{s}_0(i) \;=\; \rho(i).
$$

The IC reported on the dashboard is the time-average,

$$
\mathrm{IC} \;=\; \mathbb{E}_i[\rho(i)] \;=\; \mathbb{E}_i[\tilde{s}_0(i)].
$$

Bounded in $[-1, 1]$. Realistic cross-sectional momentum signals
sit in the $0.005$–$0.05$ range. Sign matters; flip if negative.

## IC dispersion correlation

Some predictors are good at all times; others are good only when
the cross-section is "moving" (high dispersion of returns). This
metric tells you which.

Let $\sigma(i) = \mathrm{std}_j[X_j(i+h)]$ be the cross-sectional
standard deviation of forward returns at time $i$ (with $h$ the
target horizon). Then:

$$
\text{IC dispersion corr} \;=\; \mathrm{corr}_i\!\left[\,\rho(i),\, \sigma(i+h)\,\right].
$$

Positive: your signal's IC is *higher* when the market is
dispersing — useful in volatile regimes, less in flat ones.
Negative: opposite. Near zero: your signal performs uniformly
across regimes.

Useful for portfolio construction: signals with offsetting
dispersion correlations diversify against each other better than
signals that all spike on the same days.

## Concentration

After the gauge transform, your signal at each time is a unit
vector $\tilde{s}(i)$ on the sphere $S^{J-2}$ (so $S^{18}$ for
$J=20$). The time-average $B$ of these unit vectors is *not* itself
on the sphere — its norm sits between 0 and 1:

$$
B \;=\; \mathbb{E}_i[\tilde{s}(i)],
\qquad
\text{concentration} \;=\; \|B\| \;\in\; [0,\,1].
$$

- $\|B\| = 1$ means $\tilde{s}(i)$ is constant — the signal points
  in the same direction at every timestamp.
- $\|B\| = 0$ means $\tilde{s}(i)$ averages to the origin — the
  signal points in completely uncorrelated directions over time.

In practice good predictors have concentration in the $0.1$–$0.5$
range. Very high concentration (close to $1$) often means the
signal is essentially a constant cross-sectional pattern repeated
each timestep, which can be a sign of insufficient feature use.

## Compression loss

Compression loss is the gap between the time-average IC and the
*city's* projection onto the target direction. It quantifies how
much information about the target gets "compressed away" when you
collapse the time-varying signal trajectory $\tilde{s}(i)$ into a
single direction (the city).

Define the city — the unit-normalized time-average — as

$$
\mathrm{city} \;=\; \frac{B}{\|B\|} \;\in\; S^{J-2}.
$$

Then the compression loss is

$$
\text{compression loss} \;=\; \mathrm{IC} - \mathrm{city}_0,
$$

i.e. the IC minus the city's first component (its projection on
the target axis $e_0$).

A useful equivalent form, valid after the gauge transform:

$$
\text{compression loss} \;=\; \mathrm{IC} \cdot \left(1 - \frac{1}{\|B\|}\right).
$$

Both are the same number; the second form makes the geometry clear:
- $\mathrm{IC} = B_0$ (the first component of the un-normalized
  average $B$),
- $\mathrm{city}_0 = B_0 / \|B\| = \mathrm{IC} / \|B\|$,
- so $\mathrm{IC} - \mathrm{city}_0 = \mathrm{IC}(1 - 1/\|B\|)$.

Since $\|B\| \le 1$, the term $(1 - 1/\|B\|)$ is non-positive.
Compression loss is therefore zero only when $\|B\| = 1$ (signal
is perfectly stable in direction); otherwise it has the opposite
sign of IC.

What it tells you:
- *Magnitude* tracks how much your signal's direction varies over
  time. A noisier trajectory (lower $\|B\|$) yields a city that
  sits further from the IC direction relative to its time-average,
  inflating $|\text{compression loss}|$.
- *Sign* is **opposite** to IC's: the $1 - 1/\|B\|$ multiplier is
  non-positive (since $\|B\| \le 1$), so the product flips the sign of
  the $\mathrm{IC}\cdot$ term. Profitable signals (positive IC) have
  negative compression loss; anti-predictive ones (negative IC) have
  positive compression loss — the examples were right, the word
  "mirrors" was not.

If you're trying to improve a signal: high $|\text{compression loss}|$
relative to $|\mathrm{IC}|$ indicates the signal has temporal
structure that's "lost" in the city projection — either smooth this
out (more averaging, longer windows) or capture it explicitly (regime
features, time-varying weights).

## City novelty (degrees)

The city is a single unit-vector summary of where your signal
"lives on average" in the gauge-fixed space. Two signals with
nearby cities behave similarly across the full history; two with
far-apart cities are doing structurally different things.

City novelty is the angular distance (in degrees) from your
signal's city to the *nearest* published competing city:

$$
\text{city novelty} \;=\; \min_{c \in C \setminus \{\mathrm{city}\}} \arccos(\langle \mathrm{city},\, c\rangle) \cdot \frac{180}{\pi}.
$$

A high value means your signal is doing something new. Used by the
prize allocation as a proxy for diversity (see
[COMPETITION.md](COMPETITION.md), "Quality signal selection"): two
signals with cities within $60°$ of each other are very likely
correlated, so only the higher-Sharpe one will count toward the
quality set.

Aim for at least $60°$ from the nearest existing city.

## Global novelty (degrees)

City novelty uses only the *time-average* of your signal — it
doesn't see the temporal structure. Global novelty does. It's the
angular distance based on the full pairwise PnL correlation across
the gauge-fixed time series, converted to an angle:

$$
\text{global novelty} \;=\; \min_{s \neq \text{self}} \arccos(\hat\rho(\text{self}, s)) \cdot \frac{180}{\pi},
$$

where $\hat\rho(s_1, s_2)$ is the average cross-sectional
correlation between two signals' gauge-fixed trajectories. Captures
"different *behavior* over time" rather than just "different
average direction."

A signal can have low city novelty (similar average direction to
another signal) but high global novelty (their *fluctuations*
around the average are uncorrelated), or vice versa. Both numbers
matter for diversification.

## Summary: which to optimize for

For getting on the leaderboard at all:
- **Sharpe** is the ranking criterion, and it is **net of the 5 bp
  cost**. Check the gross-vs-net gap early: if they differ a lot, you
  are paying for churn, and smoothing wins more than tuning.
- **IC** going positive is necessary for Sharpe to be positive on
  any reasonable strategy.
- Remember the target's construction is **not disclosed**, and the
  obvious first signals score near zero on it by construction — spend
  your iterations on structure the easy ideas do not capture.

For winning the prize pool:
- **City novelty** $> 60°$ keeps your signal in the quality set.
- **Global novelty** $> 60°$ adds diversification value once
  you're in the set.

For diagnosing a weak signal:
- $|\mathrm{IC}|$ small or near zero → you have no signal. Add
  features.
- $|\mathrm{IC}|$ decent but Sharpe low → portfolio construction
  issues; tighten cross-sectional de-meaning.
- $|\text{compression loss}|$ large relative to $|\mathrm{IC}|$ →
  high temporal variation; consider adding regime / time-varying
  features.
- IC dispersion corr near $\pm 1$ → your signal only fires in one
  market regime; broaden it or pair with a complementary one.
