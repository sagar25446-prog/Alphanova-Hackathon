# AlphaNova Quant Challenge — Cross-Sectional Signal Forecasting

[![Python 3.12](https://img.shields.io/badge/python-3.12-blue.svg)](https://www.python.org/downloads/)
[![Leaderboard](https://img.shields.io/badge/Global%20Leaderboard-Top%2035%20(Rank%2031%20Unique)-success.svg)](#competition-results)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

This repository contains the complete quantitative research, reverse-engineering pipelines, model iterations, and the final submission codebase for the **AlphaNova Quant Competition (Season 1)** — a bi-weekly cross-sectional signal forecasting challenge on obfuscated financial market data.

Following a **54th rank** finish in **QuantStorm**, I was invited by Dominik Keller (CEO of AlphaNova) to compete against a field comprising Kaggle Grandmasters, WorldQuant consultants, and Numerai champions.

---

## 🏆 Competition Results & Key Takeaways

- **Final Leaderboard Placement**: **Top 35 Globally** (**Rank 31** among unique competitors after deduplicating multi-submissions).
- **Core Signal Performance**: Achieved targeted portfolio concentration of **`0.2562`** (matching the theoretical asymptote of the top-ranked models) with controlled turnover under strict 5 bps transaction costs.

---

## 🔬 The Quantitative Challenge

The competition evaluated models on an undisclosed, cross-sectional target with three deliberate roadblocks:

1. **Feature Column Permutation**: The evaluation server dynamically shuffles feature column names across validation and test holdouts to prevent hardcoding column indexes.
2. **Strict Anti-Overfitting Gate**: A server-side gate trains submitted architectures on row-permuted variants. High-capacity machine learning models (e.g. standard XGBoost, LightGBM) fail the overfitting check or experience severe out-of-sample degradation.
3. **Severe 5 bps Transaction Cost**: High portfolio turnover rapidly destroys strategy Sharpe. Discrete cutoffs (e.g. going long top 2, short bottom 2) suffer catastrophic turnover due to boundary "flip-flopping".

---

## 💡 Key Engineering & Quantitative Breakthroughs

### 1. Dynamic Feature Mapping via Correlation Fingerprinting
Because the test server shuffles feature names dynamically, static indexing fails. Our solution computes cross-sectional correlation matrices on-the-fly (`_csrank` correlation):
- Features $F_1 \dots F_4$ form a collinear core ($r \approx 0.95$).
- Feature $F_5$ exhibits moderate correlation ($r \approx 0.33$).
- Feature $F_6$ is quasi-orthogonal ($r \approx 0.04$).

The predictor dynamically identifies and maps the underlying features at inference time regardless of server permutations.

### 2. Feature-Level Temporal Smoothing (Span = 240)
Rather than smoothing signals after nonlinear combinations, features are smoothed at the source using an Exponential Weighted Moving Average (EWMA):
$$\tilde{F}_{t} = \alpha F_t + (1 - \alpha) \tilde{F}_{t-1}, \quad \text{span} = 240$$
This eliminates high-frequency noise before non-linear transformation.

### 3. Closed-Form Robust Signal Formulation
Machine learning models fitted directly to the target struggled with the platform's anti-overfitting gate. Extensive screening revealed a structural cross-sectional formula:
$$\text{Signal} = -(\text{Core}^3) + \text{Core} \times F_6$$
This closed-form formulation delivers stable cross-sectional Information Coefficient (IC) with zero risk of overfitting-gate rejection.

### 4. Continuous Power Functions vs. Discrete Thresholds
Discrete portfolio selection (e.g. top $k$ / bottom $k$) mathematically yields $0.25$ concentration on a 20-asset cross-section, but boundary jitter causes excessive turnover under a 5 bps fee structure.

To resolve this, we proved and implemented continuous power transformations over uniform cross-sectional ranks:
$$\text{Ranked} = \text{CSRank}(\text{Signal}) \in [-1, 1]$$
$$\text{Weight} = \text{sign}(\text{Ranked}) \cdot |\text{Ranked}|^p, \quad p = 9$$

- **Power 4**: Produces $\approx 0.137$ concentration (matching Rank 3 on the leaderboard).
- **Power 9**: Yields **$0.2562$ concentration**, smoothly squashing intermediate assets toward zero while weighting extreme tails without boundary discontinuities.

---

## 📁 Repository Structure

```
├── submission.py                 # Final submittable model (UltimateGrandmasterPredictor)
├── submission_xgb.py             # XGBoost dynamic baseline
├── predictor.py                  # Base Predictor interface
├── runner.py                     # Local walk-forward evaluation harness
├── walkforward.py                # Walk-forward validation engine
├── data_loader.py                # Dataset loader and period partitioning
├── evaluation.py                 # Metrics computation (Sharpe, IC, Concentration)
├── city_tools.py                 # Novelty coordinate calculations
├── COMPETITION.md                # Official competition specification & rules
├── METRICS.md                    # Explanation of reported metrics
├── README_COMPETITION.md         # Original benchmark guide
│
├── alphanova_*.py                # Explored model architectures (Ridge, Causal Prism, etc.)
├── research/
│   └── scratch_experiments/      # 90+ simulation scripts, power sweeps, and diagnostics
│
├── results.csv                   # Historical validation results across iterations
├── RED_TEAM_REPORT.md            # Adversarial and stress-testing report
└── TREND240_REPORT.md            # Temporal lookback validation study
```

---

## 🚀 Quick Start

### 1. Installation
Clone the repository and install dependencies:

```bash
git clone https://github.com/sagar25446-prog/Alphanova-Hackathon.git
cd Alphanova-Hackathon
pip install -r requirements.txt
```

### 2. Run Local Walk-Forward Evaluation
Evaluate the final submission model using the competition runner:

```bash
python runner.py submission.py --full
```

Check novelty against prior signals:

```bash
python runner.py submission.py --gauge-fix
```

---

## 🛠️ Tech Stack & Methods
- **Language**: Python 3.12
- **Core Libraries**: NumPy, Pandas, Scikit-learn, XGBoost, SciPy
- **Techniques**: Cross-Sectional Ranking (`_csrank`), Exponential Smoothing (EWMA), Correlation Matrix Fingerprinting, Continuous Power Transformation, Walk-Forward Purged Cross-Validation.

---

## 📄 License
This project is open-source under the [MIT License](LICENSE).
