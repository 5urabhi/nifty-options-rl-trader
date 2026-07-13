# Experimental Reinforcement Learning Environment for Simulated Nifty Options Trading

> Status: Experimental environment; simulation and reward design under revision.
>
> This repository is an experimental reinforcement-learning research environment built primarily on synthetically generated option prices. It is intended to study environment design, feature engineering, reward construction, and policy evaluation. It is not evidence of live trading performance and should not be interpreted as an investment strategy.

## Overview

This project implements an experimental end-to-end prototype for generating simulated options data, engineering features, training a PPO agent, and evaluating actions in a custom trading environment.

The current repository is notebook-centric and centers on [new_test.ipynb](/C:/Users/akkil/Documents/nifty-options-rl-trader/new_test.ipynb), which:

- loads hourly Nifty spot data from `nifty_hourly_data_2020_to_today.csv`
- generates synthetic CE/PE option prices
- computes technical features and Black-Scholes-style Greeks
- builds a custom `SimpleNiftyCallEnv`
- trains a PPO agent with Stable-Baselines3
- exports action logs and backtest-style outputs

## What I Built

I designed and implemented the data pipeline, synthetic pricing workflow, feature engineering steps, custom trading environment, PPO training loop, evaluation notebook, and project documentation.

## Synthetic Data Warning

Option prices in the current version are simulated from Nifty spot data rather than sourced from historical option-chain records. Consequently, the environment is suitable for software and RL experimentation but not for drawing conclusions about real-world profitability.

## Research Pipeline

```text
Nifty Spot Data
   ->
Synthetic Option Price Generation
   ->
Feature Engineering and Greeks
   ->
Custom RL Environment
   ->
PPO Training
   ->
Action Export
   ->
Exploratory Backtest Metrics
```

## Repository Structure

```text
.
|-- README.md
`-- new_test.ipynb
```

The notebook reads and writes several intermediate artifacts during execution, including:

- `nifty_hourly_data_2020_to_today.csv`
- `simulated_nifty_options.csv`
- `nifty_rl_ready.csv`
- `ppo_ce_actions.csv`
- `ppo_simple_ce.zip`

These files are referenced by the notebook but are not currently committed in this repository snapshot.

## Environment Summary

| Component | Description |
| --- | --- |
| Observation space | Market features, lagged values, position flag, and normalized PnL state |
| Action space | `0 = Hold`, `1 = Buy CE`, `2 = Sell CE` |
| Position model | Single-position call-option environment |
| Reward | Currently includes a fixed positive reward on entry and realized PnL on exit |
| Episode | Chronological sequence over the prepared dataset |
| Costs | Brokerage, STT, SEBI charges, GST |
| Data | Synthetic options derived from Nifty spot data |
| Agent | PPO with `MlpPolicy` via Stable-Baselines3 |

## Synthetic Options Model

Option prices in the current notebook are simulated from Nifty spot data rather than historical option-chain records. The simulator approximates:

- intrinsic value from the spot-close price
- time decay using simplified theta-style decay
- volatility variation via Gaussian noise
- expiry handling and behavioral premiums

It does not fully reproduce:

- real implied-volatility surfaces
- bid-ask spreads
- strike-dependent liquidity
- volatility skew and smile
- discrete expiry effects
- market microstructure
- open interest and volume
- path-dependent volatility dynamics
- actual executable prices

## Feature Engineering

The notebook derives features including:

- technical indicators such as `SMA`, `RSI`, and momentum
- Black-Scholes-style `Delta`, `Vega`, and `Theta`
- time features such as hour and day-of-week
- lagged spot and synthetic option-price observations

## Reward Design

The initial environment uses a fixed positive reward when opening a position and realized PnL when closing it. This design can bias the policy toward unnecessary entry actions and does not align rewards consistently with risk-adjusted profitability.

A stronger reward design would be based on incremental marked-to-market portfolio-value change after costs, with optional penalties for turnover, drawdown, and excessive exposure. That revision is not yet implemented in the current notebook.

## Preliminary Simulation Results

Initial experiments produced unstable and economically implausible performance metrics due to synthetic option-price outliers, simplified execution assumptions, and reward-design limitations. Those outputs should be treated as debugging evidence rather than valid strategy performance.

The current focus is on correcting the environment, validating synthetic price generation, and establishing leakage-aware chronological evaluation before reporting strategy-level returns.

## Historical Debug Output - Not Validated

The screenshot below is retained as a visual record of an earlier exploratory run. It should not be interpreted as validated evidence of trading performance.

### Backtest Performance Summary (Historical Debug Output)

These metrics come from an earlier exploratory notebook run on synthetic option prices. They are retained for transparency and debugging context only.

| Metric | Value |
| --- | --- |
| Total Return | +400.24% |
| Final Portfolio Value | Rs500,244.86 |
| Max Drawdown | 1.40% |
| Sharpe Ratio | 4.54 |
| Sortino Ratio | 23.20 |
| Win Rate | 95.8% |
| Profit Factor | 88.3 |
| Avg Trade Duration | ~6 min |
| Best Trade | 369,060% |
| Worst Trade | -99.77% |

The presence of extreme values, especially the best-trade figure, is one reason these results are not treated as production-valid.

![Exploratory backtest output](https://github.com/user-attachments/assets/caa1beec-9fe8-4816-a6cd-a0b22cdfd756)

## Evaluation Protocol

The current notebook should be treated as exploratory in-sample analysis rather than a fully validated trading study.

A valid evaluation for this kind of financial time-series problem should use chronological splits:

- training period
- validation period for reward and hyperparameter selection
- fully held-out test period
- optional walk-forward retraining

Feature scalers and normalizers should be fit only on training data, and all observations should be constructed without future information. The current notebook does not yet present a strict train/validation/test protocol in the README, so live-trading conclusions should not be drawn from the current results.

## Baselines

The RL policy should eventually be compared against simple reference strategies under identical transaction-cost assumptions, such as:

- no-trade policy
- random policy
- simple momentum or RSI-based policy
- buy-and-hold call exposure
- spot-index benchmark

These benchmark comparisons are not yet documented as implemented results in this repository.

## Known Issues

- Synthetic option prices do not represent executable market quotes.
- Extreme simulated price values can distort PnL and risk metrics.
- The original reward function may incentivize unnecessary entries.
- The current environment supports limited position management.
- The model does not yet incorporate bid-ask spreads, liquidity, or market impact.
- Reported historical simulation metrics are not considered production-valid.
- Real option-chain validation remains future work.

## Limitations and Research Risks

- The repository relies on synthetic option pricing rather than historical option-chain data.
- Reported backtest-style metrics may be heavily influenced by simulator assumptions and price outliers.
- The current workflow is centered on a single notebook, which limits modularity and testability.
- The environment currently uses OpenAI Gym compatibility warnings under Stable-Baselines3 rather than a migrated Gymnasium-native implementation.
- Evaluation logs in the notebook show repeated identical mean rewards, which warrants further debugging of reward behavior and validation setup.
- A chronological, leakage-resistant validation protocol is not yet clearly established end to end.

## Reproducibility

Recommended environment:

- Python 3.10 or newer
- Jupyter Notebook

Install the libraries currently referenced by the notebook:

```bash
python -m venv .venv
.venv\Scripts\activate
pip install pandas numpy scipy gym stable-baselines3 vectorbt tqdm jupyter
jupyter notebook new_test.ipynb
```

Notes:

- No pinned `requirements.txt` is included yet.
- Runtime will depend on dataset size and PPO training length.
- The notebook writes intermediate CSVs and a saved PPO model during execution.
- No random-seed policy is documented in the current README.

## Tests

There is no dedicated automated test suite in the current repository snapshot.

Future tests should cover:

- look-ahead prevention
- reward calculation
- transaction-cost deduction
- feature alignment
- environment transitions
- terminal-state behavior

## Technology Stack

Python, Pandas, NumPy, SciPy, Gym, Stable-Baselines3, PPO, VectorBT, tqdm, Jupyter Notebook

## Future Work

- replace synthetic prices with historical option-chain data or a more defensible simulator
- redesign the reward around incremental portfolio-value changes after costs
- add strict chronological train, validation, and test splits
- migrate the environment to Gymnasium-native interfaces
- separate notebook logic into modular source files and add automated tests

## License

MIT License

## Risk Disclaimer

This project is for educational and research purposes only and does not constitute investment advice.
