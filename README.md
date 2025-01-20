# Ten Day One Percent Algorithmic Trading Bot
Implementation of a trading strategy that leverages a pre-trained LightGBM machine learning model. The algorithm is meant to be used with QuantConnect's LEAN API.

## Model Features
The model predicts whether a stock will stay within a 1% bound over the next 10 days with an 80% success rate.
- Combines changepoint detection and NN-based trend features.
- Hyperparameters fine-tuned using Optuna integration with LightGBM.
- Ensures trading success is robust over days, not just opportunities.
