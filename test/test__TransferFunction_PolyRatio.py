"""Test transfer function model parameters estimation."""
import numpy as np
import unittest
from dynm.dynamic_model import BayesianDynamicModel
from dynm.utils.format_input import compute_lagged_values
from copy import copy

# Simulating the data
nobs = 500
sd_y = 0.02
gamma_1 = 2.5
gamma_2 = 1.5
lambda_1 = 0.8
lambda_2 = -0.5

y = np.zeros(nobs)
x = np.zeros(nobs)
E_1 = np.zeros(nobs)
E_2 = np.zeros(nobs)

# First observation
np.random.seed(1111)
for t in range(1, nobs):
    # Random errors
    nu = np.random.normal(loc=0, scale=sd_y, size=1)
    x[t] = np.random.normal(loc=0, scale=1, size=1)

    # Evolution
    E_1[t] = (
        lambda_1 * E_1[t - 1] +
        lambda_2 * E_2[t - 1] +
        gamma_1 * x[t] +
        gamma_2 * x[t-1])
    E_2[t] = E_1[t-1]

    # Observation
    y[t] = E_1[t] + nu

# Estimation
m0 = np.array([0, 0, 0, 0, 0, 0])
C0 = np.identity(6)
W = np.identity(6)

np.fill_diagonal(C0, val=[9, .001, 5, 5, 1, 1])
np.fill_diagonal(W, val=[0, 0, 0, 0, 0])

x = compute_lagged_values(X=x.reshape(-1, 1), lags=2)
X = {"transfer_function": x}


class TestTransferFunctionPolyRatio(unittest.TestCase):
    """Tests BayesianDynamicModel results for Transfer Function Model."""

    def test__estimates_known_W_and_V(self):
        """Test parameters estimation with know W and V."""
        model_dict = {
            "transfer_function": {
                "m0": m0,
                "C0": C0,
                "gamma_order": 2,
                "lambda_order": 2,
                "ntfm": 1,
                "W": W
            }
        }

        # Fit
        mod = BayesianDynamicModel(model_dict=model_dict, V=sd_y**2)\
            .fit(y=y, X=X)
        m = mod.m

        self.assertTrue(np.abs(m[2] - lambda_1) < .1)
        self.assertTrue(np.abs(m[3] - lambda_2) < .1)
        self.assertTrue(np.abs(m[4] - gamma_1) < .1)
        self.assertTrue(np.abs(m[5] - gamma_2) < .1)

    def test__estimates_discount(self):
        """Test parameters estimation with discount."""
        model_dict = {
            "transfer_function": {
                "m0": m0,
                "C0": C0,
                "gamma_order": 2,
                "lambda_order": 2,
                "ntfm": 1,
                "discount": np.array([1, 1, 1, 1, 1, 1])
            }
        }

        # Fit
        mod = BayesianDynamicModel(model_dict=model_dict).fit(y=y, X=X)
        m = mod.m

        self.assertTrue(np.abs(m[2] - lambda_1) < .1)
        self.assertTrue(np.abs(m[3] - lambda_2) < .1)
        self.assertTrue(np.abs(m[4] - gamma_1) < .1)
        self.assertTrue(np.abs(m[5] - gamma_2) < .1)

    def test__BayesianDynamicModel_with_nan(self):
        """Test parameters estimation with nan in y."""
        model_dict = {
            "transfer_function": {
                "m0": m0,
                "C0": C0,
                "gamma_order": 2,
                "lambda_order": 2,
                "ntfm": 1,
                "discount": np.array([1, 1, 1, 1, 1, 1])
            }
        }

        copy_y = copy(y)
        copy_y[50] = np.nan

        # Fit
        mod = BayesianDynamicModel(model_dict=model_dict)
        fit_results = mod.fit(y=copy_y, X=X)

        forecast_df = fit_results.dict_filter.get("predictive")
        m = mod.m

        self.assertTrue(np.abs(m[2] - lambda_1) < .2)
        self.assertTrue(np.abs(m[3] - lambda_2) < .2)
        self.assertTrue(forecast_df.f.notnull().all())

    def test__predict_calc_predictive_mean_and_var_performance(self):
        """Test k steps a head performance."""
        model_dict = {
            "transfer_function": {
                "m0": m0,
                "C0": C0,
                "gamma_order": 2,
                "lambda_order": 2,
                "ntfm": 1,
                "discount": np.array([1, 1, 1, 1, 1, 1])
            }
        }

        # Insample and outsample sets
        tr__y = y[:450]
        te__y = y[450:]

        tr__X = {"transfer_function": x[:450, :, :]}
        te__X = {"transfer_function": x[450:, :, :]}

        # Fit
        mod = BayesianDynamicModel(model_dict=model_dict)\
            .fit(y=tr__y, X=tr__X)

        # Forecasting
        forecast_results = mod.predict(k=50, X=te__X)
        forecast_df = forecast_results.get("predictive")
        parameters_df = forecast_results.get("parameters")

        mape = np.mean(np.abs(forecast_df.f - te__y) / te__y)

        self.assertTrue(mape < 1)
        self.assertTrue(len(parameters_df) == 300)
        self.assertTrue(forecast_df.notnull().all().all())
        self.assertTrue(parameters_df.notnull().all().all())

    def test__smoothed_posterior_variance(self):
        """Test smooth posterior variance."""
        model_dict = {
            "transfer_function": {
                "m0": m0,
                "C0": C0,
                "gamma_order": 2,
                "lambda_order": 2,
                "ntfm": 1,
                "discount": np.array([1, 1, 1, 1, 1, 1])
            }
        }

        # Fit
        mod = BayesianDynamicModel(model_dict=model_dict)\
            .fit(y=y, X=X, smooth=True)
        smooth_posterior = mod.dict_smooth.get("posterior")

        min_var = smooth_posterior.variance.min()
        self.assertTrue(min_var >= 0.0)

    def test__smoothed_predictive_variance(self):
        """Test smooth predictive variance."""
        model_dict = {
            "transfer_function": {
                "m0": m0,
                "C0": C0,
                "gamma_order": 2,
                "lambda_order": 2,
                "ntfm": 1,
                "discount": np.array([1, 1, 1, 1, 1, 1])
            }
        }

        # Fit
        mod = BayesianDynamicModel(model_dict=model_dict)\
            .fit(y=y, X=X, smooth=True)
        smooth_predictive = mod.dict_smooth.get("predictive")

        min_var = smooth_predictive.q.min()
        self.assertTrue(min_var >= 0.0)

    def test__smoothed_predictive_errors(self):
        """Test smooth predictive mape."""
        model_dict = {
            "transfer_function": {
                "m0": m0,
                "C0": C0,
                "gamma_order": 2,
                "lambda_order": 2,
                "ntfm": 1,
                "discount": np.array([1, 1, 1, 1, 1, 1])
            }
        }

        # Fit
        mod = BayesianDynamicModel(model_dict=model_dict)\
            .fit(y=y, X=X, smooth=True)

        filter_predictive = mod\
            .dict_filter.get("predictive")\
            .sort_values("t")

        smooth_predictive = mod\
            .dict_smooth.get("predictive")\
            .sort_values("t")

        f = filter_predictive.f.values
        fk = smooth_predictive.f.values

        mse1 = np.mean((f-y)**2)
        mse2 = np.mean((fk-y)**2)

        self.assertTrue(mse2/mse1 <= 1.0)
