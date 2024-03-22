"""Test dlm model parameters estimation."""
import numpy as np
import unittest
from dynm.dynamic_model import BayesianDynamicModel

# Simulating the data
nobs = 80
sd_y = 0.02

y = np.zeros(nobs)
x = np.zeros([nobs, 2])

# Initial information
y[0] = 10
beta0 = 10
beta1 = 2
beta2 = 3

# First observation
np.random.seed(1111)
for t in range(1, nobs):
    # Random errors
    nu = np.random.normal(loc=0, scale=sd_y, size=1)

    # Input
    x[t, 0] = np.random.normal(loc=0, scale=1, size=1)
    x[t, 1] = np.random.normal(loc=0, scale=1, size=1)

    # Observation
    y[t] = beta0 + beta1 * x[t, 0] + beta2 * x[t, 1] + nu

X = {"regression": x}


class TestPolynomialandRegression(unittest.TestCase):
    """Tests BayesianDynamicModel results for Dynamic Linear Model."""

    def test__estimates_known_W_and_V(self):
        """Test parameters estimation with know W and V."""
        model_dict = {
            "polynomial": {
                "m0": np.array([10]),
                "C0": np.array([[9]]),
                "ntrend": 1,
                "W": np.array([[0]]),
            },
            "regression": {
                "m0": np.array([0, 0]),
                "C0": np.identity(2) * 9,
                "nregn": 2,
                "W": np.identity(2) * 0
            }
        }

        # Fit
        mod = BayesianDynamicModel(model_dict=model_dict, V=sd_y**2)\
            .fit(y=y, X=X)
        forecast_df = mod.dict_filter.get("predictive")

        mape = np.mean(np.abs(forecast_df.f - forecast_df.y) / forecast_df.y)

        self.assertTrue(mape < .05)

    def test__estimates_discount(self):
        """Test parameters estimation with discount."""
        model_dict = {
            "polynomial": {
                "m0": np.array([10]),
                "C0": np.array([[9]]),
                "ntrend": 1,
                "discount": 1,
            },
            "regression": {
                "m0": np.array([0, 0]),
                "C0": np.identity(2) * 9,
                "nregn": 2,
                "discount": 1
            }
        }

        # Fit
        mod = BayesianDynamicModel(model_dict=model_dict, V=sd_y**2)\
            .fit(y=y, X=X)
        forecast_df = mod.dict_filter.get("predictive")

        mape = np.mean(np.abs(forecast_df.f - forecast_df.y) / forecast_df.y)

        self.assertTrue(mape < .05)

    def test__predict_calc_predictive_mean_and_var_performance(self):
        """Test k steps a head performance."""
        model_dict = {
            "polynomial": {
                "m0": np.array([10]),
                "C0": np.array([[9]]),
                "ntrend": 1,
                "discount": 1,
            },
            "regression": {
                "m0": np.array([0, 0]),
                "C0": np.identity(2) * 9,
                "nregn": 2,
                "discount": 1
            }
        }

        # Insample and outsample sets
        tr__y = y[:60]
        te__y = y[60:]

        tr__X = {"regression": x[:60, :]}
        te__X = {"regression": x[60:, :]}

        # Fit
        mod = BayesianDynamicModel(model_dict=model_dict).fit(y=tr__y, X=tr__X)

        # Forecasting
        forecast_results = mod.predict(k=20, X=te__X)
        forecast_df = forecast_results.get("predictive")
        mape = np.mean(np.abs(forecast_df.f - te__y) / te__y)

        self.assertTrue(mape < .05)

    def test__smoothed_posterior_variance(self):
        """Test smooth posterior variance."""
        model_dict = {
            "polynomial": {
                "m0": np.array([10]),
                "C0": np.array([[9]]),
                "ntrend": 1,
                "discount": 1,
            },
            "regression": {
                "m0": np.array([0, 0]),
                "C0": np.identity(2) * 9,
                "nregn": 2,
                "discount": 1
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
            "polynomial": {
                "m0": np.array([10]),
                "C0": np.array([[9]]),
                "ntrend": 1,
                "discount": 1,
            },
            "regression": {
                "m0": np.array([0, 0]),
                "C0": np.identity(2) * 9,
                "nregn": 2,
                "discount": 1
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
            "polynomial": {
                "m0": np.array([10]),
                "C0": np.array([[9]]),
                "ntrend": 1,
                "discount": 1,
            },
            "regression": {
                "m0": np.array([0, 0]),
                "C0": np.identity(2) * 9,
                "nregn": 2,
                "discount": 1
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

    def test__invalid_model_dict_ntrend_missing(self):
        """Test incorrect model dict missing ntrend argument."""
        model_dict = {
            "polynomial": {
                "m0": np.array([10]),
                "C0": np.array([[9]]),
                "discount": 1,
            }
        }

        missing_keys = ['ntrend']
        error_message = ("Missing elements in polynomial model: " +
                         str(missing_keys))

        with self.assertRaises(ValueError) as context:
            BayesianDynamicModel(model_dict=model_dict)

        actual_error_message = str(context.exception)
        self.assertEqual(actual_error_message.strip(), error_message.strip())

    def test__invalid_model_dict_ntrend_discount_W_missing(self):
        """Test incorrect model dict missing ntrend, discount, W arguments."""
        model_dict = {
            "polynomial": {
                "m0": np.array([10]),
                "C0": np.array([[9]]),
            }
        }

        missing_keys = ['W', 'discount', 'ntrend']
        error_message = ("Missing elements in polynomial model: " +
                         str(missing_keys))

        with self.assertRaises(ValueError) as context:
            BayesianDynamicModel(model_dict=model_dict)

        actual_error_message = str(context.exception)
        self.assertEqual(actual_error_message.strip(), error_message.strip())

    def test__invalid_model_dict_nregn_discount_W_missing(self):
        """Test incorrect model dict missing nregn, discount, W arguments."""
        model_dict = {
            "polynomial": {
                "m0": np.array([10]),
                "C0": np.array([[9]]),
                "ntrend": 1,
                "discount": 1,
            },
            "regression": {
                "m0": np.array([0, 0]),
                "C0": np.identity(2) * 9
            }
        }

        missing_keys = ['W', 'discount', 'nregn']
        error_message = ("Missing elements in regression model: " +
                         str(missing_keys))

        with self.assertRaises(ValueError) as context:
            BayesianDynamicModel(model_dict=model_dict)

        actual_error_message = str(context.exception)
        self.assertEqual(actual_error_message.strip(), error_message.strip())

    def test__invalid_model_dict_discount_shape_polynomial(self):
        """Test discount shape in polynomial."""
        model_dict = {
            "polynomial": {
                "m0": np.array([10]),
                "C0": np.array([[9]]),
                "ntrend": 1,
                "discount": np.array([1, 2]),
            },
            "regression": {
                "m0": np.array([0, 0]),
                "C0": np.identity(2) * 9,
                "nregn": 2,
                "discount": 1
            }
        }

        error_message = "Discount for polynomial model is not a scalar"

        with self.assertRaises(ValueError) as context:
            BayesianDynamicModel(model_dict=model_dict)

        actual_error_message = str(context.exception)
        self.assertEqual(actual_error_message.strip(), error_message.strip())

    def test__invalid_model_dict_discount_shape_regression(self):
        """Test discount shape for regression model."""
        model_dict = {
            "polynomial": {
                "m0": np.array([10]),
                "C0": np.array([[9]]),
                "ntrend": 1,
                "discount": 1
            },
            "regression": {
                "m0": np.array([0, 0]),
                "C0": np.identity(2) * 9,
                "nregn": 2,
                "discount": np.array([1, 2]),
            }
        }

        error_message = "Discount for regression model is not a scalar"

        with self.assertRaises(ValueError) as context:
            BayesianDynamicModel(model_dict=model_dict)

        actual_error_message = str(context.exception)
        self.assertEqual(actual_error_message.strip(), error_message.strip())

    def test__invalid_model_dict_discount_values_regression(self):
        """Test discount values in regression model."""
        model_dict = {
            "polynomial": {
                "m0": np.array([10]),
                "C0": np.array([[9]]),
                "ntrend": 1,
                "discount": 1
            },
            "regression": {
                "m0": np.array([0, 0]),
                "C0": np.identity(2) * 9,
                "nregn": 2,
                "discount": 2,
            }
        }

        error_message = ("Discount for regression model" +
                         " is not in [0,1] interval")

        with self.assertRaises(ValueError) as context:
            BayesianDynamicModel(model_dict=model_dict)

        actual_error_message = str(context.exception)
        self.assertEqual(actual_error_message.strip(), error_message.strip())

    def test__with_missing_input(self):
        """Test a regression model fit with missing input."""
        model_dict = {
            "polynomial": {
                "m0": np.array([10]),
                "C0": np.array([[9]]),
                "ntrend": 1,
                "discount": 1,
            },
            "regression": {
                "m0": np.array([0, 0]),
                "C0": np.identity(2) * 9,
                "nregn": 2,
                "discount": 1
            }
        }

        error_message = ("The input X for regression is None or " +
                         "has an incompatible shape with the " +
                         "declared nregn.")

        with self.assertRaises(ValueError) as context:
            BayesianDynamicModel(model_dict=model_dict).fit(y=y, smooth=True)

        actual_error_message = str(context.exception)
        self.assertEqual(actual_error_message.strip(), error_message.strip())
