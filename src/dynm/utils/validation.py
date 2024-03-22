"""Methods for validation."""
import numpy as np


def format_missing_keys_dict(missing_keys: dict):
    """Remove 'W' or 'discount' from missing keys dictionary."""
    if 'W' in missing_keys and 'discount' in missing_keys:
        pass
    else:
        try:
            missing_keys.remove('discount')
        except Exception:
            pass

        try:
            missing_keys.remove('W')
        except Exception:
            pass
    return sorted(missing_keys)


def validate_polynomial_model_dict_keys(model_dict: dict):
    """
    Validate keys in a dictionary representing a polynomial model.

    Args:
        model_dict (dict): Dictionary containing model keys.

    Raises:
        ValueError: If required keys are missing.
    """
    user_keys = set(list(model_dict.keys()))
    expected_keys = {'m0', 'C0', 'ntrend', 'W', 'discount'}
    missing_keys = expected_keys - user_keys

    if (missing_keys == {'discount'}) | (missing_keys == {'W'}):
        pass
    else:
        formated_missing_keys = format_missing_keys_dict(
            missing_keys=missing_keys)

        error_message = ("Missing elements in polynomial model: " +
                         str(formated_missing_keys))
        raise ValueError(error_message)


def validate_regression_model_dict_keys(model_dict: dict):
    """
    Validate keys in a dictionary representing a regression model.

    Args:
        model_dict (dict): Dictionary containing model keys.

    Raises:
        ValueError: If required keys are missing.
    """
    user_keys = set(list(model_dict.keys()))
    expected_keys = set(['m0', 'C0', 'nregn', 'discount', 'W'])
    missing_keys = expected_keys - user_keys

    if (missing_keys == {'discount'}) | (missing_keys == {'W'}):
        pass
    else:
        formated_missing_keys = format_missing_keys_dict(
            missing_keys=missing_keys)

        error_message = ("Missing elements in regression model: " +
                         str(formated_missing_keys))
        raise ValueError(error_message)


def validate_seasonal_model_dict_keys(model_dict: dict):
    """
    Validate keys in a dictionary representing a seasonal model.

    Args:
        model_dict (dict): Dictionary containing model keys.

    Raises:
        ValueError: If required keys are missing.
    """
    user_keys = set(list(model_dict.keys()))
    expected_keys = set(['m0', 'C0', 'seas_period',
                         'seas_harm_components', 'discount', 'W'])
    missing_keys = expected_keys - user_keys

    if (missing_keys == {'discount'}) | (missing_keys == {'W'}):
        pass
    else:
        formated_missing_keys = format_missing_keys_dict(
            missing_keys=missing_keys)

        error_message = ("Missing elements in seasonal model: " +
                         str(formated_missing_keys))
        raise ValueError(error_message)


def validate_transfer_function_model_dict_keys(model_dict: dict):
    """
    Validate keys in a dictionary representing a transfer function model.

    Args:
        model_dict (dict): Dictionary containing model keys.

    Raises:
        ValueError: If required keys are missing.
    """
    user_keys = set(list(model_dict.keys()))
    expected_keys = set(['m0', 'C0', 'ntfm',
                         'lambda_order', 'gamma_order',
                         'discount', 'W'])
    missing_keys = expected_keys - user_keys

    if (missing_keys == {'discount'}) | (missing_keys == {'W'}):
        pass
    else:
        formated_missing_keys = format_missing_keys_dict(
            missing_keys=missing_keys)

        error_message = ("Missing elements in transfer function model: " +
                         str(formated_missing_keys))
        raise ValueError(error_message)


def validate_autoregressive_model_dict_keys(model_dict: dict):
    """
    Validate keys in a dictionary representing an autoregressive model.

    Args:
        model_dict (dict): Dictionary containing model keys.

    Raises:
        ValueError: If required keys are missing.
    """
    user_keys = set(list(model_dict.keys()))
    expected_keys = set(['m0', 'C0', 'order', 'discount', 'W'])
    missing_keys = expected_keys - user_keys

    if (missing_keys == {'discount'}) | (missing_keys == {'W'}):
        pass
    else:
        formated_missing_keys = format_missing_keys_dict(
            missing_keys=missing_keys)

        error_message = ("Missing elements in autoregressive model: " +
                         str(formated_missing_keys))
        raise ValueError(error_message)


def validate_model_dict_polynomial_mean_array(model_dict: dict):
    """
    Validate prior mean array shape for polynomial model.

    Args:
        model_dict (dict): Dictionary containing model keys.

            Obrigatory keys: {'m0', 'C0', 'ntrend''}.

            Optional keys (choose one): {'W', 'discount'}.

    Raises:
        ValueError: If prior mean array and ntrend are incompatible.
    """
    ntrend = model_dict.get('ntrend')
    m0 = model_dict.get('m0')

    if m0.shape[0] == ntrend:
        pass
    else:
        error_message = ("Prior mean array" +
                         "and declared ntrend are incomplatible")
        raise ValueError(error_message)


def validate_model_dict_regression_mean_array(model_dict: dict):
    """
    Validate prior mean array shape for regression model.

    Args:
        model_dict (dict): Dictionary containing model keys.

            Obrigatory keys: {'m0', 'C0', 'nregn'}.

            Optional keys (choose one): {'W', 'discount'}.

    Raises:
        ValueError: If prior mean array and nregn are incompatible.
    """
    nregn = model_dict.get('nregn')
    m0 = model_dict.get('m0')

    if m0.shape[0] == nregn:
        pass
    else:
        error_message = ("Prior mean array" +
                         "and declared nregn are incomplatible")
        raise ValueError(error_message)


def validate_model_dict_seasonal_mean_array(model_dict: dict):
    """
    Validate prior mean array shape for seasonal model.

    Args:
        model_dict (dict): Dictionary containing model keys.

            Obrigatory keys: {'m0', 'C0', 'seas_period',
                              'seas_harm_components'}.

            Optional keys (choose one): {'W', 'discount'}.

    Raises:
        ValueError: If prior mean array and seas_harm_components are
            incompatible.
    """
    seas_harm_components = model_dict.get('seas_harm_components')
    nseas = 2 * len(seas_harm_components)
    m0 = model_dict.get('m0')

    if m0.shape[0] == nseas:
        pass
    else:
        error_message = ("Prior mean array" +
                         "and declared seas_harm_components" +
                         "are incomplatible")
        raise ValueError(error_message)


def validate_model_dict_transfer_function_mean_array(model_dict: dict):
    """
    Validate prior mean array shape for transfer function model.

    Args:
        model_dict (dict): Dictionary containing model keys.

            Obrigatory keys: {'m0', 'C0', 'lambda_order',
                              'gamma_order', 'ntfm'}.

            Optional keys (choose one): {'W', 'discount'}.

    Raises:
        ValueError: If prior mean array and model parameters are incompatible.
    """
    gamma_order = model_dict.get('gamma_order')
    lambda_order = model_dict.get('lambda_order')
    ntfm = model_dict.get('ntfm')
    nparams = (gamma_order + 2 * lambda_order) * ntfm
    m0 = model_dict.get('m0')

    if m0.shape[0] == nparams:
        pass
    else:
        error_message = ("Prior mean array" +
                         "and declared ntfm, gamma_order, lambda_order" +
                         "are incomplatible")
        raise ValueError(error_message)


def validate_model_dict_autoregressive_mean_array(model_dict: dict):
    """
    Validate prior mean array shape for autoregressive model.

    Args:
        model_dict (dict): Dictionary containing model keys.

            Obrigatory keys: {'m0', 'C0', 'order'}.

            Optional keys (choose one): {'W', 'discount'}.

    Raises:
        ValueError: If prior mean array and model parameters are incompatible.
    """
    order = model_dict.get('order')
    nparams = 2 * order
    m0 = model_dict.get('m0')

    if m0.shape[0] == nparams:
        pass
    else:
        error_message = ("Prior mean array and declared order" +
                         "are incomplatible")
        raise ValueError(error_message)


def validate_model_dict_polynomial_covariance_matrix(model_dict: dict):
    """
    Validate prior covariance matrix shape for polynomial model.

    Args:
        model_dict (dict): Dictionary containing model keys.

            Obrigatory keys: {'m0', 'C0', 'ntrend''}.

            Optional keys (choose one): {'W', 'discount'}.

    Raises:
        ValueError: If prior covariance matrix and ntrend are incompatible.
    """
    ntrend = model_dict.get('ntrend')
    C0 = model_dict.get('C0')

    if (C0.shape[0] == ntrend) & (C0.shape[1] == ntrend):
        pass
    else:
        error_message = ("Prior covariance matrix" +
                         "and declared ntrend are incomplatible")
        raise ValueError(error_message)


def validate_model_dict_regression_covariance_matrix(model_dict: dict):
    """
    Validate prior covariance matrix shape for regression model.

    Args:
        model_dict (dict): Dictionary containing model keys.

            Obrigatory keys: {'m0', 'C0', 'nregn'}.

            Optional keys (choose one): {'W', 'discount'}.

    Raises:
        ValueError: If prior covariance matrix and nregn are incompatible.
    """
    nregn = model_dict.get('nregn')
    C0 = model_dict.get('C0')

    if (C0.shape[0] == nregn) & (C0.shape[1] == nregn):
        pass
    else:
        error_message = ("Prior covariance matrix" +
                         "and declared nregn are incomplatible")
        raise ValueError(error_message)


def validate_model_dict_seasonal_covariance_matrix(model_dict: dict):
    """
    Validate prior covariance matrix shape for seasonal model.

    Args:
        model_dict (dict): Dictionary containing model keys.

            Obrigatory keys: {'m0', 'C0', 'seas_period',
                              'seas_harm_components'}.

            Optional keys (choose one): {'W', 'discount'}.

    Raises:
        ValueError: If prior covariance matrix and seas_harm_components
        are incompatible.
    """
    seas_harm_components = model_dict.get('seas_harm_components')
    nseas = 2 * len(seas_harm_components)
    C0 = model_dict.get('C0')

    if (C0.shape[0] == nseas) & (C0.shape[1] == nseas):
        pass
    else:
        error_message = ("Prior covariance matrix" +
                         "and declared seas_harm_components" +
                         "are incomplatible")
        raise ValueError(error_message)


def validate_model_dict_transfer_function_covariance_matrix(model_dict: dict):
    """
    Validate prior covariance matrix shape for transfer function model.

    Args:
        model_dict (dict): Dictionary containing model keys.

            Obrigatory keys: {'m0', 'C0', 'seas_period',
                              'seas_harm_components'}.

            Optional keys (choose one): {'W', 'discount'}.

    Raises:
        ValueError: If prior covariance matrix and seas_harm_components
        are incompatible.
    """
    gamma_order = model_dict.get('gamma_order')
    lambda_order = model_dict.get('lambda_order')
    ntfm = model_dict.get('ntfm')
    nparams = (gamma_order + 2 * lambda_order) * ntfm
    C0 = model_dict.get('C0')

    if (C0.shape[0] == nparams) & (C0.shape[1] == nparams):
        pass
    else:
        error_message = ("Prior covariance matrix" +
                         "and declared ntfm, gamma_order, lambda_order" +
                         "are incomplatible")
        raise ValueError(error_message)


def validate_model_dict_autoregressive_covariance_matrix(model_dict: dict):
    """
    Validate prior covariance matrix shape for autoregressive model.

    Args:
        model_dict (dict): Dictionary containing model keys.

            Obrigatory keys: {'m0', 'C0', 'order'}.

            Optional keys (choose one): {'W', 'discount'}.

    Raises:
        ValueError: If prior mean array and model parameters are incompatible.
    """
    order = model_dict.get('order')
    nparams = 2 * order
    C0 = model_dict.get('C0')

    if (C0.shape[0] == nparams) & (C0.shape[1] == nparams):
        pass
    else:
        error_message = ("Prior mean array and declared order" +
                         "are incomplatible")
        raise ValueError(error_message)


def validate_model_dict_polynomial_discount_array(model_dict: dict):
    """
    Validate declared discount factor for polynomial model.

    Args:
        model_dict (dict): Dictionary containing model keys.

            Obrigatory keys: {'m0', 'C0', 'ntrend''}.

            Optional keys (choose one): {'W', 'discount'}.

    Raises:
        ValueError: If the discount factor is not a scalar or falls
        outside the [0, 1] interval.
    """
    discount = model_dict.get('discount')

    if discount is not None:
        if np.isscalar(discount):
            pass
        else:
            error_message = ("Discount for polynomial model is not a scalar")
            raise ValueError(error_message)

        if (discount >= 0) & (discount <= 1):
            pass
        else:
            error_message = ("Discount for polynomial model " +
                             "is not in [0,1] interval")
            raise ValueError(error_message)


def validate_model_dict_regression_discount_array(model_dict: dict):
    """
    Validate declared discount factor for regression model.

    Args:
        model_dict (dict): Dictionary containing model keys.

            Obrigatory keys: {'m0', 'C0', 'nregn'}.

            Optional keys (choose one): {'W', 'discount'}.

    Raises:
        ValueError: If the discount factor is not a scalar or falls
        outside the [0, 1] interval.
    """
    discount = model_dict.get('discount')

    if discount is not None:
        if np.isscalar(discount):
            pass
        else:
            error_message = ("Discount for regression model is not a scalar")
            raise ValueError(error_message)

        if (discount >= 0) & (discount <= 1):
            pass
        else:
            error_message = ("Discount for regression model " +
                             "is not in [0,1] interval")
            raise ValueError(error_message)


def validate_model_dict_seasonal_discount_array(model_dict: dict):
    """
    Validate declared discount factor for seasonal model.

    Args:
        model_dict (dict): Dictionary containing model keys.

            Obrigatory keys: {'m0', 'C0', 'nregn'}.

            Optional keys (choose one): {'W', 'discount'}.

    Raises:
        ValueError: If the discount factor is not a scalar or falls
        outside the [0, 1] interval.
    """
    discount = model_dict.get('discount')

    if discount is not None:
        if np.isscalar(discount):
            pass
        else:
            error_message = ("Discount for seasonal model is not a scalar")
            raise ValueError(error_message)

        if (discount >= 0) & (discount <= 1):
            pass
        else:
            error_message = ("Discount for seasonal model " +
                             "is not in [0,1] interval")
            raise ValueError(error_message)


def validate_model_dict_transfer_function_discount_array(model_dict: dict):
    """
    Validate declared discount factor for transfer function model.

    Args:
        model_dict (dict): Dictionary containing model keys.

            Obrigatory keys: {'m0', 'C0', 'lambda_order',
                              'gamma_order', 'ntfm'}.

            Optional keys (choose one): {'W', 'discount'}.

    Raises:
        ValueError: If the discount array is incompatible with the model
        parameters, or if any element in the discount array falls outside
        the [0, 1] interval.
    """
    gamma_order = model_dict.get('gamma_order')
    lambda_order = model_dict.get('lambda_order')
    ntfm = model_dict.get('ntfm')
    nparams = (gamma_order + 2 * lambda_order) * ntfm
    discount = model_dict.get('discount')

    if discount is not None:
        if discount.shape[0] == nparams:
            pass
        else:
            error_message = ("Discount array has a length of " +
                             str(discount.shape[0]) +
                             ", but it should have a length of " +
                             str(nparams))
            raise ValueError(error_message)

        min_discount = np.min(discount)
        max_discount = np.max(discount)
        if (min_discount >= 0) & (max_discount <= 1):
            pass
        else:
            error_message = ("Some elements in the discount array falls " +
                             "outside the [0,1] interval")
            raise ValueError(error_message)


def validate_model_dict_autoregressive_discount_array(model_dict: dict):
    """
    Validate declared discount factor for autoregressive model.

    Args:
        model_dict (dict): Dictionary containing model keys.

            Obrigatory keys: {'m0', 'C0', 'order'}.

            Optional keys (choose one): {'W', 'discount'}.

    Raises:
        ValueError: If the discount array is incompatible with the model
        parameters, or if any element in the discount array falls outside
        the [0, 1] interval.
    """
    order = model_dict.get('order')
    nparams = 2 * order
    discount = model_dict.get('discount')

    if discount is not None:
        if discount.shape[0] == nparams:
            pass
        else:
            error_message = ("Discount array has a length of " +
                             str(discount.shape[0]) +
                             ", but it should have a length of " +
                             str(nparams))
            raise ValueError(error_message)

        min_discount = np.min(discount)
        max_discount = np.max(discount)
        if (min_discount >= 0) & (max_discount <= 1):
            pass
        else:
            error_message = ("Some elements in the discount array falls " +
                             "outside the [0,1] interval")
            raise ValueError(error_message)


def validate_input_dict(mod, X: dict):
    """
    Validate input dictionary against model parameters.

    Parameters:
    - mod: Bayesian Dynamic Model object.
    - X (dict): Dictionary containing input data for regression
      and transfer function models.

    Raises:
    - ValueError: If the input data is None or has an incompatible shape
      with the declared model parameters.
    """
    nregn = mod.dlm.regression_model.nregn
    ntfm = mod.dnm.transfer_function_model.ntfm

    X_regression = X.get('regression')
    X_transfer_function = X.get('transfer_function')

    if nregn > 0:
        try:
            X_regression.shape[1] == nregn
        except Exception:
            error_message = ("The input X for regression is None or " +
                             "has an incompatible shape with the " +
                             "declared nregn.")
            raise ValueError(error_message)

    if ntfm > 0:
        try:
            X_transfer_function.shape[1] == ntfm
        except Exception:
            error_message = ("The input X for transfer function is None or " +
                             "has an incompatible shape with the " +
                             "declared ntfm.")
            raise ValueError(error_message)
