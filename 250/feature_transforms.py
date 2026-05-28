import numpy as np
from scipy import stats


def log_transform(x):
    x = np.asarray(x, dtype=float)
    if np.any(x <= 0):
        raise ValueError("log_transform requires all values to be positive")
    return np.log(x)


def sqrt_transform(x):
    x = np.asarray(x, dtype=float)
    if np.any(x < 0):
        raise ValueError("sqrt_transform requires all values to be non-negative")
    return np.sqrt(x)


def boxcox_transform(x, lmbda=None, on_nonpositive="shift"):
    x = np.asarray(x, dtype=float)
    strategies = {"shift", "yeojohnson", "raise"}
    if on_nonpositive not in strategies:
        raise ValueError(f"on_nonpositive must be one of {strategies}")

    x_min = np.min(x)
    shifted = False
    shift_amount = 0.0

    if x_min <= 0:
        if on_nonpositive == "raise":
            raise ValueError("boxcox_transform requires all values to be positive")
        if on_nonpositive == "yeojohnson":
            return yeojohnson_transform(x, lmbda=lmbda)
        if on_nonpositive == "shift":
            shift_amount = -x_min + 1e-8
            x = x + shift_amount
            shifted = True

    if lmbda is not None:
        if lmbda == 0:
            result = np.log(x)
        else:
            result = (np.power(x, lmbda) - 1) / lmbda
        if shifted:
            return result, lmbda, shift_amount
        return result, lmbda

    result, fitted_lmbda = stats.boxcox(x)
    if shifted:
        return result, fitted_lmbda, shift_amount
    return result, fitted_lmbda


def yeojohnson_transform(x, lmbda=None):
    x = np.asarray(x, dtype=float)
    if lmbda is not None:
        return _yeojohnson_with_lambda(x, lmbda)
    result, fitted_lmbda = stats.yeojohnson(x)
    return result, fitted_lmbda


def _yeojohnson_with_lambda(x, lmbda):
    out = np.empty_like(x, dtype=float)
    pos = x >= 0
    neg = ~pos

    if lmbda != 0:
        out[pos] = (np.power(x[pos] + 1, lmbda) - 1) / lmbda
    else:
        out[pos] = np.log(x[pos] + 1)

    if lmbda != 2:
        out[neg] = -((np.power(-x[neg] + 1, 2 - lmbda) - 1) / (2 - lmbda))
    else:
        out[neg] = -np.log(-x[neg] + 1)

    return out


def transform(x, method="log", **kwargs):
    x = np.asarray(x, dtype=float)
    methods = {
        "log": log_transform,
        "sqrt": sqrt_transform,
        "boxcox": boxcox_transform,
        "yeojohnson": yeojohnson_transform,
    }
    if method not in methods:
        raise ValueError(f"Unknown method '{method}', choose from {list(methods.keys())}")
    return methods[method](x, **kwargs)


def boxcox_inverse(y, lmbda, shift_amount=0.0):
    y = np.asarray(y, dtype=float)
    if lmbda == 0:
        x = np.exp(y)
    else:
        x = np.power(lmbda * y + 1, 1 / lmbda)
    if shift_amount > 0:
        x = x - shift_amount
    return x


def yeojohnson_inverse(y, lmbda):
    y = np.asarray(y, dtype=float)
    out = np.empty_like(y, dtype=float)
    pos = y >= 0
    neg = ~pos

    if lmbda != 0:
        out[pos] = np.power(lmbda * y[pos] + 1, 1 / lmbda) - 1
    else:
        out[pos] = np.exp(y[pos]) - 1

    if lmbda != 2:
        out[neg] = 1 - np.power(-(2 - lmbda) * y[neg] + 1, 1 / (2 - lmbda))
    else:
        out[neg] = 1 - np.exp(-y[neg])

    return out


def inverse_transform(y, method, lmbda=None, **kwargs):
    y = np.asarray(y, dtype=float)
    if method == "log":
        return np.exp(y)
    if method == "sqrt":
        return np.square(y)
    if method == "boxcox":
        if lmbda is None:
            raise ValueError("lmbda is required for boxcox inverse transform")
        shift_amount = kwargs.get("shift_amount", 0.0)
        return boxcox_inverse(y, lmbda, shift_amount)
    if method == "yeojohnson":
        if lmbda is None:
            raise ValueError("lmbda is required for yeojohnson inverse transform")
        return yeojohnson_inverse(y, lmbda)
    raise ValueError(f"Unknown method '{method}'")


def qq_plot_data(x, y=None):
    x = np.asarray(x, dtype=float)
    x_sorted = np.sort(x)
    n = len(x_sorted)
    theoretical = stats.norm.ppf((np.arange(1, n + 1) - 0.5) / n)

    if y is None:
        return {
            "theoretical": theoretical,
            "sample": x_sorted,
        }

    y = np.asarray(y, dtype=float)
    y_sorted = np.sort(y)
    return {
        "original": {"theoretical": theoretical, "sample": x_sorted},
        "transformed": {"theoretical": theoretical, "sample": y_sorted},
    }


def optimize_lambda(x, method="boxcox", lmbda_range=(-2, 2), n_points=100):
    x = np.asarray(x, dtype=float)

    if method == "boxcox":
        if np.any(x <= 0):
            shift_amount = -np.min(x) + 1e-8
            x = x + shift_amount
        else:
            shift_amount = 0.0
        result, fitted_lmbda = stats.boxcox(x)
        return fitted_lmbda, shift_amount

    if method == "yeojohnson":
        result, fitted_lmbda = stats.yeojohnson(x)
        return fitted_lmbda, 0.0

    raise ValueError(f"method must be 'boxcox' or 'yeojohnson', got '{method}'")

