import numpy as np
from scipy.signal import firwin, filtfilt, resample as scipy_resample
from scipy.interpolate import CubicSpline, Akima1DInterpolator


def _antialias_filter(data, k, filter_order=None):
    fs = 1.0
    nyq = fs / 2.0
    cutoff = nyq / k
    if filter_order is None:
        filter_order = min(2 * k + 1, 127)
    if filter_order % 2 == 0:
        filter_order += 1
    taps = firwin(filter_order, cutoff, fs=fs)
    pad_len = 3 * max(len(taps) - 1, 0)
    if len(data) <= pad_len:
        return data
    filtered = filtfilt(taps, [1.0], data, padlen=pad_len)
    return filtered


def downsample_stride(data, k, offset=0):
    data = np.asarray(data, dtype=float)
    if k <= 0:
        raise ValueError("k must be a positive integer")
    if not (0 <= offset < k):
        raise ValueError("offset must be in [0, k)")
    return data[offset::k]


def downsample_avg_pool(data, k, offset=0):
    data = np.asarray(data, dtype=float)
    if k <= 0:
        raise ValueError("k must be a positive integer")
    if not (0 <= offset < k):
        raise ValueError("offset must be in [0, k)")
    data = data[offset:]
    n = len(data)
    trim = n - n % k
    trimmed = data[:trim]
    return trimmed.reshape(-1, k).mean(axis=1)


def downsample_max_pool(data, k, offset=0):
    data = np.asarray(data, dtype=float)
    if k <= 0:
        raise ValueError("k must be a positive integer")
    if not (0 <= offset < k):
        raise ValueError("offset must be in [0, k)")
    data = data[offset:]
    n = len(data)
    trim = n - n % k
    trimmed = data[:trim]
    return trimmed.reshape(-1, k).max(axis=1)


def downsample_decimate(data, k, offset=0, filter_order=None):
    data = np.asarray(data, dtype=float)
    if k <= 0:
        raise ValueError("k must be a positive integer")
    if not (0 <= offset < k):
        raise ValueError("offset must be in [0, k)")
    if k == 1:
        return data.copy()
    filtered = _antialias_filter(data, k, filter_order)
    return filtered[offset::k]


def upsample_zero_fill(data, k):
    data = np.asarray(data, dtype=float)
    if k <= 0:
        raise ValueError("k must be a positive integer")
    n = len(data)
    out = np.zeros(n * k, dtype=float)
    out[::k] = data
    return out


def upsample_nearest(data, k):
    data = np.asarray(data, dtype=float)
    if k <= 0:
        raise ValueError("k must be a positive integer")
    return np.repeat(data, k)


def _resample_coords(n_in, n_out):
    x_in = np.arange(n_in, dtype=float)
    x_out = np.linspace(0, n_in - 1, n_out, endpoint=True)
    return x_in, x_out


def upsample_spline(data, target_len, method="cubic", bc_type="natural"):
    data = np.asarray(data, dtype=float)
    n_in = len(data)
    if target_len < 0:
        raise ValueError("target_len must be non-negative")
    if target_len == 0:
        return np.array([], dtype=float)
    if target_len < n_in:
        raise ValueError("target_len must be >= len(data) for upsampling; use a downsampling function instead")
    if n_in < 2:
        return np.full(target_len, data[0], dtype=float) if n_in == 1 else np.array([], dtype=float)
    x_in, x_out = _resample_coords(n_in, target_len)
    method = method.lower()
    if method == "cubic":
        interp = CubicSpline(x_in, data, bc_type=bc_type)
    elif method == "akima":
        interp = Akima1DInterpolator(x_in, data)
    else:
        raise ValueError("method must be 'cubic' or 'akima'")
    return interp(x_out)


def upsample_fft(data, target_len):
    data = np.asarray(data, dtype=float)
    n_in = len(data)
    if target_len < 0:
        raise ValueError("target_len must be non-negative")
    if target_len == 0:
        return np.array([], dtype=float)
    if target_len < n_in:
        raise ValueError("target_len must be >= len(data) for upsampling; use a downsampling function instead")
    if n_in == 0:
        return np.array([], dtype=float)
    return scipy_resample(data, target_len)


def resample(data, target_len, method="fft"):
    data = np.asarray(data, dtype=float)
    n_in = len(data)
    if target_len < 0:
        raise ValueError("target_len must be non-negative")
    if target_len == 0:
        return np.array([], dtype=float)
    if target_len == n_in:
        return data.copy()
    if target_len > n_in:
        method = method.lower()
        if method == "fft":
            return upsample_fft(data, target_len)
        elif method in ("cubic", "akima"):
            return upsample_spline(data, target_len, method=method)
        elif method == "nearest":
            ratio = target_len / n_in
            x_in = np.arange(n_in, dtype=float)
            x_out = np.linspace(0, n_in - 1, target_len, endpoint=True)
            indices = np.round(x_out).astype(int)
            indices = np.clip(indices, 0, n_in - 1)
            return data[indices]
        elif method == "zero_fill":
            k = int(np.ceil(target_len / n_in))
            upsampled = upsample_zero_fill(data, k)
            return upsampled[:target_len]
        else:
            raise ValueError("Unknown upsampling method. Choose from 'fft', 'cubic', 'akima', 'nearest', 'zero_fill'.")
    else:
        ratio = n_in / target_len
        method = method.lower()
        if method == "fft":
            return scipy_resample(data, target_len)
        elif method == "decimate":
            k = int(round(ratio))
            return downsample_decimate(data, k)[:target_len]
        elif method == "stride":
            k = int(round(ratio))
            return downsample_stride(data, k)[:target_len]
        elif method == "avg_pool":
            k = int(round(ratio))
            return downsample_avg_pool(data, k)[:target_len]
        elif method == "max_pool":
            k = int(round(ratio))
            return downsample_max_pool(data, k)[:target_len]
        else:
            raise ValueError("Unknown downsampling method. Choose from 'fft', 'decimate', 'stride', 'avg_pool', 'max_pool'.")
