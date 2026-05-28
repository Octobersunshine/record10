import numpy as np
from scipy.optimize import minimize_scalar
from sklearn.isotonic import IsotonicRegression as SklearnIsotonicRegression
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def _softmax(logits):
    shifted = logits - np.max(logits, axis=-1, keepdims=True)
    exp_shifted = np.exp(shifted)
    return exp_shifted / np.sum(exp_shifted, axis=-1, keepdims=True)


def _probs_to_logits(probs):
    eps = 1e-15
    probs_clipped = np.clip(probs, eps, 1 - eps)
    return np.log(probs_clipped)


def _nll(temperature, logits, labels):
    if temperature <= 0:
        return np.inf
    scaled = logits / temperature
    probs = _softmax(scaled)
    n = labels.shape[0]
    log_probs = np.log(np.clip(probs, 1e-15, 1.0))
    selected = log_probs[np.arange(n), labels]
    return -np.mean(selected)


class TemperatureScaling:
    def __init__(self):
        self.temperature_ = None

    def fit(self, probs, labels):
        probs = np.asarray(probs, dtype=np.float64)
        labels = np.asarray(labels, dtype=np.int64)

        if probs.ndim == 1:
            probs = np.column_stack([1 - probs, probs])
        if probs.ndim != 2:
            raise ValueError("probs must be 1D (binary) or 2D (multi-class)")
        if labels.ndim != 1:
            raise ValueError("labels must be a 1D array")
        if len(probs) != len(labels):
            raise ValueError("probs and labels must have the same length")
        if np.any(probs < 0) or np.any(probs > 1):
            raise ValueError("probs must be in [0, 1]")
        n_classes = probs.shape[1]
        if not np.all(np.isin(labels, np.arange(n_classes))):
            raise ValueError(f"labels must be in [0, {n_classes - 1}]")

        logits = _probs_to_logits(probs)

        result = minimize_scalar(
            _nll,
            bounds=(0.01, 10.0),
            args=(logits, labels),
            method='bounded',
            options={'xatol': 1e-8}
        )
        self.temperature_ = result.x
        self.n_classes_ = n_classes
        return self

    def predict_proba(self, probs):
        if self.temperature_ is None:
            raise RuntimeError("TemperatureScaling must be fitted before calling predict_proba")

        probs = np.asarray(probs, dtype=np.float64)
        input_1d = probs.ndim == 1
        if input_1d:
            probs = np.column_stack([1 - probs, probs])
        if probs.ndim != 2:
            raise ValueError("probs must be 1D (binary) or 2D (multi-class)")
        if np.any(probs < 0) or np.any(probs > 1):
            raise ValueError("probs must be in [0, 1]")

        logits = _probs_to_logits(probs)
        calibrated = _softmax(logits / self.temperature_)

        if input_1d:
            return calibrated[:, 1]
        return calibrated

    def fit_predict(self, probs, labels):
        return self.fit(probs, labels).predict_proba(probs)


class IsotonicCalibration:
    def __init__(self, out_of_bounds='clip'):
        self.out_of_bounds = out_of_bounds
        self.calibrators_ = None
        self.n_classes_ = None

    def fit(self, probs, labels):
        probs = np.asarray(probs, dtype=np.float64)
        labels = np.asarray(labels, dtype=np.int64)

        input_1d = probs.ndim == 1
        if input_1d:
            probs = np.column_stack([1 - probs, probs])
        if probs.ndim != 2:
            raise ValueError("probs must be 1D (binary) or 2D (multi-class)")
        if labels.ndim != 1:
            raise ValueError("labels must be a 1D array")
        if len(probs) != len(labels):
            raise ValueError("probs and labels must have the same length")
        if np.any(probs < 0) or np.any(probs > 1):
            raise ValueError("probs must be in [0, 1]")
        n_classes = probs.shape[1]
        if not np.all(np.isin(labels, np.arange(n_classes))):
            raise ValueError(f"labels must be in [0, {n_classes - 1}]")

        self.n_classes_ = n_classes
        self.calibrators_ = []

        for k in range(n_classes):
            y_binary = (labels == k).astype(np.float64)
            iso = SklearnIsotonicRegression(out_of_bounds=self.out_of_bounds, y_min=1e-15, y_max=1 - 1e-15)
            iso.fit(probs[:, k], y_binary)
            self.calibrators_.append(iso)

        return self

    def predict_proba(self, probs):
        if self.calibrators_ is None:
            raise RuntimeError("IsotonicCalibration must be fitted before calling predict_proba")

        probs = np.asarray(probs, dtype=np.float64)
        input_1d = probs.ndim == 1
        if input_1d:
            probs = np.column_stack([1 - probs, probs])
        if probs.ndim != 2:
            raise ValueError("probs must be 1D (binary) or 2D (multi-class)")
        if np.any(probs < 0) or np.any(probs > 1):
            raise ValueError("probs must be in [0, 1]")

        calibrated = np.zeros_like(probs)
        for k in range(self.n_classes_):
            calibrated[:, k] = self.calibrators_[k].predict(probs[:, k])

        row_sums = calibrated.sum(axis=1, keepdims=True)
        calibrated = calibrated / np.clip(row_sums, 1e-15, None)

        if input_1d:
            return calibrated[:, 1]
        return calibrated

    def fit_predict(self, probs, labels):
        return self.fit(probs, labels).predict_proba(probs)


def expected_calibration_error(probs, labels, n_bins=10):
    probs = np.asarray(probs, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.int64)

    if probs.ndim == 1:
        confidences = probs
        predictions = (probs >= 0.5).astype(np.int64)
    elif probs.ndim == 2:
        confidences = np.max(probs, axis=1)
        predictions = np.argmax(probs, axis=1)
    else:
        raise ValueError("probs must be 1D (binary) or 2D (multi-class)")

    if labels.ndim != 1:
        raise ValueError("labels must be a 1D array")
    if len(confidences) != len(labels):
        raise ValueError("probs and labels must have the same length")
    if np.any(confidences < 0) or np.any(confidences > 1):
        raise ValueError("confidences must be in [0, 1]")
    if n_bins < 1:
        raise ValueError("n_bins must be at least 1")

    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    n = len(confidences)
    correct = (predictions == labels).astype(np.float64)

    for i in range(n_bins):
        bin_lower = bin_boundaries[i]
        bin_upper = bin_boundaries[i + 1]

        if i == 0:
            mask = (confidences >= bin_lower) & (confidences <= bin_upper)
        else:
            mask = (confidences > bin_lower) & (confidences <= bin_upper)

        bin_count = np.sum(mask)
        if bin_count == 0:
            continue

        bin_weight = bin_count / n
        avg_conf = np.mean(confidences[mask])
        avg_acc = np.mean(correct[mask])
        ece += bin_weight * np.abs(avg_conf - avg_acc)

    return ece


def _get_bin_statistics(probs, labels, n_bins=10):
    probs = np.asarray(probs, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.int64)

    if probs.ndim == 1:
        confidences = probs
        predictions = (probs >= 0.5).astype(np.int64)
    elif probs.ndim == 2:
        confidences = np.max(probs, axis=1)
        predictions = np.argmax(probs, axis=1)
    else:
        raise ValueError("probs must be 1D (binary) or 2D (multi-class)")

    correct = (predictions == labels).astype(np.float64)
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    bin_centers = (bin_boundaries[:-1] + bin_boundaries[1:]) / 2
    bin_accs = np.zeros(n_bins)
    bin_confs = np.zeros(n_bins)
    bin_counts = np.zeros(n_bins)

    for i in range(n_bins):
        bin_lower = bin_boundaries[i]
        bin_upper = bin_boundaries[i + 1]

        if i == 0:
            mask = (confidences >= bin_lower) & (confidences <= bin_upper)
        else:
            mask = (confidences > bin_lower) & (confidences <= bin_upper)

        bin_count = np.sum(mask)
        bin_counts[i] = bin_count

        if bin_count > 0:
            bin_accs[i] = np.mean(correct[mask])
            bin_confs[i] = np.mean(confidences[mask])

    return bin_centers, bin_accs, bin_confs, bin_counts


def plot_reliability_diagram(probs_list, labels, method_names=None, n_bins=10, save_path=None, figsize=(12, 5)):
    if not isinstance(probs_list, list):
        probs_list = [probs_list]
    if method_names is None:
        method_names = [f"Method {i + 1}" for i in range(len(probs_list))]

    n_methods = len(probs_list)
    fig, axes = plt.subplots(1, n_methods + 1, figsize=figsize, gridspec_kw={'width_ratios': [4] * n_methods + [1]})
    if n_methods == 1:
        axes = [axes[0], axes[1]]

    colors = plt.colormaps.get_cmap('tab10').resampled(n_methods)
    max_count = 0

    for idx, (probs, name) in enumerate(zip(probs_list, method_names)):
        ax = axes[idx]
        bin_centers, bin_accs, bin_confs, bin_counts = _get_bin_statistics(probs, labels, n_bins)

        bin_width = 1.0 / n_bins
        ax.bar(bin_centers, bin_accs, width=bin_width * 0.9, alpha=0.6, color=colors(idx), label='Accuracy')
        ax.bar(bin_centers, np.abs(bin_confs - bin_accs), width=bin_width * 0.9, bottom=bin_accs,
               alpha=0.3, color='red', label='Gap')
        ax.plot([0, 1], [0, 1], 'k--', lw=1.5, label='Perfect calibration')

        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_aspect('equal')
        ax.set_xlabel('Confidence')
        if idx == 0:
            ax.set_ylabel('Accuracy')
        ax.set_title(name)
        ax.legend(loc='lower right', fontsize=8)
        ax.grid(True, alpha=0.3)

        max_count = max(max_count, np.max(bin_counts))

    ax_hist = axes[-1]
    for idx, (probs, name) in enumerate(zip(probs_list, method_names)):
        if np.asarray(probs).ndim == 1:
            confidences = np.asarray(probs)
        else:
            confidences = np.max(probs, axis=1)
        ax_hist.hist(confidences, bins=np.linspace(0, 1, n_bins + 1),
                     alpha=0.5, label=name, color=colors(idx), histtype='stepfilled')
    ax_hist.set_xlabel('Confidence')
    ax_hist.set_ylabel('Count')
    ax_hist.set_title('Confidence Distribution')
    ax_hist.legend(loc='upper left', fontsize=8)
    ax_hist.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path is not None:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        return save_path
    else:
        return fig


def compare_calibrators(probs_train, labels_train, probs_test=None, labels_test=None,
                        n_bins=10, plot_path=None):
    if probs_test is None:
        probs_test = probs_train
    if labels_test is None:
        labels_test = labels_train

    methods = {
        'Uncalibrated': None,
        'Temperature Scaling': TemperatureScaling(),
        'Isotonic Regression': IsotonicCalibration()
    }

    results = {}
    calibrated_probs_list = []
    method_names_ordered = []

    for name, model in methods.items():
        if model is None:
            calib_probs = probs_test
            ece = expected_calibration_error(probs_test, labels_test, n_bins=n_bins)
            results[name] = {
                'ece': ece,
                'calibrated_probs': calib_probs,
                'model': None
            }
        else:
            model.fit(probs_train, labels_train)
            calib_probs = model.predict_proba(probs_test)
            ece = expected_calibration_error(calib_probs, labels_test, n_bins=n_bins)
            results[name] = {
                'ece': ece,
                'calibrated_probs': calib_probs,
                'model': model
            }
        calibrated_probs_list.append(calib_probs)
        method_names_ordered.append(name)

    ece_values = {name: r['ece'] for name, r in results.items()}
    best_name = min(ece_values, key=ece_values.get)

    if plot_path is not None:
        plot_reliability_diagram(calibrated_probs_list, labels_test, method_names_ordered, n_bins, plot_path)

    return {
        'results': results,
        'best_method': best_name,
        'best_ece': ece_values[best_name],
        'best_model': results[best_name]['model'],
        'ece_comparison': ece_values
    }


if __name__ == '__main__':
    np.random.seed(42)

    print("=" * 60)
    print("Calibration Method Comparison (5 classes)")
    print("=" * 60)

    n_samples = 2000
    n_classes = 5
    labels_true = np.random.randint(0, n_classes, size=n_samples)

    logits = np.random.randn(n_samples, n_classes) * 2.0
    logits[np.arange(n_samples), labels_true] += 3.0
    probs = _softmax(logits)
    logits_noisy = np.log(np.clip(probs, 1e-15, 1.0)) + np.random.randn(n_samples, n_classes) * 1.5
    probs_uncalibrated = _softmax(logits_noisy)

    split = int(0.7 * n_samples)
    probs_train, probs_test = probs_uncalibrated[:split], probs_uncalibrated[split:]
    labels_train, labels_test = labels_true[:split], labels_true[split:]

    comparison = compare_calibrators(
        probs_train=probs_train,
        labels_train=labels_train,
        probs_test=probs_test,
        labels_test=labels_test,
        n_bins=15,
        plot_path='reliability_diagram.png'
    )

    print("\nECE Comparison:")
    for name, ece in comparison['ece_comparison'].items():
        marker = " <-- BEST" if name == comparison['best_method'] else ""
        print(f"  {name:25s}: {ece:.6f}{marker}")

    print(f"\nBest method: {comparison['best_method']} (ECE = {comparison['best_ece']:.6f})")

    ts_model = comparison['results']['Temperature Scaling']['model']
    if hasattr(ts_model, 'temperature_'):
        print(f"\nTemperature Scaling params: T = {ts_model.temperature_:.4f}")

    print("\nReliability diagram saved to: reliability_diagram.png")

    print("\n" + "=" * 60)
    print("Binary calibration test (backward compatible)")
    print("=" * 60)

    n_binary = 1000
    labels_bin = np.random.randint(0, 2, size=n_binary)
    probs_bin = np.where(
        labels_bin == 1,
        np.random.beta(7, 3, size=n_binary),
        np.random.beta(3, 7, size=n_binary)
    )
    probs_bin = 0.1 + 0.8 * probs_bin

    split_b = int(0.7 * n_binary)
    probs_bin_train, probs_bin_test = probs_bin[:split_b], probs_bin[split_b:]
    labels_bin_train, labels_bin_test = labels_bin[:split_b], labels_bin[split_b:]

    comparison_bin = compare_calibrators(
        probs_train=probs_bin_train,
        labels_train=labels_bin_train,
        probs_test=probs_bin_test,
        labels_test=labels_bin_test,
        n_bins=10
    )

    print("\nECE Comparison (Binary):")
    for name, ece in comparison_bin['ece_comparison'].items():
        marker = " <-- BEST" if name == comparison_bin['best_method'] else ""
        print(f"  {name:25s}: {ece:.6f}{marker}")
