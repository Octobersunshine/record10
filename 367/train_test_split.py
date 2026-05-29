import numpy as np


def train_test_split(X, y, test_size=0.2, random_seed=None):
    if random_seed is not None:
        np.random.seed(random_seed)

    X = np.array(X)
    y = np.array(y)

    if X.shape[0] != y.shape[0]:
        raise ValueError(
            f"X and y must have the same number of samples, "
            f"got {X.shape[0]} and {y.shape[0]}"
        )

    n_samples = X.shape[0]
    n_test = int(n_samples * test_size)
    n_train = n_samples - n_test

    if n_test == 0 or n_train == 0:
        raise ValueError(
            f"test_size={test_size} results in 0 samples for one of the splits "
            f"with {n_samples} total samples"
        )

    indices = np.arange(n_samples)
    np.random.shuffle(indices)

    train_indices = indices[:n_train]
    test_indices = indices[n_train:]

    return X[train_indices], X[test_indices], y[train_indices], y[test_indices]


def stratified_train_test_split(X, y, test_size=0.2, random_seed=None):
    if random_seed is not None:
        np.random.seed(random_seed)

    X = np.array(X)
    y = np.array(y)

    if X.shape[0] != y.shape[0]:
        raise ValueError(
            f"X and y must have the same number of samples, "
            f"got {X.shape[0]} and {y.shape[0]}"
        )

    classes, counts = np.unique(y, return_counts=True)
    n_classes = len(classes)

    if n_classes < 2:
        raise ValueError("Stratified split requires at least 2 classes in y")

    train_indices = []
    test_indices = []

    for cls in classes:
        cls_indices = np.where(y == cls)[0]
        np.random.shuffle(cls_indices)

        n_cls = len(cls_indices)
        n_cls_test = max(1, int(n_cls * test_size))
        n_cls_train = n_cls - n_cls_test

        if n_cls_train <= 0:
            raise ValueError(
                f"Class {cls} has only {n_cls} samples, which is insufficient "
                f"for test_size={test_size} (needs at least 2 samples)"
            )

        train_indices.extend(cls_indices[:n_cls_train])
        test_indices.extend(cls_indices[n_cls_train:])

    np.random.shuffle(train_indices)
    np.random.shuffle(test_indices)

    return (
        X[train_indices], X[test_indices],
        y[train_indices], y[test_indices]
    )


def time_series_split(X, y, test_size=0.3):
    X = np.array(X)
    y = np.array(y)

    if X.shape[0] != y.shape[0]:
        raise ValueError(
            f"X and y must have the same number of samples, "
            f"got {X.shape[0]} and {y.shape[0]}"
        )

    n_samples = X.shape[0]
    n_test = int(n_samples * test_size)
    n_train = n_samples - n_test

    if n_test == 0 or n_train == 0:
        raise ValueError(
            f"test_size={test_size} results in 0 samples for one of the splits "
            f"with {n_samples} total samples"
        )

    train_indices = np.arange(n_train)
    test_indices = np.arange(n_train, n_samples)

    return (
        X[train_indices], X[test_indices],
        y[train_indices], y[test_indices]
    )


def kfold_split(n_samples, n_folds=5, shuffle=True, random_seed=None):
    if n_folds < 2:
        raise ValueError("n_folds must be at least 2")
    if n_samples < n_folds:
        raise ValueError(
            f"n_samples ({n_samples}) must be >= n_folds ({n_folds})"
        )

    indices = np.arange(n_samples)
    if shuffle:
        if random_seed is not None:
            np.random.seed(random_seed)
        np.random.shuffle(indices)

    fold_sizes = np.full(n_folds, n_samples // n_folds, dtype=int)
    fold_sizes[:n_samples % n_folds] += 1

    folds = []
    current = 0
    for fold_size in fold_sizes:
        folds.append(indices[current:current + fold_size])
        current += fold_size

    result = []
    for i in range(n_folds):
        test_idx = folds[i]
        train_idx = np.concatenate([folds[j] for j in range(n_folds) if j != i])
        result.append((train_idx, test_idx))

    return result


def data_summary(X_train, X_test, y_train, y_test, label=""):
    if label:
        print(f"\n{'=' * 70}")
        print(f"  {label}")
        print(f"{'=' * 70}")
    else:
        print()

    n_train = len(X_train)
    n_test = len(X_test)
    n_total = n_train + n_test

    print(f"  Total samples:    {n_total}")
    print(f"  Train samples:    {n_train} ({n_train / n_total:.1%})")
    print(f"  Test samples:     {n_test} ({n_test / n_total:.1%})")
    print(f"  Features:         {X_train.shape[1] if X_train.ndim > 1 else 1}")

    if y_train.ndim == 1:
        classes = np.unique(np.concatenate([y_train, y_test]))
        print(f"  Classes:          {len(classes)}")
        print(f"\n  {'Class':<10} {'Train':>8} {'Test':>8} {'Train%':>8} {'Test%':>8}")
        print(f"  {'-' * 42}")
        for cls in sorted(classes):
            n_tr = np.sum(y_train == cls)
            n_te = np.sum(y_test == cls)
            print(
                f"  {cls!s:<10} {n_tr:>8d} {n_te:>8d} "
                f"{n_tr / n_train:>7.1%} {n_te / n_test:>7.1%}"
            )

    print(f"\n  X_train shape:    {X_train.shape}")
    print(f"  X_test shape:     {X_test.shape}")
    print(f"  y_train shape:    {y_train.shape}")
    print(f"  y_test shape:     {y_test.shape}")

    if X_train.ndim > 1 and X_train.shape[1] > 0:
        print(f"\n  X_train stats (per feature):")
        for col in range(X_train.shape[1]):
            vals = X_train[:, col]
            print(
                f"    Feature {col}: "
                f"min={vals.min():.3f}, max={vals.max():.3f}, "
                f"mean={vals.mean():.3f}, std={vals.std():.3f}"
            )


def _class_distribution(y):
    classes, counts = np.unique(y, return_counts=True)
    total = len(y)
    return {cls: (count, count / total) for cls, count in zip(classes, counts)}


if __name__ == "__main__":
    print("=" * 70)
    print("Demo 1: Basic balanced dataset")
    print("=" * 70)
    X = np.array([[1, 2], [3, 4], [5, 6], [7, 8], [9, 10],
                   [11, 12], [13, 14], [15, 16], [17, 18], [19, 20]])
    y = np.array([0, 1, 0, 1, 0, 1, 0, 1, 0, 1])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_seed=42
    )

    print(f"X_train:\n{X_train}")
    print(f"X_test:\n{X_test}")
    print(f"y_train: {y_train}")
    print(f"y_test: {y_test}")
    print(f"Train ratio: {len(X_train) / len(X):.0%}")
    print(f"Test ratio:  {len(X_test) / len(X):.0%}")

    print("\n" + "=" * 70)
    print("Demo 2: Imbalanced dataset - Random vs Stratified comparison")
    print("=" * 70)
    np.random.seed(99)
    n_total = 100
    n_class0 = 80
    n_class1 = 15
    n_class2 = 5
    X_imbal = np.random.randn(n_total, 2)
    y_imbal = np.array(
        [0] * n_class0 + [1] * n_class1 + [2] * n_class2
    )

    orig_dist = _class_distribution(y_imbal)
    classes = sorted(orig_dist.keys())
    n_classes = len(classes)
    print(f"\nOriginal distribution (total {n_total} samples):")
    for cls, (cnt, ratio) in sorted(orig_dist.items()):
        print(f"  Class {cls}: {cnt:3d} samples ({ratio:.1%})")

    print("\n--- Random Split (train_test_split) ---")
    X_train_rand, X_test_rand, y_train_rand, y_test_rand = train_test_split(
        X_imbal, y_imbal, test_size=0.2, random_seed=10
    )
    train_dist_rand = _class_distribution(y_train_rand)
    test_dist_rand = _class_distribution(y_test_rand)

    print(f"\nTrain set (total {len(y_train_rand)} samples):")
    for cls in classes:
        cnt, ratio = train_dist_rand.get(cls, (0, 0.0))
        print(f"  Class {cls}: {cnt:3d} samples ({ratio:.1%})")
    print(f"\nTest set (total {len(y_test_rand)} samples):")
    for cls in classes:
        cnt, ratio = test_dist_rand.get(cls, (0, 0.0))
        missing = "(MISSING!)" if cnt == 0 else ""
        print(f"  Class {cls}: {cnt:3d} samples ({ratio:.1%}) {missing}")

    print("\n--- Stratified Split (stratified_train_test_split) ---")
    X_train_strat, X_test_strat, y_train_strat, y_test_strat = \
        stratified_train_test_split(
            X_imbal, y_imbal, test_size=0.2, random_seed=10
        )
    train_dist_strat = _class_distribution(y_train_strat)
    test_dist_strat = _class_distribution(y_test_strat)

    print(f"\nTrain set (total {len(y_train_strat)} samples):")
    for cls in classes:
        cnt, ratio = train_dist_strat.get(cls, (0, 0.0))
        print(f"  Class {cls}: {cnt:3d} samples ({ratio:.1%})")
    print(f"\nTest set (total {len(y_test_strat)} samples):")
    for cls in classes:
        cnt, ratio = test_dist_strat.get(cls, (0, 0.0))
        missing = "(MISSING!)" if cnt == 0 else ""
        print(f"  Class {cls}: {cnt:3d} samples ({ratio:.1%}) {missing}")

    print("\n" + "=" * 70)
    print("Comparison Summary")
    print("=" * 70)
    print(f"{'Metric':<25} {'Random Split':<20} {'Stratified Split':<20}")
    print("-" * 70)
    for cls in classes:
        orig_ratio = orig_dist[cls][1]
        rand_test_ratio = test_dist_rand.get(cls, (0, 0.0))[1]
        strat_test_ratio = test_dist_strat.get(cls, (0, 0.0))[1]
        rand_diff = abs(rand_test_ratio - orig_ratio)
        strat_diff = abs(strat_test_ratio - orig_ratio)
        print(
            f"Class {cls} dist deviation:  "
            f"{rand_diff:.1%}                  "
            f"{strat_diff:.1%}"
        )

    n_classes_test_rand = len(test_dist_rand)
    n_classes_test_strat = len(test_dist_strat)
    print(
        f"\nClasses present in test:   "
        f"{n_classes_test_rand}/{n_classes}              "
        f"{n_classes_test_strat}/{n_classes}"
    )

    print("\n" + "=" * 70)
    print("Demo 3: Time Series Split (chronological, no randomness)")
    print("=" * 70)
    np.random.seed(77)
    n_ts = 20
    dates = np.arange(n_ts)
    X_ts = np.random.randn(n_ts, 3)
    y_ts = np.random.randint(0, 3, size=n_ts)

    print(f"\nTime-ordered data ({n_ts} samples):")
    print(f"  Index range: [0 .. {n_ts - 1}]")
    print(f"  X shape: {X_ts.shape}, y shape: {y_ts.shape}")

    X_train_ts, X_test_ts, y_train_ts, y_test_ts = time_series_split(
        X_ts, y_ts, test_size=0.3
    )

    data_summary(
        X_train_ts, X_test_ts, y_train_ts, y_test_ts,
        label="Time Series Split (70% train / 30% test)"
    )

    print(f"\n  Train index range: [0 .. {len(X_train_ts) - 1}]")
    print(f"  Test index range:  [{len(X_train_ts)} .. {n_ts - 1}]")
    print(f"  No data leakage: all train indices < all test indices")

    print("\n" + "=" * 70)
    print("Demo 4: K-Fold Cross-Validation Index Generation")
    print("=" * 70)
    n_cv = 20
    n_folds = 5

    folds = kfold_split(n_cv, n_folds=n_folds, shuffle=True, random_seed=42)

    print(f"\n  Total samples: {n_cv}")
    print(f"  Number of folds: {n_folds}")
    print(f"  Shuffle: True, random_seed: 42")
    print()

    for fold_i, (train_idx, test_idx) in enumerate(folds):
        print(
            f"  Fold {fold_i + 1}: "
            f"train={len(train_idx)} samples, "
            f"test={len(test_idx)} samples "
            f"| test indices: {sorted(test_idx.tolist())}"
        )

    all_test_indices = set()
    for fold_i, (_, test_idx) in enumerate(folds):
        all_test_indices.update(test_idx.tolist())
    print(
        f"\n  Coverage: {len(all_test_indices)}/{n_cv} samples "
        f"appear as test at least once"
    )

    print("\n" + "=" * 70)
    print("Demo 5: Data Summary for all split methods")
    print("=" * 70)
    np.random.seed(55)
    X_demo = np.random.randn(50, 3)
    y_demo = np.array([0] * 30 + [1] * 15 + [2] * 5)

    X_tr1, X_te1, y_tr1, y_te1 = train_test_split(
        X_demo, y_demo, test_size=0.2, random_seed=42
    )
    data_summary(X_tr1, X_te1, y_tr1, y_te1, label="Random Split Summary")

    X_tr2, X_te2, y_tr2, y_te2 = stratified_train_test_split(
        X_demo, y_demo, test_size=0.2, random_seed=42
    )
    data_summary(X_tr2, X_te2, y_tr2, y_te2, label="Stratified Split Summary")

    X_tr3, X_te3, y_tr3, y_te3 = time_series_split(X_demo, y_demo, test_size=0.3)
    data_summary(X_tr3, X_te3, y_tr3, y_te3, label="Time Series Split Summary")
