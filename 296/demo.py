from sklearn.datasets import load_digits
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from learning_curve import plot_learning_curve


def print_diagnosis_report(result):
    report = result["diagnosis_report"]
    if report is None:
        print("Diagnosis disabled.")
        return

    diag = report["diagnosis"].replace("_", " ").title()
    sev = report["severity"].title()

    print("\n" + "=" * 72)
    print(f"  FIT DIAGNOSIS: {diag}  (Severity: {sev})")
    print("=" * 72)
    print(f"  Note: {report['severity_note']}")
    print()

    print("  Signals Summary:")
    for k, v in report["signals_summary"].items():
        print(f"    {k:<26} : {v}")
    print()

    print("  Recommended Actions (priority order):")
    for rec in report["recommendations"]:
        print(f"    [{rec['priority']}] {rec['action']}")
        print(f"        {rec['details']}")
    print("=" * 72 + "\n")


X, y = load_digits(return_X_y=True)

print("=" * 72)
print("  CASE 1: DecisionTree (max_depth=5) — likely overfitting / high variance")
print("=" * 72)
clf1 = DecisionTreeClassifier(max_depth=5, random_state=42)
result1 = plot_learning_curve(
    clf1,
    X,
    y,
    n_repeats=10,
    scoring="accuracy",
    confidence=0.95,
    title="DecisionTree (max_depth=5) - Digits",
)
result1["fig"].savefig("learning_curve_tree.png", dpi=150)

sizes = result1["train_sizes"]
print(f"\nTotal folds (5 splits x 10 repeats): {result1['n_folds']}")
print(f"{'Samples':>8}  {'Train_Mean':>10}  {'Train_95%CI':>20}  {'Val_Mean':>10}  {'Val_95%CI':>20}")
print("-" * 78)
for i, s in enumerate(sizes):
    t_lo, t_hi = result1["train_ci_lo"][i], result1["train_ci_hi"][i]
    v_lo, v_hi = result1["val_ci_lo"][i], result1["val_ci_hi"][i]
    print(
        f"{s:>8}  {result1['train_mean'][i]:>10.4f}  [{t_lo:.4f}, {t_hi:.4f}]"
        f"  {result1['val_mean'][i]:>10.4f}  [{v_lo:.4f}, {v_hi:.4f}]"
    )
print_diagnosis_report(result1)

print("=" * 72)
print("  CASE 2: DecisionTree (max_depth=2) — likely underfitting / high bias")
print("=" * 72)
clf2 = DecisionTreeClassifier(max_depth=2, random_state=42)
result2 = plot_learning_curve(
    clf2,
    X,
    y,
    n_repeats=10,
    scoring="accuracy",
    confidence=0.95,
    title="DecisionTree (max_depth=2) - Digits",
)
result2["fig"].savefig("learning_curve_tree_underfit.png", dpi=150)
print_diagnosis_report(result2)

print("=" * 72)
print("  CASE 3: LogisticRegression (L2, C=1.0) — likely good fit or more data needed")
print("=" * 72)
clf3 = LogisticRegression(C=1.0, max_iter=5000, random_state=42)
result3 = plot_learning_curve(
    clf3,
    X,
    y,
    n_repeats=5,
    scoring="accuracy",
    confidence=0.95,
    title="LogisticRegression (C=1.0) - Digits",
)
result3["fig"].savefig("learning_curve_lr.png", dpi=150)
print_diagnosis_report(result3)

print("\nAll charts saved as PNG files.")
