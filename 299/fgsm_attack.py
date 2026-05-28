import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np
import time


class SimpleCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, 10),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x


def fgsm_attack(model, x, y, epsilon, criterion, targeted=False):
    x_adv = x.clone().detach().requires_grad_(True)
    output = model(x_adv)
    loss = -criterion(output, y) if targeted else criterion(output, y)
    loss.backward()
    perturbation = epsilon * x_adv.grad.sign()
    x_adv = torch.clamp(x_adv + perturbation, 0.0, 1.0)
    return x_adv.detach(), perturbation.detach()


def i_fgsm_attack(model, x, y, epsilon, criterion, alpha=0.03, iterations=10, targeted=False):
    x_adv = x.clone().detach()
    x_orig = x.clone().detach()
    for _ in range(iterations):
        x_adv.requires_grad_(True)
        output = model(x_adv)
        loss = -criterion(output, y) if targeted else criterion(output, y)
        loss.backward()
        with torch.no_grad():
            x_adv = x_adv + alpha * x_adv.grad.sign()
            x_adv = torch.clamp(x_orig + torch.clamp(x_adv - x_orig, -epsilon, epsilon), 0.0, 1.0)
    perturbation = x_adv - x_orig
    return x_adv.detach(), perturbation.detach()


def pgd_attack(model, x, y, epsilon, criterion, alpha=None, iterations=40, num_restarts=1, targeted=False):
    if alpha is None:
        alpha = epsilon / 4
    x_orig = x.clone().detach()
    best_x_adv = x_orig.clone()
    best_n_success = 0

    for _ in range(num_restarts):
        x_adv = x_orig + torch.empty_like(x_orig).uniform_(-epsilon, epsilon)
        x_adv = torch.clamp(x_adv, 0.0, 1.0).detach()

        for _ in range(iterations):
            x_adv.requires_grad_(True)
            output = model(x_adv)
            loss = -criterion(output, y) if targeted else criterion(output, y)
            loss.backward()
            with torch.no_grad():
                x_adv = x_adv + alpha * x_adv.grad.sign()
                x_adv = torch.clamp(x_orig + torch.clamp(x_adv - x_orig, -epsilon, epsilon), 0.0, 1.0)

        with torch.no_grad():
            pred = model(x_adv).argmax(dim=1)
            if targeted:
                n_success = (pred == y).sum().item()
            else:
                n_success = (pred != y).sum().item()

        if n_success > best_n_success:
            best_n_success = n_success
            best_x_adv = x_adv.clone()

    perturbation = best_x_adv - x_orig
    return best_x_adv.detach(), perturbation.detach()


def cw_l2_attack(model, x, y, criterion, c=100.0, kappa=0.0, lr=0.01, iterations=1000,
                  targeted=False):
    x_orig = x.clone().detach()
    batch_size = x.size(0)
    device = x.device

    w = torch.atanh(torch.clamp(x_orig * 2 - 1, -0.999, 0.999)).detach().clone().requires_grad_(True)
    optimizer = optim.Adam([w], lr=lr)

    best_adv = x_orig.clone()
    best_l2 = torch.full((batch_size,), float("inf"), device=device)

    for step in range(iterations):
        x_adv = (torch.tanh(w) + 1) / 2
        output = model(x_adv)

        one_hot = torch.zeros_like(output).scatter_(1, y.unsqueeze(1), 1)
        correct_logit = (output * one_hot).sum(dim=1)
        wrong_logit = (output - 1e4 * one_hot).max(dim=1)[0]

        if targeted:
            f_val = torch.clamp(correct_logit - wrong_logit + kappa, min=0)
        else:
            f_val = torch.clamp(wrong_logit - correct_logit + kappa, min=0)

        l2_per_sample = ((x_adv - x_orig) ** 2).sum(dim=(1, 2, 3))
        loss = (l2_per_sample + c * f_val).mean()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        with torch.no_grad():
            pred = output.argmax(dim=1)
            if targeted:
                attack_ok = pred == y
            else:
                attack_ok = pred != y

            improved = attack_ok & (l2_per_sample < best_l2)
            best_l2 = torch.where(improved, l2_per_sample, best_l2)
            mask = improved.float().view(-1, 1, 1, 1)
            best_adv = mask * x_adv + (1 - mask) * best_adv

        if step > 0 and step % 200 == 0:
            if (best_l2 == float("inf")).all():
                c = min(c * 5, 10000.0)

    failed = best_l2.isinf()
    if failed.any():
        with torch.no_grad():
            x_adv_final = (torch.tanh(w) + 1) / 2
            final_pred = model(x_adv_final).argmax(dim=1)
            if targeted:
                final_ok = final_pred == y
            else:
                final_ok = final_pred != y

            for i in range(batch_size):
                if failed[i] and final_ok[i]:
                    best_adv[i] = x_adv_final[i]
                    best_l2[i] = ((x_adv_final[i] - x_orig[i]) ** 2).sum()

    still_failed = best_l2.isinf()
    if still_failed.any():
        x_adv_pgd, _ = pgd_attack(model, x, y, epsilon=0.2, criterion=criterion,
                                   alpha=0.02, iterations=50, targeted=targeted)
        for i in range(batch_size):
            if still_failed[i]:
                best_adv[i] = x_adv_pgd[i]
                best_l2[i] = ((x_adv_pgd[i] - x_orig[i]) ** 2).sum()

    perturbation = best_adv - x_orig
    return best_adv.detach(), perturbation.detach()


ATTACK_FUNCTIONS = {
    "fgsm": lambda model, x, y, eps, crit, tgt: fgsm_attack(model, x, y, eps, crit, targeted=tgt),
    "i_fgsm": lambda model, x, y, eps, crit, tgt: i_fgsm_attack(model, x, y, eps, crit, targeted=tgt),
    "pgd": lambda model, x, y, eps, crit, tgt: pgd_attack(model, x, y, eps, crit, targeted=tgt),
    "cw": lambda model, x, y, eps, crit, tgt: cw_l2_attack(model, x, y, crit, targeted=tgt),
}


def train_model(model, train_loader, epochs=3, lr=0.001, device="cpu"):
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    model.train()
    for epoch in range(epochs):
        total_loss = 0.0
        correct = 0
        total = 0
        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            optimizer.zero_grad()
            output = model(batch_x)
            loss = criterion(output, batch_y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * batch_x.size(0)
            pred = output.argmax(dim=1)
            correct += (pred == batch_y).sum().item()
            total += batch_x.size(0)
        print(f"Epoch {epoch + 1}/{epochs}  Loss: {total_loss / total:.4f}  Acc: {correct / total:.4f}")
    return model


def adversarial_train(model, train_loader, epsilon, epochs=3, lr=0.001, device="cpu"):
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        correct = 0
        total = 0

        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)

            model.eval()
            x_adv, _ = pgd_attack(model, batch_x, batch_y, epsilon, criterion,
                                  alpha=epsilon / 4, iterations=5, num_restarts=1, targeted=False)
            x_adv = x_adv.detach()
            model.train()

            mixed_x = torch.cat([batch_x, x_adv], dim=0)
            mixed_y = torch.cat([batch_y, batch_y], dim=0)

            optimizer.zero_grad()
            output = model(mixed_x)
            loss = criterion(output, mixed_y)
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * mixed_x.size(0)
            pred = output.argmax(dim=1)
            correct += (pred == mixed_y).sum().item()
            total += mixed_x.size(0)

        print(f"Adv-Train Epoch {epoch + 1}/{epochs}  Loss: {total_loss / total:.4f}  Acc: {correct / total:.4f}")

    return model


def evaluate(model, test_loader, device="cpu"):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for x, y in test_loader:
            x, y = x.to(device), y.to(device)
            correct += (model(x).argmax(dim=1) == y).sum().item()
            total += y.size(0)
    return correct / total


def evaluate_under_attack(model, test_loader, epsilon, criterion, device="cpu",
                          attack_type="pgd", targeted=False, max_batches=None):
    model.eval()
    attack_fn = ATTACK_FUNCTIONS[attack_type]
    correct = 0
    total = 0
    cw_batch_size = 10

    for batch_idx, (x, y) in enumerate(test_loader):
        if max_batches is not None and batch_idx >= max_batches:
            break
        x, y = x.to(device), y.to(device)
        attack_labels = ((y + 1) % 10) if targeted else y

        if attack_type == "cw":
            all_pred = []
            for j in range(0, x.size(0), cw_batch_size):
                x_sub = x[j:j + cw_batch_size]
                y_sub = attack_labels[j:j + cw_batch_size]
                x_adv_sub, _ = cw_l2_attack(model, x_sub, y_sub, criterion, targeted=targeted)
                with torch.no_grad():
                    all_pred.append(model(x_adv_sub).argmax(dim=1))
            pred = torch.cat(all_pred, dim=0)
        else:
            x_adv, _ = attack_fn(model, x, attack_labels, epsilon, criterion, targeted)
            with torch.no_grad():
                pred = model(x_adv).argmax(dim=1)

        if targeted:
            correct += (pred == attack_labels).sum().item()
        else:
            correct += (pred == y).sum().item()
        total += y.size(0)

    return correct / total


def robustness_report(model, test_loader, epsilon, criterion, device="cpu", max_batches=5):
    model.eval()
    attacks = ["fgsm", "i_fgsm", "pgd", "cw"]
    results = {}

    clean_acc = evaluate(model, test_loader, device)
    results["clean"] = clean_acc

    print(f"\n{'=' * 65}")
    print(f"  ROBUSTNESS EVALUATION REPORT")
    print(f"{'=' * 65}")
    print(f"  Epsilon: {epsilon}  |  Clean Accuracy: {clean_acc:.4f}")
    print(f"{'-' * 65}")
    print(f"  {'Attack':<12} {'Mode':<12} {'Accuracy':<12} {'Atk Success':<14}")
    print(f"{'-' * 65}")

    for atk_name in attacks:
        for targeted in [False, True]:
            mode = "Targeted" if targeted else "Untargeted"
            key = f"{atk_name}_{mode.lower()}"
            acc = evaluate_under_attack(model, test_loader, epsilon, criterion, device,
                                       attack_type=atk_name, targeted=targeted,
                                       max_batches=max_batches)
            results[key] = acc
            if targeted:
                print(f"  {atk_name:<12} {mode:<12} {'---':<12} {acc:<14.4f}")
            else:
                print(f"  {atk_name:<12} {mode:<12} {acc:<12.4f} {1 - acc:<14.4f}")

    worst_case_acc = min(results.get(f"{a}_untargeted", 1.0) for a in attacks)
    robustness_gap = clean_acc - worst_case_acc
    robustness_ratio = worst_case_acc / (clean_acc + 1e-8)

    print(f"{'=' * 65}")
    print(f"  Worst-case adversarial accuracy: {worst_case_acc:.4f}")
    print(f"  Robustness gap (clean - worst):  {robustness_gap:.4f}")
    print(f"  Robustness ratio (worst/clean):  {robustness_ratio:.4f}")
    print(f"{'=' * 65}\n")

    return results


def plot_attack_comparison(std_results, rob_results, attack_names, epsilon):
    labels = []
    std_vals = []
    rob_vals = []
    for name in attack_names:
        labels.append(name.upper())
        std_vals.append(1 - std_results.get(f"{name}_untargeted", 0))
        rob_vals.append(1 - rob_results.get(f"{name}_untargeted", 0))

    x = np.arange(len(labels))
    width = 0.35
    fig, ax = plt.subplots(figsize=(10, 6))
    b1 = ax.bar(x - width / 2, std_vals, width, label="Standard Model", color="#e74c3c", alpha=0.85)
    b2 = ax.bar(x + width / 2, rob_vals, width, label="Adversarially Trained", color="#2ecc71", alpha=0.85)
    ax.set_ylabel("Attack Success Rate (Untargeted)")
    ax.set_title(f"Attack Success Rate Comparison (ε={epsilon})")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend()
    ax.set_ylim(0, 1.05)
    ax.grid(axis="y", alpha=0.3)
    for bars in [b1, b2]:
        for bar in bars:
            h = bar.get_height()
            ax.annotate(f"{h:.3f}", xy=(bar.get_x() + bar.get_width() / 2, h),
                        xytext=(0, 3), textcoords="offset points", ha="center", fontsize=9)
    plt.tight_layout()
    plt.savefig("robustness_comparison.png", dpi=150)
    print("Saved robustness_comparison.png")


def plot_robust_accuracy_curves(model_std, model_rob, test_loader, criterion, device="cpu", max_batches=3):
    epsilons = [0.0, 0.05, 0.1, 0.15, 0.2]
    attacks = [("FGSM", "fgsm"), ("I-FGSM", "i_fgsm"), ("PGD", "pgd")]
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    for idx, (atk_label, atk_key) in enumerate(attacks):
        std_accs, rob_accs = [], []
        for eps in epsilons:
            std_accs.append(evaluate_under_attack(model_std, test_loader, eps, criterion, device,
                                                  attack_type=atk_key, targeted=False, max_batches=max_batches))
            rob_accs.append(evaluate_under_attack(model_rob, test_loader, eps, criterion, device,
                                                  attack_type=atk_key, targeted=False, max_batches=max_batches))
        axes[idx].plot(epsilons, std_accs, "o-", color="#e74c3c", linewidth=2, markersize=7, label="Standard")
        axes[idx].plot(epsilons, rob_accs, "s-", color="#2ecc71", linewidth=2, markersize=7, label="Adversarially Trained")
        axes[idx].set_xlabel("Epsilon")
        axes[idx].set_ylabel("Accuracy")
        axes[idx].set_title(f"{atk_label} Attack")
        axes[idx].legend()
        axes[idx].grid(True, alpha=0.3)
        axes[idx].set_ylim(0, 1.05)

    plt.suptitle("Robust Accuracy vs. Epsilon: Standard vs. Adversarially Trained", fontsize=14)
    plt.tight_layout()
    plt.savefig("robust_accuracy_curves.png", dpi=150)
    print("Saved robust_accuracy_curves.png")


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    transform = transforms.Compose([transforms.ToTensor()])
    print("Loading MNIST dataset...")
    train_dataset = datasets.MNIST("./data", train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST("./data", train=False, download=True, transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=50, shuffle=False)

    criterion = nn.CrossEntropyLoss()
    epsilon = 0.15
    model_path = "./data/mnist_cnn_quick.pth"
    adv_model_path = "./data/mnist_cnn_adv_quick.pth"

    print("\n" + "=" * 60)
    print("  PHASE 1: Train Standard Model")
    print("=" * 60)
    model_std = SimpleCNN().to(device)
    try:
        model_std.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
        print("Loaded pre-trained standard model")
    except:
        model_std = train_model(model_std, train_loader, epochs=3, lr=0.001, device=device)
        torch.save(model_std.state_dict(), model_path)

    clean_std = evaluate(model_std, test_loader, device)
    print(f"Standard model clean accuracy: {clean_std:.4f}")

    print("\n" + "=" * 60)
    print("  PHASE 2: Robustness Evaluation (Standard Model)")
    print("=" * 60)
    t0 = time.time()
    std_results = robustness_report(model_std, test_loader, epsilon, criterion, device, max_batches=4)
    print(f"Evaluation time: {time.time() - t0:.1f}s")

    print("\n" + "=" * 60)
    print("  PHASE 3: Adversarial Training (PGD-based)")
    print("=" * 60)
    model_rob = SimpleCNN().to(device)
    try:
        model_rob.load_state_dict(torch.load(adv_model_path, map_location=device, weights_only=True))
        print("Loaded pre-trained adversarially trained model")
    except:
        model_rob.load_state_dict(model_std.state_dict())
        model_rob = adversarial_train(model_rob, train_loader, epsilon, epochs=2, lr=0.0005, device=device)
        torch.save(model_rob.state_dict(), adv_model_path)

    clean_rob = evaluate(model_rob, test_loader, device)
    print(f"Robust model clean accuracy: {clean_rob:.4f}")

    print("\n" + "=" * 60)
    print("  PHASE 4: Robustness Evaluation (Adversarially Trained)")
    print("=" * 60)
    t0 = time.time()
    rob_results = robustness_report(model_rob, test_loader, epsilon, criterion, device, max_batches=4)
    print(f"Evaluation time: {time.time() - t0:.1f}s")

    print("\n" + "=" * 60)
    print("  PHASE 5: Comparison & Visualization")
    print("=" * 60)
    plot_attack_comparison(std_results, rob_results, ["fgsm", "i_fgsm", "pgd", "cw"], epsilon)
    plot_robust_accuracy_curves(model_std, model_rob, test_loader, criterion, device, max_batches=2)

    print("\n" + "=" * 60)
    print("  FINAL SUMMARY")
    print("=" * 60)
    print(f"  {'Metric':<36} {'Standard':<12} {'Adversarial':<12} {'Delta':<10}")
    print(f"  {'-' * 70}")
    print(f"  {'Clean Accuracy':<36} {clean_std:<12.4f} {clean_rob:<12.4f} {clean_rob - clean_std:<+10.4f}")
    for atk in ["fgsm", "i_fgsm", "pgd", "cw"]:
        s = std_results.get(f"{atk}_untargeted", 0)
        r = rob_results.get(f"{atk}_untargeted", 0)
        label = atk.upper() + " Untargeted Acc"
        print(f"  {label:<36} {s:<12.4f} {r:<12.4f} {r - s:<+10.4f}")
    print(f"  {'-' * 70}")
    std_worst = min(std_results.get(f"{a}_untargeted", 0) for a in ["fgsm", "i_fgsm", "pgd", "cw"])
    rob_worst = min(rob_results.get(f"{a}_untargeted", 0) for a in ["fgsm", "i_fgsm", "pgd", "cw"])
    print(f"  {'Worst-case Adversarial Acc':<36} {std_worst:<12.4f} {rob_worst:<12.4f} {rob_worst - std_worst:<+10.4f}")
    print(f"  {'Robustness Ratio (worst/clean)':<36} {std_worst / (clean_std + 1e-8):<12.4f} {rob_worst / (clean_rob + 1e-8):<12.4f}")
    print("=" * 60)


if __name__ == "__main__":
    main()
