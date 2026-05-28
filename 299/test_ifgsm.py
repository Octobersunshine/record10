import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np


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
    if targeted:
        loss = -criterion(output, y)
    else:
        loss = criterion(output, y)
    loss.backward()
    perturbation = epsilon * x_adv.grad.sign()
    x_adv = x_adv + perturbation
    x_adv = torch.clamp(x_adv, 0.0, 1.0)
    return x_adv.detach(), perturbation.detach()


def i_fgsm_attack(model, x, y, epsilon, criterion, alpha=0.03, iterations=10, targeted=False):
    x_adv = x.clone().detach()
    x_orig = x.clone().detach()

    for _ in range(iterations):
        x_adv.requires_grad_(True)
        output = model(x_adv)
        if targeted:
            loss = -criterion(output, y)
        else:
            loss = criterion(output, y)
        loss.backward()

        with torch.no_grad():
            x_adv = x_adv + alpha * x_adv.grad.sign()
            diff = x_adv - x_orig
            diff = torch.clamp(diff, -epsilon, epsilon)
            x_adv = x_orig + diff
            x_adv = torch.clamp(x_adv, 0.0, 1.0)

    perturbation = x_adv - x_orig
    return x_adv.detach(), perturbation.detach()


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    transform = transforms.Compose([
        transforms.ToTensor(),
    ])

    print("Loading MNIST dataset...")
    test_dataset = datasets.MNIST("./data", train=False, download=True, transform=transform)
    test_loader = DataLoader(test_dataset, batch_size=100, shuffle=False)

    model = SimpleCNN().to(device)
    criterion = nn.CrossEntropyLoss()

    model_path = "./data/mnist_cnn.pth"
    try:
        model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
        print("Loaded pre-trained model")
    except:
        print("Training new model...")
        train_dataset = datasets.MNIST("./data", train=True, download=True, transform=transform)
        train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        model.train()
        for epoch in range(3):
            for batch_x, batch_y in train_loader:
                batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                optimizer.zero_grad()
                output = model(batch_x)
                loss = criterion(output, batch_y)
                loss.backward()
                optimizer.step()
        torch.save(model.state_dict(), model_path)
        print(f"Model saved to {model_path}")

    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for x, y in test_loader:
            x, y = x.to(device), y.to(device)
            pred = model(x).argmax(dim=1)
            correct += (pred == y).sum().item()
            total += y.size(0)
    clean_acc = correct / total
    print(f"\nClean test accuracy: {clean_acc:.4f}")

    epsilon = 0.15
    print(f"\n=== Quick Attack Comparison (ε={epsilon}) ===")

    fgsm_correct = 0
    ifgsm_correct = 0
    fgsm_target_correct = 0
    ifgsm_target_correct = 0
    total = 0

    for x, y in test_loader:
        x, y = x.to(device), y.to(device)
        target_labels = (y + 1) % 10

        x_adv_fgsm, _ = fgsm_attack(model, x, y, epsilon, criterion, targeted=False)
        x_adv_ifgsm, _ = i_fgsm_attack(model, x, y, epsilon, criterion, targeted=False)
        x_adv_fgsm_t, _ = fgsm_attack(model, x, target_labels, epsilon, criterion, targeted=True)
        x_adv_ifgsm_t, _ = i_fgsm_attack(model, x, target_labels, epsilon, criterion, targeted=True)

        with torch.no_grad():
            fgsm_pred = model(x_adv_fgsm).argmax(dim=1)
            ifgsm_pred = model(x_adv_ifgsm).argmax(dim=1)
            fgsm_t_pred = model(x_adv_fgsm_t).argmax(dim=1)
            ifgsm_t_pred = model(x_adv_ifgsm_t).argmax(dim=1)

        fgsm_correct += (fgsm_pred == y).sum().item()
        ifgsm_correct += (ifgsm_pred == y).sum().item()
        fgsm_target_correct += (fgsm_t_pred == target_labels).sum().item()
        ifgsm_target_correct += (ifgsm_t_pred == target_labels).sum().item()
        total += y.size(0)
        break

    fgsm_acc = fgsm_correct / total
    ifgsm_acc = ifgsm_correct / total
    fgsm_t_acc = fgsm_target_correct / total
    ifgsm_t_acc = ifgsm_target_correct / total

    print(f"Untargeted - FGSM  accuracy: {fgsm_acc:.4f} (attack success: {1-fgsm_acc:.4f})")
    print(f"Untargeted - I-FGSM accuracy: {ifgsm_acc:.4f} (attack success: {1-ifgsm_acc:.4f})")
    print(f"Targeted   - FGSM  success:  {fgsm_t_acc:.4f}")
    print(f"Targeted   - I-FGSM success:  {ifgsm_t_acc:.4f}")

    print("\n=== Visualization ===")
    x, y = next(iter(test_loader))
    x, y = x[:5].to(device), y[:5].to(device)
    target_labels = (y + 1) % 10

    fig, axes = plt.subplots(4, 5, figsize=(15, 10))
    for i in range(5):
        xi = x[i:i+1]
        yi = y[i:i+1]
        ti = target_labels[i:i+1]

        x_adv_fgsm, _ = fgsm_attack(model, xi, yi, epsilon, criterion, targeted=False)
        x_adv_ifgsm, _ = i_fgsm_attack(model, xi, yi, epsilon, criterion, targeted=False)
        x_adv_target, _ = i_fgsm_attack(model, xi, ti, epsilon, criterion, targeted=True)

        with torch.no_grad():
            orig_pred = model(xi).argmax(dim=1).item()
            fgsm_pred = model(x_adv_fgsm).argmax(dim=1).item()
            ifgsm_pred = model(x_adv_ifgsm).argmax(dim=1).item()
            target_pred = model(x_adv_target).argmax(dim=1).item()

        axes[0, i].imshow(xi.squeeze().cpu().numpy(), cmap="gray")
        axes[0, i].set_title(f"Orig: {yi.item()}\nPred: {orig_pred}")
        axes[0, i].axis("off")

        axes[1, i].imshow(x_adv_fgsm.squeeze().cpu().numpy(), cmap="gray")
        color = "green" if fgsm_pred != yi.item() else "red"
        axes[1, i].set_title(f"FGSM\nPred: {fgsm_pred}", color=color)
        axes[1, i].axis("off")

        axes[2, i].imshow(x_adv_ifgsm.squeeze().cpu().numpy(), cmap="gray")
        color = "green" if ifgsm_pred != yi.item() else "red"
        axes[2, i].set_title(f"I-FGSM\nPred: {ifgsm_pred}", color=color)
        axes[2, i].axis("off")

        axes[3, i].imshow(x_adv_target.squeeze().cpu().numpy(), cmap="gray")
        color = "green" if target_pred == ti.item() else "red"
        axes[3, i].set_title(f"Targeted I-FGSM\nTrue: {yi.item()}→{ti.item()}\nPred: {target_pred}", color=color)
        axes[3, i].axis("off")

    plt.suptitle(f"Adversarial Attack Comparison (ε={epsilon})", fontsize=16)
    plt.tight_layout()
    plt.savefig("attack_comparison_quick.png", dpi=150)
    print("Visualization saved to attack_comparison_quick.png")


if __name__ == "__main__":
    main()
