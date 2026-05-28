import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import sys
sys.path.insert(0, ".")

from fgsm_attack import (SimpleCNN, fgsm_attack, i_fgsm_attack, pgd_attack,
                          cw_l2_attack, train_model, evaluate, adversarial_train,
                          robustness_report, evaluate_under_attack)

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    transform = transforms.Compose([transforms.ToTensor()])
    test_dataset = datasets.MNIST("./data", train=False, download=True, transform=transform)
    test_loader = DataLoader(test_dataset, batch_size=50, shuffle=False)
    train_dataset = datasets.MNIST("./data", train=True, download=True, transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
    criterion = nn.CrossEntropyLoss()
    epsilon = 0.15

    model = SimpleCNN().to(device)
    model_path = "./data/mnist_cnn.pth"
    try:
        model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
        print("Loaded pre-trained model")
    except:
        print("Training model...")
        model = train_model(model, train_loader, epochs=3, lr=0.001, device=device)
        torch.save(model.state_dict(), model_path)

    model.eval()
    clean_acc = evaluate(model, test_loader, device)
    print(f"Clean accuracy: {clean_acc:.4f}")

    print("\n=== Testing all 4 attack methods (1 batch) ===")
    x, y = next(iter(test_loader))
    x, y = x.to(device), y.to(device)

    attacks = [
        ("FGSM", lambda: fgsm_attack(model, x, y, epsilon, criterion, targeted=False)),
        ("I-FGSM", lambda: i_fgsm_attack(model, x, y, epsilon, criterion, targeted=False)),
        ("PGD", lambda: pgd_attack(model, x, y, epsilon, criterion, targeted=False)),
        ("CW-L2", lambda: cw_l2_attack(model, x, y, criterion, targeted=False)),
    ]

    for name, atk_fn in attacks:
        x_adv, pert = atk_fn()
        with torch.no_grad():
            pred = model(x_adv).argmax(dim=1)
        acc = (pred == y).float().mean().item()
        l2 = (pert ** 2).sum(dim=(1,2,3)).sqrt().mean().item()
        linf = pert.abs().max().item()
        print(f"  {name:<10} Acc: {acc:.4f}  L2: {l2:.4f}  Linf: {linf:.4f}")

    print("\n=== Quick robustness_report (2 batches) ===")
    results = robustness_report(model, test_loader, epsilon, criterion, device, max_batches=2)

    print("\n=== Adversarial training (1 epoch) ===")
    model_rob = SimpleCNN().to(device)
    model_rob.load_state_dict(model.state_dict())
    model_rob = adversarial_train(model_rob, train_loader, epsilon, epochs=1, lr=0.001, device=device)
    rob_clean = evaluate(model_rob, test_loader, device)
    print(f"Robust model clean accuracy: {rob_clean:.4f}")

    print("\n=== Robust model evaluation (2 batches) ===")
    rob_results = robustness_report(model_rob, test_loader, epsilon, criterion, device, max_batches=2)

    print("\n=== SUMMARY ===")
    print(f"Standard model - Clean: {clean_acc:.4f}, PGD untargeted acc: {results.get('pgd_untargeted', 0):.4f}")
    print(f"Robust model   - Clean: {rob_clean:.4f}, PGD untargeted acc: {rob_results.get('pgd_untargeted', 0):.4f}")
    print("\n=== ALL TESTS PASSED ===")

if __name__ == "__main__":
    main()
