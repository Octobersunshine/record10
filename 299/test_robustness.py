import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from fgsm_attack import SimpleCNN, robustness_report

device = 'cpu'
model_std = SimpleCNN().to(device)
model_std.load_state_dict(torch.load('./data/mnist_cnn_quick.pth', map_location=device, weights_only=True))

model_rob = SimpleCNN().to(device)
model_rob.load_state_dict(torch.load('./data/mnist_cnn_adv_quick.pth', map_location=device, weights_only=True))

transform = transforms.Compose([transforms.ToTensor()])
test_dataset = datasets.MNIST('./data', train=False, download=True, transform=transform)
test_loader = DataLoader(test_dataset, batch_size=50, shuffle=False)
criterion = nn.CrossEntropyLoss()
epsilon = 0.15

print('=== Standard Model Robustness ===')
std_results = robustness_report(model_std, test_loader, epsilon, criterion, device, max_batches=2)

print('\n=== Adversarially Trained Model Robustness ===')
rob_results = robustness_report(model_rob, test_loader, epsilon, criterion, device, max_batches=2)

print('\n=== Improvement Summary ===')
attacks = ["fgsm", "i_fgsm", "pgd", "cw"]
std_worst = min(std_results.get(f"{a}_untargeted", 1.0) for a in attacks)
rob_worst = min(rob_results.get(f"{a}_untargeted", 1.0) for a in attacks)
print(f'Worst-case acc (standard): {std_worst:.4f}')
print(f'Worst-case acc (robust):   {rob_worst:.4f}')
print(f'Improvement:               {rob_worst - std_worst:+.4f}')
