import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split, Subset
import numpy as np
import time


class TeacherModel(nn.Module):
    def __init__(self, input_size=784, hidden_size=512, num_classes=10):
        super(TeacherModel, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.bn1 = nn.BatchNorm1d(hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size // 2)
        self.bn2 = nn.BatchNorm1d(hidden_size // 2)
        self.fc3 = nn.Linear(hidden_size // 2, hidden_size // 4)
        self.bn3 = nn.BatchNorm1d(hidden_size // 4)
        self.fc4 = nn.Linear(hidden_size // 4, num_classes)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.3)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = self.dropout(self.relu(self.bn1(self.fc1(x))))
        x = self.dropout(self.relu(self.bn2(self.fc2(x))))
        x = self.dropout(self.relu(self.bn3(self.fc3(x))))
        x = self.fc4(x)
        return x


class StudentModel(nn.Module):
    def __init__(self, input_size=784, hidden_size=128, num_classes=10):
        super(StudentModel, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, num_classes)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x


def distillation_loss(student_logits, teacher_logits, true_labels, T, alpha):
    soft_teacher = nn.functional.softmax(teacher_logits / T, dim=1)
    log_soft_student = nn.functional.log_softmax(student_logits / T, dim=1)
    distillation = nn.functional.kl_div(log_soft_student, soft_teacher, reduction='batchmean') * (T * T)
    student_loss = nn.functional.cross_entropy(student_logits, true_labels)
    return alpha * distillation + (1 - alpha) * student_loss


def train_teacher_model(model, train_loader, criterion, optimizer, device, epochs=3):
    model.train()
    for epoch in range(epochs):
        running_loss = 0.0
        correct = 0
        total = 0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        print(f'Teacher Epoch {epoch+1}/{epochs}, Loss: {running_loss/len(train_loader):.4f}, Accuracy: {100*correct/total:.2f}%')


def train_student_kd(student_model, teacher_model, train_loader, optimizer, device, T=5.0, alpha=0.7, epochs=3):
    teacher_model.eval()
    student_model.train()
    for epoch in range(epochs):
        running_loss = 0.0
        correct = 0
        total = 0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            with torch.no_grad():
                teacher_outputs = teacher_model(images)
            student_outputs = student_model(images)
            loss = distillation_loss(student_outputs, teacher_outputs, labels, T, alpha)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            _, predicted = torch.max(student_outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        print(f'Student KD Epoch {epoch+1}/{epochs}, Loss: {running_loss/len(train_loader):.4f}, Accuracy: {100*correct/total:.2f}%')


def train_student_baseline(student_model, train_loader, criterion, optimizer, device, epochs=3):
    student_model.train()
    for epoch in range(epochs):
        running_loss = 0.0
        correct = 0
        total = 0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = student_model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        print(f'Student Baseline Epoch {epoch+1}/{epochs}, Loss: {running_loss/len(train_loader):.4f}, Accuracy: {100*correct/total:.2f}%')


def evaluate_model(model, test_loader, device):
    model.eval()
    correct = 0
    total = 0
    inference_time = 0.0
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            start_time = time.time()
            outputs = model(images)
            inference_time += time.time() - start_time
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    accuracy = 100 * correct / total
    avg_inference_time = inference_time / len(test_loader.dataset)
    return accuracy, avg_inference_time


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def grid_search_temperature(teacher_model, train_subset, val_loader, device, T_candidates=[0.5, 1, 2, 4, 8], alpha=0.7):
    print('\n=== Temperature Grid Search ===')
    print(f'Candidate T values: {T_candidates}')
    print(f'Alpha: {alpha}')
    print('-'*60)

    train_subset_loader = DataLoader(train_subset, batch_size=128, shuffle=True)
    results = {}

    for T in T_candidates:
        print(f'\nTesting T = {T}...')
        student_model = StudentModel().to(device)
        optimizer = optim.Adam(student_model.parameters(), lr=0.001)
        teacher_model.eval()
        best_val_acc = 0.0
        for epoch in range(2):
            student_model.train()
            for images, labels in train_subset_loader:
                images, labels = images.to(device), labels.to(device)
                optimizer.zero_grad()
                with torch.no_grad():
                    teacher_outputs = teacher_model(images)
                student_outputs = student_model(images)
                loss = distillation_loss(student_outputs, teacher_outputs, labels, T, alpha)
                loss.backward()
                optimizer.step()

            student_model.eval()
            val_correct = 0
            val_total = 0
            with torch.no_grad():
                for images, labels in val_loader:
                    images, labels = images.to(device), labels.to(device)
                    outputs = student_model(images)
                    _, predicted = torch.max(outputs.data, 1)
                    val_total += labels.size(0)
                    val_correct += (predicted == labels).sum().item()
            val_acc = 100 * val_correct / val_total
            if val_acc > best_val_acc:
                best_val_acc = val_acc
        results[T] = best_val_acc
        print(f'T = {T}, Best Validation Accuracy: {val_acc:.2f}%')

    best_T = max(results, key=results.get)
    best_acc = results[best_T]

    print('\n' + '-'*60)
    print('Grid Search Results Summary:')
    print('-'*60)
    print(f'{"T":<10} {"Validation Accuracy":<20}')
    print('-'*30)
    for T in sorted(results.keys()):
        marker = ' <-- BEST' if T == best_T else ''
        print(f'{T:<10} {results[T]:<20.2f}{marker}')
    print('-'*60)
    print(f'\nRecommended T = {best_T} (Validation Accuracy: {best_acc:.2f}%)')

    return best_T, results


def knowledge_distillation():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    full_train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)

    indices = torch.randperm(len(full_train_dataset))[:20000]
    small_train = Subset(full_train_dataset, indices[:15000])
    small_val = Subset(full_train_dataset, indices[15000:])
    test_loader = DataLoader(test_dataset, batch_size=1000, shuffle=False)
    val_loader = DataLoader(small_val, batch_size=1000, shuffle=False)
    train_loader = DataLoader(small_train, batch_size=128, shuffle=True)

    print(f'Dataset splits (fast mode):')
    print(f'  Training: {len(small_train)} samples')
    print(f'  Validation: {len(small_val)} samples')
    print(f'  Test: {len(test_dataset)} samples')

    teacher_model = TeacherModel().to(device)
    criterion = nn.CrossEntropyLoss()
    teacher_optimizer = optim.Adam(teacher_model.parameters(), lr=0.001)

    print('\n=== Training Teacher Model ===')
    train_teacher_model(teacher_model, train_loader, criterion, teacher_optimizer, device, epochs=3)

    print('\n=== Evaluating Teacher Model ===')
    teacher_acc, teacher_time = evaluate_model(teacher_model, test_loader, device)
    teacher_params = count_parameters(teacher_model)
    print(f'Teacher Accuracy: {teacher_acc:.2f}%')
    print(f'Teacher Parameters: {teacher_params:,}')
    print(f'Teacher Avg Inference Time: {teacher_time*1000:.4f}ms')

    T_candidates = [0.5, 1, 2, 4, 8]
    best_T, grid_results = grid_search_temperature(teacher_model, small_train, val_loader, device, T_candidates=T_candidates, alpha=0.7)

    print(f'\n=== Training Student Model with Knowledge Distillation (T={best_T}) ===')
    student_kd = StudentModel().to(device)
    student_kd_optimizer = optim.Adam(student_kd.parameters(), lr=0.001)
    train_student_kd(student_kd, teacher_model, train_loader, student_kd_optimizer, device, T=best_T, alpha=0.7, epochs=3)

    print('\n=== Evaluating Student Model (KD) ===')
    student_kd_acc, student_kd_time = evaluate_model(student_kd, test_loader, device)
    student_params = count_parameters(student_kd)
    print(f'Student KD Accuracy: {student_kd_acc:.2f}%')
    print(f'Student Parameters: {student_params:,}')
    print(f'Student Avg Inference Time: {student_kd_time*1000:.4f}ms')

    print('\n=== Training Student Model Baseline (without KD) ===')
    student_baseline = StudentModel().to(device)
    student_baseline_optimizer = optim.Adam(student_baseline.parameters(), lr=0.001)
    train_student_baseline(student_baseline, train_loader, criterion, student_baseline_optimizer, device, epochs=3)

    print('\n=== Evaluating Student Model (Baseline) ===')
    student_baseline_acc, student_baseline_time = evaluate_model(student_baseline, test_loader, device)
    print(f'Student Baseline Accuracy: {student_baseline_acc:.2f}%')

    print('\n' + '='*70)
    print('PERFORMANCE COMPARISON')
    print('='*70)
    print(f'{"Model":<22} {"Accuracy":<12} {"Params":<12} {"Inf Time(ms)":<15}')
    print('-'*70)
    kd_model_label = f'Student (KD, T={best_T})'
    print(f'{"Teacher":<22} {teacher_acc:<12.2f} {teacher_params:<12,} {teacher_time*1000:<15.4f}')
    print(f'{kd_model_label:<22} {student_kd_acc:<12.2f} {student_params:<12,} {student_kd_time*1000:<15.4f}')
    print(f'{"Student (Baseline)":<22} {student_baseline_acc:<12.2f} {student_params:<12,} {student_baseline_time*1000:<15.4f}')
    print('-'*70)
    print(f'KD Improvement over Baseline: {student_kd_acc - student_baseline_acc:.2f}%')
    print(f'Teacher-Student Gap: {teacher_acc - student_kd_acc:.2f}%')
    print(f'Parameter Reduction: {(1 - student_params/teacher_params)*100:.2f}%')
    print(f'Speedup: {teacher_time/student_kd_time:.2f}x')
    print('='*70)

    print('\n' + '='*70)
    print('TEMPERATURE ANALYSIS')
    print('='*70)
    print('Temperature T controls the softness of teacher labels:')
    print('  - Small T (→1): Softmax outputs become sharper, approaching hard labels')
    print('  - Large T (→8): Softmax outputs become more uniform, may lose information')
    print('  - Optimal T balances information preservation and useful soft guidance')
    print('-'*70)
    print(f'Recommended T = {best_T}')
    print(f'Test Accuracy with T={best_T}: {student_kd_acc:.2f}%')
    print('='*70)

    return student_kd, {
        'teacher_acc': teacher_acc,
        'student_kd_acc': student_kd_acc,
        'student_baseline_acc': student_baseline_acc,
        'teacher_params': teacher_params,
        'student_params': student_params,
        'teacher_time': teacher_time,
        'student_time': student_kd_time,
        'best_T': best_T,
        'grid_search_results': grid_results
    }


if __name__ == '__main__':
    student_model, metrics = knowledge_distillation()
    torch.save(student_model.state_dict(), 'student_model_kd.pth')
    print('\nStudent model saved as student_model_kd.pth')
