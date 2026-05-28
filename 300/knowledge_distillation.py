import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader, random_split, Subset
import numpy as np
import time
import copy


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


class TeacherModelA(nn.Module):
    def __init__(self, input_size=784, num_classes=10):
        super(TeacherModelA, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_size, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Linear(64, num_classes)
        )

    def forward(self, x):
        x = x.view(x.size(0), -1)
        return self.net(x)


class TeacherModelB(nn.Module):
    def __init__(self, input_size=784, num_classes=10):
        super(TeacherModelB, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_size, 640),
            nn.BatchNorm1d(640),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(640, num_classes)
        )

    def forward(self, x):
        x = x.view(x.size(0), -1)
        return self.net(x)


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


class SmallResNet(nn.Module):
    def __init__(self, num_classes=10):
        super(SmallResNet, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1))
        )
        self.classifier = nn.Linear(64, num_classes)

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x


class ResNet18Teacher(nn.Module):
    def __init__(self, num_classes=10):
        super(ResNet18Teacher, self).__init__()
        self.resnet = models.resnet18(weights=None)
        self.resnet.conv1 = nn.Conv2d(1, 64, 7, stride=2, padding=3, bias=False)
        self.resnet.fc = nn.Linear(512, num_classes)

    def forward(self, x):
        return self.resnet(x)


class TransferLearningWrapper(nn.Module):
    def __init__(self, base_model, num_features, num_classes, freeze_ratio=0.5):
        super(TransferLearningWrapper, self).__init__()
        self.base = base_model
        layers = list(self.base.children())
        freeze_count = int(len(layers) * freeze_ratio)
        for i, layer in enumerate(layers):
            if i < freeze_count:
                for param in layer.parameters():
                    param.requires_grad = False
        self.adapter = nn.Sequential(
            nn.Linear(num_features, num_features // 2),
            nn.ReLU(),
            nn.Linear(num_features // 2, num_classes)
        )

    def forward(self, x):
        x = x.view(x.size(0), -1)
        with torch.no_grad():
            for layer in self.base.children():
                x_in = x.unsqueeze(1) if x.dim() == 2 and hasattr(layer, 'conv1') else x
                try:
                    x = layer(x_in if x_in.dim() > 2 else x)
                except Exception:
                    x = layer(x.view(x_in.size(0), -1) if x_in.dim() > 2 else x)
                if x.dim() > 2:
                    x = x.view(x.size(0), -1)
        return self.adapter(x)


def distillation_loss(student_logits, teacher_logits, true_labels, T, alpha):
    soft_teacher = F.softmax(teacher_logits / T, dim=1)
    log_soft_student = F.log_softmax(student_logits / T, dim=1)
    distillation = F.kl_div(log_soft_student, soft_teacher, reduction='batchmean') * (T * T)
    student_loss = F.cross_entropy(student_logits, true_labels)
    return alpha * distillation + (1 - alpha) * student_loss


def multi_teacher_distillation_loss(student_logits, teacher_logits_list, weights, true_labels, T, alpha):
    fused_teacher = torch.zeros_like(student_logits)
    for w, t_logits in zip(weights, teacher_logits_list):
        soft_t = F.softmax(t_logits / T, dim=1)
        fused_teacher += w * soft_t
    fused_teacher = fused_teacher / sum(weights)
    log_soft_student = F.log_softmax(student_logits / T, dim=1)
    distillation = F.kl_div(log_soft_student, fused_teacher, reduction='batchmean') * (T * T)
    student_loss = F.cross_entropy(student_logits, true_labels)
    return alpha * distillation + (1 - alpha) * student_loss


def self_distillation_loss(student_logits, past_logits, true_labels, T, alpha, momentum=0.9):
    soft_past = F.softmax(past_logits / T, dim=1)
    log_soft_student = F.log_softmax(student_logits / T, dim=1)
    distillation = F.kl_div(log_soft_student, soft_past, reduction='batchmean') * (T * T)
    student_loss = F.cross_entropy(student_logits, true_labels)
    return alpha * distillation + (1 - alpha) * student_loss


def update_ema(model, ema_model, decay=0.999):
    with torch.no_grad():
        for ema_p, model_p in zip(ema_model.parameters(), model.parameters()):
            ema_p.data.mul_(decay).add_(model_p.data, alpha=1 - decay)


def train_model(model, train_loader, criterion, optimizer, device, epochs=3, name='Model'):
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
        print(f'{name} Epoch {epoch+1}/{epochs}, Loss: {running_loss/len(train_loader):.4f}, Accuracy: {100*correct/total:.2f}%')


def train_single_teacher_kd(student_model, teacher_model, train_loader, optimizer, device, T=2.0, alpha=0.7, epochs=3):
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
        print(f'Student (Single-KD) Epoch {epoch+1}/{epochs}, Loss: {running_loss/len(train_loader):.4f}, Accuracy: {100*correct/total:.2f}%')


def train_multi_teacher_kd(student_model, teacher_models, weights, train_loader, optimizer, device, T=2.0, alpha=0.7, epochs=3):
    for t in teacher_models:
        t.eval()
    student_model.train()
    for epoch in range(epochs):
        running_loss = 0.0
        correct = 0
        total = 0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            with torch.no_grad():
                teacher_outputs_list = [t(images) for t in teacher_models]
            student_outputs = student_model(images)
            loss = multi_teacher_distillation_loss(student_outputs, teacher_outputs_list, weights, labels, T, alpha)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            _, predicted = torch.max(student_outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        print(f'Student (Multi-KD) Epoch {epoch+1}/{epochs}, Loss: {running_loss/len(train_loader):.4f}, Accuracy: {100*correct/total:.2f}%')


def train_self_distillation(student_model, train_loader, optimizer, device, T=2.0, alpha=0.5, ema_decay=0.999, epochs=3):
    ema_model = copy.deepcopy(student_model)
    ema_model.eval()
    for p in ema_model.parameters():
        p.requires_grad = False
    student_model.train()
    for epoch in range(epochs):
        running_loss = 0.0
        correct = 0
        total = 0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            with torch.no_grad():
                past_outputs = ema_model(images)
            student_outputs = student_model(images)
            loss = self_distillation_loss(student_outputs, past_outputs, labels, T, alpha)
            loss.backward()
            optimizer.step()
            update_ema(student_model, ema_model, ema_decay)
            running_loss += loss.item()
            _, predicted = torch.max(student_outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        print(f'Student (Self-KD) Epoch {epoch+1}/{epochs}, Loss: {running_loss/len(train_loader):.4f}, Accuracy: {100*correct/total:.2f}%')


def train_resnet_kd(student_model, teacher_model, train_loader, optimizer, device, T=2.0, alpha=0.7, epochs=3):
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
        print(f'Student (ResNet-KD) Epoch {epoch+1}/{epochs}, Loss: {running_loss/len(train_loader):.4f}, Accuracy: {100*correct/total:.2f}%')


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


def knowledge_distillation():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    full_train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)

    val_size = 10000
    train_size = len(full_train_dataset) - val_size
    train_dataset, val_dataset = random_split(full_train_dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=1000, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=1000, shuffle=False)

    print(f'Dataset splits:')
    print(f'  Training: {len(train_dataset)} samples')
    print(f'  Validation: {len(val_dataset)} samples')
    print(f'  Test: {len(test_dataset)} samples')

    criterion = nn.CrossEntropyLoss()

    # ============================================================
    # Part 1: Train Teacher Models
    # ============================================================
    print('\n' + '='*70)
    print('PART 1: TRAINING TEACHER MODELS')
    print('='*70)

    teacher_main = TeacherModel().to(device)
    train_model(teacher_main, train_loader, criterion, optim.Adam(teacher_main.parameters(), lr=0.001), device, epochs=3, name='Teacher-Main')

    teacher_a = TeacherModelA().to(device)
    train_model(teacher_a, train_loader, criterion, optim.Adam(teacher_a.parameters(), lr=0.001), device, epochs=3, name='Teacher-A (Narrow-Deep)')

    teacher_b = TeacherModelB().to(device)
    train_model(teacher_b, train_loader, criterion, optim.Adam(teacher_b.parameters(), lr=0.001), device, epochs=3, name='Teacher-B (Wide-Shallow)')

    teacher_acc_main, _ = evaluate_model(teacher_main, test_loader, device)
    teacher_acc_a, _ = evaluate_model(teacher_a, test_loader, device)
    teacher_acc_b, _ = evaluate_model(teacher_b, test_loader, device)
    print(f'\nTeacher-Main Accuracy: {teacher_acc_main:.2f}%')
    print(f'Teacher-A (Narrow-Deep) Accuracy: {teacher_acc_a:.2f}%')
    print(f'Teacher-B (Wide-Shallow) Accuracy: {teacher_acc_b:.2f}%')

    # ============================================================
    # Part 2: Temperature Grid Search
    # ============================================================
    best_T, grid_results = grid_search_temperature(teacher_main, train_dataset, val_loader, device, T_candidates=[0.5, 1, 2, 4, 8], alpha=0.7)
    print(f'\n>>> Optimal Temperature T = {best_T}')

    # ============================================================
    # Part 3: Single-Teacher KD
    # ============================================================
    print('\n' + '='*70)
    print('PART 2: SINGLE-TEACHER DISTILLATION')
    print('='*70)

    student_single = StudentModel().to(device)
    train_single_teacher_kd(student_single, teacher_main, train_loader, optim.Adam(student_single.parameters(), lr=0.001), device, T=best_T, alpha=0.7, epochs=3)
    single_acc, single_time = evaluate_model(student_single, test_loader, device)

    # ============================================================
    # Part 4: Multi-Teacher KD
    # ============================================================
    print('\n' + '='*70)
    print('PART 3: MULTI-TEACHER DISTILLATION')
    print('='*70)

    teacher_weights = [0.4, 0.3, 0.3]
    print(f'Teacher weights: Main={teacher_weights[0]}, A={teacher_weights[1]}, B={teacher_weights[2]}')
    student_multi = StudentModel().to(device)
    train_multi_teacher_kd(student_multi, [teacher_main, teacher_a, teacher_b], teacher_weights, train_loader, optim.Adam(student_multi.parameters(), lr=0.001), device, T=best_T, alpha=0.7, epochs=3)
    multi_acc, multi_time = evaluate_model(student_multi, test_loader, device)

    # ============================================================
    # Part 5: Self-Distillation
    # ============================================================
    print('\n' + '='*70)
    print('PART 4: SELF-DISTILLATION (EMA Teacher)')
    print('='*70)

    student_self = StudentModel().to(device)
    train_self_distillation(student_self, train_loader, optim.Adam(student_self.parameters(), lr=0.001), device, T=best_T, alpha=0.5, ema_decay=0.999, epochs=3)
    self_acc, self_time = evaluate_model(student_self, test_loader, device)

    # ============================================================
    # Part 6: ResNet Compression
    # ============================================================
    print('\n' + '='*70)
    print('PART 5: RESNET COMPRESSION (ResNet18 -> SmallCNN)')
    print('='*70)

    resnet_teacher = ResNet18Teacher().to(device)
    train_model(resnet_teacher, train_loader, criterion, optim.Adam(resnet_teacher.parameters(), lr=0.001), device, epochs=2, name='ResNet18-Teacher')
    resnet_teacher_acc, resnet_teacher_time = evaluate_model(resnet_teacher, test_loader, device)
    print(f'ResNet18 Teacher Accuracy: {resnet_teacher_acc:.2f}%')

    small_cnn = SmallResNet().to(device)
    train_resnet_kd(small_cnn, resnet_teacher, train_loader, optim.Adam(small_cnn.parameters(), lr=0.001), device, T=best_T, alpha=0.7, epochs=2)
    small_cnn_acc, small_cnn_time = evaluate_model(small_cnn, test_loader, device)

    # ============================================================
    # Part 7: Baseline
    # ============================================================
    print('\n' + '='*70)
    print('PART 6: BASELINE (Student without KD)')
    print('='*70)

    student_baseline = StudentModel().to(device)
    train_model(student_baseline, train_loader, criterion, optim.Adam(student_baseline.parameters(), lr=0.001), device, epochs=3, name='Student-Baseline')
    baseline_acc, baseline_time = evaluate_model(student_baseline, test_loader, device)

    # ============================================================
    # Final Comparison
    # ============================================================
    student_params = count_parameters(StudentModel())
    teacher_params = count_parameters(teacher_main)
    resnet_params = count_parameters(resnet_teacher)
    small_cnn_params = count_parameters(small_cnn)

    print('\n' + '='*70)
    print('FINAL PERFORMANCE COMPARISON')
    print('='*70)
    print(f'{"Method":<28} {"Accuracy":<12} {"Params":<12} {"Inf(ms)":<10}')
    print('-'*70)
    print(f'{"Teacher-Main":<28} {teacher_acc_main:<12.2f} {teacher_params:<12,} {"-":<10}')
    print(f'{"Teacher-A (Narrow-Deep)":<28} {teacher_acc_a:<12.2f} {count_parameters(teacher_a):<12,} {"-":<10}')
    print(f'{"Teacher-B (Wide-Shallow)":<28} {teacher_acc_b:<12.2f} {count_parameters(teacher_b):<12,} {"-":<10}')
    print(f'{"ResNet18 Teacher":<28} {resnet_teacher_acc:<12.2f} {resnet_params:<12,} {resnet_teacher_time*1000:<10.4f}')
    print('-'*70)
    print(f'{"Student Baseline":<28} {baseline_acc:<12.2f} {student_params:<12,} {baseline_time*1000:<10.4f}')
    print(f'{"Single-KD (T="+str(best_T)+")":<28} {single_acc:<12.2f} {student_params:<12,} {single_time*1000:<10.4f}')
    print(f'{"Multi-KD (T="+str(best_T)+")":<28} {multi_acc:<12.2f} {student_params:<12,} {multi_time*1000:<10.4f}')
    print(f'{"Self-KD (T="+str(best_T)+")":<28} {self_acc:<12.2f} {student_params:<12,} {self_time*1000:<10.4f}')
    print(f'{"ResNet->SmallCNN KD":<28} {small_cnn_acc:<12.2f} {small_cnn_params:<12,} {small_cnn_time*1000:<10.4f}')
    print('-'*70)
    print(f'Single-KD Improvement:  +{single_acc - baseline_acc:.2f}%')
    print(f'Multi-KD Improvement:   +{multi_acc - baseline_acc:.2f}%')
    print(f'Self-KD Improvement:    +{self_acc - baseline_acc:.2f}%')
    print(f'ResNet Compression Rate: {(1 - small_cnn_params/resnet_params)*100:.1f}%')
    print(f'Student Param Reduction: {(1 - student_params/teacher_params)*100:.1f}%')
    print('='*70)

    print('\n' + '='*70)
    print('DISTILLATION STRATEGY ANALYSIS')
    print('='*70)
    print('1. Single-Teacher KD: Classic Hinton approach, one teacher guides student')
    print('   - Best when single strong teacher is available')
    print('   - Simple, stable, well-understood')
    print()
    print('2. Multi-Teacher KD: Weighted fusion of multiple teacher soft labels')
    print('   - Ensembles diverse knowledge (e.g., narrow-deep + wide-shallow)')
    print('   - Weights should reflect teacher quality (better teacher = higher weight)')
    print('   - Can outperform single-teacher when teachers are complementary')
    print()
    print('3. Self-Distillation: EMA of student acts as teacher (Born-Again Network)')
    print('   - No external teacher needed')
    print('   - EMA provides smoother, more stable targets')
    print('   - Effective for model compression without a pre-trained teacher')
    print()
    print('4. ResNet Compression: Large CNN -> Small CNN via KD')
    print('   - Applicable to BERT mini, ResNet compression scenarios')
    print('   - Student architecture can be freely chosen')
    print()
    print('5. Transfer Learning: Pre-trained model + KD to new domain')
    print('   - Freeze early layers, distill from adapted teacher')
    print('='*70)

    return student_multi, {
        'teacher_acc': teacher_acc_main,
        'single_kd_acc': single_acc,
        'multi_kd_acc': multi_acc,
        'self_kd_acc': self_acc,
        'baseline_acc': baseline_acc,
        'resnet_teacher_acc': resnet_teacher_acc,
        'small_cnn_acc': small_cnn_acc,
        'best_T': best_T,
        'grid_search_results': grid_results,
    }


if __name__ == '__main__':
    student_model, metrics = knowledge_distillation()
    torch.save(student_model.state_dict(), 'student_model_kd.pth')
    print('\nStudent model saved as student_model_kd.pth')
