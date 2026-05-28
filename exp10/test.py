import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split
import matplotlib.pyplot as plt
import numpy as np

# ====================== 1. 环境检查 ======================
print("PyTorch 版本:", torch.__version__)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("使用设备:", device)

# ====================== 2. 数据加载与预处理 ======================
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

# 下载数据集
train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)

# 训练集 → 训练集 + 验证集
train_size = int(0.8 * len(train_dataset))
val_size = len(train_dataset) - train_size
train_dataset, val_dataset = random_split(train_dataset, [train_size, val_size])

# DataLoader
batch_size = 64
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# 类别名称
classes = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']

# ====================== 显示8张训练图像（改为保存图片） ======================
def imshow(img, title):
    img = img.numpy().squeeze()
    plt.imshow(img, cmap='gray')
    plt.title(title)
    plt.axis('off')

plt.figure(figsize=(12, 4))
for i in range(8):
    img, label = train_dataset[i]
    plt.subplot(1, 8, i+1)
    imshow(img, classes[label])
plt.savefig("train_samples.png", dpi=300, bbox_inches='tight')
plt.close()  # 关闭当前图，避免后续绘图被影响

# ====================== 3. 定义CNN模型 ======================
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, 3, 1)
        self.conv2 = nn.Conv2d(16, 32, 3, 1)
        self.pool = nn.MaxPool2d(2)
        self.fc1 = nn.Linear(32 * 5 * 5, 128)
        self.fc2 = nn.Linear(128, 10)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = x.view(-1, 32 * 5 * 5)
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x

model = CNN().to(device)

# ====================== 4. 损失函数 & 优化器 ======================
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
epochs = 5

# 记录曲线
train_loss_list = []
train_acc_list = []
val_loss_list = []
val_acc_list = []

# ====================== 5. 训练 + 验证 ======================
for epoch in range(epochs):
    # 训练
    model.train()
    train_loss = 0.0
    train_correct = 0
    train_total = 0

    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        _, predicted = torch.max(outputs, 1)
        train_total += labels.size(0)
        train_correct += (predicted == labels).sum().item()

    train_loss_avg = train_loss / len(train_loader)
    train_acc = train_correct / train_total

    # 验证
    model.eval()
    val_loss = 0.0
    val_correct = 0
    val_total = 0

    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            val_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            val_total += labels.size(0)
            val_correct += (predicted == labels).sum().item()

    val_loss_avg = val_loss / len(val_loader)
    val_acc = val_correct / val_total

    # 保存
    train_loss_list.append(train_loss_avg)
    train_acc_list.append(train_acc)
    val_loss_list.append(val_loss_avg)
    val_acc_list.append(val_acc)

    print(f"Epoch [{epoch+1}/{epochs}] | "
          f"Train Loss: {train_loss_avg:.4f} Acc: {train_acc:.4f} | "
          f"Val Loss: {val_loss_avg:.4f} Acc: {val_acc:.4f}")

# ====================== 6. 测试集评估 ======================
model.eval()
test_loss = 0.0
test_correct = 0
test_total = 0

with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        loss = criterion(outputs, labels)
        test_loss += loss.item()
        _, predicted = torch.max(outputs, 1)
        test_total += labels.size(0)
        test_correct += (predicted == labels).sum().item()

test_loss_avg = test_loss / len(test_loader)
test_acc = test_correct / test_total

print("\n===== 测试集结果 =====")
print(f"Test Loss: {test_loss_avg:.4f}")
print(f"Test Accuracy: {test_acc:.4f}")

# ====================== 显示8张测试图（改为保存图片） ======================
plt.figure(figsize=(12, 5))
for i in range(8):
    img, label = test_dataset[i]
    img_tensor = img.unsqueeze(0).to(device)
    output = model(img_tensor)
    _, pred = torch.max(output, 1)
    pred = pred.item()

    plt.subplot(1, 8, i+1)
    imshow(img.cpu(), f"True:{classes[label]}\nPred:{classes[pred]}")
plt.savefig("test_predictions.png", dpi=300, bbox_inches='tight')
plt.close()

# ====================== 7. 绘制训练曲线（改为保存图片） ======================
# Loss曲线
plt.figure(figsize=(10, 4))
plt.plot(range(1, epochs+1), train_loss_list, label='Train Loss', marker='o')
plt.plot(range(1, epochs+1), val_loss_list, label='Val Loss', marker='s')
plt.title('Loss Curve')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.savefig("loss_curve.png", dpi=300, bbox_inches='tight')
plt.close()

# Acc曲线
plt.figure(figsize=(10, 4))
plt.plot(range(1, epochs+1), train_acc_list, label='Train Acc', marker='o')
plt.plot(range(1, epochs+1), val_acc_list, label='Val Acc', marker='s')
plt.title('Accuracy Curve')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.savefig("accuracy_curve.png", dpi=300, bbox_inches='tight')
plt.close()