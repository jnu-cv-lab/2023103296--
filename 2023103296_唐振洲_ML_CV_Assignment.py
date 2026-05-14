# ==============================================
# 传统机器学习手写数字分类 - 完整作业代码
# 数据集：sklearn.datasets.load_digits
# ==============================================
import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

# ======================
# 任务1：数据准备
# ======================
digits = datasets.load_digits()
X = digits.data       # 特征 (1797, 64)
y = digits.target     # 标签 (1797,)
images = digits.images# 图像 (1797,8,8)

print("===== 任务1：数据信息 =====")
print("样本总数：", len(images))
print("单张图像大小：", images[0].shape)
print("类别标签：", np.unique(y))

# 显示前10张样本图
plt.figure(figsize=(10, 3))
for i in range(10):
    plt.subplot(1, 10, i+1)
    plt.imshow(images[i], cmap='gray')
    plt.title(y[i])
    plt.axis('off')
plt.suptitle('任务1：样本图像')
plt.show()

# ======================
# 任务2：数据划分 75%训练 25%测试
# ======================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42
)
print("\n===== 任务2：数据划分 =====")
print("训练集大小：", X_train.shape)
print("测试集大小：", X_test.shape)
print("训练集：用于模型学习参数")
print("测试集：用于评估模型泛化能力")

# ======================
# 任务3：特征表示（8×8 → 64维）
# ======================
print("\n===== 任务3：特征表示 =====")
print("8×8图像按行展开成一维向量 → 64维")
print("传统ML无法直接处理2D图像，必须转为向量")
print("像素特征优点：简单、直接、易提取")
print("像素特征局限：无不变性、受位置/亮度影响大")

# ======================
# 任务4：训练6个模型
# ======================
models = {
    "KNN": KNeighborsClassifier(),
    "Naive Bayes": GaussianNB(),
    "Logistic Regression": LogisticRegression(max_iter=10000),
    "SVM": SVC(),
    "Decision Tree": DecisionTreeClassifier(),
    "Random Forest": RandomForestClassifier()
}

results = {}
y_pred_dict = {}

print("\n===== 任务4：模型训练与测试 =====")
for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    results[name] = acc
    y_pred_dict[name] = y_pred
    print(f"{name:20s} 准确率：{acc:.4f}")

# ======================
# 任务5：准确率表格
# ======================
print("\n===== 任务5：准确率对比表 =====")
print("| 模型 | 测试准确率 |")
print("|------|------------|")
for name, acc in results.items():
    print(f"| {name} | {acc:.4f} |")

# ======================
# 任务6：错误样本分析（选SVM）
# ======================
best_model_name = "SVM"
y_pred = y_pred_dict[best_model_name]
cm = confusion_matrix(y_test, y_pred)

print(f"\n===== 任务6：{best_model_name} 混淆矩阵 =====")

# 绘制混淆矩阵
plt.figure(figsize=(8, 6))
ConfusionMatrixDisplay(cm, display_labels=digits.target_names).plot(cmap='Blues')
plt.title(f'任务6：{best_model_name} 混淆矩阵')
plt.show()

# 找错误样本
error_idx = np.where(y_pred != y_test)[0]
print(f"错误样本数：{len(error_idx)}")

# 显示前8个错误样本
plt.figure(figsize=(10, 4))
for i, idx in enumerate(error_idx[:8]):
    plt.subplot(2, 4, i+1)
    plt.imshow(X_test[idx].reshape(8,8), cmap='gray')
    plt.title(f'T:{y_test[idx]}\nP:{y_pred[idx]}')
    plt.axis('off')
plt.suptitle(f'任务6：{best_model_name} 错误样本')
plt.show()