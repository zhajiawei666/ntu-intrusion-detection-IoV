import pandas as pd
from rdflib import Graph, URIRef
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, precision_score, f1_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import time

# 加载 RDF 图数据集
g = Graph()
g.parse("network_traffic.rdf", format="turtle")

# 提取特征和标签
X, y = [], []
for subj, pred, obj in g:
    # 提取特征，这里假设每个节点的特征由其属性组成
    features = []
    for attr in g.predicates(subj):
        # 如果属性是标签属性，则跳过
        if attr == URIRef("http://example.org/network#hasAttackType"):
            continue
        # 提取属性值
        value = g.value(subj, attr)
        # 尝试将值转换为浮点数
        try:
            float_value = float(value)
            features.append(float_value)
        except (ValueError, TypeError):
            # 如果无法转换为浮点数，则忽略该值
            pass
    X.append(features)
    # 提取标签，假设标签存储在 http://example.org/network#hasAttackType 属性中
    label = str(g.value(subj, URIRef("http://example.org/network#hasAttackType")))
    y.append(label)

# 检查特征数组长度
feature_lengths = [len(features) for features in X]
unique_lengths = set(feature_lengths)
print(f"Unique feature lengths: {unique_lengths}")

# 确保特征长度一致
expected_length = max(unique_lengths)  # 使用最大长度作为统一长度
X = [features if len(features) == expected_length else features[:expected_length] + [0] * (expected_length - len(features)) for features in X]

print("Features of the last instance: ", features)
print("All classes:", set(y))

# 使用 pandas 计算和可视化各类攻击的比例
df = pd.DataFrame({'Label': y})
label_counts = df['Label'].value_counts()
print("Label distribution:\n", label_counts)

# 可视化标签分布
plt.figure(figsize=(10, 6))
sns.barplot(x=label_counts.index, y=label_counts.values)
plt.title('Distribution of Attack Types')
plt.xlabel('Attack Type')
plt.ylabel('Frequency')
plt.xticks(rotation=45)
plt.show()

# 绘制饼状图
plt.figure(figsize=(8, 8))
plt.pie(label_counts, labels=label_counts.index, autopct='%1.1f%%', startangle=140)
plt.title('Class Distribution')
plt.axis('equal')
plt.show()

# 使用LabelEncoder对标签进行编码
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

# 查看所有标签及其对应的编码
labels = label_encoder.classes_
print("标签及其对应的编码：")
for index, label in enumerate(labels):
    print(f"{index}: {label}")

# 查看Infiltration标签编码后的类别
infiltration_index = label_encoder.transform(['Infiltration'])[0]
print(f"Infiltration标签编码后对应的类别：{infiltration_index}")

# 划分数据集
X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)

# 打印训练集和测试集的大小
print(f"Training set size: {len(X_train)}")
print(f"Test set size: {len(X_test)}")

# 计算模型训练和预测的整体执行时间
start_time = time.time()

# 训练随机森林模型
clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train, y_train)

# 模型评估
y_pred = clf.predict(X_test)

end_time = time.time()
execution_time = end_time - start_time
print(f"模型训练和预测的整体执行时间：{execution_time:.2f}秒")

# 生成分类报告
report = classification_report(y_test, y_pred, target_names=label_encoder.classes_, output_dict=True)
print(classification_report(y_test, y_pred, target_names=label_encoder.classes_))
print(classification_report(y_test, y_pred))

# 计算准确率
accuracy = accuracy_score(y_test, y_pred)
print("准确率：", accuracy)

# 计算混淆矩阵
conf_matrix = confusion_matrix(y_test, y_pred)
print("混淆矩阵：")
print(conf_matrix)

# 可视化混淆矩阵
plt.figure(figsize=(12, 9))  # 调整图像大小
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', annot_kws={"size": 15})  # 增加注释字体大小
plt.xlabel('Predicted', fontsize=16)  # 调整X轴字体大小
plt.ylabel('Actual', fontsize=16)  # 调整Y轴字体大小
plt.title('Confusion Matrix', fontsize=18)  # 调整标题字体大小
plt.show()

# 计算假阳性率
FP = conf_matrix.sum(axis=0) - np.diag(conf_matrix)
TN = conf_matrix.sum() - (conf_matrix.sum(axis=1) + FP)
false_positive_rate = FP / (FP + TN)
print("假阳性率：", false_positive_rate)

# 计算误报率
false_alarm_rate = FP / (FP + np.diag(conf_matrix))
print("误报率：", false_alarm_rate)

# 计算平均准确度（Precision）、平均F1-Score、平均召回率（Recall）
average_precision = precision_score(y_test, y_pred, average='macro') * 100
average_f1_score = f1_score(y_test, y_pred, average='macro') * 100
average_recall = recall_score(y_test, y_pred, average='macro') * 100

print(f"平均准确度 (Precision): {average_precision:.2f}%")
print(f"平均F1-Score: {average_f1_score:.2f}%")
print(f"平均召回率 (Recall): {average_recall:.2f}%")
print(f"平均误报率 (False Alarm Rate): {false_alarm_rate.mean() * 100:.2f}%")

# 可视化混淆矩阵
f, ax = plt.subplots(figsize=(5, 5))
sns.heatmap(conf_matrix, annot=True, linewidths=0.5, linecolor="red", fmt=".0f", ax=ax)
plt.xlabel("Predicted labels")
plt.ylabel("True labels")
plt.show()

# 计算每个类别的检测准确率和误报率
accuracy_per_class = conf_matrix.diagonal() / conf_matrix.sum(axis=1)
false_positive_rate_per_class = (conf_matrix.sum(axis=0) - conf_matrix.diagonal()) / conf_matrix.sum(axis=0)
false_alarm_rate_per_class = (conf_matrix.sum(axis=0) - conf_matrix.diagonal()) / (conf_matrix.sum(axis=0))

# 准备柱状图数据
categories = label_encoder.classes_
x = np.arange(len(categories))
bar_width = 0.28

# 绘制柱状图
fig, ax = plt.subplots(figsize=(20, 12))

#bar1 = ax.bar(x - 2 * bar_width, [report[category]['precision'] * 100 for category in categories], bar_width, label='Precision')
#bar2 = ax.bar(x - bar_width, [report[category]['recall'] * 100 for category in categories], bar_width, label='Recall')
bar3 = ax.bar(x, [report[category]['f1-score'] * 100 for category in categories], bar_width, label='F1 Score')
bar4 = ax.bar(x + bar_width, accuracy_per_class * 100, bar_width, label='Accuracy')
bar5 = ax.bar(x + 2 * bar_width, false_positive_rate_per_class * 100, bar_width, label='False Alarm Rate')

# 在每个柱状上面标明具体数字
def add_labels(bars):
    for bar in bars:
        height = bar.get_height()
        ax.annotate(f'{height:.2f}%',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom',  fontsize=12)

#add_labels(bar1)
#add_labels(bar2)
add_labels(bar3)
add_labels(bar4)
add_labels(bar5)

ax.set_xlabel('Category', fontsize=14)
ax.set_ylabel('Score (%)', fontsize=14)
ax.set_title('Detection Metrics by Category', fontsize=16)
ax.set_xticks(x)
ax.set_xticklabels(categories, fontsize=12)
ax.legend(loc='center right', fontsize=12)
# 自动调整子图参数，使之填充整个图形区域
plt.tight_layout()
plt.show()

# 准备折线图数据
fig, ax = plt.subplots(figsize=(18, 10))

# 绘制准确率折线图
ax.plot(x, accuracy_per_class * 100, marker='o', label='Accuracy', color='blue')
# 在每个点上标注准确率数值
for i, txt in enumerate(accuracy_per_class * 100):
    ax.annotate(f'{txt:.6f}%', (x[i], accuracy_per_class[i] * 100), textcoords="offset points", xytext=(0, 10), ha='center', color='blue')

# 绘制误报率折线图
ax.plot(x, false_alarm_rate_per_class * 100, marker='o', label='False Alarm Rate', color='red')
# 在每个点上标注误报率数值
for i, txt in enumerate(false_alarm_rate_per_class * 100):
    ax.annotate(f'{txt:.6f}%', (x[i], false_alarm_rate_per_class[i] * 100), textcoords="offset points", xytext=(0, -30), ha='center', color='red')

# 绘制F1-score折线图
ax.plot(x, [report[category]['f1-score'] * 100 for category in categories], marker='o', label='F1 Score', color='green')
# 在每个点上标注F1-score数值
for i, txt in enumerate([report[category]['f1-score'] * 100 for category in categories]):
    ax.annotate(f'{txt:.6f}%', (x[i], txt), textcoords="offset points", xytext=(0, -50), ha='center', color='green')

ax.set_xlabel('Category')
ax.set_ylabel('Percentage (%)')
ax.set_title('Accuracy, False Alarm Rate, and F1 Score by Category')
ax.set_xticks(x)
ax.set_xticklabels(categories)
ax.legend(loc='center right')

plt.show()