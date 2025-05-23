import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_recall_fscore_support
from sklearn.ensemble import RandomForestClassifier
import shap

# 加载数据
df = pd.read_csv('data/CICIDS2017_sample.csv')

# 数据可视化
plt.title("Class Distribution")
df.groupby("Label").size().plot(kind='pie', autopct='%.2f', figsize=(20,10))
plt.show()

# Min-max normalization
numeric_features = df.dtypes[df.dtypes != 'object'].index
df[numeric_features] = df[numeric_features].apply(
    lambda x: (x - x.min()) / (x.max() - x.min()))
# Fill empty values by 0
df = df.fillna(0)

labelencoder = LabelEncoder()
df.iloc[:, -1] = labelencoder.fit_transform(df.iloc[:, -1])
df['Label'] = df['Label'].astype(int)

X = df.drop(['Label'], axis=1).values
y = df.iloc[:, -1].values.reshape(-1, 1)
y = np.ravel(y)

X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8, test_size=0.2, random_state=0, stratify=y)

# 训练随机森林模型并预测
rf = RandomForestClassifier(random_state=0)
rf.fit(X_train, y_train)
rf_score = rf.score(X_test, y_test)
y_predict = rf.predict(X_test)
y_true = y_test

print('Accuracy of RF: ' + str(rf_score))
precision, recall, fscore, _ = precision_recall_fscore_support(y_true, y_predict, average='weighted')
print('Precision of RF: ' + str(precision))
print('Recall of RF: ' + str(recall))
print('F1-score of RF: ' + str(fscore))
print(classification_report(y_true, y_predict))

cm = confusion_matrix(y_true, y_predict)
f, ax = plt.subplots(figsize=(5, 5))
sns.heatmap(cm, annot=True, linewidth=0.5, linecolor="red", fmt=".0f", ax=ax)
plt.xlabel("y_pred")
plt.ylabel("y_true")
plt.show()

# 生成SHAP值
explainer = shap.TreeExplainer(rf)
shap_values = explainer.shap_values(X_test)

# 定义特征名称
feature_names = df.drop(['Label'], axis=1).columns.tolist()

# 画出SHAP图
shap.summary_plot(shap_values, features=X_test, feature_names=feature_names, plot_type="bar", show=False)
plt.gcf().set_size_inches(10, 12)
plt.show()

# 指定特征
specified_features = [
    "Flow Duration", "Total Length of Bwd Packets", "Bwd Packet Length Std", "Flow Packets/s", "Avg Bwd Segment Size",
    "Init_Win_bytes_backward", "Bwd Packet Length Mean", "Init_Win_bytes_forward", "Average Packet Size",
    "Max Packet Length", "Fwd IAT Min", "Fwd Packet Length Mean", "Bwd Packets", "Min Packet Length",
    "Max Packet Length", "min_seg_size_forward", "Packet Length Variance", "Flow IAT Max", "Subflow Fwd Bytes",
    "Total Length of Fwd Packets", "Fwd Packet Length Max", "Flow Bytes/s", "Flow IAT Mean", "Subflow Fwd Packets"
]
'''
specified_features = [
    "Total Length of Bwd Packets", "Bwd Packet Length Std", "Avg Bwd Segment Size", "Init_Win_bytes_backward",
    "Bwd Packet Length Mean", "Init_Win_bytes_forward", "Average Packet Size", "Max Packet Length", "Fwd IAT Min",
    "Fwd Packet Length Mean", "Bwd Packets/s", "Min Packet Length", "min_seg_size_forward", "Packet Length Variance",
    "Flow IAT Max", "Subflow Fwd Bytes", "Total Length of Fwd Packets", "Subflow Fwd Packets"
]
'''
for feature in specified_features:
    shap.dependence_plot(feature, shap_values[1], X_test, feature_names=feature_names, show=False)
    plt.gcf().set_size_inches(10, 5)
    plt.show()