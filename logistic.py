import numpy as np
from sklearn.linear_model import LogisticRegression


train_dir=r"C:\Users\E507\Desktop\logsitic\horseColicTraining.txt"
test_dir=r"C:\Users\E507\Desktop\logsitic\horseColicTest.txt"

#=====================
# 1. 数据读取函数
#=====================
def load_dataset(filename):
    data = np.loadtxt(filename)
    X = data[:, :-1]   # 特征
    y = data[:, -1]    # 标签
    return X, y

#=====================
# 2. 缺失值处理函数
#   （缺失值替换为该列均值）
#=====================
def replace_nan_with_mean(X):
    for i in range(X.shape[1]):
        col = X[:, i]
        # 选择非0的数作为有效特征
        valid = col[col != 0]
        if len(valid) > 0:
            mean_val = np.mean(valid)
            col[col == 0] = mean_val
            X[:, i] = col
    return X

#=====================
# 3. 主流程
#=====================
# 读取训练集
def main():
    X_train,y_train = load_dataset(train_dir)
    X_test,y_test = load_dataset(test_dir)
    clf = LogisticRegression(solver='lbfgs',max_iter=3000)
    clf.fit(X_train,y_train)
    y_pred_sk = clf.predict(X_test)
    acc_sk=np.mean(y_pred_sk==y_test)
    print(acc_sk)
if __name__ =="__main__":
    main()
# 读取测试集


#=====================
# 4. 构建并训练逻辑回归模型
#=====================


#=====================
# 5. 测试集预测
#=====================


#=====================
# 6. 计算准确率
#=====================

