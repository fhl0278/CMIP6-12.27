import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression, LassoCV

# 读取训练数据
data_train = np.load('./data_train.npy')
df = pd.read_csv(f'./global_data/real_data1993.csv')
df[df > 0] = 1
df.columns = range(720)
# 定义模型列表
models = [
    ('LR', LinearRegression()),
    ('GBRT', GradientBoostingRegressor(n_estimators=30, random_state=0, max_depth=3)),
    ('LASSO', LassoCV(tol=1e-3, max_iter=1000)),
    ('RF', RandomForestRegressor(n_estimators=30, random_state=0, max_depth=3))
]

# 存储每个模型的训练结果
regressors = {name: [None] * 20 for name, _ in models}

# 循环训练和预测
for model_name, model in models:
    for i in range(20):
        # 准备训练数据
        train = np.delete(data_train, i, axis=0).reshape(19 * 259200, 23)
        non_zero_indices = train[:,-1] != 0

# 使用布尔索引选择对应的行
        t = train[non_zero_indices]
        X_train = t[:, :-1]
        y_train = t[:, -1]

        # 训练模型
        regressor = model
        regressor.fit(X_train, y_train)
        regressors[model_name][i] = regressor

        # 准备测试数据
        test_data = data_train[i, :, :-1]  # 假设 data_train 的最后一列是真实值
        #(259200,22)
        # 预测
        y_pred = regressor.predict(test_data)
        y_pred = pd.DataFrame(y_pred.reshape(360, 720)) * df
        # 保存预测结果
        y_pred.to_csv(f'./预测数据/Global_{model_name}{1993 + i}.csv', index=False)

        print(f'{model_name} - Year {1993 + i} completed')