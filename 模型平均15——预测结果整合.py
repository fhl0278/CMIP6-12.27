import numpy as np
import pandas as pd
df = pd.read_csv('./global_data/real_data1993.csv')
df[df > 0] = 1

model_names = [f'{prefix}{method}' for prefix in ['Global_', 'Geoarea_', 'PCA_', 'GRG_'] for method in ['LASSO', 'LR', 'RF', 'GBRT']]
model_names.append('Average_')
model_names.append('real_data')
# 数据点总数
total_points = 259200

'''
years = list(range(1993, 2013))  # 20年的时间范围

# 加载数据

num_years = len(years)
num_features = len(model_names) + 1  # 模型数量加上真实数据列

# 初始化一个形状为 (num_years, num_samples, num_features) 的 NumPy 数组
data = np.empty((num_years, total_points))

for i, name in enumerate(model_names):
    for idx, year in enumerate(years):
    # 读取模型数据并展平
        data[idx, :] = (pd.read_csv(f'./预测数据/{name}{year}.csv')*df).values.flatten()
    np.save(f'./留一验证结果/{name}.npy', data)
'''
years = list(range(2015, 2101))
num_years = len(years)
data = np.empty((num_years, total_points))
for i, name in enumerate(model_names):
    for idx, year in enumerate(years):
    # 读取模型数据并展平
        data[idx, :] = (pd.read_csv(f'./全部年份预测/{name}{year}.csv')*df).values.flatten()
    np.save(f'./未来预测数据/{name}.npy', data)

