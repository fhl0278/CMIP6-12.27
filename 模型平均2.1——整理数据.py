import numpy as np
import pandas as pd
import os
df = pd.read_csv('./global_data/real_data1993.csv')
df[df > 0] = 1
model_names = [
    "ACCESS-ESM1-5", "BCC-CSM2-MR", "CESM2", "CESM2-WACCM", "CMCC-CM2-SR5", "CMCC-ESM2", "CNRM-ESM2-1",
    "CanESM5", "CanESM5-CanOE", "E3SM-1-1", "EC-Earth3-CC", "EC-Earth3-Veg", "EC-Earth3-Veg-LR", "INM-CM4-8",
    "INM-CM5-0", "KIOST-ESM", "IPSL-CM6A-LR", "MIROC-ES2L", "MPI-ESM1-2-LR", "NorESM2-LM", "NorESM2-MM", "UKESM1-0-LL"
]

# 数据点总数
total_points = 259200
years = list(range(1993, 2013))  # 20年的时间范围

# 加载数据

num_years = len(years)
num_features = len(model_names) + 1  # 模型数量加上真实数据列

# 初始化一个形状为 (num_years, num_samples, num_features) 的 NumPy 数组
data_train = np.empty((num_years, total_points, num_features))

for idx, year in enumerate(years):
    # 读取模型数据并展平
    for i, name in enumerate(model_names):
        data = (pd.read_csv(f'./global_data/{name}_data{year}.csv')*df).values.flatten()
        data_train[idx, :, i] = data
    
    # 读取真实数据并展平
    real_data = pd.read_csv(f'./global_data/real_data{year}.csv').values.flatten()
    data_train[idx, :, -1] = real_data
np.save('./data_train.npy', data_train)
#(20,259200,23)
years = list(range(1850, 2101))
data_pred = np.empty((len(years), total_points, 22))

for idx, year in enumerate(years):
    # 读取模型数据并展平
    for i, name in enumerate(model_names):
        data = (pd.read_csv(f'./global_data/{name}_data{year}.csv')*df).values.flatten()
        data_pred[idx, :, i] = data

np.save('./data_pred.npy', data_pred)
#(251,259200,22)

