#本代码用于绘制所有模型留一验证预测结果折线图
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import r2_score

# 读取真实数据
df = pd.read_csv(f'./global_data/real_data1993.csv')
df[df > 0] = 1

# 定义模型名称
name = ['Global_', 'Geoarea_', 'PCA_', 'GRG_']

# 陆地格子数量
land_count = 52648

# 用于计算调整 R 方


# 用于区分陆地与海洋
# 固定图形大小
figsize = (10, 6)


# 绘制预测值
for n in name:
    pred_values = {model: [] for model in ['RF', 'LASSO', 'LR', 'GBRT', 'Average']}
    real_values = []
    for i in range(251):
        for model in pred_values.keys():
            if model == 'Average':
                pred_data = pd.read_csv(f'./全部年份预测/Average_{1850+i}.csv') * df
            else:
                pred_data = pd.read_csv(f'./全部年份预测/{n}{model}_{1850+i}.csv') 
            pred_value = pred_data.sum().sum() / land_count
            pred_values[model].append(pred_value)
    
    time = np.arange(1850, 2101)
    plt.figure(figsize=figsize)
    plt.plot(time, pred_values['RF'], label=f'{n}RF', linestyle=(0, (5, 3, 1, 3)))
    plt.plot(time, pred_values['LASSO'], label=f'{n}LASSO', linestyle='-.')
    plt.plot(time, pred_values['LR'], label=f'{n}LR', linestyle=':')
    plt.plot(time, pred_values['GBRT'], label=f'{n}GBRT', linestyle='--')
    plt.plot(time, pred_values['Average'], label=f'Average')
    plt.ylim(2.4, 5)
    plt.legend(loc='upper left', bbox_to_anchor=(1, 1), borderaxespad=0.)
    plt.xticks(range(1850, 2101, 50))
    plt.title('cVeg: Predict')
    plt.tight_layout()
    plt.savefig(f'./全年份预测折线图/{n}.png', dpi=1000, bbox_inches='tight')
    plt.close()

