import matplotlib.pyplot as plt
import numpy as np

plt.rcParams['font.sans-serif'] = ['SimHei']  # 使用黑体（或其他系统支持的中文字体）
plt.rcParams['axes.unicode_minus'] = False    # 解决负号'-'显示为方块的问题

data_real=np.load('./real_data.npy').mean(axis=0)

data_Average=np.load('./Average_.npy').mean(axis=0)-data_real
data_Average=data_Average[data_Average!=0]

data_Global=np.load('./Global_GBRT.npy').mean(axis=0)-data_real
data_Global=data_Global[data_Global!=0]

data_Geoarea=np.load('./Geoarea_GBRT.npy').mean(axis=0)-data_real
data_Geoarea=data_Geoarea[data_Geoarea!=0]

data_PCA=np.load('./PCA_GBRT.npy').mean(axis=0)-data_real
data_PCA=data_PCA[data_PCA!=0]

data_GRG=np.load('./GRG_GBRT.npy').mean(axis=0)-data_real
data_GRG=data_GRG[data_GRG!=0]
#如果误差为0，则这个点被删去。需要补回0点数量，保证数组长度为52648（陆地格子数）
padding_size = 52648 - len(data_GRG)
data_GRG = np.pad(data_GRG, (0, padding_size), 'constant', constant_values=(0))
# 陆地格子数量

data_list = [
    data_Average.flatten(),
    data_Global.flatten(),
    data_Geoarea.flatten(),
    data_PCA.flatten(),
    data_GRG.flatten()
]

# 绘制箱线图
plt.figure(figsize=(10, 6))
plt.boxplot(data_list, labels=['Average', 'Global', 'Geoarea', 'PCA', 'GRG'],showfliers=False, showmeans=True)
plt.title('1993——2012年简单平均、四种GBRT方预测结果误差箱线图')
plt.ylabel('误差')
plt.xlabel('Methods')
plt.grid(False)
plt.savefig('./fig2.3.png', dpi=1000, bbox_inches='tight')
plt.show()
plt.close()
