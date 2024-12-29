#本代码用于绘制所有模型留一验证预测结果折线图
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams['font.sans-serif'] = ['SimHei']  # 使用黑体（或其他系统支持的中文字体）
plt.rcParams['axes.unicode_minus'] = False    # 解决负号'-'显示为方块的问题

# 定义模型名称
name = ['Global', 'Geoarea', 'PCA_', 'GRG_']
data=np.load('./real_data.npy')
data_Global=np.load('./Global_GBRT.npy')
data_Geoarea=np.load('./Geoarea_GBRT.npy')
data_PCA=np.load('./PCA_GBRT.npy')
data_GRG=np.load('./GRG_GBRT.npy')
# 陆地格子数量
land_count = 52648

# 用于计算调整 R 方
N = land_count#52648
p = 22

figsize = (10, 6)
# 绘制预测值
time = np.arange(1993, 2013)
plt.figure(figsize=figsize)

plt.plot(time, data_Global.sum(axis=1)/N, label='Global',marker='^', markersize=3,color='greenyellow', lw=1,linestyle=(0, (5, 3, 1, 3)))
plt.plot(time, data_Geoarea.sum(axis=1)/N, label='Geoarea',marker='p', markersize=3,color='lawngreen',lw=1, linestyle='-.')
plt.plot(time, data_PCA.sum(axis=1)/N, label='PCA',marker='*', markersize=3,color='forestgreen', lw=1,linestyle=':')
plt.plot(time, data_GRG.sum(axis=1)/N, label='GRG', marker='o', markersize=3, color='darkgreen',lw=1,linestyle='--')
plt.plot(time, data.sum(axis=1)/N, label='real',lw=1,color='red')

plt.ylim(2.5, 2.8)
plt.legend(loc='upper left', bbox_to_anchor=(1, 1), borderaxespad=0.)
plt.xticks(range(1993, 2013, 2))
plt.title('1993——2012年观测值、四种GBRT方法留一验证折线图')
plt.tight_layout()
plt.savefig('./fig2.2.png', dpi=1000, bbox_inches='tight')
plt.show()
plt.close()
