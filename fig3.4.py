#该代码用于绘制1850——2100年预测结果的世界地图

import numpy as np
import matplotlib.pyplot as plt
data_average = (np.load('./Average_.npy').mean(axis=0)).reshape(360,720)  
data_GBRT = (np.load('./GRG_GBRT.npy').mean(axis=0)).reshape(360,720)  

plt.rcParams['font.sans-serif'] = ['SimHei']  # 使用黑体（或其他系统支持的中文字体）
plt.rcParams['axes.unicode_minus'] = False    # 解决负号'-'显示为方块的问题

# 计算 y 轴坐标，表示纬度
y_axis = np.linspace(-90, 90, 360)

# 计算每个纬度上 data_average 和 data_GBRT 的平均值
average_values = np.mean(data_average, axis=1)
GBRT_values = np.mean(data_GBRT, axis=1)

# 创建图表
plt.figure(figsize=(6, 10))

# 绘制 data_average 的折线
plt.plot(average_values, y_axis,  label='Average',lw=1,color='darkblue')

# 绘制 data_GBRT 的折线
plt.plot(GBRT_values, y_axis,  label='GRG', color='darkgreen',lw=1,linestyle='--')
#plt.xlim(0, 4)
# 添加图例
plt.legend()

# 设置图表标题和轴标签
plt.title('2015——2100年GRG_GBRT和简单平均预测结果关于维度的折线图对比')
plt.xlabel('Value')
plt.ylabel('Latitude')

# 显示网格
plt.grid(False)

# 显示图表
plt.savefig('./fig3.4.png', dpi=1000, bbox_inches='tight')  
plt.show()