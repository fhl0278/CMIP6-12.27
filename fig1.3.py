#该代码用于绘制1850——2100年预测结果的世界地图

import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.basemap import Basemap
from matplotlib.colors import LinearSegmentedColormap, BoundaryNorm
import gc
plt.rcParams['font.sans-serif'] = ['SimHei']  # 使用黑体（或其他系统支持的中文字体）
plt.rcParams['axes.unicode_minus'] = False    # 解决负号'-'显示为方块的问题

# 初始化经纬度网格数据
lon = np.linspace(0, 360, 720)
lat = np.linspace(-90, 90, 360)
lon1, lat1 = np.meshgrid(lon, lat)

# 定义颜色映射及级别
#levels_predicted = [-6,-4,-2.5,-1.4, -0.73, -0.12, -0.06, 0, 0.06, 0.12,0.73,  1.4, 2.5,4,6]
levels_predicted = [-2.5,-2,-1.5,-1, -0.6, -0.3, -0.1, 0, 0.1, 0.3,  0.6, 1,1.5,2,2.5]
#colors = ["maroon", "red", "orangered", "gold", "yellow", "lemonchiffon", "powderblue", "lightskyblue", "deepskyblue", "dodgerblue", "blue", "mediumblue", "darkblue"]
colors =['midnightblue',
    'darkblue',
 'mediumblue',
 'blue',
 'dodgerblue',
 'deepskyblue',
 'lightskyblue',
 'powderblue',
 'pink',
 'salmon',
 'tomato',
 'orangered',
 'red',
 'firebrick',
 'maroon',
 'darkred']
cmaps = LinearSegmentedColormap.from_list('mylist', colors, N=16)
norm = BoundaryNorm(levels_predicted, 16, extend='both')
# 绘制地图
file_path1 = './real_data.npy'
file_path2 = './GRG_GBRT.npy'
both1 = (np.load(file_path2).mean(axis=0)-np.load(file_path1).mean(axis=0)).reshape(360,720)  # 读取预测数据并与真实数据相乘
both1 = np.ma.masked_where(both1 == 0, both1)
# 创建地图对象并设置地图参数
fig, ax = plt.subplots(figsize=(10, 4))
map = Basemap(llcrnrlon=0, llcrnrlat=-90, urcrnrlon=360, urcrnrlat=90, resolution='c', projection='cyl', ax=ax)
xi, yi = map(lon1, lat1)

# 绘制地图背景
map.drawmapboundary(fill_color='white')
map.shadedrelief(scale=0.1)
map.drawparallels(np.linspace(-90, 90, 5), labels=[1, 0, 0, 0], color='gray', fontsize=8)
map.drawmeridians(np.linspace(-180, 180, 6), labels=[0, 0, 0, 1], color='gray', fontsize=8)

# 绘制数据
both1 = np.ma.masked_invalid(both1)  # 掩盖无效值
cs = map.pcolormesh(xi, yi, both1, cmap=cmaps, norm=norm)

# 添加图例
cbar = map.colorbar(cs, location='bottom', pad="10%", ticks=levels_predicted)
cbar.set_label("AGC ($kg$/$m^{-2}$)", labelpad=20, y=0.5, rotation=0, fontsize=12, weight='normal')
plt.title('1993-2012年GRG_GBRT与观测值的误差（平均）')
# 保存图像
plt.savefig('./fig1.3.png', dpi=1000, bbox_inches='tight')  
plt.show()
plt.close(fig)

# 释放内存
del both1, fig, ax, map, xi, yi, cs, cbar
gc.collect()


