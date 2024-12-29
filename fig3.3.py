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
levels_predicted = [0, 0.01,0.36, 0.44, 0.53, 0.72, 1.1, 2.2,  4, 8,20]
colors = ["#FFFFFF", "#F0FFFF", "#D0F0F6", "#AEEBEB", "#8AD3D8", "#6EB8B3", "#4A9C8D", "#2C7F6D", "#006B3F", "#004A30", "#00352A", "#00241F", "#001D18"]
cmaps = LinearSegmentedColormap.from_list('mylist', colors, N=13)
norm = BoundaryNorm(levels_predicted, 13, extend='both')
# 绘制地图
file_path = './GRG_GBRT.npy'
both1 = (np.load(file_path).mean(axis=0)).reshape(360,720)  # 读取预测数据并与真实数据相乘
both1[both1==0]=-1
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
plt.title('2015——2100年GRG_GBRT预测结果地图（平均后）')
# 保存图像
plt.savefig('./fig3.3.png', dpi=1000, bbox_inches='tight')  
plt.show()
plt.close(fig)

# 释放内存
del both1, fig, ax, map, xi, yi, cs, cbar
gc.collect()


