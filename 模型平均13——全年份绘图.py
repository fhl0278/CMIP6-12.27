

#该代码用于绘制1850——2100年预测结果的世界地图

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from mpl_toolkits.basemap import Basemap
from matplotlib.colors import LinearSegmentedColormap, BoundaryNorm
import os
import gc
import time

# 读取真实数据，并将所有正值设为1（二值化）
def load_real_data():
    df = pd.read_csv('./global_data/real_data1993.csv')
    df[df > 0] = 1
    return df.values  # 转换为 NumPy 数组以节省内存

df = load_real_data()

# 定义模型名称组合
model_names = [f'{prefix}{method}' for prefix in ['Global_', 'Geoarea_', 'PCA_', 'GRG_'] for method in ['LASSO', 'LR', 'RF', 'GBRT']]
model_names.append('Average_')
# 初始化经纬度网格数据
lon = np.linspace(0, 360, 720)
lat = np.linspace(-90, 90, 360)
lon1, lat1 = np.meshgrid(lon, lat)

# 定义颜色映射及级别
levels_predicted = [0, 0.01, 0.05, 0.1, 0.5, 1, 2, 4, 8, 12, 16, 50]
colors = ["#F0FFFF", "#FFFFFF", "#D0F0F6", "#AEEBEB", "#8AD3D8", "#6EB8B3", "#4A9C8D", "#2C7F6D", "#006B3F", "#004A30", "#00352A", "#00241F", "#001D18"]
cmaps = LinearSegmentedColormap.from_list('mylist', colors, N=13)
norm = BoundaryNorm(levels_predicted, 13, extend='both')

# 绘制地图
def process_model(model_name):
    output_dir = f'./全年份绘图/{model_name}'
    os.makedirs(output_dir, exist_ok=True)  # 创建输出目录

    for i in range(251):  # 循环遍历年份
        year = 1850 + i
        file_path = f'./全部年份预测/{model_name}{year}.csv'
        if os.path.exists(file_path):  # 检查文件是否存在
            with open(file_path, 'r') as file:
                both1 = pd.read_csv(file).values * df  # 读取预测数据并与真实数据相乘
            both1[both1 < 0] = 0  # 将负值设为0

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

            # 保存图像
            plt.savefig(os.path.join(output_dir, f'{year}.png'), dpi=300, bbox_inches='tight')  # 降低 DPI
            plt.close(fig)

            # 释放内存
            del both1, fig, ax, map, xi, yi, cs, cbar
            gc.collect()

            print(f"Processed {year} {model_name}")
            time.sleep(1)

# 处理每个模型
for model_name in model_names:#一次可能跑不完，可以分几次跑，形如model_names[0:4]
    process_model(model_name)
print("All years processed")
'''
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import matplotlib as mpl
from mpl_toolkits.basemap import Basemap
from matplotlib.colors import LinearSegmentedColormap
import os
import gc

# 读取真实数据
df = pd.read_csv(f'./global_data/real_data1993.csv')
df[df > 0] = 1

# 定义模型名称
n1 = ['Global_', 'Geoarea_', 'PCA_', 'GRG_']
n2 = ['LASSO', 'LR', 'RF', 'GBRT']

# 初始化网格数据
lon = np.linspace(0, 360, 720)
lat = np.linspace(-90, 90, 360)
lon1, lat1 = np.meshgrid(lon, lat)

# 定义颜色映射
levels_predicted = [0, 0.01, 0.05, 0.1, 0.5, 1, 2, 4, 8, 12, 16, 50]
colors = [
    "#F0FFFF",  # 1. Very light cyan
    "#FFFFFF",
    "#D0F0F6",  # 2. Light cyan
    "#AEEBEB",  # 3. Pale turquoise
    "#8AD3D8",  # 4. Light sea green
    "#6EB8B3",  # 5. Medium turquoise
    "#4A9C8D",  # 6. Dark turquoise
    "#2C7F6D",  # 7. Deep teal
    "#006B3F",  # 8. Dark green
    "#004A30",  # 9. Darker green
    "#00352A",  # 10. Very dark green
    "#00241F",  # 11. Almost black green
    "#001D18"   # 12. Very dark green with blue undertones
]
colors1 = [
    "#F0FFFF",  # 1. Very light cyan
    "#FFFFFF",
    "#D0F0F6",  # 2. Light cyan
    "#AEEBEB",  # 3. Pale turquoise
    "#8AD3D8",  # 4. Light sea green
    "#6EB8B3",  # 5. Medium turquoise
    "#4A9C8D",  # 6. Dark turquoise
    "#2C7F6D",  # 7. Deep teal
    "#006B3F",  # 8. Dark green
    "#004A30",  # 9. Darker green
    "#00352A",  # 10. Very dark green
    "#00241F",  # 11. Almost black green
    "#001D18",  # 12. Very dark green with blue undertones
    "#FFFFFF"
]
colors2 = [
    "#F0FFFF",
    "#FFFFFF",
    "#FFF0B3",  # 1. Very light yellow
    "#FFEB9C",  # 2. Pale yellow
    "#FFD785",  # 3. Gold
    "#FFC16C",  # 4. Dark gold
    "#FFAA4F",  # 5. Orange yellow
    "#FF9031",  # 6. Light orange
    "#FF7513",  # 7. Orange
    "#FF5700",  # 8. Dark orange
    "#E63900",  # 9. Very dark orange
    "#CC1A00",  # 10. Red orange
    "#B30000",  # 11. Dark red
]

cmaps = LinearSegmentedColormap.from_list('mylist', colors, N=13)
cmaps1 = LinearSegmentedColormap.from_list('mylist1', colors1, N=14)
cmaps2 = LinearSegmentedColormap.from_list('mylist2', colors2, N=13)
norm = mpl.colors.BoundaryNorm(levels_predicted, 13, extend='both')
norm1 = norm

# 绘制地图
for s1 in range(4):
    for s2 in range(4):
        name = n1[s1] + n2[s2]
        output_dir = f'./全年份绘图/{name}'
        os.makedirs(output_dir, exist_ok=True)

        for i in range(251):
            year = 1850 + i
            both1 = pd.read_csv(f'./全部年份预测/{name}{year}.csv') * df
            both1[both1 < 0] = 0

            fig, axes = plt.subplots(figsize=(10, 4))
            fontsize1 = 8
            xchi = np.round(np.linspace(-90, 90, 5), 1)

            # 创建地图对象
            map = Basemap(
                llcrnrlon=0, llcrnrlat=-90, urcrnrlon=360, urcrnrlat=90,
                lon_0=0, resolution='c', projection='cyl', ax=axes
            )
            xi, yi = map(lon1, lat1)
            xmin = np.min(xi)
            xmax = np.max(xi)
            y1 = np.min(yi)
            y2 = np.max(yi)

            # 绘制地图背景
            map.drawmapboundary(fill_color='white')
            map.shadedrelief(scale=0.1)
            map.drawparallels(circles=xchi, labels=[1, 0, 0, 0], color='gray', fontsize=fontsize1)
            map.drawmeridians(meridians=np.linspace(-180, 180, 6), labels=[0, 0, 0, 1], color='gray', fontsize=fontsize1)

            # 绘制数据
            both1 = np.ma.masked_values(both1, np.nan)
            mesh_masked = axes.pcolormesh(xi, yi, both1, cmap=cmaps1, norm=norm1)

            # 添加图例
            l = 0.2  # 左边界
            b = -0.1  # 底边界
            w = 0.6  # 宽度
            h = 0.05  # 高度
            rect = [l, b, w, h]
            cbar_ax = fig.add_axes(rect)
            cb = fig.colorbar(plt.cm.ScalarMappable(cmap=cmaps, norm=norm), cmap=cmaps, norm=norm, boundaries=levels_predicted, extend='neither', cax=cbar_ax, orientation='horizontal')
            cb.set_ticks(levels_predicted)
            cb.ax.tick_params(labelsize=fontsize1)
            cb.set_label("AGC ($kg$/$m^{-2}$)", labelpad=20, y=0.5, rotation=0, fontsize=12, weight='normal')

            # 保存图像
            fig.tight_layout()
            plt.savefig(os.path.join(output_dir, f'{year}.png'), dpi=1000, bbox_inches='tight')
            plt.close(fig)

            # 显式释放内存
            del both1, fig, axes, map, xi, yi, mesh_masked, cbar_ax, cb
            gc.collect()

            print(f"Processed {year} {name}")

print("All years processed")

'''



