import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from mpl_toolkits.basemap import Basemap
from matplotlib.colors import LinearSegmentedColormap
import matplotlib as mpl
from matplotlib.lines import Line2D  # 导入 Line2D

# 定义经纬度范围
lat = np.arange(-89.75, 90, 0.5)
lon = np.arange(0, 360, 0.5)

# 读取真实数据
df = pd.read_csv('./global_data/real_data1993.csv')
df[df > 0] = 1

# 读取地图分割数据
area = [pd.read_csv(f'./地图分割/{i}.csv') for i in range(15)]
m = sum(area[i] * (i + 1) for i in range(15))

# 初始化绘图数据
temp1 = np.full((360, 720), np.nan)
both1 = m.copy()
both1[both1 < 0] = 0

# 设置颜色映射
colors = [
    '#FFFFFF',
    '#B0E57C', '#D5E2A4', '#F0E68C', '#FFD700', '#FFA07A', '#FF6347', '#F0FFFF',
    '#FF8C00', '#4682B4', '#4169E1', '#87CEFA', '#00CED1', '#00BFFF', '#20B2AA', '#00FA9A'
]
cmaps = LinearSegmentedColormap.from_list('mylist', colors, N=16)
norm = mpl.colors.BoundaryNorm(range(1, 16), 17, extend='both')

# 创建图形和坐标轴
fig, ax = plt.subplots(figsize=(15, 12))
fontsize1 = 8
xchi = np.round(np.linspace(-90, 90, 5), 1)

# 创建地图对象
map = Basemap(llcrnrlon=0, llcrnrlat=-90, urcrnrlon=360, urcrnrlat=90, lon_0=0, resolution='c', projection='cyl', ax=ax)
lon1, lat1 = np.meshgrid(lon, lat)
xi, yi = map(lon1, lat1)
xmin, xmax = np.min(xi), np.max(xi)
y1, y2 = np.min(yi), np.max(yi)

# 绘制地图边界和地形
map.drawmapboundary(fill_color='white')
map.shadedrelief(scale=0.1)
map.drawparallels(circles=xchi, labels=[1, 0, 0, 0], color='gray', fontsize=fontsize1)
map.drawmeridians(meridians=np.linspace(-180, 180, 6), labels=[0, 0, 0, 1], color='gray', fontsize=fontsize1)

# 绘制背景色
map.pcolormesh(xi, yi, temp1, cmap=cmaps, norm=norm)

# 绘制区域划分
both1 = np.ma.masked_values(both1, np.nan)
mesh_masked = ax.pcolormesh(xi, yi, both1, cmap=cmaps, norm=norm)

# 添加图例
color_to_region = {
    '#B0E57C': 'Boreal Forests/Taiga',
    '#D5E2A4': 'Deserts & Xeric Shrublands',
    '#F0E68C': 'Flooded Grasslands & Savannas',
    '#FFD700': 'Mangroves',
    '#FFA07A': 'Mediterranean Forests, Woodlands & Scrub',
    '#FF6347': 'Montane Grasslands & Shrublands',
    '#F0FFFF': 'N/A',
    '#FF8C00': 'Temperate Broadleaf & Mixed Forests',
    '#4682B4': 'Temperate Conifer Forests',
    '#4169E1': 'Temperate Grasslands, Savannas & Shrublands',
    '#87CEFA': 'Tropical & Subtropical Coniferous Forests',
    '#00CED1': 'Tropical & Subtropical Dry Broadleaf Forests',
    '#00BFFF': 'Tropical & Subtropical Grasslands, Savannas & Shrublands',
    '#20B2AA': 'Tropical & Subtropical Moist Broadleaf Forests',
    '#00FA9A': 'Tundra'
}

legend_elements = [Line2D([0], [0], marker='o', color='w', label=region, markerfacecolor=color, markersize=15) for color, region in color_to_region.items()]
ax.legend(handles=legend_elements, loc='lower center', bbox_to_anchor=(0.5, -0.3), ncol=3)

# 添加注释
ax.annotate('(a)', xy=(0.05, 0.93), xycoords='axes fraction', fontsize=12, color='black', ha='center', va='center')

# 保存和显示图像
plt.savefig('./地图分割/完整.png', dpi=1000, bbox_inches='tight')
plt.show()

print("end")
'''
import matplotlib.pyplot as plt
import pandas as pd
import xarray as xr
import numpy as np
from mpl_toolkits.basemap import Basemap
from matplotlib.colors import ListedColormap,LinearSegmentedColormap
import matplotlib as mpl
from matplotlib.patches import Rectangle
from matplotlib.lines import Line2D
# 将列表转换为 numpy 数组
lat = np.arange(-89.75, 89.75 + 0.5, 0.5)
lon = np.arange(0, 359.5+0.5, 0.5)
df=pd.read_csv(f'./global_data/real_data1993.csv')
df[df>0]=1





area=[i for i in range(15)]
for i in range(15):
    area[i]=pd.read_csv(f'./地图分割/'+str(i)+'.csv')
m=area[0]
for i in range(1,15):
    m=m+area[i]*(i+1)
temp1 = np.full((360,720), np.nan)
both1 = np.full((360,720), 999.0)
both1=m
both1[both1<0]=0
labelneg = 'average'

levels_predicted=[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15]
label = labelneg

fig, axes = plt.subplots( figsize=(15,12))
fontsize1 = 8
xchi = np.round(np.linspace(-90, 90, 5),1)
colorbarlabel = levels_predicted

colors =[
    '#FFFFFF',
    '#B0E57C',
    '#D5E2A4',
    '#F0E68C',
    '#FFD700',
    '#FFA07A',
    '#FF6347',
    '#F0FFFF',
    '#FF8C00',
    '#4682B4',
    '#4169E1',
    '#87CEFA',
    '#00CED1',
    '#00BFFF',
    '#20B2AA',
    '#00FA9A'
]

colors1=colors

cmaps = LinearSegmentedColormap.from_list('mylist',colors,N=16)
cmaps1 = LinearSegmentedColormap.from_list('mylist1',colors1, N=16)

norm = mpl.colors.BoundaryNorm(colorbarlabel, 17, extend='both')

norm1=norm
map = Basemap(llcrnrlon=0, llcrnrlat=-90, urcrnrlon=360, urcrnrlat=90
        , lon_0=0, resolution='c', projection='cyl', ax=axes)
lon1, lat1 = np.meshgrid(lon, lat)
xi, yi = map(lon1, lat1)
xmin = np.min(xi)
xmax = np.max(xi)
y1 = np.min(yi)
y2 = np.max(yi)

map.drawmapboundary(fill_color='white')

map.shadedrelief(scale=0.1)
map.drawparallels(circles=xchi, labels=[1, 0, 0, 0], color='gray',fontsize=fontsize1)
map.drawmeridians(meridians=np.linspace(-180, 180, 6), labels=[0, 0, 0, 1], color='gray',fontsize=fontsize1)
map.pcolormesh(xi, yi, temp1, cmap = cmaps, norm = norm)
hatch = axes.fill_between([xmin,xmax],y1,y2,hatch='/////////////',color="none",edgecolor='black', label='Geoarea')
both1 = np.ma.masked_values(both1,np.nan)
mesh_masked = axes.pcolormesh(xi, yi, both1, cmap = cmaps1, norm = norm1)
axes.legend(loc="lower left",prop={'family':'SimHei','size':8})

axes.annotate('(a)', xy=(0.05, 0.93), xycoords='axes fraction', fontsize=12, color='black', ha='center', va='center')
color_to_region = {
    '#B0E57C': 'Boreal Forests/Taiga',
    '#D5E2A4': 'Deserts & Xeric Shrublands',
    '#F0E68C': 'Flooded Grasslands & Savannas',
    '#FFD700': 'Mangroves',
    '#FFA07A': 'Mediterranean Forests, Woodlands & Scrub',
    '#FF6347': 'Montane Grasslands & Shrublands',
    "#F0FFFF": 'N/A',
    '#FF8C00': 'Temperate Broadleaf & Mixed Forests',
    '#4682B4': 'Temperate Conifer Forests',
    '#4169E1': 'Temperate Grasslands, Savannas & Shrublands',
    '#87CEFA': 'Tropical & Subtropical Coniferous Forests',
    '#00CED1': 'Tropical & Subtropical Dry Broadleaf Forests',
    '#00BFFF': 'Tropical & Subtropical Grasslands, Savannas & Shrublands',
    '#20B2AA': 'Tropical & Subtropical Moist Broadleaf Forests',
    '#00FA9A': 'Tundra'
}
legend_elements = []
for color, region in color_to_region.items():

    legend_elements.append(Line2D([0], [0], marker='o', color='w', label=region,
                                  markerfacecolor=color, markersize=15))

# 添加图例
axes.legend(handles=legend_elements, loc='lower center', bbox_to_anchor=(0.5, -0.3), ncol=3)
plt.savefig('./地图分割/完整.png', dpi=1000, bbox_inches='tight')
plt.show()
print("enbd")
'''
