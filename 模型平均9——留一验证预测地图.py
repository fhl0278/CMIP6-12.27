
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from mpl_toolkits.basemap import Basemap
from matplotlib.colors import LinearSegmentedColormap
import matplotlib as mpl
from matplotlib.lines import Line2D

# 定义经纬度范围
lat = np.arange(-89.75, 90, 0.5)
lon = np.arange(0, 360, 0.5)

# 读取真实数据
df = pd.read_csv('./global_data/real_data1993.csv')
df[df > 0] = 1

# 定义文件名前缀
n1 = ['Global', 'Geoarea', 'PCA_', 'GRG_']
n2 = ['LASSO', 'LR', 'RF', 'GBRT']

# 定义颜色映射
colors = [
    "#F0FFFF", "#FFFFFF", "#D0F0F6", "#AEEBEB", "#8AD3D8", "#6EB8B3", "#4A9C8D", "#2C7F6D", "#006B3F", "#004A30", "#00352A", "#00241F", "#001D18"
]
colors1 = colors + ["#FFFFFF"]
colors2 = [
    "#F0FFFF", "#FFFFFF", "#FFF0B3", "#FFEB9C", "#FFD785", "#FFC16C", "#FFAA4F", "#FF9031", "#FF7513", "#FF5700", "#E63900", "#CC1A00", "#B30000"
]

# 创建颜色映射
cmaps = LinearSegmentedColormap.from_list('mylist', colors, N=13)
cmaps1 = LinearSegmentedColormap.from_list('mylist1', colors1, N=14)
cmaps2 = LinearSegmentedColormap.from_list('mylist2', colors2, N=13)
norm = mpl.colors.BoundaryNorm([0, 0.01, 0.05, 0.1, 0.5, 1, 2, 4, 8, 12, 16, 50], 13, extend='both')

# 循环生成图表
for s1 in range(4):
    for s2 in range(4):
        name = f'{n1[s1]}{n2[s2]}'
        for i in range(20):
            # 读取预测数据和真实数据
            pred_data = pd.read_csv(f'./预测数据/{name}{1993+i}.csv') * df
            pred_data[pred_data < 0] = 0
            real_data = pd.read_csv(f'./预测数据/real_data{1993+i}.csv')
            error_data = (pred_data - real_data) ** 2

            # 创建图形和子图
            fig, axes = plt.subplots(1, 3, figsize=(10, 4))
            fontsize1 = 8
            xchi = np.round(np.linspace(-90, 90, 5), 1)

            # 绘制预测数据
            map_pred = Basemap(llcrnrlon=0, llcrnrlat=-90, urcrnrlon=360, urcrnrlat=90, lon_0=0, resolution='c', projection='cyl', ax=axes[0])
            lon1, lat1 = np.meshgrid(lon, lat)
            xi, yi = map_pred(lon1, lat1)
            map_pred.drawmapboundary(fill_color='white')
            map_pred.shadedrelief(scale=0.1)
            map_pred.drawparallels(circles=xchi, labels=[1, 0, 0, 0], color='gray', fontsize=fontsize1)
            map_pred.drawmeridians(meridians=np.linspace(-180, 180, 6), labels=[0, 0, 0, 1], color='gray', fontsize=fontsize1)
            map_pred.pcolormesh(xi, yi, pred_data, cmap=cmaps1, norm=norm)
            axes[0].legend(loc="lower left", prop={'family': 'SimHei', 'size': 8})
            axes[0].annotate('(a)', xy=(0.05, 0.93), xycoords='axes fraction', fontsize=12, color='black', ha='center', va='center')

            # 绘制真实数据
            map_real = Basemap(llcrnrlon=0, llcrnrlat=-90, urcrnrlon=360, urcrnrlat=90, lon_0=0, resolution='c', projection='cyl', ax=axes[1])
            map_real.drawmapboundary(fill_color='white')
            map_real.shadedrelief(scale=0.1)
            map_real.drawparallels(circles=xchi, labels=[0, 0, 0, 0], color='gray', fontsize=fontsize1)
            map_real.drawmeridians(meridians=np.linspace(-180, 180, 6), labels=[0, 0, 0, 1], color='gray', fontsize=fontsize1)
            map_real.pcolormesh(xi, yi, real_data, cmap=cmaps1, norm=norm)
            axes[1].legend(loc="lower left", prop={'family': 'SimHei', 'size': 8})
            axes[1].annotate('(b)', xy=(0.05, 0.93), xycoords='axes fraction', fontsize=12, color='black', ha='center', va='center')

            # 绘制误差数据
            map_error = Basemap(llcrnrlon=0, llcrnrlat=-90, urcrnrlon=360, urcrnrlat=90, lon_0=0, resolution='c', projection='cyl', ax=axes[2])
            map_error.drawmapboundary(fill_color='white')
            map_error.shadedrelief(scale=0.1)
            map_error.drawparallels(circles=xchi, labels=[0, 0, 0, 0], color='gray', fontsize=fontsize1)
            map_error.drawmeridians(meridians=np.linspace(-180, 180, 6), labels=[0, 0, 0, 1], color='gray', fontsize=fontsize1)
            map_error.pcolormesh(xi, yi, error_data, cmap=cmaps2, norm=norm)
            axes[2].legend(loc="lower left", prop={'family': 'SimHei', 'size': 8})
            axes[2].annotate('(c)', xy=(0.05, 0.93), xycoords='axes fraction', fontsize=12, color='black', ha='center', va='center')

            # 添加颜色条
            l, b, w, h = 0.05, 0.2, 0.43, 0.018
            rect = [l, b, w, h]
            cbar_ax = fig.add_axes(rect)
            cb = fig.colorbar(plt.cm.ScalarMappable(cmap=cmaps, norm=norm), cmap=cmaps, norm=norm, boundaries=[0, 0.01, 0.05, 0.1, 0.5, 1, 2, 4, 8, 12, 16, 50], extend='neither', cax=cbar_ax, orientation='horizontal')
            cb.set_ticks([0, 0.01, 0.05, 0.1, 0.5, 1, 2, 4, 8, 12, 16, 50])
            cb.ax.tick_params(labelsize=fontsize1)
            cb.update_ticks()
            plt.title("AGC ($kg$ $m^{-2}$)", fontdict={'weight': 'normal', 'size': 12}, y=-7)

            l, b, w, h = 0.55, 0.2, 0.43, 0.018
            rect = [l, b, w, h]
            cbar_ax = fig.add_axes(rect)
            cb = fig.colorbar(plt.cm.ScalarMappable(cmap=cmaps2, norm=norm), cmap=cmaps2, norm=norm, boundaries=[0, 0.01, 0.05, 0.1, 0.5, 1, 2, 4, 8, 12, 16, 50], extend='neither', cax=cbar_ax, orientation='horizontal')
            cb.set_ticks([0, 0.01, 0.05, 0.1, 0.5, 1, 2, 4, 8, 12, 16, 50])
            cb.ax.tick_params(labelsize=fontsize1)
            cb.update_ticks()
            plt.title("Squared Error", fontdict={'weight': 'normal', 'size': 12}, y=-7)

            fig.tight_layout()
            plt.savefig(f'./预测图/{name}/{1993+i}.png', dpi=1000, bbox_inches='tight')
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

lat = np.arange(-89.75, 89.75 + 0.5, 0.5)
lon = np.arange(0, 359.5+0.5, 0.5)
df=pd.read_csv(f'./global_data/real_data1993.csv')
df[df>0]=1
n1=['Global','Geoarea','PCA_','GRG_']
n2=['LASSO','LR','RF','GBRT']
for s1 in range(4):
    for s2 in range(4):
        name=n1[s1]+n2[s2]
        temp1 = np.full((360,720), np.nan)
        temp2 = np.full((360,720), np.nan)
        temp3 = np.full((360,720), np.nan)
        both1 = np.full((360,720), 999.0)
        both2 = np.full((360,720), 999.0)
        both3 = np.full((360,720), 999.0)
        for i in range(20):
            both1 = pd.read_csv('./预测数据/'+name+''+str(1993+i)+'.csv')*df
            both1[both1<0]=0
            both2 = pd.read_csv('./预测数据/real_data'+str(1993+i)+'.csv')
            both3=(both1-both2)**2
            labelneg = 'average'
            levels_predicted=[0,0.01,0.05,0.1,0.5,1,2,4,8,12,16,50]
            label = labelneg
            
            
            
            fig, axes = plt.subplots(1, 3, figsize=(10,4))
            fontsize1 = 8
            xchi = np.round(np.linspace(-90, 90, 5),1)
            colorbarlabel = levels_predicted

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
            
            colors1= [
            
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
            "#001D18",   # 12. Very dark green with blue undertones
            "#FFFFFF"
        ]
           
            colors2 = [
                "#F0FFFF",
            "#FFFFFF",
              # 1. Very light yellow
            "#FFF0B3",  # 2. Light yellow
            "#FFEB9C",  # 3. Pale yellow
            "#FFD785",  # 4. Gold
            "#FFC16C",  # 5. Dark gold
            "#FFAA4F",  # 6. Orange yellow
            "#FF9031",  # 7. Light orange
            "#FF7513",  # 8. Orange
            "#FF5700",  # 9. Dark orange
            "#E63900",  # 10. Very dark orange
            "#CC1A00",  # 11. Red orange
            "#B30000",  # 12. Dark red
        ]
            cmaps = LinearSegmentedColormap.from_list('mylist',colors, N=13)
            cmaps1 = LinearSegmentedColormap.from_list('mylist1',colors1, N=14)
            cmaps2 = LinearSegmentedColormap.from_list('mylist1',colors2, N=13)
            norm = mpl.colors.BoundaryNorm(colorbarlabel, 13, extend='both')
 
            norm1=norm
            map = Basemap(llcrnrlon=0, llcrnrlat=-90, urcrnrlon=360, urcrnrlat=90
                    , lon_0=0, resolution='c', projection='cyl', ax=axes[0])
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
            hatch = axes[0].fill_between([xmin,xmax],y1,y2,hatch='/////////////',color="none",edgecolor='black', label='pred')
            both1 = np.ma.masked_values(both1,np.nan)
            mesh_masked = axes[0].pcolormesh(xi, yi, both1, cmap = cmaps1, norm = norm1)
            axes[0].legend(loc="lower left",prop={'family':'SimHei','size':8})
            
            map = Basemap(llcrnrlon=0, llcrnrlat=-90, urcrnrlon=360, urcrnrlat=90
                    , lon_0=0, resolution='c', projection='cyl',ax=axes[1])
            lon1, lat1 = np.meshgrid(lon, lat)
            xi, yi = map(lon1, lat1)
            map.drawmapboundary(fill_color='white')
            map.shadedrelief(scale=0.1)
            map.drawparallels(circles=xchi, labels=[0, 0, 0, 0], color='gray',fontsize=fontsize1)
            map.drawmeridians(meridians=np.linspace(-180, 180, 6), labels=[0, 0, 0, 1], color='gray',fontsize=fontsize1)
            map.pcolormesh(xi, yi, temp2, cmap = cmaps, norm = norm)
            hatch = axes[1].fill_between([xmin,xmax],y1,y2,hatch='/////////////',color="none",edgecolor='black', label='real')
            mesh_masked = axes[1].pcolormesh(xi, yi, both2, cmap = cmaps1, norm = norm1)
            axes[1].legend(loc="lower left",prop={'family':'SimHei','size':8})
            
            map = Basemap(llcrnrlon=0, llcrnrlat=-90, urcrnrlon=360, urcrnrlat=90
                    , lon_0=0, resolution='c', projection='cyl', ax=axes[2])
            lon1, lat1 = np.meshgrid(lon, lat)
            xi, yi = map(lon1, lat1)
            map.drawmapboundary(fill_color='white')
            map.shadedrelief(scale=0.1)
            map.drawparallels(circles=xchi, labels=[0, 0, 0, 0], color='gray',fontsize=fontsize1)
            map.drawmeridians(meridians=np.linspace(-180, 180, 6), labels=[0, 0, 0, 1], color='gray',fontsize=fontsize1)
            map.pcolormesh(xi, yi, temp3, hatch='/', cmap = cmaps, norm = norm)
            hatch = axes[2].fill_between([xmin,xmax],y1,y2,hatch='/////////////',color="none",edgecolor='black', label='Squared Error')
            mesh_masked = axes[2].pcolormesh(xi, yi, both3, cmap = cmaps2, norm = norm1)
            mpl.rcParams['hatch.linewidth'] = 0.5
            axes[2].legend(loc="lower left",prop={'family':'SimHei','size':8})
            axes[0].annotate('(a)', xy=(0.05, 0.93), xycoords='axes fraction', fontsize=12, color='black', ha='center', va='center')
            axes[1].annotate('(b)', xy=(0.05, 0.93), xycoords='axes fraction', fontsize=12, color='black', ha='center', va='center')
            axes[2].annotate('(c)', xy=(0.05, 0.93), xycoords='axes fraction', fontsize=12, color='black', ha='center', va='center')
            
            l = 0.05
            b = 0.2
            w = 0.43
            h = 0.018
            #对应 l,b,w,h；设置colorbar位置；
            rect = [l,b,w,h]
            cbar_ax = fig.add_axes(rect)
            cb = fig.colorbar(plt.cm.ScalarMappable(cmap=cmaps,norm=norm), cmap=cmaps, norm=norm, boundaries=levels_predicted, extend='neither', cax = cbar_ax, orientation='horizontal')
            cb.set_ticks(colorbarlabel)
            cb.ax.tick_params(labelsize=fontsize1)
            cb.update_ticks()
            plt.title("AGC ($kg$ $m^{-2}$)",fontdict={'weight':'normal','size': 12},y=-7)
            
            l = 0.55
            b = 0.2
            w = 0.43
            h = 0.018
            #对应 l,b,w,h；设置colorbar位置；
            rect = [l,b,w,h]
            cbar_ax = fig.add_axes(rect)
            cb = fig.colorbar(plt.cm.ScalarMappable(cmap=cmaps2,norm=norm1), cmap=cmaps2, norm=norm1, boundaries=levels_predicted, extend='neither', cax = cbar_ax, orientation='horizontal')
            cb.set_ticks(colorbarlabel)
            cb.ax.tick_params(labelsize=fontsize1)
            cb.update_ticks()
            fig.tight_layout()
            plt.title("Squared Error",fontdict={'weight':'normal','size': 12},y=-7)
            plt.savefig('./预测图/'+name+'/'+str(1993+i)+'.png', dpi=1000, bbox_inches='tight')
            plt.show()
            print("enbd")
'''
