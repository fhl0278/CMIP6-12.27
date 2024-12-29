
import numpy as np
import pandas as pd
import xarray as xr
import geopandas as gpd
from shapely.ops import unary_union

# 读取初始文件，其中包含划分方案
geoarea = gpd.read_file('./Ecoregions2017.shp', crs="epsg:4326")

# 按照生态区名进行分组
grouped = geoarea.groupby('BIOME_NAME')

# 合并每个分组的几何对象
new_geometries = [unary_union(group['geometry']) for _, group in grouped]
new_attributes = [{'BIOME_NAME': name} for name, _ in grouped]

# 创建新的基于生态区的 GeoDataFrame
gdf_biome = gpd.GeoDataFrame(new_attributes, geometry=new_geometries, crs="epsg:4326")

# 定义经纬度范围
lat = np.arange(-89.75, 90, 0.5)
lon = np.arange(-180, 180, 0.5)

# 创建两个 xr.Dataset，一个全为0，一个全为1
data1 = np.ones((len(lat), len(lon)))
data2 = np.zeros((len(lat), len(lon)))

dataset1 = xr.Dataset(
    {'values': (['lat', 'lon'], data1)},
    coords={'lat': lat, 'lon': lon}
)

dataset2 = xr.Dataset(
    {'values': (['lat', 'lon'], data2)},
    coords={'lat': lat, 'lon': lon}
)

# 逐个储存15个生态区数据，该生态区网格数值为1，否则为0
for i in range(len(gdf_biome)):
    # 提取单个生态区的几何信息
    gdf = gpd.GeoDataFrame(geometry=gpd.GeoSeries([gdf_biome.geometry.iloc[i]]))
    
    # 复制dataset1，并设置地理坐标系
    datatemp = dataset1.copy()
    datatemp.rio.set_spatial_dims(x_dim="lon", y_dim="lat", inplace=True)
    datatemp.rio.write_crs("epsg:4326", inplace=True)
    
    # 根据生态区的边界裁剪dataset1
    clipped = datatemp.rio.clip(gdf.geometry.apply(lambda x: mapping(x)), gdf.crs, drop=True)
    
    # 使用dataset2并替换裁剪后的区域
    data = dataset2.copy()
    data['values'].loc[dict(lat=clipped.lat, lon=clipped.lon)] = 1
    
    # 转换为 DataFrame 并进行调整
    df = pd.DataFrame(data['values']).fillna(0)
    first_half = df.iloc[:, :360]  # 前 360 列
    second_half = df.iloc[:, 360:]  # 后 360 列
    df = pd.concat([second_half, first_half], axis=1)
    df.index = range(360)
    df.columns = range(720)
    
    # 保存为 CSV 文件
    df.to_csv(f'./地图分割/{i}.csv', index=False)
'''
#该代码用于划分生态区，并以csv格式储存

import numpy as np
import pandas as pd
import xarray as xr
import rioxarray
import geopandas
from shapely. geometry import mapping
import re
import copy

#读取初始文件，其中包含划分方案
geoarea = geopandas.read_file('./Ecoregions2017.shp', crs="epsg:4326")

grouped = geoarea.groupby('BIOME_NAME')#使用生态区划分
from shapely.ops import unary_union

# 创建一个新的列表来保存合并后的几何对象和属性
new_geometries = []
new_attributes = []

for name, group in grouped:
    # 合并每个分组的几何对象
    merged_geometry = unary_union(group['geometry'])
    # 保存合并后的几何对象
    new_geometries.append(merged_geometry)
    # 保存该分组的属性（例如，ECO_BIOME_）
    new_attributes.append({'BIOME_NAME': name})

# 创建新的基于生态区的 GeoDataFrame
gdf_biome = geopandas.GeoDataFrame(new_attributes, geometry=new_geometries, crs="epsg:4326")


#创建两个xr数据，360*720，dataset1数值全为0，dataset2数值全为1，用于切割生态区

lat = np.arange(-89.75, 89.75 + 0.5, 0.5)
lon = np.arange(0-180, 359.5+0.5-180, 0.5)
# 创建数据数组，所有点的数值设为 0
data1 = np.ones((len(lat), len(lon)))
data2 = np.zeros((len(lat), len(lon)))
# 创建 Dataset 对象
dataset1 = xr.Dataset(
    {
        'values': (['lat', 'lon'], data1),
    },
    coords={
        'lat': lat,
        'lon': lon,
    }
)
dataset2 = xr.Dataset(
    {
        'values': (['lat', 'lon'], data2),
    },
    coords={
        'lat': lat,
        'lon': lon,
    }
)




#用df确定生态区和非生态区。生态区网格数值为1，非生态区数值为0。
#（这部分结果后面用不到，可以忽略）
data=dataset2.copy()
data['values'].loc[dict(lat=clipped.lat, lon=clipped.lon)] = clipped['values']

df=pd.DataFrame(data['values']).fillna(0)
first_half = df.iloc[:, :360]  # 前 360 列
second_half = df.iloc[:, 360:]  # 后 360 列
# 将后 360 列放在前面，前 360 列放在后面
df = pd.concat([second_half, first_half], axis=1)
new_index = range(360)
new_columns = range(720)
# 重新设置行标
df.index = new_index
# 重新设置列标
df.columns = new_columns
df.to_csv(f'./地图分割/-1.csv',index=False)



#逐个储存15个生态区数据，该生态区网格数值为1，否则为0

for i in range(15):
    gdf=geopandas.GeoDataFrame(geometry=geopandas.GeoSeries([gdf_biome.geometry[i]]))
    datatemp=copy.deepcopy(dataset1)
    datatemp.rio.set_spatial_dims(x_dim="lon", y_dim="lat", inplace=True)
    datatemp.rio.write_crs("epsg:4326", inplace=True)
    clipped = datatemp.rio.clip(gdf.geometry.apply(mapping), gdf.crs, drop=True)
    data=copy.deepcopy(dataset2)

    data['values'].loc[dict(lat=clipped.lat, lon=clipped.lon)] = clipped['values']

    df=pd.DataFrame(data['values']).fillna(0)
    first_half = df.iloc[:, :360]  # 前 360 列
    second_half = df.iloc[:, 360:]  # 后 360 列
    # 将后 360 列放在前面，前 360 列放在后面
    df = pd.concat([second_half, first_half], axis=1)
    new_index = range(360)
    new_columns = range(720)
    # 重新设置行标
    df.index = new_index
    # 重新设置列标
    df.columns = new_columns
    df.to_csv(f'./地图分割/'+str(i)+'.csv',index=False)

'''



