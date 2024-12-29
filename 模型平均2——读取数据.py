import numpy as np
import pandas as pd
import xarray as xr

# 模型名称列表
models = [
    "ACCESS-ESM1-5", "BCC-CSM2-MR", "CESM2", "CESM2-WACCM", "CMCC-CM2-SR5",
    "CMCC-ESM2", "CNRM-ESM2-1", "CanESM5", "CanESM5-CanOE", "E3SM-1-1",
    "EC-Earth3-CC", "EC-Earth3-Veg", "EC-Earth3-Veg-LR", "INM-CM4-8",
    "INM-CM5-0", "KIOST-ESM", "IPSL-CM6A-LR", "MIROC-ES2L",
    "MPI-ESM1-2-LR", "NorESM2-LM", "NorESM2-MM", "UKESM1-0-LL"
]

# 读取 R 值数据
r = pd.read_csv('./acgR.csv')
r.columns = range(r.shape[1])

# 遍历所有模型名称
for model in models:
    # 读取未来预测数据（2015年及以后）
    future_dataset = xr.open_dataset(f'./cVeg in 720/not_cut/cVeg_Lmon_{model}_ssp585_201501-210012_year_in_r720x360.nc')
    future_data = future_dataset.variables['cVeg'][:].data
    
    # 处理未来预测数据
    for year, data_slice in enumerate(future_data):
        transformed_data = pd.DataFrame(data_slice) / (1 + r)
        filled_data = transformed_data.fillna(0)
        filled_data.to_csv(f'./global_data/{model}_data{2015 + year}.csv', index=False)

    # 读取历史数据（2014年之前）
    historical_dataset = xr.open_dataset(f'./not_cut/cVeg_Lmon_{model}_historical_185001-201412_year_in_r720x360.nc')
    historical_data = historical_dataset.variables['cVeg'][:].data
    
    # 处理历史数据
    for year, data_slice in enumerate(historical_data):
        transformed_data = pd.DataFrame(data_slice) / (1 + r)
        filled_data = transformed_data.fillna(0)
        filled_data.to_csv(f'./global_data/{model}_data{1850 + year}.csv', index=False)

# 读取1993-2014年的真实数据
real_dataset = xr.open_dataset('./0.5_Global_annual_mean_ABC_lc2001_1993_2012_20150331.nc')
real_cVeg = real_dataset.variables['Aboveground Biomass Carbon'][:].data

# 处理真实数据
for year, data_slice in enumerate(real_cVeg):
    transformed_data = pd.DataFrame(data_slice.T) / 10
    filled_data = transformed_data.fillna(0)
    filled_data.to_csv(f'./global_data/real_data{1993 + year}.csv', index=False)

# 对1993年-2012年的数据进行简单平均并储存
for year in range(20):
    avg_data = pd.DataFrame(np.zeros((360, 720)))
    avg_data.columns = range(720)
    
    for model in models:
        model_data = pd.read_csv(f'./global_data/{model}_data{1993 + year}.csv')
        model_data.columns = range(720)
        avg_data += model_data
    
    avg_data /= len(models)
    avg_data.to_csv(f'./预测数据/Average_{1993 + year}.csv', index=False)


'''
#该代码用于读取nc格式的全球预cVeg的预测值和真实值，并用csv格式储存

import numpy as np
import xarray as xr
from sklearn.feature_selection import f_regression
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter
import cmaps
import xarray as xr
import pandas as pd

#22个模型名称
mod_name = [
"ACCESS-ESM1-5",
"BCC-CSM2-MR",
"CESM2",
"CESM2-WACCM",
"CMCC-CM2-SR5",
"CMCC-ESM2",
"CNRM-ESM2-1",
"CanESM5",
"CanESM5-CanOE",
"E3SM-1-1",
"EC-Earth3-CC",
"EC-Earth3-Veg",
"EC-Earth3-Veg-LR",
"INM-CM4-8",
"INM-CM5-0",
"KIOST-ESM",
"IPSL-CM6A-LR",
"MIROC-ES2L",
"MPI-ESM1-2-LR",
"NorESM2-LM",
"NorESM2-MM",
"UKESM1-0-LL"]
#读取R值用于acg和cveg的转化
r=pd.read_csv(f'./acgR.csv')
r.columns = range(720)
#遍历所有模型
for i in mod_name:
    #2014年以后的模型
    dataset = xr.open_dataset('./cveg in 720/not_cut/cVeg_Lmon_{}_ssp585_201501-210012_year_in_r720x360.nc'.format(i))
    data = dataset.variables['cVeg'][:].data
    #data中包含所有年份的预测值，逐年处理
    for year in range(len(data)):
        y=pd.DataFrame(data[year]) / (1 + r)#用公式转化
        desired_shape = (360, 720)
        y_filled = pd.DataFrame(np.zeros(desired_shape))
        y_filled.iloc[:y.shape[0], :y.shape[1]] = y.fillna(0)
        pd.DataFrame(y_filled).to_csv(f'./global_data/'+i+'_data'+str(2015+year)+'.csv',index=False)

    #2014年以前的模型
    dataset = xr.open_dataset('./not_cut/cVeg_Lmon_{}_historical_185001-201412_year_in_r720x360.nc'.format(i))
    data = dataset.variables['cVeg'][:].data
    for year in range(len(data)):
        y=pd.DataFrame(data[year]) / (1 + r)
        desired_shape = (360, 720)
        y_filled = pd.DataFrame(np.zeros(desired_shape))
        y_filled.iloc[:y.shape[0], :y.shape[1]] = y.fillna(0)
        pd.DataFrame(y_filled).to_csv(f'./global_data/'+i+'_data'+str(1850+year)+'.csv',index=False)

#读取1993-2014年的真实数据
Global = xr.open_dataset('./0.5_Global_annual_mean_ABC_lc2001_1993_2012_20150331.nc')
cVeg = Global.variables['Aboveground Biomass Carbon'][:].data
for year in range(len(cVeg)):
    y=pd.DataFrame(cVeg[year]).T / 10
    desired_shape = (360, 720)
    y_filled = pd.DataFrame(np.zeros(desired_shape))
    y_filled.iloc[:y.shape[0], :y.shape[1]] = y.fillna(0)
    y_filled.to_csv(f'./global_data/real_data'+str(1993+year)+'.csv',index=False)

            

#对1993年-2012年的数据各个模型数据作简单平均并储存
for year in range(20):
    df=pd.DataFrame(np.zeros(desired_shape))
    df.columns = range(720)
    for i in mod_name:
        df2=pd.read_csv(f'./global_data/'+i+'_data'+str(1993+year)+'.csv')
        df2.columns = range(720)
        df=df+df2
    df=df/len(mod_name)
    df.to_csv(f'./预测数据/Average_'+str(1993+year)+'.csv',index=False)

'''





