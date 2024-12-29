import numpy as np
import pandas as pd
from joblib import load

# 读取真实数据并处理
df = pd.read_csv('./global_data/real_data1993.csv')
df[df > 0] = 1
dif = np.array(df).reshape(1, -1)[0]

# 读取生态区数据并处理
area = [pd.read_csv(f'./地图分割/{i}.csv') * df for i in range(15)]
for i in range(14):
    for j in range(i + 1, 15):
        area[j] = area[j] - (area[i] * area[j])

# 加载PCA模型
pca = load('./模型保存/pca20.joblib')

# 加载数据
data_pred = np.load('./data_pred.npy')
#(251,259200,22)
# 预先加载所有模型
global_models = {}
geoarea_models = {}
pca_models = {}
grg_models = {}

# 全局模型
for model in ['Global_LASSO', 'Global_RF', 'Global_LR', 'Global_GBRT']:
    model_path = f'./模型保存/{model}.joblib'
    global_models[model] = load(model_path)

# 生态区模型
for model in ['Geoarea_LASSO', 'Geoarea_RF', 'Geoarea_LR', 'Geoarea_GBRT']:
    geoarea_models[model] = [load(f'./模型保存/{model}{area_num}.joblib') for area_num in range(15)]

# PCA模型
for model in ['PCA_LASSO', 'PCA_RF', 'PCA_LR', 'PCA_GBRT']:
    model_path = f'./模型保存/{model}.pkl'
    pca_models[model] = load(model_path)

# GRG模型
for model in ['GRG_GBRT', 'GRG_LR', 'GRG_RF', 'GRG_LASSO']:
    model_path = f'./模型保存/{model}.pkl'
    grg_models[model] = load(model_path)

# 主循环
for train_year in range(251):
    year = 1850 + train_year
    print('train_year=',year)
    # 全局模型
    for model in ['Global_LASSO', 'Global_RF', 'Global_LR', 'Global_GBRT']:
        output_path = f'./全部年份预测/{model}_{year}.csv'
        
        regressor = global_models[model]
        y = regressor.predict(data_pred[train_year])
        
        y = pd.DataFrame(y.reshape(360, 720))
        y.columns = range(720)
        y = y * df
        y.to_csv(output_path, index=False)
    
    # 生态区模型
    for model in ['Geoarea_LASSO', 'Geoarea_RF', 'Geoarea_LR', 'Geoarea_GBRT']:
        zero_df = pd.DataFrame(np.zeros((360, 720)))
        zero_df.columns = range(720)
        for area_num in range(15):
            regressor = geoarea_models[model][area_num]
            
            y = (regressor.predict(data_pred[train_year]))*np.array(area[area_num]).reshape(1, -1)[0]
            
            #259200
            #regressor.predict(data_pred[train_year])
            #*np.array(area[area_num])
            #.reshape(1, -1)[0]
            
            y = pd.DataFrame(y.reshape(360, 720))
            y.columns = range(720)
            zero_df += y
        output_path = f'./全部年份预测/{model}_{year}.csv'
        zero_df.to_csv(output_path, index=False)
    
    
    # PCA模型
    pca_train = data_pred[train_year][~(data_pred[train_year] == 0).all(axis=1)]
    pc_df = pca.transform(pca_train.T).T
    #pc_df(20,22)
    for model in ['PCA_LASSO', 'PCA_RF', 'PCA_LR', 'PCA_GBRT']:
        regressor = pca_models[model]
        y = [0] * 20
        for i in range(20):
            y[i] = regressor[i].predict(pc_df)[i]
            #pc_df(20,22)
        y_pred = pca.inverse_transform(pd.DataFrame(y).T)
        #50000
        temp = dif
        #259200
        indices = np.where(temp == 1)[0]
        temp[indices] = y_pred
        temp = pd.DataFrame(temp.reshape(360, 720))
        output_path = f'./全部年份预测/{model}_{year}.csv'
        temp.to_csv(output_path, index=False)
    
    # GRG模型
    for model in ['GRG_GBRT', 'GRG_LR', 'GRG_RF', 'GRG_LASSO']:
        regressor = grg_models[model]#50000
        pred = np.array(df).reshape(1, -1)[0].copy()
        #259200
        flag = 0
        for j in range(259200):
            if dif[j] == 0:
                continue
            pred[j] = regressor[flag].predict(data_pred[train_year, j, :].reshape(1, -1))[0]
            #data_pred[train_year, j, :] (1,22)
            flag += 1
        pred = pd.DataFrame(pred.reshape(360, 720))
        output_path = f'./全部年份预测/{model}_{year}.csv'
        pred.to_csv(output_path, index=False)
    
    # 简单平均
    output=np.mean(data_pred[train_year], axis=1).reshape(360,720)
    #22个模型
    output_path = f'./全部年份预测/Average_{year}.csv'
    pd.DataFrame(output).to_csv(output_path, index=False)
'''
import numpy as np
from joblib import load
import pandas as pd
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
df=pd.read_csv(f'./global_data/real_data1993.csv')
df[df>0]=1
df.columns = range(720)




area=[i for i in range(15)]
for i in range(15):
    area[i]=pd.read_csv(f'./地图分割/'+str(i)+'.csv')
    area[i].columns = range(720)
    area[i]=area[i]*df
for i in range(14):
    for j in range(i+1,15):
        temp=area[i]*area[j]
        area[j]=area[j]-temp
pca=load('./模型保存/pca20.joblib')

for train_year in range(251):
    temp = pd.DataFrame()
    for i in mod_name:
        data= pd.read_csv(f'./global_data/'+i+'_data'+str(1850+train_year)+'.csv')
        data.columns = range(720)
        data=data*df
        temp[i]=pd.DataFrame(np.array(data).reshape(1,-1)).T
    i=train_year
    data_train=temp
    #全局模型
    regressor = load('模型保存/Global_LASSO.joblib')
    y = regressor.predict(data_train)
    y=pd.DataFrame(y.reshape(360,720))
    y.columns = range(720)
    y=y*df
    y.to_csv(f'./全部年份预测/Global_LASSO'+str(1850+i)+'.csv',index=False)
    regressor = load('./模型保存/Global_RF.joblib')
    y = regressor.predict(data_train)
    y=pd.DataFrame(y.reshape(360,720))
    y.columns = range(720)
    y=y*df
    y.to_csv(f'./全部年份预测/Global_RF'+str(1850+i)+'.csv',index=False)
    regressor = load('./模型保存/Global_LR.joblib')
    y = regressor.predict(data_train)
    y=pd.DataFrame(y.reshape(360,720))
    y.columns = range(720)
    y=y*df
    y.to_csv(f'./全部年份预测/Global_LR'+str(1850+i)+'.csv',index=False)
    regressor = load('./模型保存/Global_GBRT.joblib')
    y = regressor.predict(data_train)
    y=pd.DataFrame(y.reshape(360,720))
    y.columns = range(720)
    y=y*df
    y.to_csv(f'./全部年份预测/Global_GBRT'+str(1850+i)+'.csv',index=False)
    
    #生态区
    data_pred=pd.DataFrame(np.zeros((360, 720)))
    data_pred.columns = range(720)
    for area_num in range(15):
        regressor = load('./模型保存/Geoarea_LR'+str(area_num)+'.joblib')
        y = regressor.predict(data_train)
        y=pd.DataFrame(y.reshape(360,720))
        y.columns = range(720)
        y=y*area[area_num]
        data_pred=data_pred+y
    data_pred.to_csv(f'./全部年份预测/Geoarea_LR'+str(1850+i)+'.csv',index=False)
    
    data_pred=pd.DataFrame(np.zeros((360, 720)))
    data_pred.columns = range(720)
    for area_num in range(15):
        regressor = load('./模型保存/Geoarea_RF'+str(area_num)+'.joblib')
        y = regressor.predict(data_train)
        y=pd.DataFrame(y.reshape(360,720))
        y.columns = range(720)
        y=y*area[area_num]
        data_pred=data_pred+y
    data_pred.to_csv(f'./全部年份预测/Geoarea_RF'+str(1850+i)+'.csv',index=False)
    
    data_pred=pd.DataFrame(np.zeros((360, 720)))
    data_pred.columns = range(720)
    for area_num in range(15):
        regressor = load('./模型保存/Geoarea_LASSO'+str(area_num)+'.joblib')
        y = regressor.predict(data_train)
        y=pd.DataFrame(y.reshape(360,720))
        y.columns = range(720)
        y=y*area[area_num]
        data_pred=data_pred+y
    data_pred.to_csv(f'./全部年份预测/Geoarea_LASSO'+str(1850+i)+'.csv',index=False)
    
    data_pred=pd.DataFrame(np.zeros((360, 720)))
    data_pred.columns = range(720)
    for area_num in range(15):
        regressor = load('./模型保存/Geoarea_GBRT'+str(area_num)+'.joblib')
        y = regressor.predict(data_train)
        y=pd.DataFrame(y.reshape(360,720))
        y.columns = range(720)
        y=y*area[area_num]
        data_pred=data_pred+y
    data_pred.to_csv(f'./全部年份预测/Geoarea_GBRT'+str(1850+i)+'.csv',index=False)
    

#PCA
pca=load('./模型保存/pca20.joblib')
for train_year in range(251):
    temp = pd.DataFrame()
    for i in mod_name:
        data= pd.read_csv(f'./global_data/'+i+'_data'+str(1850+train_year)+'.csv')
        data.columns = range(720)
        data=data*df
        temp[i]=pd.DataFrame(np.array(data).reshape(1,-1)).T
    data_train=temp
    y=[0]*20
    pc_df=pca.transform(data_train.T).T
    for i in range(20):
        regressor = load('./模型保存/PCA_GBRT'+str(i)+'.joblib')
        y[i] = regressor.predict(pc_df)[i]
    y_pred=pca.inverse_transform(pd.DataFrame(y).T)
    y_pred=pd.DataFrame(y_pred.reshape(360,720))
    y_pred.columns = range(720)
    y_pred=y_pred*df
    y_pred.to_csv(f'./全部年份预测/PCA_GBRT'+str(1850+train_year)+'.csv',index=False)
    for i in range(20):
        regressor = load('./模型保存/PCA_LR'+str(i)+'.joblib')
        y[i] = regressor.predict(pc_df)[i]
    y_pred=pca.inverse_transform(pd.DataFrame(y).T)
    y_pred=pd.DataFrame(y_pred.reshape(360,720))
    y_pred.columns = range(720)
    y_pred=y_pred*df
    y_pred.to_csv(f'./全部年份预测/PCA_LR'+str(1850+train_year)+'.csv',index=False)
    
    for i in range(20):
        regressor = load('./模型保存/PCA_RF'+str(i)+'.joblib')
        y[i] = regressor.predict(pc_df)[i]
    y_pred=pca.inverse_transform(pd.DataFrame(y).T)
    y_pred=pd.DataFrame(y_pred.reshape(360,720))
    y_pred.columns = range(720)
    y_pred=y_pred*df
    y_pred.to_csv(f'./全部年份预测/PCA_RF'+str(1850+train_year)+'.csv',index=False)
    
    for i in range(20):
        regressor = load('./模型保存/PCA_LASSO'+str(i)+'.joblib')
        y[i] = regressor.predict(pc_df)[i]
    y_pred=pca.inverse_transform(pd.DataFrame(y).T)
    y_pred=pd.DataFrame(y_pred.reshape(360,720))
    y_pred.columns = range(720)
    y_pred=y_pred*df
    y_pred.to_csv(f'./全部年份预测/PCA_LASSO'+str(1850+train_year)+'.csv',index=False)

#GRG

regressor1=[0]*259200
regressor2=[0]*259200
regressor3=[0]*259200
regressor4=[0]*259200
dif=np.array(df).reshape(1,-1)[0]

for j in range(259200):
    if dif[j]==0:
        continue
    if j <=172285:
        flag=''
    elif j<=218362:
        flag=1
    else:
        flag=2
    regressor1[j] = load('F:/模型平均/模型保存/GRG_GBRT'+str(flag)+'/GRG_GBRT'+str(j)+'.joblib')
    regressor2[j] = load('F:/模型平均/模型保存/GRG_LR'+str(flag)+'/GRG_LR'+str(j)+'.joblib')
    regressor3[j] = load('F:/模型平均/模型保存/GRG_RF'+str(flag)+'/GRG_RF'+str(j)+'.joblib')
    regressor4[j] = load('F:/模型平均/模型保存/GRG_LASSO'+str(flag)+'/GRG_LASSO'+str(j)+'.joblib')

for train_year in range(251):
    start_time = time.time()
    temp = pd.DataFrame()
    for i in mod_name:
        data= pd.read_csv(f'C:/Users/86189/Desktop/global_data/'+i+'_data'+str(1850+train_year)+'.csv')
        data.columns = range(720)
        data=data*df
        temp[i]=pd.DataFrame(np.array(data).reshape(1,-1)).T
    data_train=temp
    pred=[np.array(df).reshape(1,-1)[0]]*4
    flag=0
    for j in range(259200):
        if pred[0][j]==0:
            continue
        pred[0][j]=regressor1[j].predict(pd.DataFrame(data_train.iloc[j]).T)
        pred[1][j]=regressor2[j].predict(pd.DataFrame(data_train.iloc[j]).T)
        pred[2][j]=regressor3[j].predict(pd.DataFrame(data_train.iloc[j]).T)
        pred[3][j]=regressor4[j].predict(pd.DataFrame(data_train.iloc[j]).T)
    pd.DataFrame(pred[0].reshape(360,720)).to_csv(f'F:/模型平均/全部年份预测/GRG_GBRT'+str(1850+train_year)+'.csv',index=False)
    pd.DataFrame(pred[1].reshape(360,720)).to_csv(f'F:/模型平均/全部年份预测/GRG_LR'+str(1850+train_year)+'.csv',index=False)
    pd.DataFrame(pred[2].reshape(360,720)).to_csv(f'F:/模型平均/全部年份预测/GRG_RF'+str(1850+train_year)+'.csv',index=False)
    pd.DataFrame(pred[3].reshape(360,720)).to_csv(f'F:/模型平均/全部年份预测/GRG_LASSO'+str(1850+train_year)+'.csv',index=False)



#简单平均
for year in range(251):
    avg_data = pd.DataFrame(np.zeros((360, 720)))
    avg_data.columns = range(720)
    
    for model in mod_name:
        model_data = pd.read_csv(f'./global_data/'+i+'_data'+str(1850+train_year)+'.csv')
        model_data.columns = range(720)
        avg_data += model_data
    
    avg_data /= len(mod_name)
    avg_data.to_csv(f'./全部年份预测/Average_{1993 + year}.csv', index=False)
'''