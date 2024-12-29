import numpy as np
import pandas as pd
from sklearn import linear_model
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LassoCV

df = pd.read_csv('./global_data/real_data1993.csv')
df[df > 0] = 1  # 将所有正数值设置为1
df.columns = range(720)  # 设置列名

# 读取生态区分割方案
area = []
for i in range(15):
    area.append(pd.read_csv(f'./地图分割/{i}.csv'))
    area[i].columns = range(720)
    area[i] *= df  # 确保生态区与数据匹配

# 确保生态区没有重叠
for i in range(14):
    for j in range(i + 1, 15):
        temp = area[i] * area[j]
        area[j] -= temp

# 读取所需数据
data_train = np.load('./data_train.npy')

# 预留空间存放预测结果
data_pred = [pd.DataFrame(np.zeros((360, 720)), columns=range(720)) for _ in range(20)]


# 重复上述步骤，分别使用 RandomForestRegressor 和 LassoCV
for regressor_class, file_prefix in [(linear_model, 'Geoarea_LR'),(RandomForestRegressor, 'Geoarea_RF'), (LassoCV, 'Geoarea_LASSO'), (GradientBoostingRegressor, 'Geoarea_GBRT')]:
    data_pred = [pd.DataFrame(np.zeros((360, 720)), columns=range(720)) for _ in range(20)]
    for area_num in range(15):
        arealist = np.array(area[area_num]).reshape(1, -1)[0].reshape(1,259200,1)


        regressor = [None] * 20
        for i in range(20):
            result_area = (np.delete(data_train, i, axis=0)*arealist).reshape(19*259200,23)
    
            t = result_area[result_area.any(axis=1)]
            
            X_train = t[:, :-1]
            y_train = t[:, -1]

            if regressor_class is LassoCV:
                regressor[i] = LassoCV(tol=1e-3, max_iter=1000).fit(X_train, y_train)
            elif regressor_class is RandomForestRegressor:
                regressor[i] = RandomForestRegressor(n_estimators=30, random_state=0, max_depth=3).fit(X_train, y_train)
            elif regressor_class is GradientBoostingRegressor:
                regressor[i] = GradientBoostingRegressor(n_estimators=30, random_state=0, max_depth=3).fit(X_train, y_train)
            elif regressor_class is linear_model:
                regressor[i] = linear_model.LinearRegression().fit(X_train, y_train)
            y_pred = regressor[i].predict(data_train[i,:, :-1])
            y_pred = pd.DataFrame(y_pred.reshape(360, 720), columns=range(720)) * area[area_num]
            data_pred[i] += y_pred

    for i in range(20):
        data_pred[i].to_csv(f'./预测数据/{file_prefix}{1993 + i}.csv', index=False)

print('预测完成')


'''
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
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
#读取生态区分割方案
for i in range(15):
    area[i]=pd.read_csv(f'./地图分割/'+str(i)+'.csv')
    area[i].columns = range(720)
    area[i]=area[i]*df

#确保生态区没有重叠
for i in range(14):
    for j in range(i+1,15):
        temp=area[i]*area[j]
        area[j]=area[j]-temp
        
#读取所需数据
data_train=[i for i in range(20)]
for train_year in range(20):
    temp = pd.DataFrame()
    for i in mod_name:
        data= pd.read_csv(f'./global_data/'+i+'_data'+str(1993+train_year)+'.csv')
        temp[i]=pd.DataFrame(np.array(data).reshape(1,-1)).T
    t=pd.read_csv(f'./global_data/real_data'+str(1993+train_year)+'.csv')
    temp['real']=pd.DataFrame(np.array(t).reshape(1,-1)).T
    data_train[train_year]=temp

#预留空间存放预测结果
data_pred=[i for i in range(20)]
for i in range(20):
    data_pred[i]=pd.DataFrame(np.zeros((360, 720)))
    data_pred[i].columns = range(720)

#划为15个生态区，逐一预测
for area_num in range(15):
    #用0和1区分该生态区的区域
    arealist=pd.DataFrame(np.array(area[area_num]).reshape(1,-1)).T
    #整理20年留一验证训练样本
    result=[0]*20
    for i in range(20):
        result[i]=pd.DataFrame()
        for j in range(0, len(data_train)):
            if j!=i:
                result[i] = pd.concat([result[i], data_train[j].mul(arealist.iloc[:,0],axis=0)])
    from sklearn import linear_model
    rows = 20

    regressor = [0]*20
    
    for i in range(20):

        t=result[i] 
        t=t[~(t == 0).all(axis=1)]
        X_train=t.iloc[:,:-1]
        y_train=t.iloc[:, -1]

        regressor[i]=linear_model.LinearRegression()
        regressor[i].fit(X_train,y_train)

        train=data_train[i].iloc[:,:-1]
        y = regressor[i].predict(train)

        y=pd.DataFrame(y.reshape(360,720))
        y.columns = range(720)
        y=y*area[area_num]
        #用一个生态区的训练结果对全球预测，但只保留对应生态区的预测结果
        #15个生态区的预测结果相加，即为最后的结果
        data_pred[i]=data_pred[i]+y

for i in range(20):
    data_pred[i].to_csv(f'./预测数据/GeoareaLR'+str(1993+i)+'.csv',index=False)



data_pred=[i for i in range(20)]
for i in range(20):
    data_pred[i]=pd.DataFrame(np.zeros((360, 720)))
    data_pred[i].columns = range(720)
for area_num in range(15):
    arealist=pd.DataFrame(np.array(area[area_num]).reshape(1,-1)).T
    result=[0]*20
    for i in range(20):

        result[i]=pd.DataFrame()
        # 遍历剩余的 DataFrame
        for j in range(0, len(data_train)):
            if j!=i:
            # 将当前 DataFrame 的列拼接到结果 DataFrame
                result[i] = pd.concat([result[i], data_train[j].mul(arealist.iloc[:,0],axis=0)])

            # 显示最终的 DataFrame   
    from sklearn.ensemble import RandomForestRegressor

    rows = 20


    # 使用嵌套列表推导式创建二维列表
    regressor = [0]*20

    for i in range(20):

        t=result[i] 
        t=t[~(t == 0).all(axis=1)]
        X_train=t.iloc[:,:-1]
        y_train=t.iloc[:, -1]

        regressor[i] = RandomForestRegressor(n_estimators=30, random_state=0,max_depth=3)

        regressor[i].fit(X_train,y_train)

        train=data_train[i].iloc[:,:-1]
        y = regressor[i].predict(train)

        y=pd.DataFrame(y.reshape(360,720))
        y.columns = range(720)
        y=y*area[area_num]
 
        data_pred[i]=data_pred[i]+y

for i in range(20):
    data_pred[i].to_csv(f'./预测数据/GeoareaRF'+str(1993+i)+'.csv',index=False)


data_pred=[i for i in range(20)]
for i in range(20):
    data_pred[i]=pd.DataFrame(np.zeros((360, 720)))
    data_pred[i].columns = range(720)
for area_num in range(15):
    arealist=pd.DataFrame(np.array(area[area_num]).reshape(1,-1)).T
    result=[0]*20
    for i in range(20):

        result[i]=pd.DataFrame()
        # 遍历剩余的 DataFrame
        for j in range(0, len(data_train)):
            if j!=i:
            # 将当前 DataFrame 的列拼接到结果 DataFrame
                result[i] = pd.concat([result[i], data_train[j].mul(arealist.iloc[:,0],axis=0)])


    import sklearn.linear_model as skl

    rows = 20


    # 使用嵌套列表推导式创建二维列表
    regressor = [0]*20
    #对某一年进行预测，则训练集不包含这一年的数据。对15个主成分因子进行预测.。i控制年份，j控制主成分
    for i in range(20):

        t=result[i] 
        t=t[~(t == 0).all(axis=1)]
        X_train=t.iloc[:,:-1]
        y_train=t.iloc[:, -1]

        regressor[i]=skl.LassoCV(tol=1e-4, max_iter=10000)
        regressor[i].fit(X_train,y_train)

        train=data_train[i].iloc[:,:-1]
        y = regressor[i].predict(train)

        y=pd.DataFrame(y.reshape(360,720))
        y.columns = range(720)
        y=y*area[area_num]
 
        data_pred[i]=data_pred[i]+y



for i in range(20):
    data_pred[i].to_csv(f'./预测数据/GeoareaLASSO'+str(1993+i)+'.csv',index=False)


data_pred=[i for i in range(20)]
for i in range(20):
    data_pred[i]=pd.DataFrame(np.zeros((360, 720)))
    data_pred[i].columns = range(720)
for area_num in range(15):
    arealist=pd.DataFrame(np.array(area[area_num]).reshape(1,-1)).T
    result=[0]*20
    for i in range(20):

        result[i]=pd.DataFrame()
        # 遍历剩余的 DataFrame
        for j in range(0, len(data_train)):
            if j!=i:
            # 将当前 DataFrame 的列拼接到结果 DataFrame
                result[i] = pd.concat([result[i], data_train[j].mul(arealist.iloc[:,0],axis=0)])


    import sklearn.ensemble as ske

    rows = 20

    regressor = [0]*20
  
    for i in range(20):

        t=result[i] 
        t=t[~(t == 0).all(axis=1)]
        X_train=t.iloc[:,:-1]
        y_train=t.iloc[:, -1]

        regressor[i] = ske.GradientBoostingRegressor(n_estimators=30, random_state=0,max_depth=3)
        #regressor[i]=linear_model.LinearRegression()
        #regressor[i]=skl.LassoCV(tol=1e-4, max_iter=10000)
        regressor[i].fit(X_train,y_train)

        train=data_train[i].iloc[:,:-1]
        y = regressor[i].predict(train)

        y=pd.DataFrame(y.reshape(360,720))
        y.columns = range(720)
        y=y*area[area_num]
  
        data_pred[i]=data_pred[i]+y
    print('生态区数：',area_num)

for i in range(20):
    data_pred[i].to_csv(f'./预测数据/GeoareaGBRT'+str(1993+i)+'.csv',index=False)
'''
