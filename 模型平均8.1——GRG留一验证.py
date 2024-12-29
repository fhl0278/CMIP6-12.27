import numpy as np
import pandas as pd
from sklearn.linear_model import LassoCV, LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
import os
import warnings
from sklearn.exceptions import ConvergenceWarning
# 忽略所有 ConvergenceWarning
warnings.filterwarnings("ignore", category=ConvergenceWarning)

# 数据点总数
total_points = 259200
years = list(range(1993, 2013))  # 20年的时间范围


# 训练模型并预测
def train_and_predict(model, data_train, method_name):
    global predictions
    predictions = np.zeros((len(years), total_points))
    for year_idx, year in enumerate(years):
        print(f'Training for year {year}...')
        for point_idx in range(total_points):
            data = data_train[:, point_idx, :]
            #(20,22)
            data = np.delete(data, year_idx, axis=0)
            #(19,22)
            # 拆分为 X_train 和 y_train
            X_train, y_train = data[:, :-1], data[:, -1]
            if y_train.sum() == 0:  # 跳过海洋数据点
                predictions[year_idx, point_idx] = 0
                continue
            
            regressor = model()
            regressor.fit(X_train, y_train)
            
            prediction = regressor.predict(data_train[year_idx,point_idx,:-1].reshape(1, -1))
            
            predictions[year_idx, point_idx] = prediction[0]
        
        # 保存预测结果
        output_dir = './预测数据'
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        pd.DataFrame(predictions[year_idx].reshape(360, 720)).to_csv(os.path.join(output_dir, f'GPG_{method_name}{year}.csv'), index=False)
# 主函数
def main():
    data_train = np.load('./data_train.npy')

    models = {
        'LR': lambda: LinearRegression(),
        'LASSO': lambda: LassoCV(tol=1e-3, max_iter=1000),
        'RF': lambda: RandomForestRegressor(n_estimators=30, random_state=0, max_depth=3),
        'GBRT': lambda: GradientBoostingRegressor(n_estimators=30, random_state=0, max_depth=3)
    }
    
    for method_name, model in models.items():
        train_and_predict(model, data_train,method_name)
if __name__ == "__main__":
    main()
'''
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import pandas as pd
import time



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
l=259200

#读取数据

data_train=[i for i in range(20)]
for train_year in range(20):
    df = pd.DataFrame()
    for i in mod_name:
        data= pd.read_csv(f'./global_data/'+i+'_data'+str(1993+train_year)+'.csv')
        df[i]=pd.DataFrame(np.array(data).reshape(1,-1)).T
    temp=pd.read_csv(f'./global_data/real_data'+str(1993+train_year)+'.csv')
    df['real']=pd.DataFrame(np.array(temp).reshape(1,-1)).T
    data_train[train_year]=df



#dot_data保存了259200个数据点共20年的数据，用于后续训练模型
dot_data=[i for i in range(l)]
concatenated_df = pd.DataFrame()
for i in range(0,l):
    concatenated_row = pd.concat([df.iloc[i] for df in data_train], axis=1)
    concatenated_df = pd.concat([concatenated_df, concatenated_row.T], ignore_index=True)
    dot_data[i]=concatenated_df
    concatenated_df = pd.DataFrame()




import sklearn.linear_model as skl
rows = 20
cols = l
y_real=[0] * 20
y_average=[0] * 20
y_p=[0] * 20
y_pred=[[0 for _ in range(cols)] for _ in range(rows)]
#分20年训练
for k in range(20):
    regressor=[0]*259200#为每个数据点分配一个训练器
    for i in range(259200):

        data=dot_data[i].drop(dot_data[i].index[k])#出于留一验证考虑，数据点需要去除对应年份的数据
        X_train=data.iloc[:,:-1]
        y_train=data.iloc[:, -1]
        if y_train.sum()==0:#如果该点是海洋，跳过
            y_pred[k][i]=0
            continue
        #模型的训练和预测
        regressor[i] = skl.LassoCV(tol=1e-4, max_iter=10000)
        regressor[i].fit(X_train,y_train)
        y_pred[k][i] = float(regressor[i].predict(pd.DataFrame(data_train[k].iloc[i][:-1]).T))
    #将259200个数据点转化为所需格式并储存
    y_pred[k]=np.array(y_pred[k])
    y=pd.DataFrame(y_pred[k].reshape(360,720))
    y.to_csv(f'./预测数据/GPG_LASSO'+str(1993+k)+'.csv',index=False)



from sklearn.ensemble import RandomForestRegressor
rows = 20
cols = l
y_real=[0] * 20
y_average=[0] * 20
y_p=[0] * 20
y_pred=[[0 for _ in range(cols)] for _ in range(rows)]

for k in range(20):
    regressor=[0]*259200
    for i in range(259200):
        data=dot_data[i].drop(dot_data[i].index[k])
        X_train=data.iloc[:,:-1]
        y_train=data.iloc[:, -1]
        if y_train.sum()==0:
            y_pred[k][i]=0
            continue
        regressor[i] = RandomForestRegressor(n_estimators=30, random_state=0,max_depth=3)
        regressor[i].fit(X_train,y_train)
        y_pred[k][i] = float(regressor[i].predict(pd.DataFrame(data_train[k].iloc[i][:-1]).T))
    y_pred[k]=np.array(y_pred[k])
    y=pd.DataFrame(y_pred[k].reshape(360,720))
    y.to_csv(f'./预测数据/GPG_RF'+str(1993+k)+'.csv',index=False)


from sklearn import linear_model
rows = 20
cols = l
y_real=[0] * 20
y_average=[0] * 20
y_p=[0] * 20
y_pred=[[0 for _ in range(cols)] for _ in range(rows)]

for k in range(20):
    regressor=[0]*259200
    for i in range(259200):

        data=dot_data[i].drop(dot_data[i].index[k])
        X_train=data.iloc[:,:-1]
        y_train=data.iloc[:, -1]
        if y_train.sum()==0:
            y_pred[k][i]=0
            continue
        regressor[i] = linear_model.LinearRegression()
        regressor[i].fit(X_train,y_train)
        y_pred[k][i] = float(regressor[i].predict(pd.DataFrame(data_train[k].iloc[i][:-1]).T))
    y_pred[k]=np.array(y_pred[k])
    y=pd.DataFrame(y_pred[k].reshape(360,720))
    y.to_csv(f'./预测数据/GPG_LR'+str(1993+k)+'.csv',index=False)
    
import sklearn.ensemble as ske
rows = 20
cols = l
y_real=[0] * 20
y_average=[0] * 20
y_p=[0] * 20
y_pred=[[0 for _ in range(cols)] for _ in range(rows)]

for k in range(20):
    regressor=[0]*259200
    for i in range(259200):

        data=dot_data[i].drop(dot_data[i].index[k])
        X_train=data.iloc[:,:-1]
        y_train=data.iloc[:, -1]
        if y_train.sum()==0:
            y_pred[k][i]=0
            continue
        regressor[i] = ske.GradientBoostingRegressor(n_estimators=30, random_state=0,max_depth=3)
        regressor[i].fit(X_train,y_train)
        y_pred[k][i] = float(regressor[i].predict(pd.DataFrame(data_train[k].iloc[i][:-1]).T))
    y_pred[k]=np.array(y_pred[k])
    y=pd.DataFrame(y_pred[k].reshape(360,720))
    y.to_csv(f'./预测数据/GPG_LGBRT'+str(1993+k)+'.csv',index=False)
''' 





