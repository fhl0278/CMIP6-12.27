import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, LassoCV
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.neural_network import MLPRegressor
import warnings
from sklearn.exceptions import ConvergenceWarning

warnings.filterwarnings(action='ignore', category=ConvergenceWarning)

# 模型名称列表


# 读取真实数据并标记陆地
df = pd.read_csv('./global_data/real_data1993.csv')
df[df > 0] = 1
df.columns = range(720)
dif = np.array(df).reshape(1, -1)[0]
# 准备训练数据
data_train = np.load('./data_train.npy')

# 控制主成分数量
pca_count = 20

# 用留一验证的原则进行PCA，每一年的PCA降维训练数据来源于其余19年的数据
pca = [None] * 20
pca_outcome = [None] * 20

for i in range(20):
    pca_train = np.delete(data_train, i, axis=0)
    
    pca_train =pca_train.transpose(1, 0, 2).reshape(259200,23*19).T
    
    pca_train = pca_train[:, pca_train.any(axis=0)]
    pca[i] = PCA(n_components=pca_count)
    pca_outcome[i] = pca[i].fit_transform(pca_train)

# 初始化回归模型
regressors = {
    'RF': RandomForestRegressor,
    'LR': LinearRegression,
    'LASSO': LassoCV,
    'GBRT': GradientBoostingRegressor
}

for model_name, RegressorClass in regressors.items():
    regressor = [[None for _ in range(pca_count)] for _ in range(20)]
    for i in range(20):
        for j in range(pca_count):
            # 从 pca_outcome 中提取数据
            data = pca_outcome[i][:, j].reshape(19, 23)
            #(437,20)
            X_train = data[:, :-1]
            y_train = data[:, -1]
            
            # 创建一个新的对象实例
            if model_name in ['RF', 'GBRT'] :
                regressor[i][j] = RegressorClass(n_estimators=30, random_state=0, max_depth=3)
            elif model_name == 'LASSO':
                regressor[i][j]=RegressorClass(tol=1e-3, max_iter=1000)
            else:
                regressor[i][j]=RegressorClass()
            #LASSO参数
            # 训练模型
            regressor[i][j].fit(X_train, y_train)
    for i in range(20):
        data=data_train[i,:,:-1]
        #(259200,22)
        data=data[~(data == 0).all(axis=1)]
        X2_pca = pca[i].transform(data.T)
        y = [None] * pca_count
        for j in range(pca_count):
            #y[j] = regressor[i][j].predict(X2_pca[:, j].reshape(1, -1))[0]
            y[j]=regressor[i][j].predict(X2_pca.T)[j]
        y_pred = pca[i].inverse_transform(y)
        outcome = dif.copy()
        indices = np.where(dif == 1)[0]
        outcome[indices] = y_pred
        outcome = pd.DataFrame(outcome.reshape(360, 720))
        outcome.columns = range(720)
        outcome = outcome * df
        outcome.to_csv(f'./预测数据/PCA_{model_name}{1993 + i}.csv', index=False)


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



data_train=[i for i in range(20)]
for train_year in range(20):
    temp = pd.DataFrame()
    for i in mod_name:
        data= pd.read_csv(f'./global_data/'+i+'_data'+str(1993+train_year)+'.csv')*df
        temp[i]=pd.DataFrame(np.array(data).reshape(1,-1)).T
    t=pd.read_csv(f'./global_data/real_data'+str(1993+train_year)+'.csv')
    temp['real']=pd.DataFrame(np.array(t).reshape(1,-1)).T
    data_train[train_year]=temp


result=[0]*20
#result储存20年用于降维的数据，数据规模是259200（行）*（23（模型与与观察数据）*19（年））（列）
for i in range(20):
    result[i]=pd.DataFrame()
    for j in range(0, len(data_train)):
        if j!=i:
            result[i] = pd.concat([result[i], data_train[j]], axis=1)


#控制主成分数量
pca_count=20
#用留一验证的原则进行pca，每一年的pca降维训练数据来源于其余19年的数据
pca=[0]*20#用于储存模型
pc_df=[0]*20#用于储存降维后的结果
for i in range(20):
    pca[i] = PCA(n_components=pca_count)  
    principal_components = pca[i].fit_transform(result[i].T)
    pc_df[i] = pd.DataFrame(principal_components)  # 列名根据主成分数量调整

import sklearn.linear_model as skl
from sklearn.ensemble import RandomForestRegressor
import sklearn.ensemble as ske
from sklearn import linear_model
from sklearn.neural_network import MLPRegressor
import warnings
from sklearn.exceptions import ConvergenceWarning
rows = 20
cols = pca_count

# 使用嵌套列表推导式创建二维列表
regressor = [[0 for _ in range(cols)] for _ in range(rows)]
#对某一年进行预测，则训练集不包含这一年的数据。对15个主成分因子进行预测.。i控制年份，j控制主成分
for i in range(20):
    new_data = [0]*pca_count
#重新整理数据，依照主成分划分，每个主成分有19行（除了预测年份外的所有年份）*23列（模型与观测数据）
    for j in range(pca_count):
        new_data[j]=pd.DataFrame((pc_df[i].T).iloc[j].values.reshape(19,23))

    for j in range(pca_count):
        data=new_data[j]
        X_train=data.iloc[:,:-1]
        y_train=data.iloc[:, -1]
#随机森林
        regressor[i][j] = RandomForestRegressor(n_estimators=30, random_state=0,max_depth=3)
 
        regressor[i][j].fit(X_train,y_train)



y_pred=[0]*20

for i in range(20):

    X2_pca=(pd.DataFrame(pca[i].transform(data_train[i].T)).T)#把当年的259200个数据点转化为20个主成分
    train=pd.DataFrame(X2_pca).iloc[:,:-1]
    y=[0]*pca_count
    for j in range(pca_count):#预测20个主成分
        y[j] = regressor[i][j].predict(pd.DataFrame(train.iloc[j]).T)
    y_pred[i]=pca[i].inverse_transform(pd.DataFrame(y).T)#将20个主成分预测结果还原为259200个数据
    y=pd.DataFrame(y_pred[i].reshape(360,720))
    y.to_csv(f'./预测数据/PCA_RF'+str(1993+i)+'.csv',index=False)

for i in range(20):
    new_data = [0]*pca_count
#重新整理数据，依照主成分划分，每个主成分有19行（除了预测年份外的所有年份）*23列（模型与观测数据）
    for j in range(pca_count):
        new_data[j]=pd.DataFrame((pc_df[i].T).iloc[j].values.reshape(19,23))

    for j in range(pca_count):
        data=new_data[j]
        X_train=data.iloc[:,:-1]
        y_train=data.iloc[:, -1]

        regressor[i][j] = linear_model.LinearRegression()
 
        regressor[i][j].fit(X_train,y_train)



y_pred=[0]*20

for i in range(20):

    X2_pca=(pd.DataFrame(pca[i].transform(data_train[i].T)).T)#把当年的259200个数据点转化为20个主成分
    train=pd.DataFrame(X2_pca).iloc[:,:-1]
    y=[0]*pca_count
    for j in range(pca_count):#预测20个主成分
        y[j] = regressor[i][j].predict(pd.DataFrame(train.iloc[j]).T)
    y_pred[i]=pca[i].inverse_transform(pd.DataFrame(y).T)#将20个主成分预测结果还原为259200个数据
    y=pd.DataFrame(y_pred[i].reshape(360,720))
    y.to_csv(f'./预测数据/PCA_LR'+str(1993+i)+'.csv',index=False)


for i in range(20):
    new_data = [0]*pca_count
#重新整理数据，依照主成分划分，每个主成分有19行（除了预测年份外的所有年份）*23列（模型与观测数据）
    for j in range(pca_count):
        new_data[j]=pd.DataFrame((pc_df[i].T).iloc[j].values.reshape(19,23))

    for j in range(pca_count):
        data=new_data[j]
        X_train=data.iloc[:,:-1]
        y_train=data.iloc[:, -1]

        regressor[i][j] = skl.LassoCV(tol=1e-4, max_iter=10000)
 
        regressor[i][j].fit(X_train,y_train)



y_pred=[0]*20

for i in range(20):

    X2_pca=(pd.DataFrame(pca[i].transform(data_train[i].T)).T)#把当年的259200个数据点转化为20个主成分
    train=pd.DataFrame(X2_pca).iloc[:,:-1]
    y=[0]*pca_count
    for j in range(pca_count):#预测20个主成分
        y[j] = regressor[i][j].predict(pd.DataFrame(train.iloc[j]).T)
    y_pred[i]=pca[i].inverse_transform(pd.DataFrame(y).T)#将20个主成分预测结果还原为259200个数据
    y=pd.DataFrame(y_pred[i].reshape(360,720))
    y.to_csv(f'./预测数据/PCA_LASSO'+str(1993+i)+'.csv',index=False)

for i in range(20):
    new_data = [0]*pca_count
#重新整理数据，依照主成分划分，每个主成分有19行（除了预测年份外的所有年份）*23列（模型与观测数据）
    for j in range(pca_count):
        new_data[j]=pd.DataFrame((pc_df[i].T).iloc[j].values.reshape(19,23))

    for j in range(pca_count):
        data=new_data[j]
        X_train=data.iloc[:,:-1]
        y_train=data.iloc[:, -1]

        regressor[i][j] = ske.GradientBoostingRegressor(n_estimators=30, random_state=0,max_depth=3)
 
        regressor[i][j].fit(X_train,y_train)



y_pred=[0]*20

for i in range(20):

    X2_pca=(pd.DataFrame(pca[i].transform(data_train[i].T)).T)#把当年的259200个数据点转化为20个主成分
    train=pd.DataFrame(X2_pca).iloc[:,:-1]
    y=[0]*pca_count
    for j in range(pca_count):#预测20个主成分
        y[j] = regressor[i][j].predict(pd.DataFrame(train.iloc[j]).T)
    y_pred[i]=pca[i].inverse_transform(pd.DataFrame(y).T)#将20个主成分预测结果还原为259200个数据
    y=pd.DataFrame(y_pred[i].reshape(360,720))
    y.to_csv(f'./预测数据/PCA_GBRT'+str(1993+i)+'.csv',index=False)
'''