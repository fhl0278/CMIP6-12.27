
#该代码用20年数据训练最终模型，模型用于预测1850-2100年的cvge值
import numpy as np
import pandas as pd
from joblib import dump
import sklearn.linear_model as skl
from sklearn import linear_model
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.decomposition import PCA
from sklearn.exceptions import ConvergenceWarning
import warnings

# 读取真实数据
df = pd.read_csv('./global_data/real_data1993.csv')
df[df > 0] = 1


# 准备训练数据
data_train = np.load('./data_train.npy')
#(20,259200,23)
# 合并所有训练数据
result = data_train.reshape(20 * 259200, 22+1)

# 训练全局模型
t = result[result.any(axis=1)]
X_train = t[:, :-1]
y_train = t[:, -1]
#(20 * 259200, 22)
#(20 * 259200, 1)
# Gradient Boosting Regressor
regressor = GradientBoostingRegressor(n_estimators=30, random_state=0, max_depth=3)
regressor.fit(X_train, y_train)
dump(regressor, './模型保存/Global_GBRT.joblib')

# Linear Regression
regressor = linear_model.LinearRegression()
regressor.fit(X_train, y_train)
dump(regressor, './模型保存/Global_LR.joblib')

# LASSO
regressor = skl.LassoCV(tol=1e-4, max_iter=10000)
regressor.fit(X_train, y_train)
dump(regressor, './模型保存/Global_LASSO.joblib')

# Random Forest Regressor
regressor = RandomForestRegressor(n_estimators=30, random_state=0, max_depth=3)
regressor.fit(X_train, y_train)
dump(regressor, './模型保存/Global_RF.joblib')

# 训练生态区模型
area = [pd.read_csv(f'./地图分割/{i}.csv') * df for i in range(15)]
for i in range(14):
    for j in range(i + 1, 15):
        area[j] = area[j] - (area[i] * area[j])
#(0-1)
for area_num in range(15):
    arealist = np.array(area[area_num]).reshape(1,-1,1)
    #area[area_num]生态区划分方案
    #(1*259200*1)
    result_area = (data_train*arealist).reshape(20*259200,23)
    #(20*259200*23)
    t = result_area[result_area.any(axis=1)]
    X_train = t[:, :-1]
    y_train = t[:, -1]

    # Linear Regression
    regressor = linear_model.LinearRegression()
    regressor.fit(X_train, y_train)
    dump(regressor, f'./模型保存/Geoarea_LR{area_num}.joblib')

    # Random Forest Regressor
    regressor = RandomForestRegressor(n_estimators=30, random_state=0, max_depth=3)
    regressor.fit(X_train, y_train)
    dump(regressor, f'./模型保存/Geoarea_RF{area_num}.joblib')

    # Gradient Boosting Regressor
    regressor = GradientBoostingRegressor(n_estimators=30, random_state=0, max_depth=3)
    regressor.fit(X_train, y_train)
    dump(regressor, f'./模型保存/Geoarea_GBRT{area_num}.joblib')

    # LASSO
    regressor = skl.LassoCV(tol=1e-3, max_iter=1000)
    regressor.fit(X_train, y_train)
    dump(regressor, f'./模型保存/Geoarea_LASSO{area_num}.joblib')

# 训练PCA模型
pca_count = 20
pca_train = data_train.transpose(1, 0, 2).reshape(259200,23*20).T
#(20*259200*23)  #(259200,20,23)
#(259200,460)
pca_train = pca_train[:, pca_train.any(axis=0)]

pca = PCA(n_components=pca_count)
pca_outcome = pca.fit_transform(pca_train)
#(460,20)
dump(pca, './模型保存/pca20.joblib')
mode_LR=[]
mode_RF=[]
mode_GBRT=[]
mode_LASSO=[]
for j in range(pca_count):
    
    data = pca_outcome[:, j].reshape(20, 23)
    #(460,i)
    X_train = data[:, :-1]
    y_train = data[:, -1]

    # Linear Regression
    regressor = linear_model.LinearRegression()
    regressor.fit(X_train, y_train)
    mode_LR.append(regressor)

    # Random Forest Regressor
    regressor = RandomForestRegressor(n_estimators=30, random_state=0, max_depth=3)
    regressor.fit(X_train, y_train)
    mode_RF.append(regressor)

    # Gradient Boosting Regressor
    regressor = GradientBoostingRegressor(n_estimators=30, random_state=0, max_depth=3)
    regressor.fit(X_train, y_train)
    mode_GBRT.append(regressor)

    # LASSO
    regressor = skl.LassoCV(tol=1e-3, max_iter=1000)
    regressor.fit(X_train, y_train)
    mode_LASSO.append(regressor)
    
dump(mode_LR, './模型保存/PCA_LR.pkl')
dump(mode_RF, './模型保存/PCA_RF.pkl')
dump(mode_GBRT, './模型保存/PCA_GBRT.pkl')
dump(mode_LASSO, './模型保存/PCA_LASSO.pkl')
# 训练GRG模型
l = 259200
mode_LR=[]
mode_RF=[]
mode_GBRT=[]
mode_LASSO=[]
warnings.filterwarnings('ignore', category=ConvergenceWarning)

for i in range(l):
    data = data_train[:,i,:]
    #(20,1,23)
    X_train = data[:, :-1]
    y_train = data[:, -1]
    if y_train.sum() == 0:
        continue

    # Linear Regression
    regressor = linear_model.LinearRegression()
    regressor.fit(X_train, y_train)
    mode_LR.append(regressor)

    # Random Forest Regressor
    regressor = RandomForestRegressor(n_estimators=30, random_state=0, max_depth=3)
    regressor.fit(X_train, y_train)
    mode_RF.append(regressor)

    # Gradient Boosting Regressor
    regressor = GradientBoostingRegressor(n_estimators=30, random_state=0, max_depth=3)
    regressor.fit(X_train, y_train)
    mode_GBRT.append(regressor)

    # LASSO
    regressor = skl.LassoCV(tol=1e-3, max_iter=1000)
    regressor.fit(X_train, y_train)
    mode_LASSO.append(regressor)
    
dump(mode_LR, './模型保存/GRG_LR.pkl')
dump(mode_RF, './模型保存/GRG_RF.pkl')
dump(mode_GBRT, './模型保存/GRG_GBRT.pkl')
dump(mode_LASSO, './模型保存/GRG_LASSO.pkl')
'''
import numpy as np
import pandas as pd
from joblib import dump
import sklearn.linear_model as skl
from sklearn.ensemble import RandomForestRegressor
import sklearn.ensemble as ske
from sklearn import linear_model
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



data_train=[i for i in range(20)]
for train_year in range(20):
    temp = pd.DataFrame()
    for i in mod_name:
        data= pd.read_csv(f'./global_data/'+i+'_data'+str(1993+train_year)+'.csv')
        data.columns = range(720)
        data=data*df
        temp[i]=pd.DataFrame(np.array(data).reshape(1,-1)).T
    t=pd.read_csv(f'./global_data/real_data'+str(1993+train_year)+'.csv')
    temp['real']=pd.DataFrame(np.array(t).reshape(1,-1)).T
    data_train[train_year]=temp



result=pd.DataFrame()
# 遍历剩余的 DataFrame
for j in range(0, len(data_train)):
    # 将当前 DataFrame 的列拼接到结果 DataFrame
        result = pd.concat([result, data_train[j]])

    # 显示最终的 DataFrame



#训练全局模型
import sklearn.ensemble as ske
rows = 20


t=result 
t=t[~(t == 0).all(axis=1)]
X_train=t.iloc[:,:-1]
y_train=t.iloc[:, -1]

regressor = ske.GradientBoostingRegressor(n_estimators=30, random_state=0,max_depth=3)

regressor.fit(X_train,y_train)
dump(regressor, './模型保存/Global_GBRT.joblib')




from sklearn import linear_model
rows = 20


t=result 
t=t[~(t == 0).all(axis=1)]
X_train=t.iloc[:,:-1]
y_train=t.iloc[:, -1]

regressor = linear_model.LinearRegression()

regressor.fit(X_train,y_train)
dump(regressor, './模型保存/Global_LR.joblib')



import sklearn.linear_model as skl
rows = 20


t=result 
t=t[~(t == 0).all(axis=1)]
X_train=t.iloc[:,:-1]
y_train=t.iloc[:, -1]

regressor =skl.LassoCV(tol=1e-4, max_iter=10000)

regressor.fit(X_train,y_train)
dump(regressor, './模型保存/Global_LASSO.joblib')



from sklearn.ensemble import RandomForestRegressor
rows = 20


t=result 
t=t[~(t == 0).all(axis=1)]
X_train=t.iloc[:,:-1]
y_train=t.iloc[:, -1]

regressor =RandomForestRegressor(n_estimators=30, random_state=0,max_depth=3)

regressor.fit(X_train,y_train)
dump(regressor, './模型保存/Global_RF.joblib')


#训练生态区模型
area=[i for i in range(15)]
for i in range(15):
    area[i]=pd.read_csv(f'./地图分割/'+str(i)+'.csv')
    area[i].columns = range(720)
    area[i]=area[i]*df
for i in range(14):
    for j in range(i+1,15):
        temp=area[i]*area[j]
        area[j]=area[j]-temp



data_pred=[i for i in range(20)]

for area_num in range(15):
    arealist=pd.DataFrame(np.array(area[area_num]).reshape(1,-1)).T

    result=pd.DataFrame()

    for j in range(0, len(data_train)):
        result = pd.concat([result, data_train[j].mul(arealist.iloc[:,0],axis=0)])
        
    t=result
    t=t[~(t == 0).all(axis=1)]
    X_train=t.iloc[:,:-1]
    y_train=t.iloc[:, -1]

    regressor=linear_model.LinearRegression()
    regressor.fit(X_train,y_train)
    dump(regressor, './模型保存/Geoarea_LR'+str(area_num)+'.joblib')
    
    regressor=RandomForestRegressor(n_estimators=30, random_state=0,max_depth=3)
    regressor.fit(X_train,y_train)
    dump(regressor, './模型保存/Geoarea_RF'+str(area_num)+'.joblib')
    
    regressor=ske.GradientBoostingRegressor(n_estimators=30, random_state=0,max_depth=3)
    regressor.fit(X_train,y_train)
    dump(regressor, './模型保存/Geoarea_GBRT'+str(area_num)+'.joblib')
    
    regressor=skl.LassoCV(tol=1e-4, max_iter=10000)
    regressor.fit(X_train,y_train)
    dump(regressor, './模型保存/Geoarea_LASSO'+str(area_num)+'.joblib')
    

#训练pca模型

from sklearn.decomposition import PCA
pca_count=20
result2=pd.DataFrame()
for j in range(0, len(data_train)):

    result2 = pd.concat([result2, data_train[j]], axis=1)


pca = PCA(n_components=pca_count) 

principal_components = pca.fit_transform(result2.T)

pc_df = pd.DataFrame(principal_components)  # 列名根据你的主成分数量调整

dump(pca, './模型保存/pca20.joblib')




import warnings
from sklearn.exceptions import ConvergenceWarning
rows = 20
cols = pca_count


new_data = [0]*pca_count

for j in range(pca_count):
    new_data[j]=pd.DataFrame((pc_df.T).iloc[j].values.reshape(20,23))

for j in range(pca_count):
    data=new_data[j]
    X_train=data.iloc[:,:-1]
    y_train=data.iloc[:, -1]
    
    regressor=linear_model.LinearRegression()
    regressor.fit(X_train,y_train)
    dump(regressor, './模型保存/PCA_LR'+str(j)+'.joblib')
    
    regressor=RandomForestRegressor(n_estimators=30, random_state=0,max_depth=3)
    regressor.fit(X_train,y_train)
    dump(regressor, './模型保存/PCA_RF'+str(j)+'.joblib')
    
    regressor=ske.GradientBoostingRegressor(n_estimators=30, random_state=0,max_depth=3)
    regressor.fit(X_train,y_train)
    dump(regressor, './模型保存/PCA_GBRT'+str(j)+'.joblib')
    
    regressor=skl.LassoCV(tol=1e-4, max_iter=10000)
    regressor.fit(X_train,y_train)
    dump(regressor, './模型保存/PCA_LASSO'+str(j)+'.joblib')
    print(j)


#训练GRG模型
l = 259200
dot_data = [None] * l 

for i in range(l):
    # 横向拼接每个DataFrame的第i行
    concatenated_row = pd.concat([df.iloc[i] for df in data_train], axis=1)
    # 直接存储每一行的数据
    dot_data[i] = concatenated_row.T  # 注意：这里存储的是DataFrame的一个副本



import warnings
from sklearn.exceptions import ConvergenceWarning

# 忽略 ConvergenceWarning 警告
warnings.filterwarnings('ignore', category=ConvergenceWarning)

rows = 20
cols = l

for i in range(0,259200):

    #GRG模型数量过大，需要分别储存在三个文件夹内
    if j <=172285:
        flag=''
    elif j<=218362:
        flag=1
    else:
        flag=2
    data=dot_data[i]
    X_train=data.iloc[:,:-1]
    y_train=data.iloc[:, -1]
    if y_train.sum()==0:
        continue
        
    regressor=linear_model.LinearRegression()
    regressor.fit(X_train,y_train)
    dump(regressor, './模型保存/GRG_LR'+str(flag)+'/GRG_LR'+str(i)+'.joblib')
    
    regressor=RandomForestRegressor(n_estimators=30, random_state=0,max_depth=3)
    regressor.fit(X_train,y_train)
    dump(regressor, './模型保存/GRG_RF'+str(flag)+'/GRG_RF'+str(i)+'.joblib')
    
    regressor=ske.GradientBoostingRegressor(n_estimators=30, random_state=0,max_depth=3)
    regressor.fit(X_train,y_train)
    dump(regressor, './模型保存/GRG_GBRT'+str(flag)+'/GRG_GBRT'+str(i)+'.joblib')
    
    regressor= skl.LassoCV(tol=1e-4, max_iter=10000)
    regressor.fit(X_train,y_train)
    dump(regressor, './模型保存/GRG_LASSO'+str(flag)+'/GRG_LASSO'+str(i)+'.joblib')
    print(i)



'''


