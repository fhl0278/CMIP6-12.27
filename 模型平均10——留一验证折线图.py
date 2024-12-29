#本代码用于绘制所有模型留一验证预测结果折线图
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import r2_score

# 读取真实数据
df = pd.read_csv(f'./global_data/real_data1993.csv')
df[df > 0] = 1

# 定义模型名称
name = ['Global', 'Geoarea', 'PCA_', 'GRG_']

# 陆地格子数量
land_count = 52648

# 用于计算调整 R 方
N = land_count#52648
p = 22

def adj_r2(r2):
    return 1 - ((1 - r2) * (N - 1) / (N - p - 1))

# 用于区分陆地与海洋
land_mask = np.array(df).reshape(-1) == 1

# 固定图形大小
figsize = (10, 6)

# 绘制均方误差
for n in name:
    mse_values = {model: [] for model in ['Average', 'LASSO', 'LR', 'GBRT', 'RF']}
    for i in range(20):
        real_data = pd.read_csv(f'./global_data/real_data{1993+i}.csv')
        for model in mse_values.keys():
            if model == 'Average':
                pred_data = pd.read_csv(f'./预测数据/Average_{1993+i}.csv') * df
                #(360,720)->number
            else:
                pred_data = pd.read_csv(f'./预测数据/{n}{model}{1993+i}.csv') * df
            mse = ((pred_data - real_data) ** 2).sum().sum() / land_count
            mse_values[model].append(mse)
    
    time = np.arange(1993, 2013)
    plt.figure(figsize=figsize)
    plt.plot(time, mse_values['Average'], label='Average')
    plt.plot(time, mse_values['LASSO'], label=f'{n}LASSO', linestyle='-.')
    plt.plot(time, mse_values['LR'], label=f'{n}LR', linestyle=':')
    plt.plot(time, mse_values['GBRT'], label=f'{n}GBRT', linestyle='--')
    plt.plot(time, mse_values['RF'], label=f'{n}RF', linestyle=(0, (5, 3, 1, 3)))
    plt.ylim(0, 5)
    plt.legend(loc='upper left', bbox_to_anchor=(1, 1), borderaxespad=0.)
    plt.xticks(range(1993, 2013, 2))
    plt.title('cVeg: MSE')
    plt.tight_layout()
    plt.savefig(f'./留一验证折线图/均方误差/{n}.png', dpi=1000, bbox_inches='tight')
    plt.close()

# 绘制预测值
for n in name:
    pred_values = {model: [] for model in ['RF', 'LASSO', 'LR', 'GBRT', 'Average']}
    real_values = []
    for i in range(20):
        real_data = pd.read_csv(f'./global_data/real_data{1993+i}.csv').sum().sum() / land_count
        real_values.append(real_data)
        for model in pred_values.keys():
            if model == 'Average':
                pred_data = pd.read_csv(f'./预测数据/Average_{1993+i}.csv') * df
            else:
                pred_data = pd.read_csv(f'./预测数据/{n}{model}{1993+i}.csv') * df
            pred_value = pred_data.sum().sum() / land_count
            pred_values[model].append(pred_value)
    
    time = np.arange(1993, 2013)
    plt.figure(figsize=figsize)
    plt.plot(time, pred_values['RF'], label=f'{n}RF', linestyle=(0, (5, 3, 1, 3)))
    plt.plot(time, pred_values['LASSO'], label=f'{n}LASSO', linestyle='-.')
    plt.plot(time, pred_values['LR'], label=f'{n}LR', linestyle=':')
    plt.plot(time, pred_values['GBRT'], label=f'{n}GBRT', linestyle='--')
    plt.plot(time, real_values, label='Real')
    plt.ylim(2.5, 2.8)
    plt.legend(loc='upper left', bbox_to_anchor=(1, 1), borderaxespad=0.)
    plt.xticks(range(1993, 2013, 2))
    plt.title('cVeg: Predict')
    plt.tight_layout()
    plt.savefig(f'./留一验证折线图/预测值/{n}.png', dpi=1000, bbox_inches='tight')
    plt.close()

# 绘制调整 R 方
for n in name:
    r2_values = {model: [] for model in ['Average', 'LASSO', 'LR', 'GBRT', 'RF']}
    for i in range(20):
        real_data = pd.read_csv(f'./global_data/real_data{1993+i}.csv').values.reshape(-1)
        real_data = real_data[land_mask]
        for model in r2_values.keys():
            if model == 'Average':
                pred_data = pd.read_csv(f'./预测数据/Average_{1993+i}.csv') * df
            else:
                pred_data = pd.read_csv(f'./预测数据/{n}{model}{1993+i}.csv') * df
            pred_data = pred_data.values.reshape(-1)
            pred_data = pred_data[land_mask]#(1,52000)
            r2 = r2_score(real_data, pred_data)
            r2_values[model].append(adj_r2(r2))
    
    time = np.arange(1993, 2013)
    plt.figure(figsize=figsize)
    plt.plot(time, r2_values['Average'], label='Average')
    plt.plot(time, r2_values['LASSO'], label=f'{n}LASSO', linestyle='-.')
    plt.plot(time, r2_values['LR'], label=f'{n}LR', linestyle=':')
    plt.plot(time, r2_values['GBRT'], label=f'{n}GBRT', linestyle='--')
    plt.plot(time, r2_values['RF'], label=f'{n}RF', linestyle=(0, (5, 3, 1, 3)))
    plt.ylim(0.6, 1)
    plt.legend(loc='upper left', bbox_to_anchor=(1, 1), borderaxespad=0.)
    plt.xticks(range(1993, 2013, 2))
    plt.title('cVeg: Adjusted R2')
    plt.tight_layout()
    plt.savefig(f'./留一验证折线图/调整R方/{n}.png', dpi=1000, bbox_inches='tight')
    plt.close()

'''
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


df=pd.read_csv(f'./global_data/real_data1993.csv')
df[df>0]=1
name=['Global','Geoarea','PCA_','GPG']


#绘制均方误差
r0=[0]*20
r1=[0]*20
r2=[0]*20
r3=[0]*20
r4=[0]*20
y0=[0]*20
y1=[0]*20
y2=[0]*20
y3=[0]*20
y4=[0]*20
y5=[0]*20
y6=[0]*20
y7=[0]*20
y8=[0]*20
y=[0]*20
r5=[0]*20
r6=[0]*20
r7=[0]*20
r8=[0]*20

name=['Global','Geoarea','PCA_','GRG_']
#用于预测均方误差
for n in name:
    for i in range(20):
        y0[i]=pd.read_csv(f'./预测数据/Average_'+str(1993+i)+'.csv')*df
        y1[i]=pd.read_csv(f'./预测数据/'+n+'LASSO'+str(1993+i)+'.csv')*df
        y2[i]=pd.read_csv(f'./预测数据/'+n+'LR'+str(1993+i)+'.csv')*df
        y3[i]=pd.read_csv(f'C./预测数据/'+n+'GBRT'+str(1993+i)+'.csv')*df
        y4[i]=pd.read_csv(f'./预测数据/'+n+'RF'+str(1993+i)+'.csv')*df

        y[i]=pd.read_csv(f'./global_data/real_data'+str(1993+i)+'.csv')
        
        #52648是陆地格子数量
        r0[i]=((y0[i]-y[i])**2).sum().sum()/52648
        r1[i]=((y1[i]-y[i])**2).sum().sum()/52648
        r2[i]=((y2[i]-y[i])**2).sum().sum()/52648
        r3[i]=((y3[i]-y[i])**2).sum().sum()/52648
        r4[i]=((y4[i]-y[i])**2).sum().sum()/52648

    
    time = np.arange(1993, 2013)
    plt.plot(time,r0,label = 'Average')
    plt.plot(time,r1,label = n+'LASSO',linestyle='-.')
    plt.plot(time,r2,label = n+'LR',linestyle=':')
    plt.plot(time,r3,label = n+'GBRT',linestyle='--')
    plt.plot(time,r4,label = n+'RF',linestyle=(0, (5, 3, 1, 3)))
    plt.ylim(0, 5)
    plt.legend(bbox_to_anchor=(1, 1))
    plt.xticks(range(1993,2013,2))
    plt.title('cVeg: MES')
    plt.savefig('./留一验证折线图/均方误差/'+n+'.png', dpi=1000)

#用于绘制预测值
y0=[0]*20
y1=[0]*20
y2=[0]*20
y3=[0]*20
y4=[0]*20
y5=[0]*20
y=[0]*20
for n in name:
    for i in range(20):
        y0[i]=pd.read_csv(f'./预测数据/'+n+'RF'+str(1993+i)+'.csv').sum().sum()/52648
        y1[i]=pd.read_csv(f'./预测数据/'+n+'LASSO'+str(1993+i)+'.csv').sum().sum()/52648
        y2[i]=pd.read_csv(f'./预测数据/'+n+'LR'+str(1993+i)+'.csv').sum().sum()/52648
        y3[i]=pd.read_csv(f'./预测数据/'+n+'GBRT'+str(1993+i)+'.csv').sum().sum()/52648
        y4[i]=(pd.read_csv(f'./预测数据/Average_'+str(1993+i)+'.csv')*df).sum().sum()/52648
        y[i]=pd.read_csv(f'./global_data/real_data'+str(1993+i)+'.csv').sum().sum()/52648
    
    
    plt.plot(time,y0,label = n+'RF',linestyle=(0, (5, 3, 1, 3)))
    plt.plot(time,y1,label = n+'LASSO',linestyle='-.')
    plt.plot(time,y2,label = n+'LR',linestyle=':')
    plt.plot(time,y3,label = n+'GBRT',linestyle='--')
    #plt.plot(time,y4,label = 'Average')
    plt.plot(time,y,label = 'real')
    plt.ylim(2.5, 2.8)
    plt.legend(loc='upper left',bbox_to_anchor=(1, 1))
    plt.xticks(range(1993,2013,2))
    plt.title('cVeg: Predict ')
    plt.savefig('./留一验证折线图/预测值/'+n+'.png', dpi=1000)


#该代码用于绘制调整R方
N=52648
p=22
def adj(r):
    return 1 - ((1 - r) * (N - 1) / (N - p - 1))
from sklearn.metrics import r2_score

r0=[0]*20
r1=[0]*20
r2=[0]*20
r3=[0]*20
r4=[0]*20
y0=[0]*20
y1=[0]*20
y2=[0]*20
y3=[0]*20
y4=[0]*20
y5=[0]*20
y6=[0]*20
y7=[0]*20
y8=[0]*20
y9=[0]*20
y=[0]*20
r5=[0]*20
r6=[0]*20
r7=[0]*20
r8=[0]*20
r9=[0]*20

land=np.array(df).reshape(1,-1)[0]#用于区分陆地与海洋。
for n in name:
    for i in range(20):
        y0[i]=pd.read_csv(f'./预测数据/Average_'+str(1993+i)+'.csv')*df
        y0[i]=np.array(y0[i]).reshape(1,-1)[0]
        y1[i]=pd.read_csv(f'./预测数据/'+n+'LASSO'+str(1993+i)+'.csv')*df
        y1[i]=np.array(y1[i]).reshape(1,-1)[0]
        y2[i]=pd.read_csv(f'./预测数据/'+n+'LR'+str(1993+i)+'.csv')*df
        y2[i]=np.array(y2[i]).reshape(1,-1)[0]
        y3[i]=pd.read_csv(f'./预测数据/'+n+'GBRT'+str(1993+i)+'.csv')*df
        y3[i]=np.array(y3[i]).reshape(1,-1)[0]
        y4[i]=pd.read_csv(f'./预测数据/'+n+'RF'+str(1993+i)+'.csv')*df
        y4[i]=np.array(y4[i]).reshape(1,-1)[0]

        y[i]=pd.read_csv(f'./global_data/real_data'+str(1993+i)+'.csv')
        y[i]=np.array(y[i]).reshape(1,-1)[0]
    
        y0[i]=y0[i][y0[i]!=0]
        y1[i]=y1[i][land==1]
        y2[i]=y2[i][land==1]
        y3[i]=y3[i][land==1]
        y4[i]=y4[i][land==1]
        y5[i]=y5[i][land==1]
        y[i]=y[i][y[i]!=0]
        
        r0[i]=adj(r2_score(y0[i], y[i]))
        r1[i]=adj(r2_score(y1[i], y[i]))
        r2[i]=adj(r2_score(y2[i], y[i]))
        r3[i]=adj(r2_score(y3[i], y[i]))
        r4[i]=adj(r2_score(y4[i], y[i]))
        r5[i]=adj(r2_score(y5[i], y[i]))
    time = np.arange(1993, 2013)
    
    plt.plot(time,r0,label = 'Average')
    
    plt.plot(time,r1,label = n+'LASSO',linestyle='-.')
    plt.plot(time,r2,label = n+'LR',linestyle=':')
    plt.plot(time,r3,label = n+'GBRT',linestyle='--')
    plt.plot(time,r4,label = n+'RF',linestyle=(0, (5, 3, 1, 3)))
    plt.ylim(0.6, 1)
    plt.legend(loc='upper left',bbox_to_anchor=(1, 1))
    plt.xticks(range(1993,2013,2))
    plt.title('cVeg: Adjusted_R2 ')
    plt.savefig('./留一验证折线图/调整R方/'+n+'.png', dpi=1000)
'''