
from PIL import Image
import pandas as pd
import numpy as np
from scipy.ndimage import zoom

# 设置允许的最大像素数，防止因为图片过大而报错
Image.MAX_IMAGE_PIXELS = None

# 读取图像并转换为numpy数组
with Image.open('./ratio_rs_1km.tif') as img:
    img_array = np.array(img)

# 缩放图像到新的尺寸,转化成360*720的精度
new_shape = (360, 720)
scale_factors = (new_shape[0] / img_array.shape[0], new_shape[1] / img_array.shape[1])
resized_array = zoom(img_array, scale_factors, order=1)

# 调整顺序并保存为CSV文件
resized_df = pd.DataFrame(resized_array)
reordered_df = pd.concat([resized_df.iloc[:, 360:], resized_df.iloc[:, :360]], axis=1)
reordered_flipped_df = reordered_df.iloc[::-1].reset_index(drop=True)

# 保存为CSV文件
#reordered_flipped_df.to_csv('./acgR.csv', index=False)

'''
为防止GPT化简出错，以下是化简前的代码
#该代码用于读取地表R值

from PIL import Image
import pandas as pd
import numpy as np



#读取tif文件
Image.MAX_IMAGE_PIXELS = None
img = Image.open('./ratio_rs_1km.tif')


# 获取图像数据作为numpy数组
img_array = np.array(img)

# 将numpy数组转换为pandas DataFrame,
df = pd.DataFrame(img_array)

#df是一个17520*43200的表格，需要缩放成360*720
from scipy.ndimage import zoom
df_np = df.to_numpy()

# 设定新的尺寸
new_rows, new_cols = 360, 720

# 计算缩放因子
row_factor = new_rows / df_np.shape[0]
col_factor = new_cols / df_np.shape[1]

# 进行缩放
resized_np = zoom(df_np, (row_factor, col_factor), order=1)

# 转换回 DataFrame
df_resized = pd.DataFrame(resized_np)



#调整表格形式
first_half = df_resized.iloc[:, :360]  # 前 360 列
second_half = df_resized.iloc[:, 360:]  # 后 360 列

# 将后 360 列放在前面，前 360 列放在后面
new_df = pd.concat([second_half, first_half], axis=1)


df_reversed = new_df.iloc[::-1].reset_index(drop=True)


df_reversed.to_csv(f'C:/Users/86189/Desktop/acgR.csv',index=False)
#储存

'''


