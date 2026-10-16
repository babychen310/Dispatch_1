#!/usr/bin/env python
# coding: utf-8

# In[1]:


#加载各类需要用到的程序包
import os
import pywt
import pandas as pd
import numpy as np
import datetime
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['SimHei'] # 用来正常显示中文标签SimHei
plt.rcParams['axes.unicode_minus'] = False # 用来正常显示负号
from pandas import DataFrame
from pandas import concat#相空间重构用到的包
import seaborn as sns
from scipy.stats import pearsonr
from datetime import datetime
from statsmodels.tsa.stattools import adfuller
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM
from tensorflow.keras.layers import Dense, Dropout
from PyEMD import CEEMDAN
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_percentage_error, mean_absolute_error


# In[2]:


#导入文件
data = pd.read_excel(r'D:\article\wind power_96MW.xlsx', sheet_name=None)
df = pd.concat(data.values(), ignore_index=True)
df


# In[3]:


import pandas as pd
# 定义一个函数来根据索引范围划分季节
def split_season(df, spring_start, spring_end, summer_start, summer_end, autumn_start, autumn_end, winter_start, winter_end):
    # 春季数据
    spring = df.iloc[spring_start:spring_end + 1]  # +1是因为Python的切片是左闭右开的
    # 夏季数据
    summer = df.iloc[summer_start:summer_end + 1]
    # 秋季数据
    autumn = df.iloc[autumn_start:autumn_end + 1]
    # 冬季数据
    winter = df.iloc[winter_start:winter_end + 1]
    return spring, summer, autumn, winter

# 使用定义好的函数划分季节
spring_df, summer_df, autumn_df, winter_df = split_season(df, 0, 2878, 2879, 5855, 5856, 8831, 8832, 11807)

# 打印每个季节的数据框的大小，以确认划分是否正确
print(f"Spring DataFrame size: {spring_df.shape}")
print(f"Summer DataFrame size: {summer_df.shape}")
print(f"Autumn DataFrame size: {autumn_df.shape}")
print(f"Winter DataFrame size: {winter_df.shape}")


# In[4]:


#平稳性检验
from statsmodels.tsa.stattools import adfuller
seasonal_dfs = {
    'spring': spring_df,
    'summer': summer_df,
    'autumn': autumn_df,
    'winter': winter_df
}

# 定义要检验的变量列表
variables = ['windspeed', 'wind_direction', 'atmosphere ','humidity ', 'power ']

# 函数用于执行ADF检验并打印结果
def perform_adf_test(df, variable):
    result = adfuller(df[variable])
    print(f"Variable: {variable}")
    print(f"  ADF Statistic: {result[0]}")
    print(f"  p-value: {result[1]}")
    print(f"  Used lags: {result[2]}")
    print(f"  Significance level: {result[3]}")
    print(f"  Critical Values: {result[4]}")
    print("  Conclusion: ", end="")
    if result[1] > 0.05:
        print("The time series is not stationary.")
    else:
        print("The time series is stationary.")
    print()  # 打印空行以分隔不同变量的结果

# 循环遍历每个季节的数据框和每个变量
for season, df in seasonal_dfs.items():
    print(f"Season: {season}")
    for var in variables:
        if var in df.columns:
            perform_adf_test(df, var)
        else:
            print(f"  Variable '{var}' does not exist in the DataFrame.")
    print()  # 打印空行以分隔不同季节的结果


# In[22]:


#互信息可视化
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.feature_selection import mutual_info_regression

# 定义一个函数来计算互信息并绘制条形图
def plot_mi_scores(df, season_name, color):
    # 分离目标变量和特征
    X = df[['windspeed', 'wind_direction', 'temperature', 'atmosphere ','humidity ']]
    y = df[ 'power ']
    
    # 计算每个特征与目标变量之间的互信息
    mi_scores = mutual_info_regression(X, y)
    
    # 将互信息分数转换为数据框，以便更好地查看和分析
    mi_df = pd.DataFrame(mi_scores, index=X.columns, columns=[f'MI_{season_name}'])
    
    # 可视化互信息分数
    plt.figure(figsize=(10, 6))  # 设置图形大小
    plt.bar(mi_df.index, mi_df[f'MI_{season_name}'], color=color)
    plt.xlabel('Features', fontsize=16)
    plt.ylabel('Mutual Information', fontsize=16)
    plt.title(f'Mutual Information Scores for {season_name} Season', fontsize=18)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)

    # ================== 只保存春天，300dpi ==================
    if season_name == "Spring":
        plt.savefig("风电场2互信息.png", dpi=500, bbox_inches='tight')

    plt.show()

ct="#FBECB4"
xt="#B5F4DA"
qt="#F69999"
dt="#B9D5FD"

# 为每个季节调用函数并指定不同的配色
seasons = ['Spring', 'Summer', 'Autumn', 'Winter']
colors = [ct, xt, qt, dt]

for season, color in zip(seasons, colors):
    plot_mi_scores(eval(f'{season.lower()}_df'), season, color)


# In[23]:


import seaborn as sns
import matplotlib.pyplot as plt

# 计算相关系数
spring_corr = spring_df.corr()
summer_corr = summer_df.corr()
autumn_corr = autumn_df.corr()
winter_corr = winter_df.corr()

# 春天 —— 显示 + 300dpi 高清保存
plt.figure(figsize=(10, 8))
sns.heatmap(spring_corr, annot=True, annot_kws={"size":15}, cmap='coolwarm', vmax=1, square=True, cbar_kws={"shrink": .5})
plt.xticks(fontsize=11)
plt.yticks(fontsize=11)
plt.title('Spring Correlation Heatmap', fontsize=16)
plt.savefig("风电场2相关系数.png", dpi=500, bbox_inches='tight')
plt.show()

# 夏天 —— 只显示
plt.figure(figsize=(10, 8))
sns.heatmap(summer_corr, annot=True, annot_kws={"size": 15}, cmap='coolwarm', vmax=1, square=True, cbar_kws={"shrink": .5})
plt.xticks(fontsize=11)
plt.yticks(fontsize=11)
plt.title('Summer Correlation Heatmap')
plt.show()

# 秋天 —— 只显示
plt.figure(figsize=(10, 8))
sns.heatmap(autumn_corr, annot=True, annot_kws={"size": 9}, cmap='coolwarm', vmax=1, square=True, cbar_kws={"shrink": .5})
plt.xticks(fontsize=11)
plt.yticks(fontsize=11)
plt.title('Autumn Correlation Heatmap')
plt.show()

# 冬天 —— 只显示
plt.figure(figsize=(10, 8))
sns.heatmap(winter_corr, annot=True, annot_kws={"size": 9}, cmap='coolwarm', vmax=1, square=True, cbar_kws={"shrink": .5})
plt.xticks(fontsize=11)
plt.yticks(fontsize=11)
plt.title('Winter Correlation Heatmap')
plt.show()


# In[7]:


#准备雷达图进行图形绘制
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

seasons = ['spring', 'summer', 'autumn', 'winter']
season_dfs = [spring_df, summer_df, autumn_df, winter_df]

for season, df in zip(seasons, season_dfs):
    features = df.iloc[:, :-1]  # 选择前五列作为特征
    target = df.iloc[:, -1]    # 选择最后一列作为目标变量

    # 划分训练集和测试集
    x_train, x_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)

    # 训练随机森林回归模型
    rf = RandomForestRegressor(n_estimators=100, random_state=42)
    rf.fit(x_train, y_train)

    # 获取特征重要性
    importances = rf.feature_importances_

    # 将特征重要性转换为DataFrame
    feature_importances = pd.DataFrame({
        'Feature': x_train.columns,
        'Importance': importances
    }).sort_values(by='Importance', ascending=False)

    # 输出每个季节的特征重要性
    print(f"Feature importances for {season}:")
    print(feature_importances)
    print("\n")


# In[21]:


# 特征重要性 + 春天青绿色折线图（带数据标签）
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

seasons = ['spring', 'summer', 'autumn', 'winter']
season_dfs = [spring_df, summer_df, autumn_df, winter_df]

all_importances = {}

for season, df in zip(seasons, season_dfs):
    features = df.iloc[:, :-1]
    target = df.iloc[:, -1]

    x_train, x_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)

    rf = RandomForestRegressor(n_estimators=100, random_state=42)
    rf.fit(x_train, y_train)

    importances = rf.feature_importances_

    feature_importances = pd.DataFrame({
        'Feature': x_train.columns,
        'Importance': importances
    }).sort_values(by='Importance', ascending=False)

    all_importances[season] = feature_importances

    print(f"Feature importances for {season}:")
    print(feature_importances)
    print("\n")

# ===================== 春天：青绿色折线图 + 数据标签 =====================
spring_data = all_importances['spring']
features = spring_data['Feature']
scores = spring_data['Importance']

plt.figure(figsize=(10, 5))

# 青绿色折线
plt.plot(features, scores, color='#F9D978', linewidth=3, marker='o', markersize=15)

# 数据标签（放大字体）
for f, s in zip(features, scores):
    plt.text(f, s + 0.01, f'{s:.3f}', ha='center', fontsize=14, fontweight='bold')

# 标题、坐标轴标签放大
plt.title('Wind Farm 2 - Feature Importance (Spring)', fontsize=20)
plt.xlabel('Features', fontsize=19)
plt.ylabel('Importance', fontsize=19)

# X/Y轴刻度字体放大
plt.xticks(rotation=20, fontsize=16)
plt.yticks(fontsize=16)

plt.grid(alpha=0.3)
plt.tight_layout()

# 保存 300dpi 高清图
plt.savefig("风电场2特征重要性.png", dpi=500, bbox_inches='tight')
plt.show()


# In[10]:


import matplotlib.pyplot as plt
import seaborn as sns

# 定义变量列表
variables = ['windspeed', 'wind_direction', 'atmosphere ','humidity ', 'power ']

# 为每一行定义不同的颜色
palette = 'Reds'  # 只使用一个颜色板，因为我们只展示春天的数据

# 创建一个1行5列的子图布局，因为我们只展示春天的数据
fig, axs = plt.subplots(1, 5, figsize=(20, 3))
axs = axs.flatten()

# 绘制箱型图
for j, variable in enumerate(variables):
    sns.boxplot(data=spring_df[[variable]], ax=axs[j], palette=palette)
    # 子标题放大
    axs[j].set_title(f'Wind farm 2 - {variable}', fontsize=18)
    # X/Y轴标签放大
    axs[j].set_ylabel('Value', fontsize=18)
    axs[j].set_xlabel('')
    # 坐标轴刻度字体放大
    axs[j].tick_params(axis='x', labelsize=20)
    axs[j].tick_params(axis='y', labelsize=18)

# 调整子图之间的间距
plt.tight_layout()

# 保存高清图片
plt.savefig("风电场2箱线图.png", dpi=500, bbox_inches='tight')

# 显示图片
plt.show()


# In[11]:


import matplotlib.pyplot as plt
import seaborn as sns
# 定义变量列表
variables = ['windspeed', 'wind_direction', 'atmosphere ','humidity ', 'power ']
# 为每一行定义不同的颜色
palettes = ['Blues', 'Greens', 'Reds', 'Purples']

# 创建一个4行6列的子图布局
fig, axs = plt.subplots(4, 5, figsize=(20, 12))  # 调整图形大小以适应4x6布局
axs = axs.flatten()  # 将2D子图数组转换为1D，以便迭代

# 定义季节列表
seasons = ['Spring', 'Summer', 'Autumn', 'Winter']

# 绘制箱型图
for i, season_df in enumerate([spring_df, summer_df, autumn_df, winter_df]):
    for j, variable in enumerate(variables):
        sns.boxplot(data=season_df[[variable]], ax=axs[i*5 + j], palette=palettes[i])
        axs[i*5 + j].set_title(f'{seasons[i]} - {variable}', fontsize=10)
        axs[i*5 + j].set_ylabel('Value', fontsize=8)
        axs[i*5 + j].set_xlabel('')

# 调整子图之间的间距
plt.tight_layout()
plt.show()


# In[16]:


import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

# 定义变量列表
variables =['windspeed', 'wind_direction', 'atmosphere ','humidity ', 'power ']
# 为每一行定义不同的颜色
palettes = ['Blues', 'Greens', 'Reds', 'Purples']

# 创建一个4行6列的子图布局
fig, axs = plt.subplots(4, 5, figsize=(20, 12))
axs = axs.flatten()

# 定义季节列表
seasons = ['Spring', 'Summer', 'Autumn', 'Winter']

# 函数：用均值替换异常值
def replace_outliers_with_mean(df, column):
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    # 计算均值
    mean_value = df[column].mean()
    # 替换异常值为均值
    df[column] = np.where((df[column] < lower_bound) | (df[column] > upper_bound), mean_value, df[column])
    return df

# 对每个季节的数据框进行异常值校正
corrected_spring_df = spring_df.copy()
corrected_summer_df = summer_df.copy()
corrected_autumn_df = autumn_df.copy()
corrected_winter_df = winter_df.copy()

for i, season_df in enumerate([corrected_spring_df, corrected_summer_df, corrected_autumn_df, corrected_winter_df]):
    for variable in variables:
        season_df = replace_outliers_with_mean(season_df, variable)

# 绘制箱型图
for i, season_df in enumerate([corrected_spring_df, corrected_summer_df, corrected_autumn_df, corrected_winter_df]):
    for j, variable in enumerate(variables):
        sns.boxplot(data=season_df[[variable]], ax=axs[i*5 + j], palette=palettes[i])
        axs[i*5 + j].set_title(f'{seasons[i]} - {variable}', fontsize=10)
        axs[i*5 + j].set_ylabel('Value', fontsize=8)
        axs[i*5 + j].set_xlabel('')

# 调整子图之间的间距
plt.tight_layout()
plt.show()

# 输出校正后的四季数据序列
print("Corrected Spring Data:")
print(corrected_spring_df.head())
print("\nCorrected Summer Data:")
print(corrected_summer_df.head())
print("\nCorrected Autumn Data:")
print(corrected_autumn_df.head())
print("\nCorrected Winter Data:")
print(corrected_winter_df.head())


# In[19]:


import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

# 定义变量列表
variables = ['windspeed', 'wind_direction', 'atmosphere ', 'humidity ', 'power ']
# 为春季定义颜色
palette = 'Blues'

# 创建一个1行5列的子图布局
fig, axs = plt.subplots(1, 5, figsize=(20, 3))  # 修改为1行5列
axs = axs.flatten()

# 定义季节列表
season = 'Spring'

# 函数：用最大最小值替换异常值
def replace_outliers_with_extremes(df, column):
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    # 计算最大值和最小值
    min_value = df[column].min()
    max_value = df[column].max()
    # 替换异常值为最大值或最小值
    df[column] = np.where(df[column] < lower_bound, min_value, df[column])
    df[column] = np.where(df[column] > upper_bound, max_value, df[column])
    return df

# 假设你已经有了春季的数据框：spring_df
# 对春季的数据框进行异常值校正
corrected_spring_df = spring_df.copy()
for variable in variables:
    corrected_spring_df = replace_outliers_with_extremes(corrected_spring_df, variable)

# 绘制箱型图
for j, variable in enumerate(variables):
    sns.boxplot(data=corrected_spring_df[[variable]], ax=axs[j], palette=palette)
    # 子标题字体放大
    axs[j].set_title(f'{season} - {variable}', fontsize=16)
    # Y轴标签放大
    axs[j].set_ylabel('Value', fontsize=14)
    axs[j].set_xlabel('')
    # X、Y轴刻度字体全部放大
    axs[j].tick_params(axis='x', labelsize=20)
    axs[j].tick_params(axis='y', labelsize=18)

# 调整子图之间的间距
plt.tight_layout()

# ================== 保存高清图片 ==================
plt.savefig("风电场2修正箱线图.png", dpi=500, bbox_inches='tight')

plt.show()

# 输出校正后的春季数据序列
print("Corrected Spring Data:")
print(corrected_spring_df.head())


# In[ ]:





# In[ ]:





# In[12]:


# 删除四季数据框中的温度列
spring_data = corrected_spring_df.drop( ['atmosphere ','humidity '], axis=1)
summer_data = corrected_summer_df.drop(['temperature',  'atmosphere ','humidity '], axis=1)
autumn_data = corrected_autumn_df.drop('temperature', axis=1)
winter_data = corrected_winter_df.drop('temperature', axis=1)
spring_data


# In[13]:


import pandas as pd
from sklearn.model_selection import train_test_split
# 函数：划分数据集
def split_data(df, target_column):
    # 分离特征和目标变量
    X = df.drop(target_column, axis=1)
    y = df[target_column]
    
    # 划分训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42,shuffle=False)
    
    # 从训练集中进一步划分出验证集
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.25, random_state=42,shuffle=False)
    
    return X_train, X_val, X_test, y_train, y_val, y_test

# 假设 'target' 是目标变量的列名
target_column = 'power '

# 对每个季节的数据进行划分
spring_X_train, spring_X_val, spring_X_test, spring_y_train, spring_y_val, spring_y_test = split_data(spring_data, target_column)
summer_X_train, summer_X_val, summer_X_test, summer_y_train, summer_y_val, summer_y_test = split_data(summer_data, target_column)
autumn_X_train, autumn_X_val, autumn_X_test, autumn_y_train, autumn_y_val, autumn_y_test = split_data(autumn_data, target_column)
winter_X_train, winter_X_val, winter_X_test, winter_y_train, winter_y_val, winter_y_test = split_data(winter_data, target_column)
print(f"Spring DataFrame size: {spring_X_train.shape}")
print(f"Summer DataFrame size: {summer_X_train.shape}")
print(f"Autumn DataFrame size: {autumn_X_train.shape}")
print(f"Winter DataFrame size: {winter_X_train.shape}")
print(spring_X_train,spring_y_train)


# In[14]:


from sklearn.preprocessing import MinMaxScaler

# 对每个季节的X特征进行归一化，为每个季节创建独立的MinMaxScaler实例
scaler_spring_X = MinMaxScaler()
scaler_summer_X = MinMaxScaler()
scaler_autumn_X = MinMaxScaler()
scaler_winter_X = MinMaxScaler()

# 对每个季节的y目标变量进行归一化，为每个季节创建独立的MinMaxScaler实例
scaler_spring_y = MinMaxScaler()
scaler_summer_y = MinMaxScaler()
scaler_autumn_y = MinMaxScaler()
scaler_winter_y = MinMaxScaler()

# 对每个季节的训练集特征进行归一化
spring_X_train_scaled = scaler_spring_X.fit_transform(spring_X_train)
summer_X_train_scaled = scaler_summer_X.fit_transform(summer_X_train)
autumn_X_train_scaled = scaler_autumn_X.fit_transform(autumn_X_train)
winter_X_train_scaled = scaler_winter_X.fit_transform(winter_X_train)

# 对训练集目标变量进行归一化
spring_y_train_scaled = scaler_spring_y.fit_transform(spring_y_train.values.reshape(-1, 1))
summer_y_train_scaled = scaler_summer_y.fit_transform(summer_y_train.values.reshape(-1, 1))
autumn_y_train_scaled = scaler_autumn_y.fit_transform(autumn_y_train.values.reshape(-1, 1))
winter_y_train_scaled = scaler_winter_y.fit_transform(winter_y_train.values.reshape(-1, 1))

# 对验证集和测试集特征进行归一化，使用训练集的参数
spring_X_val_scaled = scaler_spring_X.transform(spring_X_val)
spring_X_test_scaled = scaler_spring_X.transform(spring_X_test)
summer_X_val_scaled = scaler_summer_X.transform(summer_X_val)
summer_X_test_scaled = scaler_summer_X.transform(summer_X_test)
autumn_X_val_scaled = scaler_autumn_X.transform(autumn_X_val)
autumn_X_test_scaled = scaler_autumn_X.transform(autumn_X_test)
winter_X_val_scaled = scaler_winter_X.transform(winter_X_val)
winter_X_test_scaled = scaler_winter_X.transform(winter_X_test)

# 对验证集和测试集目标变量进行归一化，使用训练集的参数
spring_y_val_scaled = scaler_spring_y.transform(spring_y_val.values.reshape(-1, 1))
spring_y_test_scaled = scaler_spring_y.transform(spring_y_test.values.reshape(-1, 1))
summer_y_val_scaled = scaler_summer_y.transform(summer_y_val.values.reshape(-1, 1))
summer_y_test_scaled = scaler_summer_y.transform(summer_y_test.values.reshape(-1, 1))
autumn_y_val_scaled = scaler_autumn_y.transform(autumn_y_val.values.reshape(-1, 1))
autumn_y_test_scaled = scaler_autumn_y.transform(autumn_y_test.values.reshape(-1, 1))
winter_y_val_scaled = scaler_winter_y.transform(winter_y_val.values.reshape(-1, 1))
winter_y_test_scaled = scaler_winter_y.transform(winter_y_test.values.reshape(-1, 1))
spring_X_train_scaled


# In[15]:


#相空间重构
def series_to_supervised(data, n_in=1, n_out=1, dropnan=True):
    n_vars = 1 if type(data) is list else data.shape[1]
    df = DataFrame(data)
    cols, names = list(), list()# input sequence (t-n, ... t-1)
    for i in range(n_in, 0, -1):
        cols.append(df.shift(i))
        names += [('var%d(t-%d)' % (j+1, i)) for j in range(n_vars)]# forecast sequence (t, t+1, ... t+n)
    for i in range(0, n_out):
        cols.append(df.shift(-i))
        if i == 0:
            names += [('var%d(t)' % (j+1)) for j in range(n_vars)]
        else:
            names += [('var%d(t+%d)' % (j+1, i)) for j in range(n_vars)]# put it all together
    agg = concat(cols, axis=1)
    agg.columns = names# drop rows with NaN values
    if dropnan:
        agg.dropna(inplace=True)
    return agg


# In[16]:


import numpy as np
import pandas as pd
seasonal_data = {
    'spring': {
        'train': spring_X_train_scaled,
        'val': spring_X_val_scaled,
        'test': spring_X_test_scaled
    },
    'summer': {
        'train': summer_X_train_scaled,
        'val': summer_X_val_scaled,
        'test': summer_X_test_scaled
    },
    'autumn': {
        'train': autumn_X_train_scaled,
        'val': autumn_X_val_scaled,
        'test': autumn_X_test_scaled
    },
    'winter': {
        'train': winter_X_train_scaled,
        'val': winter_X_val_scaled,
        'test': winter_X_test_scaled
    }
}

# 定义每个季节的输入时间步长（n_in）和输出时间步长（n_out）
seasonal_n_steps = {
    'spring': (19, 1),
    'summer': (19, 1),
    'autumn': (28, 1),
    'winter': (24, 1)
}

# 初始化一个字典来存储所有重构数据
all_reconstructed_data = {}

# 遍历每个季节
for season, datasets in seasonal_data.items():
    # 初始化一个字典来存储当前季节的重构数据
    all_reconstructed_data[season] = {}
    
    # 获取当前季节的n_in和n_out
    n_in, n_out = seasonal_n_steps[season]
    
    # 遍历每个数据集（训练集、验证集、测试集）
    for dataset_name, data in datasets.items():
        # 初始化一个字典来存储当前数据集的重构数据
        all_reconstructed_data[season][dataset_name] = {}
        
        # 遍历当前数据集的每一列
        for i in range(data.shape[1]):
            # 提取第 i 列的数据
            column_data = data[:, i].reshape(-1, 1)  # 重塑为二维数组
            
            # 使用 series_to_supervised 函数对每一列进行处理
            reconstructed_column = series_to_supervised(column_data, n_in=n_in, n_out=n_out)
            
            # 将重构后的数据存储在字典中
            all_reconstructed_data[season][dataset_name][f'feature_{i}'] = reconstructed_column

# 现在 all_reconstructed_data 字典中包含了所有数据集的重构数据
# 例如，访问春季训练集的第一列重构后的数据：
spring_train_feature_0_shape = all_reconstructed_data['spring']['train']['feature_0'].shape
print(spring_train_feature_0_shape)


# In[19]:


#提炼出x_spring_train
X_st1=all_reconstructed_data['spring']['train']['feature_0']
X_st2=all_reconstructed_data['spring']['train']['feature_1']
X_st3=all_reconstructed_data['spring']['train']['feature_2']
y_st=series_to_supervised(spring_y_train_scaled,19)
y_st1=y_st.iloc[:, :18]
y_spring_train=y_st.iloc[:, -1:]
x_spring_train = pd.concat([X_st1, X_st2, X_st3,y_st1], axis=1)

#提炼出x_spring_val
X_sv1=all_reconstructed_data['spring']['val']['feature_0']
X_sv2=all_reconstructed_data['spring']['val']['feature_1']
X_sv3=all_reconstructed_data['spring']['val']['feature_2']
y_sv=series_to_supervised(spring_y_val_scaled,19)
y_sv1=y_sv.iloc[:, :18]
x_spring_val = pd.concat([X_sv1, X_sv2, X_sv3,y_sv1], axis=1)
y_spring_val=y_sv.iloc[:, -1:]

#提炼出x_spring_test
X_se1=all_reconstructed_data['spring']['test']['feature_0']
X_se2=all_reconstructed_data['spring']['test']['feature_1']
X_se3=all_reconstructed_data['spring']['test']['feature_2']
y_se=series_to_supervised(spring_y_test_scaled,19)
y_se1=y_se.iloc[:, :18]
x_spring_test = pd.concat([X_se1, X_se2, X_se3,y_se1], axis=1)
y_spring_test=y_se.iloc[:, -1:]
x_spring_train


# In[ ]:





# In[ ]:





# In[23]:


import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import layers, Model, Input
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error, mean_squared_error, r2_score
from mealpy.swarm_based.PSO import OriginalPSO

import warnings
warnings.filterwarnings("ignore")
# ==============================================

# ===================== 你的数据（完全不变） =====================
train_X1 = x_spring_train.values.reshape((-1, 78, 1))
test_X1  = x_spring_test.values.reshape((-1, 78, 1))
val_X1   = x_spring_val.values.reshape((-1, 78, 1))

y_train = y_spring_train.values.ravel()
y_val   = y_spring_val.values.ravel()
y_test  = y_spring_test.values.ravel()

print("train_X1.shape:", train_X1.shape)
print("test_X1.shape:", test_X1.shape)

# ===================== iTransformer 模型（带PSO可调参数） =====================
class TransposeLayer(layers.Layer):
    def call(self, x):
        return tf.transpose(x, perm=[0, 2, 1])

def iTransformer_model(d_model, learning_rate):
    inputs = Input(shape=(78, 1))
    
    # 核心改进
    x = TransposeLayer()(inputs)
    x = layers.Dense(int(d_model))(x)
    
    # 自注意力
    x = layers.MultiHeadAttention(num_heads=2, key_dim=int(d_model)//2)(x, x)
    x = layers.LayerNormalization(epsilon=1e-6)(x)
    x = layers.Dropout(0.1)(x)
    
    # 前馈
    x = layers.Dense(int(d_model)*2, activation="gelu")(x)
    x = layers.Dense(int(d_model))(x)
    
    # 输出
    x = layers.GlobalAveragePooling1D()(x)
    outputs = layers.Dense(1)(x)
    
    model = Model(inputs, outputs)
    model.compile(optimizer=Adam(learning_rate=learning_rate), loss="mse")
    return model

# ===================== 完全照搬你 LSTM 的 PSO 格式 =====================
def cross_val(nest):
    d_model, epochs, learning_rate = nest
    model = iTransformer_model(d_model, learning_rate)
    model.fit(val_X1, y_val, epochs=int(epochs), batch_size=32, verbose=0, shuffle=False)
    y_pre = model.predict(val_X1, verbose=0)
    MSE = mean_squared_error(y_val, y_pre)
    return MSE

def fitness_function(nest):
    MSE = cross_val(nest)
    return MSE

def get_nest():
    problem_dict1 = {
         "fit_func": fitness_function,
         "lb":[16, 10, 1e-04],    # 维度、轮次、学习率 下限
         "ub":[48, 20, 1e-02],    # 维度、轮次、学习率 上限
         "minmax": "min",
         "verbose": True,
    } 
    epoch = 2
    pop_size = 16
    c1 = 2.05
    c2 = 2.05
    w_min = 0.4
    w_max = 0.9
    model = OriginalPSO(epoch, pop_size, c1, c2, w_min, w_max)
    best_position, best_fitness = model.solve(problem_dict1)
    print(f"最优参数: {best_position}, 最优MSE: {best_fitness}")
    return [best_fitness, best_position]

# ===================== 执行PSO寻优 =====================
nest = get_nest()
d_model = nest[1][0]
epochs  = nest[1][1]
lr      = nest[1][2]

# ===================== 训练最终模型 =====================
final_model = iTransformer_model(d_model, lr)
final_model.fit(train_X1, y_train, epochs=int(epochs), batch_size=16, verbose=2, shuffle=False)

# ===================== 预测 & 指标 =====================
y_pre_itrans = final_model.predict(test_X1)

r2 = r2_score(y_test, y_pre_itrans)
mse = mean_squared_error(y_test, y_pre_itrans)
mae = mean_absolute_error(y_test, y_pre_itrans)
mape = mean_absolute_percentage_error(y_test, y_pre_itrans)
rmse = np.sqrt(mse)

print("\n======= PSO-iTransformer (2026 SOTA) 结果 =======")
print(f"R²    = {r2:.4f}")
print(f"MSE   = {mse:.4f}")
print(f"RMSE  = {rmse:.4f}")
print(f"MAE   = {mae:.4f}")
print(f"MAPE  = {mape:.4f}")

# ===================== 绘图 =====================
plt.figure(figsize=(15,6))
plt.plot(y_test, label='Real', linewidth=2)
plt.plot(y_pre_itrans, label='PSO-iTransformer', linewidth=2)
plt.legend()
plt.title('PSO-iTransformer Prediction Results')
plt.show()


# In[24]:


import pandas as pd
import numpy as np
import os

# ===================== 【自动创建文件夹】 =====================
directory = 'D:\\article'
if not os.path.exists(directory):
    os.makedirs(directory)

# ===================== 计算最大值、最小值（和你原来完全一样） =====================
scaler_max = spring_y_test.values.max()
scaler_min = spring_y_test.values.min()
print("归一化的最大值:", scaler_max)
print("归一化的最小值:", scaler_min)

# ===================== 反归一化函数 =====================
def inverse_transform(y_pred_scaled, y_actual_scaled, scaler_min, scaler_max):
    y_pred = y_pred_scaled * (scaler_max - scaler_min) + scaler_min
    y_actual = y_actual_scaled * (scaler_max - scaler_min) + scaler_min
    return y_pred, y_actual

# ===================== 【替换成 iTransformer 预测变量】 =====================
# y_pre_itrans 是你iTransformer模型的预测值
y_pre_itrans_inv, y_spring_test_inv = inverse_transform(
    y_pre_itrans,          # iTransformer 预测值（替换原来的 y_pre_transformer）
    y_spring_test.values,
    scaler_min,
    scaler_max
)

# ===================== 计算误差 =====================
spring_itrans_error = y_spring_test_inv - y_pre_itrans_inv
print("反归一化后的预测误差计算完成！")

# ===================== 【保存所有结果到 D:\article 文件夹】 =====================
# 保存为 DataFrame
result_df = pd.DataFrame({
    '实际值_反归一化': y_spring_test_inv.ravel(),
    'iTransformer预测值_反归一化': y_pre_itrans_inv.ravel(),
    '预测误差': spring_itrans_error.ravel()
})

# 保存路径
save_path = os.path.join(directory, 'iTransformer_预测结果.xlsx')

# 保存为 Excel 文件
result_df.to_excel(save_path, index=False)

print(f"\n所有数据已保存至：{save_path}")
print("✅ 保存内容：实际值、iTransformer预测值、预测误差")


# In[28]:


from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error, mean_squared_error
from sklearn.metrics import r2_score
import numpy as np
import matplotlib.pyplot as plt
from mealpy.swarm_based.PSO import OriginalPSO
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import backend as K
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Layer, Input, Dense, Dropout, LayerNormalization, MultiHeadAttention, GlobalAveragePooling1D
import tensorflow as tf

# ===================== 数据（完全不变）=====================
train_X1 = x_spring_train.values.reshape((-1, 78, 1))
test_X1  = x_spring_test.values.reshape((-1, 78, 1))
val_X1   = x_spring_val.values.reshape((-1, 78, 1))

y_spring_train_flat = y_spring_train.values.ravel()
y_spring_val_flat   = y_spring_val.values.ravel()
y_spring_test_flat  = y_spring_test.values.ravel()

print("train_X1.shape:", train_X1.shape)

# ===================== 分位数损失函数（和你模板一样）=====================
def quantile_loss(y_true, y_pred, quantile):
    return K.mean(K.maximum(quantile * (y_true - y_pred), (quantile - 1) * (y_true - y_pred)), axis=-1)

# ===================== iTransformer 核心：转置层 =====================
class TransposeLayer(Layer):
    def call(self, x):
        return tf.transpose(x, perm=[0, 2, 1])

# ===================== iTransformer 模型（区间预测专用）=====================
def iTransformer_model(d_model, learning_rate, quantile):
    inputs = Input(shape=(78, 1))
    
    # iTransformer 核心结构
    x = TransposeLayer()(inputs)
    x = Dense(int(d_model))(x)
    
    x = MultiHeadAttention(num_heads=2, key_dim=int(d_model)//2)(x, x)
    x = LayerNormalization(epsilon=1e-6)(x)
    x = Dropout(0.2)(x)
    
    x = Dense(int(d_model)*2, activation="gelu")(x)
    x = Dense(int(d_model))(x)
    
    x = GlobalAveragePooling1D()(x)
    outputs = Dense(1)(x)

    model = Model(inputs, outputs)
    model.compile(optimizer=Adam(learning_rate=learning_rate), 
                  loss=lambda y_true, y_pred: quantile_loss(y_true, y_pred, quantile))
    return model

# ===================== PSO 交叉验证（完全沿用你的模板）=====================
def cross_val(nest):
    d_model, epochs, learning_rate = nest
    model = iTransformer_model(d_model, learning_rate, quantile=0.5)
    model.fit(train_X1, y_spring_train_flat, epochs=int(epochs), batch_size=32, verbose=0, shuffle=False)
    y_pre = model.predict(val_X1, verbose=0)
    return mean_squared_error(y_spring_val_flat, y_pre)

def fitness_function(nest):
    return cross_val(nest)

# ===================== PSO 寻优 =====================
def get_nest():
    problem_dict1 = {
         "fit_func": fitness_function,
         "lb":[16, 10, 1e-04],
         "ub":[48, 30, 1e-02],
         "minmax": "min",
         "verbose": True,
    }
    epoch = 2
    pop_size = 16
    c1 = 2.05
    c2 = 2.05
    w_min = 0.4
    w_max = 0.9
    model = OriginalPSO(epoch, pop_size, c1, c2, w_min, w_max)
    best_position, best_fitness = model.solve(problem_dict1)
    print(f"最优参数: {best_position}, MSE: {best_fitness}")
    return [best_fitness, best_position]

# ===================== 执行PSO =====================
nest = get_nest()
d_model     = nest[1][0]
epochs      = nest[1][1]
lr          = nest[1][2]

# ===================== 95% 区间预测：0.025 / 0.975 =====================
quantiles = [0.025, 0.975]
predictions_itrans = {}

for q in quantiles:
    model = iTransformer_model(d_model, lr, q)
    model.fit(train_X1, y_spring_train_flat, epochs=int(epochs), batch_size=16, verbose=2, shuffle=False)
    pred = model.predict(test_X1)
    predictions_itrans[q] = pred

# ===================== 指标输出（和你模板完全一样）=====================
R2, mse, mae, mape, rmse = [], [], [], [], []

for q, y_pre in predictions_itrans.items():
    R2_q    = r2_score(y_spring_test_flat, y_pre)
    mse_q   = mean_squared_error(y_spring_test_flat, y_pre)
    mae_q   = mean_absolute_error(y_spring_test_flat, y_pre)
    mape_q  = mean_absolute_percentage_error(y_spring_test_flat, y_pre)
    rmse_q  = np.sqrt(mse_q)

    R2.append(R2_q)
    mse.append(mse_q)
    mae.append(mae_q)
    mape.append(mape_q)
    rmse.append(rmse_q)

    print(f"\n=== Quantile {q} ===")
    print(f"R²:    {R2_q:.4f}")
    print(f"MSE:   {mse_q:.4f}")
    print(f"MAE:   {mae_q:.4f}")
    print(f"MAPE:  {mape_q:.4f}")
    print(f"RMSE:  {rmse_q:.4f}")

# ===================== 绘图：95% 预测区间 =====================
plt.figure(figsize=(15,6))
plt.plot(predictions_itrans[0.025], label='0.025 Quantile', linewidth=1.5)
plt.plot(predictions_itrans[0.975], label='0.975 Quantile', linewidth=1.5)
plt.plot(y_spring_test_flat, 'k--', label='Actual', linewidth=2)
plt.title('PSO-iTransformer 95% Prediction Interval')
plt.legend()
plt.show()


# In[29]:


# ===================== 计算 iTransformer 区间预测指标（95%）=====================
picp_95 = np.mean((y_spring_test.values >= predictions_itrans[0.025]) & (y_spring_test.values <= predictions_itrans[0.975]))
piaw_95 = np.mean(predictions_itrans[0.975] - predictions_itrans[0.025])
ace_95 = np.abs(0.95 - picp_95)
ais95 = np.mean((np.abs(y_spring_test.values - y_pre_itrans) / (predictions_itrans[0.975] - predictions_itrans[0.025])) ** 2)

itransformer_lb = predictions_itrans[0.025]  # 下限值
itransformer_ub = predictions_itrans[0.975]  # 上限值

print("PICP_95:", picp_95)
print("PIAW_95:", piaw_95)
print("ACE_95:", ace_95)
print("AIS_95:", ais95)


# In[30]:


import pandas as pd
import numpy as np
import os

# 创建一个DataFrame来保存iTransformer区间数据
excel_data = {
    "iTransformer_Lower_Bound": itransformer_lb.flatten(),
    "iTransformer_Upper_Bound": itransformer_ub.flatten()
}

# 将数据转换为DataFrame
excel_df = pd.DataFrame(excel_data)

# 获取spring_y_test的最大值和最小值（和原代码完全一致）
scaler_max = spring_y_test.values.max()
scaler_min = spring_y_test.values.min()
print("归一化的最大值:", scaler_max)
print("归一化的最小值:", scaler_min)

# 反归一化函数（和原代码完全一致，无改动）
def inverse_normalize(value, scaler_min, scaler_max):
    return value * (scaler_max - scaler_min) + scaler_min

# 应用反归一化，仅替换模型名称
models = ["iTransformer"]
for model in models:
    model_lower_key = model + "_Lower_Bound"
    model_upper_key = model + "_Upper_Bound"
    excel_df[model_lower_key] = excel_df[model_lower_key].apply(lambda x: inverse_normalize(x, scaler_min, scaler_max))
    excel_df[model_upper_key] = excel_df[model_upper_key].apply(lambda x: inverse_normalize(x, scaler_min, scaler_max))

# 保存到Excel文件，路径不变，仅修改文件名对应iTransformer
directory = 'D:\\article'
file_path = os.path.join(directory, 'iTransformer_predictions_quantiles.xlsx')
excel_df.to_excel(file_path, index=False)

# 显示DataFrame
excel_df


# In[ ]:





# In[ ]:





# In[30]:


##成功的PSO-LSTM参数调优
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error, mean_squared_error
from sklearn.metrics import r2_score
import numpy as np
from mealpy.swarm_based.PSO import OriginalPSO
from tensorflow.keras.optimizers import Adam
train_X1 = x_spring_train.values.reshape((x_spring_train.shape[0],78,1))
test_X1= x_spring_test.values.reshape((x_spring_test.shape[0],78,1))
print("train_X1.shape ",train_X1.shape)
print("test_X1.shape ",test_X1.shape)
def LSTM_model(units, learning_rate, train_X):
    # 使用Keras建构LSTM模型
    model = Sequential()
    model.add(LSTM(units=int(units), input_shape=(train_X1.shape[1], train_X1.shape[2]), return_sequences=True))
    model.add(Dropout(0.4))
    model.add(LSTM(units=int(units*3)))
    model.add(Dense(1))#输出几步数字为几
    optimizer = Adam(learning_rate=learning_rate)
    model.compile(optimizer=optimizer, loss='mse')
    model.summary()
    return model

def cross_val(nest):
    units,epochs,learning_rate = nest
    model=LSTM_model(units, learning_rate, x_spring_val)
    history = model.fit(x_spring_val, y_spring_val.values.ravel(), epochs=int(epochs), batch_size=64, verbose=2, shuffle=False)
    # 执行预测
    y_pre = model.predict(x_spring_val)
    # 计算MSE误差值
    MSE= mean_squared_error(y_spring_val.values.ravel(), y_pre)
    return MSE

#定义适应度函数
def fitness_function(nest):
    MSE = cross_val(nest)
    return MSE

def get_nest():
    # 适应度函数选择
    # units,epochs,learning_rate
    problem_dict1 = {
         "fit_func": fitness_function,
         "lb":[20,20,1e-05],
         "ub":[50,50,1e-03],
         "minmax": "min",
         "verbose": True,
    } 
    epoch = 2
    pop_size = 16
#大佬PSO优化函数已经封装好的，不用改动    
    c1 = 2.05
    c2 = 2.05
    w_min = 0.4
    w_max = 0.9
    model = OriginalPSO(epoch, pop_size, c1, c2, w_min, w_max)
    best_position, best_fitness = model.solve(problem_dict1)
    print(f"Solution: {best_position}, Fitness: {best_fitness}")
    return [best_fitness, best_position]#返回最优参数
#将获得的参数放入nest
nest=get_nest()
print(nest)
#取nest中的数赋给需要优化的参数
# units,epochs,learning_rate
units= nest[1][0]
epochs= nest[1][1]
learning_rate =nest[1][2]
model=LSTM_model(units, learning_rate, x_spring_train)
history = model.fit(x_spring_train, y_spring_train.values.ravel(), epochs=int(epochs), batch_size=10, verbose=2, shuffle=False)
# 执行预测
y_pre_lstm = model.predict(x_spring_test)
# 计算r方
R2_lstm = r2_score(y_spring_test.values.ravel(), y_pre_lstm)
print("计算得出的r方为：", R2_lstm)
mse_lstm= mean_squared_error(y_spring_test.values.ravel(), y_pre_lstm)
mae_lstm= mean_absolute_error(y_spring_test.values.ravel(), y_pre_lstm)
mape_lstm= mean_absolute_percentage_error(y_spring_test.values.ravel(), y_pre_lstm)
# 计算平方差
squared_diff_lstm = np.square(y_spring_test.values.ravel()- y_pre_lstm)
#计算均值
mean_squared_error_lstm = np.mean(squared_diff_lstm)
#计算RMSE
rmse_lstm= np.sqrt(mean_squared_error_lstm)
print("MSE： ", mse_lstm)
print("MAE:", mae_lstm)
print("MAPE:", mape_lstm)
print("RMSE:", rmse_lstm)
#绘制PSO-LSTM预测之后的模型
plt.figure(figsize=(15,6))
plt.plot(y_spring_test.values.ravel())
plt.plot(y_pre_lstm)
plt.show()


# In[31]:


import pandas as pd

# 计算y_spring_test的最大值和最小值
scaler_max = spring_y_test.values.max()  # 如果y_spring_test是多维的，使用max().max()获取全局最大值
scaler_min = spring_y_test.values.min()  # 如果y_spring_test是多维的，使用min().min()获取全局最小值
print("归一化的最大值:", scaler_max)
print("归一化的最小值:", scaler_min)

# 反归一化函数
def inverse_transform(y_pred_scaled, y_actual_scaled, scaler_min, scaler_max):
    y_pred = y_pred_scaled * (scaler_max - scaler_min) + scaler_min
    y_actual = y_actual_scaled * (scaler_max - scaler_min) + scaler_min
    return y_pred, y_actual

# 确保 y_pre_lstm 和 y_spring_test.values.ravel() 是一维数组
# 反归一化预测值和实际值
y_pre_lstm_inv, y_spring_test_inv = inverse_transform(y_pre_lstm, y_spring_test.values, scaler_min, scaler_max)
# 计算预测误差
spring_lstm_error = y_spring_test_inv - y_pre_lstm_inv
spring_lstm_error


# In[ ]:





# In[27]:


##LSTM区间预测（95%）
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error, mean_squared_error
from sklearn.metrics import r2_score
import numpy as np
from mealpy.swarm_based.PSO import OriginalPSO
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import backend as K
train_X1 = x_spring_train.values.reshape((x_spring_train.shape[0],78,1))
test_X1= x_spring_test.values.reshape((x_spring_test.shape[0],78,1))
val_X1= x_spring_val.values.reshape((x_spring_val.shape[0],78,1))
print("train_X1.shape ",train_X1.shape)
print("test_X1.shape ",test_X1.shape)

# 分位数损失函数
def quantile_loss(y_true, y_pred, quantile):
    return K.mean(K.maximum(quantile * (y_true - y_pred), (quantile - 1) * (y_true - y_pred)), axis=-1)

# LSTM模型
def LSTM_model(units, learning_rate, train_X, quantile):
    # 使用Keras建构LSTM模型
    model = Sequential()
    model.add(LSTM(units=int(units), input_shape=(train_X1.shape[1], train_X1.shape[2]), return_sequences=True))
    model.add(Dropout(0.5))
    model.add(LSTM(units=int(units*2)))
    model.add(Dense(1))  # 输出层
    optimizer = Adam(learning_rate=learning_rate)
    model.compile(optimizer=optimizer, loss=lambda y_true, y_pred: quantile_loss(y_true, y_pred, quantile))
    model.summary()
    return model

# 适应度函数
def fitness_function(nest):
    units, epochs, learning_rate, quantile = nest
    MSE = cross_val([units, epochs, learning_rate, quantile])
    return MSE

# 交叉验证
def cross_val(nest):
    units, epochs, learning_rate, quantile = nest
    model = LSTM_model(units, learning_rate, train_X1, quantile)
    history = model.fit(train_X1, y_spring_train.values.ravel(), epochs=int(epochs), batch_size=64, verbose=2, shuffle=False)
    y_pre_qlstm = model.predict(val_X1)
    MSE = mean_squared_error(y_spring_val.values.ravel(), y_pre_qlstm)
    return MSE

# 获取最佳参数
def get_nest(): 
    problem_dict1 = {
        "fit_func": fitness_function,
        "lb": [20, 20, 1e-05, 0.025],  # 参数的下限
        "ub": [60, 60, 1e-02, 0.975],  # 参数的上限
        "minmax": "min",
        "verbose": True,
    }
    epoch = 2
    pop_size = 16
    c1 = 2.05
    c2 = 2.05
    w_min = 0.4
    w_max = 0.9
    model = OriginalPSO(epoch, pop_size, c1, c2, w_min, w_max)
    best_position, best_fitness = model.solve(problem_dict1)
    print(f"Solution: {best_position}, Fitness: {best_fitness}")
    return [best_fitness, best_position]

# 获取最佳参数
nest = get_nest()
print(nest)

# 取nest中的数赋给需要优化的参数
units = nest[1][0]
epochs = nest[1][1]
learning_rate = nest[1][2]

# 构建并训练LSTM模型，分别预测0.025和0.975分位数
quantiles = [0.025, 0.975]
predictions_qlstm = {}

for quantile in quantiles:
    model = LSTM_model(units, learning_rate, train_X1, quantile)
    history = model.fit(train_X1, y_spring_train.values.ravel(), epochs=int(epochs), batch_size=10, verbose=2, shuffle=False)
    y_pre_qlstm = model.predict(test_X1)
    predictions_qlstm[quantile] = y_pre_qlstm

# 计算评估指标
R2, mse, mae, mape, rmse = [], [], [], [], []

for quantile, y_pre in predictions_qlstm.items():
    R2_qlstm = r2_score(y_spring_test.values.ravel(), y_pre_qlstm)
    mse_qlstm = mean_squared_error(y_spring_test.values.ravel(), y_pre_qlstm)
    mae_qlstm = mean_absolute_error(y_spring_test.values.ravel(), y_pre_qlstm)
    mape_qlstm = mean_absolute_percentage_error(y_spring_test.values.ravel(), y_pre_qlstm)
    rmse_qlstm = np.sqrt(mse_qlstm)

    R2.append(R2_qlstm)
    mse.append(mse_qlstm)
    mae.append(mae_qlstm)
    mape.append(mape_qlstm)
    rmse.append(rmse_qlstm)

    print(f"Quantile {quantile}:")
    print("计算得出的r方为：", R2_qlstm)
    print("MSE： ", mse_qlstm)
    print("MAE:", mae_qlstm)
    print("MAPE:", mape_qlstm)
    print("RMSE:", rmse_qlstm)

# 绘制PSO-LSTM预测之后的模型
plt.figure(figsize=(15,6))
for quantile, y_pre_qlstm in predictions_qlstm.items():
    plt.plot(y_pre_qlstm, label=f'{quantile} Quantile')

plt.plot(y_spring_test.values.ravel(), 'k--', label='Actual')
plt.legend()
plt.show()


# In[28]:


#计算区间预测指标
picp_95 = np.mean((y_spring_test.values >= predictions_qlstm[0.025]) & (y_spring_test.values <= predictions_qlstm[0.975]))
piaw_95 = np.mean(predictions_qlstm[0.975] - predictions_qlstm[0.025])
ace_95 = np.abs(0.95 - picp_95)
ais95 = np.mean((np.abs(y_spring_test.values - y_pre_qlstm) / (predictions_qlstm[0.975] - predictions_qlstm[0.025])) ** 2)
lstm_lb= predictions_qlstm[0.025]  # 下限值
lstm_ub = predictions_qlstm[0.975]  # 上限值
print(picp_95,piaw_95,ace_95,ais95)


# In[ ]:





# In[214]:


##成功的PSO-TCN参数调优  
from sklearn.metrics import mean_squared_error  
from mealpy.swarm_based.PSO import OriginalPSO  
from tensorflow.keras.optimizers import Adam  
from tensorflow.keras.layers import Conv1D, Dense, Dropout, Flatten  
from tensorflow.keras.models import Sequential  

# 调整 train_X1 和 test_X1 的形状  
train_X1 = x_spring_train.values.reshape((x_spring_train.shape[0], 78, 1))  
test_X1 = x_spring_test.values.reshape((x_spring_test.shape[0], 78, 1))
val_X1 = x_spring_val.values.reshape((x_spring_val.shape[0], 78, 1))

print("train_X1.shape:", train_X1.shape)  
print("test_X1.shape:", test_X1.shape)  

# 在TCN_model 函数中，确保kernel_size为整数  
def TCN_model(filters, kernel_size, learning_rate):  
    model = Sequential()  
    model.add(Conv1D(filters=int(filters), kernel_size=int(kernel_size), padding='causal', activation='relu', input_shape=(train_X1.shape[1], train_X1.shape[2])))  
    model.add(Dropout(0.3))  
    model.add(Conv1D(filters=int(filters*3), kernel_size=int(kernel_size), padding='causal', activation='relu'))  
    model.add(Flatten())  
    model.add(Dense(1))  # 输出层  

    optimizer = Adam(learning_rate=learning_rate)  
    model.compile(optimizer=optimizer, loss='mse')  
    model.summary()  
    return model  

def cross_val(nest):  
    filters, kernel_size, epochs, learning_rate = nest  
    model = TCN_model(filters, kernel_size, learning_rate)  

    # 确保在交叉验证中使用的是适当的训练集，而不是验证集  
    history = model.fit(train_X1, y_spring_train.values.ravel(), epochs=int(epochs), batch_size=64, verbose=2, shuffle=False)  

    # 在验证集上进行预测并计算 MSE  
    y_pre = model.predict(test_X1)  # 使用测试集进行预测  
    MSE = mean_squared_error(y_spring_val.values.ravel(), y_pre)  
    return MSE  

# 同样在fitness_function中确保传递的kernel_size为整数  
def fitness_function(nest):  
    # 将kernel_size转换为最接近的整数  
    kernel_size = round(nest[1])  
    nest_int = [nest[0], kernel_size, nest[2], nest[3]]  # 重建nest，确保kernel_size为整数  
    MSE = cross_val(nest_int)  
    return MSE

def get_nest():  
    problem_dict1 = {  
        "fit_func": fitness_function,  
        "lb": [30, 2, 30, 1e-06],  # 参数的下限  
        "ub": [60, 5, 60, 1e-03],  # 参数的上限  
        "minmax": "min",  
        "verbose": True,  
    }  
    epochs = 2  
    pop_size = 16  
    c1 = 2.05  
    c2 = 2.05  
    w_min = 0.4  
    w_max = 0.9  
    model = OriginalPSO(epochs, pop_size, c1, c2, w_min, w_max)  

    best_position, best_fitness = model.solve(problem_dict1)  
    print(f"Solution: {best_position}, Fitness: {best_fitness}")  
    return [best_fitness, best_position]  

# 获取最佳参数  
nest = get_nest()  
print(nest)  

# 取nest中的数赋给需要优化的参数  
filters = nest[1][0]  
kernel_size = nest[1][1]  
epochs = nest[1][2]  
learning_rate = nest[1][3]  

# 构建并训练TCN模型  
model = TCN_model(filters, kernel_size, learning_rate)  
history = model.fit(train_X1, y_spring_train.values.ravel(), epochs=int(epochs), batch_size=10, verbose=2, shuffle=False)  

# 执行预测  
y_pre_tcn = model.predict(test_X1)  

# 计算r方  
R2_tcn = r2_score(y_spring_test.values.ravel(), y_pre_tcn)  
print("计算得出的r方为：", R2_tcn)  

# 计算评估指标  
mse_tcn = mean_squared_error(y_spring_test.values.ravel(), y_pre_tcn)  
mae_tcn = mean_absolute_error(y_spring_test.values.ravel(), y_pre_tcn)  
mape_tcn = mean_absolute_percentage_error(y_spring_test.values.ravel(), y_pre_tcn)  

# 计算RMSE
rmse_tcn= np.sqrt(mse_tcn)
print("MSE： ", mse_tcn)
print("MAE:", mae_tcn)
print("MAPE:", mape_tcn)
print("RMSE:", rmse_tcn)
#绘制PSO-TCN预测之后的模型
plt.figure(figsize=(15,6))
plt.plot(y_spring_test.values.ravel())
plt.plot(y_pre_tcn)
plt.show()


# In[215]:


import pandas as pd

# 计算y_spring_test的最大值和最小值
scaler_max = spring_y_test.values.max()  # 如果y_spring_test是多维的，使用max().max()获取全局最大值
scaler_min = spring_y_test.values.min()  # 如果y_spring_test是多维的，使用min().min()获取全局最小值
print("归一化的最大值:", scaler_max)
print("归一化的最小值:", scaler_min)

# 反归一化函数
def inverse_transform(y_pred_scaled, y_actual_scaled, scaler_min, scaler_max):
    y_pred = y_pred_scaled * (scaler_max - scaler_min) + scaler_min
    y_actual = y_actual_scaled * (scaler_max - scaler_min) + scaler_min
    return y_pred, y_actual

# 确保 y_pre_tcn 和 y_spring_test.values.ravel() 是一维数组
# 反归一化预测值和实际值
y_pre_tcn_inv, y_spring_test_inv = inverse_transform(y_pre_tcn, y_spring_test.values, scaler_min, scaler_max)
# 计算预测误差
spring_tcn_error = y_spring_test_inv - y_pre_tcn_inv
spring_tcn_error


# In[216]:


#将各个点预测结果和误差保存在文件夹中
import pandas as pd
# 将它们转换为 DataFrame
y_pre_tcn_df = pd.DataFrame(y_pre_tcn_inv, columns=['y_pre_tcn'])
spring_tcn_error_df = pd.DataFrame(spring_tcn_error, columns=['spring_tcn_error'])

# 使用 concat 函数将 DataFrame 合并为一个 DataFrame
combined_df = pd.concat([ 
    y_pre_tcn_df, 
    spring_tcn_error_df
], axis=1)

# 将合并后的 DataFrame 保存到 Excel 文件中
directory = 'D:\\article'
file_path = os.path.join(directory, 'TCN_prediction.xlsx')
combined_df.to_excel(file_path, index=False)
combined_df


# In[ ]:





# In[ ]:





# In[25]:


#TCN区间预测（95%）
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error, mean_absolute_percentage_error
from mealpy.swarm_based.PSO import OriginalPSO
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import Conv1D, Dense, Dropout, Flatten
from tensorflow.keras.models import Sequential
from tensorflow.keras import backend as K

# 调整 train_X1 和 test_X1 的形状
train_X1 = x_spring_train.values.reshape((x_spring_train.shape[0], 78, 1))
test_X1 = x_spring_test.values.reshape((x_spring_test.shape[0], 78, 1))
val_X1 = x_spring_val.values.reshape((x_spring_val.shape[0], 78, 1))

print("train_X1.shape:", train_X1.shape)
print("test_X1.shape:", test_X1.shape)

# 分位数损失函数
def quantile_loss(y_true, y_pred, quantile):
    return K.mean(K.maximum(quantile * (y_true - y_pred), (quantile - 1) * (y_true - y_pred)), axis=-1)

# TCN模型
def TCN_model(filters, kernel_size, learning_rate, quantile):
    model = Sequential()
    model.add(Conv1D(filters=int(filters), kernel_size=int(kernel_size), padding='causal', activation='relu', input_shape=(train_X1.shape[1], train_X1.shape[2])))
    model.add(Dropout(0.2))
    model.add(Conv1D(filters=int(filters * 3), kernel_size=int(kernel_size), padding='causal', activation='relu'))
    model.add(Flatten())
    model.add(Dense(1))  # 输出层
    optimizer = Adam(learning_rate=learning_rate)
    model.compile(optimizer=optimizer, loss=lambda y_true, y_pred: quantile_loss(y_true, y_pred, quantile))
    model.summary()
    return model

# 交叉验证
def cross_val(nest):
    filters, kernel_size, epochs, learning_rate, quantile = nest
    model = TCN_model(filters, kernel_size, learning_rate, quantile)
    history = model.fit(train_X1, y_spring_train.values.ravel(), epochs=int(epochs), batch_size=64, verbose=2, shuffle=False)
    y_pre_qtcn = model.predict(val_X1)
    MSE = mean_squared_error(y_spring_val.values.ravel(), y_pre_qtcn)
    return MSE

# 适应度函数
def fitness_function(nest):
    kernel_size = round(nest[1])
    nest_int = [nest[0], kernel_size, nest[2], nest[3], 0.95]  # 95%分位数
    MSE = cross_val(nest_int)
    return MSE

# 获取最佳参数
def get_nest(): 
    problem_dict1 = {
        "fit_func": fitness_function,
        "lb": [20, 2, 20, 1e-06, 0.025],  # 参数的下限
        "ub": [60, 5, 60, 1e-03, 0.975],  # 参数的上限
        "minmax": "min",
        "verbose": True,
    }
    epochs = 2
    pop_size = 16
    c1 = 2.05
    c2 = 2.05
    w_min = 0.4
    w_max = 0.9
    model = OriginalPSO(epochs, pop_size, c1, c2, w_min, w_max)
    best_position, best_fitness = model.solve(problem_dict1)
    print(f"Solution: {best_position}, Fitness: {best_fitness}")
    return [best_fitness, best_position]

# 获取最佳参数
nest = get_nest()
print(nest)

# 取nest中的数赋给需要优化的参数
filters = nest[1][0]
kernel_size = nest[1][1]
epochs = nest[1][2]
learning_rate = nest[1][3]

# 构建并训练TCN模型，分别预测0.025和0.975分位数
quantiles = [0.025, 0.975]
predictions = {}

for quantile in quantiles:
    model = TCN_model(filters, kernel_size, learning_rate, quantile)
    history = model.fit(train_X1, y_spring_train.values.ravel(), epochs=int(epochs), batch_size=10, verbose=2, shuffle=False)
    y_pre_qtcn = model.predict(test_X1)
    predictions[quantile] = y_pre_qtcn

# 计算评估指标
R2, mse, mae, mape, rmse = [], [], [], [], []

for quantile, y_pre_qtcn in predictions.items():
    R2_qtcn = r2_score(y_spring_test.values.ravel(), y_pre_qtcn)
    mse_qtcn = mean_squared_error(y_spring_test.values.ravel(), y_pre_qtcn)
    mae_qtcn = mean_absolute_error(y_spring_test.values.ravel(), y_pre_qtcn)
    mape_qtcn = mean_absolute_percentage_error(y_spring_test.values.ravel(), y_pre_qtcn)
    rmse_qtcn = np.sqrt(mse_qtcn)

    R2.append(R2_qtcn)
    mse.append(mse_qtcn)
    mae.append(mae_qtcn)
    mape.append(mape_qtcn)
    rmse.append(rmse_qtcn)

    print(f"Quantile {quantile}:")
    print("计算得出的r方为：", R2_qtcn)
    print("MSE： ", mse_qtcn)
    print("MAE:", mae_qtcn)
    print("MAPE:", mape_qtcn)
    print("RMSE:", rmse_qtcn)

# 绘制PSO-TCN预测之后的模型
plt.figure(figsize=(15,6))
for quantile, y_pre_qtcn in predictions.items():
    plt.plot(y_pre_qtcn, label=f'{quantile} Quantile')

plt.plot(y_spring_test.values.ravel(), 'k--', label='Actual')
plt.legend()
plt.show()


# In[26]:


#计算区间预测指标
picp_95 = np.mean((y_spring_test.values >= predictions[0.025]) & (y_spring_test.values <= predictions[0.975]))
piaw_95 = np.mean(predictions[0.975] - predictions[0.025])
ace_95 = np.abs(0.95 - picp_95)
ais95 = np.mean((np.abs(y_spring_test.values - y_pre_qtcn) / (predictions[0.975] - predictions[0.025])) ** 2)
tcn_lb= predictions[0.025]  # 下限值
tcn_ub = predictions[0.975]  # 上限值
print(picp_95,piaw_95,ace_95,ais95)


# In[ ]:





# In[34]:


###GRU进行预测
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import GRU, Dropout, Dense
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import numpy as np
import matplotlib.pylab as plt

train_X1 = x_spring_train.values.reshape((x_spring_train.shape[0], 78, 1))
test_X1 = x_spring_test.values.reshape((x_spring_test.shape[0], 78, 1))
val_X1 = x_spring_val.values.reshape((x_spring_val.shape[0], 78, 1))

# 定义GRU模型
def GRU_model(units, learning_rate, train_X):
    model = Sequential()
    model.add(GRU(units=int(units), input_shape=(train_X1.shape[1], train_X1.shape[2]), return_sequences=True))
    model.add(Dropout(0.5))
    model.add(GRU(units=int(units*3), return_sequences=False))
    model.add(Dense(1))  # 预测的是1步输出
    optimizer = Adam(learning_rate=learning_rate)
    model.compile(optimizer=optimizer, loss='mse')
    model.summary()
    return model

##成功的PSO参数调优
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error, mean_squared_error
from sklearn.metrics import r2_score
import numpy as np
from mealpy.swarm_based.PSO import OriginalPSO

def cross_val(nest):
    units,epochs,learning_rate = nest
    model=GRU_model(units, learning_rate, val_X1)
    history = model.fit(val_X1, y_spring_val, epochs=int(epochs), batch_size=64, verbose=2, shuffle=False)
    # 执行预测
    y_pre = model.predict(val_X1)
    # 计算MSE误差值
    MSE= mean_squared_error(y_spring_val, y_pre)
    return MSE

#定义适应度函数
def fitness_function(nest):
    MSE = cross_val(nest)
    return MSE

#定义适应度函数的参数范围
def get_nest():
    problem_dict1 = {
         "fit_func": fitness_function,
         "lb": [30,30, 1e-06],
         "ub": [60,60, 1e-03],
         "minmax": "min",
    }
#设置迭代次数和粒子群数（越大运行速度越慢）
    epochs = 2  
    pop_size = 16  
    c1 = 2.05  
    c2 = 2.05  
    w_min = 0.4  
    w_max = 0.9  
    model = OriginalPSO(epochs, pop_size, c1, c2, w_min, w_max)  
    best_position, best_fitness = model.solve(problem_dict1)  
    print(f"Solution: {best_position}, Fitness: {best_fitness}")  
    return [best_fitness, best_position]  
#将获得的参数放入nest
nest=get_nest()
print(nest)
#取nest中的数赋给需要优化的参数
# units,epochs,learning_rate
units= nest[1][0]
epochs= nest[1][1]
learning_rate =nest[1][2]

model = GRU_model(units, learning_rate, train_X1)
history = model.fit(train_X1, y_spring_train, epochs=int(epochs), batch_size=64, verbose=2, shuffle=False)
y_pre_gru = model.predict(test_X1)
# 计算r方  
R2_gru = r2_score(y_spring_test, y_pre_gru)  
print("计算得出的r方为：", R2_gru)  

# 计算评估指标  
mse_gru = mean_squared_error(y_spring_test, y_pre_gru)  
mae_gru = mean_absolute_error(y_spring_test, y_pre_gru)  
mape_gru = mean_absolute_percentage_error(y_spring_test, y_pre_gru)  

# 计算RMSE
rmse_gru= np.sqrt(mse_gru)
print("MSE： ", mse_gru)
print("MAE:", mae_gru)
print("MAPE:", mape_gru)
print("RMSE:", rmse_gru)
#绘制PSO-GRU预测之后的模型
plt.figure(figsize=(15,6))
plt.plot(y_spring_test.values.ravel())
plt.plot(y_pre_gru)
plt.show()


# In[39]:


#绘制PSO-GRU预测之后的模型
plt.figure(figsize=(15,6))
plt.plot(y_spring_test.values.ravel())
plt.plot(y_pre_gru)
plt.show()


# In[35]:


import pandas as pd
# 计算y_spring_test的最大值和最小值
scaler_max = spring_y_test.values.max()  # 如果y_spring_test是多维的，使用max().max()获取全局最大值
scaler_min = spring_y_test.values.min()  # 如果y_spring_test是多维的，使用min().min()获取全局最小值
print("归一化的最大值:", scaler_max)
print("归一化的最小值:", scaler_min)

# 反归一化函数
def inverse_transform(y_pred_scaled, y_actual_scaled, scaler_min, scaler_max):
    y_pred = y_pred_scaled * (scaler_max - scaler_min) + scaler_min
    y_actual = y_actual_scaled * (scaler_max - scaler_min) + scaler_min
    return y_pred, y_actual

# 确保 y_pre_gru 和 y_spring_test.values.ravel() 是一维数组
# 反归一化预测值和实际值
y_pre_gru_inv, y_spring_test_inv = inverse_transform(y_pre_gru, y_spring_test.values, scaler_min, scaler_max)
# 计算预测误差
spring_gru_error = y_spring_test_inv - y_pre_gru_inv
# 保存误差到数据框
spring_gru_error_df = pd.DataFrame(spring_gru_error, columns=['error'])
# 打印数据框
print(spring_gru_error_df)


# In[27]:


#GRU区间预测（95%）
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import GRU, Dropout, Dense
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import numpy as np
import matplotlib.pylab as plt
from tensorflow.keras import backend as K

train_X1 = x_spring_train.values.reshape((x_spring_train.shape[0], 78, 1))
test_X1 = x_spring_test.values.reshape((x_spring_test.shape[0], 78, 1))
val_X1 = x_spring_val.values.reshape((x_spring_val.shape[0], 78, 1))

# 分位数损失函数
def quantile_loss(y_true, y_pred, quantile):
    return K.mean(K.maximum(quantile * (y_true - y_pred), (quantile - 1) * (y_true - y_pred)), axis=-1)

# GRU模型
def GRU_model(units, learning_rate, train_X, quantile):
    model = Sequential()
    model.add(GRU(units=int(units), input_shape=(train_X.shape[1], train_X.shape[2]), return_sequences=True))
    model.add(Dropout(0.5))
    model.add(GRU(units=int(units*3), return_sequences=False))
    model.add(Dense(1))  # 预测的是1步输出
    optimizer = Adam(learning_rate=learning_rate)
    model.compile(optimizer=optimizer, loss=lambda y_true, y_pred: quantile_loss(y_true, y_pred, quantile))
    model.summary()
    return model

##成功的PSO参数调优
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error, mean_squared_error
from sklearn.metrics import r2_score
import numpy as np
from mealpy.swarm_based.PSO import OriginalPSO

def cross_val(nest):
    units,epochs,learning_rate,quantile = nest
    model=GRU_model(units, learning_rate, val_X1,quantile)
    history = model.fit(val_X1, y_spring_val, epochs=int(epochs), batch_size=64, verbose=2, shuffle=False)
    # 执行预测
    y_pre = model.predict(val_X1)
    # 计算MSE误差值
    MSE= mean_squared_error(y_spring_val, y_pre)
    return MSE

# 适应度函数
def fitness_function(nest):
    units, epochs, learning_rate, quantile = nest
    MSE = cross_val([units, epochs, learning_rate, quantile])
    return MSE

#定义适应度函数的参数范围
def get_nest():
    problem_dict1 = {
         "fit_func": fitness_function,
         "lb": [30,30, 1e-06,0.025],
         "ub": [50,50, 1e-04,0.975],
         "minmax": "min",
    }
#设置迭代次数和粒子群数（越大运行速度越慢）
    epochs = 2  
    pop_size = 16  
    c1 = 2.05  
    c2 = 2.05  
    w_min = 0.4  
    w_max = 0.9  
    model = OriginalPSO(epochs, pop_size, c1, c2, w_min, w_max)  
    best_position, best_fitness = model.solve(problem_dict1)  
    print(f"Solution: {best_position}, Fitness: {best_fitness}")  
    return [best_fitness, best_position]  
#将获得的参数放入nest
nest=get_nest()
print(nest)
#取nest中的数赋给需要优化的参数
# units,epochs,learning_rate
units= nest[1][0]
epochs= nest[1][1]
learning_rate =nest[1][2]

# 构建并训练GRU模型，分别预测0.025和0.975分位数
quantiles = [0.025, 0.975]
predictions_qgru = {}

for quantile in quantiles:
    model = GRU_model(units, learning_rate, train_X1, quantile)
    history = model.fit(train_X1, y_spring_train.values.ravel(), epochs=int(epochs), batch_size=10, verbose=2, shuffle=False)
    y_pre_qgru = model.predict(test_X1)
    predictions_qgru[quantile] = y_pre_qgru

# 计算评估指标
R2, mse, mae, mape, rmse = [], [], [], [], []

for quantile, y_pre_qgru in predictions_qgru.items():
    R2_qgru = r2_score(y_spring_test.values.ravel(), y_pre_qgru)
    mse_qgru = mean_squared_error(y_spring_test.values.ravel(), y_pre_qgru)
    mae_qgru = mean_absolute_error(y_spring_test.values.ravel(), y_pre_qgru)
    mape_qgru = mean_absolute_percentage_error(y_spring_test.values.ravel(), y_pre_qgru)
    rmse_qgru = np.sqrt(mse_qgru)

    R2.append(R2_qgru)
    mse.append(mse_qgru)
    mae.append(mae_qgru)
    mape.append(mape_qgru)
    rmse.append(rmse_qgru)

    print(f"Quantile {quantile}:")
    print("计算得出的r方为：", R2_qgru)
    print("MSE： ", mse_qgru)
    print("MAE:", mae_qgru)
    print("MAPE:", mape_qgru)
    print("RMSE:", rmse_qgru)

# 绘制PSO-GRU预测之后的模型
plt.figure(figsize=(15,6))
for quantile, y_pre_qgru in predictions_qgru.items():
    plt.plot(y_pre_qgru, label=f'{quantile} Quantile')

plt.plot(y_spring_test.values.ravel(), 'k--', label='Actual')
plt.legend()
plt.show()


# In[28]:


#计算区间预测指标
picp_95 = np.mean((y_spring_test.values >= predictions_qgru[0.025]) & (y_spring_test.values <= predictions_qgru[0.975]))
piaw_95 = np.mean(predictions_qgru[0.975] - predictions_qgru[0.025])
ace_95 = np.abs(0.95 - picp_95)
ais95 = np.mean((np.abs(y_spring_test.values - y_pre_qgru) / (predictions_qgru[0.975] - predictions_qgru[0.025])) ** 2)
gru_lb= predictions_qgru[0.025]  # 下限值
gru_ub = predictions_qgru[0.975]  # 上限值
print(picp_95,piaw_95,ace_95,ais95)


# In[ ]:





# In[36]:


###RNN进行预测
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_percentage_error, mean_absolute_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import SimpleRNN

train_X1 = x_spring_train.values.reshape((x_spring_train.shape[0],78,1))
test_X1= x_spring_test.values.reshape((x_spring_test.shape[0],78,1))
val_X1= x_spring_val.values.reshape((x_spring_val.shape[0],78,1))
print("train_X1.shape ",train_X1.shape)
print("test_X1.shape ",test_X1.shape)

# 定义 RNN 模型
def RNN_model(units, learning_rate):
    model = Sequential()
    units=int(units)
    model.add(SimpleRNN(units, input_shape=(78,1), return_sequences=True))
    # 隐藏层
    model.add(SimpleRNN(units))
    model.add(Dense(1))
    # 编译模型
    optimizer = Adam(learning_rate=learning_rate)
    model.compile(optimizer=optimizer, loss='mse')
    model.summary()
    return model

##成功的PSO参数调优
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error, mean_squared_error
from sklearn.metrics import r2_score
import numpy as np
from mealpy.swarm_based.PSO import OriginalPSO

def cross_val(nest):
    units,epochs,learning_rate = nest
    model=RNN_model(units, learning_rate)
    history = model.fit(train_X1, y_spring_train, epochs=int(epochs), batch_size=64, verbose=2, shuffle=False)
    # 执行预测
    y_pre = model.predict(val_X1)
    # 计算MSE误差值
    MSE= mean_squared_error(y_spring_val, y_pre)
    return MSE
    
#定义适应度函数
def fitness_function(nest):
    MSE = cross_val(nest)
    return MSE

#定义适应度函数的参数范围
def get_nest():
    problem_dict1 = {
         "fit_func": fitness_function,
         "lb": [30,30, 1e-06],
         "ub": [60,60, 1e-03],
         "minmax": "min",
    }
#设置迭代次数和粒子群数（越大运行速度越慢）
    epochs = 2  
    pop_size = 16  
    c1 = 2.05  
    c2 = 2.05  
    w_min = 0.4  
    w_max = 0.9  
    model = OriginalPSO(epochs, pop_size, c1, c2, w_min, w_max)  
    best_position, best_fitness = model.solve(problem_dict1)  
    print(f"Solution: {best_position}, Fitness: {best_fitness}")  
    return [best_fitness, best_position]  
#将获得的参数放入nest
nest=get_nest()
print(nest)
#取nest中的数赋给需要优化的参数
# units,epochs,learning_rate
units= nest[1][0]
epochs= nest[1][1]
learning_rate =nest[1][2]
model = RNN_model(units, learning_rate)
history = model.fit(train_X1, y_spring_train, epochs=int(epochs), batch_size=64, verbose=2, shuffle=False)
y_pre_rnn = model.predict(test_X1)
# 计算r方  
R2_rnn = r2_score(y_spring_test, y_pre_rnn)  
print("计算得出的r方为：", R2_rnn)  

# 计算评估指标  
mse_rnn = mean_squared_error(y_spring_test, y_pre_rnn)  
mae_rnn = mean_absolute_error(y_spring_test, y_pre_rnn)  
mape_rnn = mean_absolute_percentage_error(y_spring_test, y_pre_rnn)  

# 计算RMSE
rmse_rnn= np.sqrt(mse_rnn)
print("MSE： ", mse_rnn)
print("MAE:", mae_rnn)
print("MAPE:", mape_rnn)
print("RMSE:", rmse_rnn)
#绘制PSO-rnn预测之后的模型
plt.figure(figsize=(15,6))
plt.plot(y_spring_test.values.ravel())
plt.plot(y_pre_rnn)
plt.show()


# In[40]:


#绘制PSO-RNN预测之后的模型
plt.figure(figsize=(15,6))
plt.plot(y_spring_test.values.ravel())
plt.plot(y_pre_rnn)
plt.show()


# In[37]:


import pandas as pd

# 计算y_spring_test的最大值和最小值
scaler_max = spring_y_test.values.max()  # 如果y_spring_test是多维的，使用max().max()获取全局最大值
scaler_min = spring_y_test.values.min()  # 如果y_spring_test是多维的，使用min().min()获取全局最小值
print("归一化的最大值:", scaler_max)
print("归一化的最小值:", scaler_min)

# 反归一化函数
def inverse_transform(y_pred_scaled, y_actual_scaled, scaler_min, scaler_max):
    y_pred = y_pred_scaled * (scaler_max - scaler_min) + scaler_min
    y_actual = y_actual_scaled * (scaler_max - scaler_min) + scaler_min
    return y_pred, y_actual

# 确保 y_pre_rnn 和 y_spring_test.values.ravel() 是一维数组
# 反归一化预测值和实际值
y_pre_rnn_inv, y_spring_test_inv = inverse_transform(y_pre_rnn, y_spring_test.values, scaler_min, scaler_max)
# 计算预测误差
spring_rnn_error = y_spring_test_inv - y_pre_rnn_inv
# 保存误差到数据框
spring_rnn_error_df = pd.DataFrame(spring_rnn_error, columns=['error'])
# 打印数据框
print(spring_rnn_error_df)


# In[29]:


###RNN进行区间预测
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_percentage_error, mean_absolute_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import SimpleRNN
from tensorflow.keras import backend as K

train_X1 = x_spring_train.values.reshape((x_spring_train.shape[0], 78, 1))
test_X1 = x_spring_test.values.reshape((x_spring_test.shape[0], 78, 1))
val_X1 = x_spring_val.values.reshape((x_spring_val.shape[0], 78, 1))

print("train_X1.shape ",train_X1.shape)
print("test_X1.shape ",test_X1.shape)
# 分位数损失函数
def quantile_loss(y_true, y_pred, quantile):
    return K.mean(K.maximum(quantile * (y_true - y_pred), (quantile - 1) * (y_true - y_pred)), axis=-1)

# RNN模型
def RNN_model(units, learning_rate, quantile):
    model = Sequential()
    units = int(units)
    model.add(SimpleRNN(units, input_shape=(78, 1), return_sequences=True))
    # 隐藏层
    model.add(SimpleRNN(units))
    # 输出层
    model.add(Dense(1))
    # 编译模型
    optimizer = Adam(learning_rate=learning_rate)
    model.compile(optimizer=optimizer, loss=lambda y_true, y_pred: quantile_loss(y_true, y_pred, quantile))
    model.summary()
    return model
    
##成功的PSO参数调优
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error, mean_squared_error
from sklearn.metrics import r2_score
import numpy as np
from mealpy.swarm_based.PSO import OriginalPSO

def cross_val(nest):
    units,epochs,learning_rate,quantile = nest
    model=RNN_model(units, learning_rate,quantile)
    history = model.fit(train_X1, y_spring_train, epochs=int(epochs), batch_size=64, verbose=2, shuffle=False)
    # 执行预测
    y_pre = model.predict(val_X1)
    # 计算MSE误差值
    MSE= mean_squared_error(y_spring_val, y_pre)
    return MSE

# 适应度函数
def fitness_function(nest):
    units, epochs, learning_rate, quantile = nest
    MSE = cross_val([units, epochs, learning_rate, quantile])
    return MSE

#定义适应度函数的参数范围
def get_nest():
    problem_dict1 = {
         "fit_func": fitness_function,
         "lb": [30,30, 1e-06,0.025],
         "ub": [60,60, 1e-03,0.975],
         "minmax": "min",
    }
#设置迭代次数和粒子群数（越大运行速度越慢）
    epochs = 2  
    pop_size = 16  
    c1 = 2.05  
    c2 = 2.05  
    w_min = 0.4  
    w_max = 0.9  
    model = OriginalPSO(epochs, pop_size, c1, c2, w_min, w_max)  
    best_position, best_fitness = model.solve(problem_dict1)  
    print(f"Solution: {best_position}, Fitness: {best_fitness}")  
    return [best_fitness, best_position]  
#将获得的参数放入nest
nest=get_nest()
print(nest)
#取nest中的数赋给需要优化的参数
# units,epochs,learning_rate
units= nest[1][0]
epochs= nest[1][1]
learning_rate =nest[1][2]

# 构建并训练RNN模型，分别预测0.025和0.975分位数
quantiles = [0.025, 0.975]
predictions_qrnn = {}

for quantile in quantiles:
    model = RNN_model(units, learning_rate, quantile)
    history = model.fit(train_X1, y_spring_train.values.ravel(), epochs=int(epochs), batch_size=64, verbose=2, shuffle=False)
    y_pre_qrnn = model.predict(test_X1)
    predictions_qrnn[quantile] = y_pre_qrnn

# 计算评估指标
R2, mse, mae, mape, rmse = [], [], [], [], []

for quantile, y_pre_qrnn in predictions_qrnn.items():
    R2_qrnn = r2_score(y_spring_test.values.ravel(), y_pre_qrnn)
    mse_qrnn = mean_squared_error(y_spring_test.values.ravel(), y_pre_qrnn)
    mae_qrnn = mean_absolute_error(y_spring_test.values.ravel(), y_pre_qrnn)
    mape_qrnn = mean_absolute_percentage_error(y_spring_test.values.ravel(), y_pre_qrnn)
    rmse_qrnn = np.sqrt(mse_qrnn)

    R2.append(R2_qrnn)
    mse.append(mse_qrnn)
    mae.append(mae_qrnn)
    mape.append(mape_qrnn)
    rmse.append(rmse_qrnn)

    print(f"Quantile {quantile}:")
    print("计算得出的r方为：", R2_qrnn)
    print("MSE： ", mse_qrnn)
    print("MAE:", mae_qrnn)
    print("MAPE:", mape_qrnn)
    print("RMSE:", rmse_qrnn)

# 绘制PSO-RNN预测之后的模型
plt.figure(figsize=(15,6))
for quantile, y_pre_qrnn in predictions_qrnn.items():
    plt.plot(y_pre_qrnn, label=f'{quantile} Quantile')

plt.plot(y_spring_test.values.ravel(), 'k--', label='Actual')
plt.legend()
plt.show()


# In[30]:


#计算区间预测指标
picp_95 = np.mean((y_spring_test.values >= predictions_qrnn[0.025]) & (y_spring_test.values <= predictions_qrnn[0.975]))
piaw_95 = np.mean(predictions_qrnn[0.975] - predictions_qrnn[0.025])
ace_95 = np.abs(0.95 - picp_95)
ais95 = np.mean((np.abs(y_spring_test.values - y_pre_qrnn) / (predictions_qrnn[0.975] - predictions_qrnn[0.025])) ** 2)
rnn_lb= predictions_qrnn[0.025]  # 下限值
rnn_ub = predictions_qrnn[0.975]  # 上限值
print(picp_95,piaw_95,ace_95,ais95)


# In[ ]:





# In[38]:


#将各个点预测结果和误差保存在文件夹中
import pandas as pd
# 将它们转换为 DataFrame
y_pre_lstm_df = pd.DataFrame(y_pre_lstm_inv, columns=['y_pre_lstm'])
spring_lstm_error_df = pd.DataFrame(spring_lstm_error, columns=['spring_lstm_error'])
y_pre_tcn_df = pd.DataFrame(y_pre_tcn_inv, columns=['y_pre_tcn'])
spring_tcn_error_df = pd.DataFrame(spring_tcn_error, columns=['spring_tcn_error'])
y_pre_gru_df = pd.DataFrame(y_pre_gru_inv, columns=['y_pre_gru'])
spring_gru_error_df = pd.DataFrame(spring_gru_error, columns=['spring_gru_error'])
y_pre_rnn_df = pd.DataFrame(y_pre_rnn_inv, columns=['y_pre_rnn'])
spring_rnn_error_df = pd.DataFrame(spring_rnn_error, columns=['spring_rnn_error'])
y_spring_test_df = pd.DataFrame(y_spring_test_inv, columns=['y_spring_test'])

# 使用 concat 函数将 DataFrame 合并为一个 DataFrame
combined_df = pd.concat([
    y_pre_lstm_df, 
    spring_lstm_error_df, 
    y_pre_tcn_df, 
    spring_tcn_error_df,
    y_pre_gru_df, 
    spring_gru_error_df,
    y_pre_rnn_df, 
    spring_rnn_error_df,
    y_spring_test_df
], axis=1)

# 将合并后的 DataFrame 保存到 Excel 文件中
directory = 'D:\\article'
file_path = os.path.join(directory, 'Model_prediction.xlsx')
combined_df.to_excel(file_path, index=False)
combined_df


# In[ ]:





# In[ ]:





# In[31]:


import pandas as pd
import numpy as np
import os

# 创建一个DataFrame来保存数据
excel_data = {
    "RNN_Lower_Bound": rnn_lb.flatten(),
    "RNN_Upper_Bound": rnn_ub.flatten(),
    "LSTM_Lower_Bound": lstm_lb.flatten(),
    "LSTM_Upper_Bound": lstm_ub.flatten(),
    "TCN_Lower_Bound": tcn_lb.flatten(),
    "TCN_Upper_Bound": tcn_ub.flatten(),
    "GRU_Lower_Bound": gru_lb.flatten(),
    "GRU_Upper_Bound": gru_ub.flatten()
}

# 将数据转换为DataFrame
excel_df = pd.DataFrame(excel_data)

# 获取spring_y_test的最大值和最小值
scaler_max = spring_y_test.values.max()  # 如果spring_y_test是多维的，使用max().max()获取全局最大值
scaler_min = spring_y_test.values.min()  # 如果spring_y_test是多维的，使用min().min()获取全局最小值
print("归一化的最大值:", scaler_max)
print("归一化的最小值:", scaler_min)

# 反归一化函数
def inverse_normalize(value, scaler_min, scaler_max):
    return value * (scaler_max - scaler_min) + scaler_min

# 应用反归一化
models = ["RNN", "LSTM", "TCN", "GRU"]
for model in models:
    model_lower_key = model + "_Lower_Bound"
    model_upper_key = model + "_Upper_Bound"
    excel_df[model_lower_key] = excel_df[model_lower_key].apply(lambda x: inverse_normalize(x, scaler_min, scaler_max))
    excel_df[model_upper_key] = excel_df[model_upper_key].apply(lambda x: inverse_normalize(x, scaler_min, scaler_max))

# 将合并后的 DataFrame 保存到 Excel 文件中
directory = 'D:\\article'
file_path = os.path.join(directory, 'predictions_quantiles2.xlsx')
excel_df.to_excel(file_path, index=False)

# 显示DataFrame
excel_df


# In[ ]:





# In[31]:


###误差修正
#导入文件
point_prediction_error = pd.read_excel(r'D:\article\point_prediction_error2.xlsx', sheet_name=None)
point_prediction_error_df = pd.concat(point_prediction_error.values(), ignore_index=True)
point_prediction_error_df


# In[32]:


import pandas as pd
from sklearn.preprocessing import MinMaxScaler

# 取出各列
y_pre_itransformer = point_prediction_error_df.iloc[:, 0].to_frame(name='y_pre_itransformer')
spring_itransformer_error = point_prediction_error_df.iloc[:, 1].to_frame(name='spring_itransformer_error')
y_spring_test = point_prediction_error_df.iloc[:, -1].to_frame(name='y_spring_test')

scaler = MinMaxScaler()
scaler.fit(train_itransformer_error.reshape(-1, 1))  # 只用训练集fit，测试集不fit

y_pre_itransformer_normalized = pd.DataFrame(
    scaler.transform(y_pre_itransformer), 
    columns=y_pre_itransformer.columns, 
    index=y_pre_itransformer.index
)
spring_itransformer_error_normalized = pd.DataFrame(
    scaler.transform(spring_itransformer_error), 
    columns=spring_itransformer_error.columns, 
    index=spring_itransformer_error.index
)
y_spring_test_normalized = pd.DataFrame(
    scaler.transform(y_spring_test), 
    columns=y_spring_test.columns, 
    index=y_spring_test.index
)


# In[33]:


import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from scipy.linalg import svd, inv

np.random.seed(0)

# ==================== DMD工具函数 ====================

def build_hankel(series, m):
    """构造Hankel矩阵，m为嵌入维度"""
    N = len(series)
    H = np.zeros((m, N - m + 1))
    for i in range(m):
        H[i, :] = series[i:i + N - m + 1]
    return H

def dmd_fit(X_train, rank):
    X1 = X_train[:, :-1]  # 前N-1列
    X2 = X_train[:, 1:]   # 后N-1列
    
    U, S, Vh = svd(X1, full_matrices=False)
    U_r = U[:, :rank]
    S_r = np.diag(S[:rank])
    Vh_r = Vh[:rank, :]
    
    # 拟合降阶DMD算子
    A_tilde = U_r.T @ X2 @ Vh_r.T @ inv(S_r)
    eigenvalues, eigenvectors = np.linalg.eig(A_tilde)
    
    # 计算全阶DMD模态
    phi = X2 @ Vh_r.T @ inv(S_r) @ eigenvectors
    
    # 计算初始模态振幅
    b = np.linalg.pinv(phi) @ X1[:, 0]
    
    return phi, eigenvalues, b

def dmd_step_predict(phi, eigenvalues, current_state):
    """用训练好的DMD算子做一步前向预测"""
    # 将当前状态投影到模态空间，一步推进
    proj = np.linalg.pinv(phi) @ current_state
    next_proj = eigenvalues * proj
    return phi @ next_proj

scaler = MinMaxScaler()
scaler.fit(train_itransformer_error.reshape(-1, 1))  # 只用训练集fit

train_error_norm = scaler.transform(train_itransformer_error.reshape(-1, 1)).flatten()
test_error_norm = scaler.transform(spring_itransformer_error.values.reshape(-1, 1)).flatten()
y_pre_norm = scaler.transform(y_pre_itransformer.values.reshape(-1, 1)).flatten()
y_test_norm = scaler.transform(y_spring_test.values.reshape(-1, 1)).flatten()


m = 12   # 嵌入维度
r = 5    # 截断秩

X_train_hankel = build_hankel(train_error_norm, m)
phi, eigenvalues, b = dmd_fit(X_train_hankel, rank=r)

n_test = len(test_error_norm)
corrected_test_error = np.zeros(n_test)

corrected_test_error[:m] = test_error_norm[:m]

for t in range(m, n_test):
    # 只用t时刻之前的m个点构造状态
    current_state = test_error_norm[t - m:t]
    # DMD一步前向预测残差变化
    dmd_pred = dmd_step_predict(phi, eigenvalues, current_state)
    # 修正当前时刻误差
    correction_strength = 0.2
    corrected_test_error[t] = test_error_norm[t] + correction_strength * dmd_pred[-1]

corrected_predictions_norm = y_pre_norm + corrected_test_error

# 反归一化
corrected_predictions = scaler.inverse_transform(corrected_predictions_norm.reshape(-1, 1)).flatten()

plt.figure(figsize=(15, 10))

plt.subplot(2, 1, 1)
plt.title("Original vs True Values")
plt.plot(y_test_norm, label='True Values', alpha=0.9)
plt.plot(y_pre_norm, label='Original Predictions', alpha=0.7)
plt.legend()
plt.xlabel("Time")
plt.ylabel("Values")

plt.subplot(2, 1, 2)
plt.title("Corrected vs True Values")
plt.plot(y_test_norm, label='True Values', linestyle='--', alpha=0.9)
plt.plot(corrected_predictions_norm, label='Corrected Predictions', alpha=0.7)
plt.legend()
plt.xlabel("Time")
plt.ylabel("Values")

plt.tight_layout()
plt.show()


# In[34]:


from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# 计算原始预测的误差指标
original_R2 = r2_score(y_spring_test.values, y_pre_itransformer.values)
original_mse = mean_squared_error(y_spring_test.values, y_pre_itransformer.values)
original_rmse = np.sqrt(original_mse)
original_mae = mean_absolute_error(y_spring_test.values, y_pre_itransformer.values)

# 计算修正后预测的误差指标（用反归一化后的原始量纲）
corrected_R2 = r2_score(y_spring_test.values, corrected_predictions)
corrected_mse = mean_squared_error(y_spring_test.values, corrected_predictions)
corrected_rmse = np.sqrt(corrected_mse)
corrected_mae = mean_absolute_error(y_spring_test.values, corrected_predictions)

print(f"Original - R2: {original_R2:.4f}, MSE: {original_mse:.4f}, RMSE: {original_rmse:.4f}, MAE: {original_mae:.4f}")
print(f"Corrected - R2: {corrected_R2:.4f}, MSE: {corrected_mse:.4f}, RMSE: {corrected_rmse:.4f}, MAE: {corrected_mae:.4f}")

# 计算变化率
R2_change = ((corrected_R2 - original_R2) / abs(original_R2)) * 100
mse_reduction = ((original_mse - corrected_mse) / original_mse) * 100
rmse_reduction = ((original_rmse - corrected_rmse) / original_rmse) * 100
mae_reduction = ((original_mae - corrected_mae) / original_mae) * 100

print(f"\nR2 Improvement: {R2_change:.2f}%")
print(f"MSE Reduction: {mse_reduction:.2f}%")
print(f"RMSE Reduction: {rmse_reduction:.2f}%")
print(f"MAE Reduction: {mae_reduction:.2f}%")


# In[ ]:





# In[35]:


import pandas as pd
from sklearn.preprocessing import MinMaxScaler

# 取出各列
y_pre_lstm = point_prediction_error_df.iloc[:, 2].to_frame(name='y_pre_lstm')
spring_lstm_error = point_prediction_error_df.iloc[:, 3].to_frame(name='spring_lstm_error')
y_spring_test = point_prediction_error_df.iloc[:, -1].to_frame(name='y_spring_test')
scaler = MinMaxScaler()
scaler.fit(train_lstm_error.reshape(-1, 1))

# 所有变量都用同一个scaler transform
y_pre_lstm_normalized = pd.DataFrame(scaler.transform(y_pre_lstm), columns=y_pre_lstm.columns, index=y_pre_lstm.index)
spring_lstm_error_normalized = pd.DataFrame(scaler.transform(spring_lstm_error), columns=spring_lstm_error.columns, index=spring_lstm_error.index)
y_spring_test_normalized = pd.DataFrame(scaler.transform(y_spring_test), columns=y_spring_test.columns, index=y_spring_test.index)


# In[36]:


import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from scipy.linalg import svd, inv

np.random.seed(0)

# ==================== DMD工具函数 ====================

def build_hankel(series, m):
    N = len(series)
    H = np.zeros((m, N - m + 1))
    for i in range(m):
        H[i, :] = series[i:i + N - m + 1]
    return H

def dmd_fit(X_train, rank):
    X1 = X_train[:, :-1]
    X2 = X_train[:, 1:]
    U, S, Vh = svd(X1, full_matrices=False)
    U_r = U[:, :rank]
    S_r = np.diag(S[:rank])
    Vh_r = Vh[:rank, :]
    A_tilde = U_r.T @ X2 @ Vh_r.T @ inv(S_r)
    eigenvalues, eigenvectors = np.linalg.eig(A_tilde)
    phi = X2 @ Vh_r.T @ inv(S_r) @ eigenvectors
    b = np.linalg.pinv(phi) @ X1[:, 0]
    return phi, eigenvalues, b

def dmd_step_predict(phi, eigenvalues, current_state):
    proj = np.linalg.pinv(phi) @ current_state
    next_proj = eigenvalues * proj
    return phi @ next_proj

scaler = MinMaxScaler()
scaler.fit(train_lstm_error.reshape(-1, 1))

train_error_norm = scaler.transform(train_lstm_error.reshape(-1, 1)).flatten()
test_error_norm = scaler.transform(spring_lstm_error.values.reshape(-1, 1)).flatten()
y_pre_norm = scaler.transform(y_pre_lstm.values.reshape(-1, 1)).flatten()
y_test_norm = scaler.transform(y_spring_test.values.reshape(-1, 1)).flatten()

# 拟合DMD算子
m = 12   # 嵌入维度
r = 5    # 截断秩（与论文一致）

X_train_hankel = build_hankel(train_error_norm, m)
phi, eigenvalues, b = dmd_fit(X_train_hankel, rank=r)

# 逐点因果修正
n_test = len(test_error_norm)
corrected_test_error = np.zeros(n_test)
corrected_test_error[:m] = test_error_norm[:m]

correction_strength = 0.2

for t in range(m, n_test):
    current_state = test_error_norm[t - m:t]
    dmd_pred = dmd_step_predict(phi, eigenvalues, current_state)
    corrected_test_error[t] = test_error_norm[t] + correction_strength * dmd_pred[-1]

corrected_predictions_norm = y_pre_norm + corrected_test_error
corrected_predictions = scaler.inverse_transform(corrected_predictions_norm.reshape(-1, 1)).flatten()

plt.figure(figsize=(15, 10))

plt.subplot(2, 1, 1)
plt.title("Original vs True Values")
plt.plot(y_test_norm, label='True Values', alpha=0.9)
plt.plot(y_pre_norm, label='Original Predictions', alpha=0.7)
plt.legend()
plt.xlabel("Time")
plt.ylabel("Values")

plt.subplot(2, 1, 2)
plt.title("Corrected vs True Values")
plt.plot(y_test_norm, label='True Values', linestyle='--', alpha=0.9)
plt.plot(corrected_predictions_norm, label='Corrected Predictions', alpha=0.7)
plt.legend()
plt.xlabel("Time")
plt.ylabel("Values")

plt.tight_layout()
plt.show()


# In[37]:


from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# 建议用原始量纲计算
original_R2 = r2_score(y_spring_test.values, y_pre_lstm.values)
original_mse = mean_squared_error(y_spring_test.values, y_pre_lstm.values)
original_rmse = np.sqrt(original_mse)
original_mae = mean_absolute_error(y_spring_test.values, y_pre_lstm.values)

# 修正后（corrected_predictions 是反归一化后的结果）
corrected_R2 = r2_score(y_spring_test.values, corrected_predictions)
corrected_mse = mean_squared_error(y_spring_test.values, corrected_predictions)
corrected_rmse = np.sqrt(corrected_mse)
corrected_mae = mean_absolute_error(y_spring_test.values, corrected_predictions)

print(f"Original - R2: {original_R2:.4f}, MSE: {original_mse:.4f}, RMSE: {original_rmse:.4f}, MAE: {original_mae:.4f}")
print(f"Corrected - R2: {corrected_R2:.4f}, MSE: {corrected_mse:.4f}, RMSE: {corrected_rmse:.4f}, MAE: {corrected_mae:.4f}")

# R2是越大越好，算提升率
R2_improvement = ((corrected_R2 - original_R2) / abs(original_R2)) * 100
# 误差指标越小越好，算减少率
mse_reduction = ((original_mse - corrected_mse) / original_mse) * 100
rmse_reduction = ((original_rmse - corrected_rmse) / original_rmse) * 100
mae_reduction = ((original_mae - corrected_mae) / original_mae) * 100

print(f"\nR2 Improvement: {R2_improvement:.2f}%")
print(f"MSE Reduction: {mse_reduction:.2f}%")
print(f"RMSE Reduction: {rmse_reduction:.2f}%")
print(f"MAE Reduction: {mae_reduction:.2f}%")


# In[ ]:





# In[38]:


import pandas as pd
from sklearn.preprocessing import MinMaxScaler

# 取出各列
y_pre_tcn = point_prediction_error_df.iloc[:, 4].to_frame(name='y_pre_tcn')
spring_tcn_error = point_prediction_error_df.iloc[:, 5].to_frame(name='spring_tcn_error')
y_spring_test = point_prediction_error_df.iloc[:, -1].to_frame(name='y_spring_test')

scaler = MinMaxScaler()
scaler.fit(train_tcn_error.reshape(-1, 1))

# 所有变量都用同一个scaler transform
y_pre_tcn_normalized = pd.DataFrame(scaler.transform(y_pre_tcn), columns=y_pre_tcn.columns, index=y_pre_tcn.index)
spring_tcn_error_normalized = pd.DataFrame(scaler.transform(spring_tcn_error), columns=spring_tcn_error.columns, index=spring_tcn_error.index)
y_spring_test_normalized = pd.DataFrame(scaler.transform(y_spring_test), columns=y_spring_test.columns, index=y_spring_test.index)


# In[39]:


import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from scipy.linalg import svd, inv

np.random.seed(0)

def build_hankel(series, m):
    N = len(series)
    H = np.zeros((m, N - m + 1))
    for i in range(m):
        H[i, :] = series[i:i + N - m + 1]
    return H

def dmd_fit(X_train, rank):
    X1 = X_train[:, :-1]
    X2 = X_train[:, 1:]
    U, S, Vh = svd(X1, full_matrices=False)
    U_r = U[:, :rank]
    S_r = np.diag(S[:rank])
    Vh_r = Vh[:rank, :]
    A_tilde = U_r.T @ X2 @ Vh_r.T @ inv(S_r)
    eigenvalues, eigenvectors = np.linalg.eig(A_tilde)
    phi = X2 @ Vh_r.T @ inv(S_r) @ eigenvectors
    return phi, eigenvalues

def dmd_step_predict(phi, eigenvalues, current_state):
    proj = np.linalg.pinv(phi) @ current_state
    next_proj = eigenvalues * proj
    return phi @ next_proj

# 归一化
scaler = MinMaxScaler()
scaler.fit(train_tcn_error.reshape(-1, 1))  # train_tcn_error替换成你的训练集残差

train_error_norm = scaler.transform(train_tcn_error.reshape(-1, 1)).flatten()
test_error_norm = scaler.transform(spring_tcn_error.values.reshape(-1, 1)).flatten()
y_pre_norm = scaler.transform(y_pre_tcn.values.reshape(-1, 1)).flatten()
y_test_norm = scaler.transform(y_spring_test.values.reshape(-1, 1)).flatten()

#训练集拟合DMD
m, r = 12, 5
X_train_hankel = build_hankel(train_error_norm, m)
phi, eigenvalues = dmd_fit(X_train_hankel, rank=r)

#测试集因果修正
n_test = len(test_error_norm)
corrected_test_error = np.zeros(n_test)
corrected_test_error[:m] = test_error_norm[:m]
correction_strength = 0.2

for t in range(m, n_test):
    current_state = test_error_norm[t - m:t]
    dmd_pred = dmd_step_predict(phi, eigenvalues, current_state)
    corrected_test_error[t] = test_error_norm[t] + correction_strength * dmd_pred[-1]

#修正后预测值
corrected_predictions_norm = y_pre_norm + corrected_test_error
corrected_predictions = scaler.inverse_transform(corrected_predictions_norm.reshape(-1, 1)).flatten()

#画图
plt.figure(figsize=(15, 10))
plt.subplot(2, 1, 1)
plt.plot(y_test_norm, label='True Values', alpha=0.9)
plt.plot(y_pre_norm, label='Original Predictions', alpha=0.7)
plt.title("Original vs True Values")
plt.legend()
plt.subplot(2, 1, 2)
plt.plot(y_test_norm, label='True Values', linestyle='--', alpha=0.9)
plt.plot(corrected_predictions_norm, label='Corrected Predictions', alpha=0.7)
plt.title("Corrected vs True Values")
plt.legend()
plt.tight_layout()
plt.show()


# In[40]:


from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# 用原始量纲计算
original_R2 = r2_score(y_spring_test.values, y_pre_tcn.values)
original_mse = mean_squared_error(y_spring_test.values, y_pre_tcn.values)
original_rmse = np.sqrt(original_mse)
original_mae = mean_absolute_error(y_spring_test.values, y_pre_tcn.values)

# 修正后
corrected_R2 = r2_score(y_spring_test.values, corrected_predictions)
corrected_mse = mean_squared_error(y_spring_test.values, corrected_predictions)
corrected_rmse = np.sqrt(corrected_mse)
corrected_mae = mean_absolute_error(y_spring_test.values, corrected_predictions)

print(f"Original - R2: {original_R2:.4f}, MSE: {original_mse:.4f}, RMSE: {original_rmse:.4f}, MAE: {original_mae:.4f}")
print(f"Corrected - R2: {corrected_R2:.4f}, MSE: {corrected_mse:.4f}, RMSE: {corrected_rmse:.4f}, MAE: {corrected_mae:.4f}")

# R2越大越好，算提升率
R2_improvement = ((corrected_R2 - original_R2) / abs(original_R2)) * 100
# 误差越小越好，算减少率
mse_reduction = ((original_mse - corrected_mse) / original_mse) * 100
rmse_reduction = ((original_rmse - corrected_rmse) / original_rmse) * 100
mae_reduction = ((original_mae - corrected_mae) / original_mae) * 100

print(f"\nR2 Improvement: {R2_improvement:.2f}%")
print(f"MSE Reduction: {mse_reduction:.2f}%")
print(f"RMSE Reduction: {rmse_reduction:.2f}%")
print(f"MAE Reduction: {mae_reduction:.2f}%")


# In[ ]:





# In[41]:


import pandas as pd
from sklearn.preprocessing import MinMaxScaler

y_pre_gru = point_prediction_error_df.iloc[:, 6].to_frame(name='y_pre_gru')
spring_gru_error = point_prediction_error_df.iloc[:, 7].to_frame(name='spring_gru_error')
y_spring_test = point_prediction_error_df.iloc[:, -1].to_frame(name='y_spring_test')

# 只用训练集fit scaler
scaler = MinMaxScaler()
scaler.fit(train_gru_error.reshape(-1, 1))

y_pre_gru_normalized = pd.DataFrame(scaler.transform(y_pre_gru), columns=y_pre_gru.columns, index=y_pre_gru.index)
spring_gru_error_normalized = pd.DataFrame(scaler.transform(spring_gru_error), columns=spring_gru_error.columns, index=spring_gru_error.index)
y_spring_test_normalized = pd.DataFrame(scaler.transform(y_spring_test), columns=y_spring_test.columns, index=y_spring_test.index)


# In[42]:


import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from scipy.linalg import svd, inv

np.random.seed(0)

def build_hankel(series, m):
    N = len(series)
    H = np.zeros((m, N - m + 1))
    for i in range(m):
        H[i, :] = series[i:i + N - m + 1]
    return H

def dmd_fit(X_train, rank):
    X1 = X_train[:, :-1]
    X2 = X_train[:, 1:]
    U, S, Vh = svd(X1, full_matrices=False)
    U_r = U[:, :rank]
    S_r = np.diag(S[:rank])
    Vh_r = Vh[:rank, :]
    A_tilde = U_r.T @ X2 @ Vh_r.T @ inv(S_r)
    eigenvalues, eigenvectors = np.linalg.eig(A_tilde)
    phi = X2 @ Vh_r.T @ inv(S_r) @ eigenvectors
    return phi, eigenvalues

def dmd_step_predict(phi, eigenvalues, current_state):
    proj = np.linalg.pinv(phi) @ current_state
    next_proj = eigenvalues * proj
    return phi @ next_proj

#归一化
scaler = MinMaxScaler()
scaler.fit(train_gru_error.reshape(-1, 1))  # train_gru_error替换成你的训练集残差

train_error_norm = scaler.transform(train_gru_error.reshape(-1, 1)).flatten()
test_error_norm = scaler.transform(spring_gru_error.values.reshape(-1, 1)).flatten()
y_pre_norm = scaler.transform(y_pre_gru.values.reshape(-1, 1)).flatten()
y_test_norm = scaler.transform(y_spring_test.values.reshape(-1, 1)).flatten()

#训练集拟合DMD
m, r = 12, 5
X_train_hankel = build_hankel(train_error_norm, m)
phi, eigenvalues = dmd_fit(X_train_hankel, rank=r)

#测试集因果修正
n_test = len(test_error_norm)
corrected_test_error = np.zeros(n_test)
corrected_test_error[:m] = test_error_norm[:m]
correction_strength = 0.2

for t in range(m, n_test):
    current_state = test_error_norm[t - m:t]
    dmd_pred = dmd_step_predict(phi, eigenvalues, current_state)
    corrected_test_error[t] = test_error_norm[t] + correction_strength * dmd_pred[-1]

# 修正后预测值
corrected_predictions_norm = y_pre_norm + corrected_test_error
corrected_predictions = scaler.inverse_transform(corrected_predictions_norm.reshape(-1, 1)).flatten()

# 画图
plt.figure(figsize=(15, 10))
plt.subplot(2, 1, 1)
plt.plot(y_test_norm, label='True Values', alpha=0.9)
plt.plot(y_pre_norm, label='Original Predictions', alpha=0.7)
plt.title("Original vs True Values")
plt.legend()
plt.subplot(2, 1, 2)
plt.plot(y_test_norm, label='True Values', linestyle='--', alpha=0.9)
plt.plot(corrected_predictions_norm, label='Corrected Predictions', alpha=0.7)
plt.title("Corrected vs True Values")
plt.legend()
plt.tight_layout()
plt.show()


# In[43]:


from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
# 原始预测指标
original_R2_gru = r2_score(y_spring_test.values, y_pre_gru.values)
original_gru_mse = mean_squared_error(y_spring_test.values, y_pre_gru.values)
original_gru_rmse = np.sqrt(original_gru_mse)
original_gru_mae = mean_absolute_error(y_spring_test.values, y_pre_gru.values)

# 修正后预测指标
corrected_R2_gru = r2_score(y_spring_test.values, corrected_gru_predictions)
corrected_gru_mse = mean_squared_error(y_spring_test.values, corrected_gru_predictions)
corrected_gru_rmse = np.sqrt(corrected_gru_mse)
corrected_gru_mae = mean_absolute_error(y_spring_test.values, corrected_gru_predictions)

print(f"Original - R2: {original_R2_gru:.4f}, MSE: {original_gru_mse:.4f}, RMSE: {original_gru_rmse:.4f}, MAE: {original_gru_mae:.4f}")
print(f"Corrected - R2: {corrected_R2_gru:.4f}, MSE: {corrected_gru_mse:.4f}, RMSE: {corrected_gru_rmse:.4f}, MAE: {corrected_gru_mae:.4f}")

# R2越大越好，算提升率
R2_gru_improvement = ((corrected_R2_gru - original_R2_gru) / abs(original_R2_gru)) * 100
# 误差指标越小越好，算减少率
mse_gru_reduction = ((original_gru_mse - corrected_gru_mse) / original_gru_mse) * 100
rmse_gru_reduction = ((original_gru_rmse - corrected_gru_rmse) / original_gru_rmse) * 100
mae_gru_reduction = ((original_gru_mae - corrected_gru_mae) / original_gru_mae) * 100

print(f"\nR2 Improvement: {R2_gru_improvement:.2f}%")
print(f"MSE Reduction: {mse_gru_reduction:.2f}%")
print(f"RMSE Reduction: {rmse_gru_reduction:.2f}%")
print(f"MAE Reduction: {mae_gru_reduction:.2f}%")


# In[ ]:





# In[44]:


import pandas as pd
from sklearn.preprocessing import MinMaxScaler

# 提取对应列
y_pre_rnn = point_prediction_error_df.iloc[:, 8].to_frame(name='y_pre_rnn')
spring_rnn_error = point_prediction_error_df.iloc[:, 9].to_frame(name='spring_rnn_error')
y_spring_test = point_prediction_error_df.iloc[:, -1].to_frame(name='y_spring_test')

# 测试集归一化
scaler = MinMaxScaler()
scaler.fit(train_rnn_error.reshape(-1, 1))

# 统一使用训练集scaler归一化所有测试数据
y_pre_rnn_normalized = pd.DataFrame(
    scaler.transform(y_pre_rnn),
    columns=y_pre_rnn.columns,
    index=y_pre_rnn.index
)
spring_rnn_error_normalized = pd.DataFrame(
    scaler.transform(spring_rnn_error),
    columns=spring_rnn_error.columns,
    index=spring_rnn_error.index
)
y_spring_test_normalized = pd.DataFrame(
    scaler.transform(y_spring_test),
    columns=y_spring_test.columns,
    index=y_spring_test.index
)


# In[45]:


import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from scipy.linalg import svd, inv

np.random.seed(0)

# 标准DMD工具函数（仅训练集拟合）
def build_hankel(series, m):
    N = len(series)
    H = np.zeros((m, N - m + 1))
    for i in range(m):
        H[i, :] = series[i:i + N - m + 1]
    return H

def dmd_fit(X_train, rank):
    X1 = X_train[:, :-1]
    X2 = X_train[:, 1:]
    U, S, Vh = svd(X1, full_matrices=False)
    U_r = U[:, :rank]
    S_r = np.diag(S[:rank])
    Vh_r = Vh[:rank, :]
    A_tilde = U_r.T @ X2 @ Vh_r.T @ inv(S_r)
    eigenvalues, eigenvectors = np.linalg.eig(A_tilde)
    phi = X2 @ Vh_r.T @ inv(S_r) @ eigenvectors
    return phi, eigenvalues

def dmd_step_predict(phi, eigenvalues, current_state):
    proj = np.linalg.pinv(phi) @ current_state
    next_proj = eigenvalues * proj
    return phi @ next_proj

# 归一化
scaler = MinMaxScaler()
scaler.fit(train_rnn_error.reshape(-1, 1))

train_error_norm = scaler.transform(train_rnn_error.reshape(-1, 1)).flatten()
test_error_norm = scaler.transform(spring_rnn_error.values.reshape(-1, 1)).flatten()
y_pre_norm = scaler.transform(y_pre_rnn.values.reshape(-1, 1)).flatten()
y_test_norm = scaler.transform(y_spring_test.values.reshape(-1, 1)).flatten()

# 拟合DMD算子
m = 12   # 嵌入维度
r = 5    # 截断秩，和论文保持一致
X_train_hankel = build_hankel(train_error_norm)
phi, eigenvalues = dmd_fit(X_train_hankel, rank=r)

# 逐点因果修正
n_test = len(test_error_norm)
corrected_test_error = np.zeros(n_test)
corrected_test_error[:m] = test_error_norm[:m]
correction_strength = 0.2

for t in range(m, n_test):
    current_state = test_error_norm[t - m:t]
    dmd_pred = dmd_step_predict(phi, eigenvalues, current_state)
    corrected_test_error[t] = test_error_norm[t] + correction_strength * dmd_pred[-1]

corrected_predictions_norm = y_pre_norm + corrected_test_error
corrected_predictions = scaler.inverse_transform(corrected_predictions_norm.reshape(-1, 1)).flatten()

plt.figure(figsize=(15, 10))
plt.subplot(2, 1, 1)
plt.title("Original vs True Values")
plt.plot(y_test_norm, label='True Values', alpha=0.9)
plt.plot(y_pre_norm, label='Original Predictions', alpha=0.7)
plt.legend()
plt.xlabel("Time")
plt.ylabel("Values")

plt.subplot(2, 1, 2)
plt.title("Corrected vs True Values")
plt.plot(y_test_norm, label='True Values', linestyle='--', alpha=0.9)
plt.plot(corrected_predictions_norm, label='Corrected Predictions', alpha=0.7)
plt.legend()
plt.xlabel("Time")
plt.ylabel("Values")
plt.tight_layout()
plt.show()


# In[46]:


from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import numpy as np

original_R2_rnn = r2_score(y_spring_test.values, y_pre_rnn.values)
original_rnn_mse = mean_squared_error(y_spring_test.values, y_pre_rnn.values)
original_rnn_rmse = np.sqrt(original_rnn_mse)
original_rnn_mae = mean_absolute_error(y_spring_test.values, y_pre_rnn.values)

corrected_R2_rnn = r2_score(y_spring_test.values, corrected_predictions)
corrected_rnn_mse = mean_squared_error(y_spring_test.values, corrected_predictions)
corrected_rnn_rmse = np.sqrt(corrected_rnn_mse)
corrected_rnn_mae = mean_absolute_error(y_spring_test.values, corrected_predictions)

print(f"Original - R2: {original_R2_rnn:.4f}, MSE: {original_rnn_mse:.4f}, RMSE: {original_rnn_rmse:.4f}, MAE: {original_rnn_mae:.4f}")
print(f"Corrected - R2: {corrected_R2_rnn:.4f}, MSE: {corrected_rnn_mse:.4f}, RMSE: {corrected_rnn_rmse:.4f}, MAE: {corrected_rnn_mae:.4f}")

# R2越大越好，计算提升比例；误差指标越小越好，计算下降比例
R2_rnn_improvement = ((corrected_R2_rnn - original_R2_rnn) / abs(original_R2_rnn)) * 100
mse_rnn_reduction = ((original_rnn_mse - corrected_rnn_mse) / original_rnn_mse) * 100
rmse_rnn_reduction = ((original_rnn_rmse - corrected_rnn_rmse) / original_rnn_rmse) * 100
mae_rnn_reduction = ((original_rnn_mae - corrected_rnn_mae) / original_rnn_mae) * 100

print(f"\nR2 Improvement: {R2_rnn_improvement:.2f}%")
print(f"MSE Reduction: {mse_rnn_reduction:.2f}%")
print(f"RMSE Reduction: {rmse_rnn_reduction:.2f}%")
print(f"MAE Reduction: {mae_rnn_reduction:.2f}%")


# In[ ]:





# In[48]:


#将DMD修正后的预测值进行反归一化
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import pandas as pd
import os

# 对归一化后的预测值进行反归一化
corrected_itransformer_predictions = scaler.inverse_transform(corrected_itransformer_predictions_normalized)
corrected_lstm_predictions = scaler.inverse_transform(corrected_lstm_predictions_normalized)
corrected_tcn_predictions = scaler.inverse_transform(corrected_tcn_predictions_normalized)
corrected_gru_predictions = scaler.inverse_transform(corrected_gru_predictions_normalized)
corrected_rnn_predictions = scaler.inverse_transform(corrected_rnn_predictions_normalized)

y_pre_itransformer_dmd = corrected_itransformer_predictions.flatten()
y_pre_lstm_dmd = corrected_lstm_predictions.flatten()
y_pre_tcn_dmd = corrected_tcn_predictions.flatten()
y_pre_gru_dmd = corrected_gru_predictions.flatten()
y_pre_rnn_dmd = corrected_rnn_predictions.flatten()

# 创建一个DataFrame
dmd_df = pd.DataFrame({
    'itranformer_DMD': y_pre_itransformer_dmd,
    'LSTM_DMD': y_pre_lstm_dmd,
    'TCN_DMD': y_pre_tcn_dmd,
    'GRU_DMD': y_pre_gru_dmd,
    'RNN_DMD': y_pre_rnn_dmd,
    'True_power': y_spring_test.values.flatten()
})

# 将合并后的 DataFrame 保存到 Excel 文件中
directory = 'D:\\article'
if not os.path.exists(directory):
    os.makedirs(directory)

file_path = os.path.join(directory, 'DMD_Corrected_Predictions.xlsx')
dmd_df.to_excel(file_path, index=False)

# 显示 DataFrame
dmd_df


# In[49]:


#电力调度成本缩减
#导入文件
dispatch_data = pd.read_excel(r'D:\article\DMD_Corrected_Predictions2.xlsx', sheet_name=None)
dispatch_df = pd.concat(dispatch_data.values(), ignore_index=True)
dispatch_df


# In[ ]:





# In[50]:


#itransformer电力调度
import pulp
import pandas as pd

# 储能系统的成本和收益
storage_initial_investment = 500  # 初始投资成本（$/kWh）
storage_degradation_maintenance = 0.05 * storage_initial_investment  # 折旧和维护成本（$/kWh/year）

# 创建一个空列表来存储Total Cost结果
total_costs_itransformer = []

# 遍历dispatch_df中的每一行数据
for index, row in dispatch_df.iterrows():
    # 定义问题
    problem = pulp.LpProblem("Power_Dispatch_Optimization_with_Storage", pulp.LpMinimize)

    # 决策变量
    wind_power = pulp.LpVariable('WindPower', lowBound=row['itransformer_Lower_Bound'], upBound=row['itransformer_Upper_Bound'], cat='Continuous')
    renewable_power = pulp.LpVariable('RenewablePower', lowBound=0, upBound=row.iloc[-2] * 0.5 -row['y_pre_itransformer'], cat='Continuous')
    thermal_power = pulp.LpVariable('ThermalPower', lowBound=0, upBound=row.iloc[-2] * 0.5, cat='Continuous')
    energy_discharge = pulp.LpVariable('EnergyDischarge', lowBound=0, upBound=100, cat='Continuous')
    energy_charge = pulp.LpVariable('EnergyCharge', lowBound=0, upBound=100, cat='Continuous')

    # 目标函数：最小化总成本，包括储能成本
    objective = (row['wind_power_cost'] * wind_power + row['renewable_energy_cost'] * renewable_power +
                row['coal_power_cost'] * thermal_power+row['storage_discharge_cost'] * energy_discharge +
                row['storage_charge_cost'] * energy_charge + storage_degradation_maintenance)
    problem += objective, "Total Cost"

    # 约束条件
    problem += row['y_spring_test'] + renewable_power + thermal_power +energy_charge  == row.iloc[-2]+energy_discharge, "Total Demand with Storage"
    problem +=row['y_pre_itransformer'] + renewable_power + thermal_power == row.iloc[-2]

    # 解决问题
    problem.solve()

    # 将Total Cost添加到列表中
    total_costs_itransformer.append(pulp.value(problem.objective))

#     # 输出决策变量的最优解
#     print(f"Row {index}:")
#     print(f"Wind Power Optimal Value: {pulp.value(wind_power)}")
#     print(f"Renewable Power Optimal Value: {pulp.value(renewable_power)}")
#     print(f"Thermal Power Optimal Value: {pulp.value(thermal_power)}")
#     print(f"Energy Discharge Optimal Value: {pulp.value(energy_discharge)}")
#     print(f"Energy Charge Optimal Value: {pulp.value(energy_charge)}\n")

# 创建一个DataFrame来存储Total Cost结果
total_costs_itransformer_df = pd.DataFrame(total_costs_itransformer, columns=["itransformer Cost"])

# 输出结果DataFrame
print(total_costs_itransformer_df)


# In[52]:


# ===================== 新增：Excel保存功能 =====================
# 确保保存目录存在
directory = 'D:\\article'
if not os.path.exists(directory):
    os.makedirs(directory)

# 定义Excel文件保存路径
file_path = os.path.join(directory, 'iTransformer_Power_Dispatch_Total_Costs.xlsx')

# 保存DataFrame到Excel
total_costs_itransformer_df.to_excel(file_path, index=False)

# 输出保存结果
print(f"✅ iTransformer电力调度成本结果已保存至：{file_path}")


# In[51]:


#itransformer_dmd电力调度
import pulp
import pandas as pd

# 储能系统的成本和收益
storage_initial_investment = 500  # 初始投资成本（$/kWh）
storage_degradation_maintenance = 0.05 * storage_initial_investment  # 折旧和维护成本（$/kWh/year）

# 创建一个空列表来存储Total Cost结果
total_costs_itransformer_dmd = []

# 遍历dispatch_df中的每一行数据
for index, row in dispatch_df.iterrows():
    # 定义问题
    problem = pulp.LpProblem("Power_Dispatch_Optimization_with_Storage", pulp.LpMinimize)

    # 决策变量
    wind_power = pulp.LpVariable('WindPower', lowBound=row['itransformer_Lower_Bound'], upBound=row['itransformer_Upper_Bound'], cat='Continuous')
    renewable_power = pulp.LpVariable('RenewablePower', lowBound=0, upBound=row.iloc[-2] * 0.5 -row['itransformer_DMD'], cat='Continuous')
    thermal_power = pulp.LpVariable('ThermalPower', lowBound=0, upBound=row.iloc[-2] * 0.5, cat='Continuous')
    energy_discharge = pulp.LpVariable('EnergyDischarge', lowBound=0, upBound=100, cat='Continuous')
    energy_charge = pulp.LpVariable('EnergyCharge', lowBound=0, upBound=100, cat='Continuous')

    # 目标函数：最小化总成本，包括储能成本
    objective = (row['wind_power_cost'] * wind_power + row['renewable_energy_cost'] * renewable_power +
                row['coal_power_cost'] * thermal_power+row['storage_discharge_cost'] * energy_discharge +
                row['storage_charge_cost'] * energy_charge + storage_degradation_maintenance)
    problem += objective, "Total Cost"

    # 约束条件
    problem +=row['y_spring_test'] + renewable_power + thermal_power +energy_charge  == row.iloc[-2]+energy_discharge, "Total Demand with Storage"
    problem +=row['itransformer_DMD'] + renewable_power + thermal_power == row.iloc[-2]
   # problem +=renewable_power +row['y_spring_test']== row.iloc[-2]*0.5

    # 解决问题
    problem.solve()

    # 将Total Cost添加到列表中
    total_costs_itransformer_dmd.append(pulp.value(problem.objective))

#     # 输出决策变量的最优解
#     print(f"Row {index}:")
#     print(f"Wind Power Optimal Value: {pulp.value(wind_power)}")
#     print(f"Renewable Power Optimal Value: {pulp.value(renewable_power)}")
#     print(f"Thermal Power Optimal Value: {pulp.value(thermal_power)}")
#     print(f"Energy Discharge Optimal Value: {pulp.value(energy_discharge)}")
#     print(f"Energy Charge Optimal Value: {pulp.value(energy_charge)}\n")

# 创建一个DataFrame来存储Total Cost结果
total_costs_itransformer_dmd = pd.DataFrame(total_costs_itransformer_dmd, columns=["itransformer Cost"])

# 输出结果DataFrame
print(total_costs_itransformer_dmd)


# In[53]:


# ===================== 新增：Excel保存功能 =====================
# 确保保存目录存在
directory = 'D:\\article'
if not os.path.exists(directory):
    os.makedirs(directory)

# 定义Excel文件保存路径
file_path = os.path.join(directory, 'total_costs_itransformer_dmd.xlsx')

# 保存DataFrame到Excel
total_costs_itransformer_dmd.to_excel(file_path, index=False)

# 输出保存结果
print(f"✅ iTransformer_DMD电力调度成本结果已保存至：{file_path}")


# In[ ]:





# In[ ]:





# In[14]:


#lstm电力调度
import pulp
import pandas as pd

# 储能系统的成本和收益
storage_initial_investment = 500  # 初始投资成本（$/kWh）
storage_degradation_maintenance = 0.05 * storage_initial_investment  # 折旧和维护成本（$/kWh/year）

# 创建一个空列表来存储Total Cost结果
total_costs_LSTM = []

# 遍历dispatch_df中的每一行数据
for index, row in dispatch_df.iterrows():
    # 定义问题
    problem = pulp.LpProblem("Power_Dispatch_Optimization_with_Storage", pulp.LpMinimize)

    # 决策变量
    wind_power = pulp.LpVariable('WindPower', lowBound=row['LSTM_Lower_Bound'], upBound=row['LSTM_Upper_Bound'], cat='Continuous')
    renewable_power = pulp.LpVariable('RenewablePower', lowBound=0, upBound=row.iloc[-2] * 0.5 -row['y_pre_lstm'], cat='Continuous')
    thermal_power = pulp.LpVariable('ThermalPower', lowBound=0, upBound=row.iloc[-2] * 0.5, cat='Continuous')
    energy_discharge = pulp.LpVariable('EnergyDischarge', lowBound=0, upBound=100, cat='Continuous')
    energy_charge = pulp.LpVariable('EnergyCharge', lowBound=0, upBound=100, cat='Continuous')

    # 目标函数：最小化总成本，包括储能成本
    objective = (row['wind_power_cost'] * wind_power + row['renewable_energy_cost'] * renewable_power +
                row['coal_power_cost'] * thermal_power+row['storage_discharge_cost'] * energy_discharge +
                row['storage_charge_cost'] * energy_charge + storage_degradation_maintenance)
    problem += objective, "Total Cost"

    # 约束条件
    problem += row['y_spring_test'] + renewable_power + thermal_power +energy_charge  == row.iloc[-2]+energy_discharge, "Total Demand with Storage"
    problem +=row['y_pre_lstm'] + renewable_power + thermal_power == row.iloc[-2]

    # 解决问题
    problem.solve()

    # 将Total Cost添加到列表中
    total_costs_LSTM.append(pulp.value(problem.objective))

#     # 输出决策变量的最优解
#     print(f"Row {index}:")
#     print(f"Wind Power Optimal Value: {pulp.value(wind_power)}")
#     print(f"Renewable Power Optimal Value: {pulp.value(renewable_power)}")
#     print(f"Thermal Power Optimal Value: {pulp.value(thermal_power)}")
#     print(f"Energy Discharge Optimal Value: {pulp.value(energy_discharge)}")
#     print(f"Energy Charge Optimal Value: {pulp.value(energy_charge)}\n")

# 创建一个DataFrame来存储Total Cost结果
total_costs_LSTM_df = pd.DataFrame(total_costs_LSTM, columns=["LSTM Cost"])

# 输出结果DataFrame
print(total_costs_LSTM_df)


# In[15]:


#lstm_dmd电力调度
import pulp
import pandas as pd

# 储能系统的成本和收益
storage_initial_investment = 500  # 初始投资成本（$/kWh）
storage_degradation_maintenance = 0.05 * storage_initial_investment  # 折旧和维护成本（$/kWh/year）

# 创建一个空列表来存储Total Cost结果
total_costs_LSTM_dmd = []

# 遍历dispatch_df中的每一行数据
for index, row in dispatch_df.iterrows():
    # 定义问题
    problem = pulp.LpProblem("Power_Dispatch_Optimization_with_Storage", pulp.LpMinimize)

    # 决策变量
    wind_power = pulp.LpVariable('WindPower', lowBound=row['LSTM_Lower_Bound'], upBound=row['LSTM_Upper_Bound'], cat='Continuous')
    renewable_power = pulp.LpVariable('RenewablePower', lowBound=0, upBound=row.iloc[-2] * 0.5 -row['LSTM_DMD'], cat='Continuous')
    thermal_power = pulp.LpVariable('ThermalPower', lowBound=0, upBound=row.iloc[-2] * 0.5, cat='Continuous')
    energy_discharge = pulp.LpVariable('EnergyDischarge', lowBound=0, upBound=100, cat='Continuous')
    energy_charge = pulp.LpVariable('EnergyCharge', lowBound=0, upBound=100, cat='Continuous')

    # 目标函数：最小化总成本，包括储能成本
    objective = (row['wind_power_cost'] * wind_power + row['renewable_energy_cost'] * renewable_power +
                row['coal_power_cost'] * thermal_power+row['storage_discharge_cost'] * energy_discharge +
                row['storage_charge_cost'] * energy_charge + storage_degradation_maintenance)
    problem += objective, "Total Cost"

    # 约束条件
    problem +=row['y_spring_test'] + renewable_power + thermal_power +energy_charge  == row.iloc[-2]+energy_discharge, "Total Demand with Storage"
    problem +=row['LSTM_DMD'] + renewable_power + thermal_power == row.iloc[-2]
   # problem +=renewable_power +row['y_spring_test']== row.iloc[-2]*0.5

    # 解决问题
    problem.solve()

    # 将Total Cost添加到列表中
    total_costs_LSTM_dmd.append(pulp.value(problem.objective))

#     # 输出决策变量的最优解
#     print(f"Row {index}:")
#     print(f"Wind Power Optimal Value: {pulp.value(wind_power)}")
#     print(f"Renewable Power Optimal Value: {pulp.value(renewable_power)}")
#     print(f"Thermal Power Optimal Value: {pulp.value(thermal_power)}")
#     print(f"Energy Discharge Optimal Value: {pulp.value(energy_discharge)}")
#     print(f"Energy Charge Optimal Value: {pulp.value(energy_charge)}\n")

# 创建一个DataFrame来存储Total Cost结果
total_costs_LSTM_dmd = pd.DataFrame(total_costs_LSTM_dmd, columns=["LSTM Cost"])

# 输出结果DataFrame
print(total_costs_LSTM_dmd)


# In[ ]:





# In[16]:


#TCN 电力调度
import pulp
import pandas as pd

# 储能系统的成本和收益
storage_initial_investment = 500  # 初始投资成本（$/kWh）
storage_degradation_maintenance = 0.05 * storage_initial_investment  # 折旧和维护成本（$/kWh/year）

# 创建一个空列表来存储Total Cost结果
total_costs_TCN  = []

# 遍历dispatch_df中的每一行数据
for index, row in dispatch_df.iterrows():
    # 定义问题
    problem = pulp.LpProblem("Power_Dispatch_Optimization_with_Storage", pulp.LpMinimize)

    # 决策变量
    wind_power = pulp.LpVariable('WindPower', lowBound=row['TCN_Lower_Bound'], upBound=row['TCN_Upper_Bound'], cat='Continuous')
    renewable_power = pulp.LpVariable('RenewablePower', lowBound=0, upBound=row.iloc[-2] * 0.5 -row['y_pre_tcn'], cat='Continuous')
    thermal_power = pulp.LpVariable('ThermalPower', lowBound=0, upBound=row.iloc[-2] * 0.5, cat='Continuous')
    energy_discharge = pulp.LpVariable('EnergyDischarge', lowBound=0, upBound=100, cat='Continuous')
    energy_charge = pulp.LpVariable('EnergyCharge', lowBound=0, upBound=100, cat='Continuous')

    # 目标函数：最小化总成本，包括储能成本
    objective = (row['wind_power_cost'] * wind_power + row['renewable_energy_cost'] * renewable_power +
                row['coal_power_cost'] * thermal_power+row['storage_discharge_cost'] * energy_discharge +
                row['storage_charge_cost'] * energy_charge + storage_degradation_maintenance)
    problem += objective, "Total Cost"

    # 约束条件
    problem += row['y_spring_test'] + renewable_power + thermal_power +energy_charge  == row.iloc[-2]+energy_discharge, "Total Demand with Storage"
    problem +=row['y_pre_tcn'] + renewable_power + thermal_power == row.iloc[-2]
   # problem +=renewable_power +row['y_spring_test']== row.iloc[-2]*0.5

    # 解决问题
    problem.solve()

    # 将Total Cost添加到列表中
    total_costs_TCN .append(pulp.value(problem.objective))

total_costs_TCN  = pd.DataFrame(total_costs_TCN , columns=["TCN Cost"])

# 输出结果DataFrame
print(total_costs_TCN )


# In[17]:


#TCN_DMD电力调度
import pulp
import pandas as pd

# 储能系统的成本和收益
storage_initial_investment = 500  # 初始投资成本（$/kWh）
storage_degradation_maintenance = 0.05 * storage_initial_investment  # 折旧和维护成本（$/kWh/year）

# 创建一个空列表来存储Total Cost结果
total_costs_TCN_dmd = []

# 遍历dispatch_df中的每一行数据
for index, row in dispatch_df.iterrows():
    # 定义问题
    problem = pulp.LpProblem("Power_Dispatch_Optimization_with_Storage", pulp.LpMinimize)

    # 决策变量
    wind_power = pulp.LpVariable('WindPower', lowBound=row['TCN_Lower_Bound'], upBound=row['TCN_Upper_Bound'], cat='Continuous')
    renewable_power = pulp.LpVariable('RenewablePower', lowBound=0, upBound=row.iloc[-2] * 0.5 -row['TCN_DMD'], cat='Continuous')
    thermal_power = pulp.LpVariable('ThermalPower', lowBound=0, upBound=row.iloc[-2] * 0.5, cat='Continuous')
    energy_discharge = pulp.LpVariable('EnergyDischarge', lowBound=0, upBound=100, cat='Continuous')
    energy_charge = pulp.LpVariable('EnergyCharge', lowBound=0, upBound=100, cat='Continuous')

    # 目标函数：最小化总成本，包括储能成本
    objective = (row['wind_power_cost'] * wind_power + row['renewable_energy_cost'] * renewable_power +
                row['coal_power_cost'] * thermal_power+row['storage_discharge_cost'] * energy_discharge +
                row['storage_charge_cost'] * energy_charge + storage_degradation_maintenance)
    problem += objective, "Total Cost"

    # 约束条件
    problem += row['y_spring_test'] + renewable_power + thermal_power +energy_charge  == row.iloc[-2]+energy_discharge, "Total Demand with Storage"
    problem +=row['TCN_DMD'] + renewable_power + thermal_power == row.iloc[-2]
    #problem +=renewable_power +row['y_spring_test']== row.iloc[-2]*0.5

    # 解决问题
    problem.solve()

    # 将Total Cost添加到列表中
    total_costs_TCN_dmd.append(pulp.value(problem.objective))

total_costs_TCN_dmd = pd.DataFrame(total_costs_TCN_dmd, columns=["TCN Cost"])

# 输出结果DataFrame
print(total_costs_TCN_dmd)


# In[ ]:





# In[18]:


#GRU 电力调度
import pulp
import pandas as pd

# 储能系统的成本和收益
storage_initial_investment = 500  # 初始投资成本（$/kWh）
storage_degradation_maintenance = 0.05 * storage_initial_investment  # 折旧和维护成本（$/kWh/year）

# 创建一个空列表来存储Total Cost结果
total_costs_GRU  = []

# 遍历dispatch_df中的每一行数据
for index, row in dispatch_df.iterrows():
    # 定义问题
    problem = pulp.LpProblem("Power_Dispatch_Optimization_with_Storage", pulp.LpMinimize)

    # 决策变量
    wind_power = pulp.LpVariable('WindPower', lowBound=row['GRU_Lower_Bound'], upBound=row['GRU_Upper_Bound'], cat='Continuous')
    renewable_power = pulp.LpVariable('RenewablePower', lowBound=0, upBound=row.iloc[-2] * 0.5 -row['y_pre_gru'], cat='Continuous')
    thermal_power = pulp.LpVariable('ThermalPower', lowBound=0, upBound=row.iloc[-2] * 0.5, cat='Continuous')
    energy_discharge = pulp.LpVariable('EnergyDischarge', lowBound=0, upBound=100, cat='Continuous')
    energy_charge = pulp.LpVariable('EnergyCharge', lowBound=0, upBound=100, cat='Continuous')

    # 目标函数：最小化总成本，包括储能成本
    objective = (row['wind_power_cost'] * wind_power + row['renewable_energy_cost'] * renewable_power +
                row['coal_power_cost'] * thermal_power+row['storage_discharge_cost'] * energy_discharge +
                row['storage_charge_cost'] * energy_charge + storage_degradation_maintenance)
    problem += objective, "Total Cost"

    # 约束条件
    problem += row['y_spring_test'] + renewable_power + thermal_power +energy_charge  == row.iloc[-2]+energy_discharge, "Total Demand with Storage"
    problem +=row['y_pre_gru'] + renewable_power + thermal_power == row.iloc[-2]
    #problem +=renewable_power +row['y_spring_test']== row.iloc[-2]*0.5

    # 解决问题
    problem.solve()

    # 将Total Cost添加到列表中
    total_costs_GRU .append(pulp.value(problem.objective))

total_costs_GRU  = pd.DataFrame(total_costs_GRU , columns=["GRU Cost"])

# 输出结果DataFrame
print(total_costs_GRU )


# In[19]:


#GRU_DMD电力调度
import pulp
import pandas as pd

# 储能系统的成本和收益
storage_initial_investment = 500  # 初始投资成本（$/kWh）
storage_degradation_maintenance = 0.05 * storage_initial_investment  # 折旧和维护成本（$/kWh/year）

# 创建一个空列表来存储Total Cost结果
total_costs_GRU_dmd = []

# 遍历dispatch_df中的每一行数据
for index, row in dispatch_df.iterrows():
    # 定义问题
    problem = pulp.LpProblem("Power_Dispatch_Optimization_with_Storage", pulp.LpMinimize)

    # 决策变量
    wind_power = pulp.LpVariable('WindPower', lowBound=row['GRU_Lower_Bound'], upBound=row['GRU_Upper_Bound'], cat='Continuous')
    renewable_power = pulp.LpVariable('RenewablePower', lowBound=0, upBound=row.iloc[-2] * 0.5 -row['GRU_DMD'], cat='Continuous')
    thermal_power = pulp.LpVariable('ThermalPower', lowBound=0, upBound=row.iloc[-2] * 0.5, cat='Continuous')
    energy_discharge = pulp.LpVariable('EnergyDischarge', lowBound=0, upBound=100, cat='Continuous')
    energy_charge = pulp.LpVariable('EnergyCharge', lowBound=0, upBound=100, cat='Continuous')

    # 目标函数：最小化总成本，包括储能成本
    objective = (row['wind_power_cost'] * wind_power + row['renewable_energy_cost'] * renewable_power +
                row['coal_power_cost'] * thermal_power+row['storage_discharge_cost'] * energy_discharge +
                row['storage_charge_cost'] * energy_charge + storage_degradation_maintenance)
    problem += objective, "Total Cost"

    # 约束条件
    problem += row['y_spring_test'] + renewable_power + thermal_power +energy_charge  == row.iloc[-2]+energy_discharge, "Total Demand with Storage"
    problem +=row['GRU_DMD'] + renewable_power + thermal_power == row.iloc[-2]
    #problem +=renewable_power +row['y_spring_test']== row.iloc[-2]*0.5

    # 解决问题
    problem.solve()

    # 将Total Cost添加到列表中
    total_costs_GRU_dmd.append(pulp.value(problem.objective))

total_costs_GRU_dmd = pd.DataFrame(total_costs_GRU_dmd, columns=["GRU Cost"])

# 输出结果DataFrame
print(total_costs_GRU_dmd)


# In[ ]:





# In[20]:


#RNN 电力调度
import pulp
import pandas as pd

# 储能系统的成本和收益
storage_initial_investment = 500  # 初始投资成本（$/kWh）
storage_degradation_maintenance = 0.05 * storage_initial_investment  # 折旧和维护成本（$/kWh/year）

# 创建一个空列表来存储Total Cost结果
total_costs_RNN  = []

# 遍历dispatch_df中的每一行数据
for index, row in dispatch_df.iterrows():
    # 定义问题
    problem = pulp.LpProblem("Power_Dispatch_Optimization_with_Storage", pulp.LpMinimize)

    # 决策变量
    wind_power = pulp.LpVariable('WindPower', lowBound=row['RNN_Lower_Bound'], upBound=row['RNN_Upper_Bound'], cat='Continuous')
    renewable_power = pulp.LpVariable('RenewablePower', lowBound=0, upBound=row.iloc[-2] * 0.5 -row['y_pre_rnn'], cat='Continuous')
    thermal_power = pulp.LpVariable('ThermalPower', lowBound=0, upBound=row.iloc[-2] * 0.5, cat='Continuous')
    energy_discharge = pulp.LpVariable('EnergyDischarge', lowBound=0, upBound=100, cat='Continuous')
    energy_charge = pulp.LpVariable('EnergyCharge', lowBound=0, upBound=100, cat='Continuous')

    # 目标函数：最小化总成本，包括储能成本
    objective = (row['wind_power_cost'] * wind_power + row['renewable_energy_cost'] * renewable_power +
                row['coal_power_cost'] * thermal_power+row['storage_discharge_cost'] * energy_discharge +
                row['storage_charge_cost'] * energy_charge + storage_degradation_maintenance)
    problem += objective, "Total Cost"

    # 约束条件
    problem += row['y_spring_test'] + renewable_power + thermal_power +energy_charge  == row.iloc[-2]+energy_discharge, "Total Demand with Storage"
    problem +=row['y_pre_rnn'] + renewable_power + thermal_power == row.iloc[-2]
    #problem +=renewable_power +row['y_spring_test']== row.iloc[-2]*0.5

    # 解决问题
    problem.solve()

    # 将Total Cost添加到列表中
    total_costs_RNN .append(pulp.value(problem.objective))

total_costs_RNN  = pd.DataFrame(total_costs_RNN , columns=["RNN Cost"])

# 输出结果DataFrame
print(total_costs_RNN )


# In[21]:


#RNN_DMD电力调度
import pulp
import pandas as pd

# 储能系统的成本和收益
storage_initial_investment = 500  # 初始投资成本（$/kWh）
storage_degradation_maintenance = 0.05 * storage_initial_investment  # 折旧和维护成本（$/kWh/year）

# 创建一个空列表来存储Total Cost结果
total_costs_RNN_dmd = []

# 遍历dispatch_df中的每一行数据
for index, row in dispatch_df.iterrows():
    # 定义问题
    problem = pulp.LpProblem("Power_Dispatch_Optimization_with_Storage", pulp.LpMinimize)

    # 决策变量
    wind_power = pulp.LpVariable('WindPower', lowBound=row['RNN_Lower_Bound'], upBound=row['RNN_Upper_Bound'], cat='Continuous')
    renewable_power = pulp.LpVariable('RenewablePower', lowBound=0, upBound=row.iloc[-2] * 0.5 -row['RNN_DMD'], cat='Continuous')
    thermal_power = pulp.LpVariable('ThermalPower', lowBound=0, upBound=row.iloc[-2] * 0.5, cat='Continuous')
    energy_discharge = pulp.LpVariable('EnergyDischarge', lowBound=0, upBound=100, cat='Continuous')
    energy_charge = pulp.LpVariable('EnergyCharge', lowBound=0, upBound=100, cat='Continuous')

    # 目标函数：最小化总成本，包括储能成本
    objective = (row['wind_power_cost'] * wind_power + row['renewable_energy_cost'] * renewable_power +
                row['coal_power_cost'] * thermal_power+row['storage_discharge_cost'] * energy_discharge +
                row['storage_charge_cost'] * energy_charge + storage_degradation_maintenance)
    problem += objective, "Total Cost"

    # 约束条件
    problem += row['y_spring_test'] + renewable_power + thermal_power +energy_charge  == row.iloc[-2]+energy_discharge, "Total Demand with Storage"
    problem +=row['RNN_DMD'] + renewable_power + thermal_power == row.iloc[-2]

    # 解决问题
    problem.solve()

    # 将Total Cost添加到列表中
    total_costs_RNN_dmd.append(pulp.value(problem.objective))

total_costs_RNN_dmd = pd.DataFrame(total_costs_RNN_dmd, columns=["RNN Cost"])

# 输出结果DataFrame
print(total_costs_RNN_dmd)


# In[23]:


import pandas as pd
# 合并DataFrame
combined_df = pd.concat([total_costs_LSTM_dmd, total_costs_LSTM_df,total_costs_TCN_dmd, total_costs_TCN,total_costs_GRU_dmd, total_costs_GRU,total_costs_RNN_dmd, total_costs_RNN], axis=1)
# 为列命名，以便区分
combined_df.columns = ['LSTM_DMD Cost', 'LSTM Cost','TCN_DMD Cost', 'TCN Cost','GRU_DMD Cost', 'GRU Cost','RNN_DMD Cost', 'RNN Cost']
# 保存到Excel文件
output_file_path = r'D:\article\Combined_Total_Costs2.xlsx'
combined_df.to_excel(output_file_path, index=False)


# In[ ]:




