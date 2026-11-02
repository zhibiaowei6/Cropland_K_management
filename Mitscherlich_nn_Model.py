#!/usr/bin/env python
# coding: utf-8

# In[3]:


import torch
import torch.nn as nn
import torch.optim as optim

import pandas as pd
import numpy as np  

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

import matplotlib.pyplot as plt


# In[2]:


import torch.nn.functional as F

# from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MaxAbsScaler
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from sklearn import set_config


# In[3]:


import os
print(os.getcwd())
#os.chdir('D:/0_PhdData/0_Paper/6_Fertilizer_P/K_test/') 需要用的数据在当前活动路径中，因而无需修改


# In[4]:


# 设置为 NumPy 输出（默认行为）
set_config(transform_output="default")  # 或直接不调用 set_config
# %% 设置随机数种子
raw_seed = 123
def setSeed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
setSeed(raw_seed)


# In[5]:


# 1. 检查CUDA可用性
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")


# In[6]:


# # 自定义物理约束层
# class PhysicsConstraintLayer(nn.Module):
#     def __init__(self, b_min=10, b_max=32):  # 控制参数b和a的范围
#         super(PhysicsConstraintLayer, self).__init__()
#         self.b_min = b_min
#         self.b_max = b_max
#         #self.a_min = a_min
#         #self.a_max = a_max

#     def forward(self, a, b, c, krate):
#         # 限制 a 的范围（使用 sigmoid）
#         #a_scaled = torch.sigmoid(a) * (self.a_max - self.a_min) + self.a_min
#         # 保证 b 不小于 b_lower_bound，增强低施钾敏感性
#         b_scaled = torch.sigmoid(b) * (self.b_max - self.b_min) + self.b_min
#         #b_scaled = torch.nn.functional.softplus(b) + self.b_min
#         #c_scaled = torch.sigmoid(c) * (self.c_max - self.c_min) + self.c_min  # 限制c为小值
#         return a * (1 - torch.exp(-(b_scaled * krate + c)))


# In[7]:


class PhysicsConstraintLayer(nn.Module):
    def __init__(self, b_min=8, b_max=45, c_min=0.5, c_max=1.3):
        super(PhysicsConstraintLayer, self).__init__()
        self.b_min = b_min
        self.b_max = b_max
        self.c_min = c_min
        self.c_max = c_max

    def forward(self, a, b, c, d, krate):
        b_scaled = torch.sigmoid(b) * (self.b_max - self.b_min) + self.b_min
        c_scaled = torch.sigmoid(c) * (self.c_max - self.c_min) + self.c_min

        exp_input = -b_scaled * krate
        exp_input_clamped = torch.clamp(exp_input, min=-60, max=0)

        base = 1 - torch.exp(exp_input_clamped)
        base_clamped = torch.clamp(base, min=1e-6)

        return a * (base_clamped ** c_scaled) + d



# In[8]:


# 设置模型
class PhysicsGuidedModel(nn.Module):
    def __init__(self):
        super(PhysicsGuidedModel, self).__init__()

        def make_network():
            return nn.Sequential(
                nn.Linear(25, 300),
                nn.BatchNorm1d(300),
                nn.SiLU(),
                nn.Dropout(0.1),
                nn.Linear(300, 128),
                nn.BatchNorm1d(128),
                nn.SiLU(),
                nn.Dropout(0.1),
                nn.Linear(128, 16),
                nn.BatchNorm1d(16),
                nn.SiLU(),
                nn.Linear(16, 1),
                nn.Softplus()  # 保证输出为正数  
            )

        self.a_network = make_network()
        self.b_network = make_network()
        self.c_network = make_network()
        self.d_network = make_network()
        self.physics_layer = PhysicsConstraintLayer()

    def forward(self, x, krate):
        a = self.a_network(x)  # [0,6000]
        b = self.b_network(x)  # [0,0.01]
        c = self.c_network(x)  # [1000,6000]
        d = self.d_network(x)
        yield_K = self.physics_layer(a, b, c, d, krate)
        return yield_K, a, b, c, d



# In[9]:


# 数据读取和预处理
df=pd.read_csv('Maize_smalldata_New5.csv', encoding='gbk')
order_name = ['Seasonal_Tem','Seasonal_Pre','Seasonal_ET0','Sand','Clay','BD','pH','OM','TN','Olsen_P','Exc_K','Irrigation','Density','Time','Approach','Type','Straw_return','K2O','Yield']
dt = df[order_name]
discrete_cols = ['Irrigation', 'Density', 'Time', 'Approach', 'Type', 'Straw_return']

dt = pd.get_dummies(dt, columns=discrete_cols, drop_first=False)  # drop_first=True 可防止共线性
print(dt.columns)
#for col in discrete_cols:
    #unique_vals = sorted(df[col].dropna().unique())
    #mapping = {val: 0.5 * i for i, val in enumerate(unique_vals)}     #为什么不用独热编码？？？
    #dt[col] = df[col].map(mapping)
#dt


# In[10]:


X = dt[['Seasonal_Tem', 'Seasonal_Pre', 'Seasonal_ET0', 'Sand', 'Clay', 'BD',
       'pH', 'OM', 'TN', 'Olsen_P', 'Exc_K', 'Irrigation_I0',
       'Irrigation_I1', 'Density_D0', 'Density_D1', 'Density_D2', 'Time_T1',
       'Time_T2', 'Approach_A0', 'Approach_A1', 'Type_Ty0',
       'Type_Ty1', 'Type_Ty2', 'Straw_return_S0', 'Straw_return_S1']].values.astype(np.float32)
Krates = dt['K2O'].values.reshape(-1,1).astype(np.float32)
Yield = dt['Yield'].values.reshape(-1,1).astype(np.float32)


# In[11]:


X_T = torch.tensor(X)     #将numpy数组转化为张量
Yield_T = torch.tensor(Yield)
Krates_T = torch.tensor(Krates)


# In[12]:


# 归一化函数
def custom_normalize(data, data_min, data_max):
    return (data - data_min) / (data_max - data_min+1e-8)  # 加上一个小的数值防止除以零

def custom_denormalize(data_normalized, data_min, data_max):
    return data_normalized * (data_max - data_min) + data_min


# In[13]:


# 获取归一化参数
X_T_min = X_T.min(dim=0, keepdim=True)[0]
X_T_max = X_T.max(dim=0, keepdim=True)[0]
X_gyh_T= custom_normalize(X_T,X_T_min,X_T_max)


Yield_T_min = Yield_T.min(dim=0, keepdim=True)[0]
Yield_T_max = Yield_T.max(dim=0, keepdim=True)[0]
Yield_gyh_T= custom_normalize(Yield_T,Yield_T_min,Yield_T_max)


Krates_T_min = Krates_T.min(dim=0, keepdim=True)[0]
Krates_T_max = Krates_T.max(dim=0, keepdim=True)[0]
Krates_gyh_T= custom_normalize(Krates_T,Krates_T_min, Krates_T_max)


# In[14]:


# 将数据集转移到cuda上
X_T_min = X_T_min.to(device)
X_T_max = X_T_max.to(device)
# 将数据集转移到cuda上
Yield_T_min = Yield_T_min.to(device)
Yield_T_max = Yield_T_max.to(device)
# 将数据集转移到cuda上
Krates_T_min = Krates_T_min.to(device)
Krates_T_max = Krates_T_max.to(device)


# In[15]:


# 划分数据集，测试集占30%
X_train, X_test, Yield_train, Yield_test, Krates_train, Krates_test = train_test_split(X_gyh_T, Yield_gyh_T, Krates_gyh_T,test_size=0.3, random_state=42)


# In[16]:


# 张量转到cuda上

data_x_train = X_train.requires_grad_(True).to(device)
data_x_test = X_test.to(device)

data_yield_train = Yield_train.to(device)
data_yield_test = Yield_test.to(device)


data_Krate_train = Krates_train.requires_grad_(True).to(device)
data_Krate_test = Krates_test.to(device)

print(data_x_train.device)
print(data_yield_train.device)
print(data_Krate_train.device)


# In[17]:


# 创建模型实例
model = PhysicsGuidedModel().to(device)

# 定义损失函数
criterion = nn.MSELoss()
from torch.optim.lr_scheduler import StepLR

# 优化器
optimizer = optim.Adam(model.parameters(), lr=0.001)

scheduler = StepLR(optimizer, step_size=500, gamma=0.45) #效果最佳最佳 epoch = 15888

# 模型训练
epochs = 2088
losses = []      # 记录损失函数变化的列表

for epoch in range(epochs):
    model.train()
    optimizer.zero_grad()
    # 前向传播
    yield_pred, a_pred, b_pred, c_pred, d_pred = model(data_x_train, data_Krate_train)

    # === 加权损失函数 ===
    # 针对低施钾（归一化后 krate < 0.5）的样本增加权重
    #low_krate_mask = (data_Krate_train < 0.3).float()
    #weights = 1.0 + 5.0 * low_krate_mask  # 权重倍数可调，比如这里是2倍
    #weighted_mse = (weights * (yield_pred - data_yield_train) ** 2).mean()

#     # === 加入 a 的惩罚项 ===
#     lambda_a = -0.03  # 惩罚权重，可以调节，如果惩罚效果不明显，可以尝试调大 lambda_a（比如 0.05, 0.1）；如果抑制过头了，让模型难以拟合数据，可以调小。
#     a_penalty = lambda_a * torch.mean(a_pred)

    #loss = weighted_mse 
    #+ a_penalty
    # 计算损失
    loss = criterion(yield_pred, data_yield_train)
    #+ a_penalty
    losses.append(loss.item())      # 记录损失函数的变化
    loss.backward()

    optimizer.step()
    scheduler.step()

    if epoch % 100 == 0:
        print(f'Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}')
        print(f'a_mean: {a_pred.mean().item():.2f}, b_mean: {b_pred.mean().item():.4f}, c_mean: {c_pred.mean().item():.2f}, d_mean: {d_pred.mean().item():.2f}')


# In[18]:


Fig = plt.figure()
plt.plot(range(epochs), losses)
plt.ylabel('loss')
plt.xlabel('epoch')
plt.show()


# In[19]:


# 开始模型评估模式
model.eval()
print("Model is now in evaluation mode.")


# In[20]:


# 在测试集和训练集上进行预测
with torch.no_grad():
    y_pred_test, _, _, _, _ = model(data_x_test, data_Krate_test)
    y_pred_train, _, _, _, _ = model(data_x_train, data_Krate_train)



# In[21]:


# 将预测值和真实值反归一化
y_pred_test = custom_denormalize(y_pred_test, Yield_T_min, Yield_T_max)
y_pred_train = custom_denormalize(y_pred_train, Yield_T_min, Yield_T_max)
y_test = custom_denormalize(data_yield_test, Yield_T_min, Yield_T_max)
y_train = custom_denormalize(data_yield_train, Yield_T_min, Yield_T_max)

# 将数据从 PyTorch 张量转换为 NumPy 数组a
y_pred_test_np = y_pred_test.cpu().numpy()
y_pred_train_np = y_pred_train.cpu().numpy()

y_test_np = y_test.cpu().numpy()
y_train_np = y_train.cpu().numpy()

# 计算 R2
r2_test = r2_score(y_test_np, y_pred_test_np)
r2_train = r2_score(y_train_np, y_pred_train_np)
print(f'R2 on test set: {r2_test:.4f}')
print(f'R2 on train set: {r2_train:.4f}')


# In[22]:


# 绘制测试集的真实值和预测值的散点图
plt.figure(figsize=(8, 6))
plt.scatter(y_test_np, y_pred_test_np, alpha=0.3,s=5, color = 'red', label='Test set')
plt.scatter(y_train_np, y_pred_train_np, alpha=0.3,s=5, color = 'blue', label='Training set')

plt.plot([y_test_np.min(), y_test_np.max()], [y_test_np.min(), y_test_np.max()], label='Perfect Prediction')
plt.xlabel('True YIELD')
plt.ylabel('Predicted YIELD')
plt.legend(loc='lower right')
plt.show()


# In[23]:


# 全部就设置这一个代码 要不都乱套了 这个就是OP 的梯度
# 定义第一个特征的范围并进行归一化
K_values = torch.arange(0, 121, 15).float().to(device)
K_values_norm = custom_normalize(K_values, Krates_T_min, Krates_T_max).view(-1)
K_num = K_values.tolist()


# In[24]:


K_values_norm


# In[25]:


# 存储预测结果的列表
y_pred_results = []

# 保持其他特征值不变，遍历每个归一化的第一个特征值
for value in K_values_norm:

    data_OP_train = value  # 将第一个特征值修改为当前归一化后的值

    # 在无梯度模式下进行预测
    with torch.no_grad():
        y_pred, a, b, c, d = model(data_x_train, data_OP_train)

    # 反归一化预测值
    y_pred_denormalized = custom_denormalize(y_pred, Yield_T_min, Yield_T_max)
    # 将反归一化后的结果转换为 NumPy 并存储
    y_pred_results.append(y_pred_denormalized.cpu().numpy().flatten()) 
# 将结果转换为 NumPy 数组，方便后续处理或分析
y_pred_results_np = np.array(y_pred_results).T 


# 将预测结果转换为 DataFrame，并将列名设置为对应的 OP 值
op_values = [f"Krate_{int(value)}" for value in K_num] # 列名称，如 OP_0, OP_2, OP_4,...
y_pred_results_df = pd.DataFrame(y_pred_results_np, columns=op_values)


# In[26]:


#y_pred_results_df


# In[27]:


# 计算每一列的平均值和95% CI
mean_values = np.array(y_pred_results_df.mean())

# 绘制曲线
plt.figure(figsize=(7, 6))

# 绘制平均值曲线
plt.plot(K_num, mean_values, label='Mean', color='blue')
# 设置 x 和 y 的刻度间隔
plt.xticks(range(0, 125, 30))

# 图形美化
plt.xlabel("K Fertilizer (kg ha$^{-1}$)", fontsize=12, labelpad=10)
plt.ylabel("Mean predicted yield (kg ha$^{-1}$)", fontsize=12, labelpad=10)
plt.tight_layout()


# 开始用中国数据预测
# 
# 算管理措施组合

# In[28]:


# 假设你已经有 dt
#cols = ['Irrigation', 'Density', 'Time', 'Approach', 'Type', 'Straw_return']

# 分析每列的数据类型与唯一值
#summary = []
#for col in cols:
 #   dtype = dt[col].dtype
  #  unique_vals = sorted(dt[col].dropna().unique().tolist())
   # summary.append([col, str(dtype), unique_vals])

# 转为 DataFrame
#df_summary = pd.DataFrame(summary, columns=['Column', 'Data Type', 'Unique Values'])
#print(df_summary)


# In[29]:


df_china = pd.read_csv('Maize_空间数据_V1.csv')  # 请替换成你的中国数据路径
continuous_cols = ['Seasonal_Tem','Seasonal_Pre','Seasonal_ET0','Sand','Clay','BD','pH','OM','TN','Olsen_P','Exc_K']
df_cont_original = df_china[continuous_cols]
df_cont_original


# In[30]:


df_manage = pd.read_csv('管理措施组合_All.csv')
discrete_combinations = df_manage.values
manage_columns=['Irrigation_I0',
       'Irrigation_I1', 'Density_D0', 'Density_D1', 'Density_D2', 'Time_T1',
       'Time_T2', 'Approach_A0', 'Approach_A1', 'Type_Ty0',
       'Type_Ty1', 'Type_Ty2', 'Straw_return_S0', 'Straw_return_S1']
#import itertools
# 定义离散变量的组合值
#Irrigation = [0, 0.5]
#Density = [0, 0.5, 1]
#Time = [0, 0.5, 1]
#Approach = [0.0, 0.5, 1.0, 1.5]
#Type = [0.0, 0.5, 1.0, 1.5]
#Straw_return = [0.0, 0.5]
#discrete_combinations = list(itertools.product(Irrigation, Density, Time, Approach, Type, Straw_return))


# In[31]:


pd.DataFrame(discrete_combinations)


# In[32]:


# 用于保存所有组合下的均值
import os, gc
from torch.utils.data import DataLoader, TensorDataset
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:16"  #设置 PyTorch 内存管理参数限制碎片化

mean_yield_per_combo = []

for comb in discrete_combinations:
    df_cont = df_cont_original.copy()
    df_cont.loc[:, manage_columns] = comb[:14]
    table_name = "Table_" + "_".join(map(str, comb[14:])) + "_predictions"

    # 转为张量 + 归一化
    df_cont = torch.tensor(df_cont.to_numpy(), dtype=torch.float32)
    #.to(device)   #数组过大，暂时不送GPU，节约显存
    X_T_min = X_T_min.to(df_cont.device) #保证下面custom_normalize时三个参数在同一设备上
    X_T_max = X_T_max.to(df_cont.device) #同上
    df_cont_tensor = custom_normalize(df_cont,X_T_min,X_T_max)

    # 构建 DataLoader，分批预测
    batch_size = 1024
    dataset = TensorDataset(df_cont_tensor)
    loader = DataLoader(dataset, batch_size=batch_size)

    y_pred_list = []

    for value in K_values_norm:
        y_pred_batches = []
        for batch in loader:
            input_batch = batch[0].to(device)
            with torch.no_grad():
                y_pred, _, _, _, _ = model(input_batch, value)
                y_pred_denorm = custom_denormalize(y_pred, Yield_T_min, Yield_T_max)
                y_pred_batches.append(y_pred_denorm.cpu().numpy())

            # 显存清理
            del input_batch, y_pred, y_pred_denorm
            torch.cuda.empty_cache()

        # 将所有批次拼接成一个完整的预测结果
        y_pred_concat = np.vstack(y_pred_batches).flatten()
        y_pred_list.append(y_pred_concat)

        del y_pred_batches, y_pred_concat
        gc.collect()

    # 将预测结果转换为 DataFrame，并将列名设置为对应的 OP 值
    y_pred_matrix = np.array(y_pred_list).T
    op_values = [f"{int(value)}" for value in K_num]
    y_pred_matrix = pd.DataFrame(y_pred_matrix, columns=op_values) #每个栅格不同施肥量的预测

    # 导出为 CSV 文件
    os.makedirs('prediction-smalldata_NoA2', exist_ok=True)
    y_pred_matrix.to_csv(f"prediction-smalldata_NoA2/{table_name}.csv", index=False)

    #mean_curve = np.mean(y_pred_matrix, axis=0)  # 平均曲线 (length = Krate数)
    mean_curve = y_pred_matrix.mean()          #此时转换成了List
    mean_yield_per_combo.append(mean_curve.values.tolist())     #同上（列为施肥量），.values.tolist()保证为纯CPU数据

     # 每轮结束后清理显存
    del df_cont, df_cont_tensor, y_pred_matrix, y_pred_list, mean_curve
    gc.collect()
    torch.cuda.empty_cache()


# In[33]:


#pd.DataFrame(y_pred_list).T           


# In[168]:


#y=pd.DataFrame(mean_yield_per_combo)


# In[169]:


#y['Index'] = df_manage[['Irrigation', 'Density', 'Time', 'Approach', 'Type', 'Straw_return']].apply(lambda row: '_'.join(row), axis=1)


# In[170]:


#y.to_csv("Mean_practices.csv", index=False)


# In[33]:


mean_yield_per_combo = np.array(mean_yield_per_combo)
print(mean_yield_per_combo.shape)  # 应该是 (86, 9)
print(len(K_num))


# In[34]:


# 转为 DataFrame
mean_yield_df = pd.DataFrame(mean_yield_per_combo, columns=[f"K_{int(k)}" for k in K_num])
mean_yield_df['Combo_ID'] =  df_manage[['Irrigation', 'Density', 'Time', 'Approach', 'Type', 'Straw_return']].apply(lambda row: '_'.join(row), axis=1)

# 排序（按K=290的平均值）
mean_yield_df = mean_yield_df.sort_values(by='K_120').reset_index(drop=True)


# In[35]:


mean_yield_df.to_csv("Mean_practices_smalldata_New2_p2.csv", index=False)
mean_yield_df


# # 开始导入作图

# In[209]:


mean_yield_df_or = pd.read_csv('Mean_practices_smalldata_New2_p2.csv')


# In[210]:


import matplotlib.cm as cm    #设置渐变的颜色梯度
from matplotlib import rcParams #设置面板的字体
import numpy as np
from scipy.interpolate import make_interp_spline  # 新引入插值函数

# 设置全局字体为Times New Roman
rcParams['font.family'] = 'Times New Roman'

# 配置字体参数
config = {
    "font.family": 'Times New Roman',            # 设置图表主字体为 Arial
    "font.size": 15,                   # 设置全局字体大小为 15pt
    "mathtext.fontset": 'custom',      #设置数学公式字体为自定义（而不是默认的 cm、stix）
    "mathtext.rm": 'Arial',            # 数学公式中的普通字体部分使用 Arial
    "mathtext.it": 'Arial:italic',     #数学公式中的斜体（如变量 x）使用 Arial italic
    "mathtext.bf": 'Arial:bold',       # 数学公式中的加粗字体（如 \mathbf{x}）使用 Arial bold
    "font.serif": ['Times New Roman'], # 设置 serif 字体（衬线体）为 Times New Roman（此项只在指定 serif 时有效）
    'axes.unicode_minus': False # 处理负号显示问题
}
rcParams.update(config)


# In[211]:


#print(mean_yield_df_or.index)


# In[212]:


num_lines=len(mean_yield_df_or)
num_lines


# In[213]:


colors = cm.get_cmap('RdYlGn', num_lines)


# In[214]:


# 可视化每种组合的响应曲线
K_num = np.array([0, 30, 45, 60, 75, 90, 105, 120])
K_smooth = np.linspace(K_num.min(), K_num.max(), 121)  # 构建更细的横坐标用于平滑


# In[215]:


# 初始化结果 DataFrame，并添加横坐标

smooth_df = pd.DataFrame({'K_smooth': K_smooth})

plt.figure(figsize=(12, 9.8))
for i in range(num_lines):
    combo_id = mean_yield_df_or.loc[i, 'Combo_ID']
    y_points = mean_yield_df_or.loc[i, [f"K_{int(k)}" for k in K_num]].values

    # 使用三次样条进行拟合（k=3 表示三次）
    spline = make_interp_spline(K_num, y_points, k=3)
    y_smooth = spline(K_smooth)

    # 添加到DataFrame中，每列对应一个组合
    smooth_df[combo_id] = y_smooth

    plt.plot(K_smooth, y_smooth, color=colors(i), label=combo_id, linewidth=1.5, alpha=0.9)

plt.xlabel("K$_2$O (kg ha$^{-1}$)", fontsize=25)
plt.ylabel("Predicted Yield (kg ha$^{-1}$)", fontsize=25)
plt.xticks(range(0, 125, 30), fontsize=20)
plt.ylim(5500, 14500)
plt.yticks(range(6000, 14001, 2000), fontsize=20)
plt.yticks(fontsize=20)
plt.grid(False)
plt.legend(
    loc='center left',           # 图例位置（左中）
    bbox_to_anchor=(1.01, 0.5),  # 图例放在图外右侧，横向略右、纵向居中
    fontsize=9,                  # 图例字体大小，可设为 'small'、8 等
    ncol=2,                      # 图例列数（将图例分成2列）
    handlelength=0.5,            # 控制图例线条长度
    handletextpad=0.5,           # 线段和文字之间的间距
    borderaxespad=0.3,           # 图像与图例的距离
    columnspacing=0.5,            # 图例列之间的间距
    labelspacing=0.4,
    frameon=False
)
plt.tight_layout(rect=[0, 0, 0.8, 1])
plt.show()

# 导出平滑数据为CSV文件，此处导出施肥梯度为1时对应的产量
smooth_df.to_csv("prediction-smalldata_NoA2/图片/K_response_curves_smooth_all.csv", index=False)


# # 再往下这些都不用管
# # 

# In[216]:


#灌溉和雨养对比
plt.figure(figsize = (8,6))
Combo_ID_I = ['I0: Rainfed','I1: Irrigation']

#提取一次施肥的索引
I0_index = mean_yield_df_or.loc[mean_yield_df_or['Combo_ID'].str.contains('I0', regex=True),[f"K_{int(k)}" for k in K_num]]

#遍历提取一次施肥的曲线
for i in I0_index.index:
    index = mean_yield_df_or['Combo_ID'][i]
    y_index = mean_yield_df_or.loc[i,[f"K_{int(k)}" for k in K_num]].values

    # 使用三次样条进行拟合（k=3 表示三次）
    spline = make_interp_spline(K_num, y_index, k=3)
    y_smooth = spline(K_smooth)

    plt.plot(K_smooth, y_smooth, alpha = 0.1, linewidth=1.5,color = 'red')

I0_mean = I0_index.mean().values
spline_I0_mean = make_interp_spline(K_num, I0_mean, k=3)
I0_smooth = spline_I0_mean(K_smooth)
plt.plot(K_smooth, I0_smooth, label = Combo_ID_I[0], linewidth=2.5,color = 'red')

#提取多次施肥的索引
I1_index = mean_yield_df_or.loc[mean_yield_df_or['Combo_ID'].str.contains('I1', regex=True),[f"K_{int(k)}" for k in K_num]]

#遍历提取多次施肥的曲线
for i in I1_index.index:
    index =mean_yield_df_or['Combo_ID'][i]
    y_index = mean_yield_df_or.loc[i,[f"K_{int(k)}" for k in K_num]].values

     # 使用三次样条进行拟合（k=3 表示三次）
    spline = make_interp_spline(K_num, y_index, k=3)
    y_smooth = spline(K_smooth)
    plt.plot(K_smooth, y_smooth, alpha = 0.1, linewidth=1.5,color = 'green')

I1_mean = I1_index.mean().values
spline_I1_mean = make_interp_spline(K_num, I1_mean, k=3)
I1_smooth = spline_I1_mean(K_smooth)
plt.plot(K_smooth, I1_smooth, label = Combo_ID_I[1], linewidth=2.5,color = 'green')

#plt.title('Comparison of the N$_2$O curve in different management',size = 20)
plt.xlabel("K$_2$O (kg ha$^{-1}$)",size = 25)
plt.ylabel("Predicted Yield (kg ha$^{-1}$)", fontsize=25)
plt.xticks(range(0, 125, 30), fontsize=20)
plt.ylim(5000, 12500)
plt.yticks(range(6000, 12001, 2000), fontsize=20)
plt.yticks(fontsize=20)
plt.grid(False)
# 显示图例
plt.legend(
    prop={'size': 25},
    handlelength=1.0,            # 控制图例线条长度
    handletextpad=0.5,           # 线段和文字之间的间距
    labelspacing=0.1,
    #title='Frequency', title_fontsize=30,
    frameon=False,
    ncol=1,loc='lower right'
)
# 将K_smooth与三条平均曲线整合成一个DataFrame
irrigation_smooth_df = pd.DataFrame({
    "K_smooth": K_smooth,
    "I0": I0_smooth,
    "I1": I1_smooth
})
# 输出为CSV文件
irrigation_smooth_df.to_csv("prediction-smalldata_NoA2/图片/Irrigation_Comparison_Smooth.csv", index=False)


# In[217]:


#密度对比
plt.figure(figsize = (8,6))
Combo_ID_D = ['D0: Low density','D1: Medium density', 'D2: High density']

#提取一次施肥的索引
D0_index = mean_yield_df_or.loc[mean_yield_df_or['Combo_ID'].str.contains('D0', regex=True),[f"K_{int(k)}" for k in K_num]]

#遍历提取一次施肥的曲线
for i in D0_index.index:
    index = mean_yield_df_or['Combo_ID'][i]
    y_index = mean_yield_df_or.loc[i,[f"K_{int(k)}" for k in K_num]].values

    # 使用三次样条进行拟合（k=3 表示三次）
    spline = make_interp_spline(K_num, y_index, k=3)
    y_smooth = spline(K_smooth)

    plt.plot(K_smooth, y_smooth, alpha = 0.1, linewidth=1.5,color = 'red')

D0_mean = D0_index.mean().values
spline_D0_mean = make_interp_spline(K_num, D0_mean, k=3)
D0_smooth = spline_D0_mean(K_smooth)
plt.plot(K_smooth, D0_smooth, label = Combo_ID_D[0], linewidth=2.5,color = 'red')

#提取多次施肥的索引
D1_index = mean_yield_df_or.loc[mean_yield_df_or['Combo_ID'].str.contains('D1', regex=True),[f"K_{int(k)}" for k in K_num]]

#遍历提取多次施肥的曲线
for i in D1_index.index:
    index =mean_yield_df_or['Combo_ID'][i]
    y_index = mean_yield_df_or.loc[i,[f"K_{int(k)}" for k in K_num]].values

     # 使用三次样条进行拟合（k=3 表示三次）
    spline = make_interp_spline(K_num, y_index, k=3)
    y_smooth = spline(K_smooth)
    plt.plot(K_smooth, y_smooth, alpha = 0.1, linewidth=1.5,color = 'orange')

D1_mean = D1_index.mean().values
spline_D1_mean = make_interp_spline(K_num, D1_mean, k=3)
D1_smooth = spline_D1_mean(K_smooth)
plt.plot(K_smooth, D1_smooth, label = Combo_ID_D[1], linewidth=2.5,color = 'orange')

#提取一次施肥的索引
D2_index = mean_yield_df_or.loc[mean_yield_df_or['Combo_ID'].str.contains('D2', regex=True),[f"K_{int(k)}" for k in K_num]]

#遍历提取一次施肥的曲线
for i in D2_index.index:
    index = mean_yield_df_or['Combo_ID'][i]
    y_index = mean_yield_df_or.loc[i,[f"K_{int(k)}" for k in K_num]].values

    # 使用三次样条进行拟合（k=3 表示三次）
    spline = make_interp_spline(K_num, y_index, k=3)
    y_smooth = spline(K_smooth)

    plt.plot(K_smooth, y_smooth, alpha = 0.1, linewidth=1.5,color = 'green')

D2_mean = D2_index.mean().values
spline_D2_mean = make_interp_spline(K_num, D2_mean, k=3)
D2_smooth = spline_D2_mean(K_smooth)
plt.plot(K_smooth, D2_smooth, label = Combo_ID_D[2], linewidth=2.5,color = 'green')

#plt.title('Comparison of the N$_2$O curve in different management',size = 20)
plt.xlabel("K$_2$O (kg ha$^{-1}$)",size = 25)
plt.ylabel("Predicted Yield (kg ha$^{-1}$)", fontsize=25)
plt.xticks(range(0, 125, 30), fontsize=20)
plt.ylim(5000, 12500)
plt.yticks(range(6000, 12001, 2000), fontsize=20)
plt.yticks(fontsize=20)
plt.grid(False)
# 显示图例
plt.legend(
    prop={'size': 25},
    handlelength=1.0,            # 控制图例线条长度
    handletextpad=0.5,           # 线段和文字之间的间距
    labelspacing=0.1,
    #title='Frequency', title_fontsize=30,
    frameon=False,
    ncol=1,loc='lower right'
)

# 将K_smooth与三条平均曲线整合成一个DataFrame
density_smooth_df = pd.DataFrame({
    "K_smooth": K_smooth,
    "D0_LowDensity": D0_smooth,
    "D1_MediumDensity": D1_smooth,
    "D2_HighDensity": D2_smooth
})
# 输出为CSV文件
density_smooth_df.to_csv("prediction-smalldata_NoA2/图片/Density_Comparison_Smooth_modified.csv", index=False)


# In[218]:


#施肥次数对比
plt.figure(figsize = (8,6))
Combo_ID_T = ['T1: Once','T2: More']

#提取一次施肥的索引
T1_index = mean_yield_df_or.loc[mean_yield_df_or['Combo_ID'].str.contains('T1', regex=True),[f"K_{int(k)}" for k in K_num]]

#遍历提取一次施肥的曲线
for i in T1_index.index:
    index = mean_yield_df_or['Combo_ID'][i]
    y_index = mean_yield_df_or.loc[i,[f"K_{int(k)}" for k in K_num]].values

    # 使用三次样条进行拟合（k=3 表示三次）
    spline = make_interp_spline(K_num, y_index, k=3)
    y_smooth = spline(K_smooth)

    plt.plot(K_smooth, y_smooth, alpha = 0.1, linewidth=1.5,color = 'red')

T1_mean = T1_index.mean().values
spline_T1_mean = make_interp_spline(K_num, T1_mean, k=3)
T1_smooth = spline_T1_mean(K_smooth)
plt.plot(K_smooth, T1_smooth, label = Combo_ID_T[0], linewidth=2.5,color = 'red')

#提取多次施肥的索引
T2_index = mean_yield_df_or.loc[mean_yield_df_or['Combo_ID'].str.contains('T2', regex=True),[f"K_{int(k)}" for k in K_num]]

#遍历提取多次施肥的曲线
for i in T2_index.index:
    index =mean_yield_df_or['Combo_ID'][i]
    y_index = mean_yield_df_or.loc[i,[f"K_{int(k)}" for k in K_num]].values

     # 使用三次样条进行拟合（k=3 表示三次）
    spline = make_interp_spline(K_num, y_index, k=3)
    y_smooth = spline(K_smooth)
    plt.plot(K_smooth, y_smooth, alpha = 0.1, linewidth=1.5,color = 'green')

T2_mean = T2_index.mean().values
spline_T2_mean = make_interp_spline(K_num, T2_mean, k=3)
T2_smooth = spline_T2_mean(K_smooth)
plt.plot(K_smooth, T2_smooth, label = Combo_ID_T[1], linewidth=2.5,color = 'green')

#plt.title('Comparison of the N$_2$O curve in different management',size = 20)
plt.xlabel("K$_2$O (kg ha$^{-1}$)",size = 25)
plt.ylabel("Predicted Yield (kg ha$^{-1}$)", fontsize=25)
plt.xticks(range(0, 125, 30), fontsize=20)
plt.ylim(5000, 12500)
plt.yticks(range(6000, 12001, 2000), fontsize=20)
plt.yticks(fontsize=20)
plt.grid(False)
# 显示图例
plt.legend(
    prop={'size': 25},
    handlelength=1.0,            # 控制图例线条长度
    handletextpad=0.5,           # 线段和文字之间的间距
    labelspacing=0.1,
    #title='Frequency', title_fontsize=30,
    frameon=False,
    ncol=1,loc='lower right'
)
# 将K_smooth与三条平均曲线整合成一个DataFrame
time_smooth_df = pd.DataFrame({
    "K_smooth": K_smooth,
    "T1": T1_smooth,
    "T2": T2_smooth
})
# 输出为CSV文件
time_smooth_df.to_csv("prediction-smalldata_NoA2/图片/Time_Comparison_Smooth.csv", index=False)


# In[219]:


#施肥方式对比
plt.figure(figsize = (8,6))
Combo_ID_A = ['A0: Broadcast','A1: Incorporation', 'A2: Broadcast & Incorporation','A3: Fertigation']

#提取一次施肥的索引
A0_index = mean_yield_df_or.loc[mean_yield_df_or['Combo_ID'].str.contains('A0', regex=True),[f"K_{int(k)}" for k in K_num]]

#遍历提取一次施肥的曲线
for i in A0_index.index:
    index = mean_yield_df_or['Combo_ID'][i]
    y_index = mean_yield_df_or.loc[i,[f"K_{int(k)}" for k in K_num]].values

    # 使用三次样条进行拟合（k=3 表示三次）
    spline = make_interp_spline(K_num, y_index, k=3)
    y_smooth = spline(K_smooth)

    plt.plot(K_smooth, y_smooth, alpha = 0.1, linewidth=1.5,color = 'red')

A0_mean = A0_index.mean().values
spline_A0_mean = make_interp_spline(K_num, A0_mean, k=3)
A0_smooth = spline_A0_mean(K_smooth)
plt.plot(K_smooth, A0_smooth, label = Combo_ID_A[0], linewidth=2.5,color = 'red')

#提取多次施肥的索引
A1_index = mean_yield_df_or.loc[mean_yield_df_or['Combo_ID'].str.contains('A1', regex=True),[f"K_{int(k)}" for k in K_num]]

#遍历提取多次施肥的曲线
for i in A1_index.index:
    index =mean_yield_df_or['Combo_ID'][i]
    y_index = mean_yield_df_or.loc[i,[f"K_{int(k)}" for k in K_num]].values

     # 使用三次样条进行拟合（k=3 表示三次）
    spline = make_interp_spline(K_num, y_index, k=3)
    y_smooth = spline(K_smooth)
    plt.plot(K_smooth, y_smooth, alpha = 0.1, linewidth=1.5,color = 'green')

A1_mean = A1_index.mean().values
spline_A1_mean = make_interp_spline(K_num, A1_mean, k=3)
A1_smooth = spline_A1_mean(K_smooth)
plt.plot(K_smooth, A1_smooth, label = Combo_ID_A[1], linewidth=2.5,color = 'green')

# #提取一次施肥的索引
# A2_index = mean_yield_df_or.loc[mean_yield_df_or['Combo_ID'].str.contains('A2', regex=True),[f"K_{int(k)}" for k in K_num]]

# #遍历提取一次施肥的曲线
# for i in A2_index.index:
#     index = mean_yield_df_or['Combo_ID'][i]
#     y_index = mean_yield_df_or.loc[i,[f"K_{int(k)}" for k in K_num]].values

#     # 使用三次样条进行拟合（k=3 表示三次）
#     spline = make_interp_spline(K_num, y_index, k=3)
#     y_smooth = spline(K_smooth)

#     plt.plot(K_smooth, y_smooth, alpha = 0.1, linewidth=1.5,color = 'blue')

# A2_mean = A2_index.mean().values
# spline_A2_mean = make_interp_spline(K_num, A2_mean, k=3)
# A2_smooth = spline_A2_mean(K_smooth)
# plt.plot(K_smooth, A2_smooth, label = Combo_ID_A[2], linewidth=2.5,color = 'blue')

# #提取一次施肥的索引
# A3_index = mean_yield_df_or.loc[mean_yield_df_or['Combo_ID'].str.contains('A3', regex=True),[f"K_{int(k)}" for k in K_num]]

# #遍历提取一次施肥的曲线
# for i in A3_index.index:
#     index = mean_yield_df_or['Combo_ID'][i]
#     y_index = mean_yield_df_or.loc[i,[f"K_{int(k)}" for k in K_num]].values

#     # 使用三次样条进行拟合（k=3 表示三次）
#     spline = make_interp_spline(K_num, y_index, k=3)
#     y_smooth = spline(K_smooth)

#     plt.plot(K_smooth, y_smooth, alpha = 0.1, linewidth=1.5,color = 'green')

# A3_mean = A3_index.mean().values
# spline_A3_mean = make_interp_spline(K_num, A3_mean, k=3)
# A3_smooth = spline_A3_mean(K_smooth)
# plt.plot(K_smooth, A3_smooth, label = Combo_ID_A[3], linewidth=2.5,color = 'green')

#plt.title('Comparison of the N$_2$O curve in different management',size = 20)
plt.xlabel("K$_2$O (kg ha$^{-1}$)",size = 25)
plt.ylabel("Predicted Yield (kg ha$^{-1}$)", fontsize=25)
plt.xticks(range(0, 125, 30), fontsize=20)
plt.ylim(5000, 12500)
plt.yticks(range(6000, 12001, 2000), fontsize=20)
plt.yticks(fontsize=20)
plt.grid(False)
# 显示图例
plt.legend(
    prop={'size': 25},
    handlelength=1.0,            # 控制图例线条长度
    handletextpad=0.5,           # 线段和文字之间的间距
    labelspacing=0.1,
    #title='Frequency', title_fontsize=30,
    frameon=False,
    ncol=1,loc='lower right'
)
# 将K_smooth与三条平均曲线整合成一个DataFrame
approach_smooth_df = pd.DataFrame({
    "K_smooth": K_smooth,
    "A0": A0_smooth,
    "A1": A1_smooth
})
# 输出为CSV文件
approach_smooth_df.to_csv("prediction-smalldata_NoA2/图片/Approach_Comparison_Smooth.csv", index=False)


# In[220]:


#肥料种类
plt.figure(figsize = (8,6))
Combo_ID_Ty = ['Ty0: KCl','Ty1: K$_2$SO$_4$']

#提取一次施肥的索引
Ty0_index = mean_yield_df_or.loc[mean_yield_df_or['Combo_ID'].str.contains('Ty0', regex=True),[f"K_{int(k)}" for k in K_num]]

#遍历提取一次施肥的曲线
for i in Ty0_index.index:
    index = mean_yield_df_or['Combo_ID'][i]
    y_index = mean_yield_df_or.loc[i,[f"K_{int(k)}" for k in K_num]].values

    # 使用三次样条进行拟合（k=3 表示三次）
    spline = make_interp_spline(K_num, y_index, k=3)
    y_smooth = spline(K_smooth)

    plt.plot(K_smooth, y_smooth, alpha = 0.1, linewidth=1.5,color = 'red')

Ty0_mean = Ty0_index.mean().values
spline_Ty0_mean = make_interp_spline(K_num, Ty0_mean, k=3)
Ty0_smooth = spline_Ty0_mean(K_smooth)
plt.plot(K_smooth, Ty0_smooth, label = Combo_ID_Ty[0], linewidth=2.5,color = 'red')

#提取多次施肥的索引
Ty1_index = mean_yield_df_or.loc[mean_yield_df_or['Combo_ID'].str.contains('Ty1', regex=True),[f"K_{int(k)}" for k in K_num]]

#遍历提取多次施肥的曲线
for i in Ty1_index.index:
    index =mean_yield_df_or['Combo_ID'][i]
    y_index = mean_yield_df_or.loc[i,[f"K_{int(k)}" for k in K_num]].values

     # 使用三次样条进行拟合（k=3 表示三次）
    spline = make_interp_spline(K_num, y_index, k=3)
    y_smooth = spline(K_smooth)
    plt.plot(K_smooth, y_smooth, alpha = 0.1, linewidth=1.5,color = 'green')

Ty1_mean = Ty1_index.mean().values
spline_Ty1_mean = make_interp_spline(K_num, Ty1_mean, k=3)
Ty1_smooth = spline_Ty1_mean(K_smooth)
plt.plot(K_smooth, Ty1_smooth, label = Combo_ID_Ty[1], linewidth=2.5,color = 'green')

#plt.title('Comparison of the N$_2$O curve in different management',size = 20)
plt.xlabel("K$_2$O (kg ha$^{-1}$)",size = 25)
plt.ylabel("Predicted Yield (kg ha$^{-1}$)", fontsize=25)
plt.xticks(range(0, 125, 30), fontsize=20)
plt.ylim(5000, 12500)
plt.yticks(range(6000, 12001, 2000), fontsize=20)
plt.yticks(fontsize=20)
plt.grid(False)
# 显示图例
plt.legend(
    prop={'size': 25},
    handlelength=1.0,            # 控制图例线条长度
    handletextpad=0.5,           # 线段和文字之间的间距
    labelspacing=0.1,
    #title='Frequency', title_fontsize=30,
    frameon=False,
    ncol=1,loc='lower right'
)
# 将K_smooth与三条平均曲线整合成一个DataFrame
type_smooth_df = pd.DataFrame({
    "K_smooth": K_smooth,
    "Ty0": Ty0_smooth,
    "Ty1": Ty1_smooth
})
# 输出为CSV文件
type_smooth_df.to_csv("prediction-smalldata_NoA2/图片/Type_Comparison_Smooth.csv", index=False)


# In[221]:


#秸秆还田对比
plt.figure(figsize = (8,6))
Combo_ID_S = ['S0: No return','S1: Return']

#提取一次施肥的索引
S0_index = mean_yield_df_or.loc[mean_yield_df_or['Combo_ID'].str.contains('S0', regex=True),[f"K_{int(k)}" for k in K_num]]

#遍历提取一次施肥的曲线
for i in S0_index.index:
    index = mean_yield_df_or['Combo_ID'][i]
    y_index = mean_yield_df_or.loc[i,[f"K_{int(k)}" for k in K_num]].values

    # 使用三次样条进行拟合（k=3 表示三次）
    spline = make_interp_spline(K_num, y_index, k=3)
    y_smooth = spline(K_smooth)

    plt.plot(K_smooth, y_smooth, alpha = 0.1, linewidth=1.5,color = 'red')

S0_mean = S0_index.mean().values
spline_S0_mean = make_interp_spline(K_num, S0_mean, k=3)
S0_smooth = spline_S0_mean(K_smooth)
plt.plot(K_smooth, S0_smooth, label = Combo_ID_S[0], linewidth=2.5,color = 'red')

#提取多次施肥的索引
S1_index = mean_yield_df_or.loc[mean_yield_df_or['Combo_ID'].str.contains('S1', regex=True),[f"K_{int(k)}" for k in K_num]]

#遍历提取多次施肥的曲线
for i in S1_index.index:
    index =mean_yield_df_or['Combo_ID'][i]
    y_index = mean_yield_df_or.loc[i,[f"K_{int(k)}" for k in K_num]].values

     # 使用三次样条进行拟合（k=3 表示三次）
    spline = make_interp_spline(K_num, y_index, k=3)
    y_smooth = spline(K_smooth)
    plt.plot(K_smooth, y_smooth, alpha = 0.1, linewidth=1.5,color = 'green')

S1_mean = S1_index.mean().values
spline_S1_mean = make_interp_spline(K_num, S1_mean, k=3)
S1_smooth = spline_S1_mean(K_smooth)
plt.plot(K_smooth, S1_smooth, label = Combo_ID_S[1], linewidth=2.5,color = 'green')

#plt.title('Comparison of the N$_2$O curve in different management',size = 20)
plt.xlabel("K$_2$O (kg ha$^{-1}$)",size = 25)
plt.ylabel("Predicted Yield (kg ha$^{-1}$)", fontsize=25)
plt.xticks(range(0, 125, 30), fontsize=20)
plt.ylim(5000, 12500)
plt.yticks(range(6000, 12001, 2000), fontsize=20)
plt.yticks(fontsize=20)
plt.grid(False)
# 显示图例
plt.legend(
    prop={'size': 25},
    handlelength=1.0,            # 控制图例线条长度
    handletextpad=0.5,           # 线段和文字之间的间距
    labelspacing=0.1,
    #title='Frequency', title_fontsize=30,
    frameon=False,
    ncol=1,loc='lower right'
)
# 将K_smooth与三条平均曲线整合成一个DataFrame
straw_smooth_df = pd.DataFrame({
    "K_smooth": K_smooth,
    "S0": S0_smooth,
    "S1": S1_smooth
})
# 输出为CSV文件
straw_smooth_df.to_csv("prediction-smalldata_NoA2/图片/Straw_Comparison_Smooth.csv", index=False)


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# ## 处理整合生成的数据

# In[4]:


import os

# 设置工作目录
folder_path = "E:/Jupyter/K/Maize2/prediction-smalldata_NoA2"
output_dir = "E:/Jupyter/K/Maize2/Exported data"

# 获取文件夹中的所有CSV文件
file_names = [f for f in os.listdir(folder_path) if f.endswith('.csv')]
# 初始化字典存储各列数据（比列表更直观）
columns_data = {
    1: {},  #（Python索引从0开始，但CSV通常有标题列）
    2: {},
    3: {},
    4: {},
    5: {},
    6: {},
    7: {},
    8: {},
    9: {}
}


# In[5]:


# 循环处理每个文件
for file in file_names:
    # 读取CSV文件
    file_path = os.path.join(folder_path, file)
    data = pd.read_csv(file_path)

    # 提取文件名（不含扩展名）
    file_name = os.path.splitext(file)[0]

    # 提取第1-9列并存入字典
    for col_idx in range(1, 10):  # 2-10列
        col_data = data.iloc[:, col_idx - 1]  # pandas列索引从0开始
        columns_data[col_idx][file_name] = col_data

# 将字典转换为DataFrame并保存,此处为了和R里的代码保持一致，文件的名称没有改，实际应该是“First_colums.....Second_colums”
output_files = {
    1: "0.Second_columns_V1.csv",
    2: "15.Third_columns_V1.csv",
    3: "30.Forth_columns_V1.csv",
    4: "45.Fifth_columns_V1.csv",
    5: "60.Sixth_columns_V1.csv",
    6: "75.Seventh_columns_V1.csv",
    7: "90.Eighth_columns_V1.csv",
    8: "105.Ninth_columns_V1.csv",
    9: "120.Tenth_columns_V1.csv"
}


# In[6]:


for col_idx, output_file in output_files.items():
    result_df = pd.DataFrame(columns_data[col_idx])
    output_path = os.path.join(output_dir, output_file)
    result_df.to_csv(output_path, index=False)
    print(f"Saved: {output_path} | Shape: {result_df.shape}")


# # 向下暂时没有用（算单个措施钾梯度的）

# In[38]:


china_single_data = df_cont_original.copy()
china_single_data


# In[39]:


china_single_data.loc[:, manage_columns] = (0.5, 1, 1, 1.5, 1.5, 0.5)
china_single_data = torch.tensor(china_single_data.to_numpy(), dtype=torch.float32).to(device)
china_single_data_nor = custom_normalize(china_single_data,X_T_min,X_T_max)


# In[40]:


y_china_pred_list = []
for value in K_values_norm:
    # 替换第一个特征为当前K值
    K_rate = value
    with torch.no_grad():
        y_pred, a, b, C = model(china_single_data_nor, K_rate)

    # 反归一化（你需要替换成自己的反归一化函数）
    y_pred_denorm = custom_denormalize(y_pred, Yield_T_min, Yield_T_max)
    y_china_pred_list.append(y_pred_denorm.cpu().numpy().flatten())


# In[41]:


y_china_pred_list = pd.DataFrame(y_china_pred_list).T
y_china_pred_list


# In[42]:


# 导出为 CSV 文件
y_china_pred_list.to_csv('./mingcheng.csv', index=False)


# In[ ]:




