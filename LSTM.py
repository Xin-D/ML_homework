import numpy as np
import pandas as pd
import matplotlib.pyplot  as plt
import torch
import torch.nn  as nn
from torch.utils.data  import Dataset, DataLoader
from datetime import datetime
from sklearn.preprocessing  import StandardScaler
from sklearn.metrics  import mean_squared_error, mean_absolute_error
from tqdm import tqdm
import copy
import warnings
warnings.filterwarnings('ignore') 

# 设置随机种子确保结果可复现
torch.manual_seed(42) 
np.random.seed(42) 

# 检查GPU可用性
device = torch.device("cuda"  if torch.cuda.is_available()  else "cpu")
print(f"Using device: {device}")
# ======================
# 3. LSTM 模型
# ======================
class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size, horizon, dropout_rate=0.2):
        super(LSTMModel, self).__init__()
        self.hidden_size  = hidden_size
        self.num_layers  = num_layers
        self.horizon  = horizon
        
        self.lstm  = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.dropout = nn.Dropout(dropout_rate)
        self.fc  = nn.Linear(hidden_size, output_size)
        
    def forward(self, x):
        # x形状: (batch_size, seq_len, input_size)
        h0 = torch.zeros(self.num_layers,  x.size(0),  self.hidden_size).to(x.device) 
        c0 = torch.zeros(self.num_layers,  x.size(0),  self.hidden_size).to(x.device) 
        
        # LSTM层
        out, _ = self.lstm(x,  (h0, c0))
        
        # 只取最后一个时间步的输出用于预测多个未来时间点
        out = out[:, -1, :]
        out = self.dropout(out)
        # 全连接层预测多个时间点
        out = self.fc(out) 
        return out.view(-1,  self.horizon,  1)
