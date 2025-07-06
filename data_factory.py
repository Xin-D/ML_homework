
import numpy as np
import pandas as pd
import torch
from torch.utils.data  import Dataset
from sklearn.preprocessing  import StandardScaler


# ======================
# 数据预处理
# ======================
class DataPreprocessor:
    def __init__(self, data_path):
        self.data_path  = data_path 
        self.columns  = [
            'DateTime', 'Global_active_power', 'Global_reactive_power', 
            'Voltage', 'Global_intensity', 'Sub_metering_1', 
            'Sub_metering_2', 'Sub_metering_3', 'RR', 'NBJRR1', 
            'NBJRR5', 'NBJRR10', 'NBJBROU'
        ]
    def load_and_preprocess(self, is_train=True):
        if is_train:
            data_df = pd.read_csv(self.data_path, dayfirst=True, low_memory=False, 
                                  na_values=['nan', '?', 'NA', 'N/A'], usecols=self.columns)
        else:
            # 加载测试数据（无列名）
            data_df = pd.read_csv(self.data_path, header=None,
                                   dayfirst=True, low_memory=False, na_values=['nan', '?', 'NA', 'N/A'])
        data_df.columns  = self.columns
        
        # 转换DateTime格式 
        data_df['DateTime'] = pd.to_datetime(data_df['DateTime'])
        
        # # 转换数据类型（字符串->浮点数）
        for col in self.columns[1:]: 
            # 处理逗号作为小数点的情况 
            data_df[col] = data_df[col].astype(str).str.replace(',',  '.')
            data_df[col] = pd.to_numeric(data_df[col],  errors='coerce')
        
        # 缺失值处理策略
        data_df = self.handle_missing_values(data_df) 
        data_df = self.aggregate_daily_data(data_df)
        return data_df
    
    def handle_missing_values(self, df):
        cols = df.columns
        mean_values = df[cols].mean()
        
        # 批量填充缺失值,均值填充
        df[cols] = df[cols].fillna(mean_values)
        
        return df
    
    def aggregate_daily_data(self, df):
        # 按天聚合数据 
        daily_df = df.copy() 
        daily_df['Date'] = daily_df['DateTime'].dt.date  
        # 创建聚合规则 
        agg_rules = {
            'Global_active_power': 'sum',
            'Global_reactive_power': 'sum',
            'Voltage': 'mean',
            'Global_intensity': 'mean',
            'Sub_metering_1': 'sum',
            'Sub_metering_2': 'sum',
            'Sub_metering_3': 'sum',
            'RR': 'first',
            'NBJRR1': 'first',
            'NBJRR5': 'first',
            'NBJRR10': 'first',
            'NBJBROU': 'first'
        }
        
        daily_agg = daily_df.groupby('Date').agg(agg_rules).reset_index() 
        # daily_agg = daily_agg.iloc[:,1:8]
        # 计算剩余能耗 
        daily_agg['Sub_metering_remainder'] = (
            daily_agg['Global_active_power'] * 1000 / 60 
        ) - (
            daily_agg['Sub_metering_1'] + 
            daily_agg['Sub_metering_2'] + 
            daily_agg['Sub_metering_3']
        )
        
        # 转换降水单位（毫米的十分之一 → 毫米）
        daily_agg['RR'] = daily_agg['RR'] / 10  
        return daily_agg 
def load_and_preprocess_data(train_path, test_path):
    # 加载数据
    train_df = DataPreprocessor(train_path).load_and_preprocess()
    test_df = DataPreprocessor(test_path).load_and_preprocess(False)
    # 合并数据集便于处理
    daily_df = pd.concat([train_df,  test_df])
    
    
    # 分割回训练集和测试集
    train_dates = daily_df.iloc[:len(train_df)]['Date'] 
    test_dates = daily_df.iloc[len(train_df):len(train_df)+len(test_df)]['Date'] 
    
    train_data = daily_df[daily_df['Date'].isin(train_dates)]
    test_data = daily_df[daily_df['Date'].isin(test_dates)]
    
    # 特征和目标列
    feature_cols = ['Global_reactive_power', 'Sub_metering_1', 'Sub_metering_2', 
                    'Sub_metering_3', 'Sub_metering_remainder', 'Voltage', 
                    'Global_intensity', 'RR', 'NBJRR1', 'NBJRR5', 'NBJRR10', 'NBJBROU']
    target_col = 'Global_active_power'
    
    # 数据标准化
    scaler_x = StandardScaler()
    scaler_y = StandardScaler()
    
    train_x = scaler_x.fit_transform(train_data[feature_cols]) 
    train_y = scaler_y.fit_transform(train_data[[target_col]]) 
    
    test_x = scaler_x.transform(test_data[feature_cols]) 
    test_y = scaler_y.transform(test_data[[target_col]]) 
    
    # 创建序列数据集
    def create_sequences(data_x, data_y, window_size, horizon):
        sequences = []
        targets = []
        
        for i in range(len(data_x) - window_size - horizon + 1):
            seq = data_x[i:i+window_size]
            target = data_y[i+window_size:i+window_size+horizon]
            sequences.append(seq) 
            targets.append(target) 
            
        return np.array(sequences),  np.array(targets) 
    
    # 创建训练和测试序列
    window_size = 90  # 使用过去90天预测未来
    
    # 短期预测序列（90天）
    train_x_seq90, train_y_seq90 = create_sequences(train_x, train_y, window_size, horizon=90)
    test_x_seq90, test_y_seq90 = create_sequences(test_x, test_y, window_size, horizon=90)
    
    # 长期预测序列（365天）
    train_x_seq365, train_y_seq365 = create_sequences(train_x, train_y, window_size, horizon=365)
    test_x_seq365, test_y_seq365 = create_sequences(test_x, test_y, window_size, horizon=365)
    
    return (train_x_seq90, train_y_seq90, test_x_seq90, test_y_seq90,
            train_x_seq365, train_y_seq365, test_x_seq365, test_y_seq365,
            scaler_y, feature_cols)

# ======================
# PyTorch数据集类
# ======================
class PowerDataset(Dataset):
    def __init__(self, sequences, targets):
        self.sequences  = sequences
        self.targets  = targets
        
    def __len__(self):
        return len(self.sequences) 
    
    def __getitem__(self, idx):
        return torch.tensor(self.sequences[idx],  dtype=torch.float32),  \
               torch.tensor(self.targets[idx],  dtype=torch.float32) 