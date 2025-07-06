import numpy as np
import torch
import torch.nn  as nn
import warnings
warnings.filterwarnings('ignore') 


# ======================
# Transformer 模型
# ======================
class TransformerModel(nn.Module):
    def __init__(self, input_size, d_model, nhead, num_layers, output_size, horizon):
        super(TransformerModel, self).__init__()
        self.horizon  = horizon
        
        # 输入嵌入层
        self.embedding  = nn.Linear(input_size, d_model)
        
        # 位置编码
        self.positional_encoding  = PositionalEncoding(d_model)
        
        # Transformer编码器
        encoder_layers = nn.TransformerEncoderLayer(d_model, nhead, dim_feedforward=4*d_model)
        self.transformer_encoder  = nn.TransformerEncoder(encoder_layers, num_layers)
        
        # 解码器
        self.decoder  = nn.Linear(d_model, output_size)
        
    def forward(self, x):
        # x形状: (batch_size, seq_len, input_size)
        x = self.embedding(x)   # (batch_size, seq_len, d_model)
        x = self.positional_encoding(x)   # (batch_size, seq_len, d_model)
        
        # 调整维度：Transformer期望 (seq_len, batch_size, d_model)
        x = x.permute(1,  0, 2)
        
        # 通过Transformer编码器
        x = self.transformer_encoder(x) 
        
        # 取最后一个时间步
        x = x[-1, :, :]  # (batch_size, d_model)
        
        # 解码
        x = self.decoder(x).view(-1,  self.horizon,  1)
        return x

# Transformer位置编码
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout  = nn.Dropout(p=0.1)
        
        position = torch.arange(max_len).unsqueeze(1) 
        div_term = torch.exp(torch.arange(0,  d_model, 2) * (-np.log(10000.0)  / d_model))
        pe = torch.zeros(max_len,  1, d_model)
        pe[:, 0, 0::2] = torch.sin(position  * div_term)
        pe[:, 0, 1::2] = torch.cos(position  * div_term)
        self.register_buffer('pe',  pe)
        
    def forward(self, x):
        # x形状: (batch_size, seq_len, d_model)
        x = x + self.pe[:x.size(1)].permute(1,  0, 2)
        return self.dropout(x) 
