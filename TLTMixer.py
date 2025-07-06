import numpy as np
import torch
import torch.nn  as nn
import warnings
warnings.filterwarnings('ignore') 


# ======================
# TCN 改进模型 (时间卷积网络)TCN + LSTM + Transformer
# ======================
class TLTMixer(nn.Module):
    def __init__(self, input_size, num_channels, kernel_size, output_size, horizon,
                 lstm_hidden=128, nhead=8, num_layers=3):
        super().__init__()
        self.horizon  = horizon
        
        # ---- 1. TCN模块 ----
        self.conv_init  = nn.Conv1d(input_size, num_channels[0], kernel_size, 
                                 padding=(kernel_size-1)//2)

        tcn_layers = []
        for i in range(1, len(num_channels)):
            dilation = 2 ** i 
            padding = (kernel_size - 1) * dilation // 2 
            tcn_layers.extend([ 
                nn.Conv1d(num_channels[i-1], num_channels[i], kernel_size,
                            padding=padding, dilation=dilation),
                nn.BatchNorm1d(num_channels[i]),
                nn.GELU(),
                nn.Dropout(0.1)
            ])
        self.tcn  = nn.Sequential(*tcn_layers)
        
        # ---- 2. LSTM模块 ----
        self.lstm  = nn.LSTM(
            input_size=num_channels[-1],    # 256
            hidden_size=lstm_hidden,
            num_layers=2,
            bidirectional=True,
            batch_first=True 
        )
        
        # ---- 3. Transformer模块 ----
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=lstm_hidden*2,  # 双向LSTM输出 128*2
            nhead=nhead,
            dim_feedforward=4*lstm_hidden,  # 128*4
            dropout=0.1,
            batch_first=True 
        )
        self.transformer  = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # ---- 4. 多尺度融合 ----
        self.fusion  = nn.Sequential(
            nn.Linear(lstm_hidden*6, lstm_hidden),  # TCN + LSTM + Transformer特征拼接 
            nn.LayerNorm(lstm_hidden),
            nn.SiLU()
        )
        
        # ---- 5. 预测头 ----
        self.predictor  = nn.Linear(lstm_hidden, output_size * horizon)
 
    def forward(self, x):
        # 输入x形状: (batch, seq_len, input_size)
        
        # === TCN路径 === 
        tcn_out = x.permute(0, 2, 1)  # (batch, input_size, seq_len)
        tcn_out = self.tcn(self.conv_init(tcn_out)) 
        tcn_feat = tcn_out[:, :, -1]  # 取最终时间步 
        # print(tcn_out.shape)
        # === LSTM路径 === 
        lstm_in = tcn_out.permute(0,  2, 1)  # (batch, seq_len, channels)
        lstm_out, _ = self.lstm(lstm_in) 
        lstm_feat = lstm_out[:, -1, :]  # 最后时间步
        # print(lstm_out.shape)
        # === Transformer路径 === 
        trans_out = self.transformer(lstm_out) 
        trans_feat = trans_out[:, -1, :]
        # print(trans_out.shape)
        # === 特征融合 === 
        combined = torch.cat([tcn_feat,  lstm_feat, trans_feat], dim=1)
        # print(combined.shape)   # (32, 768)
        fused = self.fusion(combined)
        
        # === 多步预测 ===
        pred = self.predictor(fused).view(-1,  self.horizon,  1)
        return pred 