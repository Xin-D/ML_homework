import numpy as np
import torch
import torch.nn  as nn
from torch.utils.data  import  DataLoader
from datetime import datetime
from data_factory import load_and_preprocess_data, PowerDataset
from LSTM import LSTMModel
from Transformer import TransformerModel
from TLTMixer import TLTMixer
from exp import train_model, evaluate_model, visualize_results
import warnings
warnings.filterwarnings('ignore') 

# 设置随机种子确保结果可复现
torch.manual_seed(42) 
np.random.seed(42)

# ======================
# 主函数
# ======================
def main():
    # 文件路径 - 请根据实际路径修改
    train_path = 'data/train.csv' 
    test_path = 'data/test.csv' 
    
    # 加载和预处理数据
    (train_x_seq90, train_y_seq90, test_x_seq90, test_y_seq90,
     train_x_seq365, train_y_seq365, test_x_seq365, test_y_seq365,
     scaler_y, feature_cols) = load_and_preprocess_data(train_path, test_path)
    
    print(f"训练数据短期序列形状: {train_x_seq90.shape},  {train_y_seq90.shape}") 
    print(f"测试数据短期序列形状: {test_x_seq90.shape},  {test_y_seq90.shape}") 
    print(f"训练数据长期序列形状: {train_x_seq365.shape},  {train_y_seq365.shape}") 
    print(f"测试数据长期序列形状: {test_x_seq365.shape},  {test_y_seq365.shape}") 
    print(f"特征列: {feature_cols}")
    
    # 创建数据集和数据加载器
    input_size = len(feature_cols)
    
    # 短期预测数据集 (90天)
    train_dataset90 = PowerDataset(train_x_seq90, train_y_seq90)
    test_dataset90 = PowerDataset(test_x_seq90, test_y_seq90)
    
    # 长期预测数据集 (365天)
    train_dataset365 = PowerDataset(train_x_seq365, train_y_seq365)
    test_dataset365 = PowerDataset(test_x_seq365, test_y_seq365)
    
    # 数据加载器
    batch_size = 32
    train_loader90 = DataLoader(train_dataset90, batch_size=batch_size, shuffle=True)
    test_loader90 = DataLoader(test_dataset90, batch_size=batch_size, shuffle=False)
    
    train_loader365 = DataLoader(train_dataset365, batch_size=batch_size, shuffle=True)
    test_loader365 = DataLoader(test_dataset365, batch_size=batch_size, shuffle=False)
    
    # 定义模型和训练参数
    models = {
        'LSTM': {
            'short': LSTMModel(input_size, hidden_size=512, num_layers=2, 
                              output_size=90, horizon=90),
            'long': LSTMModel(input_size, hidden_size=1024, num_layers=2, 
                             output_size=365, horizon=365)
        },
        'Transformer': {
            'short': TransformerModel(input_size, d_model=128, nhead=4, num_layers=3, 
                                     output_size=90, horizon=90),
            'long': TransformerModel(input_size, d_model=128, nhead=4, num_layers=3, 
                                    output_size=365, horizon=365)
        },
        'TLTMixer': {
            'short': TLTMixer(input_size, num_channels=[64, 64, 128, 256], kernel_size=3, 
                             output_size=1, horizon=90),
            'long': TLTMixer(input_size, num_channels=[64, 64, 128, 256], kernel_size=3, 
                            output_size=1, horizon=365)
        }
    }
    
    # 训练和评估参数
    criterion = nn.MSELoss()
    num_epochs = 150
    patience = 15
    num_experiments = 5
    
    # 存储结果
    results = {
        'short': {model_name: {'mse': [], 'mae': []} for model_name in models},
        'long': {model_name: {'mse': [], 'mae': []} for model_name in models}
    }
    
    # 进行多次实验
    for exp in range(num_experiments):
        print(f"\n=== 实验 {exp+1}/{num_experiments} ===")
        
        for model_name in models:
            print(f"\n训练 {model_name} 模型...")
            
            # 短期预测
            print("--- 短期预测 (90天) ---")
            model_short = models[model_name]['short']
            optimizer_short = torch.optim.Adam(model_short.parameters(),  lr=0.001)
            
            trained_model_short = train_model(model_short, train_loader90, test_loader90, 
                                            optimizer_short, criterion, num_epochs, patience)
            
            mse_short, mae_short, _, _ = evaluate_model(trained_model_short, test_loader90, 
                                                      criterion, scaler_y)
            
            results['short'][model_name]['mse'].append(mse_short)
            results['short'][model_name]['mae'].append(mae_short)
            print(f"[短期] {model_name} - MSE: {mse_short:.4f}, MAE: {mae_short:.4f}")
            
            # 长期预测
            print("\n--- 长期预测 (365天) ---")
            model_long = models[model_name]['long']
            optimizer_long = torch.optim.Adam(model_long.parameters(),  lr=0.0005)
            
            trained_model_long = train_model(model_long, train_loader365, test_loader365, 
                                           optimizer_long, criterion, num_epochs, patience)
            
            mse_long, mae_long, _, _ = evaluate_model(trained_model_long, test_loader365, 
                                                    criterion, scaler_y)
            
            results['long'][model_name]['mse'].append(mse_long)
            results['long'][model_name]['mae'].append(mse_long)
            print(f"[长期] {model_name} - MSE: {mse_long:.4f}, MAE: {mae_long:.4f}")
            
            # 可视化最佳模型预测结果
            visualize_results(trained_model_short, test_loader90, scaler_y, "{model_name} short term forecast".format(model_name=model_name))
            visualize_results(trained_model_long, test_loader365, scaler_y, "{model_name} long term forecast".format(model_name=model_name))
    
    # 计算平均结果和标准差
    print("\n最终结果:")
    for horizon in ['short', 'long']:
        print(f"\n=== {horizon.upper()} 预测结果 ===")
        for model_name in models:
            mse_list = results[horizon][model_name]['mse']
            mae_list = results[horizon][model_name]['mae']
            
            mse_mean = np.mean(mse_list) 
            mse_std = np.std(mse_list) 
            mae_mean = np.mean(mae_list) 
            mae_std = np.std(mae_list) 
            
            print(f"{model_name}模型:")
            print(f"  MSE: {mse_mean:.4f} ± {mse_std:.4f}")
            print(f"  MAE: {mae_mean:.4f} ± {mae_std:.4f}")
            # 写入文件
            with open(f'Result_Record/' + '{model_name} {horizon} term forecast.txt'.format(model_name=model_name,
                                            horizon=horizon,timestamp=datetime.now().strftime("%Y_%m_%d_%H_%M")), 'a+') as f:
                f.write(f'{model_name}_{horizon}:\n'.format(model_name=model_name))
                f.write(f'Average Train MSE: {mse_mean}, Std: {mse_std}\n')
                f.write(f'Average Train MAE: {mae_mean}, Std: {mae_std}\n')


if __name__ == "__main__":
    main()