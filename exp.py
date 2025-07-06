import numpy as np
import matplotlib.pyplot  as plt
import torch
from sklearn.metrics  import mean_squared_error, mean_absolute_error
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
# 训练和评估函数
# ======================
def train_model(model, train_loader, val_loader, optimizer, criterion, num_epochs=100, patience=10):
    model.to(device) 
    best_val_loss = float('inf')
    best_model = None
    counter = 0
    
    for epoch in range(num_epochs):
        model.train() 
        train_loss = 0.0
        
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device),  targets.to(device) 
            
            optimizer.zero_grad() 
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward() 
            optimizer.step() 
            
            train_loss += loss.item()  * inputs.size(0) 
        
        train_loss /= len(train_loader.dataset) 
        
        # 验证
        model.eval() 
        val_loss = 0.0
        with torch.no_grad(): 
            for inputs, targets in val_loader:
                inputs, targets = inputs.to(device),  targets.to(device) 
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                val_loss += loss.item()  * inputs.size(0) 
        
        val_loss /= len(val_loader.dataset) 
        
        # 早停
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model = copy.deepcopy(model.state_dict()) 
            counter = 0
        else:
            counter += 1
            if counter >= patience:
                print(f"Early stopping at epoch {epoch+1}")
                break
        
        print(f"Epoch {epoch+1}/{num_epochs} - Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}")
    
    # 加载最佳模型
    model.load_state_dict(best_model) 
    return model

def evaluate_model(model, test_loader, criterion, scaler_y):
    model.eval() 
    total_mse = 0.0
    total_mae = 0.0
    all_targets = []
    all_predictions = []
    
    with torch.no_grad(): 
        for inputs, targets in test_loader:
            inputs, targets = inputs.to(device),  targets.to(device) 
            outputs = model(inputs)
            
            # 反标准化
            outputs_np = outputs.cpu().numpy().reshape(-1,  outputs.shape[1]) 
            targets_np = targets.cpu().numpy().reshape(-1,  targets.shape[1]) 
            
            outputs_inv = outputs_np
            targets_inv = targets_np
            # 计算指标
            mse = mean_squared_error(targets_inv, outputs_inv)
            mae = mean_absolute_error(targets_inv, outputs_inv)
            
            total_mse += mse * inputs.size(0) 
            total_mae += mae * inputs.size(0) 
            
            all_targets.extend(targets_inv.tolist()) 
            all_predictions.extend(outputs_inv.tolist()) 
    
    total_mse /= len(test_loader.dataset) 
    total_mae /= len(test_loader.dataset) 
    
    return total_mse, total_mae, np.array(all_targets),  np.array(all_predictions) 

def visualize_results(model, test_loader, scaler_y, title):
    model.eval() 
    with torch.no_grad(): 
        # 获取一个批次的数据
        inputs, targets = next(iter(test_loader))
        inputs, targets = inputs.to(device),  targets.to(device) 
        outputs = model(inputs)
        
        # 选择一个样本进行可视化
        idx = 15
        input_seq = inputs[idx].cpu().numpy()
        target_seq = targets[idx].cpu().numpy()
        output_seq = outputs[idx].cpu().numpy()
        
        # 反标准化
        input_seq_inv = scaler_y.inverse_transform(input_seq[:,  0].reshape(-1, 1))
        target_seq_inv = scaler_y.inverse_transform(target_seq.reshape(-1,  1))
        output_seq_inv = scaler_y.inverse_transform(output_seq.reshape(-1,  1))
        
        # 创建时间轴
        target_time = np.arange(1,  len(target_seq)+1)
        
        # 绘图
        plt.figure(figsize=(12,  6))
        plt.plot(target_time,  target_seq_inv, 'g-', label='true data')
        plt.plot(target_time,  output_seq_inv, 'r-', label='predictied data')
        plt.xlabel('time (days)')
        plt.ylabel('Global active power(kW)')
        plt.title(title)
        plt.legend()
        plt.grid(True)
        plt.savefig('Result_fig/' + title + '.png')