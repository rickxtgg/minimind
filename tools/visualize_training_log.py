import re
import matplotlib.pyplot as plt
import numpy as np
import os
import argparse
import pandas as pd
from pathlib import Path
from typing import Dict, List, Union, Optional

# 解析日志文本
def detect_outliers(data: List[float], threshold: float = 3) -> List[bool]:
    """使用Z-score方法检测异常值"""
    z_scores = np.abs((data - np.mean(data)) / np.std(data))
    return z_scores > threshold

def preprocess_data(data: Dict[str, List]) -> Dict[str, List]:
    """预处理数据，包括异常值检测和处理"""
    # 检测loss中的异常值
    loss_outliers = detect_outliers(np.array(data['loss']))
    
    # 移除异常值对应的数据点
    cleaned_data = {}
    for key in data.keys():
        cleaned_data[key] = [v for i, v in enumerate(data[key]) if not loss_outliers[i]]
    
    return cleaned_data

def export_to_csv(data: Dict[str, List], output_path: str):
    """将训练数据导出为CSV格式"""
    df = pd.DataFrame(data)
    df.to_csv(output_path, index=False)
    print(f"数据已导出到: {output_path}")

def parse_log(log_text: str) -> Dict[str, List]:
    pattern = r'Epoch:\[(\d+)/(\d+)\]\((\d+)/(\d+)\) loss:([\d.]+) lr:([\d.]+) epoch_Time:([\d.]+)min:'
    matches = re.findall(pattern, log_text)
    
    data = {
        'current_epoch': [],
        'total_epochs': [],
        'current_step': [],
        'total_steps': [],
        'loss': [],
        'learning_rate': [],
        'epoch_time': []
    }
    
    for match in matches:
        data['current_epoch'].append(int(match[0]))
        data['total_epochs'].append(int(match[1]))
        data['current_step'].append(int(match[2]))
        data['total_steps'].append(int(match[3]))
        data['loss'].append(float(match[4]))
        data['learning_rate'].append(float(match[5]))
        data['epoch_time'].append(float(match[6]))
    
    return data

# 创建可视化图表
def calculate_training_stats(data: Dict[str, List]) -> Dict[str, float]:
    """计算训练统计信息"""
    stats = {}
    
    # 计算平均训练速度（steps/minute）
    total_steps = max(data['current_step'])
    total_time = sum(data['epoch_time'])
    stats['avg_training_speed'] = total_steps / total_time if total_time > 0 else 0
    
    # 计算loss下降率
    if len(data['loss']) > 1:
        initial_loss = data['loss'][0]
        final_loss = data['loss'][-1]
        total_steps = len(data['loss'])
        stats['loss_decay_rate'] = (initial_loss - final_loss) / total_steps
    else:
        stats['loss_decay_rate'] = 0
    
    return stats

def visualize_training_metrics(data: Dict[str, List], output_path: str, style: str = 'seaborn'):
    # 设置中文字体
    plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei']
    plt.rcParams['axes.unicode_minus'] = False
    # 创建一个3x3的子图布局
    fig = plt.figure(figsize=(20, 15))
    # 更新seaborn样式写法
    if style == 'seaborn':
        style = 'seaborn-v0_8'
    plt.style.use(style)
    
    # 计算每个数据点的全局step
    global_steps = [(epoch - 1) * data['total_steps'][0] + step 
                   for epoch, step in zip(data['current_epoch'], data['current_step'])]
    
    # 1. Loss趋势图
    ax1 = plt.subplot(3, 3, 1)
    ax1.plot(global_steps, data['loss'], 'b-', label='Training Loss')
    ax1.set_xlabel('Global Steps')
    ax1.set_ylabel('Loss')
    ax1.set_title('Training Loss over Steps')
    ax1.grid(True)
    ax1.legend()
    
    # 2. Learning Rate趋势图
    ax2 = plt.subplot(3, 3, 2)
    ax2.plot(global_steps, data['learning_rate'], 'g-', label='Learning Rate')
    ax2.set_xlabel('Global Steps')
    ax2.set_ylabel('Learning Rate')
    ax2.set_title('Learning Rate Schedule')
    ax2.grid(True)
    ax2.legend()
    
    # 3. 每个Epoch的平均Loss
    ax3 = plt.subplot(3, 3, 3)
    epoch_losses = {}
    for epoch, loss in zip(data['current_epoch'], data['loss']):
        if epoch not in epoch_losses:
            epoch_losses[epoch] = []
        epoch_losses[epoch].append(loss)
    
    avg_losses = [np.mean(losses) for losses in epoch_losses.values()]
    ax3.plot(list(epoch_losses.keys()), avg_losses, 'r-o', label='Average Loss per Epoch')
    ax3.set_xlabel('Epoch')
    ax3.set_ylabel('Average Loss')
    ax3.set_title('Average Loss per Epoch')
    ax3.grid(True)
    ax3.legend()
    
    # 4. 每个Epoch的训练时间
    ax4 = plt.subplot(3, 3, 4)
    epoch_times = {}
    for epoch, time in zip(data['current_epoch'], data['epoch_time']):
        if epoch not in epoch_times:
            epoch_times[epoch] = []
        epoch_times[epoch].append(time)
    
    avg_times = [np.mean(times) for times in epoch_times.values()]
    ax4.bar(list(epoch_times.keys()), avg_times, alpha=0.7, color='purple')
    ax4.set_xlabel('Epoch')
    ax4.set_ylabel('Average Time (minutes)')
    ax4.set_title('Training Time per Epoch')
    ax4.grid(True)
    
    # 5. Loss方差图
    ax5 = plt.subplot(3, 3, 5)
    loss_variances = [np.var(losses) for losses in epoch_losses.values()]
    ax5.plot(list(epoch_losses.keys()), loss_variances, 'm-o', label='Loss Variance')
    ax5.set_xlabel('Epoch')
    ax5.set_ylabel('Loss Variance')
    ax5.set_title('Loss Variance per Epoch')
    ax5.grid(True)
    ax5.legend()
    
    # 6. 训练速度趋势图
    ax6 = plt.subplot(3, 3, 6)
    steps_per_epoch = data['total_steps'][0]
    training_speeds = [steps_per_epoch / time for time in avg_times]
    ax6.plot(list(epoch_times.keys()), training_speeds, 'c-o', label='Training Speed')
    ax6.set_xlabel('Epoch')
    ax6.set_ylabel('Steps per Minute')
    ax6.set_title('Training Speed Trend')
    ax6.grid(True)
    ax6.legend()
    
    # 7. Loss下降率变化图
    ax7 = plt.subplot(3, 3, 7)
    loss_decay_rates = []
    for i in range(1, len(avg_losses)):
        decay_rate = (avg_losses[i-1] - avg_losses[i])
        loss_decay_rates.append(decay_rate)
    
    if loss_decay_rates:
        epochs = list(epoch_losses.keys())[1:]
        ax7.plot(epochs, loss_decay_rates, 'y-o', label='Loss Decay Rate')
        ax7.set_xlabel('Epoch')
        ax7.set_ylabel('Loss Decay Rate')
        ax7.set_title('Loss Decay Rate Changes')
        ax7.grid(True)
        ax7.legend()
    
    # 8. Loss分布直方图
    ax8 = plt.subplot(3, 3, 8)
    ax8.hist(data['loss'], bins=50, alpha=0.7, color='orange')
    ax8.set_xlabel('Loss Value')
    ax8.set_ylabel('Frequency')
    ax8.set_title('Loss Distribution')
    ax8.grid(True)
    
    # 9. 训练进度图
    ax9 = plt.subplot(3, 3, 9)
    progress = [epoch/data['total_epochs'][0] * 100 for epoch in data['current_epoch']]
    ax9.plot(global_steps, progress, 'k-', label='Training Progress')
    ax9.set_xlabel('Global Steps')
    ax9.set_ylabel('Progress (%)')
    ax9.set_title('Training Progress')
    ax9.grid(True)
    ax9.legend()
    
    # 调整布局，为统计信息预留空间
    plt.tight_layout(rect=[0, 0.05, 1, 1])
    
    # 添加训练统计信息
    stats = calculate_training_stats(data)
    plt.figtext(0.02, 0.01, 
                f"Training Statistics:\n" \
                f"Average Training Speed: {stats['avg_training_speed']:.2f} steps/min\n" \
                f"Loss Decay Rate: {stats['loss_decay_rate']:.4f}/step",
                fontsize=9, ha='left', va='bottom')
    
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

def read_log_file(file_path):
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return f.read()
    except FileNotFoundError:
        raise FileNotFoundError(f"训练日志文件 '{file_path}' 不存在")
    except Exception as e:
        raise Exception(f"读取日志文件时发生错误: {str(e)}")

def validate_file_path(file_path):
    if not file_path:
        raise ValueError("文件路径不能为空")
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"训练日志文件 '{file_path}' 不存在")
    if not os.path.isfile(file_path):
        raise ValueError(f"'{file_path}' 不是一个有效的文件")
    return file_path

def get_log_file_path():
    while True:
        try:
            file_path = input("请输入训练日志文件路径: ").strip()
            return validate_file_path(file_path)
        except (FileNotFoundError, ValueError) as e:
            print(f"错误: {str(e)}")
            retry = input("是否重新输入? (y/n): ").strip().lower()
            if retry != 'y':
                return None

def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='训练日志可视化工具')
    parser.add_argument('log_file', nargs='?', help='训练日志文件路径')
    parser.add_argument('--output', '-o', default='training_metrics.png',
                        help='输出图片路径 (默认: training_metrics.png)')
    parser.add_argument('--style', '-s', default='seaborn',
                        choices=['seaborn', 'classic', 'dark_background', 'bmh', 'fast'],
                        help='图表样式 (默认: seaborn)')
    parser.add_argument('--export-csv', '-e',
                        help='导出训练数据到CSV文件')
    parser.add_argument('--no-outlier-detection', action='store_true',
                        help='禁用异常值检测')
    return parser.parse_args()

def main(log_file_path: str, output_path: str, style: str = 'seaborn',
         export_csv: Optional[str] = None, detect_outliers: bool = True):
    try:
        # 读取日志文件
        log_text = read_log_file(log_file_path)
        
        # 解析日志数据
        data = parse_log(log_text)
        
        # 数据预处理
        if detect_outliers:
            data = preprocess_data(data)
        
        # 导出CSV
        if export_csv:
            export_to_csv(data, export_csv)
        
        # 生成可视化图表
        visualize_training_metrics(data, output_path, style)
        print(f"可视化图表已保存为 'training_metrics.png'")
    except Exception as e:
        print(f"错误: {str(e)}")

if __name__ == "__main__":
    args = parse_args()
    
    if args.log_file:
        try:
            log_file_path = validate_file_path(args.log_file)
        except (FileNotFoundError, ValueError) as e:
            print(f"错误: {str(e)}")
            sys.exit(1)
    else:
        print("欢迎使用训练日志可视化工具!")
        log_file_path = get_log_file_path()
    
    if log_file_path:
        main(log_file_path,
             output_path=args.output,
             style=args.style,
             export_csv=args.export_csv,
             detect_outliers=not args.no_outlier_detection)
    else:
        print("程序已退出")
        sys.exit(0)