import os
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.interpolate import CubicSpline, BSpline, splrep
from scipy.signal import savgol_filter
from scipy.ndimage import gaussian_filter1d

def read_actions_file(file_path):
    """读取actions.txt文件并解析动作数据"""
    actions = []
    with open(file_path, 'r') as f:
        lines = f.readlines()
        
        # 跳过头部注释行
        start_idx = 0
        for i, line in enumerate(lines):
            if line.startswith('#'):
                start_idx = i + 1
            else:
                break
                
        # 解析动作数据
        for line in lines[start_idx:]:
            parts = line.strip().split(',')
            if len(parts) < 2:
                continue
            action_values = [float(val) for val in parts[1:]]
            actions.append(action_values)
    
    return np.array(actions)

def cubic_spline_smoothing(sampled_times, sampled_actions, target_times):
    """使用三次样条插值平滑轨迹"""
    smoothed_actions = np.zeros((len(target_times), sampled_actions.shape[1]))
    
    for dim in range(sampled_actions.shape[1]):
        # 对每个dimension应用三次样条插值
        cs = CubicSpline(sampled_times, sampled_actions[:, dim])
        smoothed_actions[:, dim] = cs(target_times)
    
    return smoothed_actions

def savgol_smoothing(actions, window_length=15, poly_order=3):
    """使用Savitzky-Golay滤波平滑轨迹"""
    # 窗口长度必须是奇数且小于数据长度
    window_length = min(window_length, len(actions) - 1)
    if window_length % 2 == 0:
        window_length -= 1
    window_length = max(window_length, 3)  # 至少为3
    
    # 多项式阶数必须小于窗口长度
    poly_order = min(poly_order, window_length - 1)
    
    # 应用滤波器
    smoothed_actions = np.zeros_like(actions)
    for dim in range(actions.shape[1]):
        smoothed_actions[:, dim] = savgol_filter(
            actions[:, dim], window_length, poly_order)
    
    return smoothed_actions

def gaussian_smoothing(actions, sigma=2.0):
    """使用高斯滤波平滑轨迹"""
    smoothed_actions = np.zeros_like(actions)
    
    for dim in range(actions.shape[1]):
        smoothed_actions[:, dim] = gaussian_filter1d(actions[:, dim], sigma)
    
    return smoothed_actions

def b_spline_smoothing(sampled_times, sampled_actions, target_times, degree=3):
    """使用B样条曲线平滑轨迹"""
    smoothed_actions = np.zeros((len(target_times), sampled_actions.shape[1]))
    
    for dim in range(sampled_actions.shape[1]):
        # 计算B样条表示
        tck = splrep(sampled_times, sampled_actions[:, dim], k=degree)
        # 生成平滑轨迹
        smoothed_actions[:, dim] = BSpline(*tck)(target_times)
    
    return smoothed_actions

def plot_sampling_smoothing_comparison(actions, task_name, output_dir):
    """比较不同采样步长和平滑方法的效果"""
    action_dim = actions.shape[1]
    seq_length = actions.shape[0]
    
    # 创建时间步序列
    time_steps = np.arange(seq_length)
    
    # 采样步长为5
    sampling_stride = 5
    sampled_indices = np.arange(0, seq_length, sampling_stride)
    sampled_time_steps = time_steps[sampled_indices]
    sampled_actions = actions[sampled_indices]
    
    # 生成平滑轨迹的时间点
    smooth_time_steps = np.linspace(0, seq_length-1, 200)
    
    # 应用不同平滑算法
    cubic_spline_smooth = cubic_spline_smoothing(
        sampled_time_steps, sampled_actions, smooth_time_steps)
    
    savgol_smooth = savgol_smoothing(actions)
    
    gaussian_smooth = gaussian_smoothing(actions)
    
    b_spline_smooth = b_spline_smoothing(
        sampled_time_steps, sampled_actions, smooth_time_steps)
    
    # 为每个动作dimension创建比较图
    os.makedirs(output_dir, exist_ok=True)
    
    # 绘制综合比较图
    plt.figure(figsize=(18, 12))
    
    # 1. 原始数据与采样数据比较
    plt.subplot(3, 2, 1)
    for i in range(action_dim):
        plt.plot(time_steps, actions[:, i], '-', label=f'dimension {i}')
    plt.title(f'original ({seq_length} step)')
    plt.xlabel('step')
    plt.ylabel('action_value')
    plt.grid(True)
    plt.legend(loc='best')
    
    plt.subplot(3, 2, 2)
    for i in range(action_dim):
        plt.plot(sampled_time_steps, sampled_actions[:, i], 'o-', label=f'dimension {i}')
    plt.title(f'sampling_traj (step={sampling_stride})')
    plt.xlabel('step')
    plt.ylabel('action_value')
    plt.grid(True)
    plt.legend(loc='best')
    
    # 2. 不同平滑算法比较
    plt.subplot(3, 2, 3)
    for i in range(action_dim):
        plt.plot(smooth_time_steps, cubic_spline_smooth[:, i], '-', label=f'dimension {i}')
    plt.title('three_cubic_spline')
    plt.xlabel('step')
    plt.ylabel('action_value')
    plt.grid(True)
    plt.legend(loc='best')
    
    plt.subplot(3, 2, 4)
    for i in range(action_dim):
        plt.plot(time_steps, savgol_smooth[:, i], '-', label=f'dimension {i}')
    plt.title('Savitzky-Golay filter')
    plt.xlabel('step')
    plt.ylabel('action_value')
    plt.grid(True)
    plt.legend(loc='best')
    
    plt.subplot(3, 2, 5)
    for i in range(action_dim):
        plt.plot(time_steps, gaussian_smooth[:, i], '-', label=f'dimension {i}')
    plt.title('Gaussian filter')
    plt.xlabel('step')
    plt.ylabel('action_value')
    plt.grid(True)
    plt.legend(loc='best')
    
    plt.subplot(3, 2, 6)
    for i in range(action_dim):
        plt.plot(smooth_time_steps, b_spline_smooth[:, i], '-', label=f'dimension {i}')
    plt.title('B-spline')
    plt.xlabel('step')
    plt.ylabel('action_value')
    plt.grid(True)
    plt.legend(loc='best')
    
    plt.suptitle(f'traj\ntask: {task_name}', fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    
    # 保存综合比较图
    output_file = os.path.join(output_dir, "trajectory_smoothing_comparison.png")
    plt.savefig(output_file, dpi=300)
    plt.close()
    
    # 为每个dimension生成单独的对比图
    for i in range(action_dim):
        plt.figure(figsize=(15, 10))
        
        # 原始与采样数据
        plt.subplot(2, 1, 1)
        plt.plot(time_steps, actions[:, i], 'b-', label='original')
        plt.plot(sampled_time_steps, sampled_actions[:, i], 'ro-', label=f'step={sampling_stride}采样')
        plt.title(f'dimension {i} - original_vs_sampling')
        plt.xlabel('step')
        plt.ylabel('action_value')
        plt.grid(True)
        plt.legend()
        
        # 平滑算法比较
        plt.subplot(2, 1, 2)
        plt.plot(time_steps, actions[:, i], 'k-', alpha=0.5, label='original')
        plt.plot(sampled_time_steps, sampled_actions[:, i], 'ko', alpha=0.5, label='sampling')
        plt.plot(smooth_time_steps, cubic_spline_smooth[:, i], 'r-', label='three_cubic_spline')
        plt.plot(time_steps, savgol_smooth[:, i], 'g-', label='Savitzky-Golay filter')
        plt.plot(time_steps, gaussian_smooth[:, i], 'b-', label='Gaussian filter')
        plt.plot(smooth_time_steps, b_spline_smooth[:, i], 'm-', label='B-spline')
        plt.title(f'dimension {i} - traj_plan_algorithm_comparison')
        plt.xlabel('step')
        plt.ylabel('action_value')
        plt.grid(True)
        plt.legend()
        
        plt.suptitle(f'dimension {i} traj_plan_comparision\ntask: {task_name}', fontsize=14)
        plt.tight_layout(rect=[0, 0, 1, 0.95])
        
        dim_output_file = os.path.join(output_dir, f"dimension_{i}_smoothing.png")
        plt.savefig(dim_output_file, dpi=300)
        plt.close()
    
    return output_file

def main():
    # 查找所有记录文件夹
    base_dir = 'extracted_data'
    output_dir = 'trajectory_smoothing'
    os.makedirs(output_dir, exist_ok=True)
    
    record_dirs = [d for d in os.listdir(base_dir) if d.startswith('record_') and os.path.isdir(os.path.join(base_dir, d))]
    
    for record_dir in record_dirs:
        # 提取任务名称
        parts = record_dir.split('_', 2)
        if len(parts) >= 3:
            record_id = parts[1]
            task_name = parts[2]
        else:
            record_id = record_dir.split('_')[1]
            task_name = f"记录 {record_id}"
        
        # 读取动作文件
        action_file = os.path.join(base_dir, record_dir, 'actions.txt')
        if os.path.exists(action_file):
            print(f"处理记录 {record_id}: {task_name}")
            actions = read_actions_file(action_file)
            
            # 创建该记录的输出目录
            record_output_dir = os.path.join(output_dir, f"record_{record_id}")
            os.makedirs(record_output_dir, exist_ok=True)
            
            # 绘制并保存比较图
            output_file = plot_sampling_smoothing_comparison(actions, task_name, record_output_dir)
            print(f"平滑比较图已保存到: {output_file}")
            print("-" * 50)
        else:
            print(f"找不到记录 {record_id} 的动作文件")

if __name__ == "__main__":
    main()