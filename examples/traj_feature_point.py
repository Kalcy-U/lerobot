import os
import threading
import time
import mujoco
import mujoco.viewer
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.interpolate import CubicSpline, BSpline, splrep
from scipy.signal import savgol_filter
from scipy.ndimage import gaussian_filter1d
import pandas as pd
from lerobot_kinematics import lerobot_IK_5DOF, get_robot, so100_FK,so100_IK
from examples.traj_plan import mujoco_follow_traj,read_actions_file,gaussian_smoothing,linear_interpolation

def point_line_distance(point, start, end):
    """计算点到直线的距离"""
    if np.all(start == end):
        return np.linalg.norm(point - start)
    
    line_vec = end - start
    point_vec = point - start
    line_len = np.linalg.norm(line_vec)
    line_unitvec = line_vec / line_len
    point_vec_scaled = point_vec / line_len
    
    t = np.dot(line_unitvec, point_vec_scaled)
    t = max(0, min(1, t))  # 确保投影点在线段上
    nearest = start + t * line_vec
    
    return np.linalg.norm(point - nearest)

def rdp_reduce(points, epsilon, mask=None):
    """RDP算法的递归实现"""
    if mask is None:
        mask = np.zeros(len(points), dtype=bool)
        mask[0] = mask[-1] = True  # 总是保留首尾点
    
    # 找出距离最远的点
    dmax = 0
    index = -1
    start = points[0]
    end = points[-1]
    
    for i in range(1, len(points) - 1):
        if not mask[i]:
            d = point_line_distance(points[i], start, end)
            if d > dmax:
                index = i
                dmax = d
    
    # 如果最大距离大于epsilon，则递归处理
    if dmax > epsilon:
        mask[index] = True
        # 递归处理两个子区间
        rdp_reduce(points[:index + 1], epsilon, mask[:index + 1])
        rdp_reduce(points[index:], epsilon, mask[index:])
    
    return mask

def rdp_smoothing(actions, precision=0.05):
    """对每个维度应用RDP算法"""
    seq_length = len(actions)
    time_steps = np.arange(seq_length)
    reduced_actions = []
    masks = []
    
    # 对每个维度分别应用RDP
    for dim in range(actions.shape[1]):
        points = np.column_stack((time_steps, actions[:, dim]))
        # 根据action中最大最小的差值*precision决定rdp的阈值参数
        epsilon = (np.max(actions[:, dim]) - np.min(actions[:, dim])) * precision
            
        mask = rdp_reduce(points, epsilon)
        masks.append(mask)
        reduced_points = points[mask]
        reduced_actions.append(reduced_points)
    
    # 转置
    
    masks = np.array(masks).T
    return reduced_actions, masks

def enhance_curvature_points(actions, masks, curvature_threshold=0.1, points_to_add=1):
    """
    检查曲线的二阶导，在二阶导数大的RDP采样点附近添加额外的点
    
    参数:
        actions: 原始动作序列，形状为 (time_steps, dimensions)
        masks: 布尔掩码数组，指示哪些点被RDP保留
        curvature_threshold: 二阶导数阈值，超过此值的点被视为大曲率点
        points_to_add: 在大曲率点附近添加的点数
        
    返回:
        enhanced_reduced_actions: 增强后的简化动作序列列表
        enhanced_masks: 增强后的掩码数组
    """
    seq_length = len(actions)
    time_steps = np.arange(seq_length)
    enhanced_reduced_actions = []
    enhanced_masks = np.copy(masks)
    
    for dim in range(actions.shape[1]):
        # 计算原始序列的二阶导数（使用差分近似）
        first_deriv = np.gradient(actions[:, dim], time_steps)
        second_deriv = np.gradient(first_deriv, time_steps)
        second_deriv_abs = np.abs(second_deriv)
        
        # 根据二阶导数的范围计算阈值
        curvature_min = np.min(second_deriv_abs)
        curvature_max = np.max(second_deriv_abs)
        curvature_threshold = curvature_min + (curvature_max - curvature_min) * curvature_threshold
        
        # 获取当前维度RDP保留的点索引
        rdp_indices = np.where(masks[:, dim])[0]
        
        # 找出具有大二阶导的RDP点（排除首尾点）
        high_curvature_indices = []
        for idx in rdp_indices[1:-1]:  # 跳过首尾点
            if second_deriv_abs[idx] > curvature_threshold:
                high_curvature_indices.append(idx)
        
        # 在高曲率点周围添加点
        for hc_idx in high_curvature_indices:
            # 在曲率点左侧添加点
            for i in range(1, points_to_add + 1):
                i = i * 10
                target_idx = max(0, hc_idx - i)
                enhanced_masks[target_idx, dim] = True
                target_idx = min(seq_length - 1, hc_idx + i)
                enhanced_masks[target_idx, dim] = True
    
    # 根据增强后的掩码重建reduced_actions
    enhanced_reduced_actions = []
    for dim in range(actions.shape[1]):
        points = np.column_stack((time_steps, actions[:, dim]))
        indices = np.where(enhanced_masks[:, dim])[0]
        enhanced_reduced_actions.append(points[indices])
    
    return enhanced_reduced_actions, enhanced_masks

def rdp_smoothing_enhanced(actions, precision=0.05, curvature_threshold=0.1, points_to_add=2):
    """
    使用RDP算法简化轨迹并在高曲率区域增强采样点
    
    参数:
        actions: 原始动作序列
        precision: RDP算法的精度参数
        curvature_threshold: 二阶导数阈值，超过此值的点被视为大曲率点
        points_to_add: 在大曲率点附近添加的点数
        
    返回:
        enhanced_reduced_actions: 增强后的简化动作序列列表
        enhanced_masks: 增强后的掩码数组
    """
    # 先应用常规RDP算法
    reduced_actions, masks = rdp_smoothing(actions, precision)
    
    # 然后增强高曲率区域的采样点
    enhanced_reduced_actions, enhanced_masks = enhance_curvature_points(
        actions, masks, curvature_threshold, points_to_add
    )
    
    return enhanced_reduced_actions, enhanced_masks

def plot_rdp_results(actions, task_name, output_dir, precision=0.05,curvature_threshold=0.1,points_to_add=2):
    """可视化RDP结果"""
    if isinstance(actions, list):
        actions = np.stack(actions)
    
    
    
    # 使用增强版的RDP算法
    reduced_actions, masks = rdp_smoothing_enhanced(actions, precision = precision ,curvature_threshold=curvature_threshold,points_to_add=points_to_add)
    seq_length = len(actions)
    time_steps = np.arange(seq_length)
    smooth_time_steps = np.linspace(0, seq_length-1, seq_length)
    cubic_smoothed = np.zeros((len(smooth_time_steps), actions.shape[1]))
    
    # 对每个维度应用三次样条插值
    for dim in range(actions.shape[1]):
        reduced_points = reduced_actions[dim]
        # 用RDP简化后的点拟合三次样条
        x=np.where(masks[:,dim])[0]
        cs = CubicSpline(x, reduced_points[:, 1],bc_type=((1,0),(1,0)))
        # 在全部时间点上计算样条值
        cubic_smoothed[:, dim] = cs(smooth_time_steps)
        
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    # 绘制所有维度的综合图
    plt.figure(figsize=(15, 10))
    
    # 原始轨迹
    plt.subplot(2, 1, 1)
    for i in range(actions.shape[1]):
        plt.plot(time_steps, actions[:, i], '-', label=f'dimension {i}')
    plt.title('Original Trajectory')
    plt.xlabel('step')
    plt.ylabel('action_value')
    plt.grid(True)
    plt.legend()
    
    # RDP简化后的轨迹
    plt.subplot(2, 1, 2)
    for i in range(actions.shape[1]):
        reduced_points = reduced_actions[i]
        plt.plot(reduced_points[:, 0], reduced_points[:, 1], 'o-', 
                label=f'dimension {i} (points: {len(reduced_points)})')
        
        plt.plot(smooth_time_steps, cubic_smoothed[:, i], 'r-', label='three_cubic_spline')
        
       
    plt.title(f'RDP Simplified Trajectory (precision={precision})')
    plt.xlabel('step')
    plt.ylabel('action_value')
    plt.grid(True)
    plt.legend()
    
    plt.suptitle(f'RDP Trajectory Simplification\ntask: {task_name}', fontsize=14)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    
    # 保存综合图
    output_file = os.path.join(output_dir, "rdp_comparison.png")
    plt.savefig(output_file, dpi=300)
    plt.close()
    
    # 为每个维度生成单独的对比图
    for i in range(actions.shape[1]):
        plt.figure(figsize=(15, 8))
        reduced_points = reduced_actions[i]
        
        plt.plot(time_steps, actions[:, i], 'b-', label='Original', alpha=0.6)
        plt.plot(reduced_points[:, 0], reduced_points[:, 1], 'ro', 
                label=f'RDP (points: {len(reduced_points)})')
        plt.plot(smooth_time_steps, cubic_smoothed[:, i], 'r-', label='three_cubic_spline')
        
        plt.title(f'Dimension {i} - RDP Simplification')
        plt.xlabel('step')
        plt.ylabel('action_value')
        plt.grid(True)
        plt.legend()
        
        dim_output_file = os.path.join(output_dir, f"dimension_{i}_rdp.png")
        plt.savefig(dim_output_file, dpi=300)
        plt.close()
    
    return output_file


def plot_rdp_comparison(actions, task_name, output_dir, precision=0.01, curvature_threshold=0.1, points_to_add=2):
    """比较原始RDP和增强RDP的结果"""
    if isinstance(actions, list):
        actions = np.stack(actions)
    
    # 获取原始RDP和增强RDP的结果
    original_reduced_actions, original_masks = rdp_smoothing(actions, precision)
    enhanced_reduced_actions, enhanced_masks = rdp_smoothing_enhanced(
        actions, precision, curvature_threshold, points_to_add
    )
    
    seq_length = len(actions)
    time_steps = np.arange(seq_length)
    smooth_time_steps = np.linspace(0, seq_length-1, seq_length)
    
    # 计算原始RDP和增强RDP的样条插值结果
    original_cubic_smoothed = np.zeros((len(smooth_time_steps), actions.shape[1]))
    enhanced_cubic_smoothed = np.zeros((len(smooth_time_steps), actions.shape[1]))
    
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    # 为每个维度生成对比图
    for dim in range(actions.shape[1]):
        plt.figure(figsize=(15, 10))
        
        # 原始轨迹
        plt.subplot(3, 1, 1)
        plt.plot(time_steps, actions[:, dim], 'b-', label='Original Trajectory')
        plt.title(f'Dimension {dim} - Original Trajectory')
        plt.xlabel('Step')
        plt.ylabel('Value')
        plt.grid(True)
        plt.legend()
        
        # 原始RDP
        plt.subplot(3, 1, 2)
        original_points = original_reduced_actions[dim]
        x_orig = np.where(original_masks[:, dim])[0]
        cs_orig = CubicSpline(x_orig, original_points[:, 1], bc_type=((1, 0), (1, 0)))
        original_cubic_smoothed[:, dim] = cs_orig(smooth_time_steps)
        
        plt.plot(time_steps, actions[:, dim], 'b-', alpha=0.3, label='Original Trajectory')
        plt.plot(original_points[:, 0], original_points[:, 1], 'ro', 
                label=f'Original RDP Points ({len(original_points)} points)')
        plt.plot(smooth_time_steps, original_cubic_smoothed[:, dim], 'r-', 
                label='Original RDP Cubic Spline')
        plt.title(f'Dimension {dim} - Original RDP (epsilon={precision})')
        plt.xlabel('Step')
        plt.ylabel('Value')
        plt.grid(True)
        plt.legend()
        
        # 增强RDP
        plt.subplot(3, 1, 3)
        enhanced_points = enhanced_reduced_actions[dim]
        x_enh = np.where(enhanced_masks[:, dim])[0]
        cs_enh = CubicSpline(x_enh, enhanced_points[:, 1], bc_type=((1, 0), (1, 0)))
        enhanced_cubic_smoothed[:, dim] = cs_enh(smooth_time_steps)
        
        plt.plot(time_steps, actions[:, dim], 'b-', alpha=0.3, label='Original Trajectory')
        plt.plot(enhanced_points[:, 0], enhanced_points[:, 1], 'go', 
                label=f'Enhanced RDP Points ({len(enhanced_points)} points)')
        plt.plot(smooth_time_steps, enhanced_cubic_smoothed[:, dim], 'g-', 
                label='Enhanced RDP Cubic Spline')
        
        
        plt.title(f'Dimension {dim} - Enhanced RDP (Threshold={curvature_threshold}, Points to Add={points_to_add})')
        plt.xlabel('Step')
        plt.ylabel('Value')
        plt.grid(True)
        plt.legend()
        
        plt.tight_layout()
        plt.suptitle(f'Dimension {dim} - RDP Comparison\nTask: {task_name}', fontsize=14)
        plt.subplots_adjust(top=0.9)
        
        # 保存对比图
        comparison_file = os.path.join(output_dir, f"dimension_{dim}_rdp_comparison.png")
        plt.savefig(comparison_file, dpi=300)
        plt.close()
    
    # 计算并绘制曲率图
    plt.figure(figsize=(15, 5 * actions.shape[1]))
    for dim in range(actions.shape[1]):
        plt.subplot(actions.shape[1], 1, dim + 1)
        
        # 计算二阶导数
        first_deriv = np.gradient(actions[:, dim], time_steps)
        second_deriv = np.gradient(first_deriv, time_steps)
        second_deriv_abs = np.abs(second_deriv)
        
        plt.plot(time_steps, second_deriv_abs, 'b-', label='|Second Derivative|')
        plt.axhline(y=curvature_threshold, color='r', linestyle='--', 
                   label=f'Threshold ({curvature_threshold})')
        
        # 标记高曲率点
        high_curvature_mask = second_deriv_abs > curvature_threshold
        high_curvature_points = time_steps[high_curvature_mask]
        high_curvature_values = second_deriv_abs[high_curvature_mask]
        
        if len(high_curvature_points) > 0:
            plt.scatter(high_curvature_points, high_curvature_values, 
                      color='red', s=50, marker='o', 
                      label='Points Above Threshold')
        
        plt.title(f'Dimension {dim} - Second Derivative Magnitude')
        plt.xlabel('Step')
        plt.ylabel('|Second Derivative|')
        plt.grid(True)
        plt.legend()
    
    plt.tight_layout()
    curvature_file = os.path.join(output_dir, "curvature_analysis.png")
    plt.savefig(curvature_file, dpi=300)
    plt.close()
    
    return enhanced_reduced_actions, enhanced_masks


def compute_average_speed(g_actions: np.ndarray, delta_t: float = 1.0) -> float:
    """
    计算轨迹中平均速度。

    参数:
        g_actions: t x 6 数组，其中前三列是位置 (x, y, z)
        delta_t: 每两帧之间的时间间隔，默认为 1.0 单位时间

    返回:
        平均速度 (标量)
    """
    positions = g_actions[:, :3]
    displacements = np.linalg.norm(np.diff(positions, axis=0), axis=1)
    total_distance = np.sum(displacements)
    total_time = delta_t * len(displacements)
    avg_speed = total_distance / total_time
    return avg_speed
def allocate_sampled_t(sampled_a: np.ndarray, avg_speed: float, min_range: float) -> np.ndarray:
    """
    为一组采样点 sampled_a 分配时间戳 sampled_t。
    
    参数:
        sampled_a: n x m 数组，前3列表示每个采样点的 xyz 坐标
        avg_speed: 平均速度（单位距离 / 单位时间）
        min_range: 每段最小时间间隔

    返回:
        sampled_t: n 元素数组，表示每个采样点的时间戳
    """
    sampled_a=sampled_a[:,0:3]
    n = sampled_a.shape[0]
    sampled_t = [0]
    last_t = 0.0
    for i in range(1, n):
        dist = np.linalg.norm(sampled_a[i] - sampled_a[i - 1])
        delta_t = max(dist / avg_speed, min_range)
        last_t = last_t+delta_t
        sampled_t.append(int(np.round(last_t)))
    
    return np.array(sampled_t)
def test_rdp():
    """Test RDP Algorithm"""
    parquet_file='/data/tyn/so100_grasp/data/chunk-000/episode_000001.parquet'
    q_actions = read_actions_file(parquet_file)

    g_actions = np.apply_along_axis(so100_FK, 1, q_actions)
    print(len(g_actions))
    actions,masks=rdp_smoothing(g_actions,precision=0.05)
    
    mask = np.any(masks,axis=1)
    
    key_points = g_actions[mask]
    
    speed= compute_average_speed(g_actions)
    print(speed)
    sampled_t=allocate_sampled_t(key_points,speed,1)
    target_t = np.linspace(0, sampled_t[-1]-1, sampled_t[-1])
    print(len(sampled_t))
    print(len(key_points))
    print(sampled_t[-1])
    result = linear_interpolation(sampled_t, key_points, target_t)
   
    smooth = gaussian_smoothing(result)
    print(len(smooth))
    
    qpos_list=[]
    robot = get_robot()
    last_qpos = np.array([-0.703125,  -184.7461,     99.93164,   100.2832,     -2.1972656 ,  9.2514715])*np.pi/180
    for gpos in smooth:
        inv_qpos, succ,_ = lerobot_IK_5DOF(last_qpos[0:5],gpos[0:6], robot=robot)
        if succ:
            inv_qpos = np.concatenate((inv_qpos,gpos[6:]*np.pi/180))
        else:
            inv_qpos = last_qpos
        last_qpos=inv_qpos
        
        qpos_list.append(inv_qpos)
        
    mujoco_follow_traj(qpos_list)
    # print(list)
    # # 使用增强版的RDP算法，自动添加高曲率区域的采样点
    # plot_rdp_results(
    #     g_actions, 
    #     parquet_file, 
    #     '.output',
    #     precision=0.03, 
    #     curvature_threshold=0.3, 
    #     points_to_add=2)
    
    # # 对比原始RDP和增强RDP的结果
    # plot_rdp_comparison(
    #     g_actions, 
    #     parquet_file, 
    #     '.output/comparison', 
    #     precision=0.03, 
    #     curvature_threshold=0.3, 
    #     points_to_add=2
    # )
    
    # 可以通过直接调用rdp_smoothing_enhanced来测试不同参数的效果
    # enhanced_actions, enhanced_masks = rdp_smoothing_enhanced(
    #     g_actions, 
    #     precision=0.03, 
    #     curvature_threshold=0.05,  # 调整曲率阈值
    #     points_to_add=3  # 在高曲率点附近添加的点数
    # )

if __name__ == "__main__":

    test_rdp()