import os
import json
import shutil
from pathlib import Path
import numpy as np
import torchcodec
import pandas as pd

# Define paths
root_dir='/data/tyn/'
dataset_dirs = ["so100_test1","so100_test2", "so100_test3","so100_test4",  "so100_test5"]  # Replace with actual paths
dataset_dirs=[root_dir+dataset for dataset in dataset_dirs]
output_dir = Path(root_dir+"so100_grasp")
output_dir.mkdir(exist_ok=True)

# Create subdirectories in the output directory
for subdir in ["data", "meta", "videos"]:
    (output_dir / subdir).mkdir(exist_ok=True)

# Initialize aggregated metadata
aggregated_episodes = []
aggregated_stats = []
total_episodes = 0
total_frames = 0
chunk_size = 1000  # From your info.json: "chunks_size": 1000

# Get video frame count function using torchcodec
def get_video_frame_count(video_path):
    try:
        # 使用torchcodec获取视频帧数
        decoder = torchcodec.decoders.VideoDecoder(str(video_path))
        num_frames = decoder.metadata.num_frames
        
        return num_frames
    except Exception as e:
        print(f"获取视频帧数失败: {video_path}, 错误: {e}")
        return 0

# Process each dataset
current_episode_index = 0

for dataset_dir in dataset_dirs:
    dataset_path = Path(dataset_dir)
    
    # 加载当前数据集的info.json
    with open(dataset_path / "meta" / "info.json", "r") as f:
        dataset_info = json.load(f)
    
    # 获取当前数据集的episodes数量
    dataset_episodes_count = dataset_info["total_episodes"]
    
    # 1. 加载并更新episodes.jsonl
    dataset_episodes = []
    with open(dataset_path / "meta" / "episodes.jsonl", "r") as f:
        for line in f:
            episode = json.loads(line.strip())
            old_index = episode["episode_index"]
            episode["episode_index"] = current_episode_index + old_index
            
            # 验证视频帧数与episode长度是否匹配
            old_chunk_idx = old_index // chunk_size
            video_path = dataset_path / "videos" / f"chunk-{old_chunk_idx:03d}" / "observation.images.on_hand" / f"episode_{old_index:06d}.mp4"
            dataset_episodes.append(episode)
    aggregated_episodes.extend(dataset_episodes)
    
    # 2. 加载并更新episodes_stats.jsonl
    dataset_stats = []
    with open(dataset_path / "meta" / "episodes_stats.jsonl", "r") as f:
        for line in f:
            stat = json.loads(line.strip())
            old_index = stat["episode_index"]
            new_index = current_episode_index + old_index
            
            # 更新索引
            stat["episode_index"] = new_index
            
            # 更新统计信息中的episode_index
            stat["stats"]["episode_index"]["min"] = [new_index]
            stat["stats"]["episode_index"]["max"] = [new_index]
            stat["stats"]["episode_index"]["mean"] = [new_index]
            
            # 更新帧数统计信息
            if old_index < len(dataset_episodes) and "length" in dataset_episodes[old_index]:
                actual_length = dataset_episodes[old_index]["length"]
                # 更新count字段
                for key in stat["stats"]:
                    if "count" in stat["stats"][key]:
                        stat["stats"][key]["count"] = [actual_length]
            
            dataset_stats.append(stat)
    aggregated_stats.extend(dataset_stats)
    
    # 3. 复制数据和视频文件
    for old_index in range(dataset_episodes_count):
        new_index = current_episode_index + old_index
        
        # 确定源和目标的chunk索引
        old_chunk_idx = old_index // chunk_size
        new_chunk_idx = new_index // chunk_size
        
        # 创建目标chunk目录
        chunk_dir = output_dir / "data" / f"chunk-{new_chunk_idx:03d}"
        chunk_dir.mkdir(exist_ok=True, parents=True)
        
        # 复制并修改parquet文件
        old_parquet_path = dataset_path / "data" / f"chunk-{old_chunk_idx:03d}" / f"episode_{old_index:06d}.parquet"
        new_parquet_path = chunk_dir / f"episode_{new_index:06d}.parquet"
        
        if old_parquet_path.exists():
            # 读取parquet文件
            try:
                df = pd.read_parquet(old_parquet_path)
                
                # 更新episode_index列
                df['episode_index'] = new_index
                df['index']= df['index']+ total_frames
                # 保存修改后的parquet文件
                df.to_parquet(new_parquet_path)
            except Exception as e:
                print(f"处理parquet文件失败: {old_parquet_path}, 错误: {e}")
                # 如果处理失败，直接复制原文件
                shutil.copy(old_parquet_path, new_parquet_path)
        else:
            print(f"文件不存在: {old_parquet_path}")
        
        # 复制视频文件
        for camera in ["observation.images.on_hand", "observation.images.camera_pro"]:
            old_video_dir = dataset_path / "videos" / f"chunk-{old_chunk_idx:03d}" / camera
            new_video_dir = output_dir / "videos" / f"chunk-{new_chunk_idx:03d}" / camera
            new_video_dir.mkdir(exist_ok=True, parents=True)
            
            old_video_path = old_video_dir / f"episode_{old_index:06d}.mp4"
            new_video_path = new_video_dir / f"episode_{new_index:06d}.mp4"
            
            if old_video_path.exists():
                shutil.copy(old_video_path, new_video_path)
            else:
                print(f"视频文件不存在: {old_video_path}")
    
    # 计算当前数据集的总帧数
    dataset_frames = sum(episode["length"] for episode in dataset_episodes)
    total_frames += dataset_frames
    print(f"数据集 {dataset_dir} 包含 {dataset_episodes_count} 个episodes和 {dataset_frames} 帧")
    
    # 更新当前episode索引
    current_episode_index += dataset_episodes_count
    

# 更新总episodes数量
total_episodes = current_episode_index

# 4. 写入合并后的episodes.jsonl
with open(output_dir / "meta" / "episodes.jsonl", "w") as f:
    for episode in sorted(aggregated_episodes, key=lambda x: x["episode_index"]):
        f.write(json.dumps(episode) + "\n")

# 5. 写入合并后的episodes_stats.jsonl
with open(output_dir / "meta" / "episodes_stats.jsonl", "w") as f:
    for stat in sorted(aggregated_stats, key=lambda x: x["episode_index"]):
        f.write(json.dumps(stat) + "\n")

# 6. 更新并写入info.json
info_template = dataset_info  # 使用最后加载的info作为模板
info_template["total_episodes"] = total_episodes
info_template["total_frames"] = total_frames
info_template["total_videos"] = total_episodes * 2  # 每个episode有两个摄像头
info_template["total_chunks"] = (total_episodes + chunk_size - 1) // chunk_size
info_template["splits"]["train"] = f"0:{total_episodes}"

with open(output_dir / "meta" / "info.json", "w") as f:
    json.dump(info_template, f, indent=4)

# 7. 复制tasks.jsonl（假设所有数据集中的任务相同）
shutil.copy(dataset_path / "meta" / "tasks.jsonl", output_dir / "meta" / "tasks.jsonl")

print(f"合并数据集已创建于 {output_dir}，包含 {total_episodes} 个episodes和 {total_frames} 帧。")