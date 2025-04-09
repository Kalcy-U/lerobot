import copy
import os
import numpy as np
import math
from lerobot_kinematics import lerobot_IK_5DOF, lerobot_FK_5DOF, get_robot,so100_FK,so100_IK
import pandas as pd
from pathlib import Path
import json

step=10
def load_dataset_qpos(path):
    df = pd.read_parquet(path)
    return df, df['observation.state'].to_list(),df['action'].to_list()


def deg_to_rad(x):
    return x *math.pi/180

def q_to_gpos(qpos,robot):
    c_qpos=qpos.copy()
    c_qpos[0]=-c_qpos[0]
    c_qpos[1]=-c_qpos[1]
    gpos=lerobot_FK_5DOF(c_qpos[0:5],robot)
    gpos[-1]=c_qpos[5] #end effector
    return gpos # 7Dof

np.set_printoptions(linewidth=200)
# Robot Initialization
robot = get_robot('so100')

base_path = '/data/tyn/so100_grasp/'
output_path = f'/data/tyn/so100_grasp_cartesian_{step}/'

# 创建输出目录
os.makedirs(output_path, exist_ok=True)
os.makedirs(os.path.join(output_path, 'data'), exist_ok=True)
os.makedirs(os.path.join(output_path, 'meta'), exist_ok=True)

# 复制视频文件夹
os.system(f'cp -r {base_path}/videos {output_path}/')

# 加载episodes.jsonl
with open(os.path.join(base_path, 'meta', 'episodes.jsonl'), 'r') as f:
    episodes = [json.loads(line) for line in f]

# 加载info.json
with open(os.path.join(base_path, 'meta', 'info.json'), 'r') as f:
    info = json.load(f)

# 复制tasks.jsonl
os.system(f'cp {base_path}/meta/tasks.jsonl {output_path}/meta/')


# 更新episodes_stats.jsonl中的统计信息
with open(os.path.join(base_path, 'meta', 'episodes_stats.jsonl'), 'r') as f:
    stats = [json.loads(line) for line in f]
    
diff_count=0
error_file_list=[]
# 处理每个episode
for episode in episodes:
    episode_idx = episode['episode_index']
    chunk_idx = episode_idx // 1000
    
    # 创建chunk目录
    chunk_dir = os.path.join(output_path, 'data', f'chunk-{chunk_idx:03d}')
    os.makedirs(chunk_dir, exist_ok=True)
    
    # 加载parquet文件
    parquet_path = os.path.join(base_path, 'data', f'chunk-{chunk_idx:03d}', f'episode_{episode_idx:06d}.parquet')
    new_parquet_path = os.path.join(chunk_dir, f'episode_{episode_idx:06d}.parquet')
    
    if os.path.exists(parquet_path):
        df, o_qpos_list,a_qpos_list = load_dataset_qpos(parquet_path)
        # 转换为笛卡尔坐标
        o_gpos_list = []
        a_gpos_list=[]
        for o_qpos,a_qpos in zip(o_qpos_list,a_qpos_list):
            gpos=so100_FK(o_qpos)
            # test IK  
            inverse_qpos,succ = so100_IK(o_qpos, gpos)
            if succ:
                # 对比inverse_qpos和o_qpos的差异(in degree)
                diff = np.linalg.norm(np.array(inverse_qpos) - np.array(o_qpos))
                if diff>25:
                    diff_count+=1
                    if len(error_file_list)==0 or error_file_list[-1]!=parquet_path:
                        error_file_list.append(parquet_path)
            else:
                print("当前位置:", gpos)
                print("原始qpos:", [f"{x:.4f}" for x in o_qpos[0:5]])
                if len(error_file_list)==0 or  error_file_list[-1]!=parquet_path:
                    error_file_list.append(parquet_path)
            o_gpos_list.append(gpos)
            a_gpos_list.append(so100_FK(a_qpos))
            
        df['observation.state'] = df.index.map(
            lambda i: o_gpos_list[i + step] if i + step < len(o_gpos_list) else o_gpos_list[-1]
        )
        df['action'] = df.index.map(
            lambda i: a_gpos_list[i + step] if i + step < len(a_gpos_list) else a_gpos_list[-1]
        )
        df.to_parquet(new_parquet_path)
        print(df['action'][0:10])
        print(a_gpos_list[0:10])
        # 更新统计信息
        stats[episode_idx]['stats']['action']['min'] = np.min(a_gpos_list, axis=0).tolist()
        stats[episode_idx]['stats']['action']['max'] = np.max(a_gpos_list, axis=0).tolist()
        stats[episode_idx]['stats']['action']['mean'] = np.mean(a_gpos_list, axis=0).tolist()
        stats[episode_idx]['stats']['action']['std'] = np.std(a_gpos_list, axis=0).tolist()
        
        stats[episode_idx]['stats']['observation.state']['min'] = np.min(o_gpos_list, axis=0).tolist()
        stats[episode_idx]['stats']['observation.state']['max'] = np.max(o_gpos_list, axis=0).tolist()
        stats[episode_idx]['stats']['observation.state']['mean'] = np.mean(o_gpos_list, axis=0).tolist()
        stats[episode_idx]['stats']['observation.state']['std'] = np.std(o_gpos_list, axis=0).tolist()

        # print(f"处理完成: episode_{episode_idx:06d}")
    else:
        print(f"文件不存在: {parquet_path}")

print(f"error_file_list:{error_file_list}")
# 保存修改后的episodes_stats.jsonl
with open(os.path.join(output_path, 'meta', 'episodes_stats.jsonl'), 'w') as f:
    for stat in stats:
        f.write(json.dumps(stat) + '\n')

# 复制episodes.jsonl
os.system(f'cp {base_path}/meta/episodes.jsonl {output_path}/meta/')

# 更新info.json中的字段名称
info['features']['observation.state']['names'] = [
    'x',
    'y',
    'z',
    'roll',
    'pitch',
    'yaw',
    'gripper'
]
info['features']['observation.state']['shape'] = [7]

info['features']['action']['names'] = [
    'x',
    'y',
    'z',
    'roll',
    'pitch',
    'yaw',
    'gripper'
]
info['features']['action']['shape'] = [7]


# 保存修改后的info.json
with open(os.path.join(output_path, 'meta', 'info.json'), 'w') as f:
    json.dump(info, f, indent=4)

print("数据集转换完成！")

