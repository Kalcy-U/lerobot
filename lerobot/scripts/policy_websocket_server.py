#!/usr/bin/env python
# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
基于WebSocket的策略服务器，支持纯二进制通信。
启动一个WebSocket服务器，接收客户端发送的观测数据（包括二进制图像），
使用加载的策略模型生成动作预测，并将预测结果以二进制格式返回给客户端。

示例用法:

python lerobot/scripts/policy_websocket_server.py \
    --model_path=/path/to/policy/model \
    --host=0.0.0.0 \
    --port=8001 \
    --use_amp=True
"""

import pickle
import websockets
from websockets.server import serve
import asyncio
import torch
import numpy as np
from contextlib import nullcontext
import logging
import struct
import cv2
import io
import argparse
import json
from lerobot.common.policies.pi0.modeling_pi0 import PI0Policy
from lerobot.common.robot_devices.control_utils import predict_action
# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

'''
请求格式
| 4 bytes | N bytes       | M bytes (opt)     |
|--------:|----------------|------------------|
| PICKLE长度 | PICKLE字节数据   | 图像二进制 如JPEG |

'''

def bytes_to_obs(data):
        # 解析  元信息
    meta_len = struct.unpack(">I", data[:4])[0]
    meta = pickle.loads(data[4:4+meta_len])
    observation = {}
    for key in meta:
        if "images" not in key:
            observation[key]=meta[key]
        else:
            image_names = meta["images"]
            raw_images = data[4+meta_len:]
            # 切割图像部分
            offset = 0
            for name in image_names:
                end = raw_images.find(b'\xff\xd9', offset) + 2 #jpg分割字节
                img_bytes = raw_images[offset:end]
                offset = end

                img = cv2.imdecode(np.frombuffer(img_bytes, np.uint8), cv2.IMREAD_COLOR)
                tensor = torch.from_numpy(img)
                observation[name] = tensor

    return observation

class PolicyWebSocketServer:
    """策略WebSocket服务器，处理客户端请求并返回预测动作"""
    
    def __init__(self, policy, device, use_amp=True):
        """初始化服务器
        
        Args:
            policy: 预训练的策略模型
            device: 计算设备（CPU或GPU）
            use_amp: 是否使用自动混合精度
        """
        self.policy = policy
        self.device = device
        self.policy.to(device)
        self.policy.eval()
        self.use_amp = use_amp
        self.clients = set()
        logger.info(f"策略服务器初始化完成，设备: {device}, 使用AMP: {use_amp}")
    
    async def handle_client(self, websocket):
        """处理客户端连接
        
        Args:
            websocket: WebSocket连接
        """
        client_id = id(websocket)
        self.clients.add(websocket)
        client_info = f"{websocket.remote_address[0]}:{websocket.remote_address[1]}"
        logger.info(f"客户端 {client_info} (ID: {client_id}) 已连接")
        
        try:
            async for message in websocket:
                # 处理二进制消息
                if isinstance(message, bytes):
                    logger.debug(f"收到客户端 {client_id} 的二进制数据: {len(message)} 字节")
                    # 将二进制数据转换为observation
                    observation = bytes_to_obs(message)
                    
                    # 使用策略模型预测动作
                    
                    actions = predict_action( observation,self.policy, device=self.device,use_amp=self.use_amp)
                    
                    pickle_action = pickle.dumps(actions)
                    
                    await websocket.send(pickle_action)
                    logger.debug(f"已向客户端 {client_id} 发送预测动作")
                else:
                    logger.warning(f"收到非二进制消息: {message}")
        except websockets.exceptions.ConnectionClosed as e:
            logger.info(f"客户端 {client_info} (ID: {client_id}) 连接已关闭: {e}")
        except Exception as e:
            logger.error(f"处理客户端 {client_id} 请求时发生错误: {e}", exc_info=True)
        finally:
            self.clients.remove(websocket)
            logger.info(f"客户端 {client_info} (ID: {client_id}) 已断开连接")
    
    async def start(self, host, port):
        """启动WebSocket服务器
        
        Args:
            host: 服务器主机地址
            port: 服务器端口
        """
        async with serve(self.handle_client, host, port):
            logger.info(f"策略WebSocket服务器已启动，监听地址: {host}:{port}")
            await asyncio.Future()  # 无限运行，直到被中断


async def main():
    """主函数"""
    # 解析命令行参数
    parser = argparse.ArgumentParser(description="纯二进制WebSocket策略服务器")
    parser.add_argument("--model_path", type=str, default="/home/fdse/tyn/lerobot/outputs/train/pi0/checkpoints/last/pretrained_model", 
                      help="预训练策略模型的路径")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="服务器监听地址")
    parser.add_argument("--port", type=int, default=8001, help="服务器监听端口")
    parser.add_argument("--use_amp", type=bool, default=True, help="是否使用自动混合精度")
    args = parser.parse_args()
    
    # 加载预训练策略模型
    logger.info(f"正在加载模型: {args.model_path}")
    policy = PI0Policy.from_pretrained(args.model_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"模型加载完成，使用设备: {device}")
    
    # 实现服务器功能
    server = PolicyWebSocketServer(policy, device, args.use_amp)
    await server.start(args.host, args.port)


if __name__ == "__main__":
    asyncio.run(main())