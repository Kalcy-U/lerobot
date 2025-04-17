# policy_websocket_binary_client.py
import asyncio
import json
import logging
import pickle
import numpy as np
import torch
import websockets
import argparse
import cv2
import struct
import io
from typing import Dict, Any, Optional

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

'''
observations = robot.capture_observation()['state']
'''
def obs_to_bytes(observations,quality):
    # 图像压缩为 JPEG 并统计顺序
    image_data_list = []
    meta = {"images": []}

    for name in observations:
        if 'image' in name:
            img = observations[name]
            _, buf = cv2.imencode(".jpg", img, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
            image_data_list.append(buf.tobytes())
            meta["images"].append(name)  # 顺序很关键
        else:
            meta[name]=observations[name] 

    meta_bytes = pickle.dumps(meta)
    meta_len = struct.pack(">I", len(meta_bytes))

    # 构建整个 payload
    payload = meta_len + meta_bytes + b"".join(image_data_list)
    return payload

class PolicyWebSocketClient:
    """WebSocket策略客户端，用于发送二进制观测数据并接收预测动作"""
    
    def __init__(self, server_url: str, jpeg_quality: int = 85):
        """
        初始化WebSocket策略客户端
        
        Args:
            server_url: WebSocket服务器地址
            jpeg_quality: JPEG压缩质量(1-100)
        """
        self.server_url = server_url
        self.jpeg_quality = jpeg_quality
        self.websocket = None
        self.connected = False
    
    async def connect(self) -> bool:
        """连接到WebSocket服务器"""
        try:
            self.websocket = await websockets.connect(self.server_url)
            self.connected = True
            logger.info(f"已连接到服务器: {self.server_url}")
            return True
        except Exception as e:
            logger.error(f"连接服务器失败: {e}")
            self.connected = False
            return False
    
    async def disconnect(self):
        """断开与WebSocket服务器的连接"""
        if self.websocket and self.connected:
            await self.websocket.close()
            self.connected = False
            logger.info("已断开与服务器的连接")
    
    async def predict_action(self, observations: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        发送观测数据并获取预测动作
        
        Args:
            observations: 观测数据字典
            
        Returns:
            预测的动作或者None（如果发生错误）
        """
        if not self.connected or not self.websocket:
            logger.error("客户端未连接到服务器")
            return None
        
        try:
            # 将观测数据转换为二进制格式
            binary_data = obs_to_bytes(observations, self.jpeg_quality)
            
            # 发送二进制数据
            await self.websocket.send(binary_data)
            logger.debug(f"已发送观测数据，大小: {len(binary_data)} 字节")
            
            # 接收预测结果
            response = await self.websocket.recv()
            
            action = pickle.loads(response)
            
            return action
        except Exception as e:
            logger.error(f"预测动作时发生错误: {e}")
            return None

async def test_client():
    """测试二进制策略客户端的简单示例"""
    parser = argparse.ArgumentParser(description="二进制WebSocket策略客户端")
    parser.add_argument("--server", type=str, default="ws://localhost:8001", help="WebSocket服务器地址")
    parser.add_argument("--jpeg-quality", type=int, default=85, help="JPEG压缩质量(1-100)")
    args = parser.parse_args()
    
    client = PolicyWebSocketClient(args.server, args.jpeg_quality)
    
    # 创建一个测试观测数据
    observation = {
        "observation.state": torch.tensor([0.,90.,90.,90.,0.,60.]),
        "observation.images.on_hand": np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8),
       
    }
    
    # 连接到服务器
    if await client.connect():
        # 获取预测的动作
        action = await client.predict_action(observation)
        if action is not None:
            logger.info(f"收到预测动作: {action}")
    
    # 断开连接
    await client.disconnect()

if __name__ == "__main__":
    asyncio.run(test_client())