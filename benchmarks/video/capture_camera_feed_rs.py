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
"""Capture video feed from a camera as raw images."""

import argparse
import datetime as dt
from pathlib import Path

import cv2
import pyrealsense2 as rs
import numpy as np


def display_and_save_video_stream(output_dir: Path, fps: int, width: int, height: int):
    now = dt.datetime.now()
    capture_dir = output_dir / f"{now:%Y-%m-%d}" / f"{now:%H-%M-%S}"
    if not capture_dir.exists():
        capture_dir.mkdir(parents=True, exist_ok=True)

    # 初始化RealSense相机
    context = rs.context()
    devices = context.query_devices()
    if len(devices) == 0:
        print("未检测到RealSense设备")
        exit()
    else:
        print(f"检测到 {len(devices)} 个RealSense设备")
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.color, width, height, rs.format.bgr8, fps)
    config.enable_stream(rs.stream.depth, width, height, rs.format.z16, fps)
    try:
        pipeline.start(config)
        
        frame_index = 0
        while True:
            # 等待一组帧（彩色）
            frames = pipeline.wait_for_frames()
            color_frame = frames.get_color_frame()
            # 获取深度图
            
            depth_frame = frames.get_depth_frame()
            
            if depth_frame:
                # 转换深度图为可视化格式
                depth_image = np.asanyarray(depth_frame.get_data())
     
                depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET)
                
                # 显示深度图
                cv2.imshow("Depth Stream", depth_colormap)
                
            
            if color_frame:
                
                # 转换为numpy数组
                frame = np.asanyarray(color_frame.get_data())
                
                cv2.imshow("Video Stream", frame)
            
            key = cv2.waitKey(1)
            if key==ord(' '):
                frame_index+=1
                cv2.imwrite(str(capture_dir / f"depth_{frame_index:06d}.png"), depth_image)
                cv2.imwrite(str(capture_dir / f"frame_{frame_index:06d}.png"), frame)
                frame_index+=1
          
            
            if key == ord("q"):
                break

    finally:
        # 停止相机流
        pipeline.stop()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/cam_capture/"),
        help="Directory where the capture images are written. A subfolder named with the current date & time will be created inside it for each capture.",
    )
    parser.add_argument(
        "--fps",
        type=int,
        default=30,
        help="Frames Per Second of the capture.",
    )
    parser.add_argument(
        "--width",
        type=int,
        default=1280,
        help="Width of the captured images.",
    )
    parser.add_argument(
        "--height",
        type=int,
        default=720,
        help="Height of the captured images.",
    )
    args = parser.parse_args()
    display_and_save_video_stream(**vars(args))
