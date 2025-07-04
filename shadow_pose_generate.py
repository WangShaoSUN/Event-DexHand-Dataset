import numpy as np
import open3d as o3d
import torch
import torch.nn.functional as F
import os
import glob
import pickle
import sys
import os
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)

from hand_embodiment.mano import HandState
from hand_embodiment.embodiment import HandEmbodiment
from hand_embodiment.target_configurations import TARGET_CONFIG, manobase2miabase
from hand_embodiment.metrics import (
    highlight_mesh_vertices, SHADOW_CONTACT_SURFACE_VERTICES)
import tqdm

# 设置参数
hand = "shadow"
fingers = ["thumb", "index", "middle", "ring", "little"]
highlighted_finger = "thumb"
HAND_CONFIG = TARGET_CONFIG[hand]

def process_hand_data(data):
    """处理单个数据样本并计算关节角度"""
    # 创建手部状态
    hand_state = HandState(left=False)
    hand_state.betas[:] = data['mano_shape']
    hand_state.pose[:] = np.concatenate([data['mano_rot_pose'], data['mano_hand_pose']])
    
    # 重新计算网格和高亮部分
    hand_state.recompute_mesh(manobase2miabase)
    highlight_mesh_vertices(
        hand_state.hand_mesh, SHADOW_CONTACT_SURFACE_VERTICES[highlighted_finger])
    
    # 创建化身对象并计算关节角度
    emb = HandEmbodiment(hand_state, HAND_CONFIG, use_fingers=fingers)
    hand_state.recompute_mesh(HAND_CONFIG["handbase2robotbase"])
    
    # 计算关节角度
    joint_angles, _ = emb.solve(
        return_desired_positions=True,
        use_cached_forward_kinematics=False)
    
    return joint_angles

def save_with_joint_angles(pickle_path, joint_angles):
    """将关节角度添加到pickle文件并保存"""
    with open(pickle_path, 'rb') as f:
        data = pickle.load(f)
    

    data['joint_angles'] = joint_angles
    

    with open(pickle_path, 'wb') as f:
        pickle.dump(data, f)

def process_folder(folder_path):
    """处理文件夹中的所有pickle文件"""
    # 获取所有符合命名规范的pickle文件
    file_pattern = os.path.join(folder_path, "meta_data*.pickle")
    pickle_files = sorted(glob.glob(file_pattern))
    
    print(f"找到 {len(pickle_files)} 个pickle文件需要处理")
    
    for idx, file_path in enumerate(tqdm.tqdm(pickle_files)):
        try:
            with open(file_path, 'rb') as f:
                data = pickle.load(f)
            
            # 计算关节角度
            joint_angles = process_hand_data(data)
            
            # 将关节角度添加回pickle文件
            save_with_joint_angles(file_path, joint_angles)
            
            # 打印进度信息
            file_name = os.path.basename(file_path)
            print(f"[{idx+1}/{len(pickle_files)}] 已处理 {file_name} - 关节角度已添加")
            
            # 打印关节角度摘要
            # print(f"  拇指: {len(joint_angles['thumb'])}个关节")
            # print(f"  食指: {len(joint_angles['index'])}个关节")
            # print(f"  中指: {len(joint_angles['middle'])}个关节")
            # print(f"  小指: {len(joint_angles['little'])}个关节")
            
        except Exception as e:
            print(f"处理 {file_path} 时出错: {str(e)}")
            continue

if __name__ == "__main__":
    # 设置pickle文件夹路径
    pickle_folder = "/data/output2/meta_data" 
    
    # 开始处理所有文件
    process_folder(pickle_folder)