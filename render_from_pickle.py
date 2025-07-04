import numpy as np
import open3d as o3d
import torch
import torch.nn.functional as F

import sys
import os
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)

import pytransform3d.visualizer as pv
from hand_embodiment.mano import HandState
from hand_embodiment.embodiment import HandEmbodiment
from hand_embodiment.target_configurations import TARGET_CONFIG, manobase2miabase
from hand_embodiment.metrics import (
    highlight_mesh_vertices, MANO_CONTACT_SURFACE_VERTICES, SHADOW_CONTACT_SURFACE_VERTICES,
    highlight_graph_visuals, CONTACT_SURFACE_VERTICES,
    extract_mano_contact_surface, extract_graph_vertices,
    distances_robot_to_mano)
import pickle

with open('examples\meta_data_1.pickle', 'rb') as f:  # 必须用二进制模式 'rb'
    data = pickle.load(f)


# 提取数据
mano_rot_pose = data['mano_rot_pose']     # 手部的全局旋转（轴角表示）
mano_hand_pose = data['mano_hand_pose']   # 手部关节姿势（45维）
mano_shape = data['mano_shape']           # MANO形状参数（10维）
mano_trans = data['mano_trans']           # 手部全局平移（3维）
mano_root_joint = data['mano_root_joint']  # 根关节位置（3D）
joint_angles = data['joint_angles']
mano_pose = np.concatenate([mano_rot_pose, mano_hand_pose])

mean=np.array([0,0,0,0.1117, -0.0429, 0.4164, 0.1088, 0.0660, 0.7562, -0.0964, 0.0909,
                                        0.1885, -0.1181, -0.0509, 0.5296, -0.1437, -0.0552, 0.7049, -0.0192,
                                        0.0923, 0.3379, -0.4570, 0.1963, 0.6255, -0.2147, 0.0660, 0.5069,
                                        -0.3697, 0.0603, 0.0795, -0.1419, 0.0859, 0.6355, -0.3033, 0.0579,
                                        0.6314, -0.1761, 0.1321, 0.3734, 0.8510, -0.2769, 0.0915, -0.4998,
                                        -0.0266, -0.0529, 0.5356, -0.0460, 0.2774])

# hand = "mia"
# hand = "leap"
# hand = "schunk"
hand = "shadow"
# hand = "inspire"
HAND_CONFIG = TARGET_CONFIG[hand]
ROBOT_CONTACT_SURFACE_VERTICES = CONTACT_SURFACE_VERTICES[hand]
fingers = ["thumb", "index", "middle", "ring", "little"]
fingers = ["thumb", "index", "middle", "little", ]
highlighted_finger = "thumb"


hand_state = HandState(left=False)
hand_state.betas[:] = data['mano_shape']  # 形状参数 β
hand_state.pose[:] = np.concatenate([data['mano_rot_pose'], data['mano_hand_pose']])  # 全局旋转+局部关节
# hand_state.trans = data['mano_trans'] 
hand_state.recompute_mesh(manobase2miabase)
# highlight_mesh_vertices(
#     hand_state.hand_mesh, MANO_CONTACT_SURFACE_VERTICES[highlighted_finger])


highlight_mesh_vertices(
    hand_state.hand_mesh, SHADOW_CONTACT_SURFACE_VERTICES[highlighted_finger])



emb = HandEmbodiment(hand_state, HAND_CONFIG, use_fingers=fingers)
emb.joint_angles = joint_angles

# 为每个手指执行正运动学
for finger_name in fingers:
    # 更新目标链的正运动学
    emb.target_finger_chains[finger_name].forward(joint_angles[finger_name])
    
    # 如果可视化运动学对象和目标运动学对象不同，则也需要更新
    if emb.vis_kin is not emb.target_kin:
        emb.vis_finger_chains[finger_name].forward(joint_angles[finger_name])
print(HAND_CONFIG["handbase2robotbase"])
hand_state.recompute_mesh(HAND_CONFIG["handbase2robotbase"])

# joint_angles, desired_positions = emb.solve(
#     return_desired_positions=True,
#     use_cached_forward_kinematics=False)
# print("joint_angles:",joint_angles)





# join=np.concatenate((joint_angles["thumb"],joint_angles["index"],joint_angles["middle"],joint_angles["ring"],joint_angles["little"]))
# join=np.concatenate((joint_angles["thumb"],joint_angles["index"],joint_angles["middle"],joint_angles["little"]))
# bb=torch.tensor(join)
# print(bb)
graph = pv.Graph(
    emb.transform_manager_, HAND_CONFIG["base_frame"], show_frames=False,
    show_connections=False, show_visuals=True, show_collision_objects=False,
    show_name=False, s=0.02)
highlight_graph_visuals(graph, ROBOT_CONTACT_SURFACE_VERTICES[highlighted_finger])

# dists = distances_robot_to_mano(
#     hand_state, graph, ROBOT_CONTACT_SURFACE_VERTICES, fingers)
# print("error:",dists)

# mano_vertices, mano_triangles = extract_mano_contact_surface(
#     hand_state, highlighted_finger)
robot_vertices = extract_graph_vertices(
    graph, ROBOT_CONTACT_SURFACE_VERTICES, highlighted_finger)


robot_vertices = extract_graph_vertices(
    graph, ROBOT_CONTACT_SURFACE_VERTICES, highlighted_finger)

fig = pv.figure()
import pytransform3d.transformations as pt
manobase2shunk = pt.transform_from_exponential_coordinates(
   [-2.228, -0.163, 1.907, 0.066, -0.343, -0.087]  )

manobase2shadowbase = pt.transform_from_exponential_coordinates(
   [-0.340, 2.110, 2.297, -0.385, -0.119, -0.094]  )
manobase2shunk = pt.transform_from_exponential_coordinates(
[2.3, 2.184, 0.196, -0.094, -0.270, 0.251])



hand_state.recompute_mesh(manobase2shunk)
fig.add_geometry(hand_state.hand_mesh)
hand_state.hand_mesh.paint_uniform_color((1, 0.5, 0)) 
graph.add_artist(fig)

# mesh = o3d.geometry.TriangleMesh(o3d.utility.Vector3dVector(mano_vertices),
#                                  o3d.utility.Vector3iVector(mano_triangles))
# mesh.paint_uniform_color((1, 0.5, 0))
# mesh.compute_vertex_normals()
# fig.add_geometry(mesh)

pc = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(robot_vertices))
fig.add_geometry(pc)

fig.show()
