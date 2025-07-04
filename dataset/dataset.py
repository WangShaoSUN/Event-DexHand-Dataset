import os
import pickle
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms import ToTensor
import numpy as np

class HandPoseDataset(Dataset):
    def __init__(self, rgb_dir, event_dir, meta_dir, transform=None):

        self.rgb_dir = rgb_dir
        self.event_dir = event_dir
        self.meta_dir = meta_dir
        self.transform = transform if transform else ToTensor()
        
        # 收集所有有效索引
        self.valid_indices = self._find_valid_indices()
    
    def _find_valid_indices(self):
        # 收集三个文件夹中文件都存在的有效索引
        valid_indices = []
        
        # 获取每个目录中的文件列表
        rgb_files = {f.split('_')[-1].split('.')[0] for f in os.listdir(self.rgb_dir) 
                     if f.startswith('rgb_image_') and f.endswith('.npy')}
        
        event_files = {f.split('_')[-1].split('.')[0] for f in os.listdir(self.event_dir)
                       if f.startswith('event_image_') and f.endswith('.npy')}
        
        meta_files = {f.split('_')[-1].split('.')[0] for f in os.listdir(self.meta_dir)
                      if f.startswith('meta_data_') and f.endswith('.pickle')}
        
        # 查找共有的索引
        common_indices = rgb_files & event_files & meta_files
        
        # 转换为整数并排序
        valid_indices = sorted([int(idx) for idx in common_indices if idx.isdigit()])
        
        return valid_indices

    def __len__(self):
        return len(self.valid_indices)

    def __getitem__(self, idx):
        """获取对应索引的三个数据模态"""
        try:
            actual_index = self.valid_indices[idx]
            
            # 构建文件路径
            rgb_path = os.path.join(self.rgb_dir, f'rgb_image_{actual_index}.npy')
            event_path = os.path.join(self.event_dir, f'event_image_{actual_index}.npy')
            meta_path = os.path.join(self.meta_dir, f'meta_data_{actual_index}.pickle')
            
            # 加载RGB图像
            # rgb_image = Image.open(rgb_path).convert('RGB')
            rgb_image= np.load(rgb_path)
            
            # 加载Event图像（假设是单通道）
            # event_image = Image.open(event_path)
            event_image =np.load(event_path)
            with open(meta_path, 'rb') as f:
                meta_data = pickle.load(f)
            
            # 应用转换
            if self.transform:
                rgb_image = self.transform(rgb_image)
                event_image = self.transform(event_image)
            
            return {
                'rgb': rgb_image,
                'event': event_image,
                'meta': meta_data,
                'index': actual_index
            }
            
        except Exception as e:
            # 跳过损坏/缺失的文件
            print(f"Skipping index {actual_index} due to error: {str(e)}")
            return self[idx + 1]  # 尝试返回下一个有效样本（使用时需确保不会无限递归）
            # 或者返回空字典并在collate_fn中处理
            # return {}


# 使用示例
if __name__ == "__main__":
    rgb_dir='../output3/meta_data'
    print("RGB 文件是否存在:", os.path.exists(rgb_dir))
    # 初始化数据集
    dataset = HandPoseDataset(
        rgb_dir='../output3/rgb_images',
        event_dir='../output3/event_images',
        meta_dir='../output3/meta_data',
        transform=ToTensor()
    )
    
    # 创建DataLoader
    def collate_fn(batch):
        """处理跳过样本的情况"""
        # 过滤掉空样本
        batch = [b for b in batch if b]
        
        # 如果batch为空则返回空
        if len(batch) == 0:
            return None
            
        # 重组batch
        rgb_batch = torch.stack([item['rgb'] for item in batch])
        event_batch = torch.stack([item['event'] for item in batch])
        meta_batch = [item['meta'] for item in batch]
        indices = [item['index'] for item in batch]
        
        return {
            'rgb': rgb_batch,
            'event': event_batch,
            'meta': meta_batch,
            'indices': indices
        }
    
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=3,
        shuffle=True,
        num_workers=4,
        collate_fn=collate_fn
    )
    
    # 训练循环示例
    for batch in dataloader:
        if batch is None: 
            continue
            
        rgb_data = batch['rgb']
        event_data = batch['event']
        mano_params = batch['meta']

    print("meta_data:keys", mano_params[0].keys())
        