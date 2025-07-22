import pandas as pd
import numpy as np
import os
import zipfile
from typing import Tuple, Dict, List
from datetime import datetime
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split

from .utils import (
    setup_logger, 
    create_user_item_mapping, 
    create_interaction_matrix,
    build_graph_edges
)


class AliRecommendDataset:
    """阿里推荐数据集处理类"""
    
    def __init__(self, data_dir: str = 'dataset'):
        self.data_dir = data_dir
        self.logger = setup_logger('AliDataset')
        
        # 数据相关属性
        self.user_data = None
        self.item_data = None
        self.user_mapping = None
        self.item_mapping = None
        self.n_users = 0
        self.n_items = 0
        self.interaction_matrix = None
        self.edge_index = None
        
    def load_data(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        加载用户行为数据和商品数据
        
        Returns:
            user_data: 用户行为数据
            item_data: 商品数据
        """
        self.logger.info("开始加载数据...")
        
        # 加载用户行为数据
        user_file_path = os.path.join(self.data_dir, 'tianchi_mobile_recommend_train_user.zip')
        if os.path.exists(user_file_path):
            self.logger.info("从压缩包加载用户行为数据...")
            with zipfile.ZipFile(user_file_path, 'r') as zip_ref:
                # 假设压缩包内的文件名
                csv_files = [f for f in zip_ref.namelist() if f.endswith('.csv')]
                if csv_files:
                    with zip_ref.open(csv_files[0]) as f:
                        self.user_data = pd.read_csv(f)
                else:
                    raise FileNotFoundError("压缩包中未找到CSV文件")
        else:
            # 尝试直接加载CSV文件
            csv_path = os.path.join(self.data_dir, 'tianchi_mobile_recommend_train_user.csv')
            if os.path.exists(csv_path):
                self.user_data = pd.read_csv(csv_path)
            else:
                self.logger.error(f"未找到用户数据文件: {user_file_path} 或 {csv_path}")
                # 创建示例数据用于测试
                self.user_data = self._create_sample_user_data()
        
        # 加载商品数据
        item_file_path = os.path.join(self.data_dir, 'tianchi_mobile_recommend_train_item.csv')
        if os.path.exists(item_file_path):
            self.item_data = pd.read_csv(item_file_path)
        else:
            self.logger.warning(f"未找到商品数据文件: {item_file_path}")
            # 创建示例数据
            self.item_data = self._create_sample_item_data()
        
        self.logger.info(f"用户行为数据形状: {self.user_data.shape}")
        self.logger.info(f"商品数据形状: {self.item_data.shape}")
        
        return self.user_data, self.item_data
    
    def _create_sample_user_data(self) -> pd.DataFrame:
        """创建示例用户行为数据"""
        self.logger.info("创建示例用户行为数据...")
        
        np.random.seed(42)
        n_samples = 10000
        n_users = 1000
        n_items = 500
        
        data = {
            'user_id': np.random.randint(1, n_users + 1, n_samples),
            'item_id': np.random.randint(1, n_items + 1, n_samples),
            'behavior_type': np.random.choice([1, 2, 3, 4], n_samples, p=[0.6, 0.15, 0.15, 0.1]),
            'user_geohash': [''] * n_samples,  # 简化处理
            'item_category': np.random.randint(1, 50, n_samples),
            'time': pd.date_range('2014-11-18', '2014-12-18', periods=n_samples)
        }
        
        return pd.DataFrame(data)
    
    def _create_sample_item_data(self) -> pd.DataFrame:
        """创建示例商品数据"""
        self.logger.info("创建示例商品数据...")
        
        if self.user_data is not None:
            unique_items = self.user_data['item_id'].unique()
        else:
            unique_items = range(1, 501)
        
        data = {
            'item_id': unique_items,
            'item_geohash': [''] * len(unique_items),
            'item_category': np.random.randint(1, 50, len(unique_items))
        }
        
        return pd.DataFrame(data)
    
    def preprocess_data(self, behavior_types: List[int] = [4]) -> Tuple[np.ndarray, torch.Tensor]:
        """
        预处理数据，构建交互矩阵和图结构
        
        Args:
            behavior_types: 要考虑的行为类型，默认只考虑购买行为(4)
            
        Returns:
            interaction_matrix: 用户-物品交互矩阵
            edge_index: 图的边索引
        """
        if self.user_data is None:
            self.load_data()
        
        self.logger.info("开始预处理数据...")
        
        # 过滤行为类型
        filtered_data = self.user_data[self.user_data['behavior_type'].isin(behavior_types)]
        self.logger.info(f"过滤后的交互数量: {len(filtered_data)}")
        
        # 创建用户和物品映射
        self.user_mapping, self.item_mapping, self.n_users, self.n_items = create_user_item_mapping(filtered_data)
        
        self.logger.info(f"用户数量: {self.n_users}, 物品数量: {self.n_items}")
        
        # 创建交互矩阵
        self.interaction_matrix = create_interaction_matrix(
            filtered_data, self.user_mapping, self.item_mapping, self.n_users, self.n_items
        )
        
        # 构建图边索引
        self.edge_index = build_graph_edges(self.interaction_matrix)
        
        self.logger.info(f"交互矩阵形状: {self.interaction_matrix.shape}")
        self.logger.info(f"边数量: {self.edge_index.shape[1]}")
        
        return self.interaction_matrix, self.edge_index
    
    def split_data(self, test_ratio: float = 0.2, random_state: int = 42) -> Tuple[np.ndarray, np.ndarray]:
        """
        分割训练和测试数据
        
        Args:
            test_ratio: 测试集比例
            random_state: 随机种子
            
        Returns:
            train_matrix: 训练交互矩阵
            test_matrix: 测试交互矩阵
        """
        if self.interaction_matrix is None:
            raise ValueError("请先调用preprocess_data()方法")
        
        self.logger.info(f"分割数据，测试集比例: {test_ratio}")
        
        train_matrix = np.zeros_like(self.interaction_matrix)
        test_matrix = np.zeros_like(self.interaction_matrix)
        
        np.random.seed(random_state)
        
        for user_idx in range(self.n_users):
            user_items = np.where(self.interaction_matrix[user_idx] > 0)[0]
            
            if len(user_items) > 1:
                n_test = max(1, int(len(user_items) * test_ratio))
                test_items = np.random.choice(user_items, n_test, replace=False)
                train_items = np.setdiff1d(user_items, test_items)
                
                train_matrix[user_idx, train_items] = 1
                test_matrix[user_idx, test_items] = 1
            else:
                # 如果用户只有一个交互，放入训练集
                train_matrix[user_idx, user_items] = 1
        
        self.logger.info(f"训练集交互数: {train_matrix.sum()}")
        self.logger.info(f"测试集交互数: {test_matrix.sum()}")
        
        return train_matrix, test_matrix
    
    def get_negative_samples(self, interaction_matrix: np.ndarray, 
                           neg_ratio: int = 4) -> List[Tuple[int, int]]:
        """
        生成负样本
        
        Args:
            interaction_matrix: 交互矩阵
            neg_ratio: 负样本比例
            
        Returns:
            负样本列表 [(user_idx, item_idx), ...]
        """
        positive_samples = []
        negative_samples = []
        
        # 收集正样本
        for user_idx in range(self.n_users):
            for item_idx in range(self.n_items):
                if interaction_matrix[user_idx, item_idx] > 0:
                    positive_samples.append((user_idx, item_idx))
        
        # 为每个正样本生成负样本
        for user_idx, _ in positive_samples:
            neg_count = 0
            while neg_count < neg_ratio:
                neg_item = np.random.randint(0, self.n_items)
                if interaction_matrix[user_idx, neg_item] == 0:
                    negative_samples.append((user_idx, neg_item))
                    neg_count += 1
        
        return negative_samples
    
    def save_processed_data(self, save_dir: str = 'dataset/processed'):
        """保存预处理后的数据"""
        os.makedirs(save_dir, exist_ok=True)
        
        if self.interaction_matrix is not None:
            np.save(os.path.join(save_dir, 'interaction_matrix.npy'), self.interaction_matrix)
        
        if self.edge_index is not None:
            torch.save(self.edge_index, os.path.join(save_dir, 'edge_index.pt'))
        
        # 保存映射信息
        import pickle
        mappings = {
            'user_mapping': self.user_mapping,
            'item_mapping': self.item_mapping,
            'n_users': self.n_users,
            'n_items': self.n_items
        }
        
        with open(os.path.join(save_dir, 'mappings.pkl'), 'wb') as f:
            pickle.dump(mappings, f)
        
        self.logger.info(f"数据已保存到: {save_dir}")
    
    def load_processed_data(self, load_dir: str = 'dataset/processed'):
        """加载预处理后的数据"""
        try:
            self.interaction_matrix = np.load(os.path.join(load_dir, 'interaction_matrix.npy'))
            self.edge_index = torch.load(os.path.join(load_dir, 'edge_index.pt'))
            
            import pickle
            with open(os.path.join(load_dir, 'mappings.pkl'), 'rb') as f:
                mappings = pickle.load(f)
                
            self.user_mapping = mappings['user_mapping']
            self.item_mapping = mappings['item_mapping']
            self.n_users = mappings['n_users']
            self.n_items = mappings['n_items']
            
            self.logger.info(f"数据已从 {load_dir} 加载")
            return True
        except Exception as e:
            self.logger.warning(f"加载预处理数据失败: {e}")
            return False


class GraphDataset(Dataset):
    """图数据集类，用于PyTorch训练"""
    
    def __init__(self, edge_index: torch.Tensor, n_users: int, n_items: int):
        self.edge_index = edge_index
        self.n_users = n_users
        self.n_items = n_items
        self.n_nodes = n_users + n_items
        
    def __len__(self):
        return self.edge_index.shape[1]
    
    def __getitem__(self, idx):
        return self.edge_index[:, idx]


def main():
    """数据预处理主函数"""
    dataset = AliRecommendDataset()
    
    # 加载数据
    user_data, item_data = dataset.load_data()
    
    # 预处理数据
    interaction_matrix, edge_index = dataset.preprocess_data(behavior_types=[4])  # 只考虑购买行为
    
    # 分割数据
    train_matrix, test_matrix = dataset.split_data(test_ratio=0.2)
    
    # 保存预处理数据
    dataset.save_processed_data()
    
    print("数据预处理完成！")
    print(f"用户数量: {dataset.n_users}")
    print(f"物品数量: {dataset.n_items}")
    print(f"训练集交互数: {train_matrix.sum()}")
    print(f"测试集交互数: {test_matrix.sum()}")


if __name__ == "__main__":
    main()