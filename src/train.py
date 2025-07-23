import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from tqdm import tqdm
import json
from datetime import datetime
from typing import Tuple

from .dataset import AliRecommendDataset
from .model import LightGCN
from .utils import (
    setup_logger, 
    set_random_seed, 
    load_config, 
    EarlyStopping,
    compute_metrics
)


class BPRLoss(nn.Module):
    """贝叶斯个性化排序损失函数"""
    
    def __init__(self):
        super().__init__()
        
    def forward(self, pos_scores: torch.Tensor, neg_scores: torch.Tensor) -> torch.Tensor:
        """
        计算BPR损失
        
        Args:
            pos_scores: 正样本分数 [batch_size]
            neg_scores: 负样本分数 [batch_size]
            
        Returns:
            BPR损失
        """
        diff = pos_scores - neg_scores
        loss = -torch.log(torch.sigmoid(diff)).mean()
        return loss


class LightGCNTrainer:
    """LightGCN训练器"""
    
    def __init__(self, model: LightGCN, device: torch.device = None):
        """
        初始化训练器
        
        Args:
            model: LightGCN模型
            device: 计算设备
        """
        self.model = model
        self.device = device if device else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        
        # 损失函数
        self.bpr_loss = BPRLoss()
        
    def create_bpr_batch(self, interaction_matrix: np.ndarray, 
                        batch_size: int = 1024) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        创建BPR训练批次
        
        Args:
            interaction_matrix: 交互矩阵
            batch_size: 批次大小
            
        Returns:
            user_indices: 用户索引
            pos_item_indices: 正样本物品索引
            neg_item_indices: 负样本物品索引
        """
        # 获取所有正样本
        users, pos_items = np.where(interaction_matrix > 0)
        n_positive = len(users)
        
        # 随机采样批次
        batch_indices = np.random.choice(n_positive, min(batch_size, n_positive), replace=False)
        batch_users = users[batch_indices]
        batch_pos_items = pos_items[batch_indices]
        
        # 为每个正样本生成负样本
        batch_neg_items = []
        for user_idx in batch_users:
            neg_item = np.random.randint(0, self.model.n_items)
            while interaction_matrix[user_idx, neg_item] > 0:
                neg_item = np.random.randint(0, self.model.n_items)
            batch_neg_items.append(neg_item)
        
        return (
            torch.tensor(batch_users, dtype=torch.long, device=self.device),
            torch.tensor(batch_pos_items, dtype=torch.long, device=self.device),
            torch.tensor(batch_neg_items, dtype=torch.long, device=self.device)
        )
    
    def train_epoch(self, edge_index: torch.Tensor, interaction_matrix: np.ndarray,
                   optimizer: torch.optim.Optimizer, batch_size: int = 1024,
                   n_batches: int = 50) -> float:
        """
        训练一个epoch
        
        Args:
            edge_index: 边索引
            interaction_matrix: 交互矩阵
            optimizer: 优化器
            batch_size: 批次大小
            n_batches: 批次数量
            
        Returns:
            平均损失
        """
        self.model.train()
        total_loss = 0.0
        
        for _ in range(n_batches):
            optimizer.zero_grad()
            
            # 前向传播获取嵌入
            user_embeddings, item_embeddings = self.model(edge_index)
            
            # 创建训练批次
            user_indices, pos_item_indices, neg_item_indices = self.create_bpr_batch(
                interaction_matrix, batch_size
            )
            
            # 计算预测分数
            pos_scores = self.model.predict(user_indices, pos_item_indices, 
                                          user_embeddings, item_embeddings)
            neg_scores = self.model.predict(user_indices, neg_item_indices,
                                          user_embeddings, item_embeddings)
            
            # 计算损失
            loss = self.bpr_loss(pos_scores, neg_scores)
            
            # 反向传播
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        return total_loss / n_batches
    
    def train(self, edge_index: torch.Tensor, interaction_matrix: np.ndarray,
              n_epochs: int = 100, lr: float = 0.001, weight_decay: float = 1e-4,
              batch_size: int = 1024, n_batches: int = 50, 
              verbose: bool = True, eval_every: int = 10) -> dict:
        """
        完整训练过程
        
        Args:
            edge_index: 边索引
            interaction_matrix: 交互矩阵
            n_epochs: 训练轮数
            lr: 学习率
            weight_decay: 权重衰减
            batch_size: 批次大小
            n_batches: 每个epoch的批次数量
            verbose: 是否打印训练信息
            eval_every: 每隔多少轮评估一次
            
        Returns:
            训练历史信息
        """
        # 创建优化器
        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr, weight_decay=weight_decay)
        
        # 训练历史
        history = {
            'loss': [],
            'epoch': []
        }
        
        # 将边索引移到设备上
        edge_index = edge_index.to(self.device)
        
        if verbose:
            print(f"开始训练LightGCN模型...")
            print(f"设备: {self.device}")
            print(f"用户数: {self.model.n_users}, 物品数: {self.model.n_items}")
            print(f"嵌入维度: {self.model.embedding_dim}, GCN层数: {self.model.n_layers}")
            print("-" * 50)
        
        for epoch in range(n_epochs):
            # 训练一个epoch
            avg_loss = self.train_epoch(
                edge_index, interaction_matrix, optimizer, 
                batch_size, n_batches
            )
            
            # 记录训练历史
            history['loss'].append(avg_loss)
            history['epoch'].append(epoch)
            
            # 打印训练信息
            if verbose and (epoch + 1) % eval_every == 0:
                print(f"Epoch {epoch + 1:3d}/{n_epochs} | Loss: {avg_loss:.4f}")
        
        if verbose:
            print("-" * 50)
            print("训练完成!")
        
        return history
    
    def evaluate(self, edge_index: torch.Tensor, test_data: np.ndarray,
                 k: int = 20) -> dict:
        """
        评估模型性能
        
        Args:
            edge_index: 边索引
            test_data: 测试数据 (user_id, item_id) pairs
            k: top-k推荐
            
        Returns:
            评估指标
        """
        self.model.eval()
        
        with torch.no_grad():
            # 获取用户和物品嵌入
            user_embeddings, item_embeddings = self.model(edge_index.to(self.device))
            
            # 计算推荐指标
            hit_count = 0
            total_users = len(set(test_data[:, 0]))
            
            for user_id in set(test_data[:, 0]):
                # 获取该用户的测试物品
                user_test_items = set(test_data[test_data[:, 0] == user_id, 1])
                
                # 生成推荐
                recommendations = self.model.recommend(
                    user_id, user_embeddings, item_embeddings, k
                ).cpu().numpy()
                
                # 计算命中率
                if len(set(recommendations) & user_test_items) > 0:
                    hit_count += 1
            
            hit_rate = hit_count / total_users
        
        return {'hit_rate@{}'.format(k): hit_rate}


class LightGCNTrainingPipeline:
    """LightGCN训练管道"""
    
    def __init__(self, config_path: str = None):
        """
        初始化训练管道
        
        Args:
            config_path: 配置文件路径
        """
        # 加载配置
        self.config = load_config(config_path) if config_path else load_config('')
        
        # 设置随机种子
        set_random_seed(42)
        
        # 设置设备
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # 设置日志
        log_dir = 'outputs/logs'
        os.makedirs(log_dir, exist_ok=True)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        log_file = os.path.join(log_dir, f'training_{timestamp}.log')
        self.logger = setup_logger('Training', log_file)
        
        self.logger.info(f"使用设备: {self.device}")
        self.logger.info(f"配置: {json.dumps(self.config, indent=2, ensure_ascii=False)}")
        
        # 初始化组件
        self.dataset = None
        self.model = None
        self.trainer = None
        self.optimizer = None
        self.early_stopping = None
        
    def prepare_data(self):
        """准备数据"""
        self.logger.info("开始准备数据...")
        
        # 初始化数据集
        self.dataset = AliRecommendDataset()
        
        # 尝试加载预处理数据
        if not self.dataset.load_processed_data():
            self.logger.info("未找到预处理数据，开始数据预处理...")
            
            # 加载原始数据
            user_data, item_data = self.dataset.load_data()
            
            # 预处理数据
            interaction_matrix, edge_index = self.dataset.preprocess_data(behavior_types=[4])
            
            # 保存预处理数据
            self.dataset.save_processed_data()
        else:
            self.logger.info("成功加载预处理数据")
        
        # 分割训练和测试数据
        self.train_matrix, self.test_matrix = self.dataset.split_data(
            test_ratio=self.config['evaluation']['test_ratio']
        )
        
        # 将边索引移到设备
        self.edge_index = self.dataset.edge_index.to(self.device)
        
        self.logger.info(f"数据准备完成")
        self.logger.info(f"用户数: {self.dataset.n_users}, 物品数: {self.dataset.n_items}")
        self.logger.info(f"训练交互数: {self.train_matrix.sum()}, 测试交互数: {self.test_matrix.sum()}")
        
    def build_model(self):
        """构建模型"""
        self.logger.info("开始构建模型...")
        
        model_config = self.config['model']
        
        # 创建模型
        self.model = LightGCN(
            n_users=self.dataset.n_users,
            n_items=self.dataset.n_items,
            embedding_dim=model_config['embedding_dim'],
            n_layers=model_config['n_layers'],
            dropout=model_config['dropout']
        )
        
        # 创建训练器
        self.trainer = LightGCNTrainer(self.model, self.device)
        
        # 创建优化器
        training_config = self.config['training']
        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=training_config['learning_rate'],
            weight_decay=training_config['weight_decay']
        )
        
        # 创建早停机制
        self.early_stopping = EarlyStopping(patience=10, min_delta=0.001)
        
        # 计算模型参数数量
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        
        self.logger.info(f"模型构建完成")
        self.logger.info(f"总参数数: {total_params:,}")
        self.logger.info(f"可训练参数数: {trainable_params:,}")
        
    def evaluate_model(self, interaction_matrix: np.ndarray) -> dict:
        """
        评估模型
        
        Args:
            interaction_matrix: 测试交互矩阵
            
        Returns:
            评估指标字典
        """
        self.model.eval()
        
        with torch.no_grad():
            # 获取用户和物品嵌入
            user_embeddings, item_embeddings = self.model(self.edge_index)
            
            # 获取所有评分预测
            all_ratings = self.model.get_all_ratings(user_embeddings, item_embeddings)
            all_ratings = all_ratings.cpu().numpy()
            
            # 计算评估指标
            metrics = compute_metrics(
                all_ratings, 
                interaction_matrix, 
                k=self.config['evaluation']['k']
            )
        
        return metrics
    
    def train(self, config=None):
        """训练模型"""
        if config:
            self.config.update(config)
            
        if self.dataset is None:
            self.prepare_data()
        
        if self.model is None:
            self.build_model()
        
        self.logger.info("开始训练...")
        
        training_config = self.config['training']
        best_f1 = 0.0
        best_epoch = 0
        
        # 训练循环
        for epoch in range(training_config['epochs']):
            # 训练一个epoch
            train_loss = self.trainer.train_epoch(
                edge_index=self.edge_index,
                interaction_matrix=self.train_matrix,
                optimizer=self.optimizer,
                batch_size=training_config['batch_size'],
                n_batches=100
            )
            
            # 每10个epoch评估一次
            if (epoch + 1) % 10 == 0:
                # 在测试集上评估
                test_metrics = self.evaluate_model(self.test_matrix)
                
                self.logger.info(
                    f"Epoch {epoch+1}/{training_config['epochs']} - "
                    f"Train Loss: {train_loss:.4f}, "
                    f"Test F1: {test_metrics['f1']:.4f}"
                )
                
                # 保存最佳模型
                if test_metrics['f1'] > best_f1:
                    best_f1 = test_metrics['f1']
                    best_epoch = epoch + 1
                    self.save_model('best_model.pt')
                    self.logger.info(f"保存最佳模型 (F1: {best_f1:.4f})")
                
                # 早停检查
                self.early_stopping(test_metrics['f1'])
                if self.early_stopping.early_stop:
                    self.logger.info(f"早停触发，在第 {epoch+1} 轮停止训练")
                    break
            else:
                if (epoch + 1) % 5 == 0:
                    self.logger.info(f"Epoch {epoch+1}/{training_config['epochs']} - Train Loss: {train_loss:.4f}")
        
        self.logger.info(f"训练完成！最佳F1分数: {best_f1:.4f} (第 {best_epoch} 轮)")
        
        # 加载最佳模型进行最终评估
        self.load_model('best_model.pt')
        final_metrics = self.evaluate_model(self.test_matrix)
        
        self.logger.info("最终测试结果:")
        self.logger.info(f"Precision: {final_metrics['precision']:.4f}")
        self.logger.info(f"Recall: {final_metrics['recall']:.4f}")
        self.logger.info(f"F1-Score: {final_metrics['f1']:.4f}")
        
        return final_metrics
    
    def save_model(self, filename: str):
        """保存模型"""
        model_dir = 'outputs/models'
        os.makedirs(model_dir, exist_ok=True)
        
        model_path = os.path.join(model_dir, filename)
        
        # 保存模型状态和相关信息
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'config': self.config,
            'n_users': self.dataset.n_users,
            'n_items': self.dataset.n_items,
            'user_mapping': self.dataset.user_mapping,
            'item_mapping': self.dataset.item_mapping
        }
        
        torch.save(checkpoint, model_path)
        self.logger.info(f"模型已保存至: {model_path}")
    
    def load_model(self, filename: str):
        """加载模型"""
        model_dir = 'outputs/models'
        model_path = os.path.join(model_dir, filename)
        
        if os.path.exists(model_path):
            checkpoint = torch.load(model_path, map_location=self.device)
            
            # 如果模型未初始化，先初始化
            if self.model is None:
                model_config = checkpoint['config']['model']
                self.model = LightGCN(
                    n_users=checkpoint['n_users'],
                    n_items=checkpoint['n_items'],
                    embedding_dim=model_config['embedding_dim'],
                    n_layers=model_config['n_layers'],
                    dropout=model_config['dropout']
                )
                self.model.to(self.device)
            
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.logger.info(f"模型已从 {model_path} 加载")
            return True
        else:
            self.logger.warning(f"模型文件不存在: {model_path}")
            return False
    
    def generate_recommendations(self, user_ids: list = None, k: int = 20):
        """
        生成推荐结果
        
        Args:
            user_ids: 用户ID列表，如果为None则为所有用户生成推荐
            k: 推荐数量
            
        Returns:
            推荐结果列表 [(original_user_id, original_item_id), ...]
        """
        if self.model is None:
            raise ValueError("模型未初始化，请先训练或加载模型")
        
        self.model.eval()
        recommendations = []
        
        # 创建反向映射
        reverse_user_mapping = {v: k for k, v in self.dataset.user_mapping.items()}
        reverse_item_mapping = {v: k for k, v in self.dataset.item_mapping.items()}
        
        with torch.no_grad():
            # 获取用户和物品嵌入
            user_embeddings, item_embeddings = self.model(self.edge_index)
            
            # 如果未指定用户，为所有用户生成推荐
            if user_ids is None:
                user_indices = list(range(self.dataset.n_users))
            else:
                user_indices = [self.dataset.user_mapping[uid] for uid in user_ids 
                              if uid in self.dataset.user_mapping]
            
            for user_idx in user_indices:
                # 为用户生成推荐
                recommended_items = self.model.recommend(user_idx, user_embeddings, item_embeddings, k)
                
                # 转换回原始ID
                original_user_id = reverse_user_mapping[user_idx]
                
                for item_idx in recommended_items.cpu().numpy():
                    original_item_id = reverse_item_mapping[item_idx]
                    recommendations.append((original_user_id, original_item_id))
        
        return recommendations


def main():
    """主训练函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description='LightGCN模型训练')
    parser.add_argument('--config', type=str, default='config.json',
                       help='配置文件路径')
    parser.add_argument('--epochs', type=int, default=None,
                       help='训练轮数（覆盖配置文件）')
    parser.add_argument('--batch-size', type=int, default=None,
                       help='批次大小（覆盖配置文件）')
    parser.add_argument('--learning-rate', type=float, default=None,
                       help='学习率（覆盖配置文件）')
    parser.add_argument('--embedding-dim', type=int, default=None,
                       help='嵌入维度（覆盖配置文件）')
    
    args = parser.parse_args()
    
    print("🚀 开始LightGCN模型训练...")
    
    # 创建训练管道
    pipeline = LightGCNTrainingPipeline(args.config)
    
    # 覆盖配置参数
    config_override = {}
    if args.epochs is not None:
        config_override.setdefault('training', {})['epochs'] = args.epochs
    if args.batch_size is not None:
        config_override.setdefault('training', {})['batch_size'] = args.batch_size
    if args.learning_rate is not None:
        config_override.setdefault('training', {})['learning_rate'] = args.learning_rate
    if args.embedding_dim is not None:
        config_override.setdefault('model', {})['embedding_dim'] = args.embedding_dim
    
    try:
        # 训练模型
        print(f"📊 使用配置文件: {args.config}")
        if config_override:
            print(f"🔧 参数覆盖: {config_override}")
            
        final_metrics = pipeline.train(config_override)
        
        print("\n🎉 训练完成！")
        print(f"📈 最终指标:")
        print(f"  - Precision: {final_metrics['precision']:.4f}")
        print(f"  - Recall: {final_metrics['recall']:.4f}")
        print(f"  - F1-Score: {final_metrics['f1']:.4f}")
        
        # 生成推荐结果示例
        print("\n🎯 生成推荐结果示例...")
        sample_recommendations = pipeline.generate_recommendations(k=5)
        
        print(f"为前3个用户生成的推荐结果:")
        current_user = None
        count = 0
        for user_id, item_id in sample_recommendations[:15]:
            if user_id != current_user:
                if count >= 3:
                    break
                current_user = user_id
                count += 1
                print(f"\n👤 用户 {user_id} 的推荐:")
            print(f"  - 物品 {item_id}")
        
        print("\n✅ 训练和推荐生成完成！")
        print("📁 模型已保存到: outputs/models/best_model.pt")
        print("🔄 转换ONNX: python service/convert_to_onnx.py --model-path outputs/models/best_model.pt")
        print("🚀 启动服务: python service/onnx_server.py --model-path outputs/models/best_model.onnx")
        print("🎨 启动界面: python run_app.py")
        
    except Exception as e:
        print(f"❌ 训练过程中出现错误: {e}")
        raise
    
    return 0


if __name__ == "__main__":
    main()