import os
import torch
import torch.optim as optim
import numpy as np
from tqdm import tqdm
import json
from datetime import datetime

from .dataset import AliRecommendDataset
from .model import LightGCN, LightGCNTrainer
from .utils import (
    setup_logger, 
    set_random_seed, 
    load_config, 
    EarlyStopping,
    compute_metrics
)


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