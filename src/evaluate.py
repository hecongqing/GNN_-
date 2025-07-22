import os
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple
import json

from .dataset import AliRecommendDataset
from .model import LightGCN
from .train import LightGCNTrainingPipeline
from .utils import (
    setup_logger,
    compute_metrics,
    save_predictions
)


class ModelEvaluator:
    """模型评估器"""
    
    def __init__(self, model_path: str = 'outputs/models/best_model.pt'):
        """
        初始化评估器
        
        Args:
            model_path: 模型文件路径
        """
        self.model_path = model_path
        self.logger = setup_logger('Evaluator')
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # 初始化组件
        self.pipeline = None
        self.dataset = None
        self.model = None
        
    def load_model_and_data(self):
        """加载模型和数据"""
        self.logger.info("加载模型和数据...")
        
        # 创建训练管道
        self.pipeline = LightGCNTrainingPipeline()
        
        # 准备数据
        self.pipeline.prepare_data()
        self.dataset = self.pipeline.dataset
        
        # 加载模型
        if self.pipeline.load_model(os.path.basename(self.model_path)):
            self.model = self.pipeline.model
            self.logger.info("模型和数据加载完成")
        else:
            raise FileNotFoundError(f"无法加载模型: {self.model_path}")
    
    def evaluate_on_test_set(self) -> Dict[str, float]:
        """在测试集上评估模型"""
        if self.pipeline is None:
            self.load_model_and_data()
        
        self.logger.info("在测试集上评估模型...")
        
        # 分割数据
        train_matrix, test_matrix = self.dataset.split_data(test_ratio=0.2)
        
        # 评估模型
        metrics = self.pipeline.evaluate_model(test_matrix)
        
        self.logger.info("测试集评估结果:")
        self.logger.info(f"Precision@20: {metrics['precision']:.4f}")
        self.logger.info(f"Recall@20: {metrics['recall']:.4f}")
        self.logger.info(f"F1-Score@20: {metrics['f1']:.4f}")
        
        return metrics
    
    def evaluate_at_different_k(self, k_values: List[int] = [5, 10, 20, 50]) -> Dict[int, Dict[str, float]]:
        """在不同k值下评估模型"""
        if self.pipeline is None:
            self.load_model_and_data()
        
        self.logger.info(f"在不同k值下评估模型: {k_values}")
        
        # 分割数据
        train_matrix, test_matrix = self.dataset.split_data(test_ratio=0.2)
        
        # 获取模型预测
        self.model.eval()
        with torch.no_grad():
            user_embeddings, item_embeddings = self.model(self.pipeline.edge_index)
            all_ratings = self.model.get_all_ratings(user_embeddings, item_embeddings)
            all_ratings = all_ratings.cpu().numpy()
        
        # 在不同k值下计算指标
        results = {}
        for k in k_values:
            metrics = compute_metrics(all_ratings, test_matrix, k=k)
            results[k] = metrics
            self.logger.info(f"k={k}: Precision={metrics['precision']:.4f}, "
                           f"Recall={metrics['recall']:.4f}, F1={metrics['f1']:.4f}")
        
        return results
    
    def analyze_user_item_statistics(self) -> Dict[str, any]:
        """分析用户和物品的统计信息"""
        if self.dataset is None:
            self.load_model_and_data()
        
        self.logger.info("分析用户和物品统计信息...")
        
        interaction_matrix = self.dataset.interaction_matrix
        
        # 用户统计
        user_interactions = interaction_matrix.sum(axis=1)  # 每个用户的交互数
        user_stats = {
            'total_users': len(user_interactions),
            'avg_interactions_per_user': user_interactions.mean(),
            'median_interactions_per_user': np.median(user_interactions),
            'max_interactions_per_user': user_interactions.max(),
            'min_interactions_per_user': user_interactions.min(),
            'users_with_single_interaction': (user_interactions == 1).sum()
        }
        
        # 物品统计
        item_interactions = interaction_matrix.sum(axis=0)  # 每个物品的交互数
        item_stats = {
            'total_items': len(item_interactions),
            'avg_interactions_per_item': item_interactions.mean(),
            'median_interactions_per_item': np.median(item_interactions),
            'max_interactions_per_item': item_interactions.max(),
            'min_interactions_per_item': item_interactions.min(),
            'items_with_single_interaction': (item_interactions == 1).sum()
        }
        
        # 总体统计
        total_interactions = interaction_matrix.sum()
        sparsity = 1 - (total_interactions / (interaction_matrix.shape[0] * interaction_matrix.shape[1]))
        
        overall_stats = {
            'total_interactions': int(total_interactions),
            'sparsity': sparsity,
            'density': 1 - sparsity
        }
        
        stats = {
            'user_stats': user_stats,
            'item_stats': item_stats,
            'overall_stats': overall_stats
        }
        
        # 打印统计信息
        self.logger.info("用户统计:")
        for key, value in user_stats.items():
            self.logger.info(f"  {key}: {value}")
        
        self.logger.info("物品统计:")
        for key, value in item_stats.items():
            self.logger.info(f"  {key}: {value}")
        
        self.logger.info("总体统计:")
        for key, value in overall_stats.items():
            self.logger.info(f"  {key}: {value}")
        
        return stats
    
    def generate_recommendation_report(self, sample_users: List[int] = None, k: int = 20) -> pd.DataFrame:
        """生成推荐报告"""
        if self.pipeline is None:
            self.load_model_and_data()
        
        self.logger.info("生成推荐报告...")
        
        # 如果未指定用户，随机选择一些用户
        if sample_users is None:
            all_user_ids = list(self.dataset.user_mapping.keys())
            sample_users = np.random.choice(all_user_ids, min(10, len(all_user_ids)), replace=False)
        
        # 生成推荐
        recommendations = self.pipeline.generate_recommendations(sample_users, k)
        
        # 转换为DataFrame
        df_recommendations = pd.DataFrame(recommendations, columns=['user_id', 'item_id'])
        
        # 添加排名信息
        df_recommendations['rank'] = df_recommendations.groupby('user_id').cumcount() + 1
        
        # 保存推荐结果
        output_path = 'outputs/recommendations_report.csv'
        df_recommendations.to_csv(output_path, index=False)
        self.logger.info(f"推荐报告已保存至: {output_path}")
        
        return df_recommendations
    
    def plot_metrics_comparison(self, k_values: List[int] = [5, 10, 20, 50]):
        """绘制不同k值下的指标对比图"""
        # 获取不同k值下的指标
        results = self.evaluate_at_different_k(k_values)
        
        # 准备数据
        metrics_data = {
            'k': k_values,
            'precision': [results[k]['precision'] for k in k_values],
            'recall': [results[k]['recall'] for k in k_values],
            'f1': [results[k]['f1'] for k in k_values]
        }
        
        # 创建图表
        plt.figure(figsize=(12, 4))
        
        # Precision图
        plt.subplot(1, 3, 1)
        plt.plot(k_values, metrics_data['precision'], 'bo-', linewidth=2, markersize=8)
        plt.title('Precision@K')
        plt.xlabel('K')
        plt.ylabel('Precision')
        plt.grid(True, alpha=0.3)
        
        # Recall图
        plt.subplot(1, 3, 2)
        plt.plot(k_values, metrics_data['recall'], 'ro-', linewidth=2, markersize=8)
        plt.title('Recall@K')
        plt.xlabel('K')
        plt.ylabel('Recall')
        plt.grid(True, alpha=0.3)
        
        # F1图
        plt.subplot(1, 3, 3)
        plt.plot(k_values, metrics_data['f1'], 'go-', linewidth=2, markersize=8)
        plt.title('F1-Score@K')
        plt.xlabel('K')
        plt.ylabel('F1-Score')
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # 保存图表
        output_path = 'outputs/metrics_comparison.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        self.logger.info(f"指标对比图已保存至: {output_path}")
    
    def plot_interaction_distribution(self):
        """绘制交互分布图"""
        if self.dataset is None:
            self.load_model_and_data()
        
        interaction_matrix = self.dataset.interaction_matrix
        
        # 用户交互分布
        user_interactions = interaction_matrix.sum(axis=1)
        item_interactions = interaction_matrix.sum(axis=0)
        
        plt.figure(figsize=(15, 5))
        
        # 用户交互分布直方图
        plt.subplot(1, 3, 1)
        plt.hist(user_interactions, bins=50, alpha=0.7, color='blue', edgecolor='black')
        plt.title('用户交互数分布')
        plt.xlabel('交互数')
        plt.ylabel('用户数量')
        plt.grid(True, alpha=0.3)
        
        # 物品交互分布直方图
        plt.subplot(1, 3, 2)
        plt.hist(item_interactions, bins=50, alpha=0.7, color='red', edgecolor='black')
        plt.title('物品交互数分布')
        plt.xlabel('交互数')
        plt.ylabel('物品数量')
        plt.grid(True, alpha=0.3)
        
        # 交互矩阵热力图 (采样显示)
        plt.subplot(1, 3, 3)
        sample_size = min(100, interaction_matrix.shape[0])
        sample_indices = np.random.choice(interaction_matrix.shape[0], sample_size, replace=False)
        sample_matrix = interaction_matrix[sample_indices][:, :min(100, interaction_matrix.shape[1])]
        
        sns.heatmap(sample_matrix, cmap='Blues', cbar=True)
        plt.title('交互矩阵热力图 (采样)')
        plt.xlabel('物品 (采样)')
        plt.ylabel('用户 (采样)')
        
        plt.tight_layout()
        
        # 保存图表
        output_path = 'outputs/interaction_distribution.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        self.logger.info(f"交互分布图已保存至: {output_path}")
    
    def export_predictions_for_submission(self, output_path: str = 'outputs/submission.txt'):
        """导出提交格式的预测结果"""
        if self.pipeline is None:
            self.load_model_and_data()
        
        self.logger.info("导出提交格式的预测结果...")
        
        # 为所有用户生成推荐
        recommendations = self.pipeline.generate_recommendations(k=20)
        
        # 保存为提交格式
        save_predictions(recommendations, output_path)
        
        self.logger.info(f"提交结果已保存至: {output_path}")
        self.logger.info(f"总推荐数: {len(recommendations)}")
    
    def comprehensive_evaluation(self) -> Dict[str, any]:
        """综合评估"""
        self.logger.info("开始综合评估...")
        
        # 基本指标评估
        basic_metrics = self.evaluate_on_test_set()
        
        # 不同k值下的评估
        k_metrics = self.evaluate_at_different_k([5, 10, 20, 50])
        
        # 统计信息分析
        stats = self.analyze_user_item_statistics()
        
        # 生成推荐报告
        recommendation_report = self.generate_recommendation_report()
        
        # 绘制图表
        self.plot_metrics_comparison()
        self.plot_interaction_distribution()
        
        # 导出提交结果
        self.export_predictions_for_submission()
        
        # 汇总结果
        evaluation_results = {
            'basic_metrics': basic_metrics,
            'k_metrics': k_metrics,
            'statistics': stats,
            'recommendation_count': len(recommendation_report)
        }
        
        # 保存评估结果
        with open('outputs/evaluation_results.json', 'w', encoding='utf-8') as f:
            json.dump(evaluation_results, f, indent=2, ensure_ascii=False)
        
        self.logger.info("综合评估完成！")
        return evaluation_results


def main():
    """主评估函数"""
    try:
        # 创建评估器
        evaluator = ModelEvaluator()
        
        # 执行综合评估
        results = evaluator.comprehensive_evaluation()
        
        print("\n=== 综合评估结果 ===")
        print(f"测试集 Precision@20: {results['basic_metrics']['precision']:.4f}")
        print(f"测试集 Recall@20: {results['basic_metrics']['recall']:.4f}")
        print(f"测试集 F1-Score@20: {results['basic_metrics']['f1']:.4f}")
        
        print("\n不同k值下的F1分数:")
        for k, metrics in results['k_metrics'].items():
            print(f"  F1@{k}: {metrics['f1']:.4f}")
        
        print(f"\n数据稀疏度: {results['statistics']['overall_stats']['sparsity']:.4f}")
        print(f"总交互数: {results['statistics']['overall_stats']['total_interactions']:,}")
        print(f"推荐结果数: {results['recommendation_count']:,}")
        
        print("\n所有结果已保存到 outputs/ 目录")
        
    except Exception as e:
        print(f"评估过程中出现错误: {e}")
        raise


if __name__ == "__main__":
    main()