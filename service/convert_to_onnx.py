#!/usr/bin/env python3
"""
将训练好的LightGCN模型转换为ONNX格式
"""

import os
import sys
import torch
import torch.onnx
import argparse
import json
from pathlib import Path

# 添加src目录到路径
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from src.model import LightGCN
from src.dataset import AliRecommendDataset
from src.utils import setup_logger


class LightGCNONNXConverter:
    """LightGCN模型ONNX转换器"""
    
    def __init__(self, model_path: str, output_path: str = None):
        """
        初始化转换器
        
        Args:
            model_path: 训练好的模型路径
            output_path: ONNX模型输出路径
        """
        self.model_path = model_path
        self.output_path = output_path or model_path.replace('.pt', '.onnx')
        self.logger = setup_logger('ONNXConverter')
        self.device = torch.device('cpu')  # ONNX转换建议使用CPU
        
    def load_model(self):
        """加载训练好的模型"""
        self.logger.info(f"加载模型: {self.model_path}")
        
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"模型文件不存在: {self.model_path}")
        
        # 加载模型检查点
        checkpoint = torch.load(self.model_path, map_location=self.device)
        
        # 获取模型参数
        self.n_users = checkpoint.get('n_users', 1000)
        self.n_items = checkpoint.get('n_items', 500)
        self.embedding_dim = checkpoint.get('embedding_dim', 64)
        self.n_layers = checkpoint.get('n_layers', 3)
        
        # 创建模型
        self.model = LightGCN(
            n_users=self.n_users,
            n_items=self.n_items,
            embedding_dim=self.embedding_dim,
            n_layers=self.n_layers
        )
        
        # 加载模型权重
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()
        self.model.to(self.device)
        
        self.logger.info("模型加载完成")
        
    def convert_to_onnx(self):
        """转换模型为ONNX格式"""
        self.logger.info("开始转换为ONNX格式...")
        
        # 创建示例输入
        # edge_index: [2, num_edges] 的张量
        num_edges = min(self.n_users * 5, 10000)  # 限制边数以避免内存问题
        edge_index = torch.randint(
            0, max(self.n_users, self.n_items), 
            (2, num_edges), 
            dtype=torch.long
        )
        
        # 确保边索引有效
        edge_index[0] = edge_index[0] % self.n_users  # 用户索引
        edge_index[1] = edge_index[1] % self.n_items + self.n_users  # 物品索引
        
        # 输入名称和输出名称
        input_names = ['edge_index']
        output_names = ['user_embeddings', 'item_embeddings']
        
        # 动态轴配置
        dynamic_axes = {
            'edge_index': {1: 'num_edges'},
            'user_embeddings': {0: 'n_users'},
            'item_embeddings': {0: 'n_items'}
        }
        
        try:
            # 导出ONNX模型
            torch.onnx.export(
                self.model,
                edge_index,
                self.output_path,
                input_names=input_names,
                output_names=output_names,
                dynamic_axes=dynamic_axes,
                opset_version=11,
                do_constant_folding=True,
                verbose=False
            )
            
            self.logger.info(f"ONNX模型保存至: {self.output_path}")
            
            # 验证ONNX模型
            self._verify_onnx_model()
            
        except Exception as e:
            self.logger.error(f"ONNX转换失败: {e}")
            raise
    
    def _verify_onnx_model(self):
        """验证ONNX模型"""
        try:
            import onnx
            import onnxruntime as ort
            
            # 检查ONNX模型
            onnx_model = onnx.load(self.output_path)
            onnx.checker.check_model(onnx_model)
            self.logger.info("ONNX模型验证通过")
            
            # 测试推理
            ort_session = ort.InferenceSession(self.output_path)
            
            # 创建测试输入
            test_edge_index = torch.randint(
                0, max(self.n_users, self.n_items), 
                (2, 100), 
                dtype=torch.long
            ).numpy()
            
            test_edge_index[0] = test_edge_index[0] % self.n_users
            test_edge_index[1] = test_edge_index[1] % self.n_items + self.n_users
            
            # 运行推理
            outputs = ort_session.run(None, {'edge_index': test_edge_index})
            
            self.logger.info(f"ONNX推理测试成功，输出形状: {[out.shape for out in outputs]}")
            
        except ImportError:
            self.logger.warning("缺少onnx或onnxruntime库，跳过验证")
        except Exception as e:
            self.logger.warning(f"ONNX模型验证失败: {e}")
    
    def save_metadata(self):
        """保存模型元数据"""
        metadata = {
            'n_users': self.n_users,
            'n_items': self.n_items,
            'embedding_dim': self.embedding_dim,
            'n_layers': self.n_layers,
            'original_model_path': self.model_path,
            'onnx_model_path': self.output_path
        }
        
        metadata_path = self.output_path.replace('.onnx', '_metadata.json')
        with open(metadata_path, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)
        
        self.logger.info(f"元数据保存至: {metadata_path}")


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='将LightGCN模型转换为ONNX格式')
    parser.add_argument('--model-path', type=str, required=True,
                       help='训练好的模型路径(.pt文件)')
    parser.add_argument('--output-path', type=str,
                       help='ONNX模型输出路径(.onnx文件)')
    parser.add_argument('--save-metadata', action='store_true',
                       help='是否保存模型元数据')
    
    args = parser.parse_args()
    
    try:
        # 创建转换器
        converter = LightGCNONNXConverter(args.model_path, args.output_path)
        
        # 加载模型
        converter.load_model()
        
        # 转换为ONNX
        converter.convert_to_onnx()
        
        # 保存元数据
        if args.save_metadata:
            converter.save_metadata()
        
        print("✅ 模型转换完成！")
        
    except Exception as e:
        print(f"❌ 转换失败: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())