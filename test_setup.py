#!/usr/bin/env python3
"""
项目设置测试脚本
用于验证项目环境和依赖是否正确配置
"""

import sys
import os
import importlib.util
from typing import List, Tuple

def test_import(module_name: str, package: str = None) -> Tuple[bool, str]:
    """测试模块导入"""
    try:
        if package:
            module = importlib.import_module(module_name, package)
        else:
            module = importlib.import_module(module_name)
        return True, f"✓ {module_name}"
    except ImportError as e:
        return False, f"✗ {module_name}: {e}"

def test_file_exists(file_path: str) -> Tuple[bool, str]:
    """测试文件是否存在"""
    if os.path.exists(file_path):
        return True, f"✓ {file_path}"
    else:
        return False, f"✗ {file_path}: 文件不存在"

def test_directory_exists(dir_path: str) -> Tuple[bool, str]:
    """测试目录是否存在"""
    if os.path.isdir(dir_path):
        return True, f"✓ {dir_path}/"
    else:
        return False, f"✗ {dir_path}/: 目录不存在"

def main():
    """主测试函数"""
    print("=" * 60)
    print("LightGCN推荐系统 - 项目设置测试")
    print("=" * 60)
    
    # 测试结果
    results = []
    
    # 1. 测试Python基础库
    print("\n1. 测试Python基础库:")
    basic_modules = [
        'numpy', 'pandas', 'matplotlib', 'seaborn', 
        'sklearn', 'tqdm', 'json', 'pickle'
    ]
    
    for module in basic_modules:
        success, message = test_import(module)
        results.append(success)
        print(f"   {message}")
    
    # 2. 测试PyTorch和PyTorch Geometric
    print("\n2. 测试PyTorch和PyTorch Geometric:")
    torch_modules = ['torch', 'torch_geometric']
    
    for module in torch_modules:
        success, message = test_import(module)
        results.append(success)
        print(f"   {message}")
    
    # 3. 测试项目结构
    print("\n3. 测试项目结构:")
    
    # 目录结构
    directories = [
        'src', 'dataset', 'outputs', 'outputs/logs', 
        'outputs/models', 'service'
    ]
    
    for directory in directories:
        success, message = test_directory_exists(directory)
        results.append(success)
        print(f"   {message}")
    
    # 核心文件
    files = [
        'requirements.txt', 'README.md', 'config.json',
        'run_train.py', 'run_evaluate.py',
        'src/__init__.py', 'src/utils.py', 'src/dataset.py',
        'src/model.py', 'src/train.py', 'src/evaluate.py',
        'service/prediction_service.py'
    ]
    
    for file_path in files:
        success, message = test_file_exists(file_path)
        results.append(success)
        print(f"   {message}")
    
    # 4. 测试项目模块导入
    print("\n4. 测试项目模块导入:")
    
    # 添加src到路径
    sys.path.insert(0, 'src')
    
    project_modules = [
        'src.utils', 'src.dataset', 'src.model', 
        'src.train', 'src.evaluate'
    ]
    
    for module in project_modules:
        success, message = test_import(module)
        results.append(success)
        print(f"   {message}")
    
    # 5. 测试PyTorch功能
    print("\n5. 测试PyTorch功能:")
    
    try:
        import torch
        
        # 测试基本tensor操作
        x = torch.randn(3, 4)
        y = torch.randn(4, 5)
        z = torch.mm(x, y)
        print(f"   ✓ PyTorch tensor操作正常")
        results.append(True)
        
        # 测试设备
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"   ✓ 使用设备: {device}")
        results.append(True)
        
        # 测试神经网络
        model = torch.nn.Linear(10, 5)
        x = torch.randn(32, 10)
        y = model(x)
        print(f"   ✓ PyTorch神经网络正常")
        results.append(True)
        
    except Exception as e:
        print(f"   ✗ PyTorch功能测试失败: {e}")
        results.extend([False, False, False])
    
    # 6. 测试PyTorch Geometric功能
    print("\n6. 测试PyTorch Geometric功能:")
    
    try:
        import torch_geometric
        from torch_geometric.nn import MessagePassing
        from torch_geometric.utils import degree
        
        print(f"   ✓ PyTorch Geometric导入正常")
        results.append(True)
        
        # 测试基本图操作
        edge_index = torch.tensor([[0, 1, 1, 2], [1, 0, 2, 1]], dtype=torch.long)
        x = torch.randn(3, 16)
        
        # 计算度
        deg = degree(edge_index[1], num_nodes=x.size(0))
        print(f"   ✓ 图操作正常，节点度: {deg}")
        results.append(True)
        
    except Exception as e:
        print(f"   ✗ PyTorch Geometric功能测试失败: {e}")
        results.extend([False, False])
    
    # 7. 测试数据处理功能
    print("\n7. 测试数据处理功能:")
    
    try:
        from src.dataset import AliRecommendDataset
        
        # 创建数据集实例（会生成示例数据）
        dataset = AliRecommendDataset()
        user_data, item_data = dataset.load_data()
        
        print(f"   ✓ 数据集加载正常，用户数据: {user_data.shape}")
        results.append(True)
        
        # 测试预处理
        interaction_matrix, edge_index = dataset.preprocess_data()
        print(f"   ✓ 数据预处理正常，交互矩阵: {interaction_matrix.shape}")
        results.append(True)
        
    except Exception as e:
        print(f"   ✗ 数据处理功能测试失败: {e}")
        results.extend([False, False])
    
    # 8. 测试模型构建
    print("\n8. 测试模型构建:")
    
    try:
        from src.model import LightGCN
        
        # 创建模型
        model = LightGCN(n_users=100, n_items=50, embedding_dim=32, n_layers=2)
        print(f"   ✓ LightGCN模型创建正常")
        results.append(True)
        
        # 测试前向传播
        edge_index = torch.tensor([[0, 1, 1, 2], [1, 0, 2, 1]], dtype=torch.long)
        user_emb, item_emb = model(edge_index)
        print(f"   ✓ 模型前向传播正常，用户嵌入: {user_emb.shape}")
        results.append(True)
        
    except Exception as e:
        print(f"   ✗ 模型构建测试失败: {e}")
        results.extend([False, False])
    
    # 总结
    print("\n" + "=" * 60)
    passed = sum(results)
    total = len(results)
    
    if passed == total:
        print(f"🎉 所有测试通过! ({passed}/{total})")
        print("项目环境配置完成，可以开始训练模型。")
        return 0
    else:
        print(f"⚠️  部分测试失败 ({passed}/{total})")
        print("请检查失败的模块并安装相应依赖。")
        return 1

if __name__ == "__main__":
    exit(main())