#!/usr/bin/env python3
"""
LightGCN系统功能测试脚本
"""

import os
import sys
import time

def test_dependencies():
    """测试依赖是否安装"""
    print("🧪 测试依赖包...")
    
    required_packages = [
        'torch', 'torch_geometric', 'pandas', 'numpy', 
        'sklearn', 'tqdm', 'flask', 'onnx', 'onnxruntime',
        'streamlit', 'plotly', 'matplotlib', 'seaborn'
    ]
    
    missing = []
    for package in required_packages:
        try:
            __import__(package)
            print(f"  ✅ {package}")
        except ImportError:
            missing.append(package)
            print(f"  ❌ {package}")
    
    if missing:
        print(f"\n❌ 缺少依赖: {', '.join(missing)}")
        print("请运行: pip install -r requirements.txt")
        return False
    
    print("✅ 所有依赖已安装")
    return True


def test_project_structure():
    """测试项目结构"""
    print("\n🧪 测试项目结构...")
    
    required_files = [
        'src/app.py', 'src/model.py', 'src/dataset.py', 
        'src/train.py', 'src/utils.py', 'src/__init__.py',
        'service/convert_to_onnx.py', 'service/onnx_server.py',
        'requirements.txt', 'config.json', 'run_app.py'
    ]
    
    required_dirs = ['dataset', 'outputs', 'src', 'service']
    
    missing = []
    
    # 检查目录
    for dir_name in required_dirs:
        if os.path.exists(dir_name):
            print(f"  ✅ {dir_name}/")
        else:
            missing.append(dir_name)
            print(f"  ❌ {dir_name}/")
    
    # 检查文件
    for file_name in required_files:
        if os.path.exists(file_name):
            print(f"  ✅ {file_name}")
        else:
            missing.append(file_name)
            print(f"  ❌ {file_name}")
    
    if missing:
        print(f"\n❌ 缺少文件/目录: {', '.join(missing)}")
        return False
    
    print("✅ 项目结构完整")
    return True


def test_data_processing():
    """测试数据处理功能"""
    print("\n🧪 测试数据处理...")
    
    try:
        from src.dataset import AliRecommendDataset
        
        # 创建数据集实例
        dataset = AliRecommendDataset()
        
        # 加载数据
        user_data, item_data = dataset.load_data()
        print(f"  ✅ 数据加载成功 - 用户数据: {user_data.shape}, 物品数据: {item_data.shape}")
        
        # 预处理数据
        interaction_matrix, edge_index = dataset.preprocess_data()
        print(f"  ✅ 数据预处理成功 - 交互矩阵: {interaction_matrix.shape}, 边数: {edge_index.shape[1]}")
        
        return True
        
    except Exception as e:
        print(f"  ❌ 数据处理失败: {e}")
        return False


def test_model_creation():
    """测试模型创建"""
    print("\n🧪 测试模型创建...")
    
    try:
        from src.model import LightGCN
        import torch
        
        # 创建模型
        model = LightGCN(n_users=100, n_items=50, embedding_dim=32, n_layers=2)
        print(f"  ✅ 模型创建成功")
        
        # 测试前向传播
        edge_index = torch.randint(0, 150, (2, 200))
        user_emb, item_emb = model(edge_index)
        print(f"  ✅ 前向传播成功 - 用户嵌入: {user_emb.shape}, 物品嵌入: {item_emb.shape}")
        
        return True
        
    except Exception as e:
        print(f"  ❌ 模型创建失败: {e}")
        return False


def test_training_pipeline():
    """测试训练管道（快速测试）"""
    print("\n🧪 测试训练管道...")
    
    try:
        from src.train import LightGCNTrainingPipeline
        
        # 创建训练管道
        pipeline = LightGCNTrainingPipeline()
        
        # 准备数据
        pipeline.prepare_data()
        print(f"  ✅ 数据准备成功")
        
        # 构建模型
        pipeline.build_model()
        print(f"  ✅ 模型构建成功")
        
        print(f"  ✅ 训练管道测试通过")
        return True
        
    except Exception as e:
        print(f"  ❌ 训练管道测试失败: {e}")
        return False


def main():
    """主测试函数"""
    print("🚀 LightGCN系统功能测试\n")
    
    tests = [
        ("依赖检查", test_dependencies),
        ("项目结构", test_project_structure),
        ("数据处理", test_data_processing),
        ("模型创建", test_model_creation),
        ("训练管道", test_training_pipeline),
    ]
    
    passed = 0
    total = len(tests)
    
    start_time = time.time()
    
    for test_name, test_func in tests:
        if test_func():
            passed += 1
    
    end_time = time.time()
    
    print(f"\n" + "="*50)
    print(f"测试完成！")
    print(f"通过: {passed}/{total}")
    print(f"耗时: {end_time - start_time:.2f}s")
    
    if passed == total:
        print("🎉 所有测试通过！系统就绪。")
        print("\n下一步:")
        print("  1. 运行 'python run_app.py' 启动Web界面")
        print("  2. 或运行 'python -m src.app --mode train' 开始训练")
        return 0
    else:
        print("❌ 部分测试失败，请检查上述错误信息。")
        return 1


if __name__ == "__main__":
    exit(main())