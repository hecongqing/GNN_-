#!/usr/bin/env python3
"""
LightGCN推荐系统训练启动脚本
"""

import sys
import os
import argparse

# 添加src目录到Python路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from src.train import main as train_main


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='训练LightGCN推荐模型')
    parser.add_argument('--config', type=str, default='config.json', 
                       help='配置文件路径')
    parser.add_argument('--data-dir', type=str, default='dataset',
                       help='数据目录路径')
    parser.add_argument('--output-dir', type=str, default='outputs',
                       help='输出目录路径')
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("LightGCN推荐系统训练")
    print("=" * 60)
    print(f"配置文件: {args.config}")
    print(f"数据目录: {args.data_dir}")
    print(f"输出目录: {args.output_dir}")
    print("=" * 60)
    
    try:
        # 启动训练
        train_main()
        print("\n训练完成！")
        
    except KeyboardInterrupt:
        print("\n训练被用户中断")
    except Exception as e:
        print(f"\n训练失败: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())