#!/usr/bin/env python3
"""
LightGCN推荐系统评估启动脚本
"""

import sys
import os
import argparse

# 添加src目录到Python路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from src.evaluate import main as evaluate_main


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='评估LightGCN推荐模型')
    parser.add_argument('--model-path', type=str, default='outputs/models/best_model.pt',
                       help='模型文件路径')
    parser.add_argument('--output-dir', type=str, default='outputs',
                       help='输出目录路径')
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("LightGCN推荐系统评估")
    print("=" * 60)
    print(f"模型路径: {args.model_path}")
    print(f"输出目录: {args.output_dir}")
    print("=" * 60)
    
    try:
        # 启动评估
        evaluate_main()
        print("\n评估完成！")
        
    except KeyboardInterrupt:
        print("\n评估被用户中断")
    except Exception as e:
        print(f"\n评估失败: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())