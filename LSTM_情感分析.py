#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
LSTM情感分析完整脚本
包含数据处理、模型训练、评估和预测的完整流程
"""

import os
import sys
import torch
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import jieba
import warnings
warnings.filterwarnings('ignore')

# 添加src目录到Python路径
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.model import LSTMSentimentModel
from src.dataset import SentimentDataProcessor
from src.train import SentimentTrainer
from src.utils import (
    setup_logger, 
    set_random_seed, 
    plot_confusion_matrix,
    predict_sentiment,
    create_sample_data
)


def main():
    """主函数"""
    print("="*60)
    print("LSTM情感分析系统")
    print("="*60)
    
    # 设置随机种子
    set_random_seed(42)
    
    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    
    # 初始化日志
    logger = setup_logger('LSTM_Sentiment', 'outputs/logs/lstm_sentiment.log')
    logger.info("开始LSTM情感分析流程")
    
    # 1. 数据准备
    print("\n1. 数据准备...")
    data_processor = SentimentDataProcessor(
        data_dir='dataset',
        max_vocab_size=10000,
        max_length=128,
        min_freq=2
    )
    
    # 创建示例数据（如果不存在真实数据）
    if not os.path.exists('dataset'):
        os.makedirs('dataset', exist_ok=True)
    
    if not os.path.exists('dataset/sentiment_data.csv'):
        print("创建示例数据...")
        create_sample_data('dataset/sentiment_data.csv')
    
    # 准备训练、验证和测试数据
    train_loader, val_loader, test_loader = data_processor.prepare_data(
        test_size=0.2, val_size=0.1, random_state=42
    )
    
    # 保存词汇表和标签编码器
    os.makedirs('outputs', exist_ok=True)
    data_processor.save_vocab('outputs/vocab.pkl')
    data_processor.save_label_encoder('outputs/label_encoder.pkl')
    
    print(f"词汇表大小: {data_processor.get_vocab_size()}")
    print(f"类别数量: {data_processor.get_num_classes()}")
    print(f"标签: {data_processor.label_encoder.classes_}")
    
    # 2. 模型构建
    print("\n2. 模型构建...")
    model = LSTMSentimentModel(
        vocab_size=data_processor.get_vocab_size(),
        embedding_dim=128,
        hidden_dim=256,
        num_layers=2,
        num_classes=data_processor.get_num_classes(),
        dropout=0.3
    )
    
    # 初始化权重
    model.init_weights()
    
    # 计算模型参数数量
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"模型总参数数: {total_params:,}")
    print(f"可训练参数数: {trainable_params:,}")
    
    # 3. 模型训练
    print("\n3. 模型训练...")
    trainer = SentimentTrainer(model, device, logger)
    
    trainer.train(
        train_loader=train_loader,
        val_loader=val_loader,
        num_epochs=20,
        learning_rate=0.001,
        output_dir='outputs',
        patience=5
    )
    
    # 4. 模型评估
    print("\n4. 模型评估...")
    best_model_path = 'outputs/model_best.pt'
    test_results = trainer.test(test_loader, best_model_path)
    
    print(f"测试准确率: {test_results['accuracy']:.4f}")
    print(f"测试精确率: {test_results['precision']:.4f}")
    print(f"测试召回率: {test_results['recall']:.4f}")
    print(f"测试F1分数: {test_results['f1']:.4f}")
    
    # 5. 可视化结果
    print("\n5. 生成可视化结果...")
    
    # 获取测试集预测结果用于可视化
    model.eval()
    all_predictions = []
    all_labels = []
    all_texts = []
    
    with torch.no_grad():
        for batch in test_loader:
            input_ids = batch['input_ids'].to(device)
            labels = batch['labels'].to(device)
            lengths = batch['length'].to(device)
            
            outputs = model(input_ids, lengths)
            predicted = torch.argmax(outputs, dim=1)
            
            all_predictions.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    # 绘制混淆矩阵
    class_names = data_processor.label_encoder.classes_
    plot_confusion_matrix(
        all_labels, 
        all_predictions, 
        class_names, 
        save_path='outputs/confusion_matrix.png'
    )
    
    # 6. 交互式预测演示
    print("\n6. 交互式预测演示...")
    test_texts = [
        "这部电影真的很棒，强烈推荐！",
        "服务态度很差，很不满意。",
        "这个产品质量一般般。",
        "今天心情很好，阳光明媚。",
        "工作压力太大了，很焦虑。"
    ]
    
    print("测试文本预测结果:")
    print("-" * 40)
    
    for text in test_texts:
        predicted_label = predict_sentiment(
            model, text, data_processor.vocab, 
            data_processor.label_encoder, device
        )
        print(f"文本: {text}")
        print(f"预测情感: {predicted_label}")
        print("-" * 40)
    
    # 7. 保存最终结果
    print("\n7. 保存结果...")
    
    # 保存测试结果
    import json
    with open('outputs/test_results.json', 'w', encoding='utf-8') as f:
        json.dump(test_results, f, ensure_ascii=False, indent=2)
    
    # 保存模型信息
    model_info = {
        'vocab_size': data_processor.get_vocab_size(),
        'num_classes': data_processor.get_num_classes(),
        'class_names': class_names.tolist(),
        'model_config': {
            'embedding_dim': 128,
            'hidden_dim': 256,
            'num_layers': 2,
            'dropout': 0.3
        },
        'performance': {
            'accuracy': test_results['accuracy'],
            'precision': test_results['precision'],
            'recall': test_results['recall'],
            'f1': test_results['f1']
        }
    }
    
    with open('outputs/model_info.json', 'w', encoding='utf-8') as f:
        json.dump(model_info, f, ensure_ascii=False, indent=2)
    
    print("\n" + "="*60)
    print("LSTM情感分析流程完成！")
    print("="*60)
    print("输出文件:")
    print("- 最佳模型: outputs/model_best.pt")
    print("- 词汇表: outputs/vocab.pkl")
    print("- 标签编码器: outputs/label_encoder.pkl")
    print("- 测试结果: outputs/test_results.json")
    print("- 模型信息: outputs/model_info.json")
    print("- 混淆矩阵: outputs/confusion_matrix.png")
    print("- 训练日志: outputs/logs/")
    print("="*60)


def demo_prediction():
    """演示预测功能"""
    print("\n演示预测功能...")
    
    # 检查模型文件是否存在
    if not os.path.exists('outputs/model_best.pt'):
        print("错误: 未找到训练好的模型文件，请先运行完整的训练流程")
        return
    
    # 加载模型和相关文件
    import pickle
    from src.utils import load_model
    
    # 加载词汇表和标签编码器
    with open('outputs/vocab.pkl', 'rb') as f:
        vocab = pickle.load(f)
    
    with open('outputs/label_encoder.pkl', 'rb') as f:
        label_encoder = pickle.load(f)
    
    # 加载模型信息
    with open('outputs/model_info.json', 'r', encoding='utf-8') as f:
        model_info = json.load(f)
    
    # 重建模型
    model = LSTMSentimentModel(
        vocab_size=model_info['vocab_size'],
        embedding_dim=model_info['model_config']['embedding_dim'],
        hidden_dim=model_info['model_config']['hidden_dim'],
        num_layers=model_info['model_config']['num_layers'],
        num_classes=model_info['num_classes'],
        dropout=model_info['model_config']['dropout']
    )
    
    # 加载训练好的权重
    load_model(model, 'outputs/model_best.pt')
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    
    # 交互式预测
    print("输入文本进行情感分析（输入'quit'退出）:")
    while True:
        text = input("\n请输入文本: ").strip()
        if text.lower() == 'quit':
            break
        
        if text:
            predicted_label = predict_sentiment(
                model, text, vocab, label_encoder, device
            )
            print(f"预测情感: {predicted_label}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='LSTM情感分析系统')
    parser.add_argument('--mode', type=str, default='train', 
                       choices=['train', 'demo'],
                       help='运行模式: train=完整训练流程, demo=预测演示')
    
    args = parser.parse_args()
    
    if args.mode == 'train':
        main()
    elif args.mode == 'demo':
        demo_prediction()