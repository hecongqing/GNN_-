#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
将训练好的LSTM模型转换为ONNX格式
用于高效推理和跨平台部署
"""

import os
import sys
import torch
import onnx
import onnxruntime as ort
import numpy as np
import json
import pickle
from typing import Dict, Any

# 添加项目根目录到Python路径
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from src.model import LSTMSentimentModel
from src.utils import load_model, tokenize_chinese


def convert_pytorch_to_onnx(model_path: str, onnx_path: str, model_info: Dict[str, Any],
                           vocab: Dict[str, int], sample_text: str = "这是一个测试文本"):
    """
    将PyTorch模型转换为ONNX格式
    
    Args:
        model_path: PyTorch模型路径
        onnx_path: 输出ONNX模型路径
        model_info: 模型配置信息
        vocab: 词汇表
        sample_text: 用于转换的示例文本
    """
    print("开始转换PyTorch模型到ONNX格式...")
    
    # 1. 创建模型实例
    model = LSTMSentimentModel(
        vocab_size=model_info['vocab_size'],
        embedding_dim=model_info['model_config']['embedding_dim'],
        hidden_dim=model_info['model_config']['hidden_dim'],
        num_layers=model_info['model_config']['num_layers'],
        num_classes=model_info['num_classes'],
        dropout=model_info['model_config']['dropout']
    )
    
    # 2. 加载训练好的权重
    load_model(model, model_path)
    model.eval()
    
    # 3. 准备示例输入
    max_length = 128
    tokens = tokenize_chinese(sample_text)
    token_ids = [vocab.get(token, vocab.get('<UNK>', 1)) for token in tokens]
    
    if len(token_ids) > max_length:
        token_ids = token_ids[:max_length]
        length = max_length
    else:
        length = len(token_ids)
        token_ids.extend([0] * (max_length - len(token_ids)))
    
    # 创建输入张量
    input_ids = torch.tensor([token_ids], dtype=torch.long)
    lengths = torch.tensor([length], dtype=torch.long)
    
    # 4. 导出ONNX模型
    print("导出ONNX模型...")
    
    torch.onnx.export(
        model,                          # 模型
        (input_ids, lengths),          # 模型输入 (示例)
        onnx_path,                     # 保存路径
        export_params=True,            # 导出模型参数
        opset_version=11,              # ONNX算子集版本
        do_constant_folding=True,      # 常量折叠优化
        input_names=['input_ids', 'lengths'],  # 输入名称
        output_names=['output'],       # 输出名称
        dynamic_axes={                 # 动态轴
            'input_ids': {0: 'batch_size'},
            'lengths': {0: 'batch_size'},
            'output': {0: 'batch_size'}
        }
    )
    
    print(f"ONNX模型已保存到: {onnx_path}")
    
    # 5. 验证ONNX模型
    print("验证ONNX模型...")
    
    # 加载ONNX模型
    onnx_model = onnx.load(onnx_path)
    onnx.checker.check_model(onnx_model)
    print("ONNX模型验证通过")
    
    # 6. 测试推理
    print("测试ONNX模型推理...")
    
    # 使用PyTorch模型推理
    with torch.no_grad():
        pytorch_output = model(input_ids, lengths)
        pytorch_probs = torch.softmax(pytorch_output, dim=1).numpy()
    
    # 使用ONNX Runtime推理
    ort_session = ort.InferenceSession(onnx_path)
    ort_inputs = {
        'input_ids': input_ids.numpy(),
        'lengths': lengths.numpy()
    }
    ort_output = ort_session.run(None, ort_inputs)[0]
    onnx_probs = np.exp(ort_output) / np.sum(np.exp(ort_output), axis=1, keepdims=True)  # softmax
    
    # 比较输出
    diff = np.abs(pytorch_probs - onnx_probs).max()
    print(f"PyTorch vs ONNX最大差异: {diff:.6f}")
    
    if diff < 1e-5:
        print("✓ ONNX模型转换成功，输出一致性良好")
    else:
        print("⚠ ONNX模型输出与PyTorch模型存在差异")
    
    return True


def create_onnx_metadata(onnx_path: str, model_info: Dict[str, Any], 
                        vocab_path: str, label_encoder_path: str):
    """
    创建ONNX模型的元数据文件
    
    Args:
        onnx_path: ONNX模型路径
        model_info: 模型信息
        vocab_path: 词汇表路径
        label_encoder_path: 标签编码器路径
    """
    metadata = {
        'model_type': 'LSTM_Sentiment_Analysis',
        'onnx_model_path': onnx_path,
        'vocab_path': vocab_path,
        'label_encoder_path': label_encoder_path,
        'model_config': model_info['model_config'],
        'vocab_size': model_info['vocab_size'],
        'num_classes': model_info['num_classes'],
        'class_names': model_info['class_names'],
        'max_length': 128,
        'input_names': ['input_ids', 'lengths'],
        'output_names': ['output'],
        'preprocessing': {
            'tokenizer': 'jieba',
            'padding_token_id': 0,
            'unknown_token_id': 1
        },
        'performance': model_info.get('performance', {}),
        'version': '1.0.0',
        'description': 'LSTM模型用于中文文本情感分析，支持正面、负面、中性三分类'
    }
    
    metadata_path = os.path.splitext(onnx_path)[0] + '_metadata.json'
    
    with open(metadata_path, 'w', encoding='utf-8') as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)
    
    print(f"元数据已保存到: {metadata_path}")
    return metadata_path


def test_onnx_inference(onnx_path: str, vocab: Dict[str, int], label_encoder,
                       test_texts: list):
    """
    测试ONNX模型推理
    
    Args:
        onnx_path: ONNX模型路径
        vocab: 词汇表
        label_encoder: 标签编码器
        test_texts: 测试文本列表
    """
    print("\n测试ONNX模型推理...")
    
    # 创建ONNX Runtime会话
    ort_session = ort.InferenceSession(onnx_path)
    
    max_length = 128
    
    for text in test_texts:
        # 预处理文本
        tokens = tokenize_chinese(text)
        token_ids = [vocab.get(token, vocab.get('<UNK>', 1)) for token in tokens]
        
        if len(token_ids) > max_length:
            token_ids = token_ids[:max_length]
            length = max_length
        else:
            length = len(token_ids)
            token_ids.extend([0] * (max_length - len(token_ids)))
        
        # 准备输入
        input_ids = np.array([token_ids], dtype=np.int64)
        lengths = np.array([length], dtype=np.int64)
        
        ort_inputs = {
            'input_ids': input_ids,
            'lengths': lengths
        }
        
        # 推理
        ort_output = ort_session.run(None, ort_inputs)[0]
        
        # 后处理
        probabilities = np.exp(ort_output) / np.sum(np.exp(ort_output), axis=1, keepdims=True)
        predicted_class = np.argmax(probabilities, axis=1)[0]
        confidence = probabilities[0][predicted_class]
        
        predicted_label = label_encoder.inverse_transform([predicted_class])[0]
        
        print(f"文本: {text}")
        print(f"预测: {predicted_label} (置信度: {confidence:.4f})")
        print("-" * 50)


def main():
    """主函数"""
    print("="*60)
    print("PyTorch模型转换为ONNX格式")
    print("="*60)
    
    # 检查必要文件
    required_files = [
        'outputs/model_best.pt',
        'outputs/vocab.pkl',
        'outputs/label_encoder.pkl',
        'outputs/model_info.json'
    ]
    
    missing_files = []
    for file_path in required_files:
        if not os.path.exists(file_path):
            missing_files.append(file_path)
    
    if missing_files:
        print("错误: 缺少以下文件:")
        for file_path in missing_files:
            print(f"  - {file_path}")
        print("\n请先运行 'python LSTM_情感分析.py' 训练模型")
        return
    
    # 加载模型信息
    with open('outputs/model_info.json', 'r', encoding='utf-8') as f:
        model_info = json.load(f)
    
    # 加载词汇表
    with open('outputs/vocab.pkl', 'rb') as f:
        vocab = pickle.load(f)
    
    # 加载标签编码器
    with open('outputs/label_encoder.pkl', 'rb') as f:
        label_encoder = pickle.load(f)
    
    # 设置输出路径
    onnx_path = 'outputs/model_best.onnx'
    
    # 转换模型
    success = convert_pytorch_to_onnx(
        model_path='outputs/model_best.pt',
        onnx_path=onnx_path,
        model_info=model_info,
        vocab=vocab
    )
    
    if success:
        # 创建元数据
        metadata_path = create_onnx_metadata(
            onnx_path=onnx_path,
            model_info=model_info,
            vocab_path='outputs/vocab.pkl',
            label_encoder_path='outputs/label_encoder.pkl'
        )
        
        # 测试推理
        test_texts = [
            "这部电影真的很棒，强烈推荐！",
            "服务态度很差，很不满意。",
            "这个产品质量一般般。"
        ]
        
        test_onnx_inference(onnx_path, vocab, label_encoder, test_texts)
        
        print("\n" + "="*60)
        print("转换完成！")
        print("="*60)
        print("输出文件:")
        print(f"- ONNX模型: {onnx_path}")
        print(f"- 元数据: {metadata_path}")
        print("="*60)


if __name__ == "__main__":
    main()