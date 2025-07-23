# LSTM情感分析项目

基于LSTM（Long Short-Term Memory）神经网络的中文文本情感分析系统。

## 项目简介

本项目使用LSTM循环神经网络实现中文文本的情感分析，能够对输入的文本进行情感极性分类（正面/负面/中性）。项目包含完整的数据处理、模型训练、评估和部署流程。

## 项目结构

```
├── 1.1循环神经网络介绍.ipynb     # RNN基础知识介绍
├── 1.2lstm.ipynb               # LSTM原理和实现
├── LSTM_情感分析.py            # 完整的LSTM情感分析脚本
├── README.md                   # 项目说明文档
├── dataset/                    # 数据集目录
├── outputs/                    # 输出文件
│   ├── logs/                  # 训练日志
│   └── model_best.onnx        # 导出的ONNX模型
├── requirements.txt            # Python依赖包
├── service/                   # 模型服务
│   ├── Untitled.ipynb         # 实验notebook
│   ├── convert_to_onnx.py     # 模型转换ONNX格式
│   └── onnx_server.py         # ONNX模型推理服务
└── src/                       # 源代码目录
    ├── __init__.py            # 包初始化文件
    ├── app.py                 # Flask Web应用
    ├── dataset.py             # 数据处理模块
    ├── model.py               # LSTM模型定义
    ├── train.py               # 训练脚本
    └── utils.py               # 工具函数
```

## 安装依赖

```bash
pip install -r requirements.txt
```

## 使用方法

### 1. 数据准备
将情感分析数据集放入 `dataset/` 目录下，支持CSV格式，包含文本和标签列。

### 2. 训练模型
```bash
python src/train.py
```

### 3. 模型转换
```bash
python service/convert_to_onnx.py
```

### 4. 启动推理服务
```bash
python service/onnx_server.py
```

### 5. Web应用
```bash
python src/app.py
```

## 模型特点

- **LSTM网络**: 长短期记忆网络，能够有效处理序列数据
- **中文支持**: 使用jieba分词，支持中文文本处理
- **多分类**: 支持正面、负面、中性等多种情感分类
- **部署友好**: 提供ONNX模型导出和推理服务

## 技术栈

- **深度学习**: PyTorch
- **中文处理**: jieba分词
- **数据处理**: Pandas + NumPy
- **模型部署**: ONNX Runtime + Flask
- **可视化**: Matplotlib + Seaborn

## 快速开始

1. 安装依赖：`pip install -r requirements.txt`
2. 查看教程：打开 `1.1循环神经网络介绍.ipynb`
3. 学习LSTM：打开 `1.2lstm.ipynb`
4. 运行完整流程：`python LSTM_情感分析.py`