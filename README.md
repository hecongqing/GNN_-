# LightGCN推荐系统

基于LightGCN（Light Graph Convolutional Network）实现的推荐系统，支持模型训练、ONNX转换和HTTP服务部署。

## 项目简介

本项目使用图神经网络中的LightGCN模型来解决推荐系统问题。LightGCN是一种简化的图卷积网络，专门针对推荐系统进行了优化。项目支持完整的训练、转换和部署流程。

## 项目结构

```
├── README.md
├── dataset                    # 数据集存放目录
├── outputs                    # 输出文件
├── requirements.txt
├── service
│   ├── convert_to_onnx.py     # PyTorch模型转ONNX
│   └── onnx_server.py         # ONNX推理服务器
└── src
    ├── __init__.py
    ├── app.py                 # 主应用程序
    ├── dataset.py             # 数据处理
    ├── model.py               # LightGCN模型
    ├── train.py               # 训练模块
    └── utils.py               # 工具函数
```

## 安装依赖

```bash
pip install -r requirements.txt
```

## 使用方法

### 1. 安装依赖
```bash
pip install -r requirements.txt
```

### 2. 数据准备
将数据文件放入 `dataset/` 目录。如果没有真实数据，系统会自动生成示例数据用于测试。

### 3. 训练模型
```bash
python -m src.app --mode train --config config.json
```

### 4. 模型推理
```bash
python -m src.app --mode inference --user-id 0 --top-k 10
```

### 5. 转换为ONNX格式
```bash
python service/convert_to_onnx.py --model-path outputs/models/best_model.pt --save-metadata
```

### 6. 启动ONNX推理服务
```bash
python service/onnx_server.py --model-path outputs/models/best_model.onnx --host 0.0.0.0 --port 8080
```

### 7. API使用示例

#### 单用户推荐
```bash
curl -X POST "http://localhost:8080/recommend" \
  -H "Content-Type: application/json" \
  -d '{"user_id": 0, "top_k": 10}'
```

#### 批量推荐
```bash
curl -X POST "http://localhost:8080/recommend/batch" \
  -H "Content-Type: application/json" \
  -d '{"user_ids": [0, 1, 2], "top_k": 10}'
```

#### 健康检查
```bash
curl "http://localhost:8080/health"
```

#### 模型信息
```bash
curl "http://localhost:8080/info"
```

## 模型特点

- **LightGCN**: 简化的图卷积网络，去除了特征变换和非线性激活
- **高效性**: 相比传统GCN，计算更加高效
- **适用性**: 专门针对推荐系统的协同过滤场景设计
- **部署友好**: 支持ONNX格式，便于跨平台部署

## 快速开始

```bash
# 1. 安装依赖
pip install -r requirements.txt

# 2. 训练模型（使用示例数据）
python -m src.app --mode train

# 3. 转换为ONNX
python service/convert_to_onnx.py --model-path outputs/models/best_model.pt --save-metadata

# 4. 启动推理服务
python service/onnx_server.py --model-path outputs/models/best_model.onnx
```

## 项目特色

- ✅ **模块化设计**: 清晰的项目结构，易于维护和扩展
- ✅ **生产就绪**: 支持ONNX格式，便于部署
- ✅ **API服务**: 提供完整的HTTP API接口
- ✅ **轻量级**: 移除了不必要的可视化依赖，专注核心功能

## 技术栈

- **深度学习**: PyTorch + PyTorch Geometric
- **数据处理**: Pandas + NumPy
- **模型部署**: ONNX + ONNX Runtime
- **Web服务**: Flask
- **模型**: LightGCN (Light Graph Convolutional Networks)