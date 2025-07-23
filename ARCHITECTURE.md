# LightGCN推荐系统架构说明

## 🎯 核心设计理念

本项目采用**分离式架构**，将训练、推理、可视化三个核心功能完全分离，每个模块独立运行，通过标准化接口通信。

## 📁 项目结构

```
├── README.md                 # 详细文档
├── QUICKSTART.md            # 快速开始指南  
├── ARCHITECTURE.md          # 架构说明（本文件）
├── dataset/                 # 数据集目录
├── outputs/                 # 输出目录
├── requirements.txt         # Python依赖
├── run_app.py              # 界面启动脚本
├── test_system.py          # 系统测试脚本
├── config.json             # 训练配置
├── service/
│   ├── convert_to_onnx.py  # 模型转换工具
│   └── onnx_server.py      # ONNX推理API服务
└── src/
    ├── __init__.py
    ├── app.py              # 可视化界面（仅调用API）
    ├── dataset.py          # 数据处理
    ├── model.py            # LightGCN模型定义
    ├── train.py            # 模型训练（独立运行）
    └── utils.py            # 工具函数
```

## 🔄 工作流程

### 1. 训练阶段
```bash
python -m src.train
```
- **职责**: 数据加载、模型训练、模型保存
- **输入**: 配置文件、原始数据
- **输出**: PyTorch模型文件 (`.pt`)
- **特点**: 完全独立，不依赖其他模块

### 2. 转换阶段  
```bash
python service/convert_to_onnx.py --model-path outputs/models/best_model.pt
```
- **职责**: PyTorch模型转ONNX格式
- **输入**: PyTorch模型文件
- **输出**: ONNX模型文件 (`.onnx`) + 元数据
- **特点**: 一次性转换，优化推理性能

### 3. 推理服务
```bash
python service/onnx_server.py --model-path outputs/models/best_model.onnx
```
- **职责**: 提供HTTP API推理服务
- **协议**: RESTful API (JSON)
- **端口**: 8080 (可配置)
- **特点**: 高性能、可扩展、生产就绪

### 4. 可视化界面
```bash
python run_app.py  # 或 streamlit run src/app.py
```
- **职责**: 用户界面、API调用、结果可视化
- **依赖**: ONNX推理服务API
- **端口**: 8501 (可配置)
- **特点**: 响应式Web界面

## 🎨 模块详解

### src/train.py - 训练模块
**设计原则**: 单一职责，专注训练
- ✅ 独立运行，不依赖其他服务
- ✅ 支持命令行参数覆盖配置
- ✅ 完整的训练流程和日志
- ✅ 自动数据预处理和模型保存

**核心功能**:
- 数据加载和预处理
- LightGCN模型训练  
- 模型评估和验证
- 最佳模型保存
- 推荐结果生成示例

### src/app.py - 可视化界面
**设计原则**: 仅负责界面展示和API调用
- ✅ 不包含任何训练逻辑
- ✅ 通过HTTP API调用推理服务
- ✅ 丰富的数据可视化功能
- ✅ 实时API状态监控

**核心功能**:
- API服务状态检查
- 单用户/批量推荐界面
- 推荐结果可视化
- 数据分析仪表板
- API配置和测试

### service/onnx_server.py - 推理服务
**设计原则**: 高性能推理API
- ✅ 基于ONNX Runtime优化推理
- ✅ RESTful API设计
- ✅ 支持单用户和批量推荐
- ✅ 完整的健康检查和监控

**API端点**:
- `GET /health` - 健康检查
- `GET /info` - 模型信息
- `POST /recommend` - 单用户推荐
- `POST /recommend/batch` - 批量推荐

## 🔌 接口规范

### API请求格式
```json
// 单用户推荐
{
  "user_id": 0,
  "top_k": 10
}

// 批量推荐  
{
  "user_ids": [0, 1, 2],
  "top_k": 10
}
```

### API响应格式
```json
// 单用户推荐响应
{
  "user_id": 0,
  "recommendations": [1, 5, 3, 8, 2],
  "top_k": 5
}

// 批量推荐响应
{
  "recommendations": {
    "0": [1, 5, 3],
    "1": [2, 7, 4], 
    "2": [6, 1, 9]
  },
  "top_k": 3
}
```

## 🚀 部署方案

### 开发环境
```bash
# 终端1: 启动API服务
python service/onnx_server.py --model-path outputs/models/best_model.onnx --debug

# 终端2: 启动界面
python run_app.py
```

### 生产环境
```bash
# 1. 训练生产模型
python -m src.train --epochs 100 --batch-size 1024

# 2. 转换ONNX
python service/convert_to_onnx.py --model-path outputs/models/best_model.pt

# 3. 部署API服务（使用进程管理器如supervisor/systemd）
python service/onnx_server.py --model-path outputs/models/best_model.onnx --host 0.0.0.0 --port 8080

# 4. 部署界面（使用反向代理如nginx）
streamlit run src/app.py --server.port 8501 --server.address 0.0.0.0
```

## 💡 设计优势

### ✅ 职责分离
- **训练**: 专注模型优化，不涉及服务逻辑
- **推理**: 专注高性能API服务
- **界面**: 专注用户体验和数据展示

### ✅ 独立部署
- 各模块可独立部署和扩展
- API服务可水平扩展
- 界面可独立更新

### ✅ 技术选型
- **训练**: PyTorch (灵活性)
- **推理**: ONNX Runtime (性能)  
- **界面**: Streamlit (快速开发)
- **API**: Flask (轻量级)

### ✅ 生产友好
- 标准化API接口
- 完整的健康检查
- 可配置参数
- 详细的日志记录

## 🎯 使用建议

1. **开发阶段**: 使用快速训练参数测试整个流程
2. **调试阶段**: 使用debug模式启动API服务
3. **生产阶段**: 使用完整参数训练，部署到生产环境
4. **维护阶段**: 通过界面监控API状态和推荐效果