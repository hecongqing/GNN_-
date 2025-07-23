# LightGCN推荐系统

基于LightGCN（Light Graph Convolutional Network）实现的推荐系统，支持模型训练、ONNX转换、HTTP服务部署和可视化界面。

## 项目简介

本项目使用图神经网络中的LightGCN模型来解决推荐系统问题。LightGCN是一种简化的图卷积网络，专门针对推荐系统进行了优化。项目支持完整的训练、转换、部署和可视化流程。

## 项目结构

```
├── README.md
├── dataset                    # 数据集存放目录
├── outputs                    # 输出文件
├── requirements.txt           # 项目依赖
├── run_app.py                # 快速启动脚本
├── config.json               # 配置文件
├── service
│   ├── convert_to_onnx.py     # PyTorch模型转ONNX
│   └── onnx_server.py         # ONNX推理服务器
└── src
    ├── __init__.py
    ├── app.py                 # 主应用程序（包含Web界面）
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

### 🚀 快速开始（推荐）

运行可视化Web界面：

```bash
python run_app.py
```

然后在浏览器中访问：http://localhost:8501

### 📋 功能特性

- **🎯 可视化界面**: 基于Streamlit的现代化Web界面
- **📈 实时训练**: 在界面中配置参数并监控训练过程
- **🔍 推荐生成**: 交互式推荐生成和结果可视化
- **📊 数据分析**: 直观的数据分布和统计分析
- **🏗️ 模块化设计**: 清晰的项目结构，易于维护和扩展

### 🖥️ 命令行使用

#### 1. 训练模型
```bash
python -m src.app --mode train --config config.json
```

#### 2. 模型推理
```bash
python -m src.app --mode inference --user-id 0 --top-k 10
```

#### 3. 转换为ONNX格式
```bash
python service/convert_to_onnx.py --model-path outputs/models/best_model.pt --save-metadata
```

#### 4. 启动ONNX推理服务
```bash
python service/onnx_server.py --model-path outputs/models/best_model.onnx --host 0.0.0.0 --port 8080
```

### 📡 API使用示例

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

## 🎨 Web界面功能

### 🏠 首页
- 系统状态概览
- 模型和数据状态检查
- 功能导航

### 🎯 推荐
- 交互式用户推荐
- 推荐结果可视化
- 参数调节界面

### 📈 训练
- 可视化训练配置
- 实时训练进度监控
- 训练指标展示
- 训练日志查看

### 📊 分析
- 数据分布分析
- 用户行为统计
- 物品流行度分析
- 系统稀疏性分析

## 模型特点

- **LightGCN**: 简化的图卷积网络，去除了特征变换和非线性激活
- **高效性**: 相比传统GCN，计算更加高效
- **适用性**: 专门针对推荐系统的协同过滤场景设计
- **部署友好**: 支持ONNX格式，便于跨平台部署

## 技术栈

- **深度学习**: PyTorch + PyTorch Geometric
- **数据处理**: Pandas + NumPy
- **模型部署**: ONNX + ONNX Runtime
- **Web服务**: Flask + Streamlit
- **数据可视化**: Plotly + Matplotlib + Seaborn
- **模型**: LightGCN (Light Graph Convolutional Networks)

## 项目优化

相比原版本，本项目进行了以下优化：

### ✅ 新增功能
- 🎨 **现代化Web界面**: 基于Streamlit的交互式界面
- 📊 **数据可视化**: 丰富的图表和分析功能  
- 🚀 **一键启动**: 简化的启动流程
- 📈 **实时训练监控**: 可视化训练过程

### 🔧 代码优化
- 🧹 **代码精简**: 移除冗余复杂逻辑
- 🏗️ **结构优化**: 更清晰的模块划分
- ⚡ **性能提升**: 优化训练算法和批处理
- 📚 **文档完善**: 详细的使用说明和注释

### 🛠️ 易用性提升
- 🎯 **零配置启动**: 自动生成示例数据
- 🔄 **智能恢复**: 自动加载预处理数据
- 📱 **响应式界面**: 适配不同屏幕尺寸
- 🎛️ **参数可视化**: 直观的配置调节

## 许可证

本项目采用MIT许可证，详见LICENSE文件。