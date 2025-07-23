# LightGCN推荐系统

基于LightGCN（Light Graph Convolutional Network）实现的推荐系统，支持模型训练、ONNX转换、HTTP服务部署和可视化界面。

## 项目简介

本项目使用图神经网络中的LightGCN模型来解决推荐系统问题。LightGCN是一种简化的图卷积网络，专门针对推荐系统进行了优化。项目采用分离式架构，训练和推理部署分离，便于生产环境使用。

## 项目结构

```
├── README.md
├── QUICKSTART.md             # 快速开始指南
├── dataset/                  # 数据集存放目录
├── outputs/                  # 输出文件目录
├── requirements.txt          # 项目依赖
├── run_app.py               # 可视化界面启动脚本
├── test_system.py           # 系统测试脚本
├── config.json              # 训练配置文件
├── service/
│   ├── convert_to_onnx.py   # PyTorch模型转ONNX
│   └── onnx_server.py       # ONNX推理服务器
└── src/
    ├── __init__.py
    ├── app.py               # 可视化界面（调用ONNX API）
    ├── dataset.py           # 数据处理模块
    ├── model.py             # LightGCN模型定义
    ├── train.py             # 模型训练模块
    └── utils.py             # 工具函数
```

## 核心架构

### 🎯 分离式设计
- **训练模块** (`src/train.py`): 负责模型训练和保存
- **推理服务** (`service/onnx_server.py`): 基于ONNX的高性能推理API
- **可视化界面** (`src/app.py`): 调用ONNX API并进行结果可视化

### 🔄 工作流程
1. **训练阶段**: 使用 `train.py` 训练LightGCN模型
2. **转换阶段**: 将PyTorch模型转换为ONNX格式
3. **部署阶段**: 启动ONNX推理服务
4. **使用阶段**: 通过可视化界面调用API获取推荐

## 安装依赖

```bash
pip install -r requirements.txt
```

## 使用方法

### 🚀 快速开始

#### 1. 训练模型
```bash
# 使用默认配置训练
python -m src.train

# 自定义参数训练
python -m src.train --epochs 30 --batch-size 256 --learning-rate 0.001
```

#### 2. 转换为ONNX
```bash
python service/convert_to_onnx.py --model-path outputs/models/best_model.pt --save-metadata
```

#### 3. 启动推理服务
```bash
python service/onnx_server.py --model-path outputs/models/best_model.onnx --host 0.0.0.0 --port 8080
```

#### 4. 启动可视化界面
```bash
python run_app.py
```

然后在浏览器中访问：http://localhost:8501

### 🖥️ 命令行详细使用

#### 模型训练选项
```bash
python -m src.train --help

# 常用选项:
--config CONFIG          # 配置文件路径 (默认: config.json)
--epochs EPOCHS          # 训练轮数
--batch-size BATCH_SIZE  # 批次大小
--learning-rate LR       # 学习率
--embedding-dim DIM      # 嵌入维度
```

#### ONNX转换选项
```bash
python service/convert_to_onnx.py --help

# 常用选项:
--model-path PATH        # PyTorch模型路径
--output-path PATH       # ONNX输出路径
--save-metadata         # 保存模型元数据
```

#### 推理服务选项
```bash
python service/onnx_server.py --help

# 常用选项:
--model-path PATH        # ONNX模型路径
--host HOST             # 服务器地址 (默认: 0.0.0.0)
--port PORT             # 服务器端口 (默认: 8080)
--debug                 # 调试模式
```

## 🎨 可视化界面功能

### 🎯 推荐服务
- **单用户推荐**: 为指定用户生成个性化推荐
- **批量推荐**: 同时为多个用户生成推荐
- **结果可视化**: 推荐分数分布图和热力图
- **实时调用**: 直接调用ONNX API服务

### 📊 数据分析
- **用户行为分析**: 活跃度分布和统计
- **物品流行度**: 热门物品排行
- **时间趋势**: 24小时活跃度变化
- **系统指标**: 用户数、物品数、交互数等

### ⚙️ API配置
- **服务地址配置**: 设置ONNX API服务地址
- **连接测试**: 实时检测API服务状态
- **API文档**: 查看接口使用说明

## 📡 API接口

### 健康检查
```bash
curl http://localhost:8080/health
```

### 模型信息
```bash
curl http://localhost:8080/info
```

### 单用户推荐
```bash
curl -X POST "http://localhost:8080/recommend" \
  -H "Content-Type: application/json" \
  -d '{"user_id": 0, "top_k": 10}'
```

### 批量推荐
```bash
curl -X POST "http://localhost:8080/recommend/batch" \
  -H "Content-Type: application/json" \
  -d '{"user_ids": [0, 1, 2], "top_k": 10}'
```

## 模型特点

- **LightGCN**: 简化的图卷积网络，专为推荐系统优化
- **高效性**: 去除特征变换和非线性激活，计算高效
- **可扩展性**: 支持大规模用户和物品
- **部署友好**: ONNX格式，跨平台高性能推理

## 技术栈

- **深度学习**: PyTorch + PyTorch Geometric
- **数据处理**: Pandas + NumPy
- **模型推理**: ONNX Runtime
- **Web服务**: Flask (API) + Streamlit (界面)
- **数据可视化**: Plotly + Matplotlib
- **HTTP客户端**: Requests

## 项目特色

### ✅ 分离式架构
- 🎯 **职责分离**: 训练、推理、可视化各司其职
- 🚀 **独立部署**: 各模块可独立部署和扩展
- 🔄 **松耦合**: 通过API接口连接，易于维护

### 🏗️ 生产就绪
- ⚡ **高性能推理**: 基于ONNX Runtime的优化推理
- 🔧 **配置灵活**: 支持命令行参数覆盖配置文件
- 📊 **监控友好**: 完整的健康检查和状态监控

### 🎨 用户体验
- 🖥️ **现代界面**: 基于Streamlit的响应式Web界面
- 📈 **实时可视化**: 丰富的图表和交互式分析
- 🧪 **系统测试**: 完整的功能测试脚本

## 开发和部署

### 🧪 测试系统
```bash
python test_system.py
```

### 🔧 开发模式
```bash
# 训练模型（快速测试）
python -m src.train --epochs 10

# 启动调试模式的API服务
python service/onnx_server.py --model-path outputs/models/best_model.onnx --debug

# 启动界面（开发模式）
streamlit run src/app.py --server.runOnSave=true
```

### 🚀 生产部署
```bash
# 1. 训练最终模型
python -m src.train --epochs 100 --batch-size 1024

# 2. 转换ONNX
python service/convert_to_onnx.py --model-path outputs/models/best_model.pt

# 3. 启动生产服务
python service/onnx_server.py --model-path outputs/models/best_model.onnx --host 0.0.0.0 --port 8080

# 4. 启动界面（生产模式）
streamlit run src/app.py --server.port 8501 --server.address 0.0.0.0
```

## 许可证

本项目采用MIT许可证，详见LICENSE文件。