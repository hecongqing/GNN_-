# LightGCN推荐系统 - 阿里移动推荐算法挑战赛

基于阿里移动推荐算法挑战赛数据集，使用LightGCN（Light Graph Convolutional Network）实现的推荐系统。

## 项目简介

本项目使用图神经网络中的LightGCN模型来解决移动电商平台的商品推荐问题。LightGCN是一种简化的图卷积网络，专门针对推荐系统进行了优化。

## 数据集

数据来源：阿里移动推荐算法挑战赛
- 时间范围：2014.11.18 ~ 2014.12.18 (训练数据)
- 预测目标：2014.12.19 用户购买行为
- 数据文件：
  - `tianchi_mobile_recommend_train_user.zip`: 用户行为数据
  - `tianchi_mobile_recommend_train_item.csv`: 商品子集数据

## 项目结构

```
.
├── README.md
├── requirements.txt
├── dataset/                    # 数据集存放目录
├── outputs/                    # 输出文件
│   ├── logs/                  # 训练日志
│   └── models/                # 保存的模型
├── service/                   # 服务相关
│   ├── data_analysis.ipynb    # 数据分析notebook
│   └── prediction_service.py  # 预测服务
└── src/                       # 源代码
    ├── __init__.py
    ├── dataset.py             # 数据处理
    ├── model.py               # LightGCN模型
    ├── train.py               # 训练脚本
    ├── evaluate.py            # 评估脚本
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
将数据文件放在 `dataset/` 目录下：
- `tianchi_mobile_recommend_train_user.zip` 或 `tianchi_mobile_recommend_train_user.csv`
- `tianchi_mobile_recommend_train_item.csv`

如果没有真实数据，系统会自动生成示例数据用于测试。

### 3. 训练模型
```bash
# 使用默认配置训练
python run_train.py

# 使用自定义配置
python run_train.py --config custom_config.json
```

### 4. 评估模型
```bash
# 评估训练好的模型
python run_evaluate.py

# 指定模型路径
python run_evaluate.py --model-path outputs/models/best_model.pt
```

### 5. 启动推荐服务
```bash
cd service
python prediction_service.py --host 0.0.0.0 --port 5000
```

### 6. API使用示例

#### 为用户生成推荐
```bash
curl "http://localhost:5000/recommend/123?k=10"
```

#### 批量推荐
```bash
curl -X POST "http://localhost:5000/recommend/batch" \
  -H "Content-Type: application/json" \
  -d '{"user_ids": [123, 456, 789], "k": 10}'
```

#### 获取相似物品
```bash
curl "http://localhost:5000/similar/456?k=10"
```

#### 预测评分
```bash
curl -X POST "http://localhost:5000/predict" \
  -H "Content-Type: application/json" \
  -d '{"user_id": 123, "item_id": 456}'
```

## 模型特点

- **LightGCN**: 简化的图卷积网络，去除了特征变换和非线性激活
- **高效性**: 相比传统GCN，计算更加高效
- **适用性**: 专门针对推荐系统的协同过滤场景设计

## 评价指标

- Precision（精确度）
- Recall（召回率）  
- F1-Score

## 行为类型

- 1: 浏览
- 2: 收藏
- 3: 加购物车
- 4: 购买

## 快速开始

### 环境测试
```bash
python test_setup.py
```

### 完整流程
```bash
# 1. 安装依赖
pip install -r requirements.txt

# 2. 环境测试
python test_setup.py

# 3. 训练模型（使用示例数据）
python run_train.py

# 4. 评估模型
python run_evaluate.py

# 5. 启动推荐服务
cd service
python prediction_service.py
```

## 项目特色

- ✅ **完整实现**: 从数据处理到模型部署的完整推荐系统
- ✅ **先进算法**: 基于LightGCN的图神经网络推荐算法
- ✅ **易于使用**: 提供简单的命令行接口和Web API
- ✅ **可扩展性**: 模块化设计，易于扩展和定制
- ✅ **生产就绪**: 包含完整的训练、评估和部署流程

## 技术栈

- **深度学习**: PyTorch + PyTorch Geometric
- **数据处理**: Pandas + NumPy
- **可视化**: Matplotlib + Seaborn  
- **Web服务**: Flask
- **模型**: LightGCN (Light Graph Convolutional Networks)