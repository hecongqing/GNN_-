# LightGCN 训练模块

本文档说明了LightGCN训练代码的组织结构，以便于学习和使用。

## 文件结构

```
src/
├── model.py           # 模型定义 (LightGCN, LightGCNConv)
├── train.py           # 训练相关 (LightGCNTrainer, BPRLoss)
├── dataset.py         # 数据处理
└── utils.py          # 工具函数
```

## 主要组件

### 1. 模型组件 (`model.py`)

- **LightGCNConv**: LightGCN卷积层实现
- **LightGCN**: 主要的LightGCN模型类

### 2. 训练组件 (`train.py`)

- **BPRLoss**: 贝叶斯个性化排序损失函数
- **LightGCNTrainer**: 训练器类，包含完整的训练逻辑
- **LightGCNTrainingPipeline**: 完整的训练管道

## 使用方法

### 基础使用

```python
from src.model import LightGCN
from src.train import LightGCNTrainer

# 创建模型
model = LightGCN(n_users=1000, n_items=2000, embedding_dim=64)

# 创建训练器
trainer = LightGCNTrainer(model)

# 训练模型
history = trainer.train(
    edge_index=edge_index,
    interaction_matrix=interaction_matrix,
    n_epochs=100
)
```

### 自定义训练循环

```python
# 只使用训练器的组件
from src.train import BPRLoss

# 创建损失函数
criterion = BPRLoss()

# 手动训练循环
for epoch in range(n_epochs):
    # 获取批次数据
    users, pos_items, neg_items = trainer.create_bpr_batch(interaction_matrix)
    
    # 前向传播
    user_emb, item_emb = model(edge_index)
    pos_scores = model.predict(users, pos_items, user_emb, item_emb)
    neg_scores = model.predict(users, neg_items, user_emb, item_emb)
    
    # 计算损失
    loss = criterion(pos_scores, neg_scores)
```

## 训练器特性

### LightGCNTrainer 提供的功能：

1. **自动批次生成**: `create_bpr_batch()` - 自动生成BPR训练批次
2. **单轮训练**: `train_epoch()` - 训练一个epoch
3. **完整训练**: `train()` - 包含优化器、学习率调度等的完整训练流程
4. **模型评估**: `evaluate()` - 评估模型性能

### 主要参数：

- `n_epochs`: 训练轮数
- `lr`: 学习率
- `weight_decay`: 权重衰减
- `batch_size`: 批次大小
- `n_batches`: 每个epoch的批次数量

## 示例

运行训练示例：

```bash
python examples/train_example.py
```

## 优势

1. **模块化**: 模型定义和训练逻辑分离，便于理解和修改
2. **灵活性**: 可以只使用需要的组件，支持自定义训练流程
3. **可扩展**: 易于添加新的损失函数、训练策略等
4. **易学习**: 代码结构清晰，注释详细，便于学习算法原理

## 学习建议

1. 首先阅读 `model.py` 了解LightGCN的核心算法
2. 然后学习 `train.py` 中的BPR损失和训练流程
3. 运行 `examples/train_example.py` 查看完整的训练过程
4. 尝试修改参数和训练策略来理解各组件的作用