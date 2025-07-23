# 🚀 LightGCN推荐系统 - 快速开始

## 📦 安装依赖

```bash
pip install -r requirements.txt
```

## 🎯 完整工作流程

### Step 1: 模型训练
```bash
# 训练LightGCN模型
python -m src.train

# 自定义参数训练（可选）
python -m src.train --epochs 30 --batch-size 256
```

### Step 2: 模型转换
```bash
# 将PyTorch模型转换为ONNX格式
python service/convert_to_onnx.py --model-path outputs/models/best_model.pt --save-metadata
```

### Step 3: 启动API服务
```bash
# 启动ONNX推理服务
python service/onnx_server.py --model-path outputs/models/best_model.onnx
```

### Step 4: 启动可视化界面
```bash
# 启动Web界面（新终端窗口）
python run_app.py
```

然后访问：http://localhost:8501

## 🧪 测试系统

在开始前，建议先运行测试确保环境正常：

```bash
python test_system.py
```

## ⚡ 快速体验（无需训练）

如果你想快速体验系统但没有训练好的模型：

1. **仅启动界面**：
   ```bash
   python run_app.py
   ```
   
2. **查看数据分析功能**：界面会使用示例数据展示分析功能

3. **API配置测试**：在界面中测试API连接（会显示服务离线状态）

## 📱 界面功能

### 🎯 推荐服务
- 单用户推荐生成
- 批量用户推荐
- 推荐结果可视化
- 实时API调用

### 📊 数据分析
- 用户行为分布
- 物品流行度分析  
- 时间趋势分析
- 系统统计指标

### ⚙️ API配置
- 服务地址设置
- 连接状态测试
- API接口文档

## 🔧 开发模式

### 快速训练（测试用）
```bash
python -m src.train --epochs 5 --batch-size 128
```

### 调试API服务
```bash
python service/onnx_server.py --model-path outputs/models/best_model.onnx --debug
```

### 开发界面
```bash
streamlit run src/app.py --server.runOnSave=true
```

## 📡 API测试

### 检查服务状态
```bash
curl http://localhost:8080/health
```

### 获取推荐
```bash
curl -X POST http://localhost:8080/recommend \
  -H "Content-Type: application/json" \
  -d '{"user_id": 0, "top_k": 5}'
```

## 🎨 架构说明

- **src/train.py**: 模型训练（独立运行）
- **service/onnx_server.py**: API推理服务
- **src/app.py**: 可视化界面（调用API）

这种分离式设计允许：
- 🔄 独立训练和部署
- ⚡ 高性能推理服务
- 🎨 灵活的前端界面

## 🔧 故障排除

### 依赖问题
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### 端口冲突
```bash
# API服务使用其他端口
python service/onnx_server.py --port 8081

# 界面使用其他端口  
streamlit run src/app.py --server.port 8502
```

### 模型文件不存在
确保先运行训练：
```bash
python -m src.train
```

### API连接失败
1. 确认API服务正在运行
2. 检查端口是否正确（默认8080）
3. 在界面的"API配置"中测试连接

## 📚 更多信息

详细文档请参考 `README.md`