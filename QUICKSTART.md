# 🚀 LightGCN推荐系统 - 快速开始

## 📦 安装依赖

```bash
pip install -r requirements.txt
```

## 🎯 快速启动（推荐）

### 方式1: Web界面（推荐）
```bash
python run_app.py
```
然后在浏览器中访问：http://localhost:8501

### 方式2: 直接启动Web界面
```bash
streamlit run src/app.py
```

## 🧪 测试系统

在使用前，建议先运行测试脚本确保一切正常：

```bash
python test_system.py
```

## 🖥️ 命令行使用

### 训练模型
```bash
python -m src.app --mode train
```

### 生成推荐
```bash
python -m src.app --mode inference --user-id 0 --top-k 10
```

### 转换ONNX模型
```bash
python service/convert_to_onnx.py --model-path outputs/models/best_model.pt
```

### 启动API服务
```bash
python service/onnx_server.py --model-path outputs/models/best_model.onnx
```

## 🎨 Web界面功能

- **🏠 首页**: 系统状态和功能导航
- **🎯 推荐**: 交互式推荐生成和可视化
- **📈 训练**: 模型训练配置和监控
- **📊 分析**: 数据分析和统计

## ⚡ 快速测试流程

1. 运行测试: `python test_system.py`
2. 启动界面: `python run_app.py`
3. 在Web界面的"训练"选项卡中开始训练
4. 训练完成后在"推荐"选项卡生成推荐
5. 在"分析"选项卡查看数据统计

## 🔧 故障排除

### 依赖问题
如果遇到依赖安装问题，请确保Python版本 >= 3.8：
```bash
python --version
pip install --upgrade pip
pip install -r requirements.txt
```

### PyTorch Geometric安装
如果PyTorch Geometric安装失败，请参考官方安装指南：
```bash
pip install torch torchvision torchaudio
pip install torch-geometric
```

### 端口冲突
如果8501端口被占用，可以指定其他端口：
```bash
streamlit run src/app.py --server.port 8502
```

## 📚 更多信息

详细文档请参考 `README.md`