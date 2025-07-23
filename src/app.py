#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
LSTM情感分析Web应用
基于Flask的情感分析API服务
"""

import os
import sys
import pickle
import json
import torch
from flask import Flask, request, jsonify, render_template_string
import logging

# 添加项目根目录到Python路径
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from src.model import LSTMSentimentModel
from src.utils import predict_sentiment, load_model

app = Flask(__name__)
app.config['JSON_AS_ASCII'] = False  # 支持中文JSON

# 全局变量
model = None
vocab = None
label_encoder = None
device = None

# 禁用Flask的日志输出
log = logging.getLogger('werkzeug')
log.setLevel(logging.ERROR)

# HTML模板
HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>LSTM情感分析</title>
    <style>
        body {
            font-family: 'Microsoft YaHei', Arial, sans-serif;
            max-width: 800px;
            margin: 0 auto;
            padding: 20px;
            background-color: #f5f5f5;
        }
        .container {
            background-color: white;
            padding: 30px;
            border-radius: 10px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        }
        h1 {
            text-align: center;
            color: #333;
            margin-bottom: 30px;
        }
        .input-group {
            margin-bottom: 20px;
        }
        label {
            display: block;
            margin-bottom: 5px;
            font-weight: bold;
            color: #555;
        }
        textarea {
            width: 100%;
            height: 120px;
            padding: 10px;
            border: 2px solid #ddd;
            border-radius: 5px;
            font-size: 16px;
            resize: vertical;
            box-sizing: border-box;
        }
        textarea:focus {
            border-color: #4CAF50;
            outline: none;
        }
        .btn-container {
            text-align: center;
            margin: 20px 0;
        }
        button {
            background-color: #4CAF50;
            color: white;
            padding: 12px 30px;
            border: none;
            border-radius: 5px;
            font-size: 16px;
            cursor: pointer;
            transition: background-color 0.3s;
        }
        button:hover {
            background-color: #45a049;
        }
        button:disabled {
            background-color: #cccccc;
            cursor: not-allowed;
        }
        .result {
            margin-top: 20px;
            padding: 15px;
            border-radius: 5px;
            text-align: center;
            font-size: 18px;
            font-weight: bold;
        }
        .positive {
            background-color: #d4edda;
            color: #155724;
            border: 1px solid #c3e6cb;
        }
        .negative {
            background-color: #f8d7da;
            color: #721c24;
            border: 1px solid #f5c6cb;
        }
        .neutral {
            background-color: #d1ecf1;
            color: #0c5460;
            border: 1px solid #bee5eb;
        }
        .error {
            background-color: #f8d7da;
            color: #721c24;
            border: 1px solid #f5c6cb;
        }
        .loading {
            background-color: #fff3cd;
            color: #856404;
            border: 1px solid #ffeaa7;
        }
        .examples {
            margin-top: 30px;
            padding: 20px;
            background-color: #f8f9fa;
            border-radius: 5px;
        }
        .example-text {
            background-color: white;
            padding: 10px;
            margin: 5px 0;
            border-radius: 3px;
            cursor: pointer;
            border: 1px solid #ddd;
            transition: background-color 0.3s;
        }
        .example-text:hover {
            background-color: #e9ecef;
        }
        .api-info {
            margin-top: 30px;
            padding: 20px;
            background-color: #e9ecef;
            border-radius: 5px;
        }
        .api-info h3 {
            margin-top: 0;
            color: #495057;
        }
        .api-info code {
            background-color: #f8f9fa;
            padding: 2px 5px;
            border-radius: 3px;
            font-family: 'Courier New', monospace;
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>🧠 LSTM情感分析系统</h1>
        
        <div class="input-group">
            <label for="text-input">请输入要分析的文本：</label>
            <textarea id="text-input" placeholder="在这里输入您想要分析情感的文本..."></textarea>
        </div>
        
        <div class="btn-container">
            <button onclick="analyzeText()" id="analyze-btn">分析情感</button>
        </div>
        
        <div id="result" class="result" style="display:none;"></div>
        
        <div class="examples">
            <h3>示例文本（点击使用）：</h3>
            <div class="example-text" onclick="fillText('这部电影真的很棒，演员演技很好，剧情也很有趣！')">
                这部电影真的很棒，演员演技很好，剧情也很有趣！
            </div>
            <div class="example-text" onclick="fillText('服务态度很差，等了很久都没有人理我，很不满意。')">
                服务态度很差，等了很久都没有人理我，很不满意。
            </div>
            <div class="example-text" onclick="fillText('这个产品质量还可以，价格也合理。')">
                这个产品质量还可以，价格也合理。
            </div>
        </div>
        
        <div class="api-info">
            <h3>API使用说明</h3>
            <p><strong>POST</strong> <code>/predict</code></p>
            <p>请求体: <code>{"text": "要分析的文本"}</code></p>
            <p>返回: <code>{"sentiment": "情感标签", "confidence": 0.95}</code></p>
        </div>
    </div>

    <script>
        function fillText(text) {
            document.getElementById('text-input').value = text;
        }
        
        function analyzeText() {
            const text = document.getElementById('text-input').value.trim();
            const resultDiv = document.getElementById('result');
            const analyzeBtn = document.getElementById('analyze-btn');
            
            if (!text) {
                resultDiv.innerHTML = '请输入要分析的文本';
                resultDiv.className = 'result error';
                resultDiv.style.display = 'block';
                return;
            }
            
            // 显示加载状态
            resultDiv.innerHTML = '正在分析中...';
            resultDiv.className = 'result loading';
            resultDiv.style.display = 'block';
            analyzeBtn.disabled = true;
            
            // 发送请求
            fetch('/predict', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({text: text})
            })
            .then(response => response.json())
            .then(data => {
                if (data.error) {
                    resultDiv.innerHTML = '分析失败: ' + data.error;
                    resultDiv.className = 'result error';
                } else {
                    const sentiment = data.sentiment;
                    const confidence = (data.confidence * 100).toFixed(1);
                    
                    let emoticon = '';
                    let sentimentText = '';
                    let className = 'result ';
                    
                    if (sentiment === 'positive') {
                        emoticon = '😊';
                        sentimentText = '正面情感';
                        className += 'positive';
                    } else if (sentiment === 'negative') {
                        emoticon = '😞';
                        sentimentText = '负面情感';
                        className += 'negative';
                    } else {
                        emoticon = '😐';
                        sentimentText = '中性情感';
                        className += 'neutral';
                    }
                    
                    resultDiv.innerHTML = `${emoticon} ${sentimentText} (置信度: ${confidence}%)`;
                    resultDiv.className = className;
                }
            })
            .catch(error => {
                resultDiv.innerHTML = '请求失败: ' + error.message;
                resultDiv.className = 'result error';
            })
            .finally(() => {
                analyzeBtn.disabled = false;
            });
        }
        
        // 支持回车键提交
        document.getElementById('text-input').addEventListener('keydown', function(e) {
            if (e.key === 'Enter' && e.ctrlKey) {
                analyzeText();
            }
        });
    </script>
</body>
</html>
"""


def load_model_and_resources():
    """加载模型和相关资源"""
    global model, vocab, label_encoder, device
    
    try:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # 加载词汇表
        with open('outputs/vocab.pkl', 'rb') as f:
            vocab = pickle.load(f)
        
        # 加载标签编码器
        with open('outputs/label_encoder.pkl', 'rb') as f:
            label_encoder = pickle.load(f)
        
        # 加载模型信息
        with open('outputs/model_info.json', 'r', encoding='utf-8') as f:
            model_info = json.load(f)
        
        # 创建模型实例
        model = LSTMSentimentModel(
            vocab_size=model_info['vocab_size'],
            embedding_dim=model_info['model_config']['embedding_dim'],
            hidden_dim=model_info['model_config']['hidden_dim'],
            num_layers=model_info['model_config']['num_layers'],
            num_classes=model_info['num_classes'],
            dropout=model_info['model_config']['dropout']
        )
        
        # 加载训练好的权重
        load_model(model, 'outputs/model_best.pt', device)
        model.to(device)
        model.eval()
        
        print("模型加载成功！")
        return True
        
    except Exception as e:
        print(f"模型加载失败: {e}")
        return False


@app.route('/')
def index():
    """主页"""
    return render_template_string(HTML_TEMPLATE)


@app.route('/predict', methods=['POST'])
def predict():
    """情感预测API"""
    try:
        data = request.get_json()
        
        if not data or 'text' not in data:
            return jsonify({'error': '请提供text字段'}), 400
        
        text = data['text'].strip()
        if not text:
            return jsonify({'error': '文本不能为空'}), 400
        
        if model is None:
            return jsonify({'error': '模型未加载'}), 500
        
        # 进行预测
        predicted_label = predict_sentiment(
            model, text, vocab, label_encoder, device
        )
        
        # 获取预测概率
        model.eval()
        with torch.no_grad():
            from src.utils import tokenize_chinese
            tokens = tokenize_chinese(text)
            token_ids = [vocab.get(token, vocab.get('<UNK>', 1)) for token in tokens]
            
            max_length = 128
            if len(token_ids) > max_length:
                token_ids = token_ids[:max_length]
                length = max_length
            else:
                length = len(token_ids)
                token_ids.extend([0] * (max_length - len(token_ids)))
            
            input_ids = torch.tensor([token_ids], dtype=torch.long).to(device)
            lengths = torch.tensor([length], dtype=torch.long).to(device)
            
            outputs = model(input_ids, lengths)
            probabilities = torch.softmax(outputs, dim=1)
            confidence = torch.max(probabilities).item()
        
        return jsonify({
            'sentiment': predicted_label,
            'confidence': confidence,
            'text': text
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/health')
def health():
    """健康检查"""
    return jsonify({
        'status': 'healthy',
        'model_loaded': model is not None
    })


@app.route('/info')
def info():
    """模型信息"""
    try:
        if os.path.exists('outputs/model_info.json'):
            with open('outputs/model_info.json', 'r', encoding='utf-8') as f:
                model_info = json.load(f)
            return jsonify(model_info)
        else:
            return jsonify({'error': '模型信息文件不存在'}), 404
    except Exception as e:
        return jsonify({'error': str(e)}), 500


def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description='LSTM情感分析Web服务')
    parser.add_argument('--host', type=str, default='0.0.0.0', help='服务器地址')
    parser.add_argument('--port', type=int, default=5000, help='端口号')
    parser.add_argument('--debug', action='store_true', help='调试模式')
    
    args = parser.parse_args()
    
    print("="*50)
    print("LSTM情感分析Web服务")
    print("="*50)
    
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
    
    # 加载模型
    if not load_model_and_resources():
        print("模型加载失败，退出程序")
        return
    
    print(f"服务器启动中...")
    print(f"地址: http://{args.host}:{args.port}")
    print(f"API端点: http://{args.host}:{args.port}/predict")
    print("按 Ctrl+C 停止服务器")
    
    try:
        app.run(
            host=args.host,
            port=args.port,
            debug=args.debug,
            use_reloader=False
        )
    except KeyboardInterrupt:
        print("\n服务器已停止")


if __name__ == '__main__':
    main()