#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
基于ONNX Runtime的LSTM情感分析推理服务
提供高性能的情感分析API服务
"""

import os
import sys
import json
import pickle
import time
from typing import Dict, List, Any
import numpy as np
import onnxruntime as ort
from flask import Flask, request, jsonify, render_template_string
import logging

# 添加项目根目录到Python路径
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from src.utils import tokenize_chinese

app = Flask(__name__)
app.config['JSON_AS_ASCII'] = False

# 全局变量
ort_session = None
vocab = None
label_encoder = None
metadata = None

# 禁用Flask日志
log = logging.getLogger('werkzeug')
log.setLevel(logging.ERROR)

# HTML模板
HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>ONNX情感分析服务</title>
    <style>
        body {
            font-family: 'Microsoft YaHei', Arial, sans-serif;
            max-width: 900px;
            margin: 0 auto;
            padding: 20px;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
        }
        .container {
            background-color: white;
            padding: 30px;
            border-radius: 15px;
            box-shadow: 0 10px 30px rgba(0,0,0,0.3);
        }
        h1 {
            text-align: center;
            color: #333;
            margin-bottom: 10px;
            font-size: 2.5em;
        }
        .subtitle {
            text-align: center;
            color: #666;
            margin-bottom: 30px;
            font-size: 1.1em;
        }
        .performance-info {
            background: linear-gradient(45deg, #667eea, #764ba2);
            color: white;
            padding: 15px;
            border-radius: 10px;
            margin-bottom: 30px;
            text-align: center;
        }
        .input-group {
            margin-bottom: 20px;
        }
        label {
            display: block;
            margin-bottom: 8px;
            font-weight: bold;
            color: #555;
            font-size: 1.1em;
        }
        textarea {
            width: 100%;
            height: 120px;
            padding: 15px;
            border: 2px solid #ddd;
            border-radius: 10px;
            font-size: 16px;
            resize: vertical;
            box-sizing: border-box;
            transition: border-color 0.3s;
        }
        textarea:focus {
            border-color: #667eea;
            outline: none;
            box-shadow: 0 0 10px rgba(102, 126, 234, 0.3);
        }
        .btn-container {
            text-align: center;
            margin: 25px 0;
        }
        button {
            background: linear-gradient(45deg, #667eea, #764ba2);
            color: white;
            padding: 15px 40px;
            border: none;
            border-radius: 25px;
            font-size: 18px;
            font-weight: bold;
            cursor: pointer;
            transition: all 0.3s;
            box-shadow: 0 5px 15px rgba(0,0,0,0.2);
        }
        button:hover {
            transform: translateY(-2px);
            box-shadow: 0 8px 25px rgba(0,0,0,0.3);
        }
        button:disabled {
            background: #cccccc;
            cursor: not-allowed;
            transform: none;
        }
        .result {
            margin-top: 25px;
            padding: 20px;
            border-radius: 10px;
            text-align: center;
            font-size: 18px;
            font-weight: bold;
        }
        .positive {
            background: linear-gradient(45deg, #4CAF50, #45a049);
            color: white;
        }
        .negative {
            background: linear-gradient(45deg, #f44336, #da190b);
            color: white;
        }
        .neutral {
            background: linear-gradient(45deg, #2196F3, #0b7dda);
            color: white;
        }
        .error {
            background: linear-gradient(45deg, #ff5722, #e64a19);
            color: white;
        }
        .loading {
            background: linear-gradient(45deg, #ff9800, #f57c00);
            color: white;
        }
        .stats {
            margin-top: 20px;
            padding: 15px;
            background-color: #f8f9fa;
            border-radius: 10px;
            font-size: 14px;
            color: #666;
        }
        .examples {
            margin-top: 30px;
            padding: 25px;
            background-color: #f8f9fa;
            border-radius: 15px;
        }
        .examples h3 {
            margin-top: 0;
            color: #333;
            font-size: 1.3em;
        }
        .example-text {
            background-color: white;
            padding: 15px;
            margin: 10px 0;
            border-radius: 8px;
            cursor: pointer;
            border: 2px solid transparent;
            transition: all 0.3s;
        }
        .example-text:hover {
            border-color: #667eea;
            transform: translateY(-2px);
            box-shadow: 0 5px 15px rgba(0,0,0,0.1);
        }
        .api-section {
            margin-top: 30px;
            padding: 25px;
            background: linear-gradient(45deg, #e8eaf6, #f3e5f5);
            border-radius: 15px;
        }
        .api-section h3 {
            margin-top: 0;
            color: #333;
            font-size: 1.3em;
        }
        .api-section code {
            background-color: #2d3748;
            color: #68d391;
            padding: 4px 8px;
            border-radius: 4px;
            font-family: 'Courier New', monospace;
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>🚀 ONNX情感分析服务</h1>
        <div class="subtitle">基于ONNX Runtime的高性能推理</div>
        
        <div class="performance-info">
            <strong>⚡ 高性能推理引擎</strong> | 支持CPU/GPU加速 | 毫秒级响应
        </div>
        
        <div class="input-group">
            <label for="text-input">📝 请输入要分析的文本：</label>
            <textarea id="text-input" placeholder="在这里输入您想要分析情感的文本..."></textarea>
        </div>
        
        <div class="btn-container">
            <button onclick="analyzeText()" id="analyze-btn">🧠 分析情感</button>
        </div>
        
        <div id="result" class="result" style="display:none;"></div>
        <div id="stats" class="stats" style="display:none;"></div>
        
        <div class="examples">
            <h3>💡 示例文本（点击使用）：</h3>
            <div class="example-text" onclick="fillText('这部电影真的很棒，演员演技精湛，剧情引人入胜，强烈推荐！')">
                这部电影真的很棒，演员演技精湛，剧情引人入胜，强烈推荐！
            </div>
            <div class="example-text" onclick="fillText('服务态度非常差，等了很久都没人理，很不满意这次购物体验。')">
                服务态度非常差，等了很久都没人理，很不满意这次购物体验。
            </div>
            <div class="example-text" onclick="fillText('这个产品质量还可以，价格也比较合理，总体来说没什么问题。')">
                这个产品质量还可以，价格也比较合理，总体来说没什么问题。
            </div>
            <div class="example-text" onclick="fillText('今天天气很好，心情也不错，准备出去走走。')">
                今天天气很好，心情也不错，准备出去走走。
            </div>
        </div>
        
        <div class="api-section">
            <h3>🔧 API使用说明</h3>
            <p><strong>POST</strong> <code>/predict</code></p>
            <p>请求体: <code>{"text": "要分析的文本"}</code></p>
            <p>返回: <code>{"sentiment": "情感标签", "confidence": 0.95, "inference_time": 0.003}</code></p>
            <br>
            <p><strong>GET</strong> <code>/health</code> - 健康检查</p>
            <p><strong>GET</strong> <code>/info</code> - 模型信息</p>
        </div>
    </div>

    <script>
        function fillText(text) {
            document.getElementById('text-input').value = text;
        }
        
        function analyzeText() {
            const text = document.getElementById('text-input').value.trim();
            const resultDiv = document.getElementById('result');
            const statsDiv = document.getElementById('stats');
            const analyzeBtn = document.getElementById('analyze-btn');
            
            if (!text) {
                resultDiv.innerHTML = '❌ 请输入要分析的文本';
                resultDiv.className = 'result error';
                resultDiv.style.display = 'block';
                statsDiv.style.display = 'none';
                return;
            }
            
            // 显示加载状态
            resultDiv.innerHTML = '🔄 正在分析中...';
            resultDiv.className = 'result loading';
            resultDiv.style.display = 'block';
            analyzeBtn.disabled = true;
            statsDiv.style.display = 'none';
            
            const startTime = performance.now();
            
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
                const clientTime = performance.now() - startTime;
                
                if (data.error) {
                    resultDiv.innerHTML = '❌ 分析失败: ' + data.error;
                    resultDiv.className = 'result error';
                } else {
                    const sentiment = data.sentiment;
                    const confidence = (data.confidence * 100).toFixed(1);
                    const serverTime = (data.inference_time * 1000).toFixed(1);
                    
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
                    
                    resultDiv.innerHTML = `${emoticon} ${sentimentText}<br>置信度: ${confidence}%`;
                    resultDiv.className = className;
                    
                    statsDiv.innerHTML = `
                        <strong>⚡ 性能统计:</strong><br>
                        服务器推理时间: ${serverTime}ms | 
                        总响应时间: ${clientTime.toFixed(1)}ms | 
                        文本长度: ${text.length}字符
                    `;
                    statsDiv.style.display = 'block';
                }
            })
            .catch(error => {
                resultDiv.innerHTML = '❌ 请求失败: ' + error.message;
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


class ONNXSentimentPredictor:
    """ONNX情感分析预测器"""
    
    def __init__(self, onnx_path: str, vocab_path: str, label_encoder_path: str,
                 metadata_path: str = None):
        """
        初始化ONNX预测器
        
        Args:
            onnx_path: ONNX模型路径
            vocab_path: 词汇表路径
            label_encoder_path: 标签编码器路径
            metadata_path: 元数据路径
        """
        # 创建ONNX Runtime会话
        self.ort_session = ort.InferenceSession(onnx_path)
        
        # 加载词汇表
        with open(vocab_path, 'rb') as f:
            self.vocab = pickle.load(f)
        
        # 加载标签编码器
        with open(label_encoder_path, 'rb') as f:
            self.label_encoder = pickle.load(f)
        
        # 加载元数据
        self.metadata = None
        if metadata_path and os.path.exists(metadata_path):
            with open(metadata_path, 'r', encoding='utf-8') as f:
                self.metadata = json.load(f)
        
        self.max_length = 128
        
        print("ONNX模型加载成功")
        print(f"词汇表大小: {len(self.vocab)}")
        print(f"支持的情感类别: {self.label_encoder.classes_}")
    
    def preprocess_text(self, text: str) -> Dict[str, np.ndarray]:
        """
        预处理文本
        
        Args:
            text: 输入文本
            
        Returns:
            预处理后的输入字典
        """
        # 分词
        tokens = tokenize_chinese(text)
        
        # 转换为token IDs
        token_ids = [self.vocab.get(token, self.vocab.get('<UNK>', 1)) for token in tokens]
        
        # 截断或填充
        if len(token_ids) > self.max_length:
            token_ids = token_ids[:self.max_length]
            length = self.max_length
        else:
            length = len(token_ids)
            token_ids.extend([0] * (self.max_length - len(token_ids)))
        
        # 转换为numpy数组
        input_ids = np.array([token_ids], dtype=np.int64)
        lengths = np.array([length], dtype=np.int64)
        
        return {
            'input_ids': input_ids,
            'lengths': lengths
        }
    
    def predict(self, text: str) -> Dict[str, Any]:
        """
        预测文本情感
        
        Args:
            text: 输入文本
            
        Returns:
            预测结果字典
        """
        start_time = time.time()
        
        # 预处理
        inputs = self.preprocess_text(text)
        
        # ONNX推理
        outputs = self.ort_session.run(None, inputs)[0]
        
        # 后处理
        probabilities = np.exp(outputs) / np.sum(np.exp(outputs), axis=1, keepdims=True)
        predicted_class = np.argmax(probabilities, axis=1)[0]
        confidence = probabilities[0][predicted_class]
        
        predicted_label = self.label_encoder.inverse_transform([predicted_class])[0]
        
        inference_time = time.time() - start_time
        
        return {
            'sentiment': predicted_label,
            'confidence': float(confidence),
            'probabilities': {
                label: float(prob) for label, prob in 
                zip(self.label_encoder.classes_, probabilities[0])
            },
            'inference_time': inference_time
        }
    
    def get_model_info(self) -> Dict[str, Any]:
        """获取模型信息"""
        info = {
            'vocab_size': len(self.vocab),
            'num_classes': len(self.label_encoder.classes_),
            'class_names': self.label_encoder.classes_.tolist(),
            'max_length': self.max_length,
            'onnx_providers': self.ort_session.get_providers()
        }
        
        if self.metadata:
            info.update(self.metadata)
        
        return info


def load_onnx_model():
    """加载ONNX模型"""
    global ort_session, vocab, label_encoder, metadata
    
    try:
        # 检查文件
        onnx_path = 'outputs/model_best.onnx'
        vocab_path = 'outputs/vocab.pkl'
        label_encoder_path = 'outputs/label_encoder.pkl'
        metadata_path = 'outputs/model_best_metadata.json'
        
        if not os.path.exists(onnx_path):
            print(f"错误: ONNX模型文件不存在: {onnx_path}")
            print("请先运行 'python service/convert_to_onnx.py' 转换模型")
            return False
        
        # 创建预测器
        predictor = ONNXSentimentPredictor(
            onnx_path=onnx_path,
            vocab_path=vocab_path,
            label_encoder_path=label_encoder_path,
            metadata_path=metadata_path
        )
        
        # 设置全局变量
        ort_session = predictor.ort_session
        vocab = predictor.vocab
        label_encoder = predictor.label_encoder
        metadata = predictor.metadata
        
        # 保存预测器实例
        app.predictor = predictor
        
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
        
        if not hasattr(app, 'predictor'):
            return jsonify({'error': 'ONNX模型未加载'}), 500
        
        # 进行预测
        result = app.predictor.predict(text)
        result['text'] = text
        
        return jsonify(result)
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/health')
def health():
    """健康检查"""
    return jsonify({
        'status': 'healthy',
        'model_loaded': hasattr(app, 'predictor'),
        'providers': ort_session.get_providers() if ort_session else []
    })


@app.route('/info')
def info():
    """模型信息"""
    try:
        if hasattr(app, 'predictor'):
            return jsonify(app.predictor.get_model_info())
        else:
            return jsonify({'error': 'ONNX模型未加载'}), 500
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/batch_predict', methods=['POST'])
def batch_predict():
    """批量预测API"""
    try:
        data = request.get_json()
        
        if not data or 'texts' not in data:
            return jsonify({'error': '请提供texts字段（文本列表）'}), 400
        
        texts = data['texts']
        if not isinstance(texts, list):
            return jsonify({'error': 'texts必须是列表'}), 400
        
        if len(texts) > 100:
            return jsonify({'error': '批量预测最多支持100个文本'}), 400
        
        if not hasattr(app, 'predictor'):
            return jsonify({'error': 'ONNX模型未加载'}), 500
        
        # 批量预测
        results = []
        total_start_time = time.time()
        
        for text in texts:
            if text.strip():
                result = app.predictor.predict(text.strip())
                result['text'] = text.strip()
                results.append(result)
        
        total_time = time.time() - total_start_time
        
        return jsonify({
            'results': results,
            'total_time': total_time,
            'count': len(results)
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500


def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description='ONNX情感分析推理服务')
    parser.add_argument('--host', type=str, default='0.0.0.0', help='服务器地址')
    parser.add_argument('--port', type=int, default=8000, help='端口号')
    parser.add_argument('--debug', action='store_true', help='调试模式')
    
    args = parser.parse_args()
    
    print("="*60)
    print("ONNX情感分析推理服务")
    print("="*60)
    
    # 加载模型
    if not load_onnx_model():
        print("模型加载失败，退出程序")
        return
    
    print(f"🚀 服务器启动中...")
    print(f"📍 地址: http://{args.host}:{args.port}")
    print(f"🔧 API端点: http://{args.host}:{args.port}/predict")
    print(f"📊 批量预测: http://{args.host}:{args.port}/batch_predict")
    print("⏹ 按 Ctrl+C 停止服务器")
    
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