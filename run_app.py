#!/usr/bin/env python3
"""
LightGCN推荐系统启动脚本
启动可视化界面（基于ONNX API）
"""

import os
import sys
import subprocess

def main():
    """主函数"""
    print("🚀 启动LightGCN推荐系统可视化界面...")
    
    # 检查Streamlit是否安装
    try:
        import streamlit
    except ImportError:
        print("❌ 缺少Streamlit依赖，请运行: pip install streamlit")
        return 1
    
    # 检查requests是否安装
    try:
        import requests
    except ImportError:
        print("❌ 缺少requests依赖，请运行: pip install requests")
        return 1
    
    print("📋 功能说明：")
    print("  - 🎯 推荐服务: 调用ONNX API生成推荐")
    print("  - 📊 数据分析: 查看数据统计和可视化")
    print("  - ⚙️ API配置: 设置和测试API连接")
    print()
    print("📝 注意事项：")
    print("  - 本界面需要ONNX API服务运行才能进行推荐")
    print("  - 如需训练模型，请运行: python -m src.train")
    print("  - API服务启动: python service/onnx_server.py --model-path outputs/models/best_model.onnx")
    print()
    
    # 启动Streamlit应用
    try:
        cmd = [
            sys.executable, "-m", "streamlit", "run",
            "src/app.py",
            "--server.port=8501",
            "--server.address=0.0.0.0",
            "--server.headless=true",
            "--browser.gatherUsageStats=false"
        ]
        
        print("📍 应用地址: http://localhost:8501")
        print("🔄 正在启动可视化界面...")
        print("💡 提示: 按 Ctrl+C 停止应用")
        print("-" * 50)
        
        subprocess.run(cmd)
        
    except KeyboardInterrupt:
        print("\n👋 应用已停止")
    except Exception as e:
        print(f"❌ 启动失败: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())