#!/usr/bin/env python3
"""
LightGCN推荐系统启动脚本
"""

import os
import sys
import subprocess

def main():
    """主函数"""
    print("🚀 启动LightGCN推荐系统...")
    
    # 检查Streamlit是否安装
    try:
        import streamlit
    except ImportError:
        print("❌ 缺少Streamlit依赖，请运行: pip install streamlit")
        return 1
    
    # 启动Streamlit应用
    try:
        cmd = [
            sys.executable, "-m", "streamlit", "run",
            "src/app.py",
            "--server.port=8501",
            "--server.address=0.0.0.0",
            "--server.headless=true"
        ]
        
        print("📍 应用地址: http://localhost:8501")
        print("🔄 正在启动应用...")
        
        subprocess.run(cmd)
        
    except KeyboardInterrupt:
        print("\n👋 应用已停止")
    except Exception as e:
        print(f"❌ 启动失败: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())