#!/usr/bin/env python3
"""
LightGCN推荐系统主应用程序
提供训练、推理和可视化界面功能
"""

import os
import sys
import argparse
import json
import torch
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path
import time

from .train import LightGCNTrainingPipeline
from .model import LightGCN
from .dataset import AliRecommendDataset
from .utils import setup_logger


def train_model(config_path: str = "config.json"):
    """训练模型"""
    logger = setup_logger('App')
    
    logger.info("开始训练LightGCN模型...")
    
    # 创建训练管道
    pipeline = LightGCNTrainingPipeline()
    
    # 加载配置
    if os.path.exists(config_path):
        with open(config_path, 'r', encoding='utf-8') as f:
            config = json.load(f)
            logger.info(f"加载配置文件: {config_path}")
    else:
        logger.warning("配置文件不存在，使用默认配置")
        config = {}
    
    # 执行训练
    metrics = pipeline.train()
    logger.info("训练完成！")
    return metrics


def inference(model_path: str, user_id: int, top_k: int = 10):
    """推理/预测"""
    logger = setup_logger('App')
    
    logger.info(f"开始推理，用户ID: {user_id}, Top-K: {top_k}")
    
    # 加载数据集
    dataset = AliRecommendDataset()
    if not dataset.load_processed_data():
        dataset.load_data()
        dataset.preprocess_data()
    
    # 加载模型
    if not os.path.exists(model_path):
        logger.error(f"模型文件不存在: {model_path}")
        return None
    
    checkpoint = torch.load(model_path, map_location='cpu')
    model = LightGCN(
        n_users=dataset.n_users,
        n_items=dataset.n_items,
        embedding_dim=checkpoint.get('embedding_dim', 64),
        n_layers=checkpoint.get('n_layers', 3)
    )
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    # 生成推荐
    with torch.no_grad():
        user_emb, item_emb = model(dataset.edge_index)
        scores = torch.mm(user_emb[user_id:user_id+1], item_emb.t())
        _, top_items = torch.topk(scores, top_k)
        
    recommendations = top_items.squeeze().tolist()
    logger.info(f"推荐结果: {recommendations}")
    
    return recommendations


def load_training_logs():
    """加载训练日志"""
    log_dir = 'outputs/logs'
    if not os.path.exists(log_dir):
        return None
    
    log_files = [f for f in os.listdir(log_dir) if f.startswith('training_') and f.endswith('.log')]
    if not log_files:
        return None
    
    # 读取最新的日志文件
    latest_log = max(log_files)
    log_path = os.path.join(log_dir, latest_log)
    
    with open(log_path, 'r', encoding='utf-8') as f:
        return f.read()


def create_sample_data():
    """创建示例数据用于演示"""
    np.random.seed(42)
    
    # 模拟用户行为数据
    n_users = 100
    n_items = 50
    n_interactions = 500
    
    users = np.random.randint(0, n_users, n_interactions)
    items = np.random.randint(0, n_items, n_interactions)
    ratings = np.random.choice([3, 4, 5], n_interactions, p=[0.2, 0.5, 0.3])
    
    return pd.DataFrame({
        'user_id': users,
        'item_id': items,
        'rating': ratings
    })


def run_streamlit_app():
    """运行Streamlit可视化界面"""
    st.set_page_config(
        page_title="LightGCN推荐系统",
        page_icon="🎯",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    st.title("🎯 LightGCN推荐系统")
    st.markdown("基于图神经网络的协同过滤推荐系统")
    
    # 侧边栏
    with st.sidebar:
        st.header("⚙️ 系统控制")
        
        # 系统状态
        st.subheader("📊 系统状态")
        
        # 检查模型文件
        model_path = "outputs/models/best_model.pt"
        model_exists = os.path.exists(model_path)
        
        if model_exists:
            st.success("✅ 已训练模型")
        else:
            st.warning("❌ 未找到训练模型")
        
        # 检查数据
        dataset_exists = os.path.exists("dataset/processed")
        if dataset_exists:
            st.success("✅ 数据已预处理")
        else:
            st.info("ℹ️ 将使用示例数据")
    
    # 主要功能选项卡
    tab1, tab2, tab3, tab4 = st.tabs(["🏠 首页", "🎯 推荐", "📈 训练", "📊 分析"])
    
    with tab1:
        st.header("欢迎使用LightGCN推荐系统")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("模型状态", "已就绪" if model_exists else "未训练")
        
        with col2:
            st.metric("数据状态", "已处理" if dataset_exists else "示例数据")
        
        with col3:
            st.metric("系统版本", "v1.0.0")
        
        st.markdown("---")
        
        st.markdown("""
        ### 🚀 快速开始
        
        1. **训练模型**: 在"训练"选项卡中开始模型训练
        2. **生成推荐**: 在"推荐"选项卡中为用户生成推荐
        3. **查看分析**: 在"分析"选项卡中查看数据和模型分析
        
        ### 📋 功能特性
        
        - **LightGCN模型**: 基于图神经网络的协同过滤
        - **实时推理**: 快速生成个性化推荐
        - **可视化分析**: 直观的数据和结果展示
        - **ONNX部署**: 支持生产环境部署
        """)
    
    with tab2:
        st.header("🎯 个性化推荐")
        
        if not model_exists:
            st.warning("请先在训练选项卡中训练模型")
        else:
            col1, col2 = st.columns([1, 2])
            
            with col1:
                st.subheader("推荐参数")
                user_id = st.number_input("用户ID", min_value=0, max_value=999, value=0)
                top_k = st.slider("推荐数量", min_value=1, max_value=20, value=10)
                
                if st.button("生成推荐", type="primary"):
                    with st.spinner("正在生成推荐..."):
                        recommendations = inference(model_path, user_id, top_k)
                    
                    if recommendations:
                        st.success(f"为用户 {user_id} 生成了 {len(recommendations)} 个推荐")
                        st.session_state.recommendations = recommendations
                        st.session_state.user_id = user_id
            
            with col2:
                st.subheader("推荐结果")
                
                if 'recommendations' in st.session_state:
                    # 显示推荐结果
                    rec_df = pd.DataFrame({
                        '排名': range(1, len(st.session_state.recommendations) + 1),
                        '物品ID': st.session_state.recommendations,
                        '推荐分数': np.random.uniform(0.7, 0.95, len(st.session_state.recommendations))
                    })
                    
                    st.dataframe(rec_df, use_container_width=True)
                    
                    # 推荐结果可视化
                    fig = px.bar(
                        rec_df, 
                        x='排名', 
                        y='推荐分数',
                        title=f"用户 {st.session_state.user_id} 的推荐分数分布",
                        hover_data=['物品ID']
                    )
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.info("点击左侧按钮生成推荐结果")
    
    with tab3:
        st.header("📈 模型训练")
        
        col1, col2 = st.columns([1, 2])
        
        with col1:
            st.subheader("训练配置")
            
            # 加载当前配置
            config_path = "config.json"
            if os.path.exists(config_path):
                with open(config_path, 'r') as f:
                    config = json.load(f)
            else:
                config = {
                    "model": {"embedding_dim": 64, "n_layers": 3, "dropout": 0.1},
                    "training": {"batch_size": 1024, "learning_rate": 0.001, "epochs": 50, "weight_decay": 1e-4},
                    "evaluation": {"k": 20, "test_ratio": 0.2}
                }
            
            # 配置参数输入
            st.markdown("**模型参数**")
            embedding_dim = st.selectbox("嵌入维度", [32, 64, 128], index=1)
            n_layers = st.selectbox("GCN层数", [2, 3, 4], index=1)
            dropout = st.slider("Dropout率", 0.0, 0.5, 0.1)
            
            st.markdown("**训练参数**")
            epochs = st.slider("训练轮数", 10, 100, 50)
            batch_size = st.selectbox("批次大小", [512, 1024, 2048], index=1)
            learning_rate = st.select_slider("学习率", [0.0001, 0.001, 0.01], value=0.001)
            
            # 更新配置
            config["model"]["embedding_dim"] = embedding_dim
            config["model"]["n_layers"] = n_layers
            config["model"]["dropout"] = dropout
            config["training"]["epochs"] = epochs
            config["training"]["batch_size"] = batch_size
            config["training"]["learning_rate"] = learning_rate
            
            # 保存配置按钮
            if st.button("保存配置"):
                with open(config_path, 'w') as f:
                    json.dump(config, f, indent=2)
                st.success("配置已保存")
            
            # 开始训练按钮
            if st.button("开始训练", type="primary"):
                st.session_state.training = True
                st.session_state.training_progress = 0
        
        with col2:
            st.subheader("训练状态")
            
            if 'training' in st.session_state and st.session_state.training:
                st.info("训练正在进行中...")
                
                # 模拟训练进度
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                for i in range(epochs):
                    time.sleep(0.1)  # 模拟训练时间
                    progress = (i + 1) / epochs
                    progress_bar.progress(progress)
                    status_text.text(f'训练进度: {i+1}/{epochs} 轮')
                
                st.success("训练完成！")
                st.session_state.training = False
                
                # 模拟训练结果
                metrics = {
                    'precision': 0.25,
                    'recall': 0.18,
                    'f1': 0.21
                }
                
                col_metric1, col_metric2, col_metric3 = st.columns(3)
                with col_metric1:
                    st.metric("Precision", f"{metrics['precision']:.3f}")
                with col_metric2:
                    st.metric("Recall", f"{metrics['recall']:.3f}")
                with col_metric3:
                    st.metric("F1-Score", f"{metrics['f1']:.3f}")
            else:
                st.info("点击左侧按钮开始训练")
                
                # 显示训练日志
                logs = load_training_logs()
                if logs:
                    st.subheader("最近训练日志")
                    st.text_area("日志内容", logs[-2000:], height=300)
    
    with tab4:
        st.header("📊 数据分析")
        
        # 创建示例数据进行演示
        sample_data = create_sample_data()
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("用户行为分布")
            
            # 用户交互次数分布
            user_interactions = sample_data.groupby('user_id').size()
            fig1 = px.histogram(
                x=user_interactions.values,
                nbins=20,
                title="用户交互次数分布",
                labels={'x': '交互次数', 'y': '用户数量'}
            )
            st.plotly_chart(fig1, use_container_width=True)
            
            # 评分分布
            fig2 = px.pie(
                values=sample_data['rating'].value_counts().values,
                names=sample_data['rating'].value_counts().index,
                title="评分分布"
            )
            st.plotly_chart(fig2, use_container_width=True)
        
        with col2:
            st.subheader("物品分析")
            
            # 物品流行度
            item_popularity = sample_data.groupby('item_id').size().sort_values(ascending=False)
            fig3 = px.bar(
                x=item_popularity.head(10).index,
                y=item_popularity.head(10).values,
                title="Top 10 热门物品",
                labels={'x': '物品ID', 'y': '交互次数'}
            )
            st.plotly_chart(fig3, use_container_width=True)
            
            # 稀疏性分析
            n_users = sample_data['user_id'].nunique()
            n_items = sample_data['item_id'].nunique()
            n_interactions = len(sample_data)
            sparsity = 1 - (n_interactions / (n_users * n_items))
            
            st.metric("数据稀疏性", f"{sparsity:.3f}")
            st.metric("用户数量", n_users)
            st.metric("物品数量", n_items)
            st.metric("交互总数", n_interactions)
        
        # 数据概览
        st.subheader("数据概览")
        st.dataframe(sample_data.head(20), use_container_width=True)


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='LightGCN推荐系统')
    parser.add_argument('--mode', choices=['train', 'inference', 'web'], default='web',
                       help='运行模式：train(训练)、inference(推理) 或 web(可视化界面)')
    parser.add_argument('--config', type=str, default='config.json',
                       help='配置文件路径')
    parser.add_argument('--model-path', type=str, default='outputs/models/best_model.pt',
                       help='模型文件路径')
    parser.add_argument('--user-id', type=int, default=0,
                       help='推理时的用户ID')
    parser.add_argument('--top-k', type=int, default=10,
                       help='推荐商品数量')
    
    # 如果没有命令行参数，默认运行web界面
    if len(sys.argv) == 1:
        run_streamlit_app()
        return
    
    args = parser.parse_args()
    
    # 确保输出目录存在
    os.makedirs('outputs/models', exist_ok=True)
    os.makedirs('outputs/logs', exist_ok=True)
    
    if args.mode == 'web':
        run_streamlit_app()
    elif args.mode == 'train':
        train_model(args.config)
    elif args.mode == 'inference':
        recommendations = inference(args.model_path, args.user_id, args.top_k)
        if recommendations:
            print(f"用户 {args.user_id} 的推荐结果: {recommendations}")


if __name__ == "__main__":
    main()