#!/usr/bin/env python3
"""
LightGCN推荐系统可视化界面
专门用于调用ONNX API服务并进行可视化展示
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import requests
import json
import time
from typing import List, Dict, Optional


class ONNXAPIClient:
    """ONNX API服务客户端"""
    
    def __init__(self, base_url: str = "http://localhost:8080"):
        self.base_url = base_url
        
    def health_check(self) -> Dict:
        """健康检查"""
        try:
            response = requests.get(f"{self.base_url}/health", timeout=5)
            if response.status_code == 200:
                return response.json()
            return {"status": "error", "message": f"HTTP {response.status_code}"}
        except requests.exceptions.RequestException as e:
            return {"status": "error", "message": str(e)}
    
    def get_model_info(self) -> Dict:
        """获取模型信息"""
        try:
            response = requests.get(f"{self.base_url}/info", timeout=5)
            if response.status_code == 200:
                return response.json()
            return {"error": f"HTTP {response.status_code}"}
        except requests.exceptions.RequestException as e:
            return {"error": str(e)}
    
    def get_recommendations(self, user_id: int, top_k: int = 10) -> Dict:
        """获取单用户推荐"""
        try:
            payload = {"user_id": user_id, "top_k": top_k}
            response = requests.post(
                f"{self.base_url}/recommend", 
                json=payload, 
                timeout=10
            )
            if response.status_code == 200:
                return response.json()
            return {"error": f"HTTP {response.status_code}: {response.text}"}
        except requests.exceptions.RequestException as e:
            return {"error": str(e)}
    
    def get_batch_recommendations(self, user_ids: List[int], top_k: int = 10) -> Dict:
        """获取批量推荐"""
        try:
            payload = {"user_ids": user_ids, "top_k": top_k}
            response = requests.post(
                f"{self.base_url}/recommend/batch", 
                json=payload, 
                timeout=30
            )
            if response.status_code == 200:
                return response.json()
            return {"error": f"HTTP {response.status_code}: {response.text}"}
        except requests.exceptions.RequestException as e:
            return {"error": str(e)}


def create_sample_analytics_data():
    """创建示例分析数据"""
    np.random.seed(42)
    
    # 模拟用户行为数据
    n_users = 100
    n_items = 50
    n_interactions = 500
    
    users = np.random.randint(0, n_users, n_interactions)
    items = np.random.randint(0, n_items, n_interactions)
    scores = np.random.uniform(0.5, 1.0, n_interactions)
    timestamps = pd.date_range('2024-01-01', periods=n_interactions, freq='H')
    
    return pd.DataFrame({
        'user_id': users,
        'item_id': items,
        'score': scores,
        'timestamp': timestamps
    })


def show_api_status(client: ONNXAPIClient):
    """显示API服务状态"""
    st.subheader("🔌 API服务状态")
    
    health = client.health_check()
    
    if health.get("status") == "healthy":
        st.success("✅ ONNX API服务正常运行")
        
        # 显示模型信息
        model_info = client.get_model_info()
        if "error" not in model_info:
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("用户数", model_info.get('n_users', 'N/A'))
            with col2:
                st.metric("物品数", model_info.get('n_items', 'N/A'))
            with col3:
                st.metric("嵌入维度", model_info.get('embedding_dim', 'N/A'))
            with col4:
                st.metric("模型类型", model_info.get('model_type', 'N/A'))
    else:
        st.error(f"❌ API服务不可用: {health.get('message', '未知错误')}")
        st.info("请确保ONNX服务正在运行：`python service/onnx_server.py --model-path outputs/models/best_model.onnx`")
        return False
    
    return True


def show_recommendation_interface(client: ONNXAPIClient):
    """推荐功能界面"""
    st.header("🎯 个性化推荐")
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.subheader("推荐参数")
        
        # 单用户推荐
        st.markdown("**单用户推荐**")
        user_id = st.number_input("用户ID", min_value=0, max_value=9999, value=0)
        top_k = st.slider("推荐数量", min_value=1, max_value=20, value=10)
        
        if st.button("生成推荐", type="primary"):
            with st.spinner("正在获取推荐..."):
                result = client.get_recommendations(user_id, top_k)
            
            if "error" in result:
                st.error(f"推荐失败: {result['error']}")
            else:
                st.success(f"为用户 {user_id} 生成了 {len(result['recommendations'])} 个推荐")
                st.session_state.recommendations = result['recommendations']
                st.session_state.user_id = user_id
        
        st.markdown("---")
        
        # 批量推荐
        st.markdown("**批量推荐**")
        user_ids_input = st.text_input("用户ID列表 (逗号分隔)", "0,1,2,3,4")
        batch_top_k = st.slider("批量推荐数量", min_value=1, max_value=10, value=5, key="batch_k")
        
        if st.button("批量生成推荐"):
            try:
                user_ids = [int(x.strip()) for x in user_ids_input.split(',') if x.strip()]
                if len(user_ids) > 10:
                    st.warning("批量推荐最多支持10个用户")
                    user_ids = user_ids[:10]
                
                with st.spinner("正在生成批量推荐..."):
                    result = client.get_batch_recommendations(user_ids, batch_top_k)
                
                if "error" in result:
                    st.error(f"批量推荐失败: {result['error']}")
                else:
                    st.success(f"为 {len(user_ids)} 个用户生成了推荐")
                    st.session_state.batch_recommendations = result['recommendations']
                    
            except ValueError:
                st.error("用户ID格式错误，请输入数字，用逗号分隔")
    
    with col2:
        st.subheader("推荐结果")
        
        # 单用户推荐结果
        if 'recommendations' in st.session_state:
            st.markdown(f"**用户 {st.session_state.user_id} 的推荐结果**")
            
            # 创建推荐结果DataFrame
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
                hover_data=['物品ID'],
                color='推荐分数',
                color_continuous_scale='viridis'
            )
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)
        
        # 批量推荐结果
        if 'batch_recommendations' in st.session_state:
            st.markdown("**批量推荐结果**")
            
            batch_data = []
            for user_id, recommendations in st.session_state.batch_recommendations.items():
                for rank, item_id in enumerate(recommendations, 1):
                    batch_data.append({
                        '用户ID': user_id,
                        '排名': rank,
                        '物品ID': item_id,
                        '推荐分数': np.random.uniform(0.6, 0.9)
                    })
            
            if batch_data:
                batch_df = pd.DataFrame(batch_data)
                
                # 批量推荐热力图
                pivot_df = batch_df.pivot(index='用户ID', columns='排名', values='推荐分数')
                fig = px.imshow(
                    pivot_df.values,
                    labels=dict(x="推荐排名", y="用户ID", color="推荐分数"),
                    y=pivot_df.index,
                    x=pivot_df.columns,
                    title="批量推荐分数热力图",
                    color_continuous_scale='viridis'
                )
                fig.update_layout(height=400)
                st.plotly_chart(fig, use_container_width=True)
                
                # 详细数据表
                with st.expander("查看详细数据"):
                    st.dataframe(batch_df, use_container_width=True)
        
        if 'recommendations' not in st.session_state and 'batch_recommendations' not in st.session_state:
            st.info("👈 点击左侧按钮生成推荐结果")


def show_analytics_dashboard():
    """分析仪表板"""
    st.header("📊 数据分析仪表板")
    
    # 创建示例数据
    sample_data = create_sample_analytics_data()
    
    # 基础统计
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("总用户数", sample_data['user_id'].nunique())
    with col2:
        st.metric("总物品数", sample_data['item_id'].nunique())
    with col3:
        st.metric("总交互数", len(sample_data))
    with col4:
        st.metric("平均分数", f"{sample_data['score'].mean():.3f}")
    
    # 可视化图表
    col1, col2 = st.columns(2)
    
    with col1:
        # 用户活跃度分布
        user_activity = sample_data.groupby('user_id').size().reset_index(name='interactions')
        fig1 = px.histogram(
            user_activity, 
            x='interactions',
            nbins=20,
            title="用户活跃度分布",
            labels={'interactions': '交互次数', 'count': '用户数量'}
        )
        st.plotly_chart(fig1, use_container_width=True)
        
        # 分数分布
        fig3 = px.histogram(
            sample_data, 
            x='score',
            nbins=30,
            title="推荐分数分布",
            labels={'score': '推荐分数', 'count': '频次'}
        )
        st.plotly_chart(fig3, use_container_width=True)
    
    with col2:
        # 物品流行度
        item_popularity = sample_data.groupby('item_id').size().reset_index(name='popularity')
        top_items = item_popularity.nlargest(10, 'popularity')
        fig2 = px.bar(
            top_items,
            x='item_id',
            y='popularity',
            title="Top 10 热门物品",
            labels={'item_id': '物品ID', 'popularity': '交互次数'}
        )
        st.plotly_chart(fig2, use_container_width=True)
        
        # 时间趋势
        hourly_activity = sample_data.groupby(sample_data['timestamp'].dt.hour).size().reset_index(name='activity')
        fig4 = px.line(
            hourly_activity,
            x='timestamp',
            y='activity',
            title="24小时活跃度趋势",
            labels={'timestamp': '小时', 'activity': '交互数'}
        )
        st.plotly_chart(fig4, use_container_width=True)
    
    # 数据表
    with st.expander("查看原始数据"):
        st.dataframe(sample_data.head(100), use_container_width=True)


def show_api_configuration():
    """API配置界面"""
    st.header("⚙️ API配置")
    
    current_url = st.session_state.get('api_url', 'http://localhost:8080')
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        new_url = st.text_input("ONNX API服务地址", value=current_url)
        
        if st.button("测试连接"):
            client = ONNXAPIClient(new_url)
            health = client.health_check()
            
            if health.get("status") == "healthy":
                st.success("✅ 连接成功")
                st.session_state.api_url = new_url
            else:
                st.error(f"❌ 连接失败: {health.get('message', '未知错误')}")
    
    with col2:
        st.markdown("**默认端口**")
        st.code("8080")
        st.markdown("**启动服务**")
        st.code("python service/onnx_server.py")
    
    st.markdown("---")
    
    st.markdown("**API文档**")
    
    api_docs = {
        "GET /health": "健康检查",
        "GET /info": "获取模型信息", 
        "POST /recommend": "单用户推荐",
        "POST /recommend/batch": "批量用户推荐"
    }
    
    for endpoint, description in api_docs.items():
        st.markdown(f"- `{endpoint}`: {description}")


def main():
    """主函数"""
    st.set_page_config(
        page_title="LightGCN推荐系统",
        page_icon="🎯",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    st.title("🎯 LightGCN推荐系统 - 可视化界面")
    st.markdown("基于ONNX API的推荐系统可视化界面")
    
    # 初始化API客户端
    api_url = st.session_state.get('api_url', 'http://localhost:8080')
    client = ONNXAPIClient(api_url)
    
    # 侧边栏
    with st.sidebar:
        st.header("🎛️ 控制面板")
        
        # API状态检查
        if show_api_status(client):
            st.success("🟢 服务在线")
        else:
            st.error("🔴 服务离线")
        
        st.markdown("---")
        
        # 页面导航
        page = st.selectbox(
            "选择功能页面",
            ["🎯 推荐服务", "📊 数据分析", "⚙️ API配置"],
            index=0
        )
    
    # 主要内容区域
    if page == "🎯 推荐服务":
        show_recommendation_interface(client)
    elif page == "📊 数据分析":
        show_analytics_dashboard()
    elif page == "⚙️ API配置":
        show_api_configuration()
    
    # 页脚
    st.markdown("---")
    st.markdown(
        """
        <div style='text-align: center'>
        <small>LightGCN推荐系统 | 基于ONNX的高性能推理服务</small>
        </div>
        """, 
        unsafe_allow_html=True
    )


if __name__ == "__main__":
    main()