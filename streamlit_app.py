import streamlit as st
import xgboost as xgb
import numpy as np
import pandas as pd
import joblib

# 页面配置
st.set_page_config(
    page_title="早期神经功能恶化风险评估系统",
    layout="wide"
)

# 加载模型
@st.cache_resource
def load_model():
    return joblib.load('models/XGBOOST_model1113.pkl')

try:
    model = load_model()
except Exception as e:
    st.error(f"模型加载失败: {str(e)}")

# 模型特征定义
MODEL_FEATURES = ['NIHSS', 'SBP', 'NEUT', 'RDW', 'TOAST-LAA_1', 'IAS_1']

def get_risk_level(probability):
    """根据概率返回风险等级和详细信息"""
    prob_percentage = probability * 100
    
    if prob_percentage < 29:
        return "低风险", "发生早期神经功能恶化的风险较低"
    else:
        return "高风险", "发生早期神经功能恶化的风险较高"

# 页面标题
st.title("早期神经功能恶化风险评估系统")
st.subheader("Early Neurological Deterioration Risk Assessment System")

# 创建两列布局
col1, col2 = st.columns(2)

with col1:
    # TOAST-LAA
    toast_laa = st.radio(
        "大动脉粥样硬化型 / TOAST-LAA",
        options=["否 / No", "是 / Yes"],
        index=0
    )
    
    # IAS
    ias = st.radio(
        "颅内动脉狭窄≥50% / Intracranial Arterial Stenosis≥50%",
        options=["否 / No", "是 / Yes"],
        index=0
    )

with col2:
    # NIHSS
    nihss = st.number_input(
        "NIHSS评分 / NIHSS Score",
        min_value=0,
        max_value=42,
        value=0,
        help="正常范围：0-42分"
    )
    
    # SBP
    sbp = st.number_input(
        "收缩压 / Systolic Blood Pressure (mmHg)",
        min_value=0,
        max_value=300,
        value=120,
        help="正常范围：90-140 mmHg"
    )
    
    # NEUT
    neut = st.number_input(
        "中性粒细胞计数 / Neutrophil Count (×10^9/L)",
        min_value=0.0,
        max_value=20.0,
        value=4.0,
        step=0.1,
        help="正常范围：2-7 ×10^9/L"
    )
    
    # RDW
    rdw = st.number_input(
        "红细胞分布宽度 / Red Cell Distribution Width (%)",
        min_value=0.0,
        max_value=60.0,
        value=13.0,
        step=0.1,
        help="正常范围：10-60%"
    )

# 计算按钮
if st.button("计算风险评分 / Calculate Risk Score"):
    try:
        # 准备特征数据
        features = pd.DataFrame(columns=MODEL_FEATURES)
        features.loc[0] = [0] * len(MODEL_FEATURES)
        
        # 填充数据
        features.loc[0, 'NIHSS'] = nihss
        features.loc[0, 'SBP'] = sbp
        features.loc[0, 'NEUT'] = neut
        features.loc[0, 'RDW'] = rdw
        features.loc[0, 'TOAST-LAA_1'] = 1 if toast_laa == "是 / Yes" else 0
        features.loc[0, 'IAS_1'] = 1 if ias == "是 / Yes" else 0

        # 预测
        risk_prob = float(model.predict_proba(features)[0][1])
        risk_percentage = risk_prob * 100
        risk_level, risk_description = get_risk_level(risk_prob)

        # 显示结果
        st.markdown("---")
        st.subheader("评估结果 / Assessment Result")
        
        col3, col4 = st.columns(2)
        
        with col3:
            st.metric("发生概率 / Probability", f"{risk_percentage:.2f}%")
        
        with col4:
            st.metric("风险等级 / Risk Level", risk_level)
        
        st.info(risk_description)

    except Exception as e:
        st.error(f"预测过程出错: {str(e)}")