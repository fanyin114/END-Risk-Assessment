import streamlit as st  
import xgboost as xgb  
import numpy as np  
import pandas as pd  
import joblib  
import logging  
import random  

# 设置随机种子  
np.random.seed(0)  
random.seed(0)  

# 设置页面配置  
st.set_page_config(  
    page_title="END Risk Assessment System",  
    layout="wide",  
    initial_sidebar_state="collapsed"  
)  

# 自定义CSS样式  
st.markdown("""
<style>
/* 英文标题居中 */
.subtitle {
    text-align: center !important;
    width: 100% !important;
    font-size: 1.2rem !important;
    font-weight: 600 !important;
    margin: 1rem auto !important;
    color: #1e40af !important;
}

/* 评估结果标题居中 */
.result-title {
    text-align: center !important;
    font-size: 1.3rem;
    font-weight: 600;
    color: #0f172a;
    margin: 1.2rem auto;
    padding-bottom: 0.6rem;
    border-bottom: 2px solid #e2e8f0;
    max-width: 600px;
}

/* 发生概率和风险等级居中 */
.result-section p {
    text-align: center !important;
    margin: 1rem auto !important;
    font-size: 1.1rem !important;
}

/* 风险描述文本居中 */
.risk-description {
    text-align: center !important;
    margin: 1rem auto !important;
    font-size: 1.1rem !important;
    line-height: 1.6 !important;
}

/* 范围说明靠左 */
.normal-range {
    text-align: left !important;
    font-size: 0.85rem !important;
    color: #64748b !important;
    margin: 0.2rem 0 0.5rem 0 !important;
    padding-left: 0.5rem !important;
}

/* 按钮样式 */
.stButton > button {
    background: linear-gradient(145deg, #3b82f6, #1e40af) !important;
    color: white !important;
    padding: 0.6rem 2.5rem !important;
    border-radius: 6px !important;
    border: none !important;
    font-size: 1.1rem !important;
    font-weight: 600 !important;
    min-width: 250px !important;
    height: 2.8rem !important;
    box-shadow: 0 2px 4px rgba(59, 130, 246, 0.2) !important;
    transition: all 0.2s ease !important;
    display: block !important;
    margin: 1.5rem auto !important;
}

/* 高风险和低风险样式 */
.high-risk,
.low-risk {
    text-align: center !important;
    display: inline-block !important;
    padding: 0.3rem 1rem !important;
    border-radius: 4px !important;
    font-weight: 600 !important;
}

.high-risk {
    color: #dc2626 !important;
    background: #fee2e2 !important;
}

.low-risk {
    color: #166534 !important;
    background: #dcfce7 !important;
}

/* 响应式样式 */
@media (max-width: 768px) {
    .subtitle {
        font-size: 1rem !important;
    }
    
    .result-title {
        font-size: 1.2rem !important;
    }
    
    .normal-range {
        font-size: 0.8rem !important;
    }
    
    .stButton > button {
        min-width: 220px !important;
        padding: 0.6rem 1.2rem !important;
        font-size: 1rem !important;
    }
}
</style>
""", unsafe_allow_html=True)

# 设置日志  
logging.basicConfig(level=logging.DEBUG)  

# 加载模型  
@st.cache_resource  
def load_model():  
    try:  
        model = joblib.load('models/XGBOOST_model1113.pkl')  
        if isinstance(model, xgb.XGBClassifier):  
            model.random_state = 0  
        logging.info("模型加载成功")  
        return model  
    except Exception as e:  
        logging.error(f"模型加载失败: {str(e)}")  
        return None  

# 页面标题  
st.markdown("<h1 class='title'>早期神经功能恶化风险评估系统</h1>", unsafe_allow_html=True)  
st.markdown("<p class='subtitle'>Early Neurological Deterioration Risk Assessment System</p>", unsafe_allow_html=True)  

# 创建两列布局  
col1, col2 = st.columns(2)  

# 左列内容  
with col1:  
    # TOAST-LAA  
    st.markdown("<div class='input-section'>", unsafe_allow_html=True)  
    st.markdown("<div class='input-label'>大动脉粥样硬化型 / TOAST-LAA</div>", unsafe_allow_html=True)  
    toast_laa = st.radio(  
        "",  
        ["是 / Yes", "否 / No"],  
        index=1,  
        key="toast_laa",  
        label_visibility="collapsed"  
    )  
    st.markdown("</div>", unsafe_allow_html=True)  
    
    # NIHSS  
    st.markdown("<div class='input-section'>", unsafe_allow_html=True)  
    st.markdown("<div class='input-label'>NIHSS评分 / NIHSS Score</div>", unsafe_allow_html=True)  
    nihss = st.number_input(  
        "",  
        min_value=0.0,  
        max_value=42.0,  
        value=0.0,  
        step=1.0,  
        key="nihss",  
        label_visibility="collapsed"  
    )  
    st.markdown("<p class='normal-range'>范围 / Normal Range: 0-42</p>", unsafe_allow_html=True)  
    st.markdown("</div>", unsafe_allow_html=True)  

    # SBP  
    st.markdown("<div class='input-section'>", unsafe_allow_html=True)  
    st.markdown("<div class='input-label'>收缩压 / Systolic Blood Pressure</div>", unsafe_allow_html=True)  
    sbp = st.number_input(  
        "",  
        min_value=60.0,  
        max_value=300.0,  
        value=60.0,  
        step=1.0,  
        key="sbp",  
        label_visibility="collapsed"  
    )  
    st.markdown("<p class='normal-range'>范围 / Normal Range: 60-300 mmHg</p>", unsafe_allow_html=True)  
    st.markdown("</div>", unsafe_allow_html=True)  

    
# 右列内容  
with col2:  
    # IAS  
    st.markdown("<div class='input-section'>", unsafe_allow_html=True)  
    st.markdown("<div class='input-label'>颅内动脉狭窄≥50% / Intracranial Arterial Stenosis≥50%</div>", unsafe_allow_html=True)  
    ias = st.radio(  
        "",  
        ["是 / Yes", "否 / No"],  
        index=1,  
        key="ias",  
        label_visibility="collapsed"  
    )  
    st.markdown("</div>", unsafe_allow_html=True)

    # NEUT  
    st.markdown("<div class='input-section'>", unsafe_allow_html=True)  
    st.markdown("<div class='input-label'>中性粒细胞计数 / Neutrophil Count</div>", unsafe_allow_html=True)  
    neut = st.number_input(  
        "",  
        min_value=2.0,  
        max_value=30.0,  
        value=2.0,  
        step=0.1,  
        format="%.1f",  
        key="neut",  
        label_visibility="collapsed"  
    )  
    st.markdown("<p class='normal-range'>范围 / Normal Range: 2.0-30.0 ×10^9/L</p>", unsafe_allow_html=True)  
    st.markdown("</div>", unsafe_allow_html=True)  

    
    # RDW  
    st.markdown("<div class='input-section'>", unsafe_allow_html=True)  
    st.markdown("<div class='input-label'>红细胞分布宽度 / Red Cell Distribution Width</div>", unsafe_allow_html=True)  
    rdw = st.number_input(  
        "",  
        min_value=10.0,  
        max_value=60.0,  
        value=10.0,  
        step=0.1,  
        format="%.1f",  
        key="rdw",  
        label_visibility="collapsed"  
    )  
    st.markdown("<p class='normal-range'>范围 / Normal Range: 10-60 fL</p>", unsafe_allow_html=True)  
    st.markdown("</div>", unsafe_allow_html=True)  

    

    
    
# 计算按钮  
# 计算按钮（在列布局之外）  
# 计算按钮  
if st.button("计算风险评分 / Calculate Risk Score", key="calculate_button"):  
    try:  
        # 处理输入数据  
        toast_laa_value = 1 if toast_laa == "是 / Yes" else 0  
        ias_value = 1 if ias == "是 / Yes" else 0  
        
        # 创建特征DataFrame  
        features = pd.DataFrame(columns=['NIHSS', 'SBP', 'NEUT', 'RDW', 'TOAST-LAA_1', 'IAS_1'])  
        features.loc[0] = [nihss, sbp, neut, rdw, toast_laa_value, ias_value]  

        # 加载模型并预测  
        model = load_model()  
        if model is not None:  
            if hasattr(model, 'random_state'):  
                model.random_state = 0  
            risk_prob = float(model.predict_proba(features)[0][1])  
            risk_percentage = risk_prob * 100  
            risk_level = "高风险 / High Risk" if risk_percentage >= 29 else "低风险 / Low Risk"  
            
            # 显示结果  
            st.markdown("<div class='result-section'>", unsafe_allow_html=True)  
            st.markdown("<div class='result-title'>评估结果 / Assessment Result</div>", unsafe_allow_html=True)  
            st.markdown(f"<p>发生概率 / Probability: <span class='{'high-risk' if risk_percentage >= 29 else 'low-risk'}'>{risk_percentage:.2f}%</span></p>", unsafe_allow_html=True)  
            st.markdown(f"<p>风险等级 / Risk Level: <span class='{'high-risk' if risk_percentage >= 29 else 'low-risk'}'>{risk_level}</span></p>", unsafe_allow_html=True)  
            
            # 风险描述  
             
            if risk_percentage >= 29:  
                st.markdown("""  
                    <p class='risk-description'>  
                    发生早期神经功能恶化的风险较高<br>  
                    High Risk of Early Neurological Deterioration  
                    </p>  
                """, unsafe_allow_html=True)  
            else:  
                st.markdown("""  
                    <p class='risk-description'>  
                    发生早期神经功能恶化的风险较低<br>  
                    Low Risk of Early Neurological Deterioration  
                    </p>  
                """, unsafe_allow_html=True)  
            st.markdown("</div>", unsafe_allow_html=True)  

    except Exception as e:  
        st.error(f"计算过程出错 / Calculation Error: {str(e)}")
