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
    /* 全局样式 */  
* {  
    margin: 0;  
    padding: 0;  
    box-sizing: border-box;  
}  

/* 输入框容器样式 */  
.stNumberInput {  
    width: 100% !important;  
    max-width: none !important;  
    margin: 0 !important;  
}

/* 标题样式 */  
.title {  
    color: #2c3e50;  
    font-size: 2rem;  
    font-weight: 700;  
    text-align: center;  
    margin-bottom: 0.5rem;  
}  

.subtitle {  
    color: #5a6c7d;  
    font-size: 1.2rem;  
    text-align: center;  
    margin-bottom: 2rem;  
}  

/* 输入区域样式 */  
.input-section {  
    margin-bottom: 1.5rem;  
    padding: 1.2rem;  
    border-radius: 12px;  
    background: linear-gradient(145deg, #ffffff, #f0f2f5);  
    box-shadow: 3px 3px 6px #d1d9e6,  
                -3px -3px 6px #ffffff;  
}  

/* 输入标签样式 */  
.input-label {  
    color: #2c3e50;  
    font-size: 1rem;  
    font-weight: 600;  
    margin-bottom: 0.8rem;  
}  

/* 输入框样式 */  
.stNumberInput > div > div > input {  
    width: 100% !important;  
    min-width: 300px !important;  
    padding: 0.8rem !important;  
    border: 1px solid #e2e8f0 !important;  
    border-radius: 8px !important;  
    background: white !important;  
    font-size: 1.1rem !important;  
    margin-left: 0 !important;  
    box-shadow: inset 2px 2px 5px #d1d9e6,  
                inset -2px -2px 5px #ffffff;  
}  

/* Radio按钮组样式 */  
.stRadio > div {  
    padding: 0.5rem;  
}  

.stRadio > div > div > label {  
    padding: 0.5rem 1rem;  
    margin: 0.3rem;  
    border-radius: 8px;  
    background: white;  
    box-shadow: 2px 2px 5px #d1d9e6,  
                -2px -2px 5px #ffffff;  
}  

/* 正常范围提示样式 */  
.normal-range {  
    margin-top: 0.8rem;  
    padding: 0.8rem;  
    color: #5a6c7d;  
    font-size: 0.85rem;  
    background: rgba(255,255,255,0.7);  
    border-radius: 8px;  
    border-left: 3px solid #1a73e8;  
}  

/* 计算按钮样式 */  
.stButton > button {  
    width: 80%;  
    max-width: 300px;  
    margin: 2rem auto;  
    padding: 0.8rem;  
    display: block;  
    background: linear-gradient(145deg, #1a73e8, #1557b0);  
    color: white;  
    border: none;  
    border-radius: 25px;  
    font-size: 1.1rem;  
    font-weight: 600;  
    box-shadow: 3px 3px 8px #d1d9e6,  
                -3px -3px 8px #ffffff;  
    transition: all 0.3s ease;  
}  

.stButton > button:hover {  
    transform: translateY(-2px);  
    box-shadow: 4px 4px 10px #d1d9e6,  
                -4px -4px 10px #ffffff;  
}  

/* 结果区域样式 */  
.result-section {  
    margin-top: 2rem;  
    padding: 1.5rem;  
    border-radius: 15px;  
    background: linear-gradient(145deg, #ffffff, #f0f2f5);  
    box-shadow: 8px 8px 16px #d1d9e6,  
                -8px -8px 16px #ffffff;  
}  

.result-title {  
    font-size: 1.4rem;  
    font-weight: 700;  
    text-align: center;  
    color: #2c3e50;  
    margin-bottom: 1.5rem;  
}  

.high-risk {  
    color: #dc3545;  
    font-weight: 700;  
}  

.low-risk {  
    color: #28a745;  
    font-weight: 700;  
}  

.risk-description {  
    margin-top: 1rem;  
    padding: 1rem;  
    background: rgba(255,255,255,0.8);  
    border-radius: 8px;  
    color: #2c3e50;  
    line-height: 1.6;  
}  

/* 移除Streamlit默认对齐 */  
.css-1kyxreq {  
    margin: 0 !important;  
    padding: 0 !important;  
}  

.css-ocqkz7 {  
    justify-content: flex-start !important;  
}  
/* 调整输入标签样式 */  
.input-label {  
    color: #2c3e50;  
    font-size: 1.1rem;  
    font-weight: 600;  
    margin-bottom: 0.8rem;  
    text-align: left !important;  
}  
/* 调整输入标签样式 */  
.input-label {  
    color: #2c3e50;  
    font-size: 1.1rem;  
    font-weight: 600;  
    margin-bottom: 0.8rem;  
    text-align: left !important;  
}  


/* 响应式布局 */  
@media (max-width: 768px) {  
    .css-1y4p8pa {  
        width: 100% !important;  
        padding: 0 !important;  
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
