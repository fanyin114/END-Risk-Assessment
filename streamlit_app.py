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

/* 页面容器样式 */  
.block-container {  
    padding: 3rem !important;  
    max-width: 1200px !important;  
}  

/* 标题样式 */  
.title {  
    color: #2c3e50;  
    font-size: 2rem;  
    font-weight: 700;  
    text-align: center;  
    margin: 2rem 0 1rem 0;  
    line-height: 1.6;  
    padding: 0 1rem;  
    word-wrap: break-word;  
}  

.subtitle {  
    color: #5a6c7d;  
    font-size: 1.4rem;  
    text-align: center;  
    margin-bottom: 3rem;  
    line-height: 1.6;  
    padding: 0 1rem;  
}  

/* 输入区域样式 */  
.input-group {  
    background: #f8fafc;  
    border: 2px solid #4a90e2;  
    border-radius: 20px;  
    padding: 2rem;  
    margin: 1.6 rem 0;  
    box-shadow: 0 2px 4px rgba(0, 0, 0, 0.05);  
    width: 100%;  
    display: block;  
}  

.input-label {  
    color: #2c3e50;  
    font-size: 1.2rem;  
    font-weight: 600;  
    margin-bottom: 1rem;  
    display: block;  
}  

/* 输入控件样式 */  
.stNumberInput > div > div > input {  
    width: 100% !important;  
    min-width: 300px !important;  
    padding: 0.8rem 1rem !important;  
    background: white !important;  
    border: 1px solid #e2e8f0 !important;  
    border-radius: 15px !important;  
    font-size: 1rem !important;  
    box-shadow: 0 1px 3px rgba(0,0,0,0.1) !important;  
}  

/* Radio按钮样式 */  
.stRadio > div {  
    padding: 0.5rem 0 !important;  
}  

.stRadio > div > div > label {  
    padding: 0.8rem 1.2rem !important;  
    margin: 0.3rem 1rem 0.3rem 0 !important;  
    border-radius: 15px !important;  
    background: white !important;  
    border: 1px solid #e2e8f0 !important;  
    transition: all 0.2s ease !important;  
}  

.stRadio > div > div > label[data-baseweb="radio"] > div:first-child {  
    background-color: #4a90e2 !important;  
    border-color: #4a90e2 !important;  
}  

/* 计算按钮样式 */  
.stButton {  
    display: flex !important;  
    justify-content: center !important;  
    margin: 2.5rem auto !important;  
    width: 0% !important;  
}  

.stButton > button {  
    background: linear-gradient(145deg, #1a73e8, #1557b0) !important;  
    color: white !important;  
    padding: 1rem 4rem !important;  
    border-radius: 30px !important;  
    border: none !important;  
    font-size: 1.8rem !important;  
    font-weight: 800 !important;  
    box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1) !important;  
    transition: all 0.3s ease !important;  
    min-width: 400px !important;  
    max-width: 600px !important;  
    margin: 0 auto !important;  
    white-space: nowrap !important;  
    height: 3.5rem !important;  
    line-height: 1.5 !important;  
    display: flex !important;  
    align-items: center !important;  
    justify-content: center !important;  
}  

.stButton > button:hover {  
    transform: translateY(-2px) !important;  
    box-shadow: 0 6px 12px rgba(0, 0, 0, 0.15) !important;  
    background: linear-gradient(145deg, #1557b0, #1a73e8) !important;  
}  

.stButton > button:active {  
    transform: translateY(0) !important;  
    box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1) !important;  
}  

/* 结果显示样式 */  
.result-title {  
    font-size: 1.8rem;  
    font-weight: 700;  
    color: #1a365d;  
    text-align: center;  
    margin: 2rem 0;  
    padding-bottom: 1rem;  
    border-bottom: 2px solid #e2e8f0;  
}  

.result-grid {  
    display: grid;  
    grid-template-columns: 1fr;  
    gap: 2rem;  
    margin: 2rem 0;  
}  

.probability-container, .risk-level-container, .risk-description-container {  
    background: white;  
    border-radius: 12px;  
    padding: 2rem;  
    margin: 2rem 0;  
    box-shadow: 0 2px 4px rgba(0, 0, 0, 0.05);  
}  

.probability-value {  
    color: #e53e3e;  
    font-size: 2.5rem;  
    font-weight: 700;  
    text-align: center;  
    padding: 0.5rem 1rem;  
    background: #fff5f5;  
    border-radius: 8px;  
    border: 2px solid #fc8181;  
    display: inline-block;  
    margin: 0.5rem 0;  
}  

.risk-level-label {  
    font-size: 1.4rem;  
    color: #4a5568;  
    margin-bottom: 0.5rem;  
}  

.high-risk {  
    color: #e53e3e;  
    background: #fff5f5;  
    border: 2px solid #fc8181;  
}  

.low-risk {  
    color: #38a169;  
    background: #f0fff4;  
    border: 2px solid #9ae6b4;  
}  

.high-risk, .low-risk {  
    font-size: 1.6rem;  
    font-weight: 700;  
    text-align: left;  
    margin: 1rem 0;  
    padding: 0.5rem 1rem;  
    border-radius: 8px;  
}  

.risk-description {  
    font-size: 1.6rem;  
    color: #2d3748;  
    line-height: 1.8;  
    text-align: left;  
    padding: 1rem;  
    background: #f7fafc;  
    border-radius: 8px;  
}  

/* 辅助样式 */  
.normal-range {  
    color: #64748b;  
    font-size: 1.2rem;  
    margin-top: 0.5rem;  
    padding: 0.5rem 0;  
}  

/* 移除Streamlit默认样式 */  
div[data-testid="stMarkdownContainer"],  
div.stMarkdown,  
div.element-container,  
div[data-testid="stVerticalBlock"] > div {  
    background: transparent !important;  
    box-shadow: none !important;  
    border: none !important;  
}  

div[data-testid="stVerticalBlock"] {  
    gap: 2rem !important;  
}  

/* 输入区域样式 */  
.input-group {  
    background: #f8fafc;  
    border: 2px solid #4a90e2;  
    border-radius: 20px;  
    padding: 1.5rem;  /* 从2rem减小到1.5rem */  
    margin: 1rem 0;   /* 从1.6rem减小到1rem */  
    box-shadow: 0 2px 4px rgba(0, 0, 0, 0.05);  
    width: 100%;  
    display: block;  
}  

/* 输入标签间距调整 */  
.input-label {  
    color: #2c3e50;  
    font-size: 1.2rem;  
    font-weight: 600;  
    margin-bottom: 0.8rem;  /* 从1rem减小到0.8rem */  
    display: block;  
}  

/* 垂直间距调整 */  
div[data-testid="stVerticalBlock"] {  
    gap: 1rem !important;  /* 从2rem减小到1rem */  
}  

/* 结果显示区域间距调整 */  
.result-grid {  
    display: grid;  
    grid-template-columns: 1fr;  
    gap: 1.5rem;  /* 从2rem减小到1.5rem */  
    margin: 1.5rem 0;  /* 从2rem减小到1.5rem */  
}  

.probability-container, .risk-level-container, .risk-description-container {  
    background: white;  
    border-radius: 12px;  
    padding: 1.5rem;  /* 从2rem减小到1.5rem */  
    margin: 1.5rem 0;  /* 从2rem减小到1.5rem */  
    box-shadow: 0 2px 4px rgba(0, 0, 0, 0.05);  
}  

/* 响应式调整 */  
@media (max-width: 768px) {  
    .input-group {  
        padding: 1.4rem;  
    }  
    
    .input-label {  
        font-size: 1.4rem;  
    }  
    
    .stButton > button {  
        min-width: 300px !important;  
        padding: 1rem 2rem !important;  
        font-size: 1.4rem !important;  
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
