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
    padding: 1.5rem !important;  
    max-width: 900px !important;  
    background: #f8fafc;  
}  

/* 修改标题样式，确保完整显示并添加立体效果 */  
.title {  
    color: #0f172a;  
    font-size: 1.6rem;  
    font-weight: 700;  
    text-align: center;  
    margin: 1.5rem 0 0.8rem 0;  
    line-height: 1.4;  
    padding: 0.8rem;  
    background: linear-gradient(120deg, #1e40af, #3b82f6);  
    -webkit-background-clip: text;  
    -webkit-text-fill-color: transparent;  
    text-shadow: 2px 2px 4px rgba(0, 0, 0, 0.1);  
    position: relative;  
    z-index: 1;  
}  

.title::before {  
    content: '';  
    position: absolute;  
    top: 0;  
    left: 0;  
    right: 0;  
    bottom: 0;  
    background: linear-gradient(to bottom, rgba(255,255,255,0.1), rgba(255,255,255,0.05));  
    z-index: -1;  
    border-radius: 8px;  
}  
  

/* 输入区域样式 */  
.input-group {  
    background: linear-gradient(to right, #ffffff, #f1f5f9);  
    border: 1px solid #e2e8f0;  
    border-radius: 8px;  
    padding: 1rem;  
    margin: 0.6rem 0;  
    box-shadow: 0 1px 2px rgba(0, 0, 0, 0.05);  
    transition: all 0.2s ease;  
}  

.input-group:hover {  
    border-color: #3b82f6;  
    box-shadow: 0 2px 4px rgba(59, 130, 246, 0.1);  
}  

.input-label {  
    color: #1e293b;  
    font-size: 0.95rem;  
    font-weight: 600;  
    margin-bottom: 0.4rem;  
    display: block;  
}  

/* 输入控件样式 */  
.stNumberInput > div > div > input {  
    width: 100% !important;  
    min-width: 220px !important;  
    padding: 0.5rem 0.7rem !important;  
    background: white !important;  
    border: 1px solid #cbd5e1 !important;  
    border-radius: 6px !important;  
    font-size: 0.9rem !important;  
    transition: all 0.2s ease !important;  
}  

.stNumberInput > div > div > input:focus {  
    border-color: #3b82f6 !important;  
    box-shadow: 0 0 0 2px rgba(59, 130, 246, 0.1) !important;  
}  

/* Radio按钮样式 */  
.stRadio > div {  
    padding: 0.3rem 0 !important;  
}  

.stRadio > div > div > label {  
    padding: 0.5rem 0.8rem !important;  
    margin: 0.15rem 0.6rem 0.15rem 0 !important;  
    border-radius: 6px !important;  
    background: white !important;  
    border: 1px solid #cbd5e1 !important;  
    transition: all 0.2s ease !important;  
}  

.stRadio > div > div > label[data-baseweb="radio"] > div:first-child {  
    background-color: #3b82f6 !important;  
    border-color: #3b82f6 !important;  
}  

/* 计算按钮样式 */  
.stButton {  
    display: flex !important;  
    justify-content: center !important;  
    margin: 1.5rem auto !important;  
}  

.stButton > button {  
    background: linear-gradient(135deg, #3b82f6, #1e40af) !important;  
    color: white !important;  
    padding: 0.6rem 2.5rem !important;  
    border-radius: 6px !important;  
    border: none !important;  
    font-size: 1.1rem !important;  
    font-weight: 600 !important;  
    box-shadow: 0 2px 4px rgba(59, 130, 246, 0.2) !important;  
    transition: all 0.2s ease !important;  
    min-width: 250px !important;  
    height: 2.8rem !important;  
}  

.stButton > button:hover {  
    transform: translateY(-1px) !important;  
    background: linear-gradient(135deg, #2563eb, #1e3a8a) !important;  
    box-shadow: 0 4px 6px rgba(59, 130, 246, 0.25) !important;  
}  

/* 结果显示样式 */  
.result-title {  
    font-size: 1.3rem;  
    font-weight: 600;  
    color: #0f172a;  
    text-align: center;  
    margin: 1.2rem 0;  
    padding-bottom: 0.6rem;  
    border-bottom: 2px solid #e2e8f0;  
    background: linear-gradient(120deg, #1e40af, #3b82f6);  
    -webkit-background-clip: text;  
    -webkit-text-fill-color: transparent;  
}  

.result-grid {  
    display: grid;  
    grid-template-columns: 1fr;  
    gap: 1rem;  
    margin: 1.2rem 0;  
}  

.probability-container,   
.risk-level-container,   
.risk-description-container {  
    background: linear-gradient(to right, #ffffff, #f8fafc);  
    border-radius: 6px;  
    padding: 1rem;  
    margin: 0.8rem 0;  
    box-shadow: 0 1px 2px rgba(0, 0, 0, 0.05);  
    border: 1px solid #e2e8f0;  
}  

.probability-value {  
    color: #dc2626;  
    font-size: 1.8rem;  
    font-weight: 700;  
    text-align: center;  
    padding: 0.3rem 0.6rem;  
    background: linear-gradient(to right, #fee2e2, #fef2f2);  
    border-radius: 4px;  
    border: 1px solid #fecaca;  
    display: inline-block;  
    margin: 0.3rem 0;  
}  

.risk-level-label {  
    font-size: 1rem;  
    color: #334155;  
    margin-bottom: 0.3rem;  
}  

.high-risk {  
    color: #b91c1c;  
    background: linear-gradient(to right, #fee2e2, #fef2f2);  
    border: 1px solid #fecaca;  
}  

.low-risk {  
    color: #166534;  
    background: linear-gradient(to right, #dcfce7, #f0fdf4);  
    border: 1px solid #bbf7d0;  
}  

.high-risk, .low-risk {  
    font-size: 1.1rem;  
    font-weight: 600;  
    text-align: left;  
    margin: 0.6rem 0;  
    padding: 0.3rem 0.6rem;  
    border-radius: 4px;  
}  

.risk-description {  
    font-size: 1rem;  
    color: #1e293b;  
    line-height: 1.5;  
    text-align: left;  
    padding: 0.6rem;  
    background: #f8fafc;  
    border-radius: 4px;  
}  

/* 辅助样式 */  
.normal-range {  
    color: #64748b;  
    font-size: 0.85rem;  
    margin-top: 0.3rem;  
    padding: 0.3rem 0;  
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
    gap: 0.8rem !important;  
}  

/* 为指标添加立体效果 */  
.input-group {  
    background: linear-gradient(145deg, #ffffff, #f1f5f9);  
    border: 1px solid #e2e8f0;  
    border-radius: 8px;  
    padding: 1rem;  
    margin: 0.6rem 0;  
    box-shadow:   
        0 2px 4px rgba(0, 0, 0, 0.05),  
        inset 0 -2px 4px rgba(0, 0, 0, 0.02),  
        inset 0 2px 4px rgba(255, 255, 255, 0.8);  
    transition: all 0.2s ease;  
    position: relative;  
    overflow: hidden;  
}  

.input-group::before {  
    content: '';  
    position: absolute;  
    top: 0;  
    left: 0;  
    right: 0;  
    height: 4px;  
    background: linear-gradient(90deg, #3b82f6, #60a5fa);  
    opacity: 0;  
    transition: opacity 0.2s ease;  
}  

.input-group:hover::before {  
    opacity: 1;  
}  

.input-group:hover {  
    border-color: #3b82f6;  
    box-shadow:   
        0 4px 6px rgba(59, 130, 246, 0.1),  
        inset 0 -2px 4px rgba(0, 0, 0, 0.02),  
        inset 0 2px 4px rgba(255, 255, 255, 0.8);  
    transform: translateY(-1px);  
}  

/* 输入控件立体效果 */  
.stNumberInput > div > div > input {  
    background: linear-gradient(to bottom, #ffffff, #f8fafc) !important;  
    box-shadow:   
        inset 0 2px 4px rgba(0, 0, 0, 0.05),  
        0 1px 2px rgba(255, 255, 255, 0.9) !important;  
}  

/* 结果显示立体效果 */  
.probability-container,   
.risk-level-container,   
.risk-description-container {  
    background: linear-gradient(145deg, #ffffff, #f8fafc);  
    border-radius: 8px;  
    padding: 1rem;  
    margin: 0.8rem 0;  
    box-shadow:   
        0 2px 4px rgba(0, 0, 0, 0.05),  
        inset 0 -2px 4px rgba(0, 0, 0, 0.02),  
        inset 0 2px 4px rgba(255, 255, 255, 0.8);  
    position: relative;  
    overflow: hidden;  
}  

.probability-value {  
    background: linear-gradient(145deg, #fee2e2, #fef2f2);  
    box-shadow:   
        0 2px 4px rgba(220, 38, 38, 0.1),  
        inset 0 -2px 4px rgba(0, 0, 0, 0.02),  
        inset 0 2px 4px rgba(255, 255, 255, 0.8);  
}  

/* 风险等级立体效果 */  
.high-risk, .low-risk {  
    position: relative;  
    box-shadow:   
        0 2px 4px rgba(0, 0, 0, 0.05),  
        inset 0 -2px 4px rgba(0, 0, 0, 0.02),  
        inset 0 2px 4px rgba(255, 255, 255, 0.8);  
}  

.high-risk::before,  
.low-risk::before {  
    content: '';  
    position: absolute;  
    top: 0;  
    left: 0;  
    right: 0;  
    height: 2px;  
    background: linear-gradient(90deg,   
        var(--highlight-color, #dc2626),   
        var(--highlight-color-light, #ef4444));  
    opacity: 0.8;  
}  

.high-risk {  
    --highlight-color: #dc2626;  
    --highlight-color-light: #ef4444;  
}  

.low-risk {  
    --highlight-color: #16a34a;  
    --highlight-color-light: #22c55e;  
}  

/* 计算按钮立体效果 */  
.stButton > button {  
    background: linear-gradient(145deg, #3b82f6, #1e40af) !important;  
    box-shadow:   
        0 2px 4px rgba(59, 130, 246, 0.2),  
        inset 0 2px 4px rgba(255, 255, 255, 0.1) !important;  
    border: none !important;  
    position: relative;  
    overflow: hidden;  
}  

.stButton > button::before {  
    content: '';  
    position: absolute;  
    top: 0;  
    left: -100%;  
    width: 100%;  
    height: 100%;  
    background: linear-gradient(  
        120deg,  
        transparent,  
        rgba(255, 255, 255, 0.2),  
        transparent  
    );  
    transition: 0.5s;  
}  

.stButton > button:hover::before {  
    left: 100%;  
}  

/* 响应式调整 */  
@media (max-width: 768px) {  
    .block-container {  
        padding: 1rem !important;  
    }  
    
    .input-group {  
        padding: 0.8rem;  
    }  
    
    .input-label {  
        font-size: 0.9rem;  
    }  
    
    .stButton > button {  
        min-width: 220px !important;  
        padding: 0.6rem 1.2rem !important;  
        font-size: 1rem !important;  
    }  
    
    .title {  
        font-size: 1.4rem;  
    }  
    
    .subtitle {  
        font-size: 1rem;  
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
