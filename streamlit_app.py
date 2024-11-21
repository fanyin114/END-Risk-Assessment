import streamlit as st  
import xgboost as xgb  
import numpy as np  
import pandas as pd  
import joblib  
import logging  
import random 
 
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
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');  

* {  
    font-family: 'Inter', sans-serif;  
}  

.main > div {  
    padding: 0 !important;  
}  

/* 页面容器 */  
.block-container {  
    max-width: 1000px;  
    padding: 2rem;  
    margin: 0 auto;  
    background: linear-gradient(to bottom, #f8f9fa, #ffffff);  
}  

/* 标题样式 */  
.title {  
    color: #1a237e;  
    text-align: center;  
    font-size: 2.5rem;  
    font-weight: 700;  
    margin-bottom: 0.5rem;  
    padding: 1rem;  
    background: linear-gradient(120deg, #1a237e, #3949ab);  
    -webkit-background-clip: text;  
    -webkit-text-fill-color: transparent;  
    text-shadow: 2px 2px 4px rgba(0,0,0,0.1);  
}  

.subtitle {  
    color: #424242;  
    text-align: center;  
    font-size: 1.2rem;  
    margin-bottom: 2rem;  
    font-weight: 400;  
    opacity: 0.8;  
}  

/* 输入区域样式 */  
.input-section {  
    background: linear-gradient(145deg, #ffffff, #f0f2f5);  
    padding: 1.5rem 2rem;  
    border-radius: 16px;  
    margin: 1.2rem 0;  
    box-shadow: 5px 5px 15px #d1d9e6,  
                -5px -5px 15px #ffffff;  
    border: 1px solid rgba(255, 255, 255, 0.18);  
    transition: all 0.3s ease;  
}  

.input-section:hover {  
    transform: translateY(-2px);  
    box-shadow: 6px 6px 18px #d1d9e6,  
                -6px -6px 18px #ffffff;  
}  

/* 输入标签样式 */  
.input-label {  
    color: #2c3e50;  
    font-size: 1.1rem;  
    font-weight: 600;  
    margin-bottom: 0.8rem;  
    display: flex;  
    align-items: center;  
    letter-spacing: 0.5px;  
    text-shadow: 1px 1px 1px rgba(255, 255, 255, 0.5);  
}  

/* Radio按钮组样式 */  
.stRadio > div {  
    background: none !important;  
    padding: 0.5rem 0 !important;  
}  

.stRadio > div > div {  
    background: rgba(255, 255, 255, 0.8);  
    padding: 0.8rem;  
    border-radius: 12px;  
    transition: all 0.2s ease;  
    box-shadow: 3px 3px 8px #d1d9e6,  
                -3px -3px 8px #ffffff;  
}  

.stRadio > div > div:hover {  
    background: rgba(255, 255, 255, 0.9);  
    box-shadow: 4px 4px 10px #d1d9e6,  
                -4px -4px 10px #ffffff;  
}  


/* 数字输入框样式 */  
.stNumberInput > div > div > input {  
    font-size: 1.1rem;  
    padding: 0.75rem !important;  
    border: 2px solid rgba(226, 232, 240, 0.6) !important;  
    border-radius: 12px !important;  
    transition: all 0.2s ease;  
    background: rgba(255, 255, 255, 0.8) !important;  
    box-shadow: inset 2px 2px 5px #d1d9e6,  
                inset -2px -2px 5px #ffffff;  
}  

.stNumberInput > div > div > input:focus {  
    border-color: #3949ab !important;  
    box-shadow: 0 0 0 3px rgba(66, 153, 225, 0.15);  
    background: #ffffff !important;  
}  


/* 正常范围提示样式 */  
.normal-range {  
    color: #718096;  
    font-size: 0.9rem;  
    margin-top: 0.8rem;  
    padding: 0.6rem 1rem;  
    background: rgba(255, 255, 255, 0.7);  
    border-radius: 8px;  
    border-left: 3px solid #4299e1;  
    box-shadow: 2px 2px 5px rgba(209, 217, 230, 0.5);  
}  
/* 整体容器背景 */  
.block-container {  
    max-width: 1000px;  
    padding: 2rem;  
    margin: 0 auto;  
    background: #f0f2f5;  
}  

/* 按钮样式 */  
.stButton > button {  
    background: linear-gradient(135deg, #1a237e, #3949ab);  
    color: white;  
    padding: 0.75rem 2rem;  
    border-radius: 12px;  
    border: none;  
    width: 100%;  
    margin: 1.5rem 0;  
    font-size: 1.2rem;  
    font-weight: 600;  
    letter-spacing: 0.5px;  
    transition: all 0.3s ease;  
    box-shadow: 0 4px 6px rgba(26, 35, 126, 0.2);  
}  

.stButton > button:hover {  
    transform: translateY(-2px);  
    box-shadow: 0 6px 12px rgba(26, 35, 126, 0.3);  
    background: linear-gradient(135deg, #283593, #3f51b5);  
}  

/* 结果区域样式 */  
.result-section {  
    background: linear-gradient(135deg, #ffffff, #f8f9fa);  
    padding: 2rem;  
    border-radius: 16px;  
    margin-top: 2rem;  
    text-align: center;  
    box-shadow: 0 6px 12px rgba(0,0,0,0.08);  
    border: 1px solid rgba(226, 232, 240, 0.8);  
}  

.result-title {  
    color: #1a237e;  
    font-size: 1.5rem;  
    font-weight: 700;  
    margin-bottom: 1.5rem;  
    background: linear-gradient(120deg, #1a237e, #3949ab);  
    -webkit-background-clip: text;  
    -webkit-text-fill-color: transparent;  
}  

.high-risk {  
    color: #e53e3e;  
    font-weight: 700;  
    font-size: 1.3rem;  
    padding: 0.5rem 1rem;  
    background: rgba(229, 62, 62, 0.1);  
    border-radius: 8px;  
    display: inline-block;  
}  

.low-risk {  
    color: #38a169;  
    font-weight: 700;  
    font-size: 1.3rem;  
    padding: 0.5rem 1rem;  
    background: rgba(56, 161, 105, 0.1);  
    border-radius: 8px;  
    display: inline-block;  
}  

.risk-description {  
    margin-top: 1.5rem;  
    font-size: 1.1rem;  
    line-height: 1.6;  
    color: #4a5568;  
    padding: 1rem;  
    border-radius: 8px;  
}  

/* 响应式设计调整 */  
@media (max-width: 768px) {  
    .input-section {  
        padding: 1.2rem;  
        margin: 1rem 0;  
    }  
    
    .normal-range {  
        padding: 0.5rem 0.8rem;  
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
        # 设置XGBoost的随机种子  
        if isinstance(model, xgb.XGBClassifier):  
            model.random_state = 0  
        logging.info("模型加载成功")  
        return model  
    except Exception as e:  
        logging.error(f"模型加载失败: {str(e)}")  
        return None  
# 标题  
st.markdown("<h1 class='title'>早期神经功能恶化风险评估系统</h1>", unsafe_allow_html=True)  
st.markdown("<p class='subtitle'>Early Neurological Deterioration Risk Assessment System</p>", unsafe_allow_html=True)  

# 创建主容器  
with st.container():  
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

        except Exception as e:  
            st.error(f"计算过程出错 / Calculation Error: {str(e)}")
