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
body {  
    background-color: rgb(243, 244, 246);  
    min-height: 100vh;  
}  

.container {  
    max-width: 1000px;  
    margin: 0 auto;  
    padding: 2rem 1rem;  
}  

/* 标题区域 */  
.header {  
    background-color: white;  
    border-top-left-radius: 0.75rem;  
    border-top-right-radius: 0.75rem;  
    box-shadow: 0 1px 2px rgba(0, 0, 0, 0.05);  
    padding: 2rem;  
}  

.main-title {  
    font-size: 1.875rem;  
    font-weight: 700;  
    text-align: center;  
    color: rgb(17, 24, 39);  
    margin-bottom: 1rem;  
}  

.subtitle {  
    font-size: 1.25rem;  
    text-align: center;  
    color: rgb(75, 85, 99);  
    margin-bottom: 0.5rem;  
}  

/* 表单区域 */  
.form-container {  
    background-color: white;  
    box-shadow: 0 1px 2px rgba(0, 0, 0, 0.05);  
    padding: 2rem;  
}  

.form-group {  
    background-color: rgb(249, 250, 251);  
    padding: 1rem;  
    border-radius: 0.5rem;  
    border: 1px solid rgb(229, 231, 235);  
    margin-bottom: 1.5rem;  
    transition: border-color 0.2s;  
}  

.form-group:hover {  
    border-color: rgb(191, 219, 254);  
}  

/* 表单标签 */  
.form-label {  
    display: block;  
    color: rgb(17, 24, 39);  
    font-weight: 500;  
    margin-bottom: 0.5rem;  
}  

/* 单选按钮组 */  
.radio-group {  
    display: flex;  
    gap: 1rem;  
}  

.radio-label {  
    display: inline-flex;  
    align-items: center;  
}  

.radio-input {  
    color: rgb(37, 99, 235);  
}  

.radio-text {  
    margin-left: 0.5rem;  
}  

/* 数字输入框 */  
.number-input-container {  
    display: flex;  
    align-items: center;  
}  

.number-input {  
    display: block;  
    width: 100%;  
    border-radius: 0.375rem;  
    border: 1px solid rgb(209, 213, 219);  
    box-shadow: 0 1px 2px rgba(0, 0, 0, 0.05);  
}  

.number-input:focus {  
    border-color: rgb(59, 130, 246);  
    box-shadow: 0 0 0 2px rgba(59, 130, 246, 0.25);  
}  

.unit-text {  
    margin-left: 0.5rem;  
    color: rgb(107, 114, 128);  
    width: 5rem;  
}  

/* 范围提示文本 */  
.range-text {  
    margin-top: 0.25rem;  
    font-size: 0.875rem;  
    color: rgb(107, 114, 128);  
}  

/* 提交按钮 */  
.submit-button {  
    padding: 0.75rem 2rem;  
    background-color: rgb(37, 99, 235);  
    color: white;  
    font-size: 1.125rem;  
    font-weight: 600;  
    border-radius: 0.5rem;  
    box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);  
    transition: all 0.2s;  
}  

.submit-button:hover {  
    background-color: rgb(29, 78, 216);  
}  

.submit-button:focus {  
    outline: none;  
    box-shadow: 0 0 0 2px rgb(255, 255, 255), 0 0 0 4px rgb(59, 130, 246);  
}  

/* 结果显示区域 */  
.result-container {  
    background-color: white;  
    border-bottom-left-radius: 0.75rem;  
    border-bottom-right-radius: 0.75rem;  
    box-shadow: 0 1px 2px rgba(0, 0, 0, 0.05);  
    padding: 2rem;  
    border-top: 2px solid rgb(219, 234, 254);  
}  

.result-title {  
    font-size: 1.5rem;  
    font-weight: 700;  
    text-align: center;  
    color: rgb(17, 24, 39);  
    margin-bottom: 1.5rem;  
}  

.result-item {  
    display: flex;  
    justify-content: space-between;  
    align-items: center;  
    margin-bottom: 1rem;  
}  

.result-label {  
    font-size: 1.125rem;  
}  

.result-value {  
    font-size: 1.5rem;  
    font-weight: 700;  
}  

.risk-description {  
    margin-top: 1rem;  
    padding: 1rem;  
    background-color: rgb(249, 250, 251);  
    border-radius: 0.5rem;  
    color: rgb(55, 65, 81);  
}  

/* 响应式设计 */  
@media (max-width: 768px) {  
    .container {  
        padding: 1rem;  
    }  
    
    .form-group {  
        padding: 0.75rem;  
    }  
    
    .submit-button {  
        padding: 0.5rem 1.5rem;  
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
