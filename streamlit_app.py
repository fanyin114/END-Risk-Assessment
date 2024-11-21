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
    background-color: #f8f9fa;  
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif;  
}  

.container {  
    max-width: 600px;  
    margin: 0 auto;  
    padding: 2rem 1rem;  
}  

/* 标题样式 */  
.title {  
    font-size: 1.5rem;  
    font-weight: 600;  
    text-align: center;  
    color: #333;  
    margin-bottom: 0.5rem;  
}  

.subtitle {  
    font-size: 1rem;  
    text-align: center;  
    color: #666;  
    margin-bottom: 2rem;  
}  

/* 表单项样式 */  
.form-group {  
    background-color: white;  
    border-radius: 8px;  
    padding: 1rem;  
    margin-bottom: 1rem;  
    box-shadow: 0 1px 3px rgba(0,0,0,0.1);  
}  

/* 表单标签 */  
.form-label {  
    display: block;  
    color: #333;  
    font-size: 0.9rem;  
    margin-bottom: 0.75rem;  
}  

/* 单选按钮组 */  
.radio-group {  
    display: flex;  
    gap: 2rem;  
}  

.radio-label {  
    display: inline-flex;  
    align-items: center;  
    cursor: pointer;  
}  

.radio-input {  
    margin-right: 0.5rem;  
    cursor: pointer;  
}  

/* 数字输入框 */  
.number-input {  
    width: 100%;  
    padding: 0.5rem;  
    border: 1px solid #ddd;  
    border-radius: 4px;  
    font-size: 1rem;  
}  

/* 单位文本 */  
.unit-text {  
    color: #666;  
    font-size: 0.9rem;  
    margin-left: 0.5rem;  
}  

/* 正常范围提示 */  
.normal-range {  
    color: #666;  
    font-size: 0.8rem;  
    margin-top: 0.5rem;  
}  

/* 提交按钮 */  
.submit-button {  
    display: block;  
    width: 100%;  
    max-width: 400px;  
    margin: 2rem auto;  
    padding: 0.75rem;  
    background-color: #1a73e8;  
    color: white;  
    border: none;  
    border-radius: 24px;  
    font-size: 1rem;  
    font-weight: 500;  
    cursor: pointer;  
    text-align: center;  
    transition: background-color 0.2s;  
}  

.submit-button:hover {  
    background-color: #1557b0;  
}  

/* 结果区域 */  
.result-container {  
    background-color: white;  
    border-radius: 8px;  
    padding: 1.5rem;  
    margin-top: 2rem;  
    box-shadow: 0 1px 3px rgba(0,0,0,0.1);  
}  

.result-title {  
    font-size: 1.2rem;  
    font-weight: 600;  
    text-align: center;  
    color: #333;  
    margin-bottom: 1.5rem;  
}  

.result-item {  
    display: flex;  
    justify-content: space-between;  
    align-items: center;  
    margin-bottom: 1rem;  
}  

.result-label {  
    color: #333;  
}  

.probability-value {  
    color: #1a73e8;  
    font-weight: 600;  
}  

.risk-level {  
    color: #dc3545;  
    font-weight: 600;  
}  

.risk-description {  
    background-color: #f8f9fa;  
    padding: 1rem;  
    border-radius: 4px;  
    margin-top: 1rem;  
    color: #666;  
    font-size: 0.9rem;  
}  

/* 输入框样式优化 */  
input[type="number"] {  
    -moz-appearance: textfield;  
}  

input[type="number"]::-webkit-outer-spin-button,  
input[type="number"]::-webkit-inner-spin-button {  
    -webkit-appearance: none;  
    margin: 0;  
}  

/* 响应式调整 */  
@media (max-width: 480px) {  
    .container {  
        padding: 1rem;  
    }  
    
    .form-group {  
        padding: 0.75rem;  
    }  
    
    .radio-group {  
        gap: 1rem;  
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
