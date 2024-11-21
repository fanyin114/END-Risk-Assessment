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
    background: #e6e9f0;  
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif;  
}  

.container {  
    max-width: 700px;  
    margin: 0 auto;  
    padding: 2rem;  
    background: linear-gradient(145deg, #f0f2f5, #ffffff);  
}  

/* 标题样式 */  
.title {  
    font-size: 1.8rem;  
    font-weight: 700;  
    text-align: center;  
    color: #2c3e50;  
    margin-bottom: 0.5rem;  
    text-shadow: 2px 2px 4px rgba(0,0,0,0.1);  
}  

.subtitle {  
    font-size: 1.1rem;  
    text-align: center;  
    color: #5a6c7d;  
    margin-bottom: 2.5rem;  
}  

/* 表单项样式 */  
.form-group {  
    background: linear-gradient(145deg, #ffffff, #f0f2f5);  
    border-radius: 16px;  
    padding: 1.5rem;  
    margin-bottom: 1.2rem;  
    box-shadow: 5px 5px 15px #d1d9e6,  
                -5px -5px 15px #ffffff;  
    border: 1px solid rgba(255,255,255,0.18);  
    transition: transform 0.3s ease, box-shadow 0.3s ease;  
}  

.form-group:hover {  
    transform: translateY(-2px);  
    box-shadow: 6px 6px 18px #d1d9e6,  
                -6px -6px 18px #ffffff;  
}  

/* 表单标签 */  
.form-label {  
    display: block;  
    color: #2c3e50;  
    font-size: 1rem;  
    font-weight: 600;  
    margin-bottom: 1rem;  
    letter-spacing: 0.5px;  
}  

/* 单选按钮组 */  
.radio-group {  
    display: flex;  
    gap: 2.5rem;  
    padding: 0.5rem;  
    background: rgba(255,255,255,0.9);  
    border-radius: 12px;  
    box-shadow: inset 2px 2px 5px #d1d9e6,  
                inset -2px -2px 5px #ffffff;  
}  

.radio-label {  
    display: inline-flex;  
    align-items: center;  
    cursor: pointer;  
    padding: 0.5rem 1rem;  
    border-radius: 8px;  
    transition: background 0.3s ease;  
}  

.radio-label:hover {  
    background: rgba(26,115,232,0.1);  
}  

/* 数字输入框容器 */  
.input-container {  
    display: flex;  
    align-items: center;  
    max-width: 300px;  
    margin: 0 auto;  
}  

/* 数字输入框 */  
.number-input {  
    width: 70%;  
    padding: 0.8rem 1rem;  
    border: none;  
    border-radius: 12px;  
    font-size: 1rem;  
    color: #2c3e50;  
    background: white;  
    box-shadow: inset 2px 2px 5px #d1d9e6,  
                inset -2px -2px 5px #ffffff;  
    transition: all 0.3s ease;  
}  

.number-input:focus {  
    outline: none;  
    box-shadow: inset 3px 3px 7px #d1d9e6,  
                inset -3px -3px 7px #ffffff;  
}  

/* 单位文本 */  
.unit-text {  
    color: #5a6c7d;  
    font-size: 0.9rem;  
    margin-left: 1rem;  
    font-weight: 500;  
    width: 80px;  
}  

/* 正常范围提示 */  
.normal-range {  
    color: #5a6c7d;  
    font-size: 0.85rem;  
    margin-top: 0.8rem;  
    padding: 0.5rem 1rem;  
    background: rgba(255,255,255,0.7);  
    border-radius: 8px;  
    border-left: 3px solid #1a73e8;  
}  

/* 提交按钮 */  
.submit-button {  
    display: block;  
    width: 80%;  
    max-width: 300px;  
    margin: 2.5rem auto;  
    padding: 1rem;  
    background: linear-gradient(145deg, #1a73e8, #1557b0);  
    color: white;  
    border: none;  
    border-radius: 30px;  
    font-size: 1.1rem;  
    font-weight: 600;  
    cursor: pointer;  
    text-align: center;  
    transition: all 0.3s ease;  
    box-shadow: 5px 5px 10px #d1d9e6,  
                -5px -5px 10px #ffffff;  
}  

.submit-button:hover {  
    transform: translateY(-2px);  
    box-shadow: 6px 6px 12px #d1d9e6,  
                -6px -6px 12px #ffffff;  
}  

/* 结果区域 */  
.result-container {  
    background: linear-gradient(145deg, #ffffff, #f0f2f5);  
    border-radius: 20px;  
    padding: 2rem;  
    margin-top: 2.5rem;  
    box-shadow: 8px 8px 20px #d1d9e6,  
                -8px -8px 20px #ffffff;  
}  

.result-title {  
    font-size: 1.4rem;  
    font-weight: 700;  
    text-align: center;  
    color: #2c3e50;  
    margin-bottom: 2rem;  
    text-shadow: 2px 2px 4px rgba(0,0,0,0.1);  
}  

.result-item {  
    display: flex;  
    justify-content: space-between;  
    align-items: center;  
    margin-bottom: 1.2rem;  
    padding: 1rem;  
    background: rgba(255,255,255,0.8);  
    border-radius: 12px;  
    box-shadow: inset 2px 2px 5px #d1d9e6,  
                inset -2px -2px 5px #ffffff;  
}  

.probability-value {  
    color: #1a73e8;  
    font-weight: 700;  
    font-size: 1.2rem;  
}  

.risk-level {  
    color: #dc3545;  
    font-weight: 700;  
    font-size: 1.2rem;  
}  

.risk-description {  
    background: rgba(255,255,255,0.8);  
    padding: 1.2rem;  
    border-radius: 12px;  
    margin-top: 1.5rem;  
    color: #2c3e50;  
    font-size: 1rem;  
    line-height: 1.6;  
    box-shadow: inset 2px 2px 5px #d1d9e6,  
                inset -2px -2px 5px #ffffff;  
}  

/* 响应式调整 */  
@media (max-width: 480px) {  
    .container {  
        padding: 1rem;  
    }  
    
    .form-group {  
        padding: 1rem;  
    }  
    
    .input-container {  
        max-width: 100%;  
    }  
    
    .submit-button {  
        width: 90%;  
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
