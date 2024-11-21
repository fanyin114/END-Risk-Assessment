# 早期神经功能恶化风险评估系统

Early Neurological Deterioration Risk Assessment System

## 简介
本系统用于评估急性缺血性脑卒中患者发生早期神经功能恶化的风险。

## 功能特点
- 支持多个关键指标输入
- 实时风险评估
- 双语界面（中文/英文）
- 直观的结果展示

## 评估指标
- NIHSS评分
- 收缩压 (SBP)
- 中性粒细胞计数 (NEUT)
- 红细胞分布宽度 (RDW)
- 大动脉粥样硬化型 (TOAST-LAA)
- 颅内动脉狭窄 (IAS)

## 使用方法
1. 访问系统网址
2. 输入患者相关指标
3. 点击"计算风险评分"按钮
4. 查看评估结果

## 技术栈
- Python
- Streamlit
- XGBoost
- Pandas
- Scikit-learn

## 安装和运行
```bash
# 克隆仓库
git clone https://github.com/您的用户名/END-Risk-Assessment.git

# 进入项目目录
cd END-Risk-Assessment

# 安装依赖
pip install -r requirements.txt

# 运行应用
streamlit run streamlit_app.py
```

## 在线访问
[点击访问在线系统](#) (部署后添加链接)

## 作者
[您的名字/机构名称]

## 许可证
[选择合适的许可证，如 MIT License]
