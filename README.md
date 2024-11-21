# END Risk Assessment System

Early Neurological Deterioration Risk Assessment System for Acute Ischemic Stroke

## Overview
This system is designed to assess the risk of early neurological deterioration (END) in patients with acute ischemic stroke. It utilizes machine learning algorithms to provide real-time risk evaluation based on key clinical indicators.

## Features
- Real-time risk assessment
- Multiple clinical indicator support
- User-friendly interface
- Instant results visualization
- Evidence-based prediction model

## Clinical Indicators
The system evaluates the following key parameters:
- NIHSS Score (National Institutes of Health Stroke Scale)
- Systolic Blood Pressure (SBP)
- Neutrophil Count (NEUT)
- Red Cell Distribution Width (RDW)
- TOAST Classification (Large-Artery Atherosclerosis)
- Intracranial Arterial Stenosis (IAS)

## How to Use
1. Access the system through the provided URL
2. Input patient clinical indicators
3. Click "Calculate Risk Score"
4. Review the assessment results and risk prediction

## Technical Stack
- Python 3.x
- Streamlit (Web Framework)
- XGBoost (Machine Learning Model)
- Pandas (Data Processing)
- Scikit-learn (Model Support)
- Joblib (Model Loading)

## Installation
```bash
# Clone the repository
git clone https://github.com/[your-username]/END-Risk-Assessment.git

# Navigate to project directory
cd END-Risk-Assessment

# Install required packages
pip install -r requirements.txt

# Run the application
streamlit run streamlit_app.py
```

## Online Access
[Access the live system](#) (Link to be added after deployment)

## Model Information
- Algorithm: XGBoost
- Training Data: Clinical data from stroke patients
- Validation: Cross-validated on independent dataset
- Key Features: Six clinical indicators
- Model File: `XGBOOST_model1113.pkl`

## Development
This system was developed using:
- Python 3.x
- Streamlit for web interface
- XGBoost for machine learning
- Scientific computing libraries (NumPy, Pandas)

## Security and Privacy
- No patient data is stored
- All calculations are performed in real-time
- Compliant with medical data privacy requirements

## Contributing
Contributions to improve the system are welcome. Please follow these steps:
1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## Authors
[Your Name/Institution Name]

## License
This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments
- [Your Institution/Hospital Name]
- Contributors and medical professionals who provided expertise
- Research team members

## Contact
For questions or support, please contact:
[Your Contact Information]

---
© 2024 END Risk Assessment System. All Rights Reserved.
