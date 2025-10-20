# Diabetes Prediction using Machine Learning
![Python](https://img.shields.io/badge/python-3.8+-blue.svg)
![scikit-learn](https://img.shields.io/badge/scikit--learn-latest-orange.svg)
![Streamlit](https://img.shields.io/badge/streamlit-latest-red.svg)

## Overview
This project implements a machine learning solution to predict diabetes in individuals based on various health parameters. Using Support Vector Machine (SVM) classification, the model analyzes health metrics to provide accurate diabetes risk assessments. The project includes a user-friendly Streamlit web interface for real-time predictions.

## Table of Contents
- [Features](#features)
- [Dataset](#dataset)
- [Model Architecture](#model-architecture)
- [Requirements](#requirements)
- [Installation](#installation)
- [Usage](#usage)
- [Results](#results)
- [Contributing](#contributing)
- [License](#license)
- [Contact](#contact)

## Features
- **Data Preprocessing Pipeline**: Automated handling of missing values using SimpleImputer
- **Feature Engineering**: OneHotEncoder for categorical variable transformation
- **Machine Learning Model**: Support Vector Machine (SVM) classifier optimized for medical data
- **End-to-End Pipeline**: Streamlined preprocessing and modeling workflow
- **Performance Metrics**: Comprehensive evaluation using accuracy, precision, recall, and F1-score
- **Interactive Web App**: Streamlit-based deployment for instant predictions
- **Model Persistence**: Trained model saved using Pickle for production deployment

## Dataset
**Source**: Pima Indians Diabetes Database (UCI Machine Learning Repository) - National Institute of Diabetes and Digestive and Kidney Diseases.

The dataset includes the following health parameters:
- **Age**: Patient's age in years
- **BMI**: Body Mass Index (kg/m²)
- **Blood Pressure**: Systolic blood pressure measurement
- **Glucose Level**: Blood glucose concentration
- **Insulin**: Serum insulin level
- **Skin Thickness**: Triceps skin fold thickness
- **Pregnancies**: Number of pregnancies (if applicable)
- **Diabetes Pedigree Function**: Genetic predisposition score

**Target Variable**: Binary classification (0: No Diabetes, 1: Diabetes)

## Model Architecture

### Preprocessing
1. **Missing Value Imputation**: SimpleImputer with mean strategy
2. **Categorical Encoding**: OneHotEncoder for non-numeric features
3. **Feature Scaling**: Standardization for optimal SVM performance

### Classification
- **Algorithm**: Support Vector Machine (SVM)
- **Kernel**: RBF (Radial Basis Function)
- **Hyperparameter Tuning**: Grid search for optimal C and gamma values

## Requirements
```
python>=3.8
scikit-learn>=1.0.0
pandas>=1.3.0
numpy>=1.21.0
streamlit>=1.0.0
pickle-mixin>=1.0.0
```

## Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/arpanpramanik2003/diabetes-prediction.git
   cd diabetes-prediction
   ```

2. **Create a virtual environment** (recommended)
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

## Usage

### Training the Model
```python
python train_model.py
```

### Running the Streamlit Web App
```bash
streamlit run app.py
```

The web interface will be available at `http://localhost:8501`

### Making Predictions
Input the required health parameters in the web interface to receive instant diabetes risk predictions.

## Results

### Model Performance

| Metric | Score |
| --- | --- |
| Accuracy | 80.5% |
| Precision | 81% |
| Recall | 77% |
| F1-Score | 79% |

The model demonstrates strong performance in predicting diabetes, with balanced precision and recall scores indicating reliable classification for both positive and negative cases.

## Contributing

Contributions are welcome! Please follow these steps:

1. **Fork the repository**
2. **Create a feature branch**
   ```bash
   git checkout -b feature/your-feature-name
   ```
3. **Commit your changes**
   ```bash
   git commit -m "Add your descriptive commit message"
   ```
4. **Push to your branch**
   ```bash
   git push origin feature/your-feature-name
   ```
5. **Open a Pull Request**

### Contribution Guidelines
- Follow PEP 8 style guidelines for Python code
- Add unit tests for new features
- Update documentation as needed
- Ensure all tests pass before submitting PR

## License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Contact
**Arpan Pramanik**
- GitHub: [@arpanpramanik2003](https://github.com/arpanpramanik2003)
- Email: [pramanikarpan089@gmail.com](mailto:pramanikarpan089@gmail.com)
- LinkedIn: [Arpan Pramanik](https://www.linkedin.com/in/arpan-pramanik-6a409228a/)

## Acknowledgments
- Dataset source: Pima Indians Diabetes Database (UCI Machine Learning Repository) - National Institute of Diabetes and Digestive and Kidney Diseases
- Inspired by various diabetes prediction research papers
- Thanks to the open-source community for the amazing tools

---
⭐ If you find this project helpful, please consider giving it a star!

**Disclaimer**: This tool is for educational and research purposes only. Always consult healthcare professionals for medical decisions.
