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

### Pipeline Structure

```python
Pipeline([
    ('imputer', SimpleImputer()),
    ('encoder', OneHotEncoder()),
    ('classifier', SVC())
])
```

## Requirements

```
python>=3.8
numpy>=1.21.0
pandas>=1.3.0
scikit-learn>=1.0.0
streamlit>=1.10.0
matplotlib>=3.4.0
seaborn>=0.11.0
pickle-mixin>=1.0.2
```

## Installation

### Step 1: Clone the Repository

```bash
git clone https://github.com/arpanpramanik2003/diabetes-prediction.git
cd diabetes-prediction
```

### Step 2: Create Virtual Environment (Optional but Recommended)

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### Step 3: Install Dependencies

```bash
pip install -r requirements.txt
```

## Usage

### Training the Model

```bash
python train_model.py
```

This will:
- Load and preprocess the dataset
- Train the SVM classifier
- Save the trained model as `model.pkl`
- Display performance metrics

### Running the Web Application

```bash
streamlit run app.py
```

The application will be available at `http://localhost:8501`

### Making Predictions

#### Via Web Interface
1. Open the Streamlit app in your browser
2. Enter health parameters in the input fields
3. Click "Predict" to get instant results

#### Via Python Script

```python
import pickle
import numpy as np

# Load the trained model
with open('model.pkl', 'rb') as file:
    model = pickle.load(file)

# Sample input data
input_data = np.array([[age, bmi, blood_pressure, glucose, insulin, skin_thickness, pregnancies, pedigree]])

# Make prediction
prediction = model.predict(input_data)
print(f"Prediction: {'Diabetes' if prediction[0] == 1 else 'No Diabetes'}")
```

## Results

### Model Performance

| Metric | Score |
|--------|-------|
| Accuracy | XX.XX% |
| Precision | XX.XX% |
| Recall | XX.XX% |
| F1-Score | XX.XX% |

### Confusion Matrix

*(Add your confusion matrix visualization here)*

### Key Insights

- The model demonstrates strong performance in identifying diabetes cases
- Feature importance analysis reveals BMI and glucose levels as primary indicators
- Cross-validation ensures model generalization across different data splits

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
- Email: [your.email@example.com](mailto:your.email@example.com)
- LinkedIn: [Your LinkedIn Profile](https://linkedin.com/in/yourprofile)

## Acknowledgments

- Dataset source: [Specify dataset source]
- Inspired by various diabetes prediction research papers
- Thanks to the open-source community for the amazing tools

---

⭐ If you find this project helpful, please consider giving it a star!

**Disclaimer**: This tool is for educational and research purposes only. Always consult healthcare professionals for medical decisions.
