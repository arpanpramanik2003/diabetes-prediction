# Diabetes Prediction using Machine Learning

This project aims to predict whether an individual has diabetes based on various health parameters using machine learning techniques. The dataset used for this project contains information such as age, BMI, blood pressure, and other health-related factors. The machine learning model is trained using the **Support Vector Machine (SVM)** algorithm, which is effective for classification tasks.

## Features:
- **Data Preprocessing**: Utilizes **SimpleImputer** for handling missing data and **OneHotEncoder** for encoding categorical variables.
- **Modeling**: A **Support Vector Machine (SVM)** classifier is used for predicting diabetes.
- **Pipeline**: The project employs a **Pipeline** to streamline the preprocessing and modeling steps for easy scalability.
- **Evaluation**: Model performance is evaluated using metrics like accuracy and classification report.
- **Deployment**: The project also integrates with **Streamlit** for interactive deployment, allowing users to input their health data and get instant predictions.

## Technologies Used:
- Python
- **scikit-learn** for machine learning
- **Streamlit** for deployment
- **Pickle** for saving the trained model
