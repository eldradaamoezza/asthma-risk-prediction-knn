Asthma Risk Prediction Using K-Nearest Neighbor (KNN)
📌 Overview

This project implements the K-Nearest Neighbor (KNN) algorithm to predict asthma risk using a structured medical dataset. The study follows the CRISP-DM (Cross Industry Standard Process for Data Mining) methodology, covering all stages from business understanding to deployment.

The model is developed in Google Colab using Python and deployed as an interactive web application using Streamlit.

🎯 Objectives
- Develop a classification model to predict asthma risk.
- Evaluate model performance using Accuracy, Precision, Recall, and F1-Score.
- Apply hyperparameter tuning with Cross Validation.
- Deploy the final model into an interactive Streamlit dashboard.

📊 Dataset

Source: Kaggle – Asthma Synthetic Medical Dataset

Total Records: 5,400

Features include:
- Age
- Gender
- BMI
- Smoking Status
- Family History
- Allergies
- Air Pollution Level
- Physical Activity Level
- Occupation Type
- Comorbidities
- Medication Adherence
- ER Visits
- Peak Expiratory Flow
- FeNO Level

Target:
- Has_Asthma (Yes/No)

⚙️ Methodology (CRISP-DM)
- Business Understanding
- Data Understanding (EDA, duplicate check, missing values, outlier detection)
- Data Preparation (Cleaning, Encoding, Min-Max Scaling)
- Modeling (KNN Baseline & Hyperparameter Tuning)
- Evaluation (Confusion Matrix, Classification Report)
- Deployment (Streamlit Web Application)

🤖 Modeling
- Algorithm: K-Nearest Neighbor (KNN)
- Distance Metric: Euclidean Distance
- Data Split: 80% Training – 20% Testing
- Hyperparameter Tuning: Grid Search with 5-Fold Cross Validation

📈 Model Performance (After Hyperparameter Tuning)
- Accuracy: 89.41%
- Precision: 0.88 – 0.91
- Recall: 0.87 – 0.91
- F1-Score: 0.89

The model demonstrates strong performance in classifying both asthma and non-asthma cases.

🖥 Deployment
- The final model is deployed using Streamlit.

Features:
- Interactive patient data input form
- Real-time prediction
- Model performance visualization

To run locally:
pip install -r requirements.txt
streamlit run app/app.py

🛠 Tech Stack
- Python
- Pandas
- NumPy
- Scikit-learn
- Matplotlib
- Seaborn
- Google Colab
- Streamlit

👩‍💻 Author

Eldrada Intan Putri
Information Systems – Business Intelligence
Institut Informatika dan Bisnis Darmajaya
Year: 2026

📜 License

This project is developed for academic research purposes.


Repositori dengan visual preview terlihat jauh lebih profesional.

====================================================
