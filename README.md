# Smart Loan Recovery System 💸

This project uses machine learning to segment borrowers and classify their risk of default. The goal is to support recovery strategy teams by identifying higher-risk borrowers based on key financial indicators.

## 🚀 Features

- 📊 Data visualization using Plotly
- 🧹 Data preprocessing pipeline
- 🤖 Clustering borrowers via KMeans
- 🌲 Classification using RandomForest
- ✅ Evaluation metrics & model export
- 📁 Modular structure for maintainability

## 📁 Project Structure

Smart-Loan-Recovery-System-With-Machine-Learning/
project_root/  
│  
├── data/ # Raw or processed input data  
│  
├── src/ # Source code modules  
│ ├── main.py # Main pipeline script  
│ ├── preprocessing.py # Data cleaning & feature engineering  
│ ├── model_training.py # Model training and evaluation  
│ ├── segmentation.py # KMeans clustering logic  
│ └── visualization.py # Plotly chart generation  
│  
├── charts/ # Exported visualizations (PNG, HTML, etc.)  
├── artifacts/ # Saved models, metadata, and encoders  
├── config.yaml # Config file for model settings and parameters  
├── requirements.txt # Python dependencies  
├── README.md # Project documentation  
└── .gitignore # Files/folders to exclude from Git  

## 📈 Key Visualizations
You can find detailed visualizations in the charts/ folder (optional), including:
- 📊 Loan Amount Distribution vs Monthly Income
![]()
- 📦 Boxplot: Missed Payments vs Recovery Status
- 🧩 Segments of Borrowers by Loan & Income
- 📌 Recovery Status by Payment History

## 🧪 Model Training & Evaluation
- 🧬 Clustering: KMeans (4 clusters)
    - Segment borrowers into risk profiles (e.g., high loan burden, low default risk).
- 🧠 Classification: Random Forest
    - Label segments as recoverable or not.
    - Metrics:
        - ✅ Accuracy: XX%
        - 📍 Precision: XX%
        - 📈 Recall: XX%
        - 🏆 F1 Score: XX%
## 🧾 Dependencies
- Key libraries:
    - pandas, scikit-learn
    - plotly, matplotlib, seaborn
    - yaml (if using config.yaml)
    - Full list in requirements.txt.