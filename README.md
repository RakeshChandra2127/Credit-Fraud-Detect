# Credit Card Fraud Detection

A machine learning project to detect fraudulent credit card transactions using classification models.

## 📌 Project Overview
Credit card fraud is a major financial risk.  
This project builds and evaluates ML models to classify transactions as **fraudulent** or **non-fraudulent** based on transaction features.

## 🎯 Objectives
- Analyze transaction data and fraud patterns
- Handle class imbalance in fraud data
- Train and compare classification models
- Evaluate performance using fraud-focused metrics

## 🧰 Tech Stack
- Python
- Pandas, NumPy
- Matplotlib, Seaborn
- Scikit-learn
- (Optional) Imbalanced-learn (SMOTE)

## 📂 Repository Structure
- `README.md` → Project documentation
- `docs/PROJECT_REPORT.md` → Full project report
- `docs/PRESENTATION_CONTENT.md` → Slides-ready content
- `docs/SUBMISSION_LINK_TEMPLATE.md` → Submission drive link template
- `notebooks/` or `src/` → Code and experiments
- `data/` → Dataset (or instructions to access dataset)

## 🗃️ Dataset
Typical fraud datasets contain:
- anonymized PCA features (`V1...V28`)
- `Time`, `Amount`
- target column: `Class` (0 = normal, 1 = fraud)

> If your dataset is from Kaggle/UCI, add source link and license details.

## ⚙️ Methodology
1. Data loading and cleaning
2. Exploratory Data Analysis (EDA)
3. Handling class imbalance
4. Train-test split
5. Model training (e.g., Logistic Regression, Random Forest, XGBoost)
6. Evaluation (Precision, Recall, F1, ROC-AUC, PR-AUC)

## 📊 Evaluation Metrics
For fraud detection, focus on:
- **Recall (Fraud class)**: catch maximum fraud
- **Precision (Fraud class)**: reduce false alarms
- **F1-score**: balance precision and recall
- **ROC-AUC / PR-AUC**

## ✅ Results Summary
- Best model: *(fill your best model name)*
- Key metric achieved: *(fill your score)*
- Observed tradeoff: *(e.g., higher recall with acceptable precision)*

## 🚀 How to Run
```bash
# 1) Clone repo
git clone https://github.com/RakeshChandra2127/Credit-Fraud-Detect.git
cd Credit-Fraud-Detect

# 2) Create environment (optional)
python -m venv venv
# Windows: venv\Scripts\activate
# Linux/Mac: source venv/bin/activate

# 3) Install dependencies
pip install -r requirements.txt

# 4) Run project (adjust file paths/scripts as needed)
python src/train.py
```

## 🔮 Future Improvements
- Hyperparameter tuning
- Cost-sensitive learning
- Real-time fraud scoring API
- Model explainability (SHAP/LIME)

## 👤 Author
**Rakesh Chandra Behera**  
GitHub: [RakeshChandra2127](https://github.com/RakeshChandra2127)
