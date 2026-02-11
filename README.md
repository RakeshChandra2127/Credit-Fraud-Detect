# 💳 Credit Card Fraud Detection using Machine Learning & Anomaly Detection

## 📌 Project Overview

This project focuses on detecting fraudulent credit card transactions using:

- Exploratory Data Analysis (EDA)
- Correlation Analysis
- Data Preprocessing
- Anomaly Detection Algorithms (PyOD)
- Outlier Visualization Techniques

Credit card fraud detection is a highly imbalanced classification problem where fraudulent transactions represent a very small percentage of total transactions.

Dataset Source: Kaggle Credit Card Fraud Dataset  
https://www.kaggle.com/mlg-ulb/creditcardfraud

---

## 📂 Project Structure

```
Credit-Fraud-Detect/
│
├── Credit Card Fraud Detection.ipynb
├── CreditCardFraud.ipynb
├── Model_Final.ipynb
├── fraud_credit.py
├── creditcard.csv
├── requirements.txt
├── LICENSE
└── README.md
```

---

## 📊 Dataset Information

- 284,807 transactions
- 492 fraud cases (~0.17%)
- PCA-transformed features (V1–V28)
- Raw features:
  - Time
  - Amount
  - Class (0 = Normal, 1 = Fraud)

This dataset is extremely imbalanced, making accuracy a misleading metric.

---

## 🔎 Exploratory Data Analysis (EDA)

### 1️⃣ Scatter Plot: Amount vs Class
Visualizes transaction distribution and fraud concentration.

### 2️⃣ Correlation Heatmap
Shows feature relationships and highlights potential patterns.

### 3️⃣ Class Distribution
Confirms severe imbalance between normal and fraudulent transactions.

---

## ⚙️ Data Preprocessing

- Feature scaling using MinMaxScaler
- Reshaping data for anomaly detection
- Feature concatenation
- Outlier fraction set to 5%

---

## 🤖 Models Used (PyOD Library)

The following anomaly detection models were implemented:

- ABOD (Angle-based Outlier Detector)
- CBLOF (Cluster-based Local Outlier Factor)
- Feature Bagging
- Isolation Forest
- KNN
- Average KNN

Each model:

- Fits on scaled data
- Computes anomaly score
- Predicts inliers vs outliers
- Visualizes decision boundary

---

## 🧠 Core Implementation (fraud_credit.py)

Below is the main implementation code used for anomaly detection:

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pyod.models.abod import ABOD
from pyod.models.cblof import CBLOF
from pyod.models.feature_bagging import FeatureBagging
from pyod.models.iforest import IForest
from pyod.models.knn import KNN
from pyod.models.lof import LOF
from scipy import stats
from sklearn.preprocessing import MinMaxScaler
import seaborn as sns
import matplotlib

# Load dataset
df = pd.read_csv("creditcard.csv")

# Scatter plot
df.plot.scatter('Amount', 'Class')
plt.show()

# Correlation heatmap
corr = df.corr()
sns.heatmap(corr, cmap="gray")
plt.title("Imbalanced Correlation Matrix")
plt.show()

# Scaling
scaler = MinMaxScaler(feature_range=(0,1))
df[['Amount']] = scaler.fit_transform(df[['Amount']])

X1 = df['Amount'].values.reshape(-1,1)
X2 = df['Class'].values.reshape(-1,1)
X = np.concatenate((X1, X2), axis=1)

random_state = np.random.RandomState(42)
outliers_fraction = 0.05

classifiers = {
    'ABOD': ABOD(contamination=outliers_fraction),
    'CBLOF': CBLOF(contamination=outliers_fraction, random_state=random_state, check_estimator=False),
    'Feature Bagging': FeatureBagging(LOF(n_neighbors=35), contamination=outliers_fraction),
    'Isolation Forest': IForest(contamination=outliers_fraction, random_state=random_state),
    'KNN': KNN(contamination=outliers_fraction),
    'Average KNN': KNN(method='mean', contamination=outliers_fraction)
}

for clf_name, clf in classifiers.items():
    clf.fit(X)
    y_pred = clf.predict(X)
    print(clf_name, "Outliers:", np.count_nonzero(y_pred))
```

---

## 📈 Key Observations

- Fraud cases are sparse but detectable using anomaly detection.
- Isolation Forest and KNN performed effectively.
- Correlation heatmap reveals PCA features reduce linear interpretability.
- Class imbalance significantly affects supervised learning approaches.

---

## 🛠️ Technologies Used

- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn
- PyOD
- Jupyter Notebook

---

## 🚀 How to Run

### 1️⃣ Clone the Repository

```bash
git clone https://github.com/RakeshChandra2127/Credit-Fraud-Detect.git
cd Credit-Fraud-Detect
```

### 2️⃣ Install Dependencies

```bash
pip install -r requirements.txt
```

If needed:

```
numpy
pandas
matplotlib
seaborn
scikit-learn
pyod
jupyter
```

### 3️⃣ Run Notebook

```bash
jupyter notebook
```

Open and run:

```
Credit Card Fraud Detection.ipynb
```

---

## 🎯 Key Learnings

- Fraud detection is an extreme class imbalance problem.
- Accuracy is not suitable for evaluation.
- Anomaly detection models can effectively identify rare patterns.
- Feature scaling significantly impacts model performance.

---

## 🔮 Future Improvements

- Implement supervised models (Logistic Regression, XGBoost)
- Use SMOTE for balancing
- Add ROC-AUC and Precision-Recall analysis
- Deploy as real-time fraud detection API
- Compare supervised vs unsupervised approaches

---

## 👤 Author

**Rakesh Chandra Behera**  
Integrated MSc Chemistry  
NIT Rourkela  
Aspiring Data Analyst / Data Scientist  

---

## 📌 License

This project is licensed under the MIT License.
