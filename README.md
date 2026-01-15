# 🏗️ Construction Budget Prediction System

A machine learning-based predictive model designed to estimate construction project costs. This project leverages historical data to help project managers and stakeholders forecast budgets with higher accuracy.

---

## 📌 Project Overview
Estimating construction costs is notoriously difficult due to variables like material price fluctuations, labor costs, and project scope. This project uses **Machine Learning** to identify patterns in past projects and predict the budget for future ones.

* **Development Environment:** PyCharm & Jupyter Notebook
* **Libraries:** Pandas, Scikit-Learn, NumPy
* **Frontend:** Streamlit for real-time cost estimation



---

## 🛠️ Tech Stack

| Category | Tool |
| :--- | :--- |
| **Language** | Python 3.10+ |
| **IDE** | PyCharm (Application) / Jupyter (Research) |
| **Preprocessing** | NLTK (for analyzing project descriptions/notes) |
| **ML Models** | Linear Regression / Random Forest / XGBoost |
| **UI Framework** | Streamlit |

---

## ⚙️ Methodology

1. **Data Collection:** Sourced historical construction data (Kaggle/Custom CSV).
2. **Feature Engineering:** * Encoding categorical variables (Location, Building Type, Materials).
    * Scaling numerical features (Square Footage, Number of Floors).
3. **NLP Integration (NLTK):** Processed "Project Scope" text data to extract cost-driving keywords.
4. **Model Training:** Compared multiple regression models to find the one with the lowest **Mean Absolute Error (MAE)**.



---

## 🚀 How to Run

### 1. Clone the Repository
```bash
https://github.com/Adityanjr11/Minor-Project.git
