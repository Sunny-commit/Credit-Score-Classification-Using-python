# 🧠 Credit Score Classification Using Python

This project classifies credit scores using supervised machine learning techniques and visual exploratory analysis. It leverages financial data such as income, loan history, and bank behavior to predict whether a person’s credit score is **Poor**, **Standard**, or **Good**.

## 📁 Dataset

The project utilizes a dataset named `train.csv`, which includes:

* Annual income
* Monthly in-hand salary
* Number of bank accounts
* Credit history
* Number of credit cards
* Loan details
* Payment behavior

The target variable is `Credit_Score`.

## 🧰 Libraries Used

* **Pandas** & **NumPy** – for data manipulation
* **Plotly** – for interactive data visualization
* **Scikit-learn** – for building and evaluating machine learning models

## 🔍 Exploratory Data Analysis (EDA)

EDA is performed using Plotly to generate interactive box plots and distribution graphs, revealing how different features like `Occupation`, `Annual_Income`, `Num_Bank_Accounts`, etc., relate to credit scores.

Example Plots:

* Credit Score vs Occupation
* Credit Score vs Annual Income
* Credit Score vs Monthly In-hand Salary
* Credit Score vs Number of Credit Cards

## 🧪 Machine Learning Workflow

1. **Data Preprocessing**:

   * Handle missing values
   * Encode categorical variables
   * Feature selection

2. **Model Training**:

   * Multiple classification models are trained (e.g., Logistic Regression, Random Forest, etc.)
   * Hyperparameter tuning

3. **Evaluation Metrics**:

   * Accuracy
   * Confusion Matrix
   * Classification Report

## 🚀 How to Run

1. Clone this repository.
2. Ensure the dataset `train.csv` is in the same directory.
3. Run the Jupyter Notebook:

   ```bash
   jupyter notebook "Credit Score Classification Using python.ipynb"
   ```
4. Explore the visualizations and model outputs.

## 🧩 Folder Structure

```
📦Credit Score Classification
 ┣ 📜Credit Score Classification Using python.ipynb
 ┣ 📜train.csv
 ┗ 📜README.md
```

## 🏁 Future Improvements

* Include additional ensemble models
* Deploy as a web app for interactive user input
* Use SHAP values for model explainability

## 📬 Contact

For any questions or collaborations, feel free to reach out!

---

