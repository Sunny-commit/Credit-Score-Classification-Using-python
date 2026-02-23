# 💳 Credit Score Classification - Financial Machine Learning

A **machine learning system for credit score classification** that categorizes borrowers into credit tiers based on financial behavior and credit profile.

## 🎯 Overview

This project provides:
- ✅ Credit score classification (Good/Average/Poor)
- ✅ Feature importance analysis
- ✅ Class imbalance handling
- ✅ Financial feature engineering
- ✅ Credit tier prediction
- ✅ ROC-AUC evaluation

## 📊 Credit Dataset

```python
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler

class CreditDataAnalysis:
    """Analyze credit data"""
    
    def __init__(self, filepath='credit_data.csv'):
        self.df = pd.read_csv(filepath)
    
    def explore_credit_profile(self):
        """Analyze credit characteristics"""
        print(f"Total records: {len(self.df)}")
        print(f"\nCredit score distribution:")
        print(self.df['Credit_Score'].value_counts().sort_index())
        
        print(f"\nClass balance:")
        print(self.df['Credit_Score'].value_counts(normalize=True))
    
    def financial_metrics(self):
        """Analyze financial features"""
        print("\nIncome statistics:")
        print(self.df['Income'].describe())
        
        print("\nDebt statistics:")
        print(self.df['Debt'].describe())
        
        # Debt to income ratio
        self.df['Debt_to_Income'] = self.df['Debt'] / (self.df['Income'] + 1)
        print(f"\nDebt-to-income ratio:")
        print(self.df['Debt_to_Income'].describe())
```

## 🔧 Feature Engineering

```python
class CreditFeatureEngineer:
    """Engineer credit features"""
    
    def __init__(self):
        self.scaler = StandardScaler()
        self.encoders = {}
    
    def create_financial_features(self, df):
        """Create financial metrics"""
        df_copy = df.copy()
        
        # Debt to income
        df_copy['Debt_to_Income_Ratio'] = df_copy['Debt'] / (df_copy['Income'] + 1)
        
        # Savings to debt ratio
        df_copy['Savings_to_Debt_Ratio'] = df_copy['Savings'] / (df_copy['Debt'] + 1)
        
        # Monthly debt payment ability
        df_copy['Monthly_Debt_Capacity'] = (df_copy['Income'] / 12) - (df_copy['Debt'] / df_copy['Loan_Term_Months'] + 1)
        
        # Credit utilization (if credit limit available)
        if 'Credit_Limit' in df_copy.columns:
            df_copy['Credit_Utilization'] = df_copy['Outstanding_Debt'] / (df_copy['Credit_Limit'] + 1)
        
        return df_copy
    
    def create_behavioral_features(self, df):
        """Behavior-based features"""
        df_copy = df.copy()
        
        # Payment history score (1-10 scale)
        df_copy['Payment_History_Score'] = df_copy['Payment_Behavior'].apply(
            lambda x: 10 if x == 'Good' else 5 if x == 'Average' else 1
        )
        
        # Default risk
        df_copy['Default_Risk'] = (df_copy['Previous_Defaults'] > 0).astype(int)
        
        # Account age (years)
        df_copy['Account_Age_Years'] = df_copy['Account_Age_Months'] / 12
        
        return df_copy
    
    def create_account_features(self, df):
        """Account characteristics"""
        df_copy = df.copy()
        
        # Active accounts
        df_copy['Active_Accounts_Count'] = df_copy['Active_Accounts']
        
        # Bank account types
        df_copy['Account_Type_Score'] = df_copy['Account_Type'].map({
            'Savings': 3,
            'Checking': 2,
            'Money Market': 4,
            'Credit Card': 2
        })
        
        # Account diversity
        df_copy['Account_Diversity'] = df_copy['Num_of_Loan_and_Advance_Accounts'] + \
                                      df_copy['Num_of_Credit_Card_Accounts']
        
        return df_copy
```

## 🤖 Classification Models

```python
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from imblearn.over_sampling import SMOTE

class CreditClassifier:
    """Classify credit scores"""
    
    def __init__(self):
        self.models = self._build_models()
        self.smote = SMOTE(random_state=42)
    
    def _build_models(self):
        """Initialize models"""
        return {
            'Logistic Regression': LogisticRegression(max_iter=1000, multi_class='multinomial'),
            'Random Forest': RandomForestClassifier(n_estimators=100, max_depth=15, random_state=42),
            'Gradient Boosting': GradientBoostingClassifier(n_estimators=100, learning_rate=0.05, random_state=42),
            'SVM': SVC(kernel='rbf', probability=True, random_state=42)
        }
    
    def handle_imbalance(self, X_train, y_train):
        """Apply SMOTE for imbalanced classes"""
        X_resampled, y_resampled = self.smote.fit_resample(X_train, y_train)
        
        print(f"Original distribution:")
        print(pd.Series(y_train).value_counts())
        print(f"\nAfter SMOTE:")
        print(pd.Series(y_resampled).value_counts())
        
        return X_resampled, y_resampled
    
    def train_all(self, X_train, y_train):
        """Train all models"""
        trained = {}
        
        for name, model in self.models.items():
            model.fit(X_train, y_train)
            trained[name] = model
        
        self.models = trained
        return trained
    
    def predict_credit_tier(self, X_test):
        """Predict credit tier with confidence"""
        predictions = {}
        
        for name, model in self.models.items():
            pred_class = model.predict(X_test)
            pred_proba = model.predict_proba(X_test)
            predictions[name] = {
                'Class': pred_class,
                'Probability': pred_proba.max(axis=1)
            }
        
        return predictions
```

## 📊 Evaluation & Interpretation

```python
from sklearn.metrics import (
    classification_report, confusion_matrix, roc_auc_score,
    roc_curve, auc
)

class CreditEvaluator:
    """Evaluate credit classification"""
    
    @staticmethod
    def evaluate_model(y_true, y_pred):
        """Classification metrics"""
        print("Classification Report:")
        print(classification_report(y_true, y_pred, 
                                   target_names=['Good', 'Average', 'Poor']))
        
        print("\nConfusion Matrix:")
        cm = confusion_matrix(y_true, y_pred)
        print(cm)
        
        return cm
    
    @staticmethod
    def feature_importance(model, feature_names):
        """Extract important features"""
        if hasattr(model, 'feature_importances_'):
            importance_df = pd.DataFrame({
                'Feature': feature_names,
                'Importance': model.feature_importances_
            }).sort_values('Importance', ascending=False)
            
            print("\nTop 10 Important Features:")
            print(importance_df.head(10))
            
            return importance_df
    
    @staticmethod
    def plot_confusion_matrix(y_true, y_pred):
        """Visualize confusion matrix"""
        cm = confusion_matrix(y_true, y_pred)
        
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=['Good', 'Average', 'Poor'],
                   yticklabels=['Good', 'Average', 'Poor'])
        plt.title('Credit Score Classification')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.show()
```

## 💡 Interview Talking Points

**Q: Handle class imbalance?**
```
Answer:
- SMOTE oversampling
- Class weights in models
- Different metrics (F1, AUC-ROC)
- Stratified cross-validation
```

**Q: Interpretability for regulation?**
```
Answer:
- Feature importance analysis
- SHAP values for predictions
- Fair lending checks
- Decision rules visualization
```

## 🌟 Portfolio Value

✅ Credit classification
✅ Financial feature engineering
✅ Class imbalance handling
✅ Multiple algorithms
✅ Model interpretability
✅ Financial domain knowledge
✅ Regulatory compliance

---

**Technologies**: Scikit-learn, Imbalanced-learn, Pandas, NumPy

