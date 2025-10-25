# 🧠 Smart Deal Recommendation System

## 📌 Project Overview
The **Smart Deal Recommendation System** is a Machine Learning-based solution that predicts whether a user will **redeem a coupon or not**.  
It analyzes user demographics, travel patterns, lifestyle behaviors, and deal-related attributes to make personalized recommendations.

This system helps businesses **target users effectively**, increasing conversion rates and improving marketing ROI.

---

## 🎯 Objectives
- To analyze user and coupon-related data.
- To preprocess and prepare the dataset for model training.
- To build and compare multiple machine learning models.
- To identify the most accurate model for predicting coupon redemption.
- To make predictions for new users based on their profile and behavior.

---

## ⚙️ Methodology
1. **Data Preprocessing**
   - Loaded and cleaned the dataset (`smart_deal_recommendations.csv`).
   - Handled missing values and irrelevant columns.
   - Encoded categorical variables using `OneHotEncoder`/`get_dummies`.
   - Split the data into training (80%) and testing (20%) sets.

2. **Model Training**
   Trained and evaluated three models:
   - **Logistic Regression**
   - **Random Forest Classifier**
   - **XGBoost Classifier**

3. **Evaluation Metrics**
   - Accuracy  
   - Precision  
   - Recall  
   - F1-Score  
   - ROC-AUC Score  
   - Confusion Matrix Visualization

4. **Prediction**
   - A function `predict_coupon_redeem()` is used to predict whether a **new user** will redeem a coupon or not.
   - The model outputs both the **class** (Redeem/Not Redeem) and **probability**.

---

## 📊 Results Summary

| Model | Accuracy | ROC-AUC | F1-Score | Remarks |
|--------|-----------|----------|-----------|----------|
| Logistic Regression | 0.683 | 0.733 | 0.68 | Baseline model |
| Random Forest | 0.744 | 0.812 | 0.74 | Improved performance |
| XGBoost | **0.747** | **0.820** | **0.74** | ✅ Best model |

---

## 🧠 Best Model: XGBoost
- **Accuracy:** 74.7%  
- **ROC-AUC Score:** 0.82  
- **Observation:** The model effectively differentiates between users likely and unlikely to redeem offers.

---

## 🧩 How to Run the Project

### Step 1: Clone or Download
```bash
git clone https://github.com/yourusername/smart-deal-recommendation.git
Step 2: Install Dependencies
Step 3: Run the Notebook
jupyter notebook Smart_Deal_Recommendations.ipynb
Step 4: Predict for a New User
predict_coupon_redeem(new_user, xgb_model, X_train)
Sample Output:🎯 Predicted Class: Redeem ✅
🔮 Redemption Probability: 0.53

🧰 Technologies Used

Python 3.11

pandas – Data processing

NumPy – Numerical operations

scikit-learn – Model training and evaluation

XGBoost – Gradient boosting model

Matplotlib & Seaborn – Visualization

🏁 Conclusion

The project successfully predicts coupon redemption behavior.

Among all models, XGBoost delivered the best overall accuracy and robustness.

Businesses can use this model to target high-probability customers with relevant deals, improving engagement and sales.
