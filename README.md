# 📊 **HR Analytics with Decision Trees & Random Forests**

This project uses **Decision Trees** and **Random Forests** to predict employee **attrition** based on various factors like age, business travel, education level, years at the company, and other features. The goal is to help HR departments identify patterns that might indicate whether an employee is likely to leave the organization.

---

## 🧑‍💻 **Dataset Overview**

The dataset contains employee information and is used to predict the likelihood of **attrition** (whether an employee will leave the company). The columns include:

- **Age:** Employee's age.
- **Attrition:** Target variable (whether the employee left the company or not).
- **BusinessTravel:** Frequency of business travel.
- **DailyRate:** Daily wage.
- **Department:** Department the employee works in.
- **DistanceFromHome:** Distance from home to office.
- **Education:** Level of education.
- **EmployeeNumber:** Unique identifier for each employee.
- **JobSatisfaction:** Employee's satisfaction level with their job.
- **WorkLifeBalance:** Work-life balance rating.
- **YearsAtCompany:** Number of years the employee has worked at the company.
- **YearsInCurrentRole:** Number of years the employee has worked in their current role.
- And many more.

---

## 🔍 **Exploratory Data Analysis (EDA)**

Before building the model, we conducted **EDA** to understand the data better, check for missing values, and visualize important features that might impact employee attrition. Key steps include:

1. **Data Cleaning:**  
   We handled missing values, removed duplicates, and ensured data consistency.

2. **Visualization:**  
   We used histograms, bar plots, and correlation matrices to explore the relationships between different features like age, work-life balance, years at the company, and attrition rates.

3. **Feature Engineering:**  
   We transformed categorical variables (e.g., department, business travel) into numerical format using encoding techniques like **One-Hot Encoding**.

---

## 🌲 **Decision Tree Classifier**

### **Decision Trees** work by recursively splitting the dataset into subsets based on the feature that best separates the data. Each split creates a decision rule that allows us to classify data.

1. **Model Performance:**
   - **Initial Accuracy (Decision Tree Classifier):** 77%  
   - Decision trees are prone to overfitting, especially with noisy or complex datasets.

2. **Hyperparameter Tuning (Grid Search):**  
   We applied **hyperparameter tuning** (using **Grid Search**) to adjust parameters like tree depth and minimum samples per leaf, which helped the model generalize better.

   - **Tuned Model Accuracy:** 87%  
   - With hyperparameter tuning, the model's ability to predict employee attrition improved significantly, showing that it can better handle overfitting and make more accurate predictions.

---

## 🌳 **Random Forest Classifier**

### **Random Forests** are an ensemble of decision trees, trained on different subsets of data. It reduces overfitting by averaging the predictions of many decision trees, leading to better generalization.

1. **Model Performance:**
   - **Initial Accuracy (Random Forest):** 86%  
   - Random Forest typically performs better than a single decision tree due to its averaging mechanism.

2. **Hyperparameter Tuning (Randomized CV and Grid Search):**  
   We used **RandomizedSearchCV** for quicker hyperparameter optimization and **GridSearchCV** for more exhaustive search, adjusting parameters like **n_estimators**, **max_depth**, and **min_samples_split**.

   - **RandomizedSearchCV Accuracy:** 85.9%  
   - **Grid Search Accuracy:** 86%  
   - While Random Forest performed well out-of-the-box, the tuning process didn’t significantly improve the accuracy beyond the baseline due to the model's robustness.

---

## 🧰 **Key Algorithms Used**

### **1. Decision Tree Classifier:**
   - **How It Works:**  
     - Splits the dataset based on features to create a tree structure with decision nodes.
     - Each decision node tests a feature to classify data into different categories (Attrition: Yes/No).
   - **Pros:**  
     - Easy to interpret.
     - Handles both numerical and categorical data.
   - **Cons:**  
     - Prone to overfitting, especially with noisy data.

### **2. Random Forest Classifier:**
   - **How It Works:**  
     - Builds multiple decision trees using different random samples from the dataset.
     - Each tree predicts a result, and the random forest takes the majority vote as the final prediction.
   - **Pros:**  
     - Reduces overfitting compared to a single decision tree.
     - Handles large datasets with higher dimensionality.
   - **Cons:**  
     - Less interpretable than a single decision tree.

---

## 📈 **Model Performance Summary**

| Model                       | Accuracy  |
|-----------------------------|-----------|
| **Decision Tree (Base)**     | 77%       |
| **Decision Tree (Tuned)**    | 87%       |
| **Random Forest (Base)**     | 86%       |
| **Random Forest (Randomized CV)** | 85.9%  |
| **Random Forest (Grid Search)** | 86%     |

---

## ⚙️ **Tools and Techniques**

- **Libraries Used:** `pandas`, `NumPy`, `matplotlib`, `seaborn`, `scikit-learn`  
- **Modeling Techniques:**  
  - **Decision Tree Classifier**
  - **Random Forest Classifier**
  - **Hyperparameter Tuning:** Grid Search and Randomized CV

---

## 🌟 **Key Takeaways**

- **Decision Trees** are a good starting point for classification tasks like attrition prediction, but hyperparameter tuning is crucial to avoid overfitting and improve performance.
- **Random Forests** typically outperform single decision trees by leveraging the power of multiple trees and reducing variance.
- Hyperparameter tuning is important for both models to enhance accuracy and make them more robust against overfitting.
- The combination of **model selection** and **hyperparameter optimization** significantly improves prediction accuracy, aiding HR departments in better decision-making.
