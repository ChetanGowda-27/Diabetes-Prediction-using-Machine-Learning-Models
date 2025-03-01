

# **Diabetes Prediction using Machine Learning**

### **📊 Overview**
Diabetes is a growing global health concern, and early detection can be crucial in reducing its long-term effects. This project explores machine learning models to predict the likelihood of an individual developing diabetes based on various medical attributes. By using algorithms like **K-Nearest Neighbors (KNN)**, **Random Forest (RF)**, and **Support Vector Machines (SVM)**, we aim to provide an early warning system to assist in medical decision-making.

---

### **🔧 Technologies Used**
- **Python**: Core programming language for this project.
- **Libraries**:  
  - **Pandas**: Data manipulation and analysis.
  - **Scikit-learn**: Machine learning models and evaluation.
  - **Seaborn / Matplotlib**: Data visualization.
  - **NumPy**: Numerical operations.

---

### **🚀 Features**
- **Data Preprocessing**: Features like BMI and glucose are engineered to enhance model performance.
- **Hyperparameter Tuning**: The models are optimized using **GridSearchCV** to find the best hyperparameters.
- **Model Evaluation**: Performance is evaluated based on metrics such as **Accuracy**, **Precision**, **Recall**, and **F1-Score**.
- **Visualization**: Confusion matrices and classification reports are visualized for in-depth model performance analysis.

---

### **💻 How to Run This Project**

#### **1. Clone the Repository**
Start by cloning the repository to your local machine using the following command:
```bash
git clone https://github.com/yourusername/diabetes-prediction.git
```

#### **2. Install Dependencies**
Ensure you have Python 3.8+ installed. Then, install the required libraries using `pip`:
```bash
pip install -r requirements.txt
```

#### **3. Prepare the Dataset**
The dataset used in this project is the **Pima Indians Diabetes Database**, available publicly. Ensure you have the dataset in the `data/` folder (or update the path to where it’s stored).

#### **4. Run the Project**
After setting up, run the **`diabetes_prediction.py`** script to train the models and evaluate them:
```bash
python diabetes_prediction.py
```

#### **5. View Results**
The models will output performance metrics and display visualizations such as confusion matrices for each model. You can compare the **Accuracy**, **Precision**, **Recall**, and **F1-Score** for each model.

---

### **📈 Results**

The following table summarizes the performance of each model:

| Model           | Accuracy | Precision | Recall | F1-Score |
|-----------------|----------|-----------|--------|----------|
| **KNN**         | 0.74     | 0.75      | 0.73   | 0.74     |
| **Random Forest** | 0.78  | 0.79      | 0.77   | 0.78     |
| **SVM**         | 0.75     | 0.75      | 0.75   | 0.75     |

- **Best Performing Model**: Random Forest outperforms KNN and SVM in accuracy and F1-score.
- **Key Insights**: All models demonstrated promising results, with Random Forest being the most accurate and balanced in terms of other metrics.

---

### **🔍 Project Walkthrough**

1. **Data Preprocessing**:  
   - Loaded the dataset and performed **feature engineering** to add new features like BMI squared and Age × Glucose interaction.
   - Scaled features to standardize them before feeding them into the models.
   
2. **Model Training**:  
   - **KNN**: Tuned the number of neighbors and distance metric.
   - **Random Forest**: Tuned the number of trees, tree depth, and minimum samples per split.
   - **SVM**: Tuned regularization, kernel types, and gamma.

3. **Model Evaluation**:  
   - Used **GridSearchCV** to find the optimal hyperparameters.
   - Evaluated the models using metrics like **Confusion Matrix**, **Precision**, **Recall**, and **F1-Score**.

---

### **🛠️ Contribution**

Feel free to contribute to the project by:
- Reporting issues.
- Forking the repository and submitting pull requests.
- Improving the data preprocessing pipeline.
- Adding additional evaluation metrics and visualizations.

---

### **📑 License**
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

### **📢 Acknowledgments**
- The **Pima Indians Diabetes Database** is publicly available and used for research purposes.
- The project makes use of the **Scikit-learn** library for machine learning and **Matplotlib/Seaborn** for visualization.

---


