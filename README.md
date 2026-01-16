# 🛍️ Online Shoppers Purchasing Intention Prediction

## 📖 Project Overview
This project aims to predict whether an online store visitor will make a purchase based on their browsing behavior. By identifying high-intent users, e-commerce platforms can optimize their marketing strategies and resource allocation.

The solution features a robust machine learning pipeline that handles **class imbalance**, performs **advanced feature engineering**, and optimizes decision thresholds based on **business costs**.

## 💡 Key Highlights
- **Business-Centric Optimization**: Implemented a cost-sensitive learning approach (Cost of False Negative = 5x Cost of False Positive) to optimize decision thresholds, minimizing expected revenue loss.
- **Advanced Feature Engineering**: Created composite behavioral metrics such as *Interaction Efficiency* and *Exit-Bounce differentials* to capture user intent more accurately.
- **Leakage-Free Pipeline**: Utilized `imbalanced-learn` pipelines to ensure SMOTE oversampling applies only to training data, maintaining rigorous validation integrity.

## 🛠️ Tech Stack
- **Python**: Pandas, NumPy
- **Visualization**: Matplotlib, Seaborn
- **Machine Learning**: Scikit-learn, XGBoost, Imbalanced-learn (SMOTE)
- **Workflow**: Jupyter Notebook

## 📊 Methodology
1. **EDA**: Analyzed session duration, bounce rates, and special day impacts.
2. **Preprocessing**: Applied Yeo-Johnson transformation for skewed features and One-Hot Encoding for categorical variables.
3. **Modeling**: Benchmarked Random Forest, XGBoost, and MLP. **Random Forest** was selected as the champion model.
4. **Post-Processing**: Adjusted classification thresholds based on a custom cost matrix to align with business profitability.

## 🚀 Results
- **Final Model**: Random Forest
- **F1-Score (Class 1)**: ~0.69 (at optimized threshold)
- **ROC AUC**: ~0.94
- **Strategic Insight**: A "November-specific" threshold policy was analyzed to handle seasonal traffic spikes.

## 📂 File Structure
- `notebooks/01_Data_Exploration.ipynb`: Data cleaning and exploratory analysis.
- `notebooks/02_Modeling_and_Evaluation.ipynb`: Full ML pipeline, training, and cost analysis.
- `presentation/`: Project slides summarizing the business value.

## 🔧 How to Run
1. Clone the repository:
   ```bash
   git clone [https://github.com/YourUsername/shopper-intention-prediction.git](https://github.com/YourUsername/shopper-intention-prediction.git)

2. Install dependencies
    ```bash
    pip install -r requirements.txt

3. Run the notebooks in order
