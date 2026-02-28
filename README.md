# Algerian-_Forestfire_Model
🔥 Algerian Forest Fires – Machine Learning Regression Analysis 📌 Overview  This project performs predictive modeling on the Algerian Forest Fires Dataset to estimate fire weather index–related outcomes using multiple regression techniques.

The objective is to:

Perform structured data preprocessing

Analyze feature correlation

Apply feature encoding and scaling

Train multiple regression models

Evaluate and compare model performance




📂 Dataset Information

Dataset: Algerian Forest Fires Dataset
Source: Public wildfire dataset containing meteorological and fire index attributes

Features Include:

Temperature

Relative Humidity (RH)

Wind Speed (Ws)

Rain

FFMC

DMC

DC

ISI

BUI

FWI

Region / Class (Categorical)



🛠 Tech Stack

Python 3.x

NumPy

Pandas

Matplotlib

Seaborn

Scikit-Learn



📊 Project Workflow
1️⃣ Data Preprocessing

Loaded cleaned dataset

Removed inconsistencies

Converted categorical features using Label Encoding

Performed correlation analysis

Removed highly correlated features using custom threshold function

2️⃣ Train-Test Split
train_test_split(X, Y, test_size=0.2, random_state=42)

80% training data

20% testing data

Random state fixed for reproducibility

3️⃣ Feature Scaling

Standard scaling applied to normalize feature distributions.

Improves regression stability

Prevents bias due to feature magnitude differences

Visualization:

Boxplot before scaling

Boxplot after scaling

🤖 Models Implemented

The following regression models were trained and evaluated:

✔ Linear Regression
✔ Lasso Regression
✔ Ridge Regression
✔ ElasticNet Regression


📈 Evaluation Metrics

Models were evaluated using:

Mean Absolute Error (MAE)

R² Score

🔹 Best Observed Performance
Mean Absolute Error: 2.1428
R² Score: 0.8807
Interpretation

MAE ≈ 2.14 → Low average prediction error

R² ≈ 0.88 → Model explains 88% of variance

Indicates strong predictive capability



📉 Visualization

Correlation Heatmap

Feature Scaling Comparison (Boxplots)

Actual vs Predicted Scatter Plots

These visualizations help assess:

Feature relationships

Distribution normalization

Model accuracy and variance



📌 Key Insights

Feature scaling significantly improves regression performance

Regularization (Lasso, Ridge, ElasticNet) helps control overfitting

Linear Regression performs strongly on this dataset

Correlation filtering enhances model stability



🔮 Future Enhancements

Hyperparameter tuning (GridSearchCV)

Cross-validation strategy

Feature engineering

Ensemble methods (Random Forest, Gradient Boosting)

Model deployment (Flask / FastAPI)
