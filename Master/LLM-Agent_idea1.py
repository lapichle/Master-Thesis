import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, accuracy_score, precision_score, recall_score, mean_squared_error
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression

#For my master thesis, it's essential to develop a robust and flexible system that allows users to choose from different models, train them, and evaluate their performance using various metrics
#This idea is a more interactive approach where users can select models, and the agent will handle training, evaluation, and metrics reporting
#With this agent, the user has a User Model Selection, Flexible Target Selection, Evaluation Metrics, Cross-Validation Option 

# Step 1: Data Exploration
def data_exploration(df):
    print("Summary Statistics:")
    print(df.describe())
    print("\nMissing Values:")
    print(df.isnull().sum())
    print("\nData Types:")
    print(df.dtypes)

# Step 2: Model Suggestions Based on the Data
def suggest_models(df):
    print("Based on your dataset, here are some potential models:")
    print("- Logistic Regression: Suitable for binary/multiclass classification (e.g., predicting 'weapontype').")
    print("- Random Forest: Suitable for both classification and regression tasks with complex data.")
    print("- Decision Tree: Good for both classification and regression, provides interpretable models.")
    print("- Linear Regression: Suitable for predicting continuous variables (regression tasks).")
    print("- Random Forest Regressor: For more complex regression tasks.")
    print("- K-means Clustering: If you're looking to cluster similar incidents based on features.")
    print("- XGBoost: If you're working with tabular data")

# Step 3: Training the Model (Based on User Choice)
def train_model(df, target_col, model_choice, task_type='classification'):
    # Handling missing values and encoding categorical variables
    df_encoded = df.copy()

    # Impute missing values
    imputer = SimpleImputer(strategy='most_frequent')
    df_encoded = pd.DataFrame(imputer.fit_transform(df_encoded), columns=df_encoded.columns)
    
    # Encoding categorical variables
    for col in df_encoded.select_dtypes(include=['object']).columns:
        df_encoded[col] = LabelEncoder().fit_transform(df_encoded[col].astype(str))
    
    # Prepare the data
    X = df_encoded.drop(target_col, axis=1)
    y = df_encoded[target_col]
    
    # Splitting the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    # User Model Selection
    if task_type == 'classification':
        if model_choice == 'logistic_regression':
            model = LogisticRegression(max_iter=100000)
        elif model_choice == 'random_forest':
            model = RandomForestClassifier()
        elif model_choice == 'decision_tree':
            model = DecisionTreeClassifier()
        elif model_choice == 'xgboost':
            model = XGBClassifier()
        else:
            raise ValueError("Unsupported classification model selected.")
    elif task_type == 'regression':
        if model_choice == 'linear_regression':
            model = LinearRegression()
        elif model_choice == 'random_forest_regressor':
            model = RandomForestRegressor()
        else:
            raise ValueError("Unsupported regression model selected.")
    else:
        raise ValueError("Unsupported task type.")

    # Train the model
    model.fit(X_train, y_train)
    
    # Predict and Evaluate
    y_pred = model.predict(X_test)
    
    if task_type == 'classification':
        evaluate_classification(y_test, y_pred)
    elif task_type == 'regression':
        evaluate_regression(y_test, y_pred)

# Step 4: Evaluation Metrics for Classification
def evaluate_classification(y_test, y_pred):
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    print("Accuracy Score:", accuracy_score(y_test, y_pred))
    print("Precision Score:", precision_score(y_test, y_pred, average='weighted'))
    print("Recall Score:", recall_score(y_test, y_pred, average='weighted'))

# Step 5: Evaluation Metrics for Regression
def evaluate_regression(y_test, y_pred):
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    print("\nRegression Metrics:")
    print(f"Mean Squared Error: {mse}")
    print(f"Root Mean Squared Error: {rmse}")

# Example User Interaction

# 1. Explore the data
data = pd.read_csv('/Users/laurapichler/Desktop/Repository/Master-Thesis/Data/cleaned_ssdb_data_2022.csv')
data_exploration(data)

# 2. Suggest models
suggest_models(data)

# 3. User selects a model and a target variable
# Example: Logistic Regression on 'weapontype' column (classification task)
train_model(data, target_col='weapontype', model_choice='logistic_regression', task_type='classification')

# 4. Another example: Random Forest Regressor on 'reliability' column (regression task)
train_model(data, target_col='reliability', model_choice='random_forest_regressor', task_type='regression')
