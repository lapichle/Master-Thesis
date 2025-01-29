#Importing all necessary packages
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
from sklearn.preprocessing import LabelEncoder

#In my Master Thesis I want to create a LLM-Agent capable of suggesting statistical models
#The user should be able to provide a dataset, with which the agent will be able to work with 
#The agent should be able to explore the dataset, suggest statistical models and 
#Train models on the data if applicable (for example, predicting incident characteristics or classifications based on available variables).
#In this example I will use the data of the K-12 Dataset to train and test the agent


#Lets start with the data exploration 
#The agent can help with basic exploration of the dataset (e.g., summary statistics, correlations, missing values)

#Data Exploration: The data_exploration() function provides an overview of the dataset (missing values, data types, etc.).
#Model Suggestions: The suggest_models() function recommends statistical models based on the dataset's structure.
#Training Logistic Regression: The train_logistic_regression() function is an example of how to encode the data, split it, and train a logistic regression model. A user can adapt it to other models (Random Forest, Decision Trees, etc.).


# Loading the data
data = pd.read_csv('/Users/laurapichler/Desktop/Repository/Master-Thesis/Data/cleaned_ssdb_data_2022.csv')

# Step 1: Data Exploration
def data_exploration(df):
    print("Summary Statistics:")
    print(df.describe())
    print("\nMissing Values:")
    print(df.isnull().sum())
    print("\nData Types:")
    print(df.dtypes)

#Furthermore, I want the agent to suggest statistical models to the user, this agent will provide,
#Logistic Regression, Random Forest, Linear Regression and K-Means Clustering

# Step 2: Suggest Statistical Models
def suggest_models(df):
    print("Based on the data, here are some potential models:")
    print("- Logistic Regression: If predicting categorical outcomes (e.g., weapon type, shooter outcome).")
    print("- Random Forest: For more complex classification tasks, especially if there are many features.")
    print("- Linear Regression: If predicting continuous variables (e.g., incident severity).")
    print("- K-means Clustering: If you're looking to cluster similar incidents based on features.")

# In the next step I will train a Logistic Regression for Classification as an example 

# Step 3: Train Logistic Regression for Classification
def train_logistic_regression(df, target_col):
    # Encoding categorical variables
    df_encoded = df.copy()
    for col in df_encoded.select_dtypes(include=['object']).columns:
        df_encoded[col] = LabelEncoder().fit_transform(df_encoded[col].astype(str))
    
    # Prepare data
    X = df_encoded.drop(target_col, axis=1)
    y = df_encoded[target_col]

    # Splitting the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    # Logistic Regression Model
    model = LogisticRegression(max_iter=100000)
    model.fit(X_train, y_train)
    
    # Predictions and Evaluation
    y_pred = model.predict(X_test)
    print("Accuracy Score:", accuracy_score(y_test, y_pred))
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))

# Example Use of the Agent
data_exploration(data)  # Explore the data
suggest_models(data)    # Suggest statistical models

# Train a logistic regression model on a sample target column (e.g., 'weapontype')
train_logistic_regression(data, 'weapontype')

