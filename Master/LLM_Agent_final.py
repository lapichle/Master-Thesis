import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, accuracy_score, precision_score, recall_score
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer
from imblearn.over_sampling import SMOTE  
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

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
    print("- XGBoost: If you're working with tabular data.")

from imblearn.over_sampling import SMOTE

# Step 3; Train model with Smote

def train_model_with_smote(df, target_col, model_choice, task_type='classification'):
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
    
    # Applying SMOTE for class balancing if classification
    if task_type == 'classification':
        smote = SMOTE(random_state=42, k_neighbors=1)  # Set k_neighbors=1 for rare classes
        X_train, y_train = smote.fit_resample(X_train, y_train)

    # User Model Selection
    if task_type == 'classification':
        if model_choice == 'logistic_regression':
            model = LogisticRegression(max_iter=100000)
        elif model_choice == 'random_forest':
            model = RandomForestClassifier(random_state=42)
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
        plot_confusion_matrix(y_test, y_pred, classes=np.unique(y))
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
    rmse = np.sqrt(y_test, y_pred)
    print("\nRegression Metrics:")
    print(f"Root Mean Squared Error: {rmse}")


# Step 6: Confusion Matrix Visualization
def plot_confusion_matrix(y_test, y_pred, classes):
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(6, 5))
    plt.title("Confusion Matrix")
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

    # Adding text annotations
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, format(cm[i, j], 'd'),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > cm.max() / 2. else "black")

    plt.tight_layout()
    plt.show()


# Example Usage
data = pd.read_csv('/Users/laurapichler/Desktop/Repository/Master-Thesis/Data/cleaned_ssdb_data_2022.csv')

# 1. Explore the data
data_exploration(data)

# 2. Suggest models
suggest_models(data)

# 3. Train and evaluate with SMOTE
train_model_with_smote(data, target_col='weapontype', model_choice='random_forest', task_type='classification')

from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

# Generate Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=np.unique(y))
disp.plot(cmap=plt.cm.Blues)
plt.title("Confusion Matrix for `weapontype` Predictions")
plt.show()


# Feature Importance Plot
importances = model.feature_importances_
features = X_train.columns
plt.figure(figsize=(10, 6))
plt.barh(features, importances, color='skyblue')
plt.xlabel("Importance")
plt.ylabel("Feature")
plt.title("Feature Importance for `weapontype` Predictions")
plt.show()