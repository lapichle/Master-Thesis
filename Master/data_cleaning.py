import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


# Load the Excel file that contains multiple sheets
file_path = "/Users/laurapichler/Desktop/Repository/Master-Thesis/Data/ssdb_data_2022.xlsx"

# Load all sheets into separate DataFrames
xls = pd.ExcelFile(file_path)
sheet_names = xls.sheet_names

# Combine all the dataframes into one
dataframes = [xls.parse(sheet_name) for sheet_name in sheet_names]
combined_df = pd.concat(dataframes, ignore_index=True)

# Looking at the data
print("Initial Data Information:\n")
print(combined_df.info())
print("\nMissing Value Count:\n")
print(combined_df.isnull().sum())

# Cleaning the Dataset

# Handle Missing Values
# Drop columns with too many missing values or fill them with appropriate values.
threshold = 0.1  # If more than 10% of the values are missing, drop the column
combined_df = combined_df.dropna(axis=1, thresh=int(threshold * len(combined_df)))

# Fill remaining missing values with median for numeric or mode for categorical
for column in combined_df.columns:
    if combined_df[column].dtype in ['int64', 'float64']:
        combined_df[column].fillna(combined_df[column].median(), inplace=True)
    else:
        combined_df[column].fillna(combined_df[column].mode()[0], inplace=True)

# Remove Duplicates
combined_df.drop_duplicates(inplace=True)

# Standardizing Column Names
combined_df.columns = [col.strip().lower().replace(' ', '_') for col in combined_df.columns]

# Handle Inconsistent Data Types
# Convert object columns with numeric-like values to numeric type
for column in combined_df.columns:
    if combined_df[column].dtype == 'object':
        try:
            combined_df[column] = pd.to_numeric(combined_df[column])
        except ValueError:
            pass  

# Handling Categorical Variables for LLM

# Converting categorical columns to string type
for column in combined_df.columns:
    if combined_df[column].dtype == 'object':
        combined_df[column] = combined_df[column].astype(str)

# Saving the Cleaned Dataset to a New CSV
output_file_path = "/Users/laurapichler/Desktop/Repository/Master-Thesis/Data/cleaned_ssdb_data_2022.csv"
combined_df.to_csv(output_file_path, index=False)

output_file_path
