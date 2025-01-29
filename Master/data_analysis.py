import pandas as pd
import numpy as np 

# Load the data to examine its contents
file_path = '/Users/laurapichler/Desktop/Repository/Master-Thesis/Data/cleaned_ssdb_data_2022.csv'
data = pd.read_csv(file_path)

# Display the first few rows of the dataset to understand its structure
print(data.head())

# Let's perform an analysis: investigating the relationship between gender, race, and shooter outcomes

# Grouping the data by gender and race to analyze their relation with the shooter's outcome
# Focusing on the relevant columns: 'gender', 'race', and 'shooteroutcome'

# First, let's get a count of shooter outcomes based on gender and race
outcome_by_gender_race = data.groupby(['gender', 'race', 'shooteroutcome']).size().reset_index(name='count')

# Pivoting the data for clearer analysis
pivot_outcome_gender_race = outcome_by_gender_race.pivot_table(index=['gender', 'race'], columns='shooteroutcome', values='count', fill_value=0)

print(pivot_outcome_gender_race)