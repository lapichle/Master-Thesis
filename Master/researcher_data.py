#The research paper that I want to replicate with the help of my agent provides data about each researcher who participated in the research
#Using this data I can create agents that are based on the personas of the researchers and see which models they would choose based on the information they get on the person 
#Furthermore I want compare the model selection of my agents for each persona 
#And the model they actually chose in the paper to answer the question 
import pandas as pd 
import numpy as np 

file_path = '/Users/laurapichler/Desktop/dataverse_files/cri_survey_long_public.dta'
data = pd.read_stata(file_path, convert_categoricals= False)

#Missing values indicated throughout the codebook by “.a”

print(data.describe())
print(data.info())


import pandas as pd


# List of selected columns based on your criteria
selected_columns = [
    'u_id',
    'backgr_degree',               # Area of Highest Degree
    'v_18',                        # Published on Statistics/Methods
    'backgr_exp_famil_mlm',        # Familiarity with Multilevel Modelling
    'backgr_exp_teach_stat',       # Teaching Statistics
    'belief_H1_1', 'belief_H1_2', 'belief_H1_3',  # Personal Belief about Hypothesis
    'belief_certainty_1', 'belief_certainty_2', 'belief_certainty_3',  # Certainty in Belief about Immigration Hypothesis
    'v_33',                        # Perceived Individual Time Spent on Replication
    'v_98', 'v_99', 'v_100',       # Constraints due to Time and Resources
    'v_41',                        # Enjoyment of Replication
    'v_88', 'v_90', 'v_94',        # Motivational Factors
    'v_110'                        # Gender
]

# Selecting the chosen columns and storing them in a new DataFrame
selected_data = data[selected_columns]

# Save the selected data to a new CSV file
output_path = '/Users/laurapichler/Desktop/Repository/Master-Thesis/Data/research_data.csv'  
selected_data.to_csv(output_path, index=False)

print(f"Selected data saved to {output_path}")

#Perfect, this is stored now, now let's clean our data 

