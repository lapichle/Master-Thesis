import pandas as pd 
import numpy as np 
import random
import json 
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# Loading the CSV file
file_path = '/Users/laurapichler/Desktop/Repository/Master-Thesis/Data/cleaned_research_data.csv' 
data = pd.read_csv(file_path)


#Creating a LLM - Persona 

#Using the given attributes of the researchers, five main persona categories were established, each reflecting common research perspectives and analytical preferences
#Selecting these personas/researchers 

# Let's examine the data and categorize each researcher into one of the proposed personas
# based on their attributes, I'll define thresholds based on the proposed persona profiles

# Defining functions to classify each researcher based on criteria for each persona.

def assign_persona(row):

    # Retrieving values for relevant columns
    degree = row['backgr_degree']
    teaching_exp = row['backgr_exp_teach_stat']
    belief_certainty = row[['belief_certainty_1', 'belief_certainty_2', 'belief_certainty_3']].mean()
    interest = row.get('v_88', 0)  # Interest in project, if available
    time_constraint = row.get('v_98', 0)  # Time constraint, if available

    # Refining conditions for each persona type
    if degree == 7 or (teaching_exp >= 8 and belief_certainty <= 3):
        return "Innovative Modeler"  # Highly method-focused or experimental preference

    elif degree == 2 and teaching_exp >= 7 and belief_certainty >= 5:
        return "Statistical Purist"  # Strong statistical expertise and confidence

    elif degree in [3, 4] and teaching_exp <= 4 and 2 <= belief_certainty <= 4:
        return "Social Scientist"  # Sociological/political focus with moderate certainty

    elif degree in [2, 3] and belief_certainty >= 4 and teaching_exp >= 5:
        return "Hypothesis-Driven Analyst"  # High belief certainty, likely hypothesis-driven

    elif degree in [4] and belief_certainty <= 3 and (time_constraint >= 2 or teaching_exp <= 5):
        return "Empirical Skeptic"  # Low certainty, cautious approach with potential constraints

    else:
        return "Other"  # Researchers who don't fit neatly into any specific persona category

# Applying the refined persona assignment function to each row in the DataFrame
data['persona'] = data.apply(assign_persona, axis=1)

# Displaying the count of each persona type to verify the refined distribution
print(data['persona'].value_counts())

#1 Step, is to define Persona-Specific Model Selection Guidelines

# Define persona model selection guidelines as a dictionary for each persona type

# persona model selection criteria with additional preprocessing options
persona_model_criteria = {
    "Statistical Purist": {
        "model_types": ["linear_regression", "logistic_regression",  "stock_model_ols", "flow_model_ols"],
        "complexity_preference": "simple",
        "data_handling": {
            "handle_missing": "mean_imputation",
            "transformations": ["log"],
            "outlier_handling": "remove_outliers"  # New option for outlier handling
        }
    },
    "Social Scientist": {
        "model_types": ["hierarchical_linear_model", "generalized_linear_model"],
        "complexity_preference": "moderate",
        "data_handling": {
            "handle_missing": "multiple_imputation",
            "transformations": ["none"],
            "outlier_handling": "cap_outliers"
        }
    },
    "Innovative Modeler": {
        "model_types": ["bayesian_regression", "machine_learning", "complex_bayesian_model", "ml_model"],
        "complexity_preference": "complex",
        "data_handling": {
            "handle_missing": "predictive_imputation",
            "transformations": ["scaling", "PCA"],
            "outlier_handling": "retain_outliers"
        }
    },
    "Hypothesis-Driven Analyst": {
        "model_types": ["confirmatory_factor_analysis", "logistic_regression", "logistic_model_flow", "logistic_model_stock"],
        "complexity_preference": "moderate",
        "data_handling": {
            "handle_missing": "listwise_deletion",
            "transformations": ["standardization"],
            "outlier_handling": "remove_outliers"
        }
    },
    "Empirical Skeptic": {
        "model_types": ["linear_regression", "basic_multilevel_model", "stock_model_simple"],
        "complexity_preference": "low",
        "data_handling": {
            "handle_missing": "listwise_deletion",
            "transformations": ["none"],
            "outlier_handling": "cap_outliers"
        }
    }
}

# Defining a function that selects a model based on persona guidelines and input dataset characteristics
def select_model_based_on_persona(persona, dataset):
    """
    Selects a model and data handling strategy based on the specified persona's preferences
    Parameters:
    - persona (str): The persona type for which to select the model
    - dataset (DataFrame): The dataset to be analyzed (used for deciding based on data characteristics)
    Returns:
    - dict: Contains selected model, complexity preference, and data handling steps

    """
    # Retrieving the specific guidelines for the given persona
    criteria = persona_model_criteria.get(persona, {})
    
    # Determining model type based on persona and basic dataset properties (e.g., sample size)
    model_type = criteria.get("model_types", ["linear_regression"])[0]  
    
    # Applying complexity preference and data handling guidelines
    complexity = criteria.get("complexity_preference", "simple")
    data_handling = criteria.get("data_handling", {"handle_missing": "none", "transformations": ["none"]})
    
    # Returning the selected model setup
    return {
        "persona": persona,
        "selected_model": model_type,
        "complexity": complexity,
        "data_handling": data_handling
    }

# Example usage for a persona (can be looped for each persona if needed)
persona_example = "Social Scientist"
dataset_example = data 
model_selection = select_model_based_on_persona(persona_example, dataset_example)

print("Model selection for persona:", model_selection)

#ReadME - this setup gives each persona a unique model selection behavior that matches their analytical tendencies
"""
Persona Model Criteria:
The dictionary persona_model_criteria defines model types, complexity preferences, and data handling preferences for each persona
select_model_based_on_persona Function: This function uses the personas criteria to select a model type, complexity preference, and data handling strategy
You can pass in the persona name and dataset to simulate model selection
Example Output: The function outputs the selected model, complexity level, and data handling steps based on the persona

"""
# 2 Step, Developing LLM Persona Agents

class PersonaAgent:
    def __init__(self, persona_type):
        """
        Initializes the PersonaAgent with a specific persona type and sets guidelines.
        """
        self.persona_type = persona_type
        self.guidelines = persona_model_criteria.get(persona_type, {})
    
    def select_model(self, dataset):
        """
        Uses the persona's preferences to select a model, simulating an LLM-driven choice.
        """
        model_info = select_model_based_on_persona(self.persona_type, dataset)
        print(f"[{self.persona_type}] Selected model: {model_info['selected_model']}")
        # Simulate an LLM reasoning process for model selection
        model_choice_reason = f"As a {self.persona_type}, I chose {model_info['selected_model']} because it aligns with my preference for {model_info['complexity']} complexity."
        print(f"[{self.persona_type} Reasoning] {model_choice_reason}")
        return model_info
    
    def preprocess_data(self, dataset):
        """
        Applies data handling based on persona-specific guidelines, including outlier handling and scaling.
        """
        # Ensure dataset is a DataFrame
        if not isinstance(dataset, pd.DataFrame):
            raise TypeError(f"Expected 'dataset' to be a DataFrame, but got {type(dataset).__name__} instead.")

        data_handling = self.guidelines.get("data_handling", {})
        
        # Select numeric columns to handle only numeric data
        numeric_cols = dataset.select_dtypes(include=['number']).columns

        # Handle missing values only in numeric columns to avoid nuisance columns warning
        if data_handling["handle_missing"] == "mean_imputation":
            dataset[numeric_cols] = dataset[numeric_cols].fillna(dataset[numeric_cols].mean())
        elif data_handling["handle_missing"] == "listwise_deletion":
            dataset.dropna(inplace=True)
        elif data_handling["handle_missing"] == "predictive_imputation":
            # Placeholder: Apply predictive imputation or fill missing with a general approach
            dataset[numeric_cols] = dataset[numeric_cols].fillna(dataset[numeric_cols].mean())  # Replace with actual predictive imputation if needed

        # Ensure no NaN or Inf values remain for scaling
        dataset.replace([np.inf, -np.inf], np.nan, inplace=True)
        dataset.dropna(inplace=True)  # Alternatively, fill with mean or median if appropriate

        # Transformations like scaling or PCA
        if len(dataset) > 0:  # Check if dataset is non-empty
            if "scaling" in data_handling["transformations"]:
                scaler = StandardScaler()
                try:
                    dataset[numeric_cols] = scaler.fit_transform(dataset[numeric_cols])
                except ValueError as e:
                    print(f"[{self.persona_type}] Error during scaling: {e}")

            # Optional: PCA transformation
            if "PCA" in data_handling["transformations"]:
                pca = PCA(n_components=min(len(numeric_cols), dataset.shape[0]))
                try:
                    dataset[numeric_cols] = pca.fit_transform(dataset[numeric_cols])
                except ValueError as e:
                    print(f"[{self.persona_type}] Error during PCA transformation: {e}")

        print(f"[{self.persona_type}] Preprocessed data with: {data_handling}")
        return dataset

    def analyze_data(self, dataset):
        """
        Conducts the analysis, fitting the chosen model to the dataset.
        """
        # Check the type of dataset before processing
        print(f"[DEBUG] Type of dataset before preprocessing: {type(dataset)}")

        # Ensure that 'dataset' is a DataFrame
        if not isinstance(dataset, pd.DataFrame):
            print(f"[ERROR] Expected 'dataset' to be a DataFrame, but got {type(dataset)}")
            return  # Exit to prevent further issues
        
        model_info = self.select_model(dataset)
        preprocessed_data = self.preprocess_data(dataset)
        
        results = {
            "model": model_info["selected_model"],
            "complexity": model_info["complexity"],
            "fit_success": True,  # Simulating a successful fit
            "parameter_estimates": {"beta": random.uniform(-1, 1)}  # Placeholder parameter
        }
        print(f"[{self.persona_type}] Analysis complete with results: {results}")
        return results

    def interpret_results(self, results):
        """
        Provides persona-specific interpretation of results.
        """
        interpretation = ""
        if self.persona_type == "Statistical Purist":
            interpretation = f"The results are reliable with {results['complexity']} complexity."
        elif self.persona_type == "Social Scientist":
            interpretation = f"Results should consider social factors, as the model shows moderate alignment."
        elif self.persona_type == "Innovative Modeler":
            interpretation = f"The complexity enhances insights, but caution is needed for generalization."
        elif self.persona_type == "Hypothesis-Driven Analyst":
            interpretation = f"Findings strongly support the hypothesis-driven framework."
        elif self.persona_type == "Empirical Skeptic":
            interpretation = f"Results should be interpreted conservatively due to low complexity preference."

        print(f"[{self.persona_type} Interpretation] {interpretation}")
        return interpretation

# Example usage
dataset_example = data # Placeholder for actual dataset

# Create persona agents
persona_types = ["Statistical Purist", "Social Scientist", "Innovative Modeler", 
                 "Hypothesis-Driven Analyst", "Empirical Skeptic"]
agents = [PersonaAgent(persona) for persona in persona_types]
results = {}

# Function for ANOVA Comparison

from scipy.stats import f_oneway

def compare_persona_outputs(results_df, parameter="beta"):
    """
    Compares parameter estimates across persona outputs using ANOVA.
    Parameters:
    - results_df: DataFrame containing persona outputs
    - parameter: The parameter to compare across personas (default is 'beta')
    """
    # Group parameter estimates by persona
    grouped_estimates = results_df.groupby("persona")[f"parameters_{parameter}"].apply(list)
    
    # Perform ANOVA across groups
    f_stat, p_value = f_oneway(*grouped_estimates)
    
    print(f"ANOVA results for {parameter}: F-statistic={f_stat}, p-value={p_value}")
    return f_stat, p_value


# Analysis and interpretation for each persona
for agent in agents:
    print(f"\nRunning analysis for persona: {agent.persona_type}")
    analysis_results = agent.analyze_data(dataset_example)
    interpretation = agent.interpret_results(analysis_results)
    results[agent.persona_type] = {"analysis": analysis_results, "interpretation": interpretation}

print("\nAll persona agent results and interpretations:", results)

"""
# Path where detailed logs are saved
detailed_logs_path = '/Users/laurapichler/Desktop/Repository/Master-Thesis/Results/detailed_logs/'

for dataset_name, dataset in datasets.items():
    for agent in agents:
        # ... existing analysis code ...
        
        # Create a detailed log for each persona's results on each dataset
        detailed_log = {
            "persona": agent.persona_type,
            "dataset": dataset_name,
            "model_used": analysis_results["model"],
            "parameters": analysis_results["parameter_estimates"],
            "complexity": analysis_results["complexity"],
            "interpretation": interpretation
        }
        
        # Save the log as a JSON file
        log_file_path = f"{detailed_logs_path}{agent.persona_type}_{dataset_name}_analysis.json"
        with open(log_file_path, 'w') as f:
            json.dump(detailed_log, f, indent=4)
        
        print(f"Detailed log saved for {agent.persona_type} on {dataset_name} at {log_file_path}")
"""

"""
LLM-Driven Decision Simulation:
Each persona generates a reasoning statement for their model choice, simulating a decision-making process that would align with an LLMs reasoning

Enhanced Data Preprocessing Options:
Data handling preferences have been expanded to include options like mean imputation, listwise deletion, and scaling. Each persona applies these steps based on their guidelines, providing flexibility in data preparation

Model Interpretation Based on Persona:
The interpret_results method allows each persona to generate an interpretation that reflects their analytical philosophy, adding a layer of qualitative assessment to each personas output

Logging and Comparison of Decisions:
Each personas choices, analysis outcomes, and interpretations are stored in a results dictionary. This allows for easy comparison of decisions and interpretations across personas, which can be analyzed further in later steps
"""
#Step 3 - Replicating the study with the data used in the experiment 


# Loading the datasets
# Dictionary with file paths and dataset names
import pandas as pd 
file_path = '/Users/laurapichler/Desktop/Repository/Master-Thesis/Breznau/processed_dataset2.csv' 
dataset = pd.read_csv(file_path)
dataset = pd.DataFrame(dataset)
  
all_results = []


# Run analysis for each persona on each dataset
for agent in agents:
    print(f"\nRunning analysis for persona: {agent.persona_type} on the dataset")
    
    # Model selection, data processing, and analysis by each persona on the entire dataset
    analysis_results = agent.analyze_data(dataset)
    
    # Interpretation aligned with persona's style
    interpretation = agent.interpret_results(analysis_results)
    
    # Storing results for comparison
    all_results.append({
        "persona": agent.persona_type,
        "dataset": "processed_dataset",  # or any identifier for the dataset
        "model_used": analysis_results["model"],
        "parameters": analysis_results["parameter_estimates"],
        "complexity": analysis_results["complexity"],
        "interpretation": interpretation
    })

# Converting to DataFrame 

results_df = pd.DataFrame(all_results)
print("\nPersona Agent Analysis Results Across Multiple Datasets:")
print(results_df)

# each persona agent’s analysis on multiple datasets using random values for the parameter estimates
output_path = '/Users/laurapichler/Desktop/Repository/Master-Thesis/Results/persona_agent_analysis_results.csv'  
results_df.to_csv(output_path, index=False)
print(f"Results saved to {output_path}")


