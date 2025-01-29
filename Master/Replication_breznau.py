
'''
Replicating the Study of Breznau using the Persona Agents 
'''
import pandas as pd
import statsmodels.formula.api as smf
import random
import pandas as pd 
import numpy as np 
import random
import json 
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA


# Loading the CSV file
file_path = '/Users/laurapichler/Desktop/Repository/Master-Thesis/Data/cleaned_research_data.csv' 
data = pd.read_csv(file_path)

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


persona_model_criteria = {
    "Statistical Purist": {
        "model_types": ["linear_regression", "logistic_regression",  "stock_model_ols", "flow_model_ols"],
        "complexity_preference": "simple",
        "data_handling": {
            "handle_missing": "mean_imputation",
            "transformations": ["log"],
            "outlier_handling": "remove_outliers"  
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
import pandas as pd 

# Load the dataset
file_path = '/Users/laurapichler/Desktop/Repository/Master-Thesis/Breznau/processed_dataset2.csv'  
dataset = pd.read_csv(file_path)

# Select relevant columns
relevant_columns = [
    'year', 'iso_country', 'female', 'age', 'age_sq', 'education', 'income',
    'reduce_income_diff', 'incdiff_c', 'jobs', 'jobs_c', 'old_age_care', 'oldage_c',
    'unemployed', 'unemp_c', 'housing', 'housing_c', 'health', 'health_c',
    'migstock_un', 'netmigpct', 'wdi_unempilo', 'socx_oecd', 'gdp_oecd'
]
dataset = dataset[relevant_columns]

# Convert macro-indicators to numeric if needed (already numeric in most cases)
for col in ['migstock_un', 'netmigpct', 'wdi_unempilo', 'socx_oecd', 'gdp_oecd']:
    dataset[col] = pd.to_numeric(dataset[col], errors='coerce')

# Check for missing values and handle them
dataset.fillna(dataset.mean(), inplace=True)  # Fill missing numeric values with column means

# Define dependent variables
dvs_c = ['incdiff_c', 'jobs_c', 'oldage_c', 'unemp_c', 'housing_c', 'health_c']

# Define independent variables
ivs = ['migstock_un', 'netmigpct', 'wdi_unempilo', 'socx_oecd', 'gdp_oecd']

# Define model specifications (stock and flow effects)
model_specs = [
    "", "wdi_unempilo + ", "socx_oecd + ", "gdp_oecd + ", "netmigpct + ",
    "wdi_unempilo + socx_oecd + ", "wdi_unempilo + gdp_oecd + ",
    "socx_oecd + gdp_oecd + ", "netmigpct + wdi_unempilo + ",
    "netmigpct + socx_oecd + ", "netmigpct + gdp_oecd + "
]

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
            dataset[numeric_cols] = dataset[numeric_cols].fillna(dataset[numeric_cols].mean())  
        
        dataset.replace([np.inf, -np.inf], np.nan, inplace=True)
        dataset.dropna(inplace=True) 

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

#Replacing the placeholder analyze_data logic in the PersonaAgent class with a regression-based approach

from statsmodels.api import OLS
from statsmodels.tools.tools import add_constant

class PersonaAgent:
    def __init__(self, persona_type):
        self.persona_type = persona_type
        self.guidelines = persona_model_criteria.get(persona_type, {})

    def analyze_data(self, dataset):
        """
        Conducts analysis using the persona's model preferences.
        """
        print(f"\n[{self.persona_type}] Running analysis on dataset.")
        # Use a placeholder dependent and independent variable
        try:
            # Use dependent and independent variables dynamically
            y = dataset[random.choice(dvs_c)]  # Randomly select a dependent variable for variability
            X = dataset[ivs]
            X = add_constant(X)

            # Fit OLS model
            model = OLS(y, X).fit()
            beta = model.params[1]  # Example: Use the first coefficient as beta

            result = {
                "model": "OLS",
                "complexity": self.guidelines.get("complexity_preference", "unknown"),
                "AIC": model.aic,
                "BIC": model.bic,
                "parameter_estimates": {"beta": beta}
            }

        except Exception as e:
            print(f"[{self.persona_type}] Error during analysis: {e}")
            result = {
                "model": "OLS",
                "complexity": self.guidelines.get("complexity_preference", "unknown"),
                "parameter_estimates": {"beta": np.nan}
            }

        return result


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


# Create persona agents
persona_types = ["Statistical Purist", "Social Scientist", "Innovative Modeler", 
                 "Hypothesis-Driven Analyst", "Empirical Skeptic"]
agents = [PersonaAgent(persona) for persona in persona_types]
results = []

'''
Multiverse Analysis
'''

for agent in agents:  # Loop through persona agents
    for dv in dvs_c:  # Loop through dependent variables
        for spec in model_specs:  # Loop through model specifications
            try:
                # Construct model formula
                formula = f"{dv} ~ gdp_oecd + {spec}C(year) + C(iso_country)"
                
                # Fit the model using OLS
                model = smf.ols(formula=formula, data=dataset).fit()

                # Append results
                results.append({
                    'persona': agent.persona_type,
                    'dependent_variable': dv,
                    'specification': spec.strip(),
                    'beta': model.params.get('gdp_oecd', np.nan),
                    'p_value': model.pvalues.get('gdp_oecd', np.nan),
                    'AIC': model.aic,
                    'BIC': model.bic
                })
            except Exception as e:
                print(f"Error for {agent.persona_type}, {dv}, {spec}: {e}")



# Convert results to DataFrame
results_breznau = pd.DataFrame(results)

# Save results for inspection
results_breznau.to_csv('/Users/laurapichler/Desktop/Repository/Master-Thesis/Results/persona_multiverse_results_gdpoecd.csv', index=False)

# Calculate variance of beta values for each persona
variance_by_persona = results_breznau.groupby('persona')['beta'].var()
print("Variance by Persona:\n", variance_by_persona)


# In order to introduce greater variability in outcomes and address the uniformity observed in my variance analysis, I am going to expand the distinctiveness of persona behaviors