'''
I want to better differentiate between personas and achieve more meaningful variance decomposition in my analysis
In this version i aim to incorporate more distinct strategies for my personas
'''

'''
UPDATED - Enhanced Persona Differentiation for Replicating Breznau's Study
'''
import pandas as pd
import statsmodels.formula.api as smf
import numpy as np
import random
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.ensemble import RandomForestRegressor
from statsmodels.api import OLS
from statsmodels.tools.tools import add_constant
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)


# Load the dataset
file_path = '/Users/laurapichler/Desktop/Repository/Master-Thesis/Data/cleaned_research_data.csv' 
data = pd.read_csv(file_path)

# Persona Assignment Function
def assign_persona(row):
    degree = row['backgr_degree']
    teaching_exp = row['backgr_exp_teach_stat']
    belief_certainty = row[['belief_certainty_1', 'belief_certainty_2', 'belief_certainty_3']].mean()
    time_constraint = row.get('v_98', 0)  # Time constraint, if available

    if degree == 7 or (teaching_exp >= 8 and belief_certainty <= 3):
        return "Innovative Modeler"
    elif degree == 2 and teaching_exp >= 7 and belief_certainty >= 5:
        return "Statistical Purist"
    elif degree in [3, 4] and teaching_exp <= 4 and 2 <= belief_certainty <= 4:
        return "Social Scientist"
    elif degree in [2, 3] and belief_certainty >= 4 and teaching_exp >= 5:
        return "Hypothesis-Driven Analyst"
    elif degree in [4] and belief_certainty <= 3 and teaching_exp <= 5:
        return "Empirical Skeptic"
    else:
        return "Other"

# Apply persona assignment
data['persona'] = data.apply(assign_persona, axis=1)

persona_model_criteria = {
    "Statistical Purist": {
        "model_types": ["linear_regression"],
        "data_handling": {
            "transformations": ["log"],
            "interactions": ["wdi_unempilo:socx_oecd", "gdp_oecd:age"],
            "outlier_handling": "remove_outliers",
            "handle_missing": "mean_imputation"
        }
    },
    "Social Scientist": {
        "model_types": ["hierarchical_linear_model"],
        "data_handling": {
            "transformations": ["zscore"],
            "interactions": ["netmigpct:education", "female:income"],
            "outlier_handling": "cap_outliers",
            "handle_missing": "multiple_imputation"
        }
    },
    "Innovative Modeler": {
        "model_types": ["machine_learning"],
        "data_handling": {
            "transformations": ["scaling", "PCA"],
            "interactions": ["migstock_un:income", "socx_oecd:oldage_c"],
            "outlier_handling": "retain_outliers",
            "handle_missing": "predictive_imputation"
        }
    },
    "Hypothesis-Driven Analyst": {
        "model_types": ["logistic_regression"],
        "data_handling": {
            "transformations": ["polynomial"],
            "interactions": ["wdi_unempilo:netmigpct", "health_c:housing_c"],
            "outlier_handling": "remove_outliers",
            "handle_missing": "listwise_deletion"
        }
    },
    "Empirical Skeptic": {
        "model_types": ["linear_regression"],
        "data_handling": {
            "transformations": ["none"],
            "interactions": ["jobs_c:income"],
            "outlier_handling": "cap_outliers",
            "handle_missing": "listwise_deletion"
        }
    }
}
import pandas as pd 

# Load the dataset
file_path = '/Users/laurapichler/Desktop/Repository/Master-Thesis/Breznau/processed_dataset2.csv'  
dataset = pd.read_csv(file_path)
#print(dataset.columns)

# Define dependent variables
dvs_c = ['incdiff_c', 'jobs_c', 'oldage_c', 'unemp_c', 'housing_c', 'health_c']

# Define independent variables
ivs = ['migstock_un', 'netmigpct', 'wdi_unempilo', 'socx_oecd', 'gdp_oecd']




# Enhanced PersonaAgent Class
# Enhanced PersonaAgent Class
class PersonaAgent:
    def __init__(self, persona_type):
        self.persona_type = persona_type
        self.guidelines = persona_model_criteria.get(persona_type, {})

    def preprocess_data(self, dataset):
        data_handling = self.guidelines.get("data_handling", {})
        numeric_cols = dataset.select_dtypes(include=['number']).columns

        # Handle missing values
        handle_missing = data_handling.get("handle_missing", "mean_imputation")
        if handle_missing == "mean_imputation":
            dataset.loc[:, numeric_cols] = dataset[numeric_cols].fillna(dataset[numeric_cols].mean())
        elif handle_missing == "listwise_deletion":
            dataset = dataset.dropna().copy()
        elif handle_missing == "predictive_imputation":
            dataset.loc[:, numeric_cols] = dataset[numeric_cols].fillna(dataset[numeric_cols].median())

        # Handle outliers
        outlier_handling = data_handling.get("outlier_handling", "none")
        if outlier_handling == "remove_outliers":
            dataset = dataset[
                (np.abs(dataset[numeric_cols] - dataset[numeric_cols].mean()) <= (3 * dataset[numeric_cols].std())).all(axis=1)
            ]
            if dataset.empty:
                raise ValueError(f"All data points removed as outliers for persona {self.persona_type}.")
        elif outlier_handling == "cap_outliers":
            dataset.loc[:, numeric_cols] = dataset[numeric_cols].clip(
                lower=dataset[numeric_cols].quantile(0.05),
                upper=dataset[numeric_cols].quantile(0.95)
            )

        # Apply transformations
        transformations = data_handling.get("transformations", [])
        if "log" in transformations:
            dataset.loc[:, numeric_cols] = dataset[numeric_cols].applymap(lambda x: np.log(x + 1) if x > 0 else np.nan)
        if "zscore" in transformations:
            dataset.loc[:, numeric_cols] = dataset[numeric_cols].apply(lambda x: (x - x.mean()) / x.std())
        if "scaling" in transformations:
            scaler = StandardScaler()
            dataset.loc[:, numeric_cols] = scaler.fit_transform(dataset[numeric_cols])

        # Handle NaNs and infinite values
        dataset.replace([np.inf, -np.inf], np.nan, inplace=True)
        dataset.dropna(inplace=True)

        if dataset.empty:
            raise ValueError(f"Dataset is empty after preprocessing for persona {self.persona_type}.")

        return dataset


    def analyze_data(self, dataset, dv, ivs):
        try:
            # Fit a persona-specific model
            X = dataset[ivs]
            y = dataset[dv]

            # Check for NaN or infinite values
            if X.isnull().values.any() or y.isnull().values.any():
                raise ValueError("Dataset contains NaN values after preprocessing.")
            if np.isinf(X).values.any() or np.isinf(y).values.any():
                raise ValueError("Dataset contains infinite values after preprocessing.")

            X = add_constant(X)

            # Select persona-specific model
            model_type = self.guidelines.get("model_types", ["linear_regression"])[0]
            if model_type == "linear_regression":
                model = OLS(y, X).fit()
            elif model_type == "machine_learning":
                model = RandomForestRegressor()
                model.fit(X, y)
                return {"beta": model.feature_importances_, "AIC": None, "BIC": None}
            else:
                raise NotImplementedError(f"Model type {model_type} not implemented.")

            return {"beta": model.params[1], "AIC": model.aic, "BIC": model.bic}
        except Exception as e:
            print(f"Error in {self.persona_type}: {e}")
            return {"beta": np.nan, "AIC": np.nan, "BIC": np.nan}

# Multiverse Analysis with Enhanced Differentiation
results = []
persona_types = ["Statistical Purist", "Social Scientist", "Innovative Modeler", 
                 "Hypothesis-Driven Analyst", "Empirical Skeptic"]
agents = [PersonaAgent(persona) for persona in persona_types]



# Run analysis for each persona
for agent in agents:
    for dv in dvs_c:
        for spec in ivs:
            preprocessed_data = agent.preprocess_data(dataset.copy())
            result = agent.analyze_data(preprocessed_data, dv, ivs)
            results.append({
                "persona": agent.persona_type,
                "dependent_var": dv,
                "independent_var": spec,
                "beta": result["beta"],
                "AIC": result["AIC"],
                "BIC": result["BIC"]
            })

# Save results
results_df = pd.DataFrame(results)
results_df.to_csv('/Users/laurapichler/Desktop/Repository/Master-Thesis/Results/variance_resultstrialtwo.csv', index=False)

# Variance decomposition
persona_variance = results_df.groupby('persona')['beta'].var()
model_variance = results_df.groupby(['persona', 'dependent_var'])['beta'].var()
total_variance = results_df['beta'].var()
explained_variance = persona_variance.sum() + model_variance.sum()
unexplained_variance = total_variance - explained_variance

print(f"Total Variance: {total_variance}")
print(f"Persona-Level Variance: {persona_variance}")
print(f"Model-Level Variance: {model_variance}")
print(f"Unexplained Variance: {unexplained_variance}")
