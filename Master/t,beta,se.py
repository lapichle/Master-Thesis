import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np


# Load Breznau et al.'s replication data

results_beta_extraction_df = pd.read_csv('/Users/laurapichler/Desktop/Repository/Master-Thesis/Results/results_beta_extraction.csv')

# Load your persona agent results
updated_persona_df = pd.read_csv('/Users/laurapichler/Desktop/Repository/Master-Thesis/Results/updated_persona_multiverse_results_updated.csv')


def calculate_t_values_(df, beta_col, p_value_col):
    clipped_p_values = df[p_value_col].clip(upper=0.9999)
    return np.sign(df[beta_col]) * (np.abs(df[beta_col]) / np.sqrt(1 - clipped_p_values))

# Recalculate t_values and standard_errors if necessary
updated_persona_df['t_value'] = calculate_t_values_(updated_persona_df, 'beta', 'p_value')
results_beta_extraction_df['t_value'] = calculate_t_values_(results_beta_extraction_df, 'beta', 'p_value')

# Calculate standard_error from beta and t_value
updated_persona_df['standard_error'] = updated_persona_df['beta'] / updated_persona_df['t_value']
results_beta_extraction_df['standard_error'] = results_beta_extraction_df['beta'] / results_beta_extraction_df['t_value']

# Verify the recalculated columns
updated_persona_df[['beta', 'p_value', 't_value', 'standard_error']].head(), \
results_beta_extraction_df[['beta', 'p_value', 't_value', 'standard_error']].head()


# Extract values for visualization
persona_betas = updated_persona_df['beta']
breznau_betas = results_beta_extraction_df['beta']

persona_t_values = updated_persona_df['t_value']
breznau_t_values = results_beta_extraction_df['t_value']

persona_standard_errors = updated_persona_df['standard_error']
breznau_standard_errors = results_beta_extraction_df['standard_error']


# Plot Betas
plt.figure(figsize=(8, 6))
persona_betas.plot(kind='kde', color="#ff7f0e", label="Persona Agent Betas")
breznau_betas.plot(kind='kde', color="#1f77b4", linestyle='--', label="Breznau et al. Betas")

plt.title("Density Plot of Beta Coefficients", fontsize=16, fontweight='bold')
plt.xlabel("Beta Coefficient", fontsize=12)
plt.ylabel("Density", fontsize=12)
plt.legend(title="Source", fontsize=10)
plt.grid(alpha=0.3)
plt.show()

# Plot T-Values
plt.figure(figsize=(8, 6))
persona_t_values.plot(kind='kde', color="#ff7f0e", label="Persona Agent T-Values")
breznau_t_values.plot(kind='kde', color="#1f77b4", linestyle='--', label="Breznau et al. T-Values")

plt.title("Density Plot of T-Values", fontsize=16, fontweight='bold')
plt.xlabel("T-Value", fontsize=12)
plt.ylabel("Density", fontsize=12)
plt.legend(title="Source", fontsize=10)
plt.grid(alpha=0.3)
plt.show()

# Plot Standard Errors
plt.figure(figsize=(8, 6))
persona_standard_errors.plot(kind='kde', color="#ff7f0e", label="Persona Agent Standard Errors")
breznau_standard_errors.plot(kind='kde', color="#1f77b4", linestyle='--', label="Breznau et al. Standard Errors")

plt.title("Density Plot of Standard Errors", fontsize=16, fontweight='bold')
plt.xlabel("Standard Error", fontsize=12)
plt.ylabel("Density", fontsize=12)
plt.legend(title="Source", fontsize=10)
plt.grid(alpha=0.3)
plt.show()

#The spike in the standard error distribution for the "Persona Agent" dataset suggests that the computed standard errors are nearly constant or exhibit minimal variability