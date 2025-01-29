import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import ks_2samp

# Load Breznau et al.'s replication data

breznau_data = pd.read_csv('/Users/laurapichler/Desktop/Repository/Master-Thesis/Results/results_beta_extraction.csv')
breznau_betas = breznau_data['beta']  
# Load your persona agent results
persona_results = pd.read_csv('/Users/laurapichler/Desktop/Repository/Master-Thesis/Results/updated_persona_multiverse_results_updated.csv')
persona_betas = persona_results['beta']  

# Visualize the distributions using density plots
plt.figure(figsize=(10, 6))
sns.kdeplot(breznau_betas, label='Breznau et al.', shade=True, color='blue')
sns.kdeplot(persona_betas, label='Persona Agents', shade=True, color='orange')

# Add plot labels and title
plt.title("Comparison of Beta Distributions: Breznau et al. vs Persona Agents", fontsize=16)
plt.xlabel("Beta Coefficients", fontsize=14)
plt.ylabel("Density", fontsize=14)
plt.legend(fontsize=12)
plt.grid(True)
plt.show()

# Conduct Kolmogorov-Smirnov (K-S) test
ks_stat, p_value = ks_2samp(breznau_betas, persona_betas)
print(f"K-S Test Statistic: {ks_stat}")
print(f"P-Value: {p_value}")

# Interpret the K-S test result
if p_value < 0.05:
    print("The distributions are statistically different (p < 0.05).")
else:
    print("The distributions are not statistically different (p >= 0.05).")
