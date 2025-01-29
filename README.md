# 🎓 Master Thesis: Battling Idiosyncratic Uncertainty
**Exploring Persona Agent LLMs for Enhancing Research Replicability**  
📍 *University of Mannheim*  
👩‍💻 *Author: Laura Pichler*  
📅 *Date: January 31, 2025*  

---

## 📖 **Overview**
This repository contains all the datasets, code, and analysis scripts associated with my master's thesis:  
> *"Battling Idiosyncratic Uncertainty: The Conflict between Artificial Intelligence and Unpredictable Factors."*

The study investigates how **Persona Agent Large Language Models (LLMs)** can reduce idiosyncratic biases in research by simulating the decision-making processes of diverse researchers. Inspired by **Breznau et al. (2022)**, this research explores how **interpretive flexibility** affects scientific reproducibility and how structured AI personas can enhance transparency in data analysis.

---

## 🏛 **Research Goals**
✅ Develop **persona agents** to mimic diverse researcher profiles.  
✅ Examine **interpretive variability** in statistical modeling decisions.  
✅ Enhance **replicability** by structuring LLMs for research workflows.  
✅ Analyze how AI-based decision frameworks compare to **human analysts**.

---

## 📂 **Repository Structure**
```plaintext
Master-Thesis/
├── 📁 code/              # Python scripts & Jupyter notebooks for analysis
│   ├── LLM_Persona.py   # Core script for Persona Agent model selection
│   ├── t_beta_se.py     # Script for computing and visualizing t-values
│   ├── data_analysis.py # Statistical analysis workflow
│   ├── prompt_engineering.py # Rules for guiding LLM persona responses
│   ├── visualization.py # Density plots and comparative analyses
│   └── README.md        # Documentation for using the scripts
│
├── 📁 data/              # Processed & raw datasets (not included if large)
│   ├── persona_agent_results.csv
│   ├── results_beta_extraction.csv
│   ├── cleaned_research_data.csv
│   ├── research_data.csv
│   ├── ISSP_replication_data.csv
│   └── README.md        # Data dictionary and variable explanations
│
├── 📁 figures/           # All generated plots and visualizations
│   ├── densityplot_beta.png
│   ├── tvalues_ratco.png
│   ├── standard_errors.png
│   ├── violin_beta_distributions.png
│   └── README.md        # Explanation of figures
│
├── 📁 results/           # Statistical outputs & summary results
│   ├── persona_multiverse_results.csv
│   ├── variance_analysis.csv
│   ├── factor_loadings.csv
│   ├── ANOVA_results.txt
│   ├── README.md        # Notes on statistical findings
│
└── README.md            # Main repository documentation
