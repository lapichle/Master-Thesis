import matplotlib.pyplot as plt

# Data for the pie chart
labels = [
    "Statistical Purist (2)",
    "Social Scientist (32)",
    "Hypothesis-Driven Analyst (21)",
    "Innovative Modeler (12)",
    "Empirical Skeptic (1)",
    "Other (138)"
]
sizes = [2, 32, 21, 12, 1, 138]
colors = ["#ff9999","#66b3ff","#99ff99","#ffcc99", "#c2c2f0", "#ffb3e6"]
explode = (0, 0, 0, 0, 0, 0)  # Highlight the largest category (Other)

# Create the pie chart
plt.figure(figsize=(4, 4))
plt.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=140, explode=explode)
plt.title("Persona Distribution in Researcher Dataset", fontsize=14, fontweight='bold')
plt.axis('equal')  # Equal aspect ratio ensures the pie chart is circular.

# Save the pie chart
pie_chart_path = "/Users/laurapichler/Desktop/Repository/Plots/Persona_Distribution_Pie_Chart.png"
plt.savefig(pie_chart_path, bbox_inches="tight", dpi=300)
plt.show()

pie_chart_path
