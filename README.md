🌍 SDG 7 Policy & Energy Simulator
Group Assignment 2 | Master of Data Science

📖 Project Overview
This project provides a data-driven framework to analyze global SDG 7 (Affordable and Clean Energy) progress. By integrating Time-Series Forecasting (Prophet/ARIMA) and Unsupervised Learning (K-Means/Hierarchical Clustering), this tool assists policymakers in profiling nation-specific energy strategies and forecasting future renewable energy trajectories toward 2030.

🛠️ Key Analytical Modules
1. Prescriptive Policy Profiler (Clustering)
Methodology: Uses K-Means Clustering (validated against Hierarchical Agglomerative Clustering) to categorize nations into three distinct policy profiles: Green Pioneers, Transitioning Nations, and Fossil Dependents.

Insight: Users can input hypothetical energy targets to visualize where their proposed policy sits in the global energy landscape via a dynamic Voronoi Decision Boundary plot.

2. Policy Gap Forecaster (Regression)
Methodology: Utilizes Prophet (Meta's time-series library) compared against a baseline ARIMA model to forecast renewable energy shares.

Insight: Calculates the "Policy Deficit"—the gap between a nation's 2030 aspirational target and the ML-predicted trajectory—providing a metric for necessary structural intervention.

🚀 Deployment & Installation
This application is built with Streamlit and is hosted live at:
[👉 Link to Streamlit App Here](https://github.com/antaru-ops/WQD7001_Principles_of_Data_Science.git)

Local Setup
To run this project locally, ensure you have Python 3.9+ installed:

Bash
# Clone the repository
git clone [Your-GitHub-Link]

# Install dependencies
pip install -r requirements.txt

# Launch the app
streamlit run app.py
📊 Technical Stack
Language: Python 3.x

Frontend: Streamlit

ML Libraries: Scikit-Learn, Prophet, Statsmodels

Visualization: Matplotlib, Seaborn

👤 Credits
Developed by Group 5 for the WQD7001 Principles of Data Science course.
