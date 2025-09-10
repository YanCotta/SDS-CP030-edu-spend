# Welcome to the SuperDataScience Community Project!
Welcome to the **EduSpend: Global Higher-Education Cost Analytics & Planning** repository! 🎉

This project is a collaborative initiative brought to you by SuperDataScience, a thriving community dedicated to advancing the fields of data science, machine learning, and AI. We are excited to have you join us in this journey of learning, experimentation, and growth.

To contribute to this project, please follow the guidelines avilable in our [CONTRIBUTING.md](CONTRIBUTING.md) file.


-> shared rep with all the team members work: https://github.com/SuperDataScience-Community-Projects/SDS-CP030-edu-spend


# Project Scope of Works:

## Project Overview
**EduSpend** transforms a rich international dataset of tuition fees, living-cost indices, rent, visa charges, and insurance premiums into actionable insights for students, consultants, and policymakers. The project will:

- benchmark real costs across countries, cities, and degree levels;
- predict total cost of attendance (TCA) for any program profile;
- classify destinations into affordability tiers;
- segment universities by cost structures;
- deliver an interactive Streamlit app for budget planning and market comparison.

**Link to Dataset:** https://www.kaggle.com/datasets/adilshamim8/cost-of-international-education

## Project Objectives
### Exploratory Data Analysis
- Profile cost distributions by **Country**, **City**, **Level**, and **Program**.
- Visualize correlations among **Tuition_USD**, **Living_Cost_Index**, **Rent_USD**, and **Exchange_Rate**.
- Compute baseline **Total Cost of Attendance** (tuition + rent × months + visa + insurance × years).
- Detect outliers and assess currency-conversion stability.

### Feature Engineering + Model Development
1. **Derived Features**
    - One-hot encode categoricals; scale/transform numeric costs.
    - Create TCA and affordability-tier labels (e.g., Low/Medium/High via quantiles).

2. **Predictive Tasks**
    - **TCA Regressor**: estimate total study cost from program specs and location features.
    - **Affordability Classifier**: label each record _Low_, _Medium_, or _High_ cost.

3. **Clustering**
    - Group universities/destinations into cost archetypes via K-Means or HDBSCAN on scaled features.

4. **Evaluation & Tuning**
    - Regression → MAE, RMSE, R².
    - Classification → Accuracy, Macro F1, ROC-AUC.
    - Clustering → Silhouette Score + qualitative review.
    - Hyperparameter search for tree-based and gradient-boosted models.

### Model Deployment
- **Streamlit Web App**
    - Budget Planner: users enter target country, level, duration, and see TCA forecasts.
    - Affordability Map & Dashboards.
    - Cluster Explorer with segment descriptions.

- Host on **Streamlit Community Cloud** with README and user guide.

## Technical Requirements

* **Data handling**: `pandas`, `numpy`
* **Visualization**: `matplotlib`, `seaborn`, `plotly`
* **Machine-learning & evaluation**: `scikit-learn`, `xgboost`, `lightgbm`, `mlflow`
* **Clustering & dimensionality reduction**: `KMeans`, `HDBSCAN`, `PCA` or `UMAP`
* **Deployment stack**: `Streamlit` for the web interface
* **Runtime environment**: Python 3.9+ in a managed `conda` or `venv` environment


## Workflow

### **Phase 1: Setup & EDA (1 Week)**
- Setup GitHub repo and project folders
- Setup virtual environment and respective libraries
- Data anlysis of key features effecting study costs and affordability
- Answering questions set out in the Objectives section for EDA.

### **Phase 2: Model Development (3 Weeks)**
- Feature engineering
- Model experimentation with ML Flow
- Build, traing, test, evaluate different models
- Build a pipeline to use for Deployment phase

### **Phase 3: Deployment (1 Week)**
- Build a basic Streamlit UI to use your model
- Deploy to Streamlit cloud

## Timeline

| **Phase**     | **Task**                    | **Duration** |
|---------------|-----------------------------|--------------|
| **Phase 1**:  | Setup & EDA                 |  Week 1      |
| **Phase 2**:  | Model Development           |  Week 2 - 4  |
| **Phase 3**:  | Deployment                  |  Week 5      |

