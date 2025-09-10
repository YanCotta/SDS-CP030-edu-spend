# 🎓 EduSpend: Global Education Cost Prediction Platform

A comprehensive machine learning platform for predicting Total Cost of Attendance (TCA) for international higher education, featuring AI-powered cost predictions, affordability classification, and market cluster analysis.

## 🌟 Project Overview

EduSpend is an end-to-end machine learning solution that helps students and families plan their international education budget with precision. The platform combines advanced predictive modeling with intuitive web interfaces to provide accurate cost estimates for studying abroad.

### Key Achievements
- **Final Model Performance**: 96.44% R² score for TCA prediction
- **Multi-Platform Deployment**: Streamlit web app and FastAPI REST service
- **Production-Ready**: Containerized with Docker and automated CI/CD
- **Interactive Analytics**: Real-time cost predictions and market analysis

## 🚀 Tech Stack

- **Machine Learning**: Python, Scikit-learn, XGBoost, MLflow
- **Web Applications**: Streamlit, FastAPI
- **Data Processing**: Pandas, NumPy
- **Visualization**: Plotly, Matplotlib, Seaborn
- **Containerization**: Docker
- **CI/CD**: GitHub Actions
- **Cloud Deployment**: Docker Hub

## 📊 Model Performance

Our final TCA prediction model achieves exceptional accuracy:
- **R² Score**: 96.44%
- **Algorithm**: XGBoost Regressor with feature engineering
- **Features**: Country, city, program, education level, living costs, and more
- **Validation**: Cross-validated with robust preprocessing pipeline

## 🛠️ Local Development & Usage

### Prerequisites
- Python 3.9+
- Git

### Setup Instructions

1. **Clone the repository**
   ```bash
   git clone https://github.com/YanCotta/SDS-CP030-edu-spend.git
   cd SDS-CP030-edu-spend/submissions/team-members/yan-cotta
   ```

2. **Create and activate virtual environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

### Running the Applications

#### Streamlit Web Application
Launch the interactive web interface:
```bash
streamlit run app.py
```
Access at: `http://localhost:8501`

**Features**:
- 💰 TCA Budget Planner - AI-powered cost predictions
- 🌍 Affordability Explorer - Destination cost classification
- 📊 Market Cluster Analysis - Global education market insights

#### FastAPI REST Service
Start the REST API server:
```bash
uvicorn api:app --reload --host 0.0.0.0 --port 8000
```
Access at: `http://localhost:8000`
API Documentation: `http://localhost:8000/docs`

**Endpoints**:
- `POST /predict` - TCA prediction
- `POST /predict/batch` - Batch predictions
- `GET /health` - Health check

## 🐳 Containerized Deployment

### Build Docker Image Locally
```bash
docker build -t eduSpend-app .
```

### Run Docker Container
```bash
docker run -p 8501:8501 eduSpend-app
```
Access the application at: `http://localhost:8501`

### Production Docker Image
Our production image is automatically built and available on Docker Hub:

```bash
# Pull the latest image
docker pull yancotta/sds-cp030-edu-spend:latest

# Run the production container
docker run -p 8501:8501 yancotta/sds-cp030-edu-spend:latest
```

**Docker Hub Repository**: https://hub.docker.com/r/yancotta/sds-cp030-edu-spend

**Available Tags**:
- `latest` - Always points to the latest stable build
- `95b35abc7aa1a88999ef8e6393fab2f52b0d9a2c` - Latest commit build (863.7 MB)

## 🔄 CI/CD Pipeline

### Automated Deployment
Our GitHub Actions workflow provides seamless continuous integration and deployment:

- **Trigger**: Every push to the `main` branch
- **Process**: 
  1. Code checkout and validation
  2. Docker image build with multi-platform support (AMD64/ARM64)
  3. Automatic push to Docker Hub
  4. Image tagging with both `latest` and commit SHA

### Pipeline Features
- ✅ **Automated Building**: Docker images built on every commit
- ✅ **Multi-Platform Support**: Compatible with Intel and ARM architectures
- ✅ **Efficient Caching**: GitHub Actions cache for faster builds
- ✅ **Security**: Non-root user execution in containers
- ✅ **Health Monitoring**: Built-in health checks

### Docker Image Tags
- `yancotta/sds-cp030-edu-spend:latest` - Always points to the latest stable build
- `yancotta/sds-cp030-edu-spend:[commit-sha]` - Specific commit versions for rollback

## 📁 Project Structure

```
yan-cotta/
├── 01_EDA_EduSpend.ipynb         # Exploratory Data Analysis
├── 02_Model_Development.ipynb     # Model training and validation
├── 03_Final_Models.ipynb          # Final model selection
├── 04_Final_Pipeline.ipynb        # Production pipeline
├── app.py                         # Streamlit web application
├── api.py                         # FastAPI REST service
├── Dockerfile                     # Container configuration
├── .dockerignore                  # Docker build context exclusions
├── requirements.txt               # Python dependencies
├── tca_predictor.joblib          # Trained model artifact
├── feature_columns.pkl           # Feature engineering pipeline
├── tca_predictor_metadata.json   # Model metadata
├── data/
│   ├── final_labeled_data.csv    # Production dataset
│   └── International_Education_Costs.csv  # Original dataset
└── .github/workflows/
    └── main.yml                  # CI/CD pipeline configuration
```

## 🎯 Key Features

### 1. TCA Budget Planner
- AI-powered cost predictions for international education
- Interactive parameter adjustment
- Detailed cost breakdown analysis
- Visual cost distribution charts

### 2. Affordability Explorer
- Destination affordability classification (Low/Medium/High)
- Budget comparison tools
- Market trend analysis
- Cost-effectiveness insights

### 3. Market Cluster Analysis
- Global education market segmentation
- University cost structure analysis
- Country-specific market insights
- Downloadable filtered datasets

## 🔧 Configuration

### Environment Variables
```bash
# Production Streamlit Configuration
STREAMLIT_SERVER_HEADLESS=true
STREAMLIT_SERVER_ENABLE_CORS=false
STREAMLIT_SERVER_ENABLE_XSRF_PROTECTION=false
STREAMLIT_SERVER_FILE_WATCHER_TYPE=none
```

### Model Configuration
- **Model Type**: XGBoost Regressor
- **Input Features**: 10+ engineered features
- **Output**: Total Cost of Attendance (USD)
- **Preprocessing**: StandardScaler, LabelEncoder

## 📈 Usage Examples

### Web Interface Prediction
1. Navigate to the TCA Budget Planner
2. Select destination (Country/City)
3. Choose academic details (Level/Program)
4. Adjust cost parameters
5. Get instant AI-powered predictions

### API Usage
```python
import requests

# Prediction request
response = requests.post("http://localhost:8000/predict", json={
    "country": "USA",
    "city": "New York",
    "program": "Computer Science",
    "level": "Masters",
    "duration_years": 2.0,
    "living_cost_index": 120,
    "rent_usd": 2000,
    "insurance_usd": 2000,
    "visa_fee_usd": 500
})

prediction = response.json()
print(f"Predicted TCA: ${prediction['tca']:,.2f}")
```

## 🚀 Quick Start

**Option 1: Docker (Recommended)**
```bash
docker run -p 8501:8501 yancotta/sds-cp030-edu-spend:latest
```

**Option 2: Local Development**
```bash
git clone https://github.com/YanCotta/SDS-CP030-edu-spend.git
cd SDS-CP030-edu-spend/submissions/team-members/yan-cotta
pip install -r requirements.txt
streamlit run app.py
```

## 🤝 Contributing

This project is part of the SDS-CP030 program. For contributions or questions, please refer to the main repository guidelines.

## 📝 License

This project is developed as part of an educational program. Please refer to the main repository for licensing information.

## 👨‍💻 Author

**Yan Cotta**  
*Data Scientist & ML Engineer*

---

<div align="center">
  <strong>🎓 EduSpend - Making International Education Accessible Through AI</strong>
</div>
